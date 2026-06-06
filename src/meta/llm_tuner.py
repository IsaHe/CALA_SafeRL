"""
llm_tuner.py - Proponente de permisividad del shield via Gemma/Ollama.

Cuatro capas de defensa para que un LLM local nunca pueda inyectar un valor
inseguro o no parseable:
  1. Decodificacion restringida por esquema - ``format=`` es un JSON schema
     (ollama >= 0.4), asi que el modelo se restringe a emitir JSON que cumpla
     ``ShieldTuningResponse``.
  2. Determinismo - temperature 0, seed fijo, top_p 1.
  3. Validacion - la respuesta se parsea con un modelo pydantic; los problemas
     estructurales (no-JSON / falta clave / tipo erroneo) lanzan y disparan el
     fallback.
  4. Clamp + fallback no-op - el objetivo numerico se clampa a [0, 1]; ante
     CUALQUIER fallo (conexion, timeout, parse, validacion) el proponente
     devuelve un HOLD en la permisividad actual -- el shield se deja igual.

La llamada real ``ollama.chat`` es inyectable (``chat_fn``) para que los tests
unitarios corran con un mock y no necesiten ni GPU ni servidor Ollama.
``keep_alive=0`` pide a Ollama descargar el modelo tras cada llamada, liberando
VRAM para CARLA + PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import BaseModel

from src.meta.kpi_snapshot import KpiSnapshot
from src.meta.tunable_shield import clamp_permissiveness


class ShieldTuningResponse(BaseModel):
    """Contrato estricto para la salida JSON del modelo."""

    model_config = {"extra": "forbid"}

    target_permissiveness: float
    action: str  # "loosen" | "hold" | "tighten"
    reasoning: str


@dataclass(frozen=True)
class TunerProposal:
    target_permissiveness: float
    action: str
    reasoning: str
    ok: bool  # False -> fallback (LLM inusable esta ronda)
    raw_response: str = ""


SYSTEM_PROMPT = (
    "You are a SAFETY-FIRST curriculum controller for a reinforcement-learning "
    "driving agent in CARLA. A safety shield currently overrides the agent's "
    "unsafe actions; your job is to decide how much to LOOSEN that shield so the "
    "agent gradually learns to drive safely on its own - WITHOUT letting crash or "
    "off-road rates rise. You only propose a target; a deterministic governor "
    "enforces hard safety limits on top of your proposal. Respond with JSON only."
)


def build_user_prompt(snapshot: KpiSnapshot) -> str:
    return (
        "Shield permissiveness is a scalar in [0,1]: 1.0 = strict (shield helps "
        "maximally), 0.0 = fully weaned (agent drives unaided). LOWER it only when "
        "the agent is provably safe; RAISE it if safety is degrading.\n\n"
        f"Current permissiveness: {snapshot.current_permissiveness:.3f}\n"
        f"Metrics over the last {snapshot.window_episodes} episodes:\n"
        f"{snapshot.to_json()}\n\n"
        "Guidance:\n"
        "- crash_rate and offroad_rate are the hard priorities. If either is "
        "elevated, propose action='tighten' (higher target_permissiveness).\n"
        "- If crash_rate and offroad_rate are very low AND shielded_fraction is "
        "still high, the agent leans on the shield - propose a SMALL loosening "
        "(action='loosen', slightly lower target_permissiveness).\n"
        "- Otherwise action='hold'.\n"
        "- Move in small steps (~0.05-0.1). Never jump straight to 0.\n"
        "Return JSON with keys: target_permissiveness (float 0..1), action "
        "('loosen'|'hold'|'tighten'), reasoning (one sentence)."
    )


class LLMShieldTuner:
    def __init__(
        self,
        model: str = "gemma4:e4b",
        seed: int = 42,
        temperature: float = 0.0,
        host: Optional[str] = None,
        keep_alive: float | int | str = 0,
        num_gpu: Optional[int] = None,
        request_timeout: float = 600.0,
        chat_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.host = host
        self.keep_alive = keep_alive
        # num_gpu=0 fuerza inferencia en CPU (no compite por VRAM con CARLA y
        # evita 'timed out waiting for llama-server to start' bajo contención de
        # GPU). None deja decidir a Ollama (usa GPU). request_timeout es el
        # timeout del cliente httpx: cargas en frío de modelos grandes son lentas.
        self.num_gpu = num_gpu
        self.request_timeout = request_timeout
        self._chat_fn = chat_fn

    def _options(self) -> dict:
        options = {
            "temperature": self.temperature,
            "seed": self.seed,
            "top_p": 1.0,
        }
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        return options

    def _client(self):
        import ollama  # lazy: el modulo importa sin el paquete / servidor

        return ollama.Client(host=self.host, timeout=self.request_timeout)

    def _chat(self, messages: list[dict], fmt: dict):
        options = self._options()
        if self._chat_fn is not None:
            return self._chat_fn(
                model=self.model,
                messages=messages,
                format=fmt,
                options=options,
                keep_alive=self.keep_alive,
            )
        return self._client().chat(
            model=self.model,
            messages=messages,
            format=fmt,
            options=options,
            keep_alive=self.keep_alive,
        )

    def warmup(self) -> tuple[bool, str]:
        """Pre-carga el modelo con una llamada trivial: falla rápido y con un
        mensaje claro si Ollama no está disponible, y amortiza la carga fuera del
        bucle de entrenamiento. No lanza: devuelve (ok, mensaje)."""
        messages = [{"role": "user", "content": "ready?"}]
        options = {"temperature": 0.0, "num_predict": 1}
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        try:
            if self._chat_fn is not None:
                self._chat_fn(
                    model=self.model,
                    messages=messages,
                    format=None,
                    options=options,
                    keep_alive=self.keep_alive,
                )
            else:
                self._client().chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                    keep_alive=self.keep_alive,
                )
            return True, "ok"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def propose(self, snapshot: KpiSnapshot) -> TunerProposal:
        current_p = clamp_permissiveness(snapshot.current_permissiveness)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(snapshot)},
        ]
        fmt = ShieldTuningResponse.model_json_schema()
        resp = None
        try:
            resp = self._chat(messages, fmt)
            content = resp["message"]["content"]
            parsed = ShieldTuningResponse.model_validate_json(content)
            target = clamp_permissiveness(parsed.target_permissiveness)
            action = (
                parsed.action
                if parsed.action in ("loosen", "hold", "tighten")
                else "hold"
            )
            return TunerProposal(
                target_permissiveness=target,
                action=action,
                reasoning=parsed.reasoning,
                ok=True,
                raw_response=content,
            )
        except Exception as exc:  # parse/validacion/conexion/timeout -> no-op
            return TunerProposal(
                target_permissiveness=current_p,
                action="hold",
                reasoning=f"fallback: {type(exc).__name__}: {exc}",
                ok=False,
                raw_response=_safe_raw(resp),
            )


def _safe_raw(resp) -> str:
    try:
        return resp["message"]["content"] if resp else ""
    except Exception:
        return ""
