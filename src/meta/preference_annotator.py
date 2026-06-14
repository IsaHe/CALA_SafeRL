"""
preference_annotator.py - Anotador de PREFERENCIAS de seguridad via LLM local
(estilo Motif) para el reward model de evitacion de colisiones.

Reutiliza el transporte y las defensas de ``LocalLLMClient`` (decodificacion
restringida por esquema + determinismo) y anade su propia validacion/fallback
para una salida de PREFERENCIA en vez de una permisividad:

    dadas dos captions de escena de conduccion (A y B), el LLM decide cual es la
    situacion MAS SEGURA (menor riesgo de colision), con una confianza.

A diferencia del tuner del shield, este anotador corre OFFLINE (no en el bucle de
20 Hz): no hay contencion de GPU ni latencia que importe, asi que se puede usar un
modelo mas fuerte y/o muchas llamadas. El reward model (fase 2) se entrena luego
con perdida Bradley-Terry sobre estas preferencias.

Defensas (heredadas + propias):
  1-2. schema-constrained decoding + determinismo  -> ``LocalLLMClient``.
  3.   validacion pydantic (``PreferenceResponse``, extra=forbid, ``safer`` en
       {0,1} via Literal -> enum en el schema).
  4.   fallback ABSTENCION: ante CUALQUIER fallo se devuelve ``safer=-1`` (ok=False)
       y el par se DESCARTA del dataset, nunca contamina las etiquetas.

``annotate_symmetric`` pregunta en ambos ordenes (A,B) y (B,A) y solo acepta el
par si las dos respuestas coinciden: cancela el sesgo de posicion conocido de los
LLM como jueces de pares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from src.meta.llm_tuner import LocalLLMClient


class PreferenceResponse(BaseModel):
    """Contrato estricto para la salida JSON del anotador."""

    model_config = {"extra": "forbid"}

    safer: Literal[0, 1]  # indice (0=A, 1=B) de la escena MAS SEGURA
    confidence: float  # 0..1
    reasoning: str


@dataclass(frozen=True)
class PreferenceLabel:
    safer: int  # 0 (A) | 1 (B) | -1 (abstencion / fallback)
    confidence: float
    reasoning: str
    ok: bool  # False -> par descartado (LLM inusable / inconsistente)
    raw_response: str = ""


PREFERENCE_SYSTEM_PROMPT = (
    "You are a SAFETY examiner judging COLLISION risk ONLY. You are shown two single "
    "snapshots of a car driving in CARLA, labelled A and B. Choose the snapshot with "
    "the LOWER risk of hitting something or running off the road. Apply these rules "
    "strictly:\n"
    "1. A road user in the path DIRECTLY AHEAD is the main hazard: the closer it is, "
    "the MORE dangerous. You cannot see whether it will brake, so always treat a "
    "closer in-path road user as less safe, even if the car is keeping pace with it.\n"
    "2. Being off-centre, near a road edge, or poorly aligned with the lane is "
    "dangerous (run-off-road risk).\n"
    "3. A clear road ahead with the car under control is SAFE - even if the car is "
    "momentarily stopped. A stopped car with nothing nearby has almost NO collision "
    "risk; do NOT penalise it for being slow or stopped.\n"
    "4. Judge ONLY the risk of a collision or going off-road. Do NOT reward speed, "
    "progress, smoothness or comfort.\n"
    "Respond with JSON only."
)


def build_preference_user_prompt(caption_a: str, caption_b: str) -> str:
    return (
        "Snapshot A:\n"
        f"{caption_a}\n\n"
        "Snapshot B:\n"
        f"{caption_b}\n\n"
        "Which snapshot has the LOWER risk of a collision or running off the road? "
        "Return JSON with keys: safer (0 if A is safer, 1 if B is safer), confidence "
        "(float 0..1, how sure you are), reasoning (one short sentence)."
    )


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, float(x))))


class PreferenceAnnotator(LocalLLMClient):
    """Anota pares de captions con una preferencia de seguridad."""

    def _system_prompt(self) -> str:
        return PREFERENCE_SYSTEM_PROMPT

    def annotate(self, caption_a: str, caption_b: str) -> PreferenceLabel:
        """Una unica llamada direccional (A primero, B segundo)."""
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": build_preference_user_prompt(caption_a, caption_b),
            },
        ]
        fmt = PreferenceResponse.model_json_schema()
        resp = None
        try:
            resp = self._chat(messages, fmt)
            content = resp["message"]["content"]
            parsed = PreferenceResponse.model_validate_json(content)
            return PreferenceLabel(
                safer=int(parsed.safer),
                confidence=_clamp01(parsed.confidence),
                reasoning=parsed.reasoning,
                ok=True,
                raw_response=content,
            )
        except Exception as exc:  # parse/validacion/conexion/timeout -> abstencion
            return PreferenceLabel(
                safer=-1,
                confidence=0.0,
                reasoning=f"fallback: {type(exc).__name__}: {exc}",
                ok=False,
                raw_response=_safe_raw(resp),
            )

    def annotate_symmetric(self, caption_a: str, caption_b: str) -> PreferenceLabel:
        """Dos llamadas en ordenes opuestos; acepta solo si coinciden.

        Cancela el sesgo de posicion: ``ab`` decide sobre (A,B) y ``ba`` sobre
        (B,A). 'A es mas seguro' equivale a ``ab.safer==0`` y a ``ba.safer==1``.
        Si discrepan (o cualquiera falla), se abstiene (par descartado).
        """
        ab = self.annotate(caption_a, caption_b)
        ba = self.annotate(caption_b, caption_a)
        if not (ab.ok and ba.ok):
            return PreferenceLabel(-1, 0.0, "fallback in one direction", False)

        a_safer_from_ab = ab.safer == 0
        a_safer_from_ba = ba.safer == 1
        if a_safer_from_ab != a_safer_from_ba:
            return PreferenceLabel(
                -1, 0.0, "position-bias disagreement between A,B and B,A", False
            )

        confidence = 0.5 * (ab.confidence + ba.confidence)
        return PreferenceLabel(
            safer=0 if a_safer_from_ab else 1,
            confidence=confidence,
            reasoning=ab.reasoning,
            ok=True,
            raw_response=ab.raw_response,
        )


def _safe_raw(resp) -> str:
    try:
        return resp["message"]["content"] if resp else ""
    except Exception:
        return ""
