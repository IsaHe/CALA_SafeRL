"""
meta_controller.py - Orquesta KPI snapshot -> propuesta LLM -> governor -> aplicar.

Envuelve las tres piezas puras (LLMShieldTuner, SafetyGovernor, KpiSnapshot) en un
unico objeto que llaman los bucles de entrenamiento. El shield es duck-typed
(cualquier cosa que exponga ``get_permissiveness()`` / ``set_permissiveness(p)``),
asi que el controlador es totalmente testeable con un shield falso -- sin CARLA.

El Modo A (entre fases) y el Modo B (intra-run cada N episodios) usan ambos
``decide()``; el Modo B ademas regula con ``should_run(episode)`` y trackea dwell.
Cada decision produce un registro de auditoria para persistencia (la tabla SQLite
meta_decisions).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Protocol

from src.meta.kpi_snapshot import KpiSnapshot
from src.meta.llm_tuner import LLMShieldTuner, TunerProposal
from src.meta.safety_governor import GovernorDecision, SafetyGovernor
from src.meta.tunable_shield import clamp_permissiveness


class TunableShield(Protocol):
    def get_permissiveness(self) -> float: ...

    def set_permissiveness(self, p: float) -> dict: ...


@dataclass
class MetaDecision:
    episode: int
    snapshot: KpiSnapshot
    proposal: TunerProposal
    governor: GovernorDecision
    applied_permissiveness: float
    changed: bool

    def audit_record(self) -> dict:
        """Dict plano apto para insertar en la tabla meta_decisions."""
        return {
            "episode": self.episode,
            "applied_permissiveness": self.applied_permissiveness,
            "previous_permissiveness": self.snapshot.current_permissiveness,
            "changed": int(self.changed),
            "llm_ok": int(self.proposal.ok),
            "llm_target": self.proposal.target_permissiveness,
            "llm_action": self.proposal.action,
            "llm_reasoning": self.proposal.reasoning,
            "governor_verdict": self.governor.verdict,
            "governor_reason": self.governor.reason,
            "crash_rate": self.snapshot.crash_rate,
            "offroad_rate": self.snapshot.offroad_rate,
            "shielded_fraction": self.snapshot.shielded_fraction,
            "kpi_json": self.snapshot.to_json(indent=None),
            "raw_response": self.proposal.raw_response,
            "ts": time.time(),
        }


class MetaController:
    def __init__(
        self,
        tuner: LLMShieldTuner,
        governor: SafetyGovernor,
        interval_episodes: int = 100,
        enabled: bool = True,
    ):
        self.tuner = tuner
        self.governor = governor
        self.interval = max(1, int(interval_episodes))
        self.enabled = enabled
        self._last_change_episode = 0

    def should_run(self, episode: int) -> bool:
        """True si toca disparar el meta-paso en este episodio."""
        return self.enabled and episode > 0 and episode % self.interval == 0

    def decide(self, snapshot: KpiSnapshot, episode: int) -> MetaDecision:
        """Pipeline puro: tuner.propose -> governor.govern -> MetaDecision.

        No muta el shield (eso lo hace ``apply``), de modo que la decision puede
        calcularse en un hilo de fondo y aplicarse luego en el hilo principal en
        un boundary seguro.
        """
        current_p = clamp_permissiveness(snapshot.current_permissiveness)
        proposal = self.tuner.propose(snapshot)
        episodes_since = episode - self._last_change_episode
        gov = self.governor.govern(
            current_p=current_p,
            proposed_p=proposal.target_permissiveness,
            kpi=snapshot,
            episodes_since_change=episodes_since,
        )
        applied = clamp_permissiveness(gov.applied_permissiveness)
        changed = abs(applied - current_p) > 1e-9
        if changed:
            self._last_change_episode = episode
        return MetaDecision(
            episode=episode,
            snapshot=snapshot,
            proposal=proposal,
            governor=gov,
            applied_permissiveness=applied,
            changed=changed,
        )

    def apply(self, shield: TunableShield, decision: MetaDecision) -> dict:
        """Aplica la permisividad decidida al shield vivo (hilo principal)."""
        return shield.set_permissiveness(decision.applied_permissiveness)


class AsyncMetaRunner:
    """Ejecuta ``MetaController.decide`` en un hilo de fondo (single-flight).

    Modo B (intra-run): la llamada al LLM puede tardar segundos; ejecutarla en
    el bucle de 20 Hz lo bloquearia. Este runner la lanza en un worker y el hilo
    principal recoge el resultado en un boundary posterior, donde aplica la
    decision al shield. Solo hay UNA decision en vuelo a la vez.

    El worker NO toca el shield (eso lo hace el hilo principal via
    ``controller.apply``), de modo que no hay carrera sobre los atributos del
    shield: el worker solo calcula la propuesta/veredicto.
    """

    def __init__(self, controller: MetaController):
        self.controller = controller
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None

    def submit_if_idle(self, snapshot: KpiSnapshot, episode: int) -> bool:
        """Lanza una decision si no hay otra en vuelo. True si se lanzo."""
        if self._future is not None and not self._future.done():
            return False
        self._future = self._executor.submit(self.controller.decide, snapshot, episode)
        return True

    def poll(self) -> Optional[MetaDecision]:
        """Devuelve la MetaDecision lista (y la consume), o None si no hay."""
        if self._future is None or not self._future.done():
            return None
        fut, self._future = self._future, None
        try:
            return fut.result()
        except Exception:
            # decide() no deberia lanzar (tuner.propose ya hace fallback), pero
            # si lo hiciera, lo tragamos: ninguna excepcion del meta-loop debe
            # tumbar un entrenamiento de horas.
            return None

    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
