"""
safety_governor.py - Guardia DETERMINISTA sobre las propuestas de destete del LLM.

Este es el componente que hace de "la seguridad es la maxima prioridad" una
propiedad del CODIGO, no del modelo de lenguaje. El LLM solo PROPONE una
permisividad objetivo; el governor decide que (si algo) se aplica realmente,
bajo cuatro reglas duras:

  1. SUELO DE SEGURIDAD - si crash_rate u offroad_rate superan su techo, fuerza el
     shield MAS ESTRICTO (sube permisividad) sin importar la propuesta.
  2. RATCHET           - afloja SOLO cuando los KPIs de seguridad estan por debajo
     de sus objetivos (mas estrictos); en otro caso mantiene. El destete se gana,
     nunca se asume.
  3. PASO ACOTADO      - cambia la permisividad como mucho ``max_step`` por
     decision, para que cada desplazamiento de la politica de comportamiento sea
     pequeno frente a la tasa de adaptacion del critico.
  4. DWELL             - espera ``dwell_episodes`` tras cualquier cambio antes de
     volver a aflojar, para que el meta-bucle no oscile sobre ventanas ruidosas
     de 100 episodios.

Apretar (tighten) siempre se permite de inmediato (solo puede aumentar la
seguridad). El governor es una funcion PURA de (current_p, proposed_p, kpi,
episodes_since_change), por lo que es trivialmente testeable sin CARLA ni LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.meta.kpi_snapshot import KpiSnapshot
from src.meta.tunable_shield import clamp_permissiveness


@dataclass(frozen=True)
class GovernorConfig:
    crash_ceiling: float = 0.10  # superar -> force tighten
    offroad_ceiling: float = 0.15
    crash_target: float = 0.05  # por debajo -> aflojar permitido
    offroad_target: float = 0.08
    max_step: float = 0.10  # |Δp| maximo por decision voluntaria
    force_tighten_step: float = 0.20  # Δp de force-tighten (asimetrico, mas rapido)
    dwell_episodes: int = 100


@dataclass(frozen=True)
class GovernorDecision:
    applied_permissiveness: float
    # force_tighten | loosen | tighten | hold_dwell | hold_unsafe | hold
    verdict: str
    reason: str


class SafetyGovernor:
    def __init__(self, config: GovernorConfig | None = None):
        self.config = config or GovernorConfig()

    def govern(
        self,
        current_p: float,
        proposed_p: float,
        kpi: KpiSnapshot,
        episodes_since_change: int,
    ) -> GovernorDecision:
        c = self.config
        current_p = clamp_permissiveness(current_p)
        proposed_p = clamp_permissiveness(proposed_p)

        # Regla 1 - suelo duro de seguridad: el peligro empirico anula al LLM.
        if kpi.crash_rate > c.crash_ceiling or kpi.offroad_rate > c.offroad_ceiling:
            new_p = clamp_permissiveness(current_p + c.force_tighten_step)
            return GovernorDecision(
                new_p,
                "force_tighten",
                f"safety breach (crash={kpi.crash_rate:.3f}>{c.crash_ceiling} or "
                f"offroad={kpi.offroad_rate:.3f}>{c.offroad_ceiling}); tightening",
            )

        want_loosen = proposed_p < current_p - 1e-9
        want_tighten = proposed_p > current_p + 1e-9

        # Apretar es siempre seguro -> permitir de inmediato (acotado).
        if want_tighten:
            new_p = clamp_permissiveness(min(proposed_p, current_p + c.max_step))
            return GovernorDecision(new_p, "tighten", "LLM requested a tighter shield")

        # Regla 4 - dwell: sin aflojar mas hasta que el cambio se asiente.
        if want_loosen and episodes_since_change < c.dwell_episodes:
            return GovernorDecision(
                current_p,
                "hold_dwell",
                f"dwell not met ({episodes_since_change}<{c.dwell_episodes})",
            )

        # Regla 2 - ratchet: aflojar solo si esta probado seguro bajo objetivos.
        if want_loosen:
            safe_enough = (
                kpi.crash_rate < c.crash_target and kpi.offroad_rate < c.offroad_target
            )
            if not safe_enough:
                return GovernorDecision(
                    current_p,
                    "hold_unsafe",
                    f"not under targets (crash={kpi.crash_rate:.3f}/{c.crash_target}, "
                    f"offroad={kpi.offroad_rate:.3f}/{c.offroad_target}); holding",
                )
            # Regla 3 - paso acotado hacia la propuesta.
            new_p = clamp_permissiveness(max(proposed_p, current_p - c.max_step))
            return GovernorDecision(new_p, "loosen", "safe; loosening one bounded step")

        return GovernorDecision(current_p, "hold", "no change proposed")
