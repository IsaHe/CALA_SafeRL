"""
kpi_snapshot.py - Snapshot canonico de KPIs INDEPENDIENTES del shield para el
meta-controlador.

El LLM y el SafetyGovernor deben juzgar el destete con metricas que NO dependan
de cuanto esta ayudando el shield ahora mismo. El reward conformado por episodio
esta explicitamente CONFUNDIDO (cambia segun el shield retrocede aun con funcion
de reward fija), asi que se transporta solo como contexto de baja prioridad,
nunca como objetivo.

``build_kpi_snapshot`` es una funcion pura que ensambla el snapshot desde valores
que los bucles de entrenamiento YA calculan (las ventanas deslizantes de 100
episodios en main_train.py y ``shield.get_statistics()``). No hace ninguna
metrica nueva.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KpiSnapshot:
    episode: int
    window_episodes: int  # cuantos episodios resumen las tasas
    current_permissiveness: float

    # -- KPIs primarios de seguridad/competencia (el objetivo del governor) --
    crash_rate: float
    offroad_rate: float
    stuck_rate: float
    success_rate: float
    shielded_fraction: float  # fraccion de pasos que el shield modifico

    # -- Reflexion del shield (desglose por causa estilo Eureka) --
    mean_shield_alpha: float = 0.0
    intervention_dynamic: float = 0.0
    intervention_static: float = 0.0
    intervention_pedestrian: float = 0.0
    intervention_recovery: float = 0.0
    safe_rate: float = 0.0
    warning_rate: float = 0.0
    critical_rate: float = 0.0

    # -- Contexto (NO objetivos; el reward esta confundido bajo destete) --
    avg_reward_100: float = 0.0
    mean_speed_kmh: float = 0.0
    num_npc: int = 0

    # -- Curriculum de EXPOSICION (tuner de trafico en ruta) --
    # route_npc_count: NPCs lentos inyectados ahora en la ruta del ego.
    # encounter_rate : fraccion de episodios (ventana) con un vehiculo en el cono
    #   frontal (<~35 m) — el KPI de PROGRESO del tuner de exposicion.
    route_npc_count: int = 0
    encounter_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def build_kpi_snapshot(
    *,
    episode: int,
    current_permissiveness: float,
    crash_rate: float,
    offroad_rate: float,
    stuck_rate: float,
    success_rate: float,
    shielded_fraction: float,
    window_episodes: int = 100,
    shield_stats: dict | None = None,
    avg_reward_100: float = 0.0,
    mean_speed_kmh: float = 0.0,
    num_npc: int = 0,
    mean_shield_alpha: float = 0.0,
    route_npc_count: int = 0,
    encounter_rate: float = 0.0,
) -> KpiSnapshot:
    """Ensambla un KpiSnapshot desde valores ya calculados en el bucle.

    ``shield_stats`` es el dict devuelto por ``shield.get_statistics()``
    (CarlaAdaptiveHorizonShield); las claves ausentes caen a 0.0 para que los
    casos basic y sin-shield tambien produzcan un snapshot valido.
    """
    s = shield_stats or {}
    return KpiSnapshot(
        episode=int(episode),
        window_episodes=int(window_episodes),
        current_permissiveness=float(current_permissiveness),
        crash_rate=float(crash_rate),
        offroad_rate=float(offroad_rate),
        stuck_rate=float(stuck_rate),
        success_rate=float(success_rate),
        shielded_fraction=float(shielded_fraction),
        mean_shield_alpha=float(mean_shield_alpha),
        intervention_dynamic=float(s.get("interventions_dynamic", 0.0)),
        intervention_static=float(s.get("interventions_static", 0.0)),
        intervention_pedestrian=float(s.get("interventions_pedestrian", 0.0)),
        intervention_recovery=float(
            s.get("interventions_recovery", s.get("recovery_activations", 0.0))
        ),
        safe_rate=float(s.get("safe_rate", 0.0)),
        warning_rate=float(s.get("warning_rate", 0.0)),
        critical_rate=float(s.get("critical_rate", 0.0)),
        avg_reward_100=float(avg_reward_100),
        mean_speed_kmh=float(mean_speed_kmh),
        num_npc=int(num_npc),
        route_npc_count=int(route_npc_count),
        encounter_rate=float(encounter_rate),
    )
