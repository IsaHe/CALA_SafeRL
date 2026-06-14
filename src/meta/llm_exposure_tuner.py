"""
llm_exposure_tuner.py - Proponente LLM del curriculum de EXPOSICIÓN a tráfico.

Subclase de LLMShieldTuner que sólo cambia los prompts: el escalar p ya no
gobierna los umbrales del shield sino cuánto tráfico lento se inyecta en la ruta
del ego (ver tunable_exposure.py). Hereda intactas las cuatro capas de defensa
(schema-constrained decoding, determinismo, validación pydantic, clamp+fallback
no-op), y el SafetyGovernor determinista sigue vetando cada propuesta.

La tarea del LLM aquí NO es trivial de fijar con un schedule: debe subir la
exposición lo bastante rápido para que el agente acumule encuentros frontales
(encounter_rate es el KPI de progreso) pero retroceder ante señales de colapso
(crash_rate), con KPIs ruidosos en ventanas de 100 episodios. Es exactamente el
trade-off seguridad/aprendizaje que motivó el patrón proposer+governor.
"""

from __future__ import annotations

from src.meta.kpi_snapshot import KpiSnapshot
from src.meta.llm_tuner import LLMShieldTuner

EXPOSURE_SYSTEM_PROMPT = (
    "You are a SAFETY-FIRST curriculum controller for a reinforcement-learning "
    "driving agent in CARLA. The agent already lane-keeps well but rarely meets "
    "other vehicles head-on, so it cannot learn collision avoidance. You control "
    "how much SLOW LEAD TRAFFIC is injected directly on the agent's route: more "
    "traffic means more collision-avoidance experience but more risk. Your job "
    "is to raise that exposure as fast as SAFETY allows - and back off when "
    "crash rates degrade. You only propose a target; a deterministic governor "
    "enforces hard safety limits on top of your proposal. Respond with JSON only."
)


def build_exposure_user_prompt(snapshot: KpiSnapshot) -> str:
    return (
        "Exposure permissiveness is a scalar in [0,1]: 1.0 = NO injected route "
        "traffic (safest, but the agent learns nothing about vehicles), 0.0 = "
        "maximum injected lead traffic. LOWER it to give the agent more "
        "collision-avoidance experience; RAISE it if safety is degrading.\n\n"
        f"Current permissiveness: {snapshot.current_permissiveness:.3f}\n"
        f"Currently injected route NPCs: {snapshot.route_npc_count}\n"
        f"Metrics over the last {snapshot.window_episodes} episodes:\n"
        f"{snapshot.to_json()}\n\n"
        "Guidance:\n"
        "- crash_rate and offroad_rate are the hard priorities. If either is "
        "elevated, propose action='tighten' (higher target_permissiveness, less "
        "traffic).\n"
        "- encounter_rate is the fraction of episodes where the agent actually "
        "met a vehicle in its front cone (<35 m). It is the PROGRESS signal: if "
        "it is low and crash_rate is comfortably under control, the agent is "
        "not getting the experience it needs - propose a SMALL loosening "
        "(action='loosen', slightly lower target_permissiveness).\n"
        "- If encounter_rate is already high, prefer action='hold' and let the "
        "agent learn at the current exposure before adding more.\n"
        "- Otherwise action='hold'.\n"
        "- Move in small steps (~0.05-0.1). Never jump straight to 0.\n"
        "Return JSON with keys: target_permissiveness (float 0..1), action "
        "('loosen'|'hold'|'tighten'), reasoning (one sentence)."
    )


class LLMExposureTuner(LLMShieldTuner):
    """Mismo plumbing/defensas que LLMShieldTuner; sólo cambian los prompts."""

    def _system_prompt(self) -> str:
        return EXPOSURE_SYSTEM_PROMPT

    def _user_prompt(self, snapshot: KpiSnapshot) -> str:
        return build_exposure_user_prompt(snapshot)
