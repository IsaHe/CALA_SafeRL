"""
tunable_exposure.py - Superficie de "exposición a tráfico" para el meta-controlador.

Adaptador que implementa el mismo Protocol duck-typed que el shield destetable
(``get_permissiveness()`` / ``set_permissiveness(p) -> dict``), pero cuyo escalar
controla cuántos NPCs lentos se inyectan DELANTE del ego sobre su propia ruta
(``CarlaEnv.route_npc_count``), en vez de los umbrales del shield.

Motivación (diagnóstico learn_ttc4): el cuello de botella de la evitación de
colisiones es la EXPOSICIÓN, no el reward ni el shield — con 60 NPCs globales el
cono frontal del ego ve un vehículo en ~3% de episodios, así que ni un TTC
penalty ni un umbral del shield tienen estados sobre los que actuar. El dial con
palanca real es el tráfico que el agente efectivamente ENCUENTRA.

Convención del escalar (idéntica a tunable_shield, para reutilizar
SafetyGovernor/MetaController sin tocarlos):

    permissiveness = 1.0  -> SIN tráfico inyectado (= producción, máxima seguridad)
    permissiveness = 0.0  -> exposición máxima (``max_route_npcs`` leads en ruta)

Así "tighten" (subir p) = retirar tráfico = más seguro, y el suelo de seguridad
del governor (crash por encima del techo -> force_tighten) CORTA tráfico
automáticamente; "loosen" (bajar p) = añadir exposición, y se gana con el
ratchet (KPIs bajo objetivos) + dwell, nunca se asume.

El adaptador no toca CARLA: escribe ``route_npc_count`` en el CarlaEnv base y el
valor se consume en el siguiente ``reset()`` — la mutación ocurre en el boundary
de episodio, igual que el set_permissiveness del shield.
"""

from __future__ import annotations

from src.meta.tunable_shield import clamp_permissiveness

DEFAULT_MAX_ROUTE_NPCS = 6


def exposure_to_route_npcs(p: float, max_route_npcs: int) -> int:
    """Mapea el escalar de permisividad a un número de NPCs en ruta.

    Lineal y acotado por construcción: p=1 -> 0 NPCs, p=0 -> max_route_npcs.
    """
    p = clamp_permissiveness(p)
    return int(round((1.0 - p) * max(0, int(max_route_npcs))))


class TunableRouteExposure:
    """Adaptador Protocol-compatible sobre ``CarlaEnv.route_npc_count``.

    Duck-typed contra ``meta_controller.TunableShield``: MetaController /
    SafetyGovernor / AsyncMetaRunner funcionan sin cambios. Mantiene su propio
    escalar p (el env sólo conoce el conteo resuelto).
    """

    def __init__(self, base_env, max_route_npcs: int = DEFAULT_MAX_ROUTE_NPCS):
        self.base_env = base_env
        self.max_route_npcs = max(0, int(max_route_npcs))
        self._permissiveness = 1.0
        # Sincroniza el estado inicial: p=1.0 -> 0 NPCs inyectados.
        self.base_env.route_npc_count = exposure_to_route_npcs(
            self._permissiveness, self.max_route_npcs
        )

    def get_permissiveness(self) -> float:
        return self._permissiveness

    def set_permissiveness(self, p: float) -> dict:
        """Aplica p y devuelve los parámetros resueltos (para auditoría/logs)."""
        self._permissiveness = clamp_permissiveness(p)
        count = exposure_to_route_npcs(self._permissiveness, self.max_route_npcs)
        self.base_env.route_npc_count = count
        return {"route_npc_count": float(count)}

    @property
    def route_npc_count(self) -> int:
        return int(self.base_env.route_npc_count)
