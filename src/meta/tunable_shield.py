"""
tunable_shield.py - Superficie de parametros "destetables" del shield para el
meta-controlador LLM.

FUENTE UNICA DE VERDAD de QUE umbrales de CarlaAdaptiveHorizonShield puede mover
el meta-controlador, y de COMO un escalar ``shield_permissiveness`` in [0, 1] se
mapea a valores concretos de umbral.

    permissiveness = 1.0  -> STRICT     (baseline = comportamiento de produccion)
    permissiveness = 0.0  -> PERMISSIVE (maximo destete aun considerado seguro)

Cada valor se interpola linealmente entre sus extremos strict y permissive:

    value(p) = strict + (1 - p) * (permissive - strict)

de modo que p=1 reproduce EXACTAMENTE el shield actual y bajar p desteta al
agente. Como p se clampa a [0, 1] e interpolamos entre dos extremos fijos, cada
valor resuelto queda acotado por construccion: el LLM nunca puede empujar un
umbral fuera de su intervalo seguro.

Este modulo es deliberadamente CARLA-free (sin ``import carla``) para que toda la
pipeline meta y sus tests corran sin simulador. El shield llama a
``permissiveness_to_params`` desde su propio ``set_permissiveness``.

PARAMETROS INVARIANTES (nunca destetables) -- el suelo duro de seguridad -- se
listan en ``INVARIANT_PARAMS`` para documentacion/asserts y estan ausentes a
proposito de ``WEANABLE_ENDPOINTS``:
  - PED_EMERGENCY_M           (distancia de frenado de emergencia ante peaton)
  - rama front<0.5*thr -> -1.0 (freno duro de colision inminente)
  - EDGE_GUARD_MIN_NORM       (guardia dura de borde de carretera)
  - STALL_* / recovery        (recuperacion de deadlock, ortogonal al destete)
"""

from __future__ import annotations

# name -> (extremo strict @p=1, extremo permissive @p=0)
# Las claves con punto direccionan entradas anidadas de HORIZON_CONFIG:
# "horizon.<nivel>.<clave>".
WEANABLE_ENDPOINTS: dict[str, tuple[float, float]] = {
    # Umbrales LIDAR frontal/lateral de colision (menor = mas permisivo).
    # Extremos permissive conservadores: incluso totalmente desteted el shield
    # reacciona a distancia segura; el techo de crash del SafetyGovernor atrapa
    # cualquier regresion empirica.
    "front_threshold_base": (0.15, 0.08),
    "side_threshold_base": (0.02, 0.008),
    # Asistencia de mantenimiento de carril -- la superficie principal del shield
    # de conduccion (mayor = mas permisivo).
    "LATERAL_WARNING_OFFSET": (0.50, 0.75),
    "LATERAL_CRITICAL_OFFSET": (0.70, 0.90),
    "HEADING_WARNING_DEG": (10.0, 20.0),
    "HEADING_CRITICAL_DEG": (20.0, 35.0),
    "MAX_HEADING_DEV_RAD": (0.52, 0.80),
    "MAX_LATERAL_DRIFT_M": (1.75, 2.50),
    # Horizontes de verificacion de trayectoria del BicycleModel
    # (multiplier hacia abajo o lateral_thr hacia arriba = mas permisivo).
    "horizon.warning.threshold_multiplier": (1.5, 1.0),
    "horizon.critical.threshold_multiplier": (2.0, 1.0),
    "horizon.safe.lateral_thr": (0.55, 0.75),
    "horizon.warning.lateral_thr": (0.40, 0.60),
    "horizon.critical.lateral_thr": (0.30, 0.50),
}

INVARIANT_PARAMS: frozenset[str] = frozenset(
    {
        "PED_EMERGENCY_M",
        "EDGE_GUARD_MIN_NORM",
        "STALL_SPEED_KMH",
        "STALL_RECOVERY_EXIT_KMH",
        "STALL_PATIENCE",
        "STALL_RECOVERY_THROTTLE",
        "LATERAL_RECOVERY_THROTTLE",
        "emergency_brake",
    }
)

STRICT_PERMISSIVENESS: float = 1.0
MAX_PERMISSIVE: float = 0.0


def clamp_permissiveness(p: float) -> float:
    """Clampa el escalar de permisividad a su intervalo valido [0, 1]."""
    return float(min(1.0, max(0.0, float(p))))


def _interpolate(strict: float, permissive: float, p: float) -> float:
    return strict + (1.0 - p) * (permissive - strict)


def permissiveness_to_params(p: float) -> dict[str, float]:
    """Resuelve el escalar de permisividad a valores concretos de umbral.

    Devuelve un dict plano con las mismas claves que ``WEANABLE_ENDPOINTS``
    (claves con punto para entradas anidadas de HORIZON_CONFIG). Todos los
    valores quedan acotados entre los dos extremos porque ``p`` se clampa a
    [0, 1].
    """
    p = clamp_permissiveness(p)
    return {
        name: _interpolate(strict, permissive, p)
        for name, (strict, permissive) in WEANABLE_ENDPOINTS.items()
    }
