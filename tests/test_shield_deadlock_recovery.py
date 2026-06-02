"""
test_shield_deadlock_recovery.py — Invariantes de la recuperación de deadlock
del CarlaAdaptiveHorizonShield (anti-stall, Phase C).

El deadlock que ataca: coche parado con error de heading que el shield no
podía resolver porque la acción de emergencia SÓLO frenaba. La recuperación
introduce un avance suave + steer hacia el carril, pero SÓLO en una ventana
demostrablemente segura. Estos tests fijan esa ventana para que un refactor
futuro no debilite silenciosamente el frenado proactivo del shield.

Se exercita `_build_emergency_action` y `_update_stall_counter` sin instanciar
el entorno (object.__new__ + atributos mínimos), así que sólo requieren el
paquete `carla` importable, no un servidor.
"""

import numpy as np
import pytest

carla = pytest.importorskip("carla")  # noqa: F841  (sólo se necesita el paquete)

from src.Adaptative_Shield.adaptive_horizon_shield import (  # noqa: E402
    CarlaAdaptiveHorizonShield as Shield,
)


def _make_shield() -> Shield:
    """Instancia el shield sin __init__ y fija los atributos que usa la lógica."""
    sh = object.__new__(Shield)
    sh.heading_correction_gain = 1.5
    sh.lane_correction_gain = 0.5
    sh.front_threshold_base = 0.15
    sh.side_threshold_base = 0.02
    sh.emergency_brake = -0.6
    sh._stall_steps = 0
    sh._last_emergency_was_recovery = False
    sh.shield_activations = 0
    sh.last_info = {}
    sh.stats = {
        "safe_steps": 0,
        "warning_steps": 0,
        "critical_steps": 0,
        "interventions_dynamic": 0,
        "interventions_static": 0,
        "interventions_pedestrian": 0,
        "interventions_recovery": 0,
        "interventions_by_horizon": {1: 0, 5: 0, 10: 0},
    }
    return sh


def _emergency_tb(
    sh: Shield,
    *,
    stall: int,
    front: float,
    lat: float = 0.4,
    hdg: float = 10.0,
    edge_l: float = 0.5,
    edge_r: float = 0.5,
    ped: float = 999.0,
) -> float:
    """Devuelve el throttle/brake (action[1]) de la acción de emergencia."""
    sh._stall_steps = stall
    sh.last_info = {
        "lateral_offset_norm": lat,
        "heading_error": hdg,
        "dist_left_edge_norm": edge_l,
        "dist_right_edge_norm": edge_r,
    }
    analysis = {
        "nearest_pedestrian_m": ped,
        "min_l_side_static": 0.5,
        "min_r_side_static": 0.5,
        "min_front_combined": front,
    }
    return float(sh._build_emergency_action(analysis)[1])


# ── La recuperación SÓLO engancha en la ventana segura ─────────────────────


def test_recovery_engages_when_stalled_and_safe():
    sh = _make_shield()
    tb = _emergency_tb(sh, stall=Shield.STALL_PATIENCE, front=0.5)
    assert tb == pytest.approx(Shield.STALL_RECOVERY_THROTTLE)
    assert tb > 0.0, "la recuperación debe AVANZAR para desencallar"


def test_no_recovery_before_patience():
    sh = _make_shield()
    tb = _emergency_tb(sh, stall=Shield.STALL_PATIENCE - 1, front=0.5)
    assert tb == pytest.approx(Shield.LATERAL_RECOVERY_THROTTLE)
    assert tb < 0.0, "sin paciencia agotada no debe avanzar"


# ── Cualquier peligro real tiene PRIORIDAD y sigue frenando ────────────────


def test_front_obstacle_brakes_even_when_stalled():
    sh = _make_shield()
    # front por debajo del umbral → freno de emergencia, NO recovery
    tb = _emergency_tb(sh, stall=Shield.STALL_PATIENCE, front=0.10)
    assert tb == pytest.approx(sh.emergency_brake)


def test_front_very_close_hard_brakes():
    sh = _make_shield()
    tb = _emergency_tb(sh, stall=Shield.STALL_PATIENCE, front=0.05)
    assert tb == pytest.approx(-1.0)


def test_near_road_edge_blocks_recovery():
    sh = _make_shield()
    # min_edge < EDGE_GUARD_MIN_NORM → nada de avanzar
    tb = _emergency_tb(
        sh,
        stall=Shield.STALL_PATIENCE,
        front=0.5,
        edge_l=Shield.EDGE_GUARD_MIN_NORM - 0.01,
    )
    assert tb == pytest.approx(Shield.LATERAL_RECOVERY_THROTTLE)
    assert tb < 0.0


def test_critical_lateral_brakes():
    sh = _make_shield()
    tb = _emergency_tb(
        sh,
        stall=Shield.STALL_PATIENCE,
        front=0.5,
        lat=Shield.LATERAL_CRITICAL_OFFSET + 0.05,
    )
    assert tb == pytest.approx(sh.emergency_brake)


def test_pedestrian_blocks_recovery():
    sh = _make_shield()
    tb = _emergency_tb(
        sh,
        stall=Shield.STALL_PATIENCE,
        front=0.5,
        ped=Shield.PED_EMERGENCY_M - 0.5,
    )
    assert tb == pytest.approx(Shield.LATERAL_RECOVERY_THROTTLE)
    assert tb < 0.0


# ── Contador de stall con histéresis ───────────────────────────────────────


def test_stall_counter_increments_below_threshold():
    sh = _make_shield()
    sh.last_info = {"speed_kmh": Shield.STALL_SPEED_KMH - 0.5}
    for expected in (1, 2, 3):
        sh._update_stall_counter(None)
        assert sh._stall_steps == expected


def test_stall_counter_resets_above_exit_speed():
    sh = _make_shield()
    sh._stall_steps = 20
    sh.last_info = {"speed_kmh": Shield.STALL_RECOVERY_EXIT_KMH + 1.0}
    sh._update_stall_counter(None)
    assert sh._stall_steps == 0


def test_stall_counter_hysteresis_holds_between_thresholds():
    sh = _make_shield()
    sh._stall_steps = 20
    mid_speed = 0.5 * (Shield.STALL_SPEED_KMH + Shield.STALL_RECOVERY_EXIT_KMH)
    sh.last_info = {"speed_kmh": mid_speed}
    sh._update_stall_counter(None)
    assert sh._stall_steps == 20, "en la banda de histéresis el contador se mantiene"


# ── La acción de emergencia siempre es válida ──────────────────────────────


def test_emergency_action_in_bounds():
    sh = _make_shield()
    sh._stall_steps = Shield.STALL_PATIENCE
    sh.last_info = {
        "lateral_offset_norm": 0.4,
        "heading_error": 10.0,
        "dist_left_edge_norm": 0.5,
        "dist_right_edge_norm": 0.5,
    }
    analysis = {
        "nearest_pedestrian_m": 999.0,
        "min_l_side_static": 0.5,
        "min_r_side_static": 0.5,
        "min_front_combined": 0.5,
    }
    act = sh._build_emergency_action(analysis)
    assert act.shape == (2,)
    assert np.all(act >= -1.0) and np.all(act <= 1.0)


# ── Telemetría de la recuperación (Safety/Semantic/Recovery_Activations) ───


def test_recovery_sets_flag_when_engaged():
    sh = _make_shield()
    _emergency_tb(sh, stall=Shield.STALL_PATIENCE, front=0.5)
    assert sh._last_emergency_was_recovery is True


def test_recovery_flag_false_when_braking():
    sh = _make_shield()
    _emergency_tb(sh, stall=Shield.STALL_PATIENCE, front=0.10)  # front blocked
    assert sh._last_emergency_was_recovery is False


def test_recovery_flag_false_when_not_stalled():
    sh = _make_shield()
    _emergency_tb(sh, stall=0, front=0.5)
    assert sh._last_emergency_was_recovery is False


def test_get_statistics_exposes_recovery():
    sh = _make_shield()
    sh.stats["safe_steps"] = 10
    sh.stats["interventions_recovery"] = 3
    out = sh.get_statistics()
    assert out["recovery_activations"] == 3
    assert out["recovery_rate"] == pytest.approx(3 / 10)
