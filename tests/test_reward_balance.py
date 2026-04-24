"""
Tests de estabilidad para el stack Shielded-PPO.

CUBREN:
  A) Reward shaping monótono en velocidad (anti-paralysis, sesiones 3-4)
     1.  Conducción normal → shaped reward claramente positivo.
     2.  Off-road → shaped reward muy negativo.
     3.  Lane change permitido → invasion/edge/drift suprimidos.
     4.  Idle penalty ESCALONADA (pico dead-stop, suave al arrancar).
     5.  Supervivencia > suicidio off-road.
     6.  Reward independiente del shield.
     7.  MONOTONÍA on-road + centrado, tras movimiento inicial.
     8.  STUCK < MOVING.
     9.  NO DEAD ZONE en [IDLE_THRESHOLD, target].
     10. PARADO es NEGATIVO con defaults del argparse de training.
     11. MOVIENDOSE v=3 >> PARADO v=0 con defaults del argparse.
     12. GATING de centering/heading por has_moved_recently.
     13. OUTCOME flags consistency (sum == 1).
     14. PROGRESS_REWARD satura a 10 km/h (sesión 4).
     15. ACCELERATION_REWARD sólo por dv>0 (sesión 4).
     16. ACTOR BIAS → throttle inicial >0.5 (sesión 4 cold-start fix).
     17. IDLE_PENALTY atenuada por throttle>0.3 en v<2 km/h (sesión 4).

  B) Estabilidad numérica del actor
     18-20. log_prob y shifts (con nuevo LOG_STD_MAX=-0.7 y bias throttle).

  C) Proyección del shield
     21. Blend continuo + emergencia peatonal vía α=1.

  D) PPO con shield_mask
     22-24. KL/grad bounded + hard-stop + clean batch.
"""

import math
import os
import sys
from collections import deque

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.PPO.ActorCritic import ActorCritic  # noqa: E402
from src.PPO.ppo_agent import PPOAgent  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# Helper: reward shaper simulado (espeja src/reward_shaper.py, sin CARLA)
# ──────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    target_speed_kmh=30.0,  # fallback cuando info no trae speed_limit
    speed_weight=0.10,
    smoothness_weight=0.10,
    lane_centering_weight=0.15,
    heading_alignment_weight=0.04,
    lane_invasion_penalty=0.25,
    off_road_penalty=1.00,
    edge_warning_weight=0.30,
    progress_bonus_weight=0.30,
    wrong_heading_penalty=0.50,
    speed_limit_margin=0.05,
    idle_penalty_weight=0.25,
    curvature_speed_scale=0.4,
    lane_drift_penalty_weight=0.08,
    progress_reward_weight=0.30,
    acceleration_reward_weight=0.08,
)

# Defaults reales que `main_train.py` pasa al shaper (si fueran distintos)
TRAINING_DEFAULTS = dict(DEFAULTS)

INTENTIONAL_STEER_THRESHOLD = 0.25
IDLE_SPEED_THRESHOLD_KMH = 0.5
IDLE_TIER_MID_KMH = 2.0
IDLE_TIER_HIGH_KMH = 5.0
IDLE_MULT_DEAD_STOP = 1.0
IDLE_MULT_CRAWL = 0.5
IDLE_MULT_SLOW = 0.2
MOVEMENT_WINDOW_STEPS = 5

# Sesión 4
PROGRESS_SATURATION_KMH = 10.0
IDLE_ACTION_ATTENUATION = 0.3
IDLE_ACTION_THROTTLE_THRESHOLD = 0.3
ACCELERATION_DELTA_CAP_KMH = 2.0

# Nuevos términos (diag plan) — cruce de línea sólida + coste por cambio de carril.
SOLID_INVASION_PENALTY = 5.0
LANE_CHANGE_COST = 0.05
LANE_CHANGE_COOLDOWN_STEPS = 20


def _idle_penalty_scaled(speed_kmh: float, weight: float) -> float:
    if speed_kmh < IDLE_SPEED_THRESHOLD_KMH:
        return weight * IDLE_MULT_DEAD_STOP
    if speed_kmh < IDLE_TIER_MID_KMH:
        return weight * IDLE_MULT_CRAWL
    if speed_kmh < IDLE_TIER_HIGH_KMH:
        return weight * IDLE_MULT_SLOW
    return 0.0


def simulate_step(
    speed_kmh: float,
    lateral_offset_norm: float,
    on_road: bool,
    lane_change_permitted: bool,
    dist_left_edge_norm: float,
    dist_right_edge_norm: float,
    lane_invasion: bool = False,
    on_edge_warning: float = 0.0,
    heading_error_deg: float = 0.0,
    heading_error_norm: float = 0.0,
    road_curvature_norm: float = 0.0,
    steering_diff: float = 0.0,
    current_steering: float = 0.0,
    milestone_crossed: bool = False,
    effective_limit_kmh: float = None,
    recent_speeds: list = None,
    prev_speed_kmh: float = 0.0,
    executed_throttle: float = 0.0,
    lane_change_event: bool = False,
    **overrides,
) -> dict:
    p = {**DEFAULTS, **overrides}
    # Límite efectivo: si no se da, usa el fallback `target_speed_kmh`.
    effective_limit = (
        float(effective_limit_kmh)
        if effective_limit_kmh is not None
        else p["target_speed_kmh"]
    )

    in_lane_transition = lane_change_permitted and abs(lateral_offset_norm) > 0.5

    # has_moved_recently: cualquier velocidad > umbral en la ventana.
    if recent_speeds is None:
        recent_speeds = [speed_kmh]
    window = deque(recent_speeds, maxlen=MOVEMENT_WINDOW_STEPS)
    has_moved = any(v >= IDLE_SPEED_THRESHOLD_KMH for v in window)

    # 1. Progress reward lineal — satura a PROGRESS_SATURATION_KMH=10 km/h
    # (sesión 4: ∂R/∂v amplificado en el tramo 0-10 para escapar el reposo).
    if on_road:
        speed_ratio = float(np.clip(speed_kmh / PROGRESS_SATURATION_KMH, 0.0, 1.0))
        progress_reward = speed_ratio * p["progress_reward_weight"]
    else:
        progress_reward = 0.0

    # 1b. Acceleration reward (sesión 4) — señal densa desde dv>0.
    if on_road:
        delta_v = speed_kmh - prev_speed_kmh
        acceleration_reward = (
            float(np.clip(delta_v, 0.0, ACCELERATION_DELTA_CAP_KMH))
            * p["acceleration_reward_weight"]
        )
    else:
        acceleration_reward = 0.0

    # 2. Speed reward Gaussiana sobre effective_limit
    curvature_factor = 1.0 - p["curvature_speed_scale"] * min(
        abs(road_curvature_norm) / 0.6, 1.0
    )
    curve_adjusted_limit = effective_limit * max(curvature_factor, 0.4)
    if on_road and speed_kmh > 0.1:
        speed_diff = abs(speed_kmh - curve_adjusted_limit)
        sigma = 0.35 * curve_adjusted_limit
        speed_reward = math.exp(-(speed_diff**2) / (2.0 * sigma**2)) * p["speed_weight"]
        speed_ceiling = effective_limit * (1.0 + p["speed_limit_margin"])
        if speed_kmh > speed_ceiling:
            overspeed = (speed_kmh - speed_ceiling) / effective_limit
            speed_reward -= overspeed * p["speed_weight"] * 0.8
    else:
        speed_reward = 0.0

    # 3. Lane centering (GATED por has_moved_recently)
    min_edge_dist = min(dist_left_edge_norm, dist_right_edge_norm)
    centering_score = float(np.clip(min_edge_dist / 0.5, 0.0, 1.0))
    if in_lane_transition:
        lane_centering = 0.3 * p["lane_centering_weight"]
    elif on_road and has_moved:
        lane_centering = centering_score * p["lane_centering_weight"]
    else:
        lane_centering = 0.0

    # 4. Heading alignment (GATED)
    if on_road and has_moved:
        heading_alignment = (
            math.exp(-(heading_error_norm**2) / (2.0 * 0.40**2))
            * p["heading_alignment_weight"]
        )
    else:
        heading_alignment = 0.0

    # 5. Smoothness
    smoothness_penalty = steering_diff * p["smoothness_weight"]

    # 6. Invasion
    if in_lane_transition:
        invasion_pen = 0.0
    elif lane_invasion:
        intentional = abs(current_steering) >= INTENTIONAL_STEER_THRESHOLD
        if not intentional:
            invasion_severity = min(abs(lateral_offset_norm), 1.0)
            invasion_pen = p["lane_invasion_penalty"] * (0.5 + 0.5 * invasion_severity)
        else:
            invasion_pen = 0.0
    else:
        invasion_pen = 0.0

    # 7. Road / off-road
    if not on_road:
        road_penalty = p["off_road_penalty"]
    elif in_lane_transition:
        road_penalty = 0.0
    else:
        critical_edge = min(dist_left_edge_norm, dist_right_edge_norm)
        edge_threshold = 0.4
        if critical_edge < edge_threshold:
            edge_proximity = ((edge_threshold - critical_edge) / edge_threshold) ** 2
            road_penalty = edge_proximity * p["edge_warning_weight"]
        elif on_edge_warning > 0.3:
            road_penalty = on_edge_warning * p["edge_warning_weight"] * 0.5
        else:
            road_penalty = 0.0

    # 8. Wrong heading
    if abs(heading_error_deg) > 90.0:
        wrong_heading_pen = (
            (abs(heading_error_deg) - 90.0) / 90.0 * p["wrong_heading_penalty"]
        )
    else:
        wrong_heading_pen = 0.0

    # 9. Progress milestone
    progress_bonus = p["progress_bonus_weight"] if milestone_crossed else 0.0

    # 10. Idle ESCALONADA con atenuación action-gated (sesión 4).
    if on_road:
        idle_penalty = _idle_penalty_scaled(speed_kmh, p["idle_penalty_weight"])
        if (
            speed_kmh < IDLE_TIER_MID_KMH
            and executed_throttle > IDLE_ACTION_THROTTLE_THRESHOLD
        ):
            idle_penalty *= IDLE_ACTION_ATTENUATION
    else:
        idle_penalty = 0.0

    # 11. Drift
    edge_asymmetry = abs(dist_left_edge_norm - dist_right_edge_norm)
    if in_lane_transition:
        drift_penalty = 0.0
    elif edge_asymmetry > 0.3 and min_edge_dist < 0.35:
        drift_penalty = (edge_asymmetry - 0.3) * p["lane_drift_penalty_weight"]
    else:
        drift_penalty = 0.0

    # 12. Cruce de línea SÓLIDA (hard) — aplica incluso en in_lane_transition.
    solid_invasion_pen = SOLID_INVASION_PENALTY if lane_invasion else 0.0

    # 13. Coste por evento de cambio de carril.
    lane_change_cost = LANE_CHANGE_COST if lane_change_event else 0.0

    shaped_reward = (
        progress_reward
        + acceleration_reward
        + speed_reward
        + lane_centering
        + heading_alignment
        + progress_bonus
        - smoothness_penalty
        - invasion_pen
        - road_penalty
        - wrong_heading_pen
        - idle_penalty
        - drift_penalty
        - solid_invasion_pen
        - lane_change_cost
    )

    return {
        "shaped_reward": shaped_reward,
        "progress_reward": progress_reward,
        "acceleration_reward": acceleration_reward,
        "speed_reward": speed_reward,
        "lane_centering": lane_centering,
        "heading_alignment": heading_alignment,
        "progress_bonus": progress_bonus,
        "smoothness_penalty": smoothness_penalty,
        "invasion_pen": invasion_pen,
        "road_penalty": road_penalty,
        "wrong_heading_pen": wrong_heading_pen,
        "idle_penalty": idle_penalty,
        "drift_penalty": drift_penalty,
        "solid_invasion_pen": solid_invasion_pen,
        "lane_change_cost": lane_change_cost,
        "in_lane_transition": in_lane_transition,
        "has_moved_recently": has_moved,
    }


# ──────────────────────────────────────────────────────────────────────────
# A) Reward shaping
# ──────────────────────────────────────────────────────────────────────────


def test_normal_driving_positive():
    """Centrado, a velocidad objetivo, tras moverse: claramente positivo."""
    r = simulate_step(
        speed_kmh=30.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        recent_speeds=[30.0] * MOVEMENT_WINDOW_STEPS,
    )
    assert r["progress_reward"] > 0.0
    assert r["speed_reward"] > 0.0
    assert r["lane_centering"] > 0.0
    assert r["idle_penalty"] == 0.0
    assert r["road_penalty"] == 0.0
    assert r["shaped_reward"] > 0.30


def test_off_road_net_negative():
    r = simulate_step(
        speed_kmh=5.0,
        lateral_offset_norm=0.0,
        on_road=False,
        lane_change_permitted=False,
        dist_left_edge_norm=0.0,
        dist_right_edge_norm=0.0,
        recent_speeds=[5.0] * MOVEMENT_WINDOW_STEPS,
    )
    assert r["road_penalty"] == DEFAULTS["off_road_penalty"]
    assert r["progress_reward"] == 0.0
    assert r["shaped_reward"] < -0.5


def test_lane_change_suppresses_penalties():
    """
    Cambio de carril LEGAL (línea discontinua, lane_invasion=False desde el
    sensor porque éste sólo dispara para sólidas). Las penalties graduales
    de invasión/borde/drift se suprimen durante el tránsito.
    """
    r = simulate_step(
        speed_kmh=25.0,
        lateral_offset_norm=0.8,
        on_road=True,
        lane_change_permitted=True,
        dist_left_edge_norm=0.1,
        dist_right_edge_norm=0.9,
        lane_invasion=False,
        recent_speeds=[25.0] * MOVEMENT_WINDOW_STEPS,
    )
    assert r["in_lane_transition"] is True
    assert r["invasion_pen"] == 0.0
    assert r["road_penalty"] == 0.0
    assert r["drift_penalty"] == 0.0
    assert r["solid_invasion_pen"] == 0.0
    assert r["shaped_reward"] > 0.0


def test_idle_penalty_tiered():
    """Idle penalty ESCALONADA por velocidad (no binaria)."""
    base = dict(
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    r_stop = simulate_step(speed_kmh=0.0, **base)
    r_crawl = simulate_step(speed_kmh=1.0, **base)
    r_slow = simulate_step(speed_kmh=3.0, **base)
    r_above = simulate_step(speed_kmh=6.0, **base)

    assert r_stop["idle_penalty"] == pytest.approx(
        DEFAULTS["idle_penalty_weight"] * IDLE_MULT_DEAD_STOP
    )
    assert r_crawl["idle_penalty"] == pytest.approx(
        DEFAULTS["idle_penalty_weight"] * IDLE_MULT_CRAWL
    )
    assert r_slow["idle_penalty"] == pytest.approx(
        DEFAULTS["idle_penalty_weight"] * IDLE_MULT_SLOW
    )
    assert r_above["idle_penalty"] == 0.0
    # Monotonía: a menor velocidad, mayor idle_penalty.
    assert (
        r_stop["idle_penalty"]
        > r_crawl["idle_penalty"]
        > r_slow["idle_penalty"]
        > r_above["idle_penalty"]
    )


def test_survival_beats_suicide():
    good = simulate_step(
        speed_kmh=30.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        recent_speeds=[30.0] * MOVEMENT_WINDOW_STEPS,
    )
    offroad = simulate_step(
        speed_kmh=0.0,
        lateral_offset_norm=0.0,
        on_road=False,
        lane_change_permitted=False,
        dist_left_edge_norm=0.0,
        dist_right_edge_norm=0.0,
    )
    survival = 200 * good["shaped_reward"]
    death = 80 * good["shaped_reward"] + offroad["shaped_reward"] - 10.0
    assert survival > death


def test_reward_independent_of_shield_presence():
    base_args = dict(
        speed_kmh=20.0,
        lateral_offset_norm=0.1,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.45,
        dist_right_edge_norm=0.55,
        recent_speeds=[20.0] * MOVEMENT_WINDOW_STEPS,
    )
    a = simulate_step(**base_args)
    b = simulate_step(**base_args)
    assert math.isclose(a["shaped_reward"], b["shaped_reward"], rel_tol=1e-12)


def test_reward_monotonic_in_speed_on_road():
    """Monotonía en [IDLE_THRESHOLD, target] cuando se ha movido recientemente."""
    speeds = [0.5, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    rewards = [
        simulate_step(
            speed_kmh=v,
            lateral_offset_norm=0.0,
            on_road=True,
            lane_change_permitted=False,
            dist_left_edge_norm=0.5,
            dist_right_edge_norm=0.5,
            recent_speeds=[v] * MOVEMENT_WINDOW_STEPS,
        )["shaped_reward"]
        for v in speeds
    ]
    for i in range(1, len(rewards)):
        assert rewards[i] > rewards[i - 1], (
            f"No monótono v={speeds[i]}: R({speeds[i - 1]})={rewards[i - 1]:.4f} "
            f">= R({speeds[i]})={rewards[i]:.4f}"
        )


def test_stuck_reward_below_moving_reward():
    """
    Parado y centrado (tras historial parado) < moviéndose despacio y
    centrado (tras historial moviéndose).
    """
    stopped = simulate_step(
        speed_kmh=0.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        recent_speeds=[0.0] * MOVEMENT_WINDOW_STEPS,
    )
    moving_slow = simulate_step(
        speed_kmh=5.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        recent_speeds=[5.0] * MOVEMENT_WINDOW_STEPS,
    )
    assert moving_slow["shaped_reward"] > stopped["shaped_reward"], (
        f"stuck={stopped['shaped_reward']:.4f} vs moving={moving_slow['shaped_reward']:.4f}"
    )


def test_no_dead_zone_between_idle_and_target():
    """No debe existir ningún tramo decreciente en [0.5, target]."""
    speeds = np.arange(0.5, DEFAULTS["target_speed_kmh"] + 0.25, 0.25)
    rewards = np.array(
        [
            simulate_step(
                speed_kmh=float(v),
                lateral_offset_norm=0.0,
                on_road=True,
                lane_change_permitted=False,
                dist_left_edge_norm=0.5,
                dist_right_edge_norm=0.5,
                recent_speeds=[float(v)] * MOVEMENT_WINDOW_STEPS,
            )["shaped_reward"]
            for v in speeds
        ]
    )
    dR = np.diff(rewards)
    violations = [(speeds[i + 1], dR[i]) for i in range(len(dR)) if dR[i] < -1e-6]
    assert not violations, f"Dead-zone detectada: {violations[:5]}"


def test_parked_reward_is_negative_with_training_defaults():
    """
    Con los defaults reales del argparse de training, tras estar parado
    y centrado durante MOVEMENT_WINDOW_STEPS pasos, el shaped_reward DEBE
    ser negativo (no bolsillo estable positivo).
    """
    r = simulate_step(
        speed_kmh=0.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        recent_speeds=[0.0] * MOVEMENT_WINDOW_STEPS,
        **TRAINING_DEFAULTS,
    )
    # lane_centering=0 (gated off), heading=0 (gated off), progress=0,
    # idle_penalty=−0.25. El único positivo es alive_bonus que NO está.
    assert r["lane_centering"] == 0.0, "Gating debe suprimir centering sin movimiento"
    assert r["heading_alignment"] == 0.0
    assert r["idle_penalty"] == TRAINING_DEFAULTS["idle_penalty_weight"]
    assert r["shaped_reward"] < -0.2, (
        f"Parado debería ser ~−0.25, obtuve {r['shaped_reward']:.3f}"
    )


def test_moving_dominates_parked_with_training_defaults():
    """
    Con defaults reales, moverse a 3 km/h (tras historial de movimiento)
    DEBE superar a estar parado por más de 0.3/paso.
    """
    stopped = simulate_step(
        speed_kmh=0.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        recent_speeds=[0.0] * MOVEMENT_WINDOW_STEPS,
        **TRAINING_DEFAULTS,
    )
    moving = simulate_step(
        speed_kmh=3.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        recent_speeds=[3.0] * MOVEMENT_WINDOW_STEPS,
        **TRAINING_DEFAULTS,
    )
    delta = moving["shaped_reward"] - stopped["shaped_reward"]
    assert delta > 0.3, f"Δ parado→v=3 = {delta:.3f}, debería ser > 0.3"


def test_idle_gating_requires_recent_motion():
    """
    Si TODOS los pasos recientes fueron parados, lane_centering y heading
    se suprimen. Un solo paso reciente con v≥umbral los reactiva.
    """
    no_motion = simulate_step(
        speed_kmh=0.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        recent_speeds=[0.0] * MOVEMENT_WINDOW_STEPS,
    )
    some_motion = simulate_step(
        speed_kmh=0.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        # Un paso con v=1 en la ventana
        recent_speeds=[0.0, 1.0, 0.0, 0.0, 0.0],
    )
    assert no_motion["lane_centering"] == 0.0
    assert no_motion["heading_alignment"] == 0.0
    assert some_motion["lane_centering"] > 0.0
    assert some_motion["heading_alignment"] > 0.0


def test_progress_reward_saturates_at_10_kmh():
    """Progress_reward satura a PROGRESS_SATURATION_KMH=10; de 10 a 30 km/h
    el componente permanece constante (ya está el speed_reward Gaussiano
    para diferenciar ese tramo)."""
    base = dict(
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    r5 = simulate_step(speed_kmh=5.0, recent_speeds=[5.0] * 5, **base)
    r10 = simulate_step(speed_kmh=10.0, recent_speeds=[10.0] * 5, **base)
    r20 = simulate_step(speed_kmh=20.0, recent_speeds=[20.0] * 5, **base)
    r30 = simulate_step(speed_kmh=30.0, recent_speeds=[30.0] * 5, **base)

    # Progress crece en 0-10, satura después.
    assert r5["progress_reward"] < r10["progress_reward"]
    assert r10["progress_reward"] == pytest.approx(DEFAULTS["progress_reward_weight"])
    assert r20["progress_reward"] == pytest.approx(DEFAULTS["progress_reward_weight"])
    assert r30["progress_reward"] == pytest.approx(DEFAULTS["progress_reward_weight"])

    # Aun así, el shaped_reward total sigue subiendo (speed_reward Gaussiano).
    assert r30["shaped_reward"] > r20["shaped_reward"] > r10["shaped_reward"]


def test_acceleration_reward_only_positive_dv():
    """acceleration_reward > 0 al acelerar, == 0 al decelerar o mantener."""
    base = dict(
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        recent_speeds=[5.0] * 5,
    )
    accel = simulate_step(speed_kmh=6.0, prev_speed_kmh=5.0, **base)
    same = simulate_step(speed_kmh=5.0, prev_speed_kmh=5.0, **base)
    decel = simulate_step(speed_kmh=3.0, prev_speed_kmh=5.0, **base)
    big_accel = simulate_step(speed_kmh=10.0, prev_speed_kmh=5.0, **base)

    assert accel["acceleration_reward"] == pytest.approx(
        1.0 * DEFAULTS["acceleration_reward_weight"]
    )
    assert same["acceleration_reward"] == 0.0
    assert decel["acceleration_reward"] == 0.0
    # Cap a ACCELERATION_DELTA_CAP_KMH (2.0)
    assert big_accel["acceleration_reward"] == pytest.approx(
        ACCELERATION_DELTA_CAP_KMH * DEFAULTS["acceleration_reward_weight"]
    )


def test_actor_bias_points_toward_forward_throttle():
    """El bias inicial del actor debe producir action[1]>0.5 en modo
    determinista — es la semilla que rompe el cold-start de la parálisis."""
    policy = _make_policy(seed=0)
    # En modo determinista el output es tanh(action_mean). Con state=zeros
    # las salidas dependen dominantemente de las biases (pesos init ~3e-3).
    state = torch.zeros(1, ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM)
    with torch.no_grad():
        features_in = policy._encode(state)
        features = policy.actor(features_in)
        action_mean = policy.actor_mean(features)
        action = torch.tanh(action_mean)

    # action[1] = throttle/brake; queremos claramente positivo (gas).
    assert action[0, 1].item() > 0.5, (
        f"Bias throttle debería dar tanh(~0.8)≈0.66, obtuve {action[0, 1].item():.3f}"
    )


def test_idle_penalty_action_gated():
    """
    Con speed<2 km/h y throttle>0.3, la idle_penalty se atenúa al 30%
    (el agente está intentando arrancar — la física aún no ha respondido).
    """
    base = dict(
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    # Sin throttle → idle_penalty completa del tramo crawl
    passive = simulate_step(speed_kmh=1.0, executed_throttle=0.0, **base)
    # Con throttle alto → atenuada ×0.3
    trying = simulate_step(speed_kmh=1.0, executed_throttle=0.8, **base)

    assert passive["idle_penalty"] == pytest.approx(
        DEFAULTS["idle_penalty_weight"] * IDLE_MULT_CRAWL
    )
    assert trying["idle_penalty"] == pytest.approx(
        passive["idle_penalty"] * IDLE_ACTION_ATTENUATION
    )
    # Fuera del tramo crítico (v>=2), el throttle NO atenúa.
    high_v = simulate_step(speed_kmh=3.0, executed_throttle=0.8, **base)
    baseline_high = simulate_step(speed_kmh=3.0, executed_throttle=0.0, **base)
    assert high_v["idle_penalty"] == baseline_high["idle_penalty"]


def test_solid_invasion_applies_hard_penalty():
    """
    Si `lane_invasion=True` (LaneInvasionSensor filtra a sólo sólidas),
    shaped_reward baja por SOLID_INVASION_PENALTY=5.0 en un solo step.
    """
    base = dict(
        speed_kmh=20.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        recent_speeds=[20.0] * MOVEMENT_WINDOW_STEPS,
    )
    no_invasion = simulate_step(lane_invasion=False, **base)
    with_invasion = simulate_step(lane_invasion=True, **base)

    assert no_invasion["solid_invasion_pen"] == 0.0
    assert with_invasion["solid_invasion_pen"] == SOLID_INVASION_PENALTY
    delta = no_invasion["shaped_reward"] - with_invasion["shaped_reward"]
    # Diferencia debe ser al menos la penalty sólida (puede haber pequeños
    # acoples por invasion_pen, pero 5.0 es holgadamente dominante).
    assert delta >= SOLID_INVASION_PENALTY - 0.4, (
        f"delta={delta:.3f} debería ≥ {SOLID_INVASION_PENALTY - 0.4}"
    )


def test_solid_invasion_applies_even_during_legal_lane_change():
    """
    Crítico: incluso si `lane_change_permitted=True`, cruzar una sólida
    (lane_invasion=True, que sólo dispara para sólidas) castiga. El
    waypoint puede estar desactualizado o la línea ser sólida-discontinua
    desde el lado equivocado.
    """
    r = simulate_step(
        speed_kmh=20.0,
        lateral_offset_norm=0.8,
        on_road=True,
        lane_change_permitted=True,  # transición "legal" según waypoint
        dist_left_edge_norm=0.1,
        dist_right_edge_norm=0.9,
        lane_invasion=True,
        recent_speeds=[20.0] * MOVEMENT_WINDOW_STEPS,
    )
    assert r["in_lane_transition"] is True
    assert r["invasion_pen"] == 0.0  # gradual suprimida por transición legal
    assert r["solid_invasion_pen"] == SOLID_INVASION_PENALTY  # hard no se suprime
    assert r["shaped_reward"] < -SOLID_INVASION_PENALTY + 1.0


def test_lane_change_cost_applied_on_event():
    """Un lane_change_event=True añade −LANE_CHANGE_COST al reward."""
    base = dict(
        speed_kmh=25.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        recent_speeds=[25.0] * MOVEMENT_WINDOW_STEPS,
    )
    no_event = simulate_step(lane_change_event=False, **base)
    with_event = simulate_step(lane_change_event=True, **base)

    assert no_event["lane_change_cost"] == 0.0
    assert with_event["lane_change_cost"] == LANE_CHANGE_COST
    delta = no_event["shaped_reward"] - with_event["shaped_reward"]
    assert delta == pytest.approx(LANE_CHANGE_COST, abs=1e-6)


def test_lane_change_cost_applies_even_when_legal():
    """
    El coste por cambio se aplica aunque `lane_change_permitted=True`.
    El propósito es desincentivar cambios innecesarios, no castigar solo
    los ilegales (eso ya lo hace solid_invasion).
    """
    r = simulate_step(
        speed_kmh=25.0,
        lateral_offset_norm=0.5,
        on_road=True,
        lane_change_permitted=True,
        dist_left_edge_norm=0.2,
        dist_right_edge_norm=0.8,
        lane_change_event=True,
        recent_speeds=[25.0] * MOVEMENT_WINDOW_STEPS,
    )
    assert r["lane_change_cost"] == LANE_CHANGE_COST


def test_outcome_flags_consistency():
    """
    Los 5 booleanos derivados de Outcome/Type deben sumar exactamente 1.
    """
    for outcome in range(5):
        flags = [
            1 if outcome == 0 else 0,
            1 if outcome == 1 else 0,
            1 if outcome == 2 else 0,
            1 if outcome == 3 else 0,
            1 if outcome == 4 else 0,
        ]
        assert sum(flags) == 1, f"outcome={outcome}: flags {flags}"


# ──────────────────────────────────────────────────────────────────────────
# B) Estabilidad numérica del actor
# ──────────────────────────────────────────────────────────────────────────


def _make_policy(seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(
        state_dim=ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM,
        action_dim=2,
        hidden_dim=64,
    )


def test_log_prob_bounded_for_saturated_actions():
    policy = _make_policy()
    state = torch.zeros(4, ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM)
    raw_extreme = torch.tensor(
        [[3.0, -3.0], [2.5, 2.5], [-3.0, -2.0], [1.5, 1.5]],
        dtype=torch.float32,
    )
    _, log_prob, _, _ = policy.get_action_and_value(state, raw_extreme)
    assert torch.isfinite(log_prob).all()
    assert (log_prob > -1e3).all()


def test_log_det_jacobian_stable_at_extremes():
    raw = torch.tensor([[-10.0, 10.0], [0.0, 0.0], [5.0, -5.0]])
    log_det = ActorCritic._log_det_tanh_jacobian(raw)
    assert torch.isfinite(log_det).all()


def test_policy_update_robust_to_mean_shift_on_saturated_raw():
    policy_old = _make_policy(seed=0)
    policy_new = _make_policy(seed=0)
    with torch.no_grad():
        policy_new.actor_mean.bias.add_(0.1)

    state = torch.zeros(1, ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM)
    raw = torch.tensor([[3.0, -3.0]])
    _, old_lp, _, _ = policy_old.get_action_and_value(state, raw)
    _, new_lp, _, _ = policy_new.get_action_and_value(state, raw)
    delta = (new_lp - old_lp).abs().max().item()
    assert delta < 3.0


# ──────────────────────────────────────────────────────────────────────────
# C) Proyección del shield
# ──────────────────────────────────────────────────────────────────────────


def test_projection_continuous_blend():
    proposed = np.array([0.2, 0.5], dtype=np.float32)
    emergency = np.array([-0.3, -1.0], dtype=np.float32)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    prev = proposed.copy()
    for alpha in alphas[1:]:
        blended = (1 - alpha) * proposed + alpha * emergency
        assert np.all(np.minimum(proposed, emergency) - 1e-6 <= blended)
        assert np.all(blended <= np.maximum(proposed, emergency) + 1e-6)
        assert np.linalg.norm(blended - prev) <= 1.0
        prev = blended


# ──────────────────────────────────────────────────────────────────────────
# D) PPO update con shield_mask
# ──────────────────────────────────────────────────────────────────────────


def _build_agent(kl_target=0.05, lr=1e-4, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM
    return PPOAgent(
        state_dim=state_dim,
        action_dim=2,
        lr=lr,
        scheduler_t_max=100,
        k_epochs=4,
        hidden_dim=64,
        kl_target=kl_target,
        normalize_obs=False,
    )


def _make_rollout(agent, n_steps=64, shielded_fraction=0.5, saturate_shielded=True):
    state_dim = ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM
    rng = np.random.default_rng(42)
    states = rng.uniform(0.0, 1.0, size=(n_steps, state_dim)).astype(np.float32)

    raw_actions = []
    log_probs = []
    shield_mask = []
    for i in range(n_steps):
        _, raw, lp, _ = agent.select_action(states[i])
        is_shielded = i < int(shielded_fraction * n_steps)
        if is_shielded and saturate_shielded:
            raw = raw + rng.normal(0.0, 2.0, size=raw.shape).astype(np.float32)
        raw_actions.append(raw)
        log_probs.append(lp)
        shield_mask.append(1.0 if is_shielded else 0.0)

    rewards = rng.normal(0.0, 0.5, size=n_steps).astype(np.float32)
    dones = np.zeros(n_steps, dtype=np.float32)
    dones[-1] = 1.0

    return {
        "states": states.tolist(),
        "raw_actions": raw_actions,
        "log_probs": log_probs,
        "rewards": rewards.tolist(),
        "dones": dones.tolist(),
        "truncated": [False] * n_steps,
        "final_values": [0.0] * n_steps,
        "shield_mask": shield_mask,
    }


def test_ppo_update_with_shielded_samples_bounded_kl_and_grad():
    agent = _build_agent(kl_target=0.05)
    memory = _make_rollout(
        agent, n_steps=64, shielded_fraction=0.5, saturate_shielded=True
    )
    metrics = agent.update(memory)
    assert np.isfinite(metrics["approx_kl"])
    assert np.isfinite(metrics["grad_norm"])
    assert metrics["grad_norm"] < 5.0
    assert metrics["approx_kl"] < 0.15
    assert metrics["shielded_fraction"] == pytest.approx(0.5, abs=0.05)


def test_ppo_kl_hard_stop_before_optimizer_step():
    agent = _build_agent(kl_target=0.01)
    memory = _make_rollout(
        agent, n_steps=32, shielded_fraction=0.0, saturate_shielded=False
    )
    memory["log_probs"] = [lp - 10.0 for lp in memory["log_probs"]]
    before = [p.detach().clone() for p in agent.policy.parameters()]
    metrics = agent.update(memory)
    after = [p.detach().clone() for p in agent.policy.parameters()]
    assert metrics["epochs_rejected"] >= 1 or metrics["epochs_run"] < 4
    if metrics["epochs_run"] == 0:
        for b, a in zip(before, after):
            assert torch.allclose(b, a)


def test_ppo_update_clean_batch_no_rejection():
    agent = _build_agent(kl_target=0.05)
    memory = _make_rollout(
        agent, n_steps=64, shielded_fraction=0.0, saturate_shielded=False
    )
    metrics = agent.update(memory)
    assert metrics["epochs_rejected"] == 0
    assert metrics["epochs_run"] >= 1
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert metrics["shielded_fraction"] == 0.0


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import inspect

    tests = [
        obj
        for name, obj in sorted(inspect.getmembers(sys.modules[__name__]))
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    for t in tests:
        print(f"--- {t.__name__} ---")
        try:
            t()
            print("  PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
