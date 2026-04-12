"""
Test de balance de recompensas: verifica que los escenarios problemáticos
detectados en el diagnóstico ya no producen incentivos perversos.

Escenarios:
  1. Conducción normal (30 km/h, centrado) — debe ser el mayor reward/paso
  2. Paso tras intervención del shield (5 km/h, borde) — debe ser > 0 o ligeramente negativo
  3. Cambio de carril permitido (entre carriles, 25 km/h) — debe ser > 0 (no penalizado)
  4. Stuck con shield activo (0 km/h, shield loop) — idle_penalty DEBE aplicarse tras grace period

Uso:
    python -m pytest tests/test_reward_balance.py -v
    # o directamente:
    python tests/test_reward_balance.py
"""

import math
import numpy as np


def simulate_step_reward(
    speed_kmh: float,
    lateral_offset_norm: float,
    on_road: bool,
    lane_change_permitted: bool,
    shield_active: bool,
    dist_left_edge_norm: float,
    dist_right_edge_norm: float,
    lane_invasion: bool = False,
    heading_error_norm: float = 0.0,
    steering_diff: float = 0.0,
    road_curvature_norm: float = 0.0,
    action_divergence: float = 0.0,
    grace_steps_remaining: int = 0,
) -> dict:
    """Simula un paso del reward shaper con los parámetros por defecto actuales."""
    # Parámetros del reward shaper (valores por defecto actuales)
    speed_weight = 0.10
    lane_centering_weight = 0.10
    heading_alignment_weight = 0.04
    smoothness_weight = 0.10
    lane_invasion_penalty = 0.25
    off_road_penalty = 2.00
    edge_warning_weight = 0.30
    idle_penalty_weight = 0.04
    lane_drift_penalty_weight = 0.08
    alive_bonus = 0.15
    shield_intervention_penalty = 0.05
    min_moving_speed_kmh = 5.0
    speed_gate_full_kmh = 10.0
    target_speed_kmh = 50.0
    curvature_speed_scale = 0.4

    # Base reward (from CarlaEnv)
    speed_ms = speed_kmh / 3.6
    base_reward = max(speed_ms, 0.0) * 0.05 * 0.3

    # Speed gate
    if speed_gate_full_kmh > min_moving_speed_kmh:
        speed_gate = float(np.clip(
            (speed_kmh - min_moving_speed_kmh) /
            (speed_gate_full_kmh - min_moving_speed_kmh), 0.0, 1.0))
    else:
        speed_gate = float(np.clip(speed_kmh / max(speed_gate_full_kmh, 1.0), 0.0, 1.0))

    # 1. Speed reward
    curvature_factor = 1.0 - curvature_speed_scale * min(abs(road_curvature_norm) / 0.6, 1.0)
    curve_adjusted_limit = target_speed_kmh * max(curvature_factor, 0.4)
    if on_road and speed_kmh > 0.5:
        speed_diff = abs(speed_kmh - curve_adjusted_limit)
        sigma = 0.35 * curve_adjusted_limit
        speed_reward = math.exp(-(speed_diff**2) / (2.0 * sigma**2)) * speed_weight
    else:
        speed_reward = 0.0

    # Lane transition detection
    in_lane_transition = lane_change_permitted and abs(lateral_offset_norm) > 0.5

    # 2. Lane centering
    min_edge_dist = min(dist_left_edge_norm, dist_right_edge_norm)
    centering_score = float(np.clip(min_edge_dist / 0.5, 0.0, 1.0))
    if in_lane_transition:
        lane_centering = 0.3 * lane_centering_weight
    else:
        lane_centering = max(speed_gate, 0.3) * centering_score * lane_centering_weight

    # 3. Heading alignment
    heading_alignment = (
        speed_gate * math.exp(-(heading_error_norm**2) / (2.0 * 0.40**2))
        * heading_alignment_weight
    )

    # 4. Smoothness
    smoothness_penalty = steering_diff * smoothness_weight

    # 5. Invasion penalty
    if in_lane_transition:
        invasion_pen = 0.0
    elif lane_invasion:
        invasion_pen = lane_invasion_penalty * 0.75
    else:
        invasion_pen = 0.0

    # 6. Edge/road penalty
    if not on_road:
        road_penalty = off_road_penalty
    elif in_lane_transition:
        road_penalty = 0.0
    else:
        critical_edge = min(dist_left_edge_norm, dist_right_edge_norm)
        if critical_edge < 0.4:
            edge_proximity = ((0.4 - critical_edge) / 0.4) ** 2
            road_penalty = edge_proximity * edge_warning_weight
        else:
            road_penalty = 0.0

    # 9. Shield intervention penalty
    if shield_active and action_divergence > 0:
        shield_pen = action_divergence / 2.83 * shield_intervention_penalty
    else:
        shield_pen = 0.0

    # 10. Idle penalty (with grace period logic)
    if speed_kmh < min_moving_speed_kmh and on_road and grace_steps_remaining <= 0:
        idle_fraction = 1.0 - speed_kmh / max(min_moving_speed_kmh, 1.0)
        idle_pen = idle_fraction * idle_penalty_weight
    else:
        idle_pen = 0.0

    # 11. Drift penalty
    edge_asymmetry = abs(dist_left_edge_norm - dist_right_edge_norm)
    if in_lane_transition:
        drift_pen = 0.0
    elif edge_asymmetry > 0.3 and min_edge_dist < 0.35:
        drift_pen = (edge_asymmetry - 0.3) * lane_drift_penalty_weight
    else:
        drift_pen = 0.0

    # 12. Alive bonus
    alive_val = alive_bonus if on_road else 0.0

    shaped = (
        base_reward + alive_val + speed_reward + lane_centering
        + heading_alignment - smoothness_penalty - invasion_pen
        - road_penalty - shield_pen - idle_pen - drift_pen
    )

    return {
        "shaped_reward": shaped,
        "base_reward": base_reward,
        "alive_bonus": alive_val,
        "speed_reward": speed_reward,
        "lane_centering": lane_centering,
        "heading_alignment": heading_alignment,
        "smoothness_penalty": smoothness_penalty,
        "invasion_pen": invasion_pen,
        "road_penalty": road_penalty,
        "shield_pen": shield_pen,
        "idle_pen": idle_pen,
        "drift_pen": drift_pen,
        "in_lane_transition": in_lane_transition,
    }


def test_normal_driving_is_positive():
    """Conducción normal a 30 km/h, centrado — debe dar reward positivo alto."""
    r = simulate_step_reward(
        speed_kmh=30.0, lateral_offset_norm=0.0, on_road=True,
        lane_change_permitted=False, shield_active=False,
        dist_left_edge_norm=0.5, dist_right_edge_norm=0.5,
    )
    print(f"Normal driving:  {r['shaped_reward']:+.4f}  {r}")
    assert r["shaped_reward"] > 0.30, f"Expected > 0.30, got {r['shaped_reward']:.4f}"


def test_post_shield_not_heavily_negative():
    """Tras shield, lento y cerca del borde — debe ser > -0.10 (no incentiva suicidio)."""
    r = simulate_step_reward(
        speed_kmh=3.0, lateral_offset_norm=0.7, on_road=True,
        lane_change_permitted=False, shield_active=True,
        dist_left_edge_norm=0.15, dist_right_edge_norm=0.85,
        action_divergence=1.0, grace_steps_remaining=30,
    )
    print(f"Post-shield:     {r['shaped_reward']:+.4f}  {r}")
    assert r["shaped_reward"] > -0.15, f"Expected > -0.15, got {r['shaped_reward']:.4f}"


def test_lane_change_not_penalized():
    """Cambio de carril permitido a 25 km/h — debe dar reward > 0."""
    r = simulate_step_reward(
        speed_kmh=25.0, lateral_offset_norm=0.8, on_road=True,
        lane_change_permitted=True, shield_active=False,
        dist_left_edge_norm=0.1, dist_right_edge_norm=0.9,
        lane_invasion=True, heading_error_norm=0.1,
    )
    print(f"Lane change:     {r['shaped_reward']:+.4f}  {r}")
    assert r["in_lane_transition"] is True
    assert r["invasion_pen"] == 0.0, "Invasion should be suppressed during lane change"
    assert r["road_penalty"] == 0.0, "Edge penalty should be suppressed during lane change"
    assert r["drift_pen"] == 0.0, "Drift should be suppressed during lane change"
    assert r["shaped_reward"] > 0.0, f"Expected > 0, got {r['shaped_reward']:.4f}"


def test_stuck_gets_idle_penalty_after_grace():
    """Stuck con shield loop — idle_penalty DEBE aplicarse tras grace period."""
    # Grace period expired (grace_steps_remaining = 0), shield still active
    r = simulate_step_reward(
        speed_kmh=0.5, lateral_offset_norm=0.6, on_road=True,
        lane_change_permitted=False, shield_active=True,
        dist_left_edge_norm=0.2, dist_right_edge_norm=0.8,
        action_divergence=0.5, grace_steps_remaining=0,
    )
    print(f"Stuck post-grace: {r['shaped_reward']:+.4f}  {r}")
    assert r["idle_pen"] > 0.0, "Idle penalty should fire after grace period expires"


def test_suicide_not_optimal():
    """
    Escenario completo: sobrevivir 200 pasos luchando debe dar MÁS reward
    que morir (off-road) en el paso 80.
    """
    # Scenario A: 100 good steps + 100 struggling steps
    good = simulate_step_reward(
        speed_kmh=30.0, lateral_offset_norm=0.0, on_road=True,
        lane_change_permitted=False, shield_active=False,
        dist_left_edge_norm=0.5, dist_right_edge_norm=0.5,
    )
    struggling = simulate_step_reward(
        speed_kmh=3.0, lateral_offset_norm=0.5, on_road=True,
        lane_change_permitted=False, shield_active=True,
        dist_left_edge_norm=0.25, dist_right_edge_norm=0.75,
        action_divergence=0.8, grace_steps_remaining=0,
    )
    survival_reward = 100 * good["shaped_reward"] + 100 * struggling["shaped_reward"]

    # Scenario B: 80 good steps then off-road terminal (-20 from env + -2 from shaper)
    death_terminal = -22.0  # out_of_road_penalty=20 + shaper off_road_penalty=2
    death_reward = 80 * good["shaped_reward"] + death_terminal

    print(f"\nSurvival (200 steps): {survival_reward:+.1f}")
    print(f"Death (80 steps):     {death_reward:+.1f}")
    print(f"Survival advantage:   {survival_reward - death_reward:+.1f}")

    assert survival_reward > death_reward, (
        f"Suicide is still optimal! Survival={survival_reward:.1f} vs Death={death_reward:.1f}"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("REWARD BALANCE VALIDATION")
    print("=" * 70)

    tests = [
        test_normal_driving_is_positive,
        test_post_shield_not_heavily_negative,
        test_lane_change_not_penalized,
        test_stuck_gets_idle_penalty_after_grace,
        test_suicide_not_optimal,
    ]

    passed = 0
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        try:
            test()
            print("  PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")

    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{len(tests)} passed")
    print("=" * 70)
