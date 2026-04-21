"""
Smoke tests for CarlaRewardShaper (pre-refactor structure).

Mirrors the per-step arithmetic of src/reward_shaper.py so the tests run
without a CARLA connection. Covers the essential behaviors:
  1. Normal driving → net positive shaped reward.
  2. Off-road → large net negative (terminal-like).
  3. Lane-change transition suppresses invasion / edge / drift penalties.
  4. Shield active → shield_not_activated_bonus is disabled; shield_intervention_pen
     fires only when shield_intervention_penalty > 0 and divergence > 0.
  5. Idle penalty fires only after the shield grace period expires.
  6. Surviving (struggling + slow) is still better than suiciding off-road.

Usage:
    python -m pytest tests/test_reward_balance.py -v
    python tests/test_reward_balance.py
"""

import math
import numpy as np


# Defaults taken from CarlaRewardShaper.__init__ (pre-refactor revert).
DEFAULTS = dict(
    target_speed_kmh=30.0,
    speed_weight=0.08,
    smoothness_weight=0.10,
    lane_centering_weight=0.15,
    heading_alignment_weight=0.04,
    lane_invasion_penalty=0.25,
    off_road_penalty=3.00,
    edge_warning_weight=0.30,
    progress_bonus_weight=0.30,
    wrong_heading_penalty=0.50,
    shield_intervention_penalty=0.0,
    speed_limit_margin=0.05,
    idle_penalty_weight=0.04,
    min_moving_speed_kmh=5.0,
    speed_gate_full_kmh=10.0,
    curvature_speed_scale=0.4,
    lane_drift_penalty_weight=0.08,
    alive_bonus=0.15,
    shield_grace_duration=10,
    shield_not_activated_bonus=0.02,
)

INTENTIONAL_STEER_THRESHOLD = 0.25


def _speed_gate(speed_kmh: float, p: dict) -> float:
    if p["speed_gate_full_kmh"] > p["min_moving_speed_kmh"]:
        return float(
            np.clip(
                (speed_kmh - p["min_moving_speed_kmh"])
                / (p["speed_gate_full_kmh"] - p["min_moving_speed_kmh"]),
                0.0,
                1.0,
            )
        )
    return float(np.clip(speed_kmh / max(p["speed_gate_full_kmh"], 1.0), 0.0, 1.0))


def simulate_step(
    speed_kmh: float,
    lateral_offset_norm: float,
    on_road: bool,
    lane_change_permitted: bool,
    shield_active: bool,
    dist_left_edge_norm: float,
    dist_right_edge_norm: float,
    lane_invasion: bool = False,
    on_edge_warning: float = 0.0,
    heading_error_deg: float = 0.0,
    heading_error_norm: float = 0.0,
    road_curvature_norm: float = 0.0,
    action_divergence: float = 0.0,
    steering_diff: float = 0.0,
    current_steering: float = 0.0,
    shield_grace_steps_before: int = 0,
    milestone_crossed: bool = False,
    **overrides,
) -> dict:
    """Replica the per-step computation of CarlaRewardShaper."""
    p = {**DEFAULTS, **overrides}
    effective_limit = p["target_speed_kmh"]
    speed_gate = _speed_gate(speed_kmh, p)

    in_lane_transition = lane_change_permitted and abs(lateral_offset_norm) > 0.5

    # 1. Speed reward
    curvature_factor = 1.0 - p["curvature_speed_scale"] * min(
        abs(road_curvature_norm) / 0.6, 1.0
    )
    curve_adjusted_limit = effective_limit * max(curvature_factor, 0.4)
    if on_road and speed_kmh > 0.5:
        speed_diff = abs(speed_kmh - curve_adjusted_limit)
        sigma = 0.35 * curve_adjusted_limit
        speed_reward = math.exp(-(speed_diff**2) / (2.0 * sigma**2)) * p["speed_weight"]
        speed_ceiling = effective_limit * (1.0 + p["speed_limit_margin"])
        if speed_kmh > speed_ceiling:
            overspeed = (speed_kmh - speed_ceiling) / effective_limit
            speed_reward -= overspeed * p["speed_weight"] * 0.8
    else:
        speed_reward = 0.0

    # 2. Lane centering
    min_edge_dist = min(dist_left_edge_norm, dist_right_edge_norm)
    centering_score = float(np.clip(min_edge_dist / 0.5, 0.0, 1.0))
    if in_lane_transition:
        lane_centering = 0.3 * p["lane_centering_weight"]
    else:
        lane_centering = (
            max(speed_gate, 0.3) * centering_score * p["lane_centering_weight"]
        )

    # 3. Heading alignment
    heading_alignment = (
        speed_gate
        * math.exp(-(heading_error_norm**2) / (2.0 * 0.40**2))
        * p["heading_alignment_weight"]
    )

    # 4. Smoothness
    smoothness_penalty = steering_diff * p["smoothness_weight"]

    # 5. Lane invasion
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

    # 6. Edge / road penalty
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

    # 7. Wrong heading
    if abs(heading_error_deg) > 90.0:
        wrong_heading_pen = (
            (abs(heading_error_deg) - 90.0) / 90.0 * p["wrong_heading_penalty"]
        )
    else:
        wrong_heading_pen = 0.0

    # 8. Progress milestone
    progress_bonus = p["progress_bonus_weight"] if milestone_crossed else 0.0

    # 9. Shield
    shield_intervention_pen = 0.0
    shield_not_activated_bonus = 0.0
    if shield_active and p["shield_intervention_penalty"] > 0.0:
        shield_intervention_pen = (action_divergence / 2.83) * p[
            "shield_intervention_penalty"
        ]
    elif not shield_active and p["shield_not_activated_bonus"] > 0.0:
        shield_not_activated_bonus = p["shield_not_activated_bonus"]

    # 10. Idle + grace bookkeeping
    grace_before = shield_grace_steps_before
    if shield_active and grace_before <= 0:
        grace_before = p["shield_grace_duration"]

    if speed_kmh < p["min_moving_speed_kmh"] and on_road and grace_before <= 0:
        idle_fraction = 1.0 - speed_kmh / max(p["min_moving_speed_kmh"], 1.0)
        idle_penalty = idle_fraction * p["idle_penalty_weight"]
    else:
        idle_penalty = 0.0

    grace_after = max(grace_before - 1, 0) if grace_before > 0 else 0

    # 11. Drift
    edge_asymmetry = abs(dist_left_edge_norm - dist_right_edge_norm)
    if in_lane_transition:
        drift_penalty = 0.0
    elif edge_asymmetry > 0.3 and min_edge_dist < 0.35:
        drift_penalty = (edge_asymmetry - 0.3) * p["lane_drift_penalty_weight"]
    else:
        drift_penalty = 0.0

    # 12. Alive bonus
    alive_bonus_val = p["alive_bonus"] * speed_gate if on_road else 0.0

    shaped_reward = (
        alive_bonus_val
        + speed_reward
        + lane_centering
        + heading_alignment
        + progress_bonus
        + shield_not_activated_bonus
        - smoothness_penalty
        - invasion_pen
        - road_penalty
        - wrong_heading_pen
        - shield_intervention_pen
        - idle_penalty
        - drift_penalty
    )

    return {
        "shaped_reward": shaped_reward,
        "alive_bonus": alive_bonus_val,
        "speed_reward": speed_reward,
        "lane_centering": lane_centering,
        "heading_alignment": heading_alignment,
        "progress_bonus": progress_bonus,
        "shield_not_activated_bonus": shield_not_activated_bonus,
        "smoothness_penalty": smoothness_penalty,
        "invasion_pen": invasion_pen,
        "road_penalty": road_penalty,
        "wrong_heading_pen": wrong_heading_pen,
        "shield_intervention_pen": shield_intervention_pen,
        "idle_penalty": idle_penalty,
        "drift_penalty": drift_penalty,
        "in_lane_transition": in_lane_transition,
        "speed_gate": speed_gate,
        "grace_after": grace_after,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_normal_driving_positive():
    """On-road, centered, at target speed → net positive shaped reward."""
    r = simulate_step(
        speed_kmh=30.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        shield_active=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    print(
        f"Normal: shaped={r['shaped_reward']:+.4f} "
        f"(alive={r['alive_bonus']:.3f} speed={r['speed_reward']:.3f} "
        f"center={r['lane_centering']:.3f} shield_bonus={r['shield_not_activated_bonus']:.3f})"
    )
    assert r["alive_bonus"] > 0.0, "Alive bonus must fire when moving on-road"
    assert r["speed_reward"] > 0.0, "Speed bonus must fire at target speed"
    assert r["lane_centering"] > 0.0, "Centering bonus must fire when centered"
    assert r["road_penalty"] == 0.0, "No edge penalty expected when far from edges"
    assert r["shaped_reward"] > 0.10, (
        f"Normal driving should be clearly positive, got {r['shaped_reward']:.4f}"
    )


def test_off_road_net_negative():
    """on_road=False applies the full off_road_penalty."""
    r = simulate_step(
        speed_kmh=5.0,
        lateral_offset_norm=0.0,
        on_road=False,
        lane_change_permitted=False,
        shield_active=False,
        dist_left_edge_norm=0.0,
        dist_right_edge_norm=0.0,
    )
    print(
        f"Off-road: shaped={r['shaped_reward']:+.4f} road_penalty={r['road_penalty']:.3f}"
    )
    assert r["road_penalty"] == DEFAULTS["off_road_penalty"]
    assert r["alive_bonus"] == 0.0, "Alive bonus must be zero off-road"
    assert r["shaped_reward"] < -1.0, (
        f"Off-road must be strongly negative, got {r['shaped_reward']:.4f}"
    )


def test_lane_change_suppresses_penalties():
    """Permitted lane change → invasion, edge and drift penalties suppressed."""
    r = simulate_step(
        speed_kmh=25.0,
        lateral_offset_norm=0.8,
        on_road=True,
        lane_change_permitted=True,
        shield_active=False,
        dist_left_edge_norm=0.1,
        dist_right_edge_norm=0.9,
        lane_invasion=True,
    )
    print(
        f"Lane change: shaped={r['shaped_reward']:+.4f} "
        f"in_transition={r['in_lane_transition']} "
        f"invasion={r['invasion_pen']:.3f} edge={r['road_penalty']:.3f} drift={r['drift_penalty']:.3f}"
    )
    assert r["in_lane_transition"] is True
    assert r["invasion_pen"] == 0.0
    assert r["road_penalty"] == 0.0
    assert r["drift_penalty"] == 0.0
    assert r["shaped_reward"] > 0.0, (
        f"Permitted lane change should be net positive, got {r['shaped_reward']:.4f}"
    )


def test_shield_active_suppresses_bonus():
    """shield_active=True → shield_not_activated_bonus must be zero."""
    r_no_shield = simulate_step(
        speed_kmh=20.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        shield_active=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    r_shield = simulate_step(
        speed_kmh=20.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        shield_active=True,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        action_divergence=0.5,
    )
    print(
        f"No shield bonus={r_no_shield['shield_not_activated_bonus']:.3f} "
        f"Shield bonus={r_shield['shield_not_activated_bonus']:.3f} "
        f"Shield pen={r_shield['shield_intervention_pen']:.3f}"
    )
    assert (
        r_no_shield["shield_not_activated_bonus"]
        == DEFAULTS["shield_not_activated_bonus"]
    )
    assert r_shield["shield_not_activated_bonus"] == 0.0
    # Default penalty is 0.0, so no intervention cost either
    assert r_shield["shield_intervention_pen"] == 0.0


def test_shield_intervention_penalty_when_configured():
    """When shield_intervention_penalty > 0 and divergence > 0 → penalty applies."""
    r = simulate_step(
        speed_kmh=20.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        shield_active=True,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        action_divergence=1.0,
        shield_intervention_penalty=0.25,
    )
    print(f"Shield pen (weighted): {r['shield_intervention_pen']:.4f}")
    assert r["shield_intervention_pen"] > 0.0
    assert r["shield_not_activated_bonus"] == 0.0


def test_idle_penalty_after_grace():
    """Idle penalty suppressed during grace, fires once grace expires."""
    # Inside grace period (grace_before > 0) → idle suppressed
    r_grace = simulate_step(
        speed_kmh=0.5,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        shield_active=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        shield_grace_steps_before=5,
    )
    # No grace → idle should fire
    r_no_grace = simulate_step(
        speed_kmh=0.5,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        shield_active=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        shield_grace_steps_before=0,
    )
    print(
        f"Grace idle={r_grace['idle_penalty']:.4f} no-grace idle={r_no_grace['idle_penalty']:.4f}"
    )
    assert r_grace["idle_penalty"] == 0.0, (
        "Idle penalty must be suppressed during grace"
    )
    assert r_no_grace["idle_penalty"] > 0.0, "Idle penalty must fire after grace"


def test_survival_beats_suicide():
    """A struggling-but-alive episode must outscore a short off-road suicide."""
    good = simulate_step(
        speed_kmh=30.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        shield_active=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    struggling = simulate_step(
        speed_kmh=6.0,
        lateral_offset_norm=0.5,
        on_road=True,
        lane_change_permitted=False,
        shield_active=True,
        dist_left_edge_norm=0.25,
        dist_right_edge_norm=0.75,
        action_divergence=0.8,
        shield_intervention_penalty=0.1,
    )
    offroad_step = simulate_step(
        speed_kmh=0.0,
        lateral_offset_norm=0.0,
        on_road=False,
        lane_change_permitted=False,
        shield_active=False,
        dist_left_edge_norm=0.0,
        dist_right_edge_norm=0.0,
    )
    # 200 steps alive (100 good + 100 struggling) vs 80 good + 1 offroad step + -20 terminal.
    survival_reward = 100 * good["shaped_reward"] + 100 * struggling["shaped_reward"]
    death_reward = 80 * good["shaped_reward"] + offroad_step["shaped_reward"] - 20.0
    print(f"Survival (200 steps): {survival_reward:+.2f}")
    print(f"Suicide  (80 steps):  {death_reward:+.2f}")
    assert survival_reward > death_reward, (
        f"Surviving should dominate: survival={survival_reward:.2f} vs suicide={death_reward:.2f}"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("REWARD SHAPER SMOKE TESTS (pre-refactor structure)")
    print("=" * 70)

    tests = [
        test_normal_driving_positive,
        test_off_road_net_negative,
        test_lane_change_suppresses_penalties,
        test_shield_active_suppresses_bonus,
        test_shield_intervention_penalty_when_configured,
        test_idle_penalty_after_grace,
        test_survival_beats_suicide,
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
