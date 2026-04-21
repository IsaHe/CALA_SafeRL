"""
Test de balance de recompensas para la jerarquía lexicográfica de dos niveles.

Verifica que la nueva arquitectura garantice:
  1. Conducción normal → efficiency_reward > 0 y safety_reward = 0
  2. Safety domina efficiency — off-road nunca cancelable por bonuses
  3. Cambio de carril permitido → sin penalizaciones de invasión/borde
  4. Shield intervention → smoothness_penalty suprimida (sin doble penalización)
  5. Supervivir luchando es mejor que suicidarse (sin incentivo al off-road)

Uso:
    python -m pytest tests/test_reward_balance.py -v
    python tests/test_reward_balance.py
"""

import math
import numpy as np


# ── Parámetros por defecto de CarlaRewardShaper ───────────────────────────────
DEFAULTS = dict(
    target_speed_kmh=30.0,
    speed_weight=0.10,
    progress_weight=0.40,
    comfort_weight=0.08,
    efficiency_cap=0.20,
    safety_edge_weight=0.40,
    lane_invasion_penalty=0.35,
    off_road_penalty=2.00,
    off_road_penalty_k=4.0,
    shield_intervention_penalty=0.25,
    speed_limit_margin=0.05,
    curvature_speed_scale=0.4,
    min_moving_speed_kmh=5.0,
    speed_gate_full_kmh=10.0,
    max_steps=1000,
)

INTENTIONAL_STEER_THRESHOLD = 0.25


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
    road_curvature_norm: float = 0.0,
    action_divergence: float = 0.0,
    steering_diff: float = 0.0,
    current_step: int = 500,
    **overrides,
) -> dict:
    """Simula un paso del CarlaRewardShaper con la jerarquía de dos niveles."""
    p = {**DEFAULTS, **overrides}

    effective_limit = p["target_speed_kmh"]

    # Speed gate
    if p["speed_gate_full_kmh"] > p["min_moving_speed_kmh"]:
        speed_gate = float(
            np.clip(
                (speed_kmh - p["min_moving_speed_kmh"])
                / (p["speed_gate_full_kmh"] - p["min_moving_speed_kmh"]),
                0.0, 1.0,
            )
        )
    else:
        speed_gate = float(np.clip(speed_kmh / max(p["speed_gate_full_kmh"], 1.0), 0.0, 1.0))

    in_lane_transition = lane_change_permitted and abs(lateral_offset_norm) > 0.5

    # ── LEVEL 1: SAFETY ────────────────────────────────────────────────
    # S1. Off-road / edge proximity
    if not on_road:
        remaining_fraction = max(0.0, (p["max_steps"] - current_step) / p["max_steps"])
        edge_penalty = p["off_road_penalty"] * (1.0 + p["off_road_penalty_k"] * remaining_fraction)
    elif in_lane_transition:
        edge_penalty = 0.0
    else:
        critical_edge = min(dist_left_edge_norm, dist_right_edge_norm)
        edge_threshold = 0.4
        if critical_edge < edge_threshold:
            edge_proximity = ((edge_threshold - critical_edge) / edge_threshold) ** 2
            edge_penalty = edge_proximity * p["safety_edge_weight"]
        elif on_edge_warning > 0.3:
            edge_penalty = on_edge_warning * p["safety_edge_weight"] * 0.5
        else:
            edge_penalty = 0.0

    # S2. Lane invasion
    if in_lane_transition or not lane_invasion:
        invasion_pen = 0.0
    else:
        invasion_pen = p["lane_invasion_penalty"] * 0.75  # mid-severity

    # S3. Shield intervention
    shield_pen = (action_divergence / 2.83) * p["shield_intervention_penalty"] if shield_active else 0.0

    safety_reward = -(edge_penalty + invasion_pen + shield_pen)

    # ── LEVEL 2: EFFICIENCY ────────────────────────────────────────────
    # E1. Speed
    curvature_factor = 1.0 - p["curvature_speed_scale"] * min(abs(road_curvature_norm) / 0.6, 1.0)
    curve_adjusted_limit = effective_limit * max(curvature_factor, 0.4)
    if on_road and speed_kmh > 0.5:
        speed_diff = abs(speed_kmh - curve_adjusted_limit)
        sigma = 0.35 * curve_adjusted_limit
        speed_reward = math.exp(-(speed_diff ** 2) / (2.0 * sigma ** 2)) * p["speed_weight"]
    else:
        speed_reward = 0.0

    # E2. Progress (milestone not triggered in unit tests)
    progress_bonus = 0.0

    # E3. Comfort: smoothness zeroed on shield intervention
    if shield_active:
        smoothness_penalty = 0.0
    else:
        smoothness_penalty = steering_diff * p["comfort_weight"]

    min_edge_dist = min(dist_left_edge_norm, dist_right_edge_norm)
    centering_score = float(np.clip(min_edge_dist / 0.5, 0.0, 1.0))
    if in_lane_transition:
        centering_bonus = 0.3 * p["comfort_weight"]
    else:
        centering_bonus = max(speed_gate, 0.3) * centering_score * p["comfort_weight"]
    comfort_reward = centering_bonus - smoothness_penalty

    # Clip for lexicographic dominance
    efficiency_raw = speed_reward + progress_bonus + comfort_reward
    efficiency_reward = float(np.clip(efficiency_raw, -p["efficiency_cap"], p["efficiency_cap"]))
    efficiency_clipped = efficiency_raw != efficiency_reward

    return {
        "safety_reward": safety_reward,
        "efficiency_reward": efficiency_reward,
        "efficiency_raw": efficiency_raw,
        "efficiency_clipped": efficiency_clipped,
        "edge_penalty": edge_penalty,
        "invasion_pen": invasion_pen,
        "shield_pen": shield_pen,
        "speed_reward": speed_reward,
        "comfort_reward": comfort_reward,
        "smoothness_penalty": smoothness_penalty,
        "in_lane_transition": in_lane_transition,
        "shaped_reward": safety_reward + efficiency_reward,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_normal_driving_positive():
    """Conducción normal a 30 km/h, centrado → safety=0, efficiency > 0."""
    r = simulate_step(
        speed_kmh=30.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        shield_active=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    print(f"Normal driving: safety={r['safety_reward']:+.4f}  efficiency={r['efficiency_reward']:+.4f}")
    assert r["safety_reward"] == 0.0, f"No safety event should trigger, got {r['safety_reward']}"
    assert r["efficiency_reward"] > 0.0, f"Efficiency should be positive, got {r['efficiency_reward']}"
    assert r["shaped_reward"] > 0.10, f"Overall reward should be clearly positive, got {r['shaped_reward']:.4f}"


def test_lexicographic_dominance_off_road():
    """Off-road penalty NUNCA cancelable por efficiency bonus."""
    # Best possible efficiency (at cap) vs any off-road event
    r = simulate_step(
        speed_kmh=30.0,
        lateral_offset_norm=0.0,
        on_road=False,
        lane_change_permitted=False,
        shield_active=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        current_step=500,
    )
    print(f"Off-road:  safety={r['safety_reward']:+.4f}  efficiency={r['efficiency_reward']:+.4f}")
    assert r["safety_reward"] < -DEFAULTS["efficiency_cap"], (
        f"Off-road safety penalty {r['safety_reward']:.4f} must be < -efficiency_cap ({-DEFAULTS['efficiency_cap']})"
    )
    assert r["shaped_reward"] < 0.0, f"Off-road must be net negative, got {r['shaped_reward']:.4f}"


def test_lexicographic_dominance_lane_invasion():
    """Lane invasion penalty > max efficiency_cap → safety dominates."""
    r = simulate_step(
        speed_kmh=30.0,
        lateral_offset_norm=0.5,
        on_road=True,
        lane_change_permitted=False,
        shield_active=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        lane_invasion=True,
    )
    print(f"Lane invasion: safety={r['safety_reward']:+.4f}  efficiency={r['efficiency_reward']:+.4f}")
    assert abs(r["invasion_pen"]) > DEFAULTS["efficiency_cap"], (
        f"invasion_pen {r['invasion_pen']:.4f} must exceed efficiency_cap {DEFAULTS['efficiency_cap']}"
    )
    assert r["shaped_reward"] < 0.0, f"Lane invasion must be net negative, got {r['shaped_reward']:.4f}"


def test_lane_change_suppresses_penalties():
    """Cambio de carril permitido → invasion, edge y drift suprimidos."""
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
    print(f"Lane change: safety={r['safety_reward']:+.4f}  in_transition={r['in_lane_transition']}")
    assert r["in_lane_transition"] is True
    assert r["invasion_pen"] == 0.0, "Invasion penalty must be suppressed during lane change"
    assert r["edge_penalty"] == 0.0, "Edge penalty must be suppressed during lane change"
    assert r["shaped_reward"] > 0.0, f"Lane change should be net positive, got {r['shaped_reward']:.4f}"


def test_shield_zeroes_smoothness():
    """Intervención del shield → smoothness_penalty = 0 (sin doble penalización)."""
    # Without shield: large steering diff → smoothness penalty
    r_no_shield = simulate_step(
        speed_kmh=20.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        shield_active=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        steering_diff=1.5,
        action_divergence=0.0,
    )
    # With shield: same steering diff but smoothness suppressed
    r_shield = simulate_step(
        speed_kmh=20.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        shield_active=True,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        steering_diff=1.5,
        action_divergence=0.5,
    )
    print(f"No-shield smoothness_penalty: {r_no_shield['smoothness_penalty']:+.4f}")
    print(f"Shield smoothness_penalty:    {r_shield['smoothness_penalty']:+.4f}")
    assert r_no_shield["smoothness_penalty"] > 0.0, "Smoothness should fire without shield"
    assert r_shield["smoothness_penalty"] == 0.0, "Smoothness must be zeroed when shield intervenes"


def test_efficiency_cap_enforced():
    """efficiency_reward está siempre dentro de [-efficiency_cap, +efficiency_cap].
    Se verifica con un cap artificialmente pequeño para forzar el clip.
    """
    tiny_cap = 0.05  # mucho menor que el efficiency_raw natural (~0.18)
    r = simulate_step(
        speed_kmh=30.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        shield_active=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
        efficiency_cap=tiny_cap,
    )
    print(f"Efficiency raw={r['efficiency_raw']:+.4f}  clipped={r['efficiency_reward']:+.4f}  cap=±{tiny_cap}")
    assert r["efficiency_clipped"] is True, "Should have clipped with tiny_cap"
    assert -tiny_cap <= r["efficiency_reward"] <= tiny_cap, (
        f"efficiency_reward {r['efficiency_reward']:.4f} outside [-{tiny_cap}, {tiny_cap}]"
    )


def test_survival_beats_suicide():
    """
    Sobrevivir 200 pasos luchando es mejor que ir off-road en el paso 80.
    """
    good = simulate_step(
        speed_kmh=30.0, lateral_offset_norm=0.0, on_road=True,
        lane_change_permitted=False, shield_active=False,
        dist_left_edge_norm=0.5, dist_right_edge_norm=0.5,
        current_step=100,
    )
    struggling = simulate_step(
        speed_kmh=4.0, lateral_offset_norm=0.5, on_road=True,
        lane_change_permitted=False, shield_active=True,
        dist_left_edge_norm=0.25, dist_right_edge_norm=0.75,
        action_divergence=0.8, current_step=150,
    )
    survival_reward = 100 * good["shaped_reward"] + 100 * struggling["shaped_reward"]

    # Off-road at step 80: CarlaEnv terminal penalty (-20) + shaper off-road
    offroad_step = simulate_step(
        speed_kmh=0.0, lateral_offset_norm=0.0, on_road=False,
        lane_change_permitted=False, shield_active=False,
        dist_left_edge_norm=0.0, dist_right_edge_norm=0.0,
        current_step=80,
    )
    death_reward = 80 * good["shaped_reward"] + offroad_step["shaped_reward"] - 20.0

    print(f"\nSurvival (200 steps): {survival_reward:+.2f}")
    print(f"Suicide  (80 steps):  {death_reward:+.2f}")
    print(f"Survival advantage:   {survival_reward - death_reward:+.2f}")

    assert survival_reward > death_reward, (
        f"Suicide is still optimal! Survival={survival_reward:.2f} vs Suicide={death_reward:.2f}"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("REWARD HIERARCHY VALIDATION (two-level lexicographic)")
    print("=" * 70)

    tests = [
        test_normal_driving_positive,
        test_lexicographic_dominance_off_road,
        test_lexicographic_dominance_lane_invasion,
        test_lane_change_suppresses_penalties,
        test_shield_zeroes_smoothness,
        test_efficiency_cap_enforced,
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
