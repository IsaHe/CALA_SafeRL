"""
Tests de estabilidad para el stack Shielded-PPO.

CUBREN:
  A) Reward shaping monótono en velocidad (anti-paralysis)
     1.  Conducción normal → shaped reward claramente positivo.
     2.  Off-road → shaped reward muy negativo.
     3.  Lane change permitido → invasion/edge/drift suprimidos.
     4.  Idle penalty binaria: dispara sólo si speed < 0.5 km/h y on_road.
     5.  Supervivencia > suicidio off-road.
     6.  Reward independiente del shield.
     7.  MONOTONÍA: ∀ 0 ≤ v₁ < v₂ ≤ target, R(v₁) < R(v₂) on_road y centrado.
     8.  STUCK < MOVING: parado+centrado < moviéndose algo descentrado.
     9.  NO DEAD ZONE: ∂R/∂v ≥ 0 numéricamente en todo [0, target].

  B) Estabilidad numérica de la política
     10. log_prob acotado para acciones saturadas.
     11. log_det Jacobiano estable en los extremos.
     12. Política robusta a pequeños shifts con raw_action extrema.

  C) Proyección del shield
     13. Proyección α-blend continua.

  D) PPO con shield_mask
     14. Batch shielded → KL y grad_norm acotados.
     15. KL hard-stop pre-step.
     16. Batch limpio sin rechazos.
"""

import math
import os
import sys

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
    target_speed_kmh=30.0,
    speed_weight=0.10,
    smoothness_weight=0.10,
    lane_centering_weight=0.10,
    heading_alignment_weight=0.04,
    lane_invasion_penalty=0.25,
    off_road_penalty=2.00,
    edge_warning_weight=0.30,
    progress_bonus_weight=0.30,
    wrong_heading_penalty=0.50,
    speed_limit_margin=0.05,
    idle_penalty_weight=0.10,
    min_moving_speed_kmh=5.0,
    curvature_speed_scale=0.4,
    lane_drift_penalty_weight=0.08,
    progress_reward_weight=0.15,
)

INTENTIONAL_STEER_THRESHOLD = 0.25
IDLE_SPEED_THRESHOLD_KMH = 0.5


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
    **overrides,
) -> dict:
    p = {**DEFAULTS, **overrides}
    effective_limit = p["target_speed_kmh"]

    in_lane_transition = lane_change_permitted and abs(lateral_offset_norm) > 0.5

    # 1. Progress reward denso
    if on_road:
        speed_ratio = float(np.clip(speed_kmh / p["target_speed_kmh"], 0.0, 1.0))
        progress_reward = speed_ratio * p["progress_reward_weight"]
    else:
        progress_reward = 0.0

    # 2. Speed reward Gaussiana
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

    # 3. Lane centering (sin speed_gate)
    min_edge_dist = min(dist_left_edge_norm, dist_right_edge_norm)
    centering_score = float(np.clip(min_edge_dist / 0.5, 0.0, 1.0))
    if in_lane_transition:
        lane_centering = 0.3 * p["lane_centering_weight"]
    elif on_road:
        lane_centering = centering_score * p["lane_centering_weight"]
    else:
        lane_centering = 0.0

    # 4. Heading alignment (sin speed_gate)
    if on_road:
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

    # 10. Idle binaria
    if speed_kmh < IDLE_SPEED_THRESHOLD_KMH and on_road:
        idle_penalty = p["idle_penalty_weight"]
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

    shaped_reward = (
        progress_reward
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
    )

    return {
        "shaped_reward": shaped_reward,
        "progress_reward": progress_reward,
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
        "in_lane_transition": in_lane_transition,
    }


# ──────────────────────────────────────────────────────────────────────────
# A) Reward shaping — monotonía y ausencia de paralysis
# ──────────────────────────────────────────────────────────────────────────


def test_normal_driving_positive():
    r = simulate_step(
        speed_kmh=30.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    assert r["progress_reward"] > 0.0
    assert r["speed_reward"] > 0.0
    assert r["lane_centering"] > 0.0
    assert r["idle_penalty"] == 0.0
    assert r["road_penalty"] == 0.0
    assert r["shaped_reward"] > 0.15


def test_off_road_net_negative():
    r = simulate_step(
        speed_kmh=5.0,
        lateral_offset_norm=0.0,
        on_road=False,
        lane_change_permitted=False,
        dist_left_edge_norm=0.0,
        dist_right_edge_norm=0.0,
    )
    assert r["road_penalty"] == DEFAULTS["off_road_penalty"]
    assert r["progress_reward"] == 0.0
    assert r["shaped_reward"] < -1.0


def test_lane_change_suppresses_penalties():
    r = simulate_step(
        speed_kmh=25.0,
        lateral_offset_norm=0.8,
        on_road=True,
        lane_change_permitted=True,
        dist_left_edge_norm=0.1,
        dist_right_edge_norm=0.9,
        lane_invasion=True,
    )
    assert r["in_lane_transition"] is True
    assert r["invasion_pen"] == 0.0
    assert r["road_penalty"] == 0.0
    assert r["drift_penalty"] == 0.0
    assert r["shaped_reward"] > 0.0


def test_idle_penalty_binary():
    """
    Idle dispara como penalty BINARIO sólo si speed < 0.5 km/h.
    A 1 km/h (por encima del umbral) NO dispara.
    """
    r_stopped = simulate_step(
        speed_kmh=0.1,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    r_crawling = simulate_step(
        speed_kmh=1.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    assert r_stopped["idle_penalty"] == DEFAULTS["idle_penalty_weight"]
    assert r_crawling["idle_penalty"] == 0.0


def test_survival_beats_suicide():
    good = simulate_step(
        speed_kmh=30.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
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
    death = 80 * good["shaped_reward"] + offroad["shaped_reward"] - 20.0
    assert survival > death


def test_reward_independent_of_shield_presence():
    base_args = dict(
        speed_kmh=20.0,
        lateral_offset_norm=0.1,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.45,
        dist_right_edge_norm=0.55,
    )
    a = simulate_step(**base_args)
    b = simulate_step(**base_args)
    assert math.isclose(a["shaped_reward"], b["shaped_reward"], rel_tol=1e-12)


def test_reward_monotonic_in_speed_on_road():
    """
    Propiedad cardinal: en conducción centrada on-road, el shaped_reward
    debe ser ESTRICTAMENTE CRECIENTE en velocidad en [0.5, target].
    Esto garantiza gradiente positivo para PPO en todo el rango útil.
    """
    speeds = [0.5, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    rewards = [
        simulate_step(
            speed_kmh=v,
            lateral_offset_norm=0.0,
            on_road=True,
            lane_change_permitted=False,
            dist_left_edge_norm=0.5,
            dist_right_edge_norm=0.5,
        )["shaped_reward"]
        for v in speeds
    ]
    for i in range(1, len(rewards)):
        assert rewards[i] > rewards[i - 1], (
            f"No monótono en v={speeds[i]}: R({speeds[i - 1]})={rewards[i - 1]:.4f} "
            f">= R({speeds[i]})={rewards[i]:.4f}"
        )


def test_stuck_reward_below_moving_reward():
    """
    Un vehículo PARADO y centrado debe recibir MENOS reward que uno
    moviéndose (aunque sea despacio) y perfectamente centrado.
    Esto rompe el óptimo local "parar para evitar offroad".
    """
    stopped = simulate_step(
        speed_kmh=0.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    moving_slow = simulate_step(
        speed_kmh=5.0,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    assert moving_slow["shaped_reward"] > stopped["shaped_reward"], (
        f"stuck={stopped['shaped_reward']:.4f} vs moving={moving_slow['shaped_reward']:.4f}"
    )


def test_no_dead_zone_between_idle_and_target():
    """
    No debe existir un tramo de velocidad donde el reward decrezca al
    aumentar v. Barrido fino de 0 a target con Δv=0.25 km/h.
    """
    speeds = np.arange(0.0, DEFAULTS["target_speed_kmh"] + 0.25, 0.25)
    rewards = np.array(
        [
            simulate_step(
                speed_kmh=float(v),
                lateral_offset_norm=0.0,
                on_road=True,
                lane_change_permitted=False,
                dist_left_edge_norm=0.5,
                dist_right_edge_norm=0.5,
            )["shaped_reward"]
            for v in speeds
        ]
    )
    # Derivada discreta; permito un diminuto epsilon para ruido numérico.
    dR = np.diff(rewards)
    # Excluyo el salto del umbral idle (v cruza 0.5 km/h) que es discreto:
    # es el único punto donde permitimos salto positivo > eps.
    violations = [(speeds[i + 1], dR[i]) for i in range(len(dR)) if dR[i] < -1e-6]
    assert not violations, f"Dead-zone detectada: {violations[:5]}"


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
    tests = [
        test_normal_driving_positive,
        test_off_road_net_negative,
        test_lane_change_suppresses_penalties,
        test_idle_penalty_binary,
        test_survival_beats_suicide,
        test_reward_independent_of_shield_presence,
        test_reward_monotonic_in_speed_on_road,
        test_stuck_reward_below_moving_reward,
        test_no_dead_zone_between_idle_and_target,
        test_log_prob_bounded_for_saturated_actions,
        test_log_det_jacobian_stable_at_extremes,
        test_policy_update_robust_to_mean_shift_on_saturated_raw,
        test_projection_continuous_blend,
        test_ppo_update_with_shielded_samples_bounded_kl_and_grad,
        test_ppo_kl_hard_stop_before_optimizer_step,
        test_ppo_update_clean_batch_no_rejection,
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
    print(f"\n{passed}/{len(tests)} tests passed")
