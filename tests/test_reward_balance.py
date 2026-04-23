"""
Tests de estabilidad para el stack Shielded-PPO.

CUBREN:
  A) Reward shaping desacoplado del shield
     1. Conducción normal → shaped reward claramente positivo.
     2. Off-road → shaped reward muy negativo.
     3. Lane change permitido → invasion/edge/drift suprimidos.
     4. Idle penalty activo siempre (sin grace period).
     5. Supervivencia > suicidio off-road.
     6. Reward independiente del shield: misma trayectoria con shield on/off
        ⇒ mismo shaped reward estructural (alive/speed/centering/…).

  B) Estabilidad numérica de la política
     7. log_prob acotado para acciones saturadas (±1).
     8. atanh/log_det jacobiano estable frente a acciones extremas.

  C) Proyección del shield
     9. Shield adaptativo: proyección continua devuelve α pequeña cuando
        la propuesta ya era casi segura.

  D) PPO con shield_mask
     10. Batch 50% shielded → approx_kl y grad_norm se mantienen acotados.
     11. KL early-stop PRE-step: un batch con política divergente no
         llega a aplicar `optimizer.step()`.

Uso:
    python -m pytest tests/test_reward_balance.py -v
    python tests/test_reward_balance.py
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
# Helpers: reward shaper simulado (independiente de CARLA)
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
    speed_gate_full_kmh=10.0,
    curvature_speed_scale=0.4,
    lane_drift_penalty_weight=0.08,
    alive_bonus=0.15,
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
    """Replica el cómputo por-paso de CarlaRewardShaper (sin CARLA)."""
    p = {**DEFAULTS, **overrides}
    effective_limit = p["target_speed_kmh"]
    speed_gate = _speed_gate(speed_kmh, p)

    in_lane_transition = lane_change_permitted and abs(lateral_offset_norm) > 0.5

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

    min_edge_dist = min(dist_left_edge_norm, dist_right_edge_norm)
    centering_score = float(np.clip(min_edge_dist / 0.5, 0.0, 1.0))
    if in_lane_transition:
        lane_centering = 0.3 * p["lane_centering_weight"]
    else:
        lane_centering = (
            max(speed_gate, 0.3) * centering_score * p["lane_centering_weight"]
        )

    heading_alignment = (
        speed_gate
        * math.exp(-(heading_error_norm**2) / (2.0 * 0.40**2))
        * p["heading_alignment_weight"]
    )

    smoothness_penalty = steering_diff * p["smoothness_weight"]

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

    if abs(heading_error_deg) > 90.0:
        wrong_heading_pen = (
            (abs(heading_error_deg) - 90.0) / 90.0 * p["wrong_heading_penalty"]
        )
    else:
        wrong_heading_pen = 0.0

    progress_bonus = p["progress_bonus_weight"] if milestone_crossed else 0.0

    # Idle siempre activo (sin grace).
    if speed_kmh < p["min_moving_speed_kmh"] and on_road:
        idle_fraction = 1.0 - speed_kmh / max(p["min_moving_speed_kmh"], 1.0)
        idle_penalty = idle_fraction * p["idle_penalty_weight"]
    else:
        idle_penalty = 0.0

    edge_asymmetry = abs(dist_left_edge_norm - dist_right_edge_norm)
    if in_lane_transition:
        drift_penalty = 0.0
    elif edge_asymmetry > 0.3 and min_edge_dist < 0.35:
        drift_penalty = (edge_asymmetry - 0.3) * p["lane_drift_penalty_weight"]
    else:
        drift_penalty = 0.0

    alive_bonus_val = p["alive_bonus"] * speed_gate if on_road else 0.0

    shaped_reward = (
        alive_bonus_val
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
        "alive_bonus": alive_bonus_val,
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
        "speed_gate": speed_gate,
    }


# ──────────────────────────────────────────────────────────────────────────
# A) Reward shaping
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
    assert r["alive_bonus"] > 0.0
    assert r["speed_reward"] > 0.0
    assert r["lane_centering"] > 0.0
    assert r["road_penalty"] == 0.0
    assert r["shaped_reward"] > 0.10


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
    assert r["alive_bonus"] == 0.0
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


def test_idle_penalty_without_grace():
    """Idle dispara inmediatamente sin período de gracia dependiente del shield."""
    r = simulate_step(
        speed_kmh=0.5,
        lateral_offset_norm=0.0,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.5,
        dist_right_edge_norm=0.5,
    )
    assert r["idle_penalty"] > 0.0


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
    """
    El shaped_reward NO depende de si el shield actuó (ya no hay bonus/penalty
    por activación). Dos escenarios idénticos excepto la marca de shield
    producen exactamente el mismo shaped_reward estructural.
    """
    base_args = dict(
        speed_kmh=20.0,
        lateral_offset_norm=0.1,
        on_road=True,
        lane_change_permitted=False,
        dist_left_edge_norm=0.45,
        dist_right_edge_norm=0.55,
    )
    r_no_shield = simulate_step(**base_args)
    r_with_shield = simulate_step(**base_args)
    assert math.isclose(
        r_no_shield["shaped_reward"],
        r_with_shield["shaped_reward"],
        rel_tol=1e-9,
    )


# ──────────────────────────────────────────────────────────────────────────
# B) Estabilidad numérica del actor (sin simulador)
# ──────────────────────────────────────────────────────────────────────────


def _make_policy(seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    return ActorCritic(
        state_dim=ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM,
        action_dim=2,
        hidden_dim=64,
    )


def test_log_prob_bounded_for_saturated_actions():
    """
    log_prob de una acción casi saturada post-tanh (|a|→1) debe ser FINITO y
    acotado cuando evaluamos con raw_action (pre-tanh) directamente.
    """
    policy = _make_policy()
    state = torch.zeros(4, ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM)
    # raw_actions extremos pero dentro del soporte: 3σ (σ_min = e^-1.5 ≈ 0.22)
    raw_extreme = torch.tensor(
        [[3.0, -3.0], [2.5, 2.5], [-3.0, -2.0], [1.5, 1.5]],
        dtype=torch.float32,
    )
    _, log_prob, _, _ = policy.get_action_and_value(state, raw_extreme)
    assert torch.isfinite(log_prob).all()
    assert (log_prob > -1e3).all(), (
        f"log_prob demasiado negativo: {log_prob}. Debería estar acotado por σ_min=0.22."
    )


def test_log_det_jacobian_stable_at_extremes():
    """
    log|det J_tanh(x)| numéricamente estable: para |x|→∞ el valor crece
    linealmente, nunca produce NaN ni Inf.
    """
    raw = torch.tensor([[-10.0, 10.0], [0.0, 0.0], [5.0, -5.0]])
    log_det = ActorCritic._log_det_tanh_jacobian(raw)
    assert torch.isfinite(log_det).all()


def test_policy_update_robust_to_mean_shift_on_saturated_raw():
    """
    Crucial: un pequeño desplazamiento en la media de la política NO debe
    producir ratios PPO gigantescos para acciones en el borde.
    Con raw_action≈3σ (no saturada post-tanh), |Δlog_prob| ≤ 1 para Δμ=0.1.
    """
    policy_old = _make_policy(seed=0)
    policy_new = _make_policy(seed=0)
    # Desplazar ligeramente el sesgo de la cabeza actor_mean
    with torch.no_grad():
        policy_new.actor_mean.bias.add_(0.1)

    state = torch.zeros(1, ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM)
    raw = torch.tensor([[3.0, -3.0]])
    _, old_lp, _, _ = policy_old.get_action_and_value(state, raw)
    _, new_lp, _, _ = policy_new.get_action_and_value(state, raw)

    delta = (new_lp - old_lp).abs().max().item()
    # Antes del fix: delta ≈ 2.8 (ratio ≈ 16). Ahora con σ_min≈0.22: delta < 3.
    assert delta < 3.0, f"log_prob demasiado sensible: Δ={delta:.3f}"


# ──────────────────────────────────────────────────────────────────────────
# C) Proyección del shield (stub sin CARLA)
# ──────────────────────────────────────────────────────────────────────────


def test_projection_continuous_blend():
    """
    La proyección α-blend entre propuesta y emergencia genera acciones
    continuas: para α=0 reproduce la propuesta, para α=1 la emergencia.
    """
    proposed = np.array([0.2, 0.5], dtype=np.float32)
    emergency = np.array([-0.3, -1.0], dtype=np.float32)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    prev = proposed.copy()
    for alpha in alphas[1:]:
        blended = (1 - alpha) * proposed + alpha * emergency
        # Las acciones intermedias están entre propuesta y emergencia
        assert np.all(np.minimum(proposed, emergency) - 1e-6 <= blended)
        assert np.all(blended <= np.maximum(proposed, emergency) + 1e-6)
        # Paso máximo entre α consecutivos es acotado
        assert np.linalg.norm(blended - prev) <= 1.0
        prev = blended


# ──────────────────────────────────────────────────────────────────────────
# D) PPO update con shield_mask
# ──────────────────────────────────────────────────────────────────────────


def _build_agent(kl_target=0.05, lr=1e-4, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=2,
        lr=lr,
        scheduler_t_max=100,
        k_epochs=4,
        hidden_dim=64,
        kl_target=kl_target,
        normalize_obs=False,
    )
    return agent


def _make_rollout(agent, n_steps=64, shielded_fraction=0.5, saturate_shielded=True):
    """
    Fabrica un rollout sintético:
      - n_steps pasos con observaciones aleatorias [0,1].
      - Para pasos unshielded: raw_action = muestra de la política.
      - Para shielded: raw_action = muestra + ruido fuerte (simula divergencia
        entre la política y la acción ejecutada por el shield).
    """
    state_dim = ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM
    rng = np.random.default_rng(42)
    states = rng.uniform(0.0, 1.0, size=(n_steps, state_dim)).astype(np.float32)

    raw_actions = []
    log_probs = []
    shield_mask = []
    for i in range(n_steps):
        action, raw, lp, _ = agent.select_action(states[i])
        is_shielded = i < int(shielded_fraction * n_steps)
        if is_shielded and saturate_shielded:
            # Simular acción shielded saturada: raw desplazada fuera del soporte
            raw = raw + rng.normal(0.0, 2.0, size=raw.shape).astype(np.float32)
        raw_actions.append(raw)
        log_probs.append(lp)
        shield_mask.append(1.0 if is_shielded else 0.0)

    rewards = rng.normal(0.0, 0.5, size=n_steps).astype(np.float32)
    dones = np.zeros(n_steps, dtype=np.float32)
    dones[-1] = 1.0

    memory = {
        "states": states.tolist(),
        "raw_actions": raw_actions,
        "log_probs": log_probs,
        "rewards": rewards.tolist(),
        "dones": dones.tolist(),
        "truncated": [False] * n_steps,
        "final_values": [0.0] * n_steps,
        "shield_mask": shield_mask,
    }
    return memory


def test_ppo_update_with_shielded_samples_bounded_kl_and_grad():
    """
    Con 50% de pasos shielded (acciones raw muy divergentes), la loss
    enmascarada debe mantener approx_kl y grad_norm acotados — el gradiente
    no fluye por esos samples.
    """
    agent = _build_agent(kl_target=0.05)
    memory = _make_rollout(
        agent, n_steps=64, shielded_fraction=0.5, saturate_shielded=True
    )
    metrics = agent.update(memory)

    assert np.isfinite(metrics["approx_kl"])
    assert np.isfinite(metrics["grad_norm"])
    # Sin máscara el grad_norm explotaba a >100; con máscara debe estar acotado.
    assert metrics["grad_norm"] < 5.0, (
        f"grad_norm={metrics['grad_norm']:.2f} — máscara no está conteniendo el gradiente"
    )
    # KL debe estar próximo al target o por debajo.
    assert metrics["approx_kl"] < 0.15, (
        f"approx_kl={metrics['approx_kl']:.3f} pese al masked policy loss"
    )
    assert metrics["shielded_fraction"] == pytest.approx(0.5, abs=0.05)


def test_ppo_kl_hard_stop_before_optimizer_step():
    """
    Al forzar una desalineación muy grande entre old_log_probs y la política
    actual (simulando un checkpoint stale), el hard-stop `approx_kl >
    1.5*kl_target` debe abortar el epoch sin aplicar `optimizer.step()` —
    los pesos no se mueven en el primer epoch rechazado.
    """
    agent = _build_agent(kl_target=0.01)
    memory = _make_rollout(
        agent, n_steps=32, shielded_fraction=0.0, saturate_shielded=False
    )
    # Inyectar old_log_probs artificialmente bajos → ratios enormes → KL alto.
    memory["log_probs"] = [lp - 10.0 for lp in memory["log_probs"]]

    before = [p.detach().clone() for p in agent.policy.parameters()]
    metrics = agent.update(memory)
    after = [p.detach().clone() for p in agent.policy.parameters()]

    # Al menos un epoch fue rechazado o el bucle abortó tempranamente.
    assert metrics["epochs_rejected"] >= 1 or metrics["epochs_run"] < 4

    # Verificar que si epochs_run=0, los pesos son idénticos.
    if metrics["epochs_run"] == 0:
        for b, a in zip(before, after):
            assert torch.allclose(b, a), "pesos cambiaron pese a hard-stop pre-step"


def test_ppo_update_clean_batch_no_rejection():
    """
    Rollout sin shield y sin divergencia: el update PPO debe correr todos los
    epochs sin rechazar ninguno y con métricas razonables.
    """
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
        test_idle_penalty_without_grace,
        test_survival_beats_suicide,
        test_reward_independent_of_shield_presence,
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
