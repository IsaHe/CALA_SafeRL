"""
test_shield_learning.py - Tests de las palancas anti-dependencia del shield.

Cubren (sin CARLA salvo el bypass del shield, que usa importorskip):
  - Reward: penalización por dependencia del shield (R1) + gating en lane-change.
  - PPO: BC loss "shield-as-teacher" (R2) — 0 cuando coef=0 (comportamiento
    previo EXACTO), >0 y finito cuando coef>0; annealing.
  - ActorCritic.squashed_mean: forma y rango.
  - Shield bypass: pass-through total (requiere `carla`).
"""

import os
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.PPO.ActorCritic import ActorCritic  # noqa: E402
from src.PPO.ppo_agent import PPOAgent  # noqa: E402
from src.reward_shaper import CarlaRewardShaper  # noqa: E402

STATE_DIM = ActorCritic.LIDAR_TOTAL + ActorCritic.VECTOR_DIM


# ── Reward: shield-reliance penalty (R1) ─────────────────────────────────────


class _FakeInnerEnv(gym.Env):
    """Env mínimo que devuelve un info configurable (espeja la cadena interna)."""

    def __init__(self, info):
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (STATE_DIM,), np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._info = info

    def reset(self, **kwargs):
        return np.zeros(STATE_DIM, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(STATE_DIM, dtype=np.float32),
            0.0,
            False,
            False,
            dict(self._info),
        )


def _shaper(info, weight):
    sh = CarlaRewardShaper(_FakeInnerEnv(info), shield_reliance_weight=weight)
    sh.reset()
    return sh


_BASE_INFO = dict(
    speed_kmh=20.0,
    lateral_offset_norm=0.0,
    heading_error=0.0,
    on_road=True,
    lane_change_permitted=False,
    dist_left_edge_norm=0.5,
    dist_right_edge_norm=0.5,
    total_distance=10.0,
    speed_limit_kmh=30.0,
    shield_intensity=0.5,
)


def test_shield_reliance_penalty_subtracted():
    act = np.array([0.0, 0.5], dtype=np.float32)
    _, r_off, _, _, info_off = _shaper({**_BASE_INFO}, weight=0.0).step(act)
    _, r_on, _, _, info_on = _shaper({**_BASE_INFO}, weight=0.2).step(act)
    # penalización = weight * shield_intensity = 0.2 * 0.5 = 0.1
    assert info_off["shield_reliance_penalty"] == pytest.approx(0.0)
    assert info_on["shield_reliance_penalty"] == pytest.approx(0.1)
    assert (r_off - r_on) == pytest.approx(0.1, abs=1e-6)


def test_shield_reliance_penalty_scales_with_alpha():
    act = np.array([0.0, 0.5], dtype=np.float32)
    _, _, _, _, lo = _shaper({**_BASE_INFO, "shield_intensity": 0.25}, 0.4).step(act)
    _, _, _, _, hi = _shaper({**_BASE_INFO, "shield_intensity": 1.0}, 0.4).step(act)
    assert lo["shield_reliance_penalty"] == pytest.approx(0.1)
    assert hi["shield_reliance_penalty"] == pytest.approx(0.4)


def test_shield_reliance_penalty_gated_in_lane_change():
    # lane_change_permitted + |lat_off|>0.5 -> in_lane_transition -> pen=0.
    act = np.array([0.0, 0.5], dtype=np.float32)
    info = {**_BASE_INFO, "lane_change_permitted": True, "lateral_offset_norm": 0.7}
    _, _, _, _, out = _shaper(info, weight=0.5).step(act)
    assert out["in_lane_transition"] is True
    assert out["shield_reliance_penalty"] == pytest.approx(0.0)


def test_shield_reliance_zero_when_no_shield_info():
    # Sin shield_intensity en info (shield_type none) -> alpha=0 -> pen=0.
    act = np.array([0.0, 0.5], dtype=np.float32)
    info = {k: v for k, v in _BASE_INFO.items() if k != "shield_intensity"}
    _, _, _, _, out = _shaper(info, weight=0.5).step(act)
    assert out["shield_reliance_penalty"] == pytest.approx(0.0)


# ── PPO: shield-as-teacher BC loss (R2) ──────────────────────────────────────


def _build_agent(coef=0.0, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return PPOAgent(
        STATE_DIM,
        2,
        lr=1e-4,
        scheduler_t_max=100,
        k_epochs=2,
        hidden_dim=64,
        kl_target=0.05,
        normalize_obs=False,
        shield_imitation_coef=coef,
        shield_imitation_anneal_updates=10,
    )


def _make_rollout(agent, n=64, shielded_fraction=0.5):
    rng = np.random.default_rng(42)
    states = rng.uniform(0.0, 1.0, size=(n, STATE_DIM)).astype(np.float32)
    raw_actions, log_probs, shield_mask = [], [], []
    for i in range(n):
        _, raw, lp, _ = agent.select_action(states[i])
        shielded = i < int(shielded_fraction * n)
        if shielded:
            # acción "del shield" distinta de la media de la política -> BC>0.
            raw = raw + rng.normal(0.0, 1.5, size=raw.shape).astype(np.float32)
        raw_actions.append(raw)
        log_probs.append(lp)
        shield_mask.append(1.0 if shielded else 0.0)
    rewards = rng.normal(0.0, 0.5, size=n).astype(np.float32)
    dones = np.zeros(n, dtype=np.float32)
    dones[-1] = 1.0
    return {
        "states": states.tolist(),
        "raw_actions": raw_actions,
        "log_probs": log_probs,
        "rewards": rewards.tolist(),
        "dones": dones.tolist(),
        "truncated": [False] * n,
        "final_values": [0.0] * n,
        "shield_mask": shield_mask,
    }


def test_bc_loss_zero_when_disabled():
    agent = _build_agent(coef=0.0)
    metrics = agent.update(_make_rollout(agent))
    assert metrics["bc_loss"] == 0.0
    assert metrics["shield_imitation_coef_current"] == 0.0


def test_bc_loss_positive_and_finite_when_enabled():
    agent = _build_agent(coef=0.5)
    metrics = agent.update(_make_rollout(agent, shielded_fraction=0.5))
    assert np.isfinite(metrics["bc_loss"])
    assert metrics["bc_loss"] > 0.0
    assert np.isfinite(metrics["grad_norm"])
    assert metrics["shield_imitation_coef_current"] == pytest.approx(0.5)


def test_bc_loss_zero_when_no_shielded_steps():
    # coef>0 pero ningún paso shielded -> denominador clamp, BC=0.
    agent = _build_agent(coef=0.5)
    metrics = agent.update(_make_rollout(agent, shielded_fraction=0.0))
    assert metrics["bc_loss"] == pytest.approx(0.0)


def _agent_k1(steer_only=True, seed=1):
    # lr=0 -> la política no cambia durante el update, así el BC se mide sobre el
    # MISMO init para comparar configuraciones de forma justa.
    torch.manual_seed(seed)
    np.random.seed(seed)
    return PPOAgent(
        STATE_DIM,
        2,
        lr=0.0,
        scheduler_t_max=100,
        k_epochs=1,
        hidden_dim=64,
        kl_target=None,
        normalize_obs=False,
        shield_imitation_coef=0.5,
        shield_imitation_anneal_updates=10,
        shield_imitation_steer_only=steer_only,
    )


def test_bc_steer_only_excludes_throttle_term():
    # steer_only (default) imita sólo dim 0 -> BC <= full (que añade el término
    # de throttle). El throttle del shield = frenos; imitarlo colapsa a "parar".
    a_s = _agent_k1(steer_only=True)
    roll = _make_rollout(a_s, shielded_fraction=0.5)
    a_f = _agent_k1(steer_only=False)  # mismo seed -> mismo init y mismo rollout
    m_s = a_s.update(roll)
    m_f = a_f.update(roll)
    assert m_f["bc_loss"] > 0.0
    assert m_s["bc_loss"] <= m_f["bc_loss"] + 1e-9
    assert m_s["bc_loss"] < m_f["bc_loss"]  # el throttle difiere -> full estricto


def test_imitation_coef_anneals_to_zero():
    agent = _build_agent(coef=0.5)
    assert agent.shield_imitation_coef_current == pytest.approx(0.5)
    for _ in range(10):  # anneal_updates=10
        agent.step_imitation_decay()
    assert agent.shield_imitation_coef_current == pytest.approx(0.0)


def test_squashed_mean_shape_and_range():
    torch.manual_seed(0)
    policy = ActorCritic(STATE_DIM, 2, hidden_dim=64)
    states = torch.randn(5, STATE_DIM)
    mean = policy.squashed_mean(states)
    assert mean.shape == (5, 2)
    assert torch.all(mean >= -1.0) and torch.all(mean <= 1.0)


# ── Shield bypass (probe shield-OFF) — requiere `carla` ──────────────────────


def test_adaptive_shield_bypass_passes_action_through():
    pytest.importorskip("carla")
    from src.Adaptative_Shield.adaptive_horizon_shield import (
        CarlaAdaptiveHorizonShield as Shield,
    )

    sh = object.__new__(Shield)
    sh._bypass = False
    sh.env = _FakeInnerEnv({"speed_kmh": 5.0})
    sh.set_bypass(True)

    proposed = np.array([0.3, 0.7], dtype=np.float32)
    _, _, _, _, info = sh.step(proposed)
    assert info["shield_bypassed"] is True
    assert info["shield_activated"] is False
    assert info["shield_intensity"] == 0.0
    assert np.allclose(info["executed_action"], proposed)
