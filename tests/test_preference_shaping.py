"""
test_preference_shaping.py - Shaping por preferencias potential-based (fase 3).

Sin CARLA ni torch: un FakeEnv y un reward model trucado (predict_np = obs[0])
verifican que CarlaRewardShaper aplica F = beta*(gamma*r_phi(s') - r_phi(s)),
con Phi(terminal)=0, encadenando s correctamente y sin tocar el reward cuando
esta desactivado.
"""

import os
import sys

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reward_shaper import CarlaRewardShaper  # noqa: E402

BETA = 0.1
GAMMA = 0.99


class _FakeEnv(gym.Env):
    """Env minimo: reset/step devuelven observaciones fijadas e info vacia."""

    def __init__(self, reset_obs, step_obs, done_list=None, reward=0.0):
        self.observation_space = Box(-np.inf, np.inf, (739,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, (2,), dtype=np.float32)
        self._reset_obs = reset_obs
        self._step_obs = list(step_obs)
        self._done = done_list or [False] * len(self._step_obs)
        self._reward = reward
        self._i = 0

    def reset(self, **kwargs):
        self._i = 0
        return self._reset_obs.copy(), {}

    def step(self, action):
        o, d = self._step_obs[self._i], self._done[self._i]
        self._i += 1
        return o.copy(), self._reward, d, False, {}


class _FakeRewardModel:
    """r_phi(obs) = obs[0]  (control directo del potencial en los tests)."""

    def predict_np(self, obs):
        return float(np.asarray(obs)[0])


def _obs(phi: float) -> np.ndarray:
    o = np.zeros(739, dtype=np.float32)
    o[0] = phi
    return o


def _shaper(env, *, active: bool):
    return CarlaRewardShaper(
        env,
        preference_reward_model=_FakeRewardModel() if active else None,
        preference_reward_coef=BETA if active else 0.0,
        preference_reward_gamma=GAMMA,
    )


# ── desactivado ────────────────────────────────────────────────────────────────


def test_disabled_when_coef_zero():
    env = _FakeEnv(_obs(1.0), [_obs(2.0)])
    sh = CarlaRewardShaper(
        env, preference_reward_model=_FakeRewardModel(), preference_reward_coef=0.0
    )
    sh.reset()
    _, _, _, _, info = sh.step(np.zeros(2, dtype=np.float32))
    assert info["preference_shaping"] == 0.0


def test_disabled_when_no_model():
    env = _FakeEnv(_obs(1.0), [_obs(2.0)])
    sh = CarlaRewardShaper(env, preference_reward_coef=BETA)  # sin modelo
    sh.reset()
    _, _, _, _, info = sh.step(np.zeros(2, dtype=np.float32))
    assert info["preference_shaping"] == 0.0


# ── shaping activo ───────────────────────────────────────────────────────────


def test_potential_term_value_non_terminal():
    phi0, phi1 = 1.5, 4.0
    sh = _shaper(_FakeEnv(_obs(phi0), [_obs(phi1)], done_list=[False]), active=True)
    sh.reset()
    _, _, _, _, info = sh.step(np.zeros(2, dtype=np.float32))
    assert info["preference_shaping"] == pytest.approx(BETA * (GAMMA * phi1 - phi0))


def test_terminal_uses_zero_potential():
    phi0, phi1 = 3.0, 9.0
    sh = _shaper(_FakeEnv(_obs(phi0), [_obs(phi1)], done_list=[True]), active=True)
    sh.reset()
    _, _, _, _, info = sh.step(np.zeros(2, dtype=np.float32))
    # done -> Phi(s')=0  =>  F = beta*(gamma*0 - phi0) = -beta*phi0
    assert info["preference_shaping"] == pytest.approx(BETA * (-phi0))


def test_prev_obs_chains_across_steps():
    phi0, phi1, phi2 = 1.0, 2.0, 5.0
    sh = _shaper(
        _FakeEnv(_obs(phi0), [_obs(phi1), _obs(phi2)], done_list=[False, False]),
        active=True,
    )
    sh.reset()
    _, _, _, _, i1 = sh.step(np.zeros(2, dtype=np.float32))
    _, _, _, _, i2 = sh.step(np.zeros(2, dtype=np.float32))
    assert i1["preference_shaping"] == pytest.approx(BETA * (GAMMA * phi1 - phi0))
    # el segundo paso usa phi1 como s (prev_obs se actualizo), NO phi0.
    assert i2["preference_shaping"] == pytest.approx(BETA * (GAMMA * phi2 - phi1))


def test_shaping_added_into_shaped_reward():
    # La diferencia entre shaper activo e inactivo en la MISMA transicion es
    # exactamente el termino de shaping (todo lo demas es identico).
    phi0, phi1 = 2.0, 1.0  # acercarse a un lider (phi baja) -> shaping negativo
    on = _shaper(_FakeEnv(_obs(phi0), [_obs(phi1)]), active=True)
    off = _shaper(_FakeEnv(_obs(phi0), [_obs(phi1)]), active=False)
    on.reset()
    off.reset()
    _, _, _, _, i_on = on.step(np.zeros(2, dtype=np.float32))
    _, _, _, _, i_off = off.step(np.zeros(2, dtype=np.float32))
    diff = i_on["shaped_reward"] - i_off["shaped_reward"]
    assert diff == pytest.approx(i_on["preference_shaping"])
    assert i_on["preference_shaping"] < 0.0  # cerrar distancia al lider penaliza
