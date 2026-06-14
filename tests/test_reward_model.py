"""
test_reward_model.py - Reward model r_phi por preferencias (fase 2).

Sin CARLA ni Ollama: forma/IO del modelo, signo de la perdida Bradley-Terry, y
que el bucle de entrenamiento separa 'via libre' (seguro) de 'lider cercano'
(inseguro) en datos sinteticos.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.imitation.reward_model import (  # noqa: E402
    REWARD_FEATURE_DIM,
    RewardModel,
    bradley_terry_loss,
    load_reward_model,
    save_reward_model,
    train_bradley_terry,
)


def _free_obs() -> np.ndarray:
    o = np.zeros(739, dtype=np.float32)
    o[0:720] = 1.0
    o[737] = 30.0 / 130.0  # speed_limit_norm
    o[738] = 1.0  # speed_ratio -> 30 km/h
    return o


def _close_lead() -> np.ndarray:
    o = _free_obs()
    o[240 + 0] = 0.1  # 5 m directamente delante
    return o


def _make_dataset(n: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    obs_a = np.zeros((n, 739), dtype=np.float32)
    obs_b = np.zeros((n, 739), dtype=np.float32)
    safer = np.zeros(n, dtype=np.int8)
    for k in range(n):
        safe, unsafe = _free_obs(), _close_lead()
        if rng.random() < 0.5:
            obs_a[k], obs_b[k], safer[k] = safe, unsafe, 0  # A seguro
        else:
            obs_a[k], obs_b[k], safer[k] = unsafe, safe, 1  # B seguro
    return {
        "obs_a": obs_a,
        "obs_b": obs_b,
        "safer": safer,
        "confidence": np.ones(n, dtype=np.float32),
    }


# ── forma / IO ────────────────────────────────────────────────────────────────


def test_feature_dim_and_forward_shapes():
    assert REWARD_FEATURE_DIM == 250
    m = RewardModel()
    batch = torch.zeros(8, 739)
    assert m(batch).shape == (8,)
    assert m.features(batch).shape == (8, 250)


def test_predict_np_single_and_batch():
    m = RewardModel()
    assert isinstance(m.predict_np(_free_obs()), float)
    out = m.predict_np(np.stack([_free_obs(), _close_lead()]))
    assert out.shape == (2,)


# ── perdida Bradley-Terry ──────────────────────────────────────────────────────


def test_bt_loss_sign_prefers_higher_reward_for_safer():
    # safer=0 (A) -> la perdida es MENOR cuando r_a > r_b.
    safer = torch.tensor([0.0])
    good = bradley_terry_loss(torch.tensor([2.0]), torch.tensor([0.0]), safer)
    bad = bradley_terry_loss(torch.tensor([0.0]), torch.tensor([2.0]), safer)
    assert good < bad


def test_bt_loss_confidence_weighting():
    safer = torch.tensor([0.0, 0.0])
    r_a = torch.tensor([0.0, 0.0])
    r_b = torch.tensor([2.0, 2.0])  # ambos "mal" (B mas alto pero A es el seguro)
    full = bradley_terry_loss(r_a, r_b, safer, torch.tensor([1.0, 1.0]))
    half = bradley_terry_loss(r_a, r_b, safer, torch.tensor([1.0, 0.0]))
    assert half < full  # bajar la confianza de un par reduce su peso en la perdida


# ── entrenamiento ──────────────────────────────────────────────────────────────


def test_training_separates_clear_from_close_lead():
    torch.manual_seed(0)
    model = RewardModel()
    hist = train_bradley_terry(
        model, _make_dataset(300, seed=1), epochs=120, lr=1e-3, seed=1, device="cpu"
    )
    assert hist["val_acc"][-1] > 0.9
    # la via libre debe puntuar MAS SEGURA que el lider cercano.
    assert model.predict_np(_free_obs()) > model.predict_np(_close_lead())


# ── persistencia ───────────────────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path):
    torch.manual_seed(0)
    model = RewardModel()
    p = str(tmp_path / "rm.pth")
    save_reward_model(model, p, meta={"x": 1})
    loaded = load_reward_model(p, device="cpu")
    o = _close_lead()
    assert loaded.predict_np(o) == pytest.approx(model.predict_np(o), abs=1e-5)


def test_load_rejects_incompatible_features(tmp_path):
    model = RewardModel()
    p = str(tmp_path / "rm.pth")
    save_reward_model(model, p)
    ckpt = torch.load(p, weights_only=False)
    ckpt["feature_indices"] = [0, 1, 2]  # incompatible
    torch.save(ckpt, p)
    with pytest.raises(ValueError):
        load_reward_model(p)
