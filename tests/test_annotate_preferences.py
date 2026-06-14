"""
test_annotate_preferences.py - Logica pura del anotador offline (fase 1).

Sin Ollama ni CARLA: prueba el muestreo de pares, la deteccion de encuentros y el
ensamblado del dataset (descartando abstenciones y baja confianza) con un anotador
mock.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meta.preference_annotator import PreferenceLabel  # noqa: E402
from tools.annotate_preferences import (  # noqa: E402
    is_encounter,
    is_front_encounter,
    run_annotation,
    select_pairs,
)

DYN_START = 240


def _free_obs() -> np.ndarray:
    o = np.zeros(739, dtype=np.float32)
    o[0:720] = 1.0
    return o


def _obs_with_front(norm: float) -> np.ndarray:
    o = _free_obs()
    o[DYN_START + 0] = norm
    return o


def _obs_with_side(norm: float) -> np.ndarray:
    o = _free_obs()
    o[DYN_START + 60] = norm  # bin 60 = a la izquierda, fuera del cono frontal
    return o


class _FakeAnnotator:
    def __init__(self, labels):
        self._labels = labels
        self._i = 0

    def annotate_symmetric(self, a, b):
        lbl = self._labels[min(self._i, len(self._labels) - 1)]
        self._i += 1
        return lbl

    annotate = annotate_symmetric


# ── is_encounter ─────────────────────────────────────────────────────────────


def test_is_encounter_true_when_obstacle_close():
    assert is_encounter(_obs_with_front(0.2), max_m=35.0) is True  # 10 m


def test_is_encounter_false_when_clear():
    assert is_encounter(_free_obs(), max_m=35.0) is False


def test_is_encounter_respects_max_m():
    assert is_encounter(_obs_with_front(0.9), max_m=35.0) is False  # 45 m > 35


def test_is_front_encounter_only_in_path():
    assert is_front_encounter(_obs_with_front(0.4), max_m=25.0) is True  # 20 m delante
    assert is_front_encounter(_obs_with_front(0.6), max_m=25.0) is False  # 30 m > 25
    # Un obstaculo de lado (bin 60) NO es un encuentro frontal aunque este cerca.
    assert is_front_encounter(_obs_with_side(0.2), max_m=25.0) is False


# ── select_pairs ─────────────────────────────────────────────────────────────


def test_select_pairs_no_self_pairs_and_in_range():
    rng = np.random.default_rng(0)
    pairs = select_pairs(np.arange(10), 50, rng, oversample_idx=np.array([], dtype=int))
    assert len(pairs) == 50
    for i, j in pairs:
        assert 0 <= i < 10 and 0 <= j < 10
        assert i != j


def test_select_pairs_oversamples_encounters():
    rng = np.random.default_rng(1)
    pairs = select_pairs(
        np.arange(10), 30, rng, oversample_idx=np.array([3], dtype=int), oversample=1.0
    )
    # oversample=1.0 -> el primer endpoint es SIEMPRE el unico indice de encuentro.
    assert all(i == 3 for i, _ in pairs)


def test_select_pairs_endpoints_restricted_to_eligible():
    rng = np.random.default_rng(7)
    eligible = np.array([2, 4, 6, 8], dtype=int)
    pairs = select_pairs(eligible, 40, rng, oversample_idx=np.array([], dtype=int))
    allowed = set(eligible.tolist())
    for i, j in pairs:
        assert i in allowed and j in allowed
        assert i != j


def test_select_pairs_empty_when_too_few_eligible():
    rng = np.random.default_rng(2)
    assert (
        select_pairs(np.array([0]), 10, rng, oversample_idx=np.array([], dtype=int))
        == []
    )


# ── run_annotation ───────────────────────────────────────────────────────────


def test_run_annotation_drops_abstentions_and_low_confidence():
    obs = np.stack([_free_obs(), _obs_with_front(0.2), _free_obs()])
    pairs = [(0, 1), (1, 2), (2, 0)]
    labels = [
        PreferenceLabel(safer=0, confidence=0.9, reasoning="keep", ok=True),
        PreferenceLabel(safer=-1, confidence=0.0, reasoning="abstain", ok=False),
        PreferenceLabel(safer=1, confidence=0.2, reasoning="low", ok=True),
    ]
    dataset, audit = run_annotation(
        obs, pairs, _FakeAnnotator(labels), symmetric=True, min_confidence=0.5
    )
    # Solo el primer par sobrevive (ok + confianza >= 0.5).
    assert dataset["safer"].tolist() == [0]
    assert dataset["obs_a"].shape == (1, 739)
    assert dataset["obs_b"].shape == (1, 739)
    assert dataset["confidence"][0] == pytest.approx(0.9)
    # La auditoria conserva TODOS los pares.
    assert len(audit) == 3
    assert audit[1]["ok"] is False


def test_run_annotation_keeps_all_when_no_min_confidence():
    obs = np.stack([_free_obs(), _obs_with_front(0.2)])
    pairs = [(0, 1), (1, 0)]
    labels = [
        PreferenceLabel(safer=0, confidence=0.1, reasoning="a", ok=True),
        PreferenceLabel(safer=1, confidence=0.3, reasoning="b", ok=True),
    ]
    dataset, _ = run_annotation(obs, pairs, _FakeAnnotator(labels), min_confidence=0.0)
    assert dataset["safer"].size == 2
