"""
Tests del feature de tipo de marca de carril (solid/dashed left/right).

Verifican la lógica pura de clasificación (sin requerir CARLA en ejecución):
  1. Un waypoint con sólida a la izquierda y discontinua a la derecha
     emite flags [solid_left=1, solid_right=0, dashed_left=0, dashed_right=1].
  2. Marcas NONE → todos los flags a 0.
  3. SolidSolid, SolidBroken, BrokenSolid también se clasifican como "sólida".
  4. BrokenBroken se clasifica como "dashed".
"""

import os
import sys

import pytest

carla = pytest.importorskip("carla")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.CARLA.Env.carla_env import CarlaEnv  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# Helper — clasificador "puro" que refleja la lógica de _get_lane_features
# ──────────────────────────────────────────────────────────────────────────


def classify(left_type, right_type):
    solid_left = left_type in CarlaEnv._SOLID_MARKING_TYPES
    solid_right = right_type in CarlaEnv._SOLID_MARKING_TYPES
    dashed_left = left_type in CarlaEnv._DASHED_MARKING_TYPES
    dashed_right = right_type in CarlaEnv._DASHED_MARKING_TYPES
    return (
        float(solid_left),
        float(solid_right),
        float(dashed_left),
        float(dashed_right),
    )


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


def test_solid_left_dashed_right():
    flags = classify(carla.LaneMarkingType.Solid, carla.LaneMarkingType.Broken)
    assert flags == (1.0, 0.0, 0.0, 1.0)


def test_both_none():
    flags = classify(carla.LaneMarkingType.NONE, carla.LaneMarkingType.NONE)
    assert flags == (0.0, 0.0, 0.0, 0.0)


def test_solid_variants_classified_as_solid():
    for solid_variant in (
        carla.LaneMarkingType.Solid,
        carla.LaneMarkingType.SolidSolid,
        carla.LaneMarkingType.SolidBroken,
        carla.LaneMarkingType.BrokenSolid,
    ):
        flags = classify(solid_variant, carla.LaneMarkingType.NONE)
        assert flags[0] == 1.0, f"{solid_variant} no clasificó como solid_left"
        assert flags[2] == 0.0, f"{solid_variant} no debería ser dashed"


def test_dashed_variants_classified_as_dashed():
    for dashed_variant in (
        carla.LaneMarkingType.Broken,
        carla.LaneMarkingType.BrokenBroken,
    ):
        flags = classify(carla.LaneMarkingType.NONE, dashed_variant)
        assert flags[3] == 1.0, f"{dashed_variant} no clasificó como dashed_right"
        assert flags[1] == 0.0, f"{dashed_variant} no debería ser solid"


def test_solid_and_dashed_are_mutually_exclusive():
    """Una marca concreta no puede ser sólida Y discontinua a la vez."""
    for marking_type in (
        carla.LaneMarkingType.Solid,
        carla.LaneMarkingType.SolidSolid,
        carla.LaneMarkingType.SolidBroken,
        carla.LaneMarkingType.BrokenSolid,
        carla.LaneMarkingType.Broken,
        carla.LaneMarkingType.BrokenBroken,
        carla.LaneMarkingType.NONE,
    ):
        flags = classify(marking_type, carla.LaneMarkingType.NONE)
        # solid_left + dashed_left <= 1
        assert flags[0] + flags[2] <= 1.0, f"{marking_type} es a la vez solid y dashed"


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
