"""
Tests de distancia del SemanticLidarProcessor.

CUBREN:
  1. Normalización `dist / lidar_range` exacta para puntos a 5, 25, 0 m.
  2. Scan vacío → todas las celdas = 1.0 (libre).
  3. Filtro asimétrico de altura: puntos en el suelo se rechazan, puntos
     a altura de guardarraíl bajo (~0.5 m) se mantienen.
  4. Filtro techo: puntos por encima de Z_ABOVE_MAX se descartan.
  5. Filtro ego por object_idx exacto.
  6. Verificación específica del plan: un obstáculo bajo a 5 m debe
     leerse exactamente a 5 m tras des-normalizar (= 5.0 m).

Si `carla` no está instalado en el entorno de test, se saltan todos.
"""

import os
import sys

import numpy as np
import pytest

carla = pytest.importorskip("carla")  # noqa: F841  — habilita skip limpio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.CARLA.Sensors.SemanticLidarProcessor import (  # noqa: E402
    SemanticLidarProcessor,
    _SEMANTIC_DTYPE,
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers — construir un measurement sintético
# ──────────────────────────────────────────────────────────────────────────


class _FakeMeasurement:
    """Objeto mínimo con `.raw_data` para alimentar a process()."""

    def __init__(self, points: np.ndarray):
        self.raw_data = points.tobytes()
        self.frame = 0


def _make_points(entries):
    """entries: lista de dicts con x, y, z, object_idx, object_tag."""
    arr = np.zeros(len(entries), dtype=_SEMANTIC_DTYPE)
    for i, e in enumerate(entries):
        arr["x"][i] = e.get("x", 0.0)
        arr["y"][i] = e.get("y", 0.0)
        arr["z"][i] = e.get("z", 0.0)
        arr["cos_inc_angle"][i] = e.get("cos_inc_angle", 1.0)
        arr["object_idx"][i] = e.get("object_idx", 0)
        # Tag 14 = Car en CARLA 0.9.14+ (era 10 en 0.9.10-0.9.13). Default
        # dinámico para que el punto caiga en `combined` sin que el test
        # tenga que especificarlo cada vez.
        arr["object_tag"][i] = e.get("object_tag", 14)
    return arr


def _make_processor(z_mount: float = 1.0, lidar_range: float = 50.0):
    return SemanticLidarProcessor(
        num_rays=240,
        lidar_range=lidar_range,
        height_filter=1.5,
        ego_id=-1,
        z_mount=z_mount,
    )


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


def test_normalization_5m():
    """Punto a 5 m en frente → combined[0] == 5/50 == 0.10."""
    p = _make_processor(z_mount=1.0, lidar_range=50.0)
    # Punto en frente: x positivo, y=0. z relativo al sensor = 0 (a la misma
    # altura que el sensor). Tag 14 (Car en CARLA 0.9.14+) para que caiga
    # en el grupo dinámico y por tanto en combined.
    pts = _make_points([{"x": 5.0, "y": 0.0, "z": 0.0, "object_tag": 14}])
    result = p.process(_FakeMeasurement(pts))
    assert result.combined[0] == pytest.approx(0.10, abs=1e-6)


def test_normalization_25m():
    p = _make_processor(z_mount=1.0, lidar_range=50.0)
    pts = _make_points([{"x": 25.0, "y": 0.0, "z": 0.0, "object_tag": 14}])
    result = p.process(_FakeMeasurement(pts))
    assert result.combined[0] == pytest.approx(0.50, abs=1e-6)


def test_no_hit_returns_one():
    """Sin puntos en el buffer → todas las celdas quedan en 1.0 (libre)."""
    p = _make_processor()
    pts = _make_points([])
    result = p.process(_FakeMeasurement(pts))
    assert np.all(result.combined == 1.0)
    assert np.all(result.dynamic == 1.0)
    assert np.all(result.static == 1.0)


def test_ground_rejection_with_asymmetric_filter():
    """
    Sensor a z_mount=1.0 m. Un punto a z_sensor = -0.95 (world z = 0.05 m,
    por debajo de GROUND_CLEARANCE=0.15) debe RECHAZARSE → scan libre.
    Esto elimina los falsos obstáculos de `sidewalk` a ~3.7 m que eran la
    causa principal del sesgo "todo está cerca" en el polar plot.
    """
    p = _make_processor(z_mount=1.0, lidar_range=50.0)
    # Tag 2 = SideWalk en CARLA 0.9.14+ (era 8 en la tabla vieja).
    pts = _make_points(
        [{"x": 10.0, "y": 0.0, "z": -0.95, "object_tag": 2}]  # sidewalk
    )
    result = p.process(_FakeMeasurement(pts))
    # El punto cayó en el bin frontal (0). Debe estar en 1.0 (libre).
    assert result.combined[0] == 1.0


def test_low_obstacle_kept():
    """
    Guardarraíl bajo a world z=0.5 m, sensor a z_mount=1.0 → z_sensor = -0.5.
    Está por encima de z_min_sensor = -(1.0 - 0.15) = -0.85, así que PASA.
    Distancia horizontal 10 m → combined[0] == 10/50 == 0.20.
    Tag 28 = GuardRail en CARLA 0.9.14+ (era 17 en la tabla vieja).
    """
    p = _make_processor(z_mount=1.0, lidar_range=50.0)
    pts = _make_points(
        [{"x": 10.0, "y": 0.0, "z": -0.5, "object_tag": 28}]  # guardrail
    )
    result = p.process(_FakeMeasurement(pts))
    assert result.combined[0] == pytest.approx(0.20, abs=1e-6)


def test_overhead_clip():
    """Punto a z_sensor=2.0 (> Z_ABOVE_MAX=1.5) debe rechazarse."""
    p = _make_processor(z_mount=1.0)
    pts = _make_points([{"x": 10.0, "y": 0.0, "z": 2.0, "object_tag": 14}])
    result = p.process(_FakeMeasurement(pts))
    assert result.combined[0] == 1.0


def test_ego_filter():
    """Puntos con object_idx == ego_id deben descartarse."""
    p = SemanticLidarProcessor(
        num_rays=240, lidar_range=50.0, height_filter=1.5, ego_id=42, z_mount=1.0
    )
    pts = _make_points(
        [{"x": 5.0, "y": 0.0, "z": 0.0, "object_idx": 42, "object_tag": 14}]
    )
    result = p.process(_FakeMeasurement(pts))
    # El único punto era del ego → se filtra → bin frontal libre.
    assert result.combined[0] == 1.0


def test_low_obstacle_5m_reads_exactly_5m_after_denormalization():
    """
    Criterio de verificación del plan: un obstáculo bajo a exactamente 5 m
    de distancia debe leerse como 5.0 m tras multiplicar por lidar_range.
    Sensor bajo del parachoques: z_mount=0.5, lidar_range=30.
    Tag 28 = GuardRail (CARLA 0.9.14+).
    """
    p = _make_processor(z_mount=0.5, lidar_range=30.0)
    # Obstáculo a world z=0.5 m (altura de bordillo/guardarraíl bajo);
    # desde el sensor bajo a z=0.5 m → z_sensor=0.0 (mismo plano).
    pts = _make_points(
        [{"x": 5.0, "y": 0.0, "z": 0.0, "object_tag": 28}]  # guardrail
    )
    result = p.process(_FakeMeasurement(pts))
    normalized = result.combined[0]
    denormalized_m = normalized * 30.0
    assert denormalized_m == pytest.approx(5.0, abs=1e-5), (
        f"Expected 5.0 m, got {denormalized_m} m (norm={normalized})"
    )


def test_nearest_meters_returns_raw_distance():
    """
    `nearest_vehicle_m` devuelve la distancia RAW en metros (no normalizada),
    para que las métricas `Safety/Min_*_Distance_m` sean legibles.
    Tag 14 = Car (CARLA 0.9.14+).
    """
    p = _make_processor(z_mount=1.0, lidar_range=50.0)
    pts = _make_points([{"x": 12.5, "y": 0.0, "z": 0.0, "object_tag": 14}])
    result = p.process(_FakeMeasurement(pts))
    assert result.nearest_vehicle_m == pytest.approx(12.5, abs=1e-5)


def test_nearest_sentinel_when_no_detection():
    """Sin detección → nearest_* = 999.0 (sentinela explícito)."""
    p = _make_processor()
    pts = _make_points([])
    result = p.process(_FakeMeasurement(pts))
    assert result.nearest_vehicle_m == 999.0
    assert result.nearest_pedestrian_m == 999.0
    assert result.nearest_static_m == 999.0


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
