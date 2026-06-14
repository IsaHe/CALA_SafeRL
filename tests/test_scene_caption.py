"""
test_scene_caption.py - Caption de escena desde la observacion (fase 1).

Verifican, sin CARLA, que el captioner es fiel a las convenciones del proyecto:
  - recuperacion de velocidad real (speed_ratio * speed_limit_norm * 130),
  - cono frontal +-FRONT_N bins y conversion norm->metros (val*50),
  - convencion angular atan2(-y,x): bin 0 = de frente, CCW = izquierda,
  - robustez (obs corta -> ValueError).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meta.scene_caption import (  # noqa: E402
    IDX_LATERAL_OFFSET,
    IDX_ON_EDGE,
    IDX_SPEED_LIMIT_NORM,
    IDX_SPEED_RATIO,
    LIDAR_RANGE_M,
    _bearing_label,
    caption_from_obs,
    extract_scene_features,
)

DYN_START = 240


def _free_obs() -> np.ndarray:
    """Carretera libre: los 3 canales LIDAR a 1.0 (libre), features a 0."""
    o = np.zeros(739, dtype=np.float32)
    o[0:720] = 1.0
    return o


def _set_dyn(o: np.ndarray, bin_idx: int, norm: float) -> None:
    o[DYN_START + bin_idx] = norm


# ── bearing ─────────────────────────────────────────────────────────────────


def test_bearing_front_left_right_rear():
    assert _bearing_label(0) == "directly ahead"
    assert _bearing_label(30) == "ahead and to the left"  # 45 deg, CCW = izquierda
    assert _bearing_label(60) == "to the left"  # 90 deg
    assert _bearing_label(120) == "directly behind"  # 180 deg
    assert _bearing_label(180) == "to the right"  # 270 deg
    assert _bearing_label(210) == "ahead and to the right"  # 315 deg


# ── velocidad ────────────────────────────────────────────────────────────────


def test_speed_recovered_from_route_features():
    o = _free_obs()
    o[IDX_SPEED_LIMIT_NORM] = 0.5  # -> 65 km/h
    o[IDX_SPEED_RATIO] = 0.5  # -> 32.5 km/h
    f = extract_scene_features(o)
    assert f.speed_limit_kmh == pytest.approx(65.0)
    assert f.speed_kmh == pytest.approx(32.5)


def test_caption_stopped_when_speed_zero():
    cap = caption_from_obs(_free_obs())
    assert "stopped" in cap.lower()
    assert "no other road users" in cap.lower()


# ── obstaculos dinamicos ─────────────────────────────────────────────────────


def test_front_obstacle_distance_in_metres():
    o = _free_obs()
    _set_dyn(o, 0, 0.2)  # bin 0 (frente), 0.2 * 50 = 10 m
    f = extract_scene_features(o)
    assert f.front_dynamic_m == pytest.approx(10.0)
    assert f.nearest_bearing == "directly ahead"
    assert "10 m directly ahead" in caption_from_obs(o)


def test_obstacle_to_the_left_when_front_clear():
    o = _free_obs()
    _set_dyn(o, 60, 0.3)  # 90 deg, 15 m; cono frontal sigue libre
    f = extract_scene_features(o)
    assert not np.isfinite(f.front_dynamic_m)
    assert f.nearest_dynamic_m == pytest.approx(15.0)
    assert f.nearest_bearing == "to the left"
    cap = caption_from_obs(o)
    assert "directly ahead is clear" in cap
    assert "to the left" in cap


def test_obstacle_to_the_right():
    o = _free_obs()
    _set_dyn(o, 180, 0.3)  # 270 deg
    assert extract_scene_features(o).nearest_bearing == "to the right"


def test_front_cone_wraps_around_bin_zero():
    # Un hit en el ultimo bin (239) cae dentro del cono frontal (FRONT_N a cada lado).
    o = _free_obs()
    _set_dyn(o, 239, 0.4)  # 20 m, justo a la izquierda del frente
    f = extract_scene_features(o)
    assert f.front_dynamic_m == pytest.approx(20.0)


# ── posicion en carril ───────────────────────────────────────────────────────


def test_centered_vs_on_edge():
    o = _free_obs()
    o[IDX_LATERAL_OFFSET] = 0.0
    assert "well centred" in caption_from_obs(o)

    o[IDX_ON_EDGE] = 1.0
    assert "edge of the road" in caption_from_obs(o)


# ── robustez ─────────────────────────────────────────────────────────────────


def test_short_obs_raises():
    with pytest.raises(ValueError):
        extract_scene_features(np.zeros(100, dtype=np.float32))


def test_free_road_reports_infinite_distances():
    f = extract_scene_features(_free_obs())
    assert not np.isfinite(f.front_dynamic_m)
    assert not np.isfinite(f.nearest_dynamic_m)
    assert f.nearest_bearing == "none"


def test_caption_is_single_string_ending_in_period():
    cap = caption_from_obs(_free_obs())
    assert isinstance(cap, str)
    assert cap.endswith(".")
    assert LIDAR_RANGE_M == 50.0  # convencion asumida por el captioner
