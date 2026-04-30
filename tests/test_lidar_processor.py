"""
Tests de convención angular del SemanticLidarProcessor.

Cubren la fase D1 del plan de debugging del visualizador LIDAR semántico
(docs/plan_debug_lidar.md). Verifican que la proyección de un hit en el
sistema local del sensor (UE LH: x=adelante, y=derecha, z=arriba) al bin
del scan polar respeta la convención documentada:

    bin   0   →  FRENTE
    bin  60   →  IZQUIERDA
    bin 120   →  ATRÁS
    bin 180   →  DERECHA

Esta convención es la que asume tanto el dashboard de main_eval.py
(set_theta_zero_location("N") + set_theta_direction(1)) como los shields
en sus comprobaciones por arco. Si cambia, todos esos consumidores
fallan silenciosamente.

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
# Helpers
# ──────────────────────────────────────────────────────────────────────────


class _FakeMeasurement:
    """Objeto mínimo con `.raw_data` para alimentar a process()."""

    def __init__(self, points: np.ndarray, frame: int = 0):
        self.raw_data = points.tobytes()
        self.frame = frame


def _make_points(entries):
    arr = np.zeros(len(entries), dtype=_SEMANTIC_DTYPE)
    for i, e in enumerate(entries):
        arr["x"][i] = e.get("x", 0.0)
        arr["y"][i] = e.get("y", 0.0)
        arr["z"][i] = e.get("z", 0.0)
        arr["cos_inc_angle"][i] = e.get("cos_inc_angle", 1.0)
        arr["object_idx"][i] = e.get("object_idx", 99)
        arr["object_tag"][i] = e.get("object_tag", 11)  # Wall (static)
    return arr


def _make_processor():
    return SemanticLidarProcessor(
        num_rays=240,
        lidar_range=50.0,
        height_filter=1.5,
        ego_id=-1,
        z_mount=1.0,
    )


# ──────────────────────────────────────────────────────────────────────────
# Test principal — D1
# ──────────────────────────────────────────────────────────────────────────


def test_processor_angular_convention():
    """
    4 puntos sintéticos a 10 m a frente / izquierda / atrás / derecha.
    Verificamos que cada uno cae exactamente en el bin que la convención
    del proyecto documenta.

      Frente    → bin   0
      Izquierda → bin  60
      Atrás     → bin 120
      Derecha   → bin 180

    Es el test que ancla la coherencia entre:
      - SemanticLidarProcessor.process() (atan2(-y, x) → bin)
      - El dashboard polar de main_eval.py (theta_zero=N, dir=+1)
      - Los arcos de seguridad de los shields (FRONT_N, R_START..R_END,
        L_START..L_END).
    """
    p = _make_processor()
    pts = _make_points(
        [
            {"x": 10.0, "y": 0.0, "z": 0.0, "object_tag": 11},  # frente
            {"x": 0.0, "y": -10.0, "z": 0.0, "object_tag": 11},  # izquierda
            {"x": -10.0, "y": 0.0, "z": 0.0, "object_tag": 11},  # atrás
            {"x": 0.0, "y": 10.0, "z": 0.0, "object_tag": 11},  # derecha
        ]
    )
    res = p.process(_FakeMeasurement(pts))

    # Cada punto debe haber depositado dist=10/50=0.2 en su bin esperado
    # y los demás bins frontales/laterales deben quedar libres (=1.0).
    assert res.combined[0] == pytest.approx(0.2, abs=1e-6), (
        f"FRENTE: bin 0 esperado 0.2, leído {res.combined[0]}"
    )
    assert res.combined[60] == pytest.approx(0.2, abs=1e-6), (
        f"IZQUIERDA: bin 60 esperado 0.2, leído {res.combined[60]}"
    )
    assert res.combined[120] == pytest.approx(0.2, abs=1e-6), (
        f"ATRÁS: bin 120 esperado 0.2, leído {res.combined[120]}"
    )
    assert res.combined[180] == pytest.approx(0.2, abs=1e-6), (
        f"DERECHA: bin 180 esperado 0.2, leído {res.combined[180]}"
    )

    # Bins intermedios deben estar libres — no se "fugaron" hits a bins
    # adyacentes por errores de redondeo del bin index.
    for free_bin in (30, 90, 150, 210):
        assert res.combined[free_bin] == 1.0, (
            f"bin {free_bin} debería estar libre (=1.0), leído "
            f"{res.combined[free_bin]}"
        )


def test_processor_arcs_cover_expected_bins():
    """
    Verifica que las constantes de arco del procesador (FRONT_N, R_*, L_*)
    envuelven los bins canónicos esperados. Los nombres R_/L_ del procesador
    se mapean a bins así:

        FRONT_N = 15           → bin 0 cae en [n-15, n) ∪ [0, 15)
        R_START..R_END = 40,80 → cubre bin 60
        L_START..L_END = 160,200 → cubre bin 180

    NOTA importante (no resuelta por este plan):
      Por la convención `atan2(-y, x)` y el sistema LH de CARLA, un hit
      físicamente a la IZQUIERDA del vehículo (y<0) cae en bin 60, y un
      hit físicamente a la DERECHA (y>0) cae en bin 180. Por tanto
      `min_r_side_*` (que opera sobre bins 40-80) en realidad mide la
      IZQUIERDA física, y `min_l_side_*` mide la DERECHA física. Es un
      desfase nominal entre código y docstrings de SemanticScanResult /
      carla_sensors.py — flagged como TODO de auditoría aparte.
    """
    p = _make_processor()
    n = p.num_rays
    fn = p.FRONT_N

    assert 0 < fn, "FRONT_N debe ser positivo"
    assert p.R_START <= 60 < p.R_END, (
        f"bin 60 debería caer en [R_START={p.R_START}, R_END={p.R_END})"
    )
    assert p.L_START <= 180 < p.L_END, (
        f"bin 180 debería caer en [L_START={p.L_START}, L_END={p.L_END})"
    )
    # Frente con wrap-around: bin 0 ∈ [n-fn, n) ∪ [0, fn)
    assert (0 < fn) or ((n - fn) <= 0 < n)


def test_processor_arc_minima_match_synthetic_hit():
    """
    Hit a la DERECHA física (y=+10) → cae en bin 180 → afecta a
    `min_l_side_combined` (bins 160-200), NO a `min_r_side_combined`.
    Ver nota en test_processor_arcs_cover_expected_bins sobre la
    inversión nominal de los nombres R_/L_ del procesador.
    """
    p = _make_processor()
    pts = _make_points(
        [{"x": 0.0, "y": 10.0, "z": 0.0, "object_tag": 11}]  # y>0 = DERECHA
    )
    res = p.process(_FakeMeasurement(pts))
    # bin 180 ∈ [L_START=160, L_END=200) → afecta al mínimo del arco "L_*"
    assert res.min_l_side_combined == pytest.approx(0.2, abs=1e-6)
    assert res.min_r_side_combined == 1.0
    assert res.min_front_combined == 1.0


def test_processor_front_arc_wraps_around_bin_zero():
    """
    Un hit en bin 0 (frente exacto) debe afectar a min_front_combined,
    confirmando que el front_min wrappea bins [n-FRONT_N, n) ∪ [0, FRONT_N).
    """
    p = _make_processor()
    pts = _make_points([{"x": 10.0, "y": 0.0, "z": 0.0, "object_tag": 11}])
    res = p.process(_FakeMeasurement(pts))
    assert res.min_front_combined == pytest.approx(0.2, abs=1e-6)


def test_processor_exposes_raw_points_post_filters():
    """
    El procesador debe exponer points_x / points_y / points_tag con los
    puntos efectivamente usados para construir los scans (post-filtros
    ego + altura + rango). Esos arrays alimentan el BEV point map del
    dashboard de evaluación; si no llegan, el dashboard no muestra
    nada y el debug visual se rompe.
    """
    p = _make_processor()
    pts = _make_points(
        [
            {"x": 10.0, "y": 0.0, "z": 0.0, "object_tag": 10},  # OK: vehicle
            {"x": 0.0, "y": -10.0, "z": 0.0, "object_tag": 4},  # OK: ped
            {"x": 5.0, "y": 0.0, "z": -1.95, "object_tag": 8},  # SUELO: descartar
            {"x": 100.0, "y": 0.0, "z": 0.0, "object_tag": 11}, # FUERA RANGO
        ]
    )
    res = p.process(_FakeMeasurement(pts))
    # 2 puntos esperados: el vehículo y el peatón. El de suelo cae por el
    # filtro asimétrico de altura; el lejano cae por el filtro de rango.
    assert len(res.points_x) == 2, (
        f"esperado 2 puntos post-filtros, leídos {len(res.points_x)}"
    )
    assert len(res.points_y) == 2
    assert len(res.points_tag) == 2
    # Los tags supervivientes deben ser exactamente {4, 10}.
    assert set(int(t) for t in res.points_tag) == {4, 10}


def test_processor_exposes_no_points_when_empty():
    """Frame vacío → arrays de puntos vacíos (no None ni longitud rara)."""
    p = _make_processor()
    res = p.process(_FakeMeasurement(_make_points([])))
    assert len(res.points_x) == 0
    assert len(res.points_y) == 0
    assert len(res.points_tag) == 0


def test_processor_filters_ego_from_raw_points():
    """
    Los puntos del propio ego no deben aparecer ni en los scans bin-eados
    ni en los arrays crudos. Si lo hicieran, el BEV pintaría el coche
    como obstáculo a 0 m y el shield activaría continuamente.
    """
    p = SemanticLidarProcessor(
        num_rays=240, lidar_range=50.0, height_filter=1.5, ego_id=42, z_mount=1.0
    )
    pts = _make_points(
        [
            {"x": 0.5, "y": 0.0, "z": 0.0, "object_idx": 42, "object_tag": 10},
            {"x": 10.0, "y": 0.0, "z": 0.0, "object_idx": 99, "object_tag": 10},
        ]
    )
    res = p.process(_FakeMeasurement(pts))
    # Solo el segundo punto (object_idx=99) debe sobrevivir.
    assert len(res.points_x) == 1
    assert float(res.points_x[0]) == pytest.approx(10.0, abs=1e-5)


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
