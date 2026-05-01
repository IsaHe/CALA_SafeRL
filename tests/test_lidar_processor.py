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
    DYNAMIC_TAGS,
    STATIC_OBS_TAGS,
    PEDESTRIAN_TAGS,
    VEHICLE_TAGS,
    ROAD_EDGE_TAGS,
    ROAD_SURFACE_TAGS,
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
        # Tag 4 = Wall en CARLA 0.9.14+ (era 11 en la tabla vieja). Default
        # estático para que el punto caiga en `combined` por la rama static.
        arr["object_tag"][i] = e.get("object_tag", 4)
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
    # Tag 4 = Wall en CARLA 0.9.14+ → cae en STATIC_OBS_TAGS → llega a combined.
    pts = _make_points(
        [
            {"x": 10.0, "y": 0.0, "z": 0.0, "object_tag": 4},  # frente
            {"x": 0.0, "y": -10.0, "z": 0.0, "object_tag": 4},  # izquierda
            {"x": -10.0, "y": 0.0, "z": 0.0, "object_tag": 4},  # atrás
            {"x": 0.0, "y": 10.0, "z": 0.0, "object_tag": 4},  # derecha
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
        [{"x": 0.0, "y": 10.0, "z": 0.0, "object_tag": 4}]  # y>0 = DERECHA
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
    pts = _make_points([{"x": 10.0, "y": 0.0, "z": 0.0, "object_tag": 4}])
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
    # CARLA 0.9.14+ tags: 14=Car, 12=Pedestrian, 2=SideWalk (a nivel del
    # suelo → debe filtrarse), 4=Wall (estático).
    pts = _make_points(
        [
            {"x": 10.0, "y": 0.0, "z": 0.0, "object_tag": 14},  # OK: Car
            {"x": 0.0, "y": -10.0, "z": 0.0, "object_tag": 12}, # OK: Ped
            {"x": 5.0, "y": 0.0, "z": -1.95, "object_tag": 2},  # SUELO
            {"x": 100.0, "y": 0.0, "z": 0.0, "object_tag": 4},  # FUERA RANGO
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
    # Los tags supervivientes deben ser exactamente {12, 14}.
    assert set(int(t) for t in res.points_tag) == {12, 14}


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
    # Tag 14 = Car (CARLA 0.9.14+). El primer punto es un hit del propio
    # cuerpo del coche; debe filtrarse aunque tenga tag de vehículo.
    pts = _make_points(
        [
            {"x": 0.5, "y": 0.0, "z": 0.0, "object_idx": 42, "object_tag": 14},
            {"x": 10.0, "y": 0.0, "z": 0.0, "object_idx": 99, "object_tag": 14},
        ]
    )
    res = p.process(_FakeMeasurement(pts))
    # Solo el segundo punto (object_idx=99) debe sobrevivir.
    assert len(res.points_x) == 1
    assert float(res.points_x[0]) == pytest.approx(10.0, abs=1e-5)


# ──────────────────────────────────────────────────────────────────────────
# Blindajes contra el bug de tabla de tags (CARLA 0.9.14+)
# ──────────────────────────────────────────────────────────────────────────


def test_pedestrian_tags_match_carla_0_9_14():
    """
    Tag 12 = Pedestrian, tag 13 = Rider en CARLA 0.9.14+. Si vuelven a
    cablearse contra tag 4 (Wall en la nueva tabla), las paredes saldrán
    como peatones — bug confirmado en evaluación real antes del fix.
    """
    assert 12 in PEDESTRIAN_TAGS, (
        "Pedestrian (tag 12) debe estar en PEDESTRIAN_TAGS para CARLA 0.9.14+"
    )
    assert 13 in PEDESTRIAN_TAGS, (
        "Rider (tag 13) debe estar en PEDESTRIAN_TAGS"
    )
    assert 4 not in PEDESTRIAN_TAGS, (
        "Tag 4 = Wall en CARLA 0.9.14+ — NO debe estar en PEDESTRIAN_TAGS"
    )


def test_vehicle_tags_match_carla_0_9_14():
    """
    En CARLA 0.9.14+ la antigua clase Vehicles (tag 10) se subdividió en
    Car/Truck/Bus/Train/Motorcycle/Bicycle (tags 14-19). El tag 10 ahora
    es Terrain. Si VEHICLE_TAGS sigue conteniendo 10, los vehículos no
    se detectarán como dinámicos y caerán en static (bug reportado).
    """
    assert VEHICLE_TAGS == frozenset({14, 15, 16, 17, 18, 19}), (
        f"VEHICLE_TAGS desalineado con CARLA 0.9.14+: {VEHICLE_TAGS}"
    )
    assert 10 not in VEHICLE_TAGS, (
        "Tag 10 = Terrain en CARLA 0.9.14+ — NO debe estar en VEHICLE_TAGS"
    )


def test_wall_does_not_become_pedestrian():
    """
    Caso reproductor del bug: una pared (tag 4 en CARLA 0.9.14+) NO debe
    aparecer en el sub-scan de peatones. Antes del fix, el procesador
    asumía tag 4 = Pedestrian y la pared activaba `min_front_pedestrian`
    sin que hubiera ningún humano en la escena.
    """
    p = _make_processor()
    pts = _make_points(
        [{"x": 5.0, "y": 0.0, "z": 0.0, "object_tag": 4}]  # Wall
    )
    res = p.process(_FakeMeasurement(pts))
    # El scan de peatones debe quedar libre.
    assert res.pedestrian[0] == 1.0, (
        "Una pared (tag 4) no debe aparecer en pedestrian_scan"
    )
    # En cambio sí debe contar como estático.
    assert res.static[0] == pytest.approx(0.10, abs=1e-6)
    assert res.n_pedestrian_pts == 0
    assert res.n_static_pts == 1


def test_car_appears_as_dynamic_not_static():
    """
    Caso reproductor del bug: un coche (tag 14 en CARLA 0.9.14+) debe
    aparecer en el sub-scan dinámico, NO en el estático. Antes del fix,
    DYNAMIC_TAGS contenía tag 10 (que en la tabla nueva es Terrain) y
    los coches caían en static_obs por la suerte de coincidir tag 14
    con el antiguo "Ground" — pero ni siquiera estaba agrupado allí.
    """
    p = _make_processor()
    pts = _make_points(
        [{"x": 8.0, "y": 0.0, "z": 0.0, "object_tag": 14}]  # Car
    )
    res = p.process(_FakeMeasurement(pts))
    # Aparece en dynamic.
    assert res.dynamic[0] == pytest.approx(0.16, abs=1e-6)
    # NO aparece en static.
    assert res.static[0] == 1.0
    assert res.n_vehicle_pts == 1
    assert res.n_static_pts == 0


def test_road_surface_captured_pre_height_filter():
    """
    Las marcas de carril (tag 24 = RoadLine) están a nivel del suelo y
    el filtro asimétrico de altura las descarta — correcto para safety
    pero el dashboard las necesita como referencia visual. El procesador
    debe exponerlas en el canal road_points_* aparte (PRE-filtro-altura).
    """
    p = _make_processor()  # z_mount=1.0 → z_min_sensor=-0.85
    # World z=0 → z_sensor=-1.0 < -0.85 → descartado por filtro altura.
    pts = _make_points(
        [
            {"x": 5.0, "y": 0.0, "z": -1.0, "object_tag": 24},  # RoadLine
            {"x": 7.0, "y": 0.0, "z": -1.0, "object_tag": 1},   # Roads
        ]
    )
    res = p.process(_FakeMeasurement(pts))
    # No están en los scans (el filtro de altura los retiró).
    assert res.combined[0] == 1.0
    # PERO están en road_points_*: capa visual de fondo.
    assert len(res.road_points_x) == 2
    assert set(int(t) for t in res.road_points_tag) == {1, 24}


def test_road_surface_not_in_combined_or_obstacle_points():
    """
    Las superficies de carretera (1 Roads, 24 RoadLine) NO deben
    contaminar `points_*` ni `combined`. Si lo hicieran, el BEV
    pintaría líneas blancas como "objetos" y el shield podría dispararse
    contra ellas.
    """
    p = SemanticLidarProcessor(
        num_rays=240,
        lidar_range=50.0,
        height_filter=1.5,
        ego_id=-1,
        z_mount=0.5,  # ZONA donde el filtro asimétrico no las descarta
    )
    # z_sensor = -0.5, z_min = -(0.5-0.15) = -0.35 → -0.5 < -0.35 → filtra.
    # Para forzar que pasen el filtro de altura, usamos z=-0.2 (sobre el
    # umbral). Aún así NO deben acabar en points_* ni en combined porque
    # excluimos road_surface explícitamente del point map.
    pts = _make_points(
        [
            {"x": 5.0, "y": 0.0, "z": -0.2, "object_tag": 24},  # RoadLine
            {"x": 5.0, "y": 0.0, "z": 0.5, "object_tag": 4},    # Wall sí
        ]
    )
    res = p.process(_FakeMeasurement(pts))
    # Solo el Wall (tag 4) debe estar en points_*.
    assert len(res.points_x) == 1, (
        f"esperado 1 punto en points_*, leído {len(res.points_x)} "
        f"con tags {set(int(t) for t in res.points_tag)}"
    )
    assert int(res.points_tag[0]) == 4
    # RoadLine sí debe aparecer en road_points_*.
    assert 24 in set(int(t) for t in res.road_points_tag)


def test_no_overlap_between_groups():
    """
    Las cuatro categorías de obstáculo (vehículo, peatón, estático,
    road_edge) deben ser mutuamente excluyentes. Si un tag cae en dos
    grupos, los conteos `n_*_pts` se inflarán y el debug visual mentirá.
    """
    groups = {
        "vehicle": VEHICLE_TAGS,
        "pedestrian": PEDESTRIAN_TAGS,
        "static": STATIC_OBS_TAGS,
        "road_edge": ROAD_EDGE_TAGS,
        "road_surface": ROAD_SURFACE_TAGS,
    }
    items = list(groups.items())
    for i, (a_name, a) in enumerate(items):
        for b_name, b in items[i + 1 :]:
            inter = a & b
            assert not inter, (
                f"Tags duplicados entre {a_name} y {b_name}: {inter}"
            )


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
