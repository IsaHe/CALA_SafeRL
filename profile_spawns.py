"""
profile_spawns.py - Clasifica spawn points de un mapa CARLA por dificultad.

Para cada spawn point del mapa: walks forward `--forward_dist` metros a lo
largo de los waypoints, suma la curvatura absoluta acumulada (deg) y
calcula estadísticas de lane_width y disponibilidad de lane vecinas.

Cada spawn point recibe una clasificación:

    easy   : curvatura total < TIER_EASY_DEG y lane_width media >= 3.0 m
    medium : curvatura total < TIER_MEDIUM_DEG
    hard   : el resto

El resultado se guarda como JSON en `spawn_profiles/<map_name>.json` con
una estructura consumible por `train_curriculum.py`:

    {
      "map": "Town04",
      "n_spawn_points": 234,
      "forward_dist_m": 100.0,
      "tiers": {
        "easy":   [1, 5, 42, ...],
        "medium": [...],
        "hard":   [...]
      },
      "details": {
        "0": {"curvature_deg": 12.3, "tier": "easy", "lane_width_mean": 3.5, ...},
        ...
      }
    }

USO:
    # CARLA server debe estar corriendo
    python profile_spawns.py --map Town04
    python profile_spawns.py --map Town04 --forward_dist 150 --output custom.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] profile: %(message)s",
)
logger = logging.getLogger("profile_spawns")


# Umbrales en grados de curvatura ABSOLUTA acumulada en la distancia
# forward analizada. Calibrados para Town04 a 100 m:
#   easy:  rectas y curvas suaves de autopista (< 15°)
#   medium: curvas moderadas (15°-60°)
#   hard:  cruces, rotondas, ramales (>= 60°)
TIER_EASY_DEG = 15.0
TIER_MEDIUM_DEG = 60.0

# Lane_width mínima para considerar 'easy'. Por debajo de esto, aunque
# sea recta, los rieles están demasiado cerca y la dificultad de
# lane-keeping crece. Town04 highway tiene ~3.5 m; rampas estrechan.
EASY_MIN_LANE_WIDTH_M = 3.0

# Paso de muestreo del waypoint API (metros).
SAMPLE_STEP_M = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Perfila spawn points de un mapa CARLA por dificultad."
    )
    p.add_argument(
        "--map",
        type=str,
        default="Town04",
        help="Nombre del mapa CARLA a perfilar. Carga el mapa antes de "
        "ejecutar; espera que CARLA esté corriendo.",
    )
    p.add_argument(
        "--host", type=str, default="localhost", help="Host del servidor CARLA"
    )
    p.add_argument("--port", type=int, default=2000, help="Puerto del servidor CARLA")
    p.add_argument(
        "--timeout", type=float, default=20.0, help="Timeout de conexión CARLA (s)"
    )
    p.add_argument(
        "--forward_dist",
        type=float,
        default=100.0,
        help="Distancia (m) a explorar hacia adelante desde cada spawn para "
        "medir curvatura acumulada. Default 100m alinea con success_distance "
        "de la primera fase del curriculum.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de salida del JSON. Default: ./spawn_profiles/<map>.json",
    )
    p.add_argument(
        "--easy_max_deg",
        type=float,
        default=TIER_EASY_DEG,
        help=f"Umbral max curvatura (deg) para tier 'easy'. Default {TIER_EASY_DEG}.",
    )
    p.add_argument(
        "--medium_max_deg",
        type=float,
        default=TIER_MEDIUM_DEG,
        help=f"Umbral max curvatura (deg) para tier 'medium'. Default {TIER_MEDIUM_DEG}.",
    )
    return p.parse_args()


def _yaw_diff_deg(a: float, b: float) -> float:
    """Diferencia mínima signada entre dos yaw en grados, en [-180, 180]."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return d


def profile_spawn(carla_map, sp_idx: int, sp_transform, forward_dist: float) -> dict:
    """Camina forward_dist metros desde un spawn y devuelve métricas.

    Returns dict con:
        curvature_deg: suma de |yaw_diff| acumulado (deg)
        lane_width_mean: lane_width medio en el path
        lane_width_min: lane_width mínimo en el path
        path_length_m: distancia real recorrida (puede ser < forward_dist
                       si el spawn está cerca del fin del road network)
        has_left_neighbor: True si todo el path tiene Driving lane a la izq.
        has_right_neighbor: ídem a la derecha
        starts_in_driving: el waypoint inicial es de tipo Driving
    """
    # Importar carla aquí para no cargarlo si solo se hace --help
    import carla

    loc = sp_transform.location
    wp = carla_map.get_waypoint(loc, project_to_road=True)
    if wp is None:
        return {
            "curvature_deg": float("inf"),
            "lane_width_mean": 0.0,
            "lane_width_min": 0.0,
            "path_length_m": 0.0,
            "has_left_neighbor": False,
            "has_right_neighbor": False,
            "starts_in_driving": False,
            "error": "no_waypoint",
        }

    starts_in_driving = wp.lane_type == carla.LaneType.Driving

    cur_wp = wp
    prev_yaw = cur_wp.transform.rotation.yaw
    total_curv = 0.0
    lane_widths = [cur_wp.lane_width]
    has_left = wp.get_left_lane() is not None
    has_right = wp.get_right_lane() is not None
    distance_walked = 0.0

    n_steps = int(forward_dist / SAMPLE_STEP_M)
    for _ in range(n_steps):
        nxt = cur_wp.next(SAMPLE_STEP_M)
        if not nxt:
            break
        cur_wp = nxt[0]
        cur_yaw = cur_wp.transform.rotation.yaw
        total_curv += abs(_yaw_diff_deg(prev_yaw, cur_yaw))
        prev_yaw = cur_yaw
        lane_widths.append(cur_wp.lane_width)
        distance_walked += SAMPLE_STEP_M
        # Si en algún momento perdemos lane vecino, marcamos False.
        if cur_wp.get_left_lane() is None:
            has_left = False
        if cur_wp.get_right_lane() is None:
            has_right = False

    return {
        "curvature_deg": total_curv,
        "lane_width_mean": statistics.mean(lane_widths),
        "lane_width_min": min(lane_widths),
        "path_length_m": distance_walked,
        "has_left_neighbor": has_left,
        "has_right_neighbor": has_right,
        "starts_in_driving": starts_in_driving,
    }


def classify(detail: dict, easy_max_deg: float, medium_max_deg: float) -> str:
    """Decide tier basado en el detalle. Reglas:
    - hard: spawn inválido / no en driving lane / path corto
    - easy: low curvature AND wide lane
    - medium: low/moderate curvature
    - hard: high curvature
    """
    if not detail.get("starts_in_driving", False):
        return "hard"
    if detail["path_length_m"] < 50.0:
        # Spawn cerca del fin del road network — el agente no tiene espacio
        # para aprender lane-keep largo.
        return "hard"
    if (
        detail["curvature_deg"] < easy_max_deg
        and detail["lane_width_mean"] >= EASY_MIN_LANE_WIDTH_M
    ):
        return "easy"
    if detail["curvature_deg"] < medium_max_deg:
        return "medium"
    return "hard"


def main() -> int:
    args = parse_args()

    # Lazy import para que --help funcione sin CARLA disponible
    try:
        import carla
    except ImportError:
        logger.error(
            "No se pudo importar `carla`. Activa el venv del proyecto y asegúrate "
            "de tener `carla` Python API instalado."
        )
        return 2

    out_path = (
        Path(args.output)
        if args.output
        else Path("./spawn_profiles") / f"{args.map}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Conectando a CARLA en {args.host}:{args.port}…")
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.load_world(args.map)
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    logger.info(f"Mapa '{args.map}' cargado: {len(spawn_points)} spawn points.")

    details: dict[str, dict] = {}
    tier_lists: dict[str, list[int]] = {"easy": [], "medium": [], "hard": []}

    t_start = time.time()
    for idx, sp in enumerate(spawn_points):
        det = profile_spawn(carla_map, idx, sp, args.forward_dist)
        tier = classify(det, args.easy_max_deg, args.medium_max_deg)
        det["tier"] = tier
        # Coordenadas para inspección manual / dashboard
        det["x"] = float(sp.location.x)
        det["y"] = float(sp.location.y)
        det["yaw"] = float(sp.rotation.yaw)
        details[str(idx)] = det
        tier_lists[tier].append(idx)

    dur = time.time() - t_start

    # Resumen
    summary = {
        "map": args.map,
        "n_spawn_points": len(spawn_points),
        "forward_dist_m": args.forward_dist,
        "easy_max_deg": args.easy_max_deg,
        "medium_max_deg": args.medium_max_deg,
        "easy_min_lane_width_m": EASY_MIN_LANE_WIDTH_M,
        "tiers": tier_lists,
        "tier_counts": {k: len(v) for k, v in tier_lists.items()},
        "details": details,
    }

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Guardado JSON en {out_path}")
    logger.info(
        f"Procesado en {dur:.1f}s · "
        f"easy={len(tier_lists['easy'])} "
        f"medium={len(tier_lists['medium'])} "
        f"hard={len(tier_lists['hard'])}"
    )
    if not tier_lists["easy"]:
        logger.warning(
            "Ningún spawn clasificado como 'easy'. Considera relajar "
            "--easy_max_deg o revisar el mapa."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
