from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass
class SemanticScanResult:
    """
    Resultado completo de un frame de LIDAR semántico (v2).

    Scans (np.ndarray float32 shape=(num_rays,), rango [0,1]):
      combined   → min(dynamic, static, road_edge): scan de observación principal.
                   Incluye bordes de carretera para que el agente perciba límites físicos.
      dynamic    → solo vehículos (no-ego) + peatones + objetos dinámicos
      static     → solo obstáculos estáticos (muros, quitamiedos, postes, ...)
      pedestrian → sub-scan solo peatones (señal independiente de alta prioridad)
      road_edge  → acera (8) + terreno (22): límite físico de la calzada

    Distancias globales (metros reales, no normalizadas):
      nearest_vehicle_m / nearest_pedestrian_m / nearest_static_m / nearest_road_edge_m

    Mínimos por arco (normalizados [0,1], pre-calculados):
      Frente  (±FRONT_N bins): min_front_combined / _dynamic / _static
      Derecha (bins R_START:R_END): min_r_side_combined / _static / _road_edge
      Izquierda (bins L_START:L_END): min_l_side_combined / _static / _road_edge

    Uso recomendado en el shield adaptativo:
      - risk_level frontal ← min_front_dynamic
      - riesgo lateral     ← min(min_r_side_road_edge, min_l_side_road_edge)
      - freno check        ← min_front_combined
      - steer check        ← min_r/l_side_static
      - TTC peatón         ← nearest_pedestrian_m / speed_ms
    """

    combined: np.ndarray = field(default_factory=lambda: np.ones(240, dtype=np.float32))
    dynamic: np.ndarray = field(default_factory=lambda: np.ones(240, dtype=np.float32))
    static: np.ndarray = field(default_factory=lambda: np.ones(240, dtype=np.float32))
    pedestrian: np.ndarray = field(
        default_factory=lambda: np.ones(240, dtype=np.float32)
    )
    road_edge: np.ndarray = field(
        default_factory=lambda: np.ones(240, dtype=np.float32)
    )

    nearest_vehicle_m: float = 999.0
    nearest_pedestrian_m: float = 999.0
    nearest_static_m: float = 999.0
    nearest_road_edge_m: float = 999.0

    min_front_combined: float = 1.0
    min_front_dynamic: float = 1.0
    min_front_static: float = 1.0
    min_r_side_combined: float = 1.0
    min_r_side_static: float = 1.0
    min_r_side_road_edge: float = 1.0
    min_l_side_combined: float = 1.0
    min_l_side_static: float = 1.0
    min_l_side_road_edge: float = 1.0

    n_vehicle_pts: int = 0
    n_pedestrian_pts: int = 0
    n_static_pts: int = 0
    n_road_edge_pts: int = 0
    tag_counts: Dict[int, int] = field(default_factory=dict)

    # ── Nube de puntos cruda post-filtros (ego + altura) ──────────────────
    # Coordenadas en frame del sensor (UE LH: x=adelante, y=derecha).
    # Se rellenan con los puntos efectivamente usados para construir los
    # scans bin-eados, lo que permite verificar visualmente en BEV qué
    # está viendo el agente (point map de debug). Por defecto vacíos para
    # no penalizar memoria si no se consumen.
    points_x: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    points_y: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    points_tag: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.uint32)
    )

    # ── Nube de carretera (PRE-filtro-altura) — debug visual ────────────
    # Roads (tag 1) y RoadLine (tag 24) están a nivel del suelo. El filtro
    # asimétrico de altura los descarta — correcto para safety, pero el
    # dashboard los necesita como referencia visual de la calzada y las
    # marcas de carril. Se capturan ANTES del filtro de altura y NO se
    # usan para construir los scans bin-eados.
    road_points_x: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    road_points_y: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    road_points_tag: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.uint32)
    )

    def to_info_dict(self) -> Dict[str, float]:
        """Entradas del info dict de CarlaEnv listas para usar."""
        return {
            # ── Scans completos ──
            "lidar_scan": self.combined,
            "lidar_dynamic_scan": self.dynamic,
            "lidar_static_scan": self.static,
            "lidar_pedestrian_scan": self.pedestrian,
            "lidar_road_edge_scan": self.road_edge,
            # ── Distancias globales (metros) ──
            "nearest_vehicle_m": self.nearest_vehicle_m,
            "nearest_pedestrian_m": self.nearest_pedestrian_m,
            "nearest_static_m": self.nearest_static_m,
            "nearest_road_edge_m": self.nearest_road_edge_m,
            # ── Mínimos por arco combinados ──
            "min_front_dist": self.min_front_combined,
            "min_lidar_dist": float(np.min(self.combined)),
            # ── Mínimos semánticos por arco ──
            "min_front_combined": self.min_front_combined,
            "min_front_dynamic": self.min_front_dynamic,
            "min_front_static": self.min_front_static,
            "min_r_side_combined": self.min_r_side_combined,
            "min_r_side_static": self.min_r_side_static,
            "min_r_side_road_edge": self.min_r_side_road_edge,
            "min_l_side_combined": self.min_l_side_combined,
            "min_l_side_static": self.min_l_side_static,
            "min_l_side_road_edge": self.min_l_side_road_edge,
            # ── Conteos y densidad ──
            "n_vehicle_pts": self.n_vehicle_pts,
            "n_pedestrian_pts": self.n_pedestrian_pts,
            "n_static_pts": self.n_static_pts,
            "n_road_edge_pts": self.n_road_edge_pts,
            "semantic_tag_counts": self.tag_counts,
            # ── Nube de puntos post-filtros (debug / point map BEV) ──
            "lidar_points_x": self.points_x,
            "lidar_points_y": self.points_y,
            "lidar_points_tag": self.points_tag,
            # ── Carretera + marcas (PRE-filtro-altura, solo visual) ──
            "lidar_road_points_x": self.road_points_x,
            "lidar_road_points_y": self.road_points_y,
            "lidar_road_points_tag": self.road_points_tag,
        }
