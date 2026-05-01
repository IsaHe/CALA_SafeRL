import math

import carla
import numpy as np

from src.CARLA.Sensors.SemanticScanResult import SemanticScanResult

# Tags semánticos de CARLA 0.9.14+ (alineados con CityScapes).
# IMPORTANTE: la versión 0.9.14 (release 12/2022) reasignó completamente
# los IDs respecto a 0.9.10-0.9.13. La numeración antigua era:
#   4 = Pedestrian, 10 = Vehicles, 11 = Wall, 17 = GuardRail, ...
# La nueva (que usa 0.9.16, la del proyecto) es:
#   4 = Wall, 10 = Terrain, 12 = Pedestrian, 14..19 = Car/Truck/Bus/...
# Si se usan los IDs viejos contra un servidor 0.9.14+, el labeling sale
# completamente roto: paredes salen como "peatones" y vehículos caen en
# tags que ni siquiera están en DYNAMIC. Verificado en evaluación real.
# Ref: https://carla.org/2022/12/23/release-0.9.14/
#      https://carla.readthedocs.io/en/latest/ref_sensors/

# Vehículos (Cityscapes amplía la antigua clase "Vehicles" en 6 sub-clases)
VEHICLE_TAGS: frozenset = frozenset(
    {
        14,  # Car
        15,  # Truck
        16,  # Bus
        17,  # Train
        18,  # Motorcycle
        19,  # Bicycle
    }
)

# Personas: peatón a pie + Rider (humano sobre moto/bici)
PEDESTRIAN_TAGS: frozenset = frozenset(
    {
        12,  # Pedestrian
        13,  # Rider
    }
)

# Dinámicos = vehículos + personas + tag 21 (Dynamic).
# Filtramos al ego por object_idx ANTES de llegar a estas máscaras, así
# que un vehículo en DYNAMIC nunca es el coche propio.
DYNAMIC_TAGS: frozenset = (
    VEHICLE_TAGS | PEDESTRIAN_TAGS | frozenset({21})
)  # 21 = Dynamic

# Obstáculos estáticos altos: cualquier estructura vertical no-carretera
# que pueda dañar al vehículo si lo golpea.
STATIC_OBS_TAGS: frozenset = frozenset(
    {
        3,  # Building
        4,  # Wall
        5,  # Fence
        6,  # Pole
        7,  # TrafficLight (caja)
        8,  # TrafficSign
        9,  # Vegetation
        20,  # Static (props inmovibles)
        26,  # Bridge (estructura)
        28,  # GuardRail
    }
)

# Bordes/transiciones de la calzada — útiles como "borde físico"
# pero NO son obstáculos altos. El shield los usa para riesgo lateral.
ROAD_EDGE_TAGS: frozenset = frozenset(
    {
        2,  # SideWalk — bordillo/acera, límite inmediato al salirse
        10,  # Terrain — hierba/tierra al lado de la carretera
        25,  # Ground  — superficie horizontal genérica no-carretera
        27,  # RailTrack — vías no-conducibles
    }
)

# Carretera y marcas. NO son obstáculo, pero sirven de referencia
# visual en el BEV. Los puntos en estos tags se procesan en una capa
# aparte que NO pasa por el filtro de altura (de lo contrario el
# filtro asimétrico los descartaría por ser hits a nivel del suelo).
ROAD_SURFACE_TAGS: frozenset = frozenset(
    {
        1,  # Roads
        24,  # RoadLine (marcas de carril)
    }
)

# Dtype estructurado para parsear el payload semántico (24 bytes/punto)
_SEMANTIC_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("cos_inc_angle", np.float32),
        ("object_idx", np.uint32),
        ("object_tag", np.uint32),
    ]
)


class SemanticLidarProcessor:
    """
    Procesa SemanticLidarMeasurement y produce SemanticScanResult.

    Filtro de altura ASIMÉTRICO (consciente de la altura de montaje):
      Aceptamos hits cuyo z_sensor ∈ [-(z_mount - GROUND_CLEARANCE), Z_ABOVE_MAX].
      Esto rechaza impactos contra el suelo (que generaban falsos obstáculos a
      ~3.7 m con sensor a z=1.0 y lower_fov=-15°) y mantiene guardarraíles
      bajos como obstáculos válidos.

    Pipeline (sin bucles Python sobre puntos):
      1. frombuffer con dtype estructurado  → arrays tipados en un paso
      2. Filtro ego por object_idx exacto   → elimina cuerpo propio sin heurística
      3. Filtro de altura asimétrico        → descarta suelo y señales muy altas
      4. np.isin para separar grupos semánticos en una sola operación
      5. _build_scan x 5 (combined, dynamic, static, pedestrian, road_edge)
      6. np.minimum.at vectorizado para distancia mínima por bin
      7. Pre-cálculo de mínimos por arco (frente, lados)
    """

    FRONT_N = 15
    R_START, R_END = 40, 80
    L_START, L_END = 160, 200

    GROUND_CLEARANCE = 0.15  # metros sobre la calzada que aún consideramos obstáculo
    Z_ABOVE_MAX = 1.5  # metros por encima del sensor que aún consideramos relevante

    def __init__(
        self,
        num_rays: int = 240,
        lidar_range: float = 50.0,
        height_filter: float = 1.5,
        ego_id: int = -1,
        z_mount: float = 1.0,
    ):
        self.num_rays = num_rays
        self.lidar_range = lidar_range
        self.height_filter = height_filter
        self.ego_id = ego_id
        self.z_mount = z_mount
        self.z_min_sensor = -(z_mount - self.GROUND_CLEARANCE)
        self.z_max_sensor = self.Z_ABOVE_MAX
        self._angle_step = 2.0 * math.pi / num_rays

        self._empty = np.ones(num_rays, dtype=np.float32)
        self._last = SemanticScanResult()

        # Pre-convertir frozensets a arrays numpy para isin eficiente
        self._dyn_arr = np.array(list(DYNAMIC_TAGS), dtype=np.uint32)
        self._stat_arr = np.array(list(STATIC_OBS_TAGS), dtype=np.uint32)
        self._road_edge_arr = np.array(list(ROAD_EDGE_TAGS), dtype=np.uint32)
        self._veh_arr = np.array(list(VEHICLE_TAGS), dtype=np.uint32)
        self._ped_arr = np.array(list(PEDESTRIAN_TAGS), dtype=np.uint32)
        self._road_surface_arr = np.array(list(ROAD_SURFACE_TAGS), dtype=np.uint32)

    def set_ego_id(self, ego_id: int):
        self.ego_id = ego_id

    def process(
        self, measurement: carla.SemanticLidarMeasurement
    ) -> SemanticScanResult:
        raw = np.frombuffer(measurement.raw_data, dtype=_SEMANTIC_DTYPE)

        if raw.size == 0:
            return self._empty_result()

        x = raw["x"].copy()
        y = raw["y"].copy()
        z = raw["z"].copy()
        idx = raw["object_idx"]
        tag = raw["object_tag"]

        # ── 1. Filtro ego (exacto por actor_id) ──────────────────────
        if self.ego_id >= 0:
            keep = idx != np.uint32(self.ego_id)
            x = x[keep]
            y = y[keep]
            z = z[keep]
            tag = tag[keep]

        if x.size == 0:
            return self._empty_result()

        # ── 1.5. Capturar carretera y marcas (PRE-filtro-altura) ──────
        # Roads (1) y RoadLine (24) están a nivel del suelo (world_z ≈ 0)
        # y el filtro asimétrico las descarta — correcto desde el punto
        # de vista del shield (no queremos que líneas blancas disparen
        # frenadas), pero el dashboard sí necesita verlas como referencia
        # visual. Por eso las extraemos aquí, antes del filtro de altura,
        # y las exponemos en un canal de "ground reference" aparte. NO se
        # usan para construir scans bin-eados.
        m_road = np.isin(tag, self._road_surface_arr)
        if m_road.any():
            road_dist = np.sqrt(x[m_road] ** 2 + y[m_road] ** 2)
            road_in_range = road_dist <= self.lidar_range
            road_pts_x = x[m_road][road_in_range].astype(np.float32, copy=False)
            road_pts_y = y[m_road][road_in_range].astype(np.float32, copy=False)
            road_pts_tag = tag[m_road][road_in_range].astype(np.uint32, copy=False)
        else:
            road_pts_x = np.zeros(0, dtype=np.float32)
            road_pts_y = np.zeros(0, dtype=np.float32)
            road_pts_tag = np.zeros(0, dtype=np.uint32)

        # ── 2. Filtro de altura asimétrico (rechaza suelo) ────────────
        # Sin esto, el canal inferior (lower_fov negativo) impacta el suelo
        # a d = z_mount/tan(|lower_fov|) y mete falsos obstáculos en el scan
        # (sidewalk/terrain tags) → el agente percibía obstáculos a ~3.7 m
        # constantemente y el shield disparaba sin causa real.
        hm = (z >= self.z_min_sensor) & (z <= self.z_max_sensor)
        x = x[hm]
        y = y[hm]
        tag = tag[hm]

        if x.size == 0:
            empty = self._empty_result()
            # Aun sin obstáculos válidos, conservamos los puntos de road
            # para el dashboard.
            empty.road_points_x = road_pts_x
            empty.road_points_y = road_pts_y
            empty.road_points_tag = road_pts_tag
            return empty

        # ── 3. Distancias horizontales ────────────────────────────────
        dist = np.sqrt(x**2 + y**2)

        # ── 4. Ángulos → índices de bin (convención RH anti-horaria) ─
        angles = np.arctan2(-y, x)
        angles[angles < 0] += 2.0 * math.pi
        bins = (angles / self._angle_step).astype(np.int32) % self.num_rays

        # ── 5. Máscaras semánticas (CARLA 0.9.14+) ────────────────────
        # Pedestrian agrupa Pedestrian (12) + Rider (13). Vehicle agrupa
        # Car/Truck/Bus/Train/Motorcycle/Bicycle (14..19). Es esencial
        # usar np.isin para los grupos amplios; la versión antigua usaba
        # `== np.uint32(N)` cableado a un único ID, lo que mezclaba
        # categorías al cambiar la tabla de tags entre versiones de CARLA.
        m_dyn = np.isin(tag, self._dyn_arr)
        m_stat = np.isin(tag, self._stat_arr)
        m_ped = np.isin(tag, self._ped_arr)
        m_veh = np.isin(tag, self._veh_arr)
        m_road_edge = np.isin(tag, self._road_edge_arr)
        m_comb = m_dyn | m_stat | m_road_edge

        # ── 6. Construir los 5 scans ──────────────────────────────────
        combined_s = self._build_scan(bins, dist, m_comb)
        dynamic_s = self._build_scan(bins, dist, m_dyn)
        static_s = self._build_scan(bins, dist, m_stat)
        pedestrian_s = self._build_scan(bins, dist, m_ped)
        road_edge_s = self._build_scan(bins, dist, m_road_edge)

        # ── 7. Estadísticas globales ──────────────────────────────────
        n_veh = int(m_veh.sum())
        n_ped = int(m_ped.sum())
        n_stat = int(m_stat.sum())
        n_road_edge = int(m_road_edge.sum())

        nv_m = float(dist[m_veh].min()) if n_veh > 0 else 999.0
        np_m = float(dist[m_ped].min()) if n_ped > 0 else 999.0
        ns_m = float(dist[m_stat].min()) if n_stat > 0 else 999.0
        nre_m = float(dist[m_road_edge].min()) if n_road_edge > 0 else 999.0

        u_tags, u_counts = np.unique(tag, return_counts=True)
        tag_counts = {int(t): int(c) for t, c in zip(u_tags, u_counts)}

        # ── 8. Mínimos por arco ───────────────────────────────────────
        n = self.num_rays
        fn = self.FRONT_N

        def arc_min(scan, s, e):
            return float(scan[s:e].min())

        def front_min(scan):
            return float(min(scan[n - fn :].min(), scan[:fn].min()))

        # Nube cruda post-filtros para depuración visual en BEV. Los hits
        # más allá del rango se descartan también del point map para que
        # coincida con lo que el scan bin-eado considera.
        # Excluimos road/roadline aquí: ya van en road_points_* aparte
        # (capa visual de fondo). Si los dejáramos también en points_*,
        # el dashboard los pintaría dos veces y el conteo "By tag"
        # confundiría obstáculos con superficie de carretera.
        not_road = ~np.isin(tag, self._road_surface_arr)
        in_range = (dist <= self.lidar_range) & not_road
        pts_x = x[in_range].astype(np.float32, copy=False)
        pts_y = y[in_range].astype(np.float32, copy=False)
        pts_tag = tag[in_range].astype(np.uint32, copy=False)

        result = SemanticScanResult(
            combined=combined_s,
            dynamic=dynamic_s,
            static=static_s,
            pedestrian=pedestrian_s,
            road_edge=road_edge_s,
            nearest_vehicle_m=nv_m,
            nearest_pedestrian_m=np_m,
            nearest_static_m=ns_m,
            nearest_road_edge_m=nre_m,
            min_front_combined=front_min(combined_s),
            min_front_dynamic=front_min(dynamic_s),
            min_front_static=front_min(static_s),
            min_r_side_combined=arc_min(combined_s, self.R_START, self.R_END),
            min_r_side_static=arc_min(static_s, self.R_START, self.R_END),
            min_r_side_road_edge=arc_min(road_edge_s, self.R_START, self.R_END),
            min_l_side_combined=arc_min(combined_s, self.L_START, self.L_END),
            min_l_side_static=arc_min(static_s, self.L_START, self.L_END),
            min_l_side_road_edge=arc_min(road_edge_s, self.L_START, self.L_END),
            n_vehicle_pts=n_veh,
            n_pedestrian_pts=n_ped,
            n_static_pts=n_stat,
            n_road_edge_pts=n_road_edge,
            tag_counts=tag_counts,
            points_x=pts_x,
            points_y=pts_y,
            points_tag=pts_tag,
            road_points_x=road_pts_x,
            road_points_y=road_pts_y,
            road_points_tag=road_pts_tag,
        )
        self._last = result
        return result

    # ── helpers ───────────────────────────────────────────────────────

    def _build_scan(
        self,
        bins: np.ndarray,
        dist: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Construye scan 1D para los puntos que pasan `mask`."""
        if not mask.any():
            return self._empty.copy()

        b = bins[mask]
        d = dist[mask]
        in_range = d <= self.lidar_range
        b = b[in_range]
        d = d[in_range]
        if b.size == 0:
            return self._empty.copy()

        min_d = np.full(self.num_rays, self.lidar_range, dtype=np.float32)
        np.minimum.at(min_d, b, d)

        scan = np.ones(self.num_rays, dtype=np.float32)
        hit = min_d < self.lidar_range
        scan[hit] = min_d[hit] / self.lidar_range
        return scan

    def _empty_result(self) -> SemanticScanResult:
        r = SemanticScanResult(
            combined=self._empty.copy(),
            dynamic=self._empty.copy(),
            static=self._empty.copy(),
            pedestrian=self._empty.copy(),
        )
        self._last = r
        return r

    def get_last(self) -> SemanticScanResult:
        return self._last
