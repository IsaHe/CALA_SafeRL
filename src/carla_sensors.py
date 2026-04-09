"""
carla_sensors.py - Gestión de sensores CARLA para Safe RL


DISEÑO DEL LIDAR:
    CARLA LIDAR con channels=1 y rotation_frequency=20 Hz (igual que fixed_delta_seconds)
    → 1 rotación completa por tick de simulación
    → points_per_second = num_rays * rotation_frequency = 240 * 20 = 4800
    → Resultado: exactamente 240 puntos angularmente equidistantes por tick
    → Proyectamos al plano horizontal → scan 2D exactamente como MetaDrive
 
FORMATO DEL BUFFER SEMÁNTICO (carla.SemanticLidarMeasurement.raw_data):
    Cada punto = 24 bytes = 4 float32 + 2 uint32:
        [x, y, z, cos_inc_angle, object_idx, object_tag]
    donde x=adelante, y=DERECHA, z=arriba (frame left-hand Unreal/CARLA).
    Parseado con dtype estructurado numpy (sin bucle Python).
 
CONVENCIÓN ANGULAR (idéntica a versión anterior):
    ángulo = atan2(-y, x)  [y negada para LH→RH, anti-horario]
    Índice 0 → frente; creciente → izquierda del vehículo
    Frente:    bins [n-15:n] U [0:15]
    Izquierda: bins [40:80]
    Derecha:   bins [160:200]

GRUPOS DE TAGS SEMÁNTICOS (CityScapes):
    DYNAMIC_TAGS      = {4 Peatón, 10 Vehículo, 20 Dinámico}
    STATIC_OBS_TAGS   = {1 Edificio, 2 Valla, 5 Poste, 9 Vegetación,
                         11 Muro, 12 Señal, 15 Puente, 17 Quitamiedos,
                         18 Semáforo, 19 Estático}
    GROUND_TAGS       = {6 RoadLine, 7 Road, 8 Acera, 14 Suelo, 16 Via, 22 Terreno}
    EXCLUDED          = {0 Unlabeled, 3 Other, 13 Sky, 21 Water}
    EGO               = filtrado por object_idx == ego_vehicle.id (exacto, sin heurística)
 
BLUEPRINTS:
    ray_cast_semantic NO tiene: atmosphere_attenuation_rate, noise_stddev, dropoff_*
    Sí comparte con ray_cast: channels, range, points_per_second,
                               rotation_frequency, upper_fov, lower_fov
"""

import carla
import numpy as np
import queue
import weakref
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


DYNAMIC_TAGS: frozenset = frozenset({
    4,   # Pedestrian
    10,  # Vehicles  ← ego filtrado por object_idx antes de llegar aquí
    20,  # Dynamic
})
 
STATIC_OBS_TAGS: frozenset = frozenset({
    1,   # Building
    2,   # Fence
    5,   # Pole
    9,   # Vegetation
    11,  # Wall
    12,  # TrafficSign
    15,  # Bridge
    17,  # GuardRail
    18,  # TrafficLight
    19,  # Static
})
 
GROUND_TAGS: frozenset = frozenset({
    6,   # RoadLine
    7,   # Road
    8,   # SideWalk
    14,  # Ground
    16,  # RailTrack
    22,  # Terrain
})
 
EXCLUDED_TAGS: frozenset = frozenset({0, 3, 13, 21})  # Unlabeled, Other, Sky, Water
 
# Dtype estructurado para parsear el payload semántico (24 bytes/punto)
_SEMANTIC_DTYPE = np.dtype([
    ("x",             np.float32),
    ("y",             np.float32),
    ("z",             np.float32),
    ("cos_inc_angle", np.float32),
    ("object_idx",    np.uint32),
    ("object_tag",    np.uint32),
])
 
TAG_NAMES: Dict[int, str] = {
    0: "Unlabeled", 1: "Building",  2: "Fence",       3: "Other",
    4: "Pedestrian",5: "Pole",      6: "RoadLine",     7: "Road",
    8: "SideWalk",  9: "Vegetation",10: "Vehicle",    11: "Wall",
    12: "TrafficSign",13:"Sky",     14:"Ground",      15:"Bridge",
    16:"RailTrack", 17:"GuardRail", 18:"TrafficLight",19:"Static",
    20:"Dynamic",   21:"Water",     22:"Terrain",
}

@dataclass
class SemanticScanResult:
    """
    Resultado completo de un frame de LIDAR semántico.
 
    Scans (np.ndarray float32 shape=(num_rays,), rango [0,1]):
      combined   → min(dynamic, static): scan de observación, compat con shields
      dynamic    → solo vehículos (no-ego) + peatones + objetos dinámicos
      static     → solo obstáculos estáticos (muros, quitamiedos, postes, ...)
      pedestrian → sub-scan solo peatones (señal independiente de alta prioridad)
 
    Distancias globales (metros reales, no normalizadas):
      nearest_vehicle_m / nearest_pedestrian_m / nearest_static_m
 
    Mínimos por arco (normalizados [0,1], pre-calculados):
      Frente  (±FRONT_N bins): min_front_combined / _dynamic / _static
      Derecha (bins R_START:R_END): min_r_side_combined / _static
      Izquierda (bins L_START:L_END): min_l_side_combined / _static
 
    Uso recomendado en el shield adaptativo:
      - risk_level  ← min_front_dynamic   (evitar falsos críticos por quitamiedos)
      - freno check ← min_front_combined  (muros frontales son peligrosos)
      - steer check ← min_r/l_side_static (quitamiedos definen límite lateral)
      - TTC peatón  ← nearest_pedestrian_m / speed_ms
    """
    combined:    np.ndarray = field(default_factory=lambda: np.ones(240, dtype=np.float32))
    dynamic:     np.ndarray = field(default_factory=lambda: np.ones(240, dtype=np.float32))
    static:      np.ndarray = field(default_factory=lambda: np.ones(240, dtype=np.float32))
    pedestrian:  np.ndarray = field(default_factory=lambda: np.ones(240, dtype=np.float32))
 
    nearest_vehicle_m:    float = 999.0
    nearest_pedestrian_m: float = 999.0
    nearest_static_m:     float = 999.0
 
    min_front_combined:   float = 1.0
    min_front_dynamic:    float = 1.0
    min_front_static:     float = 1.0
    min_r_side_combined:  float = 1.0
    min_r_side_static:    float = 1.0
    min_l_side_combined:  float = 1.0
    min_l_side_static:    float = 1.0
 
    n_vehicle_pts:    int = 0
    n_pedestrian_pts: int = 0
    n_static_pts:     int = 0
    tag_counts: Dict[int, int] = field(default_factory=dict)
 
    def to_info_dict(self) -> Dict:
        """Entradas del info dict de CarlaEnv listas para usar."""
        return {
            # ── Scans completos ──
            "lidar_scan":            self.combined,     # backward compat shields
            "lidar_dynamic_scan":    self.dynamic,
            "lidar_static_scan":     self.static,
            "lidar_pedestrian_scan": self.pedestrian,
 
            # ── Distancias globales (metros) ──
            "nearest_vehicle_m":    self.nearest_vehicle_m,
            "nearest_pedestrian_m": self.nearest_pedestrian_m,
            "nearest_static_m":     self.nearest_static_m,
 
            # ── Mínimos por arco combinados (compat shields) ──
            "min_front_dist":        self.min_front_combined,
            "min_lidar_dist":        float(np.min(self.combined)),
 
            # ── Mínimos semánticos por arco ──
            "min_front_combined":    self.min_front_combined,
            "min_front_dynamic":     self.min_front_dynamic,
            "min_front_static":      self.min_front_static,
            "min_r_side_combined":   self.min_r_side_combined,
            "min_r_side_static":     self.min_r_side_static,
            "min_l_side_combined":   self.min_l_side_combined,
            "min_l_side_static":     self.min_l_side_static,
 
            # ── Conteos y densidad ──
            "n_vehicle_pts":         self.n_vehicle_pts,
            "n_pedestrian_pts":      self.n_pedestrian_pts,
            "n_static_pts":          self.n_static_pts,
            "semantic_tag_counts":   self.tag_counts,
        }
    
class SemanticLidarProcessor:
    """
    Procesa SemanticLidarMeasurement y produce SemanticScanResult.
 
    Pipeline (sin bucles Python sobre puntos):
      1. frombuffer con dtype estructurado  → arrays tipados en un paso
      2. Filtro ego por object_idx exacto   → elimina cuerpo propio sin heurística
      3. Filtro de altura ±height_filter    → descarta señales elevadas y suelo rasante
      4. np.isin para separar grupos semánticos en una sola operación
      5. _build_scan x 4 (combined, dynamic, static, pedestrian)
      6. np.minimum.at vectorizado para distancia mínima por bin
      7. Pre-cálculo de mínimos por arco (frente, lados)
    """
 
    FRONT_N         = 15
    R_START, R_END  = 40, 80
    L_START, L_END  = 160, 200
 
    def __init__(
        self,
        num_rays:      int   = 240,
        lidar_range:   float = 50.0,
        height_filter: float = 0.5,
        ego_id:        int   = -1,
    ):
        self.num_rays      = num_rays
        self.lidar_range   = lidar_range
        self.height_filter = height_filter
        self.ego_id        = ego_id
        self._angle_step   = 2.0 * math.pi / num_rays
 
        self._empty  = np.ones(num_rays, dtype=np.float32)
        self._last   = SemanticScanResult()
 
        # Pre-convertir frozensets a arrays numpy para isin eficiente
        self._dyn_arr  = np.array(list(DYNAMIC_TAGS),     dtype=np.uint32)
        self._stat_arr = np.array(list(STATIC_OBS_TAGS),  dtype=np.uint32)
 
    def set_ego_id(self, ego_id: int):
        self.ego_id = ego_id
 
    def process(self, measurement: carla.SemanticLidarMeasurement) -> SemanticScanResult:
        raw = np.frombuffer(measurement.raw_data, dtype=_SEMANTIC_DTYPE)
 
        if raw.size == 0:
            return self._empty_result()
 
        x   = raw["x"].copy()
        y   = raw["y"].copy()
        z   = raw["z"].copy()
        idx = raw["object_idx"]
        tag = raw["object_tag"]
 
        # ── 1. Filtro ego (exacto por actor_id) ──────────────────────
        if self.ego_id >= 0:
            keep = idx != np.uint32(self.ego_id)
            x = x[keep]; y = y[keep]; z = z[keep]; tag = tag[keep]
 
        if x.size == 0:
            return self._empty_result()
 
        # ── 2. Filtro de altura ───────────────────────────────────────
        hm = np.abs(z) <= self.height_filter
        x = x[hm]; y = y[hm]; tag = tag[hm]
 
        if x.size == 0:
            return self._empty_result()
 
        # ── 3. Distancias horizontales ────────────────────────────────
        dist = np.sqrt(x**2 + y**2)
 
        # ── 4. Ángulos → índices de bin (convención RH anti-horaria) ─
        angles  = np.arctan2(-y, x)            # -y: CARLA LH → RH
        angles[angles < 0] += 2.0 * math.pi
        bins = (angles / self._angle_step).astype(np.int32) % self.num_rays
 
        # ── 5. Máscaras semánticas ────────────────────────────────────
        m_dyn  = np.isin(tag, self._dyn_arr)
        m_stat = np.isin(tag, self._stat_arr)
        m_ped  = (tag == np.uint32(4))
        m_veh  = (tag == np.uint32(10))
        m_comb = m_dyn | m_stat        # excluye suelo, sky, ego ya filtrado
 
        # ── 6. Construir los 4 scans ──────────────────────────────────
        combined_s    = self._build_scan(bins, dist, m_comb)
        dynamic_s     = self._build_scan(bins, dist, m_dyn)
        static_s      = self._build_scan(bins, dist, m_stat)
        pedestrian_s  = self._build_scan(bins, dist, m_ped)
 
        # ── 7. Estadísticas globales ──────────────────────────────────
        n_veh  = int(m_veh.sum())
        n_ped  = int(m_ped.sum())
        n_stat = int(m_stat.sum())
 
        nv_m = float(dist[m_veh].min())  if n_veh  > 0 else 999.0
        np_m = float(dist[m_ped].min())  if n_ped  > 0 else 999.0
        ns_m = float(dist[m_stat].min()) if n_stat > 0 else 999.0
 
        u_tags, u_counts = np.unique(tag, return_counts=True)
        tag_counts = {int(t): int(c) for t, c in zip(u_tags, u_counts)}
 
        # ── 8. Mínimos por arco ───────────────────────────────────────
        n = self.num_rays
        fn = self.FRONT_N
 
        def arc_min(scan, s, e):
            return float(scan[s:e].min())
 
        def front_min(scan):
            return float(min(scan[n - fn:].min(), scan[:fn].min()))
 
        result = SemanticScanResult(
            combined    = combined_s,
            dynamic     = dynamic_s,
            static      = static_s,
            pedestrian  = pedestrian_s,
 
            nearest_vehicle_m    = nv_m,
            nearest_pedestrian_m = np_m,
            nearest_static_m     = ns_m,
 
            min_front_combined   = front_min(combined_s),
            min_front_dynamic    = front_min(dynamic_s),
            min_front_static     = front_min(static_s),
 
            min_r_side_combined  = arc_min(combined_s, self.R_START, self.R_END),
            min_r_side_static    = arc_min(static_s,   self.R_START, self.R_END),
            min_l_side_combined  = arc_min(combined_s, self.L_START, self.L_END),
            min_l_side_static    = arc_min(static_s,   self.L_START, self.L_END),
 
            n_vehicle_pts    = n_veh,
            n_pedestrian_pts = n_ped,
            n_static_pts     = n_stat,
            tag_counts       = tag_counts,
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
        b = b[in_range];  d = d[in_range]
        if b.size == 0:
            return self._empty.copy()
 
        min_d = np.full(self.num_rays, self.lidar_range, dtype=np.float32)
        np.minimum.at(min_d, b, d)
 
        scan = np.ones(self.num_rays, dtype=np.float32)
        hit  = min_d < self.lidar_range
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

class SemanticLidarSensor:
    """
    Wrapper sobre sensor.lidar.ray_cast_semantic de CARLA.
 
    Misma configuración de frecuencia/puntos que la versión anterior
    (channels=1, 20Hz, 240 pts/rev) para mantener la densidad angular.
 
    Diferencias de blueprint vs ray_cast:
      - NO tiene: atmosphere_attenuation_rate, noise_stddev, dropoff_*
      - Payload por punto: +8 bytes (object_idx uint32 + object_tag uint32)
      - Coste CPU: ~3-5% mayor que ray_cast en el parse, pero se elimina
        el filtro min_dist que perdía puntos de NPCs cercanos.
    """
 
    def __init__(
        self,
        world:              carla.World,
        vehicle:            carla.Vehicle,
        num_rays:           int   = 240,
        lidar_range:        float = 50.0,
        rotation_frequency: float = 20.0,
        height_offset:      float = 1.8,
        height_filter:      float = 0.5,
    ):
        self._num_rays = num_rays
        self.processor = SemanticLidarProcessor(
            num_rays      = num_rays,
            lidar_range   = lidar_range,
            height_filter = height_filter,
            ego_id        = vehicle.id,
        )
        self._queue: queue.Queue = queue.Queue()
 
        bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        bp.set_attribute("channels",           "1")
        bp.set_attribute("range",              str(lidar_range))
        bp.set_attribute("rotation_frequency", str(rotation_frequency))
        bp.set_attribute("points_per_second",  str(int(num_rays * rotation_frequency)))
        bp.set_attribute("upper_fov",          "1.0")
        bp.set_attribute("lower_fov",          "-1.0")
        # Note: ray_cast_semantic does NOT accept atmosphere_attenuation_rate
 
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=0.0, z=height_offset)),
            attach_to=vehicle,
        )
        self.sensor.listen(lambda data: self._queue.put(data))
        self._last = SemanticScanResult()
        self._last_was_fresh = False
        self._stale_reads = 0
        self._fresh_reads = 0
        self._last_frame = -1
        logger.debug(f"SemanticLidarSensor: ego_id={vehicle.id}")
 
    def update_ego_id(self, new_id: int):
        """Llamar tras respawn del ego vehicle."""
        self.processor.set_ego_id(new_id)
 
    def get_result(self) -> SemanticScanResult:
        latest = None
        while not self._queue.empty():
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._last = self.processor.process(latest)
            self._last_was_fresh = True
            self._fresh_reads += 1
            self._last_frame = int(getattr(latest, "frame", -1))
        else:
            self._last_was_fresh = False
            self._stale_reads += 1
        return self._last

    def get_status(self) -> Dict[str, int]:
        return {
            "fresh": int(self._last_was_fresh),
            "stale_reads": int(self._stale_reads),
            "fresh_reads": int(self._fresh_reads),
            "last_frame": int(self._last_frame),
        }
 
    def get_scan(self) -> np.ndarray:
        """Backward compat: combined_scan."""
        return self.get_result().combined.copy()
 
    def destroy(self):
        if self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()

class CollisionSensor:
    """
    Sensor de colisión CARLA con tracking de impulso físico.
    """

    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        self._collision_flag = False
        self._collision_impulse = 0.0
        self._collision_actor_type = ""

        bp = world.get_blueprint_library().find("sensor.other.collision")
        transform = carla.Transform()
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event)
        )

    @staticmethod
    def _on_collision(weak_self, event: carla.CollisionEvent):
        self = weak_self()
        if not self:
            return
        self._collision_flag = True
        imp = event.normal_impulse
        self._collision_impulse = math.sqrt(imp.x**2 + imp.y**2 + imp.z**2)
        self._collision_actor_type = event.other_actor.type_id

    def get_collision(self) -> bool:
        """Retorna True si hubo colisión desde el último get (y la resetea)."""
        result = self._collision_flag
        self._collision_flag = False
        return result

    def get_impulse(self) -> float:
        """Retorna el impulso de la última colisión."""
        return self._collision_impulse

    def destroy(self):
        if self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


class LaneInvasionSensor:
    """
    Sensor de invasión de carril NATIVO de CARLA (solo detecta cruces de líneas sólidas).
    """

    SOLID_TYPES = frozenset({
        carla.LaneMarkingType.Solid,
        carla.LaneMarkingType.SolidSolid,
        carla.LaneMarkingType.SolidBroken,
        carla.LaneMarkingType.BrokenSolid,
    })

    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        self._invasion_flag = False

        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        transform = carla.Transform()
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event)
        )

    @staticmethod
    def _on_invasion(weak_self, event: carla.LaneInvasionEvent):
        self = weak_self()
        if not self:
            return
        crossed = event.crossed_lane_markings
        # Solo marcar si cruza una línea sólida (no discontinua)
        for marking in crossed:
            if marking.type in LaneInvasionSensor.SOLID_TYPES:
                self._invasion_flag = True
                break

    def get_invasion(self) -> bool:
        """Retorna True si hubo invasión de carril desde el último get."""
        result = self._invasion_flag
        self._invasion_flag = False
        return result

    def destroy(self):
        if self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


class SensorManager:
    """
    Gestor unificado de todos los sensores del vehículo ego.

    Interfaz única para CarlaEnv, que abstrae la complejidad
    de múltiples sensores y sus callbacks asíncronos.
    """

    def __init__(
        self,
        world: carla.World,
        vehicle: carla.Vehicle,
        num_lidar_rays: int = 240,
        lidar_range: float = 50.0,
        height_filter:  float = 0.5,
    ):
        self.lidar = SemanticLidarSensor(
            world, vehicle,
            num_rays = num_lidar_rays,
            lidar_range = lidar_range,
            height_filter = height_filter,
        )
        self.collision = CollisionSensor(world, vehicle)
        self.lane_invasion = LaneInvasionSensor(world, vehicle)

        logger.info(
            f"SensorManager: SemanticLIDAR(ego={vehicle.id}, "
            f"rays={num_lidar_rays}, range={lidar_range}m, h±{height_filter}m)"
        )

    def get_semantic_result(self) -> SemanticScanResult:
        return self.lidar.get_result()

    def get_semantic_status(self) -> Dict[str, int]:
        return self.lidar.get_status()
 
    def get_lidar_scan(self) -> np.ndarray:
        return self.lidar.get_scan()
 
    def get_collision(self) -> bool:
        return self.collision.get_collision()
 
    def get_lane_invasion(self) -> bool:
        return self.lane_invasion.get_invasion()
 
    def destroy(self):
        for s in [self.lidar, self.collision, self.lane_invasion]:
            try:
                s.destroy()
            except Exception as e:
                logger.warning(f"Error destroying sensor: {e}")