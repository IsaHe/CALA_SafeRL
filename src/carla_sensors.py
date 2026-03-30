"""
carla_sensors.py - Gestión de sensores CARLA para Safe RL

Reemplaza la inferencia heurística de MetaDrive con sensores físicos reales:

    LIDARSensor       → 240 rayos horizontales, procesados a scan 1D [0,1]
                        (compatible con todos los shields del proyecto)
    CollisionSensor   → evento + impulso físico (permite discriminar raspaduras)
    LaneInvasionSensor→ NATIVO de CARLA; detecta cruce de líneas de carril
                        (elimina completamente LaneAwarenessWrapper)
    SensorManager     → interfaz unificada, gestión de ciclo de vida

DISEÑO DEL LIDAR:
    CARLA LIDAR con channels=1 y rotation_frequency=20 Hz (igual que fixed_delta_seconds)
    → 1 rotación completa por tick de simulación
    → points_per_second = num_rays * rotation_frequency = 240 * 20 = 4800
    → Resultado: exactamente 240 puntos angularmente equidistantes por tick
    → Proyectamos al plano horizontal → scan 2D exactamente como MetaDrive

CONVENCIÓN ANGULAR (compatible con MetaDrive shields):
    Índice 0   → frente del vehículo (0°)
    Índices crecientes → sentido anti-horario
    Frente: lidar[-15:] + lidar[:15]
    Lado derecho: lidar[40:80]
    Lado izquierdo: lidar[160:200]
"""

import carla
import numpy as np
import queue
import weakref
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LIDARProcessor:
    """
    Convierte el point cloud 3D de CARLA en un scan 2D de N rayos normalizado.

    El scan resultante tiene la misma semántica que el LIDAR de MetaDrive:
      - valor 1.0  = sin obstáculo (distancia máxima)
      - valor ~0.0 = obstáculo muy cercano

    Frame del sensor CARLA (left-hand): x=adelante, y=izquierda, z=arriba
    Convertimos a frame right-hand: ángulo 0=adelante, crece antihorario.
    """

    def __init__(self, num_rays: int = 240, lidar_range: float = 50.0):
        self.num_rays = num_rays
        self.lidar_range = lidar_range
        self._scan = np.ones(num_rays, dtype=np.float32)
        self._angle_step = 2.0 * math.pi / num_rays

    def process(self, lidar_measurement: carla.LidarMeasurement) -> np.ndarray:
        """
        Procesa un LidarMeasurement y retorna scan normalizado de forma (num_rays,).

        Usa raw_data (buffer numpy float32) en lugar de iterar punto a punto.
        El formato del buffer de CARLA LidarMeasurement es:
            [x0, y0, z0, intensity0, x1, y1, z1, intensity1, ...]
        donde x=adelante, y=izquierda, z=arriba (frame left-hand de CARLA).

        Filtra puntos por:
          - Distancia mínima de 0.5m (elimina reflexiones del propio vehículo)
          - Distancia máxima = lidar_range
          - Altura: solo puntos entre -0.5m y +0.5m del sensor (plano horizontal)
        """
        min_dist_per_bin = np.full(self.num_rays, self.lidar_range, dtype=np.float32)

        # Convertir buffer binario a array numpy: shape (N, 4) → x, y, z, intensity
        raw = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32)

        if raw.size == 0:
            # Sin puntos este tick (puede ocurrir en los primeros frames)
            self._scan = np.ones(self.num_rays, dtype=np.float32)
            return self._scan.copy()

        # Reshape: cada fila = [x, y, z, intensity]
        points = raw.reshape(-1, 4)
        x_arr = points[:, 0]
        y_arr = points[:, 1]
        z_arr = points[:, 2]

        # Filtro de altura: mantener solo capa horizontal (±0.5m)
        height_mask = np.abs(z_arr) <= 0.5
        x_arr = x_arr[height_mask]
        y_arr = y_arr[height_mask]

        if x_arr.size == 0:
            self._scan = np.ones(self.num_rays, dtype=np.float32)
            return self._scan.copy()

        # Distancia en plano horizontal
        dist_arr = np.sqrt(x_arr ** 2 + y_arr ** 2)

        # Filtro de distancia
        dist_mask = (dist_arr >= 0.5) & (dist_arr <= self.lidar_range)
        x_arr    = x_arr[dist_mask]
        y_arr    = y_arr[dist_mask]
        dist_arr = dist_arr[dist_mask]

        if x_arr.size == 0:
            self._scan = np.ones(self.num_rays, dtype=np.float32)
            return self._scan.copy()

        # Ángulo right-hand: 0=adelante, aumenta anti-horario
        # CARLA frame: x=adelante, y=izquierda (left-hand)
        # atan2(y, x) da directamente el ángulo con esa convención
        angles = np.arctan2(y_arr, x_arr)
        angles[angles < 0] += 2.0 * math.pi  # [0, 2π)

        # Bin angular para cada punto
        ray_indices = (angles / self._angle_step).astype(np.int32) % self.num_rays

        # Asignar distancia mínima por bin (vectorizado)
        np.minimum.at(min_dist_per_bin, ray_indices, dist_arr)

        # Normalizar a [0, 1]: 1=libre, ~0=obstáculo muy cercano
        scan = np.ones(self.num_rays, dtype=np.float32)
        mask = min_dist_per_bin < self.lidar_range
        scan[mask] = min_dist_per_bin[mask] / self.lidar_range

        self._scan = scan
        return scan.copy()

    def get_scan(self) -> np.ndarray:
        return self._scan.copy()


class LIDARSensor:
    """
    Sensor LIDAR de CARLA configurado para generar un scan 2D de N rayos.

    Configuración:
      channels=1          → una única capa horizontal
      rotation_frequency  → sincronizado con delta_seconds (20 Hz)
      points_per_second   → num_rays * rotation_frequency
      upper/lower_fov     → ±1° alrededor del plano horizontal
    """

    def __init__(
        self,
        world: carla.World,
        vehicle: carla.Vehicle,
        num_rays: int = 240,
        lidar_range: float = 50.0,
        rotation_frequency: float = 20.0,
        height_offset: float = 1.8,  # metros sobre el centro del vehículo
    ):
        self.processor = LIDARProcessor(num_rays=num_rays, lidar_range=lidar_range)
        self._data_queue: queue.Queue = queue.Queue()

        bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        bp.set_attribute("channels", "1")
        bp.set_attribute("range", str(lidar_range))
        bp.set_attribute("rotation_frequency", str(rotation_frequency))
        bp.set_attribute("points_per_second", str(int(num_rays * rotation_frequency)))
        bp.set_attribute("upper_fov", "1.0")
        bp.set_attribute("lower_fov", "-1.0")
        bp.set_attribute("atmosphere_attenuation_rate", "0.0")  # sin atenuación

        transform = carla.Transform(carla.Location(x=0.0, z=height_offset))
        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.sensor.listen(lambda data: self._data_queue.put(data))

        # Scan inicial (todo 1.0 = libre)
        self._last_scan = np.ones(num_rays, dtype=np.float32)

    def get_scan(self) -> np.ndarray:
        """Retorna el scan más reciente disponible."""
        latest = None
        while not self._data_queue.empty():
            try:
                latest = self._data_queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            self._last_scan = self.processor.process(latest)

        return self._last_scan.copy()

    def destroy(self):
        if self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


class CollisionSensor:
    """
    Sensor de colisión CARLA con tracking de impulso físico.

    A diferencia de MetaDrive (flag binario), CARLA provee:
      - impulso vectorial real (N·s)
      - actor con el que colisionó
    Esto permite al shield distinguir raspaduras de colisiones graves.
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
    Sensor de invasión de carril NATIVO de CARLA.

    Detecta el cruce de líneas de carril sin heurísticas.
    Solo reporta cruce de líneas sólidas (no líneas discontinuas ni bordes).

    Ventaja sobre MetaDrive LaneAwarenessWrapper:
      - Detección física real, no basada en posición Y
      - Distingue tipo de línea (sólida, discontinua, borde de carretera)
      - Cero falsos positivos en intersecciones
    """

    SOLID_TYPES = {
        carla.LaneMarkingType.Solid,
        carla.LaneMarkingType.SolidSolid,
        carla.LaneMarkingType.SolidBroken,
        carla.LaneMarkingType.BrokenSolid,
    }

    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        self._invasion_flag = False
        self._crossed_types: list = []

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
                self._crossed_types.append(str(marking.type))
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
    ):
        self.lidar = LIDARSensor(
            world, vehicle,
            num_rays=num_lidar_rays,
            lidar_range=lidar_range,
        )
        self.collision = CollisionSensor(world, vehicle)
        self.lane_invasion = LaneInvasionSensor(world, vehicle)

        logger.info("SensorManager initialized: LIDAR + Collision + LaneInvasion")

    def get_lidar_scan(self) -> np.ndarray:
        return self.lidar.get_scan()

    def get_collision(self) -> bool:
        return self.collision.get_collision()

    def get_lane_invasion(self) -> bool:
        return self.lane_invasion.get_invasion()

    def destroy(self):
        """Destruye todos los sensores de forma segura."""
        for sensor in [self.lidar, self.collision, self.lane_invasion]:
            try:
                sensor.destroy()
            except Exception as e:
                logger.warning(f"Error destroying sensor: {e}")