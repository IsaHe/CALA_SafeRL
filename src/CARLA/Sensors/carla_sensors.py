"""
carla_sensors.py - Gestión de sensores CARLA para Safe RL

DISEÑO DEL LIDAR:
    LIDAR ALTO (techo, z=1.0 m, 3 ch, FOV [-15°, +5°], range=50 m)
        → Cobertura larga distancia, vehículos y obstáculos altos.
        → Filtro de altura ASIMÉTRICO: rechaza suelo (sin él, los hits del
          canal inferior contra el asfalto generaban falsos obstáculos a
          ~3.7 m con sidewalks como "obstáculo cercano").
        → Es el ÚNICO LIDAR del sistema. La versión v3 incluía también un
          LIDAR bajo a z=0.5 m destinado a detectar guardarraíles bajos,
          pero en evaluación se demostró redundante (todo lo que veía el
          bajo también lo veía el alto) y se eliminó.

FORMATO DEL BUFFER SEMÁNTICO (carla.SemanticLidarMeasurement.raw_data):
    Cada punto = 24 bytes = 4 float32 + 2 uint32:
        [x, y, z, cos_inc_angle, object_idx, object_tag]
    donde x=adelante, y=DERECHA, z=arriba (frame left-hand Unreal/CARLA).
    Parseado con dtype estructurado numpy (sin bucle Python).

CONVENCIÓN ANGULAR:
    ángulo = atan2(-y, x)  [y negada para LH→RH, anti-horario]
    Índice 0 → frente; creciente → izquierda del vehículo
    Frente:    bins [n-15:n] U [0:15]

GRUPOS DE TAGS SEMÁNTICOS (CityScapes, CARLA 0.9.14+):
    PEDESTRIAN_TAGS  = {12 Pedestrian, 13 Rider}
    VEHICLE_TAGS     = {14 Car, 15 Truck, 16 Bus, 17 Train,
                        18 Motorcycle, 19 Bicycle}
    DYNAMIC_TAGS     = PEDESTRIAN_TAGS ∪ VEHICLE_TAGS ∪ {21 Dynamic}
    STATIC_OBS_TAGS  = {3 Building, 4 Wall, 5 Fence, 6 Pole,
                        7 TrafficLight, 8 TrafficSign, 9 Vegetation,
                        20 Static, 26 Bridge, 28 GuardRail}
    ROAD_EDGE_TAGS   = {2 SideWalks, 10 Terrain, 25 Ground, 27 RailTrack}
    ROAD_SURFACE_TAGS= {1 Roads, 24 RoadLine}  ← solo visualización
    EGO              = filtrado por object_idx == ego_vehicle.id (exacto)
    El scan `combined` = min(dynamic, static, road_edge): el agente percibe
    obstáculos móviles, estáticos Y bordes físicos en un único canal.

BLUEPRINTS:
    ray_cast_semantic NO tiene: atmosphere_attenuation_rate, noise_stddev,
    dropoff_*. Sí comparte con ray_cast: channels, range, points_per_second,
    rotation_frequency, upper_fov, lower_fov.
"""

import carla
import numpy as np
import logging
from typing import Dict

from src.CARLA.Sensors.CollisionSensor import CollisionSensor
from src.CARLA.Sensors.SemanticLidarSensor import SemanticLidarSensor
from src.CARLA.Sensors.LaneInvasionSensor import LaneInvasionSensor
from src.CARLA.Sensors.SemanticScanResult import SemanticScanResult


logger = logging.getLogger(__name__)


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
        height_filter: float = 0.5,
    ):
        self.lidar = SemanticLidarSensor(
            world,
            vehicle,
            num_rays=num_lidar_rays,
            lidar_range=lidar_range,
            height_filter=height_filter,
        )
        self.collision = CollisionSensor(world, vehicle)
        self.lane_invasion = LaneInvasionSensor(world, vehicle)

    def get_semantic_result(
        self, expected_frame: int = None, timeout: float = 1.0
    ) -> SemanticScanResult:
        """
        Si expected_frame se pasa, bloquea hasta llegar un frame coincidente
        (patrón canónico CARLA synchronous_mode.py). Sin él, comportamiento
        legacy (drenado no bloqueante).
        """
        return self.lidar.get_result(expected_frame=expected_frame, timeout=timeout)

    def get_semantic_status(self) -> Dict[str, int]:
        return self.lidar.get_status()

    def update_ego_id(self, new_id: int) -> None:
        """
        Defensa en profundidad: actualizar el ego_id del filtro del LIDAR
        semántico. Hoy el SensorManager se recrea en cada reset() del
        entorno y el ego_id se recoge bien al construirse, pero llamar
        a esta función explícitamente garantiza que cualquier cambio en
        el flujo de reset no introduzca un bug silencioso de ego visible
        como obstáculo.
        """
        self.lidar.update_ego_id(new_id)

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
