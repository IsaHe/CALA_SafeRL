"""
carla_sensors.py - Gestión de sensores CARLA para Safe RL


DISEÑO DEL LIDAR (v2 — multi-canal con detección de bordes de carretera):
    CARLA SemanticLIDAR con channels=3, rotation_frequency=20 Hz
    → upper_fov=+5°, lower_fov=-15°  → cubre altura 0.2 - 1.7 m (sensor a 1.0 m)
    → A 3 m lateral, el canal inferior impacta a ≈0.2 m: detecta guardarraíles (~0.8 m)
    → points_per_second = num_rays * channels * rotation_frequency = 240 * 3 * 20 = 14400
    → height_filter = 1.5 m  → mantiene puntos entre -0.5 m y 2.5 m (sensor frame)
    → Proyectamos al plano horizontal con np.minimum por bin → scan 2D de 240 rayos
 
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
    ROAD_EDGE_TAGS    = {8 Acera, 22 Terreno}  ← NUEVO: marcan límite físico de carretera
    GROUND_TAGS       = {6 RoadLine, 7 Road, 8 Acera, 14 Suelo, 16 Via, 22 Terreno}
    EXCLUDED          = {0 Unlabeled, 3 Other, 13 Sky, 21 Water}
    EGO               = filtrado por object_idx == ego_vehicle.id (exacto, sin heurística)
    El scan `combined` = min(dynamic, static, road_edge): el agente percibe obstáculos
    móviles, obstáculos estáticos Y bordes físicos de carretera en un único canal.
 
BLUEPRINTS:
    ray_cast_semantic NO tiene: atmosphere_attenuation_rate, noise_stddev, dropoff_*
    Sí comparte con ray_cast: channels, range, points_per_second,
                               rotation_frequency, upper_fov, lower_fov
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