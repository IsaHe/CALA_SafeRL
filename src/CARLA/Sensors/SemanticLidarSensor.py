import queue
from typing import Dict

import carla
import numpy as np

from src.CARLA.Sensors.SemanticLidarProcessor import SemanticLidarProcessor
from src.CARLA.Sensors.SemanticScanResult import SemanticScanResult


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
        height_offset:      float = 1.0,
        height_filter:      float = 1.5,
        lidar_channels:     int   = 3,
        upper_fov:          float = 5.0,
        lower_fov:          float = -15.0,
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
        bp.set_attribute("channels",           str(lidar_channels))
        bp.set_attribute("range",              str(lidar_range))
        bp.set_attribute("rotation_frequency", str(rotation_frequency))
        bp.set_attribute("points_per_second", str(int(num_rays * lidar_channels * rotation_frequency)))
        bp.set_attribute("upper_fov",          str(upper_fov))
        bp.set_attribute("lower_fov",          str(lower_fov))
 
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