import logging
import queue
from typing import Dict, Optional

import carla
import numpy as np

from src.CARLA.Sensors.SemanticLidarProcessor import SemanticLidarProcessor
from src.CARLA.Sensors.SemanticScanResult import SemanticScanResult

logger = logging.getLogger(__name__)


class SemanticLidarSensor:
    """
    Wrapper sobre sensor.lidar.ray_cast_semantic de CARLA.

    Acepta un `transform` arbitrario para soportar múltiples LIDARs
    montados a alturas distintas (p. ej. uno alto a z=1.0 y uno bajo a
    z=0.5 en el parachoques delantero para detectar guardarraíles bajos).

    El procesador recibe `z_mount` para construir un filtro de altura
    asimétrico que rechaza impactos contra el suelo.
    """

    def __init__(
        self,
        world: carla.World,
        vehicle: carla.Vehicle,
        num_rays: int = 240,
        lidar_range: float = 50.0,
        rotation_frequency: float = 20.0,
        height_offset: float = 1.0,
        height_filter: float = 1.5,
        lidar_channels: int = 3,
        upper_fov: float = 5.0,
        lower_fov: float = -15.0,
        transform: Optional[carla.Transform] = None,
    ):
        self._num_rays = num_rays

        if transform is None:
            transform = carla.Transform(carla.Location(x=0.0, z=height_offset))
        z_mount = float(transform.location.z)

        self.processor = SemanticLidarProcessor(
            num_rays=num_rays,
            lidar_range=lidar_range,
            height_filter=height_filter,
            ego_id=vehicle.id,
            z_mount=z_mount,
        )
        self._queue: queue.Queue = queue.Queue()

        bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        bp.set_attribute("channels", str(lidar_channels))
        bp.set_attribute("range", str(lidar_range))
        bp.set_attribute("rotation_frequency", str(rotation_frequency))
        bp.set_attribute(
            "points_per_second",
            str(int(num_rays * lidar_channels * rotation_frequency)),
        )
        bp.set_attribute("upper_fov", str(upper_fov))
        bp.set_attribute("lower_fov", str(lower_fov))

        # A2: log de los atributos efectivos del blueprint para detectar
        # cualquier divergencia entre lo configurado y lo aceptado por CARLA.
        bp_attrs = {a.id: bp.get_attribute(a.id).as_string() for a in bp}
        logger.info(f"[LIDAR_DBG] bp.attrs={bp_attrs}")

        self.sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        self.sensor.listen(lambda data: self._queue.put(data))

        # A1: log de instanciación (id del actor sensor, parent, transform).
        try:
            parent_id = self.sensor.parent.id if self.sensor.parent else None
        except Exception:
            parent_id = None
        logger.info(
            f"[LIDAR_DBG] spawn id={self.sensor.id} parent={parent_id} "
            f"local_transform={transform} z_mount={z_mount:.2f}"
        )

        self._last = SemanticScanResult()
        self._last_was_fresh = False
        self._stale_reads = 0
        self._fresh_reads = 0
        self._last_frame = -1
        # Métricas extra para auditar: nº de puntos del último frame
        # procesado y nº de puntos del ego filtrados antes de bin-ear.
        self._last_pts_total = 0
        self._last_pts_after_ego_filter = 0

    def update_ego_id(self, new_id: int):
        """Llamar tras respawn del ego vehicle."""
        self.processor.set_ego_id(new_id)

    def get_result(
        self,
        expected_frame: Optional[int] = None,
        timeout: float = 1.0,
    ) -> SemanticScanResult:
        """
        Recupera el último SemanticScanResult.

        Modo legacy (expected_frame=None):
            Drena la cola con get_nowait() y procesa el último frame disponible.
            Si la cola está vacía, devuelve el resultado anterior y marca stale.

        Modo frame-match (expected_frame=<int>):
            Bloquea con queue.get(timeout=...) hasta recibir un frame cuyo
            `frame` coincida con `expected_frame`. Frames anteriores se
            descartan. Si tras `timeout` no llega el frame esperado, devuelve
            el último resultado y suma stale_reads. Esto reproduce el patrón
            canónico de CARLA (synchronous_mode.py) y garantiza que los datos
            usados por la política y el shield corresponden al tick actual.
        """
        if expected_frame is None:
            return self._legacy_get()
        return self._frame_matched_get(expected_frame, timeout)

    def _legacy_get(self) -> SemanticScanResult:
        latest = None
        while not self._queue.empty():
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self._record_pts(latest)
            self._last = self.processor.process(latest)
            self._last_was_fresh = True
            self._fresh_reads += 1
            self._last_frame = int(getattr(latest, "frame", -1))
        else:
            self._last_was_fresh = False
            self._stale_reads += 1
        return self._last

    def _record_pts(self, measurement) -> None:
        """Cuenta puntos del frame para diagnóstico (C1 del plan)."""
        try:
            from src.CARLA.Sensors.SemanticLidarProcessor import _SEMANTIC_DTYPE
            n = len(np.frombuffer(measurement.raw_data, dtype=_SEMANTIC_DTYPE))
        except Exception:
            n = -1
        self._last_pts_total = int(n)

    def _frame_matched_get(
        self, expected_frame: int, timeout: float
    ) -> SemanticScanResult:
        """
        Bloquea hasta que llega un dato con frame == expected_frame.
        Descarta frames anteriores sin procesarlos.
        """
        while True:
            try:
                data = self._queue.get(timeout=timeout)
            except queue.Empty:
                # CARLA no entregó el frame esperado dentro del timeout.
                # Mantener el último _last como fallback y marcar stale.
                self._last_was_fresh = False
                self._stale_reads += 1
                return self._last

            data_frame = int(getattr(data, "frame", -1))
            if data_frame == expected_frame:
                self._record_pts(data)
                self._last = self.processor.process(data)
                self._last_was_fresh = True
                self._fresh_reads += 1
                self._last_frame = data_frame
                return self._last
            # frame antiguo: descartar y seguir esperando.
            # No incrementamos stale_reads — fueron entregas tardías, no
            # fallos. El bucle continúa hasta agotar timeout o llegar al
            # frame correcto.
            continue

    def get_status(self) -> Dict[str, int]:
        return {
            "fresh": int(self._last_was_fresh),
            "stale_reads": int(self._stale_reads),
            "fresh_reads": int(self._fresh_reads),
            "last_frame": int(self._last_frame),
            "pts_per_frame": int(self._last_pts_total),
        }

    def get_scan(self) -> np.ndarray:
        """Backward compat: combined_scan."""
        return self.get_result().combined.copy()

    def destroy(self):
        if self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
