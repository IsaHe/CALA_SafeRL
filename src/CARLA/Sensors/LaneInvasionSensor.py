import weakref

import carla


class LaneInvasionSensor:
    """
    Sensor de invasión de carril NATIVO de CARLA.
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