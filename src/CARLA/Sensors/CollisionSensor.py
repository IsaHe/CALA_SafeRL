import math
import weakref

import carla


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
