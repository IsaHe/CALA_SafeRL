import math
from typing import Tuple, List


class BicycleModel:
    """
    Modelo cinemático de bicicleta para predicción de trayectoria.

    Parámetros calibrados para el Tesla Model 3 de CARLA:
      wheelbase    = 2.87 m (distancia entre ejes)
      max_steer    = 0.6 rad (~34°, máximo físico del vehículo)
      dt           = 0.05 s (sincronizado con fixed_delta_seconds = 20 Hz)

    La predicción es físicamente correcta para velocidades y radios
    de giro típicos de entorno urbano/autopista.

    Ecuaciones del modelo de bicicleta:
      Δθ = (v / L) * tan(δ) * dt
      Δx = v * cos(θ) * dt
      Δy = v * sin(θ) * dt
    """

    def __init__(
        self,
        wheelbase: float = 2.87,
        max_steer_rad: float = 0.60,
        dt: float = 0.05,
        max_accel_ms2: float = 3.0,
        max_decel_ms2: float = 7.0,
    ):
        self.L = wheelbase
        self.max_steer_rad = max_steer_rad
        self.dt = dt
        self.max_accel = max_accel_ms2
        self.max_decel = max_decel_ms2

    def predict_trajectory(
        self,
        x: float,
        y: float,
        yaw_rad: float,
        speed_ms: float,
        steering_norm: float,
        tb_norm: float,  # throttle_brake normalizado
        horizon: int,
    ) -> List[Tuple[float, float, float]]:
        """
        Predice una trayectoria de `horizon` pasos.

        Args:
            x, y          : Posición inicial (metros, frame global CARLA)
            yaw_rad       : Heading inicial (radianes)
            speed_ms      : Velocidad inicial (m/s)
            steering_norm : Acción de steering [-1, 1]
            tb_norm       : Acción throttle_brake [-1, 1]
            horizon       : Número de pasos a predecir

        Returns:
            Lista de (x, y, yaw_rad) para cada paso incluyendo el inicial.
        """
        trajectory = [(x, y, yaw_rad)]

        steer_rad = steering_norm * self.max_steer_rad
        if tb_norm >= 0.0:
            accel = tb_norm * self.max_accel
        else:
            accel = tb_norm * self.max_decel  # negativo = frenado

        cx, cy, cyaw = x, y, yaw_rad
        cspeed = speed_ms

        for _ in range(horizon):
            # Actualizar velocidad (clampear a 0)
            cspeed = max(0.0, cspeed + accel * self.dt)

            if cspeed < 0.01:
                # Vehículo parado: posición no cambia
                trajectory.append((cx, cy, cyaw))
                continue

            if abs(steer_rad) < 1e-4:
                # Trayectoria recta
                cx += cspeed * math.cos(cyaw) * self.dt
                cy += cspeed * math.sin(cyaw) * self.dt
            else:
                # Radio de giro: R = L / tan(δ)
                R = self.L / math.tan(abs(steer_rad))
                R = math.copysign(R, steer_rad)  # signo según dirección

                # Cambio de heading
                d_yaw = (cspeed / abs(R)) * self.dt
                if steer_rad < 0:
                    d_yaw = -d_yaw

                # Actualizar posición con arco de círculo
                cx += (
                    abs(R)
                    * (math.sin(cyaw + d_yaw) - math.sin(cyaw))
                    * math.copysign(1, R)
                )
                cy += (
                    abs(R)
                    * (math.cos(cyaw) - math.cos(cyaw + d_yaw))
                    * math.copysign(1, R)
                )
                cyaw += d_yaw

            trajectory.append((cx, cy, cyaw))

        return trajectory
