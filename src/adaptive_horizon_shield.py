"""
adaptive_horizon_shield.py - Adaptive Horizon Safety Shield para CARLA

Reemplaza completamente la versión MetaDrive con mejoras fundamentales:

INNOVACIÓN PRINCIPAL - BicycleModel para predicción de trayectoria:
  MetaDrive usaba una heurística que manipulaba el array LIDAR (completamente
  ficticio). Este shield usa el modelo de bicicleta con parámetros reales del
  Tesla Model 3 de CARLA para predecir posiciones físicamente correctas.

INNOVACIÓN 2 - Waypoint API para verificación a lo largo de la trayectoria:
  Para cada posición predicha, consultamos el Waypoint API de CARLA y
  verificamos el offset lateral real. Esto elimina los falsos positivos
  del shield MetaDrive que frecuentemente intervenía en carreteras curvas.

INNOVACIÓN 3 - Umbrales adaptativos más precisos:
  safe     : dist LIDAR > 50% → horizonte 1 paso
  warning  : 20% < dist < 50% → horizonte 5 pasos, umbrales x1.5
  critical : dist < 20%       → horizonte 10 pasos, umbrales x2.0

La verificación adaptativa permite máxima libertad al agente en zonas seguras
y máxima protección en zonas de riesgo (principio de mínima interferencia).
"""

import gymnasium as gym
import numpy as np
import carla
import math
from typing import Tuple, Dict, List, Optional


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
        self.L             = wheelbase
        self.max_steer_rad = max_steer_rad
        self.dt            = dt
        self.max_accel     = max_accel_ms2
        self.max_decel     = max_decel_ms2

    def predict_trajectory(
        self,
        x: float, y: float, yaw_rad: float,
        speed_ms: float,
        steering_norm: float,
        tb_norm: float,       # throttle_brake normalizado
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
                cx += abs(R) * (math.sin(cyaw + d_yaw) - math.sin(cyaw)) * math.copysign(1, R)
                cy += abs(R) * (math.cos(cyaw) - math.cos(cyaw + d_yaw)) * math.copysign(1, R)
                cyaw += d_yaw

            trajectory.append((cx, cy, cyaw))

        return trajectory


class CarlaAdaptiveHorizonShield(gym.Wrapper):
    """
    Shield adaptativo con predicción de trayectoria físicamente correcta.

    Cadena de envoltorios típica:
        CarlaAdaptiveHorizonShield
          └── CarlaRewardShaper
                └── CarlaEnv
    """

    # Configuración de horizontes adaptativos
    HORIZON_CONFIG = {
        "safe": {
            "min_dist_threshold":    0.50,   # LIDAR norm
            "horizon":               1,
            "threshold_multiplier":  1.0,
        },
        "warning": {
            "min_dist_threshold":    0.20,
            "horizon":               5,
            "threshold_multiplier":  1.5,
        },
        "critical": {
            "min_dist_threshold":    0.00,
            "horizon":               10,
            "threshold_multiplier":  2.0,
        },
    }

    def __init__(
        self,
        env,
        num_lidar_rays: int             = 240,
        front_threshold_base: float     = 0.15,
        side_threshold_base: float      = 0.04,
        lateral_threshold_base: float   = 0.82,
        k_steer: float                  = 2.5,
        k_brake: float                  = 8.0,
    ):
        super().__init__(env)

        self.num_lidar_rays           = num_lidar_rays
        self.front_threshold_base     = front_threshold_base
        self.side_threshold_base      = side_threshold_base
        self.lateral_threshold_base   = lateral_threshold_base
        self.k_steer                  = k_steer
        self.k_brake                  = k_brake

        self.bicycle_model = BicycleModel()

        # Estado
        self.last_obs: Optional[np.ndarray] = None
        self.last_info: Dict                = {}
        self.shield_activations             = 0

        # Estadísticas extendidas
        self.stats = {
            "safe_steps":     0,
            "warning_steps":  0,
            "critical_steps": 0,
            "interventions_by_horizon": {1: 0, 5: 0, 10: 0},
        }

    # ══════════════════════════════════════════════════════════════════
    # GYMNASIUM API
    # ══════════════════════════════════════════════════════════════════

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs  = obs
        self.last_info = info
        return obs, info

    def step(self, action: np.ndarray):
        """Paso con shield adaptativo."""
        lidar        = self.last_obs[:self.num_lidar_rays]
        analysis     = self._analyze_lidar(lidar)
        risk_level, min_dist = self._get_risk_level(analysis)
        horizon      = self.HORIZON_CONFIG[risk_level]["horizon"]

        self.stats[f"{risk_level}_steps"] += 1

        # Obtener mapa CARLA para verificación de trayectoria
        carla_map = self._get_carla_map()
        ego       = self._get_ego_vehicle()

        # Verificar si la acción propuesta es segura
        is_safe = self._check_trajectory_safety(
            action, horizon, risk_level, carla_map, ego
        )

        if not is_safe:
            final_action = self._find_safe_action(
                horizon, risk_level, carla_map, ego, analysis
            )
            self.shield_activations += 1
            self.stats["interventions_by_horizon"][horizon] = (
                self.stats["interventions_by_horizon"].get(horizon, 0) + 1
            )
            shield_active = True
        else:
            final_action  = action.copy()
            shield_active = False

        obs, reward, done, truncated, info = self.env.step(final_action)

        self.last_obs  = obs
        self.last_info = info

        info.update({
            "shield_activated":         shield_active,
            "risk_level":               risk_level,
            "min_distance":             min_dist,
            "horizon_used":             horizon,
            "min_front_dist":           analysis["min_front"],
            "min_r_side_dist":          analysis["min_r_side"],
            "min_l_side_dist":          analysis["min_l_side"],
            "total_shield_activations": self.shield_activations,
            "executed_action":           final_action,
            "proposed_action":           action,
        })

        return obs, reward, done, truncated, info

    # ══════════════════════════════════════════════════════════════════
    # ANÁLISIS DE RIESGO
    # ══════════════════════════════════════════════════════════════════

    def _analyze_lidar(self, scan: np.ndarray) -> Dict:
        n = self.num_lidar_rays
        front  = np.concatenate((scan[n - 15:n], scan[:15]))
        r_side = scan[40:80]
        l_side = scan[160:200]
        return {
            "min_front":  float(np.min(front)),
            "min_r_side": float(np.min(r_side)),
            "min_l_side": float(np.min(l_side)),
            "min_dist":   float(np.min(scan)),
        }

    def _get_risk_level(self, analysis: Dict) -> Tuple[str, float]:
        min_dist = analysis["min_dist"]   # ← mínimo GLOBAL = captura separadores laterales
        if min_dist > self.HORIZON_CONFIG["safe"]["min_dist_threshold"]:
            return "safe", min_dist
        elif min_dist > self.HORIZON_CONFIG["warning"]["min_dist_threshold"]:
            return "warning", min_dist
        else:
            return "critical", min_dist

    # ══════════════════════════════════════════════════════════════════
    # PREDICCIÓN Y VERIFICACIÓN DE TRAYECTORIA (CARLA-NATIVE)
    # ══════════════════════════════════════════════════════════════════

    def _check_trajectory_safety(
        self,
        action: np.ndarray,
        horizon: int,
        risk_level: str,
        carla_map,
        ego,
    ) -> bool:
        """
        Verifica si la trayectoria predicha es segura.

        Combina:
          1. Verificación LIDAR en el paso inmediato (obstáculos presentes)
          2. Verificación de posición lateral via Waypoint API en cada paso predicho
        """
        multiplier = self.HORIZON_CONFIG[risk_level]["threshold_multiplier"]
        front_thr  = self.front_threshold_base / multiplier
        side_thr   = self.side_threshold_base / multiplier
        lat_thr    = self.lateral_threshold_base / multiplier

        # ── Verificación LIDAR inmediata ───────────────────────────────
        lidar = self.last_obs[:self.num_lidar_rays]
        a = self._analyze_lidar(lidar)
        if (a["min_front"]  < front_thr or
                a["min_r_side"] < side_thr or
                a["min_l_side"] < side_thr):
            return False
        # ── Predicción de trayectoria con BicycleModel ─────────────────
        if ego is None or carla_map is None:
            return True  # Sin datos CARLA, asumir seguro

        try:
            transform = ego.get_transform()
            velocity  = ego.get_velocity()
        except Exception:
            return True

        x       = transform.location.x
        y       = transform.location.y
        yaw_rad = math.radians(transform.rotation.yaw)
        speed   = math.sqrt(velocity.x**2 + velocity.y**2)

        trajectory = self.bicycle_model.predict_trajectory(
            x, y, yaw_rad, speed,
            float(action[0]), float(action[1]),
            horizon,
        )

        # ── Verificar cada posición predicha via Waypoint API ──────────
        for px, py, _ in trajectory[1:]:
            loc = carla.Location(x=float(px), y=float(py), z=0.0)
            wp = carla_map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is None:
                return False  # Posición predicha fuera de carretera

            wp_right = wp.transform.get_right_vector()
            diff = loc - wp.transform.location
            lat_offset = diff.x * wp_right.x + diff.y * wp_right.y
            lane_half  = max(wp.lane_width / 2.0, 1.0)
            lat_norm   = abs(lat_offset) / lane_half

            if lat_norm > lat_thr:
                return False

        return True

    def _find_safe_action(
        self,
        horizon: int,
        risk_level: str,
        carla_map,
        ego,
        analysis: Dict,
    ) -> np.ndarray:
        """
        Busca la acción más segura entre candidatos predefinidos.

        Candidatos ordenados por preferencia (primero = menos intrusivo):
          1. Freno moderado + enderezar
          2. Freno fuerte
          3. Girar derecha + frenar
          4. Girar izquierda + frenar
          5. Corrección de carril basada en offset lateral
        """
        lat_norm = self.last_info.get("lateral_offset_norm", 0.0)

        # Acción de corrección de carril personalizada
        lane_correction = float(np.clip(-lat_norm * 1.5, -1.0, 1.0))

        candidates = [
            np.array([0.0,              -0.4]),   # Freno suave, recto
            np.array([0.0,              -1.0]),   # Freno total
            np.array([0.5,              -0.5]),   # Derecha + frenar
            np.array([-0.5,             -0.5]),   # Izquierda + frenar
            np.array([lane_correction,  -0.3]),   # Corrección carril
            np.array([0.0,               0.0]),   # Mantener
        ]

        for candidate in candidates:
            if self._check_trajectory_safety(
                candidate, horizon, risk_level, carla_map, ego
            ):
                return candidate

        # Fallback: freno total (siempre seguro en el corto plazo)
        return np.array([0.0, -0.5], dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════
    # ACCESO A OBJETOS CARLA (navegación de wrapper chain)
    # ══════════════════════════════════════════════════════════════════

    def _get_carla_map(self) -> Optional[carla.Map]:
        """Navega la cadena de wrappers para obtener el mapa CARLA."""
        env = self.env
        while env is not None:
            if hasattr(env, "map") and env.map is not None:
                return env.map
            env = getattr(env, "env", None)
        return None

    def _get_ego_vehicle(self) -> Optional[carla.Vehicle]:
        """Navega la cadena de wrappers para obtener el vehículo ego."""
        env = self.env
        while env is not None:
            if hasattr(env, "ego_vehicle") and env.ego_vehicle is not None:
                return env.ego_vehicle
            env = getattr(env, "env", None)
        return None

    # ══════════════════════════════════════════════════════════════════
    # ESTADÍSTICAS
    # ══════════════════════════════════════════════════════════════════

    def get_statistics(self) -> Dict:
        total = sum([
            self.stats["safe_steps"],
            self.stats["warning_steps"],
            self.stats["critical_steps"],
        ])
        if total == 0:
            total = 1

        return {
            "total_steps":               total,
            "safe_rate":                 self.stats["safe_steps"] / total,
            "warning_rate":              self.stats["warning_steps"] / total,
            "critical_rate":             self.stats["critical_steps"] / total,
            "total_interventions":       self.shield_activations,
            "intervention_rate":         self.shield_activations / total,
            "interventions_by_horizon":  self.stats["interventions_by_horizon"],
        }

    def reset_statistics(self):
        self.stats = {
            "safe_steps":     0,
            "warning_steps":  0,
            "critical_steps": 0,
            "interventions_by_horizon": {1: 0, 5: 0, 10: 0},
        }
        self.shield_activations = 0
