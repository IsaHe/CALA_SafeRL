"""
adaptive_horizon_shield.py - Adaptive Horizon Safety Shield para CARLA

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
        "safe": {"min_dist_threshold": 0.50, "horizon": 1, "threshold_multiplier": 1.0},
        "warning": {"min_dist_threshold": 0.20, "horizon": 5, "threshold_multiplier": 1.5},
        "critical": {"min_dist_threshold": 0.00, "horizon": 10, "threshold_multiplier": 2.0},
    }

    PED_EMERGENCY_M: float = 4.0

    def __init__(
        self,
        env,
        num_lidar_rays: int = 240,
        front_threshold_base: float = 0.15,
        side_threshold_base: float = 0.04,
        lateral_threshold_base: float = 0.82,
        k_steer: float = 2.5,
        k_brake: float = 8.0,
    ):
        super().__init__(env)

        self.num_lidar_rays = num_lidar_rays
        self.front_threshold_base = front_threshold_base
        self.side_threshold_base = side_threshold_base
        self.lateral_threshold_base = lateral_threshold_base
        self.k_steer = k_steer
        self.k_brake = k_brake

        self.bicycle_model = BicycleModel()

        # Estado
        self.last_obs: Optional[np.ndarray] = None
        self.last_info: Dict = {}
        self.shield_activations = 0

        # Estadísticas extendidas
        self.stats = {
            "safe_steps": 0,
            "warning_steps": 0,
            "critical_steps": 0,
            "interventions_dynamic":    0,
            "interventions_static":     0,
            "interventions_pedestrian": 0,
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
        sem_analysis = self._analyze_semantic(self.last_obs, self.last_info)
        risk_level, _ = self._get_risk_level_semantic(sem_analysis)
        horizon = self.HORIZON_CONFIG[risk_level]["horizon"]

        self.stats[f"{risk_level}_steps"] += 1

        # Obtener mapa CARLA para verificación de trayectoria
        carla_map = self._get_carla_map()
        ego = self._get_ego_vehicle()

        # Verificar si la acción propuesta es segura
        is_safe = self._check_trajectory_safety(
            action, horizon, risk_level, carla_map, ego, sem_analysis
        )

        if not is_safe:
            final_action = self._find_safe_action(
                horizon, risk_level, carla_map, ego, sem_analysis
            )
            self.shield_activations += 1
            self.stats["interventions_by_horizon"][horizon] = (
                self.stats["interventions_by_horizon"].get(horizon, 0) + 1
            )
            self._categorize_intervention(sem_analysis)
            shield_active = True
        else:
            final_action  = action.copy()
            shield_active = False

        obs, reward, done, truncated, info = self.env.step(final_action)

        self.last_obs  = obs
        self.last_info = info

        info.update({
            "shield_activated": shield_active,
            "risk_level": risk_level,
            "min_distance": sem_analysis["min_dist_for_risk"],
            "horizon_used": horizon,
            "min_front_dist": sem_analysis["min_front_combined"],
            "min_front_dynamic": sem_analysis["min_front_dynamic"],
            "min_front_static": sem_analysis["min_front_static"],
            "min_r_side_dist": sem_analysis["min_r_side_combined"],
            "min_l_side_dist": sem_analysis["min_l_side_combined"],
            "nearest_vehicle_m": sem_analysis["nearest_vehicle_m"],
            "nearest_pedestrian_m": sem_analysis["nearest_pedestrian_m"],
            "nearest_static_m": sem_analysis["nearest_static_m"],
            "total_shield_activations": self.shield_activations,
            "executed_action": final_action,
            "proposed_action": action,
        })

        return obs, reward, done, truncated, info

    # ══════════════════════════════════════════════════════════════════
    # ANÁLISIS DE RIESGO
    # ══════════════════════════════════════════════════════════════════

    def _analyze_semantic(self, obs: np.ndarray, info: Dict) -> Dict:
        """
        Extrae las métricas relevantes para el shield.
 
        Prioridad: campos semánticos del info dict (si vienen del SemanticLidarSensor)
        Fallback:  calcular desde obs[:num_lidar_rays] como antes (compat)
        """
        n = self.num_lidar_rays
        if "min_front_dynamic" in info:
            return {
                "min_front_combined":  info["min_front_combined"],
                "min_front_dynamic":   info["min_front_dynamic"],
                "min_front_static":    info["min_front_static"],
                "min_r_side_combined": info["min_r_side_combined"],
                "min_r_side_static":   info.get("min_r_side_static", info["min_r_side_combined"]),
                "min_l_side_combined": info["min_l_side_combined"],
                "min_l_side_static":   info.get("min_l_side_static", info["min_l_side_combined"]),
                "nearest_vehicle_m":   info.get("nearest_vehicle_m",    999.0),
                "nearest_pedestrian_m":info.get("nearest_pedestrian_m", 999.0),
                "nearest_static_m":    info.get("nearest_static_m",     999.0),
                "min_dist_for_risk":   info["min_front_dynamic"],
                "has_semantics":       True,
            }
        
        scan  = obs[:n]
        front = np.concatenate((scan[n-15:], scan[:15]))
        r_s   = scan[40:80]
        l_s   = scan[160:200]
        mf    = float(front.min())
        return {
            "min_front_combined":  mf,
            "min_front_dynamic":   mf,
            "min_front_static":    mf,
            "min_r_side_combined": float(r_s.min()),
            "min_r_side_static":   float(r_s.min()),
            "min_l_side_combined": float(l_s.min()),
            "min_l_side_static":   float(l_s.min()),
            "nearest_vehicle_m":   float(scan.min()) * 50.0,
            "nearest_pedestrian_m":999.0,
            "nearest_static_m":    float(scan.min()) * 50.0,
            "min_dist_for_risk":   float(scan.min()),
            "has_semantics":       False,
        }

    def _get_risk_level_semantic(self, analysis: Dict) -> Tuple[str, float]:
        """
        Risk level basado en la distancia frontal a obstáculos DINÁMICOS.
 
        Esto resuelve el problema anterior donde los quitamiedos laterales
        (tag 17, GuardRail) a 3-4m forzaban el modo 'warning' de forma
        permanente en autopistas, activando el horizonte 5 y causando
        frenos innecesarios.
 
        Los quitamiedos siguen siendo manejados por el Waypoint API en
        _check_trajectory_safety (verificación de offset lateral).
        """
        distance = analysis["min_dist_for_risk"]

        if distance > self.HORIZON_CONFIG["safe"]["min_dist_threshold"]:
            return "safe", distance
        elif distance > self.HORIZON_CONFIG["warning"]["min_dist_threshold"]:
            return "warning", distance
        else:
            return "critical", distance
    
    def _categorize_intervention(self, a: Dict):
        """Incrementa el contador de intervención por categoría semántica."""
        if a["nearest_pedestrian_m"] < self.PED_EMERGENCY_M:
            self.stats["interventions_pedestrian"] += 1
        elif a["min_front_dynamic"] < a["min_front_static"]:
            self.stats["interventions_dynamic"] += 1
        else:
            self.stats["interventions_static"] += 1

    # ══════════════════════════════════════════════════════════════════
    # PREDICCIÓN Y VERIFICACIÓN DE TRAYECTORIA
    # ══════════════════════════════════════════════════════════════════

    def _check_trajectory_safety(
        self,
        action: np.ndarray,
        horizon: int,
        risk_level: str,
        carla_map,
        ego,
        analysis: Dict,
    ) -> bool:
        """
        Verificación dual (semántica + Waypoint API).
 
        Checks LIDAR:
          1. front_combined_thr: cualquier obstáculo frontal (vehículo o muro)
          2. side_static_thr:    quitamiedos / muros laterales
          3. Emergencia peatón:  si hay peatón muy cerca → unsafe siempre
 
        Check Waypoint API (sin cambios vs versión anterior):
          Verifica que cada posición predicha esté dentro del carril.
        """
        multiplier = self.HORIZON_CONFIG[risk_level]["threshold_multiplier"]
        front_thr = self.front_threshold_base / multiplier
        side_thr = self.side_threshold_base / multiplier
        lat_thr = self.lateral_threshold_base / multiplier

        # ── Emergencia peatón (override inmediato) ────────────────────
        if analysis["nearest_pedestrian_m"] < self.PED_EMERGENCY_M:
            return False
        
        # ── Checks LIDAR semánticos ───────────────────────────────────
        # Frente: combined (muro o vehículo, ambos peligrosos en frontal)
        if analysis["min_front_combined"] < front_thr:
            return False
 
        # Lados: solo obstáculos estáticos definen el límite real de carril;
        # si el lado tiene un vehículo (dinámico) es un NPC que se cruza,
        # eso lo gestiona el Waypoint API (trajetoria saldrá del carril).
        if analysis["min_r_side_static"] < side_thr:
            return False
        if analysis["min_l_side_static"] < side_thr:
            return False

        # ── Predicción de trayectoria con BicycleModel ─────────────────
        if ego is None or carla_map is None:
            return True

        try:
            transform = ego.get_transform()
            velocity = ego.get_velocity()
        except Exception:
            return True

        x = transform.location.x
        y = transform.location.y
        yaw_rad = math.radians(transform.rotation.yaw)
        speed = math.sqrt(velocity.x**2 + velocity.y**2)

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
                return False

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

        Si la amenaza principal es DINÁMICA (vehículo cerca en frente):
          → candidatos con freno fuerte primero
        Si la amenaza es ESTÁTICA LATERAL (quitamiedos, muros de carril):
          → candidatos con corrección de carril primero, freno moderado
        Si hay PEATÓN muy cerca:
          → freno de emergencia total, sin búsqueda
        """
        # ── Emergencia peatón: no buscar, frenar ya ───────────────────
        if analysis["nearest_pedestrian_m"] < self.PED_EMERGENCY_M:
            return np.array([0.0, -1.0], dtype=np.float32)
        
        lat_norm = self.last_info.get("lateral_offset_norm", 0.0)
        # Acción de corrección de carril personalizada
        lane_correction = float(np.clip(-lat_norm * 1.5, -1.0, 1.0))
        speed_ms = self.last_info.get("speed_ms", 0.0)
        # Umbral TTC adaptativo a la velocidad
        ttc_vehicle_thr = max(5.0, speed_ms * 1.5)  # metros

        is_dynamic_threat = analysis["nearest_vehicle_m"] < ttc_vehicle_thr
        is_static_threat  = (analysis["min_r_side_static"] < self.side_threshold_base or
                             analysis["min_l_side_static"]  < self.side_threshold_base)

        if is_dynamic_threat:
            # Freno primero, luego corrección
            candidates = [
                np.array([0.0, -0.6]),   # Freno moderado, recto
                np.array([0.0, -1.0]),   # Freno total
                np.array([lane_correction, -0.5]),   # Corrección + freno
                np.array([0.5, -0.5]),   # Derecha + frenar
                np.array([-0.5, -0.5]),   # Izquierda + frenar
                np.array([0.0, 0.0]),   # Mantener
            ]
        elif is_static_threat:
            # Steering primero, freno ligero
            candidates = [
                np.array([lane_correction, -0.3]),   # Corrección carril prioritaria
                np.array([lane_correction, -0.5]),   # Corrección + freno moderado
                np.array([0.0, -0.4]),   # Freno suave, recto
                np.array([0.0, -0.8]),   # Freno fuerte
                np.array([0.5, -0.4]),   # Derecha + freno
                np.array([-0.5, -0.4]),   # Izquierda + freno
            ]
        else:
            # Candidatos balanceados (comportamiento anterior)
            candidates = [
                np.array([0.0, -0.4]),
                np.array([0.0, -1.0]),
                np.array([0.5, -0.5]),
                np.array([-0.5, -0.5]),
                np.array([lane_correction, -0.3]),
                np.array([0.0, 0.0]),
            ]

        for candidate in candidates:
            if self._check_trajectory_safety(
                candidate, horizon, risk_level, carla_map, ego, analysis
            ):
                return candidate

        # Fallback: freno total (siempre seguro en el corto plazo)
        return np.array([0.0, -0.5], dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════
    # ACCESO A OBJETOS CARLA
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
            "total_steps": total,
            "safe_rate": self.stats["safe_steps"] / total,
            "warning_rate": self.stats["warning_steps"] / total,
            "critical_rate": self.stats["critical_steps"] / total,
            "total_interventions": self.shield_activations,
            "intervention_rate": self.shield_activations / total,
            "interventions_by_horizon": self.stats["interventions_by_horizon"],
            "interventions_dynamic": self.stats["interventions_dynamic"],
            "interventions_static": self.stats["interventions_static"],
            "interventions_pedestrian": self.stats["interventions_pedestrian"],
        }

    def reset_statistics(self):
        self.stats = {
            "safe_steps":     0,
            "warning_steps":  0,
            "critical_steps": 0,
            "interventions_by_horizon": {1: 0, 5: 0, 10: 0},
            "interventions_dynamic": 0,
            "interventions_static": 0,
            "interventions_pedestrian": 0,
        }
        self.shield_activations = 0
