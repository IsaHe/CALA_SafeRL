"""
adaptive_horizon_shield.py - Adaptive Horizon Safety Shield para CARLA

La verificación adaptativa permite máxima libertad al agente en zonas seguras
y máxima protección en zonas de riesgo (principio de mínima interferencia).
"""

import gymnasium as gym
import numpy as np
import carla
import math
from typing import Tuple, Dict, Optional

from src.Adaptative_Shield.BicycleModel import BicycleModel


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
        lateral_threshold_base: float = 0.65,
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

    # GYMNASIUM API

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

    # ANÁLISIS DE RIESGO

    def _analyze_semantic(self, obs: np.ndarray, info: Dict) -> Dict:
        """
        Extrae las métricas relevantes para el shield.
 
        Prioridad: campos semánticos del info dict (si vienen del SemanticLidarSensor)
        Fallback:  calcular desde obs[:num_lidar_rays]
        """
        n = self.num_lidar_rays
        if "min_front_dynamic" in info:
            return {
                "min_front_combined":    info["min_front_combined"],
                "min_front_dynamic":     info["min_front_dynamic"],
                "min_front_static":      info["min_front_static"],
                "min_r_side_combined":   info["min_r_side_combined"],
                "min_r_side_static":     info.get("min_r_side_static",    info["min_r_side_combined"]),
                "min_r_side_road_edge":  info.get("min_r_side_road_edge", 1.0),
                "min_l_side_combined":   info["min_l_side_combined"],
                "min_l_side_static":     info.get("min_l_side_static",    info["min_l_side_combined"]),
                "min_l_side_road_edge":  info.get("min_l_side_road_edge", 1.0),
                "nearest_vehicle_m":     info.get("nearest_vehicle_m",    999.0),
                "nearest_pedestrian_m":  info.get("nearest_pedestrian_m", 999.0),
                "nearest_static_m":      info.get("nearest_static_m",     999.0),
                "nearest_road_edge_m":   info.get("nearest_road_edge_m",  999.0),
                "min_dist_for_risk":     info["min_front_dynamic"],
                "has_semantics":         True,
            }
        
        scan  = obs[:n]
        front = np.concatenate((scan[n-15:], scan[:15]))
        r_s   = scan[40:80]
        l_s   = scan[160:200]
        mf    = float(front.min())
        return {
            "min_front_combined":    mf,
            "min_front_dynamic":     mf,
            "min_front_static":      mf,
            "min_r_side_combined":   float(r_s.min()),
            "min_r_side_static":     float(r_s.min()),
            "min_r_side_road_edge":  float(r_s.min()),
            "min_l_side_combined":   float(l_s.min()),
            "min_l_side_static":     float(l_s.min()),
            "min_l_side_road_edge":  float(l_s.min()),
            "nearest_vehicle_m":     float(scan.min()) * 50.0,
            "nearest_pedestrian_m":  999.0,
            "nearest_static_m":      float(scan.min()) * 50.0,
            "nearest_road_edge_m":   999.0,
            "min_dist_for_risk":     float(scan.min()),
            "has_semantics":         False,
        }

    def _is_lane_change_context(self) -> bool:
        """Detecta si el vehículo está en contexto de cambio de carril permitido."""
        return bool(self.last_info.get("lane_change_permitted", False))

    def _get_risk_level_semantic(self, analysis: Dict) -> Tuple[str, float]:
        """
        Risk level combinado: riesgo FRONTAL dinámico + riesgo LATERAL de posición.

        Si el cambio de carril está permitido por las marcas viales, el riesgo
        lateral se limita a "warning" máximo para no bloquear la maniobra.
        """
        frontal_distance = analysis["min_dist_for_risk"]

        # Riesgo frontal
        if frontal_distance > self.HORIZON_CONFIG["safe"]["min_dist_threshold"]:
            frontal_level = "safe"
        elif frontal_distance > self.HORIZON_CONFIG["warning"]["min_dist_threshold"]:
            frontal_level = "warning"
        else:
            frontal_level = "critical"

        # Riesgo lateral: absoluto de lateral_offset_norm del info dict del último paso
        lat_norm = abs(self.last_info.get("lateral_offset_norm", 0.0))
        if lat_norm > 0.85:
            lateral_level = "critical"
        elif lat_norm > 0.70:
            lateral_level = "warning"
        else:
            lateral_level = "safe"

        # Si el cambio de carril es legítimo (marcas viales lo permiten),
        # no escalar riesgo lateral a "critical" — eso bloquearía la maniobra.
        if self._is_lane_change_context() and lateral_level == "critical":
            lateral_level = "warning"

        # Nivel final = el más restrictivo de los dos
        level_rank = {"safe": 0, "warning": 1, "critical": 2}
        if level_rank[frontal_level] >= level_rank[lateral_level]:
            final_level = frontal_level
        else:
            final_level = lateral_level

        return final_level, frontal_distance
    
    def _categorize_intervention(self, a: Dict):
        """Incrementa el contador de intervención por categoría semántica."""
        if a["nearest_pedestrian_m"] < self.PED_EMERGENCY_M:
            self.stats["interventions_pedestrian"] += 1
        elif a["min_front_dynamic"] < a["min_front_static"]:
            self.stats["interventions_dynamic"] += 1
        else:
            self.stats["interventions_static"] += 1

    # PREDICCIÓN Y VERIFICACIÓN DE TRAYECTORIA

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
 
        Check Waypoint API:
          Verifica que cada posición predicha esté dentro del carril.
        """
        multiplier = self.HORIZON_CONFIG[risk_level]["threshold_multiplier"]
        front_thr = self.front_threshold_base / multiplier
        side_thr = self.side_threshold_base / multiplier
        # Floor de 0.45 para evitar que en modo critical (mult=2.0) el
        # umbral lateral sea tan estricto que rechace todos los candidatos
        # de corrección (cuya trayectoria aún pasa cerca del borde).
        lat_thr = max(self.lateral_threshold_base / multiplier, 0.45)

        # Durante cambio de carril permitido, relajar umbral lateral:
        # el vehículo cruza la zona inter-carril donde lat_norm ≈ 1.0
        # respecto al carril más cercano. Permitimos hasta 1.2 para no
        # bloquear la transición, manteniendo el check frontal/peatón intacto.
        if self._is_lane_change_context():
            lat_thr = max(lat_thr, 1.2)

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
        Si la amenaza es PURAMENTE LATERAL (borde de carril sin obstáculo frontal):
          → corrección de vuelta al centro con gas suave como primer candidato
        """
        # ── Emergencia peatón: no buscar, frenar ya ───────────────────
        if analysis["nearest_pedestrian_m"] < self.PED_EMERGENCY_M:
            return np.array([0.0, -1.0], dtype=np.float32)

        lat_norm = self.last_info.get("lateral_offset_norm", 0.0)
        speed_ms = self.last_info.get("speed_ms", 0.0)
        lane_change_ok = self._is_lane_change_context()

        # Durante cambio de carril permitido: NO corregir de vuelta al carril
        # original — eso bloquea la maniobra. Usar steering 0 (mantener rumbo).
        if lane_change_ok and abs(lat_norm) > 0.5:
            lane_correction = 0.0
        else:
            lane_correction = float(np.clip(-lat_norm * 1.5, -1.0, 1.0))

        # Umbral TTC adaptativo a la velocidad
        ttc_vehicle_thr = max(5.0, speed_ms * 1.5)  # metros

        is_dynamic_threat = analysis["nearest_vehicle_m"] < ttc_vehicle_thr
        is_static_threat  = (analysis["min_r_side_static"] < self.side_threshold_base or
                             analysis["min_l_side_static"]  < self.side_threshold_base)
        is_lateral_only   = (abs(lat_norm) > 0.65
                             and not is_dynamic_threat
                             and not is_static_threat
                             and not lane_change_ok)

        if is_dynamic_threat:
            # Freno primero, luego corrección
            candidates = [
                np.array([0.0, -0.6]),   # Freno moderado, recto
                np.array([0.0, -1.0]),   # Freno total
                np.array([lane_correction, -0.5]),   # Corrección + freno
                np.array([0.5, -0.5]),   # Derecha + frenar
                np.array([-0.5, -0.5]),   # Izquierda + frenar
                np.array([lane_correction, 0.1]),   # Corrección + gas suave (mantener control a baja vel.)
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
                np.array([lane_correction, 0.1]),   # Corrección + gas suave
            ]
        elif is_lateral_only:
            # Primera prioridad: volver al centro del carril manteniendo marcha
            # (frenar haría perder control de dirección a baja velocidad)
            candidates = [
                np.array([lane_correction, 0.15]),          # Corrección + gas suave
                np.array([lane_correction, 0.0]),            # Corrección, mantener
                np.array([lane_correction * 0.8, -0.2]),     # Corrección + freno leve
                np.array([lane_correction, -0.4]),           # Corrección + freno
                np.array([0.0, -0.3]),                       # Recto + freno suave
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

        # Fallback: freno con corrección lateral para no agravar deriva.
        # Durante cambio de carril permitido, mantener rumbo actual (no corregir).
        if lane_change_ok and abs(lat_norm) > 0.5:
            fallback_steer = 0.0
        else:
            fallback_steer = float(np.clip(-lat_norm * 1.2, -0.8, 0.8))
        return np.array([fallback_steer, -0.4], dtype=np.float32)

    # ACCESO A OBJETOS CARLA

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

    # ESTADÍSTICAS

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
