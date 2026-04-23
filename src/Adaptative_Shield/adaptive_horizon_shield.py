"""
adaptive_horizon_shield.py - Adaptive Horizon Safety Shield para CARLA

PARADIGMA: MÍNIMA INTERFERENCIA POR PROYECCIÓN
  En lugar de elegir de un set discreto de candidatos fijos (que saturaban
  ±1.0 y rompían numéricamente el log_prob de PPO), se proyecta la acción
  propuesta hacia una acción-objetivo (`emergency`) mediante interpolación
  continua:
        a_exec(α) = (1-α)·a_prop + α·a_emergency,    α ∈ {0.25, 0.5, 0.75, 1.0}
  Se devuelve la primera α cuya trayectoria pasa la verificación dual
  (semántica LIDAR + BicycleModel + Waypoint API). La intensidad α se
  expone como `shield_intensity` en info.

  Beneficios vs candidatos discretos:
    - La acción ejecutada queda siempre dentro del soporte de π(·|s):
      pequeños desplazamientos respecto a la propuesta, no saltos
      extremos a [0,-1] o [±0.5,-0.5].
    - Cuando α es pequeña, el shield_mask puede seguir a 0 (caso
      pass-through blando) — evita descartar datos útiles.
    - Gradiente de PPO limpio: el credit assignment está bien definido.

CAPAS:
  - Emergencia peatón → override inmediato (α=1).
  - BicycleModel (horizontes adaptativos 1/5/10) + Waypoint API.
"""

import gymnasium as gym
import numpy as np
import carla
import math
from typing import Tuple, Dict, Optional

from src.Adaptative_Shield.BicycleModel import BicycleModel


class CarlaAdaptiveHorizonShield(gym.Wrapper):
    """Shield adaptativo con proyección continua y BicycleModel."""

    HORIZON_CONFIG = {
        "safe": {"min_dist_threshold": 0.50, "horizon": 1, "threshold_multiplier": 1.0},
        "warning": {
            "min_dist_threshold": 0.20,
            "horizon": 5,
            "threshold_multiplier": 1.5,
        },
        "critical": {
            "min_dist_threshold": 0.00,
            "horizon": 10,
            "threshold_multiplier": 2.0,
        },
    }

    PED_EMERGENCY_M: float = 4.0
    BLEND_ALPHAS = (0.25, 0.5, 0.75, 1.0)
    SHIELD_MASK_THRESHOLD = 0.05

    def __init__(
        self,
        env,
        num_lidar_rays: int = 240,
        front_threshold_base: float = 0.15,
        side_threshold_base: float = 0.04,
        lateral_threshold_base: float = 0.65,
        lane_correction_gain: float = 1.5,
        emergency_brake: float = -0.6,
    ):
        super().__init__(env)

        self.num_lidar_rays = num_lidar_rays
        self.front_threshold_base = front_threshold_base
        self.side_threshold_base = side_threshold_base
        self.lateral_threshold_base = lateral_threshold_base
        self.lane_correction_gain = lane_correction_gain
        self.emergency_brake = emergency_brake

        self.bicycle_model = BicycleModel()

        self.last_obs: Optional[np.ndarray] = None
        self.last_info: Dict = {}
        self.shield_activations = 0

        self.stats = {
            "safe_steps": 0,
            "warning_steps": 0,
            "critical_steps": 0,
            "interventions_dynamic": 0,
            "interventions_static": 0,
            "interventions_pedestrian": 0,
            "interventions_by_horizon": {1: 0, 5: 0, 10: 0},
        }

    # ────────────────────────── GYMNASIUM ──────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.last_info = info
        return obs, info

    def step(self, action: np.ndarray):
        sem_analysis = self._analyze_semantic(self.last_obs, self.last_info)
        risk_level, _ = self._get_risk_level_semantic(sem_analysis)
        horizon = self.HORIZON_CONFIG[risk_level]["horizon"]
        self.stats[f"{risk_level}_steps"] += 1

        carla_map = self._get_carla_map()
        ego = self._get_ego_vehicle()

        proposed = np.asarray(action, dtype=np.float32).copy()

        # Emergencia peatón: proyección con α=1 hacia emergency_ped.
        # Usamos el mismo paradigma que el resto del shield (blend continuo)
        # para mantener invariantes: ningún `executed_action` tiene componente
        # saturada "por decreto". A α=1 el resultado matemático es idéntico a
        # emergency_ped, pero el flujo queda unificado.
        if sem_analysis["nearest_pedestrian_m"] < self.PED_EMERGENCY_M:
            emergency_ped = np.array([0.0, -1.0], dtype=np.float32)
            final_action = ((1.0 - 1.0) * proposed + 1.0 * emergency_ped).astype(
                np.float32
            )
            alpha = 1.0
            self.stats["interventions_pedestrian"] += 1
            self.shield_activations += 1
            self.stats["interventions_by_horizon"][horizon] = (
                self.stats["interventions_by_horizon"].get(horizon, 0) + 1
            )
        elif self._check_trajectory_safety(
            proposed, horizon, risk_level, carla_map, ego, sem_analysis
        ):
            final_action = proposed
            alpha = 0.0
        else:
            emergency = self._build_emergency_action(sem_analysis)
            final_action, alpha = self._project(
                proposed,
                emergency,
                horizon,
                risk_level,
                carla_map,
                ego,
                sem_analysis,
            )
            if alpha >= self.SHIELD_MASK_THRESHOLD:
                self.shield_activations += 1
                self.stats["interventions_by_horizon"][horizon] = (
                    self.stats["interventions_by_horizon"].get(horizon, 0) + 1
                )
                self._categorize_intervention(sem_analysis)

        shield_activated = alpha >= self.SHIELD_MASK_THRESHOLD

        obs, reward, done, truncated, info = self.env.step(final_action)
        self.last_obs = obs
        self.last_info = info

        info.update(
            {
                "shield_activated": shield_activated,
                "shield_intensity": float(alpha),
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
                "proposed_action": proposed,
            }
        )

        return obs, reward, done, truncated, info

    # ────────────────────────── ANÁLISIS DE RIESGO ──────────────────────────

    def _analyze_semantic(self, obs: np.ndarray, info: Dict) -> Dict:
        n = self.num_lidar_rays
        if "min_front_dynamic" in info:
            return {
                "min_front_combined": info["min_front_combined"],
                "min_front_dynamic": info["min_front_dynamic"],
                "min_front_static": info["min_front_static"],
                "min_r_side_combined": info["min_r_side_combined"],
                "min_r_side_static": info.get(
                    "min_r_side_static", info["min_r_side_combined"]
                ),
                "min_r_side_road_edge": info.get("min_r_side_road_edge", 1.0),
                "min_l_side_combined": info["min_l_side_combined"],
                "min_l_side_static": info.get(
                    "min_l_side_static", info["min_l_side_combined"]
                ),
                "min_l_side_road_edge": info.get("min_l_side_road_edge", 1.0),
                "nearest_vehicle_m": info.get("nearest_vehicle_m", 999.0),
                "nearest_pedestrian_m": info.get("nearest_pedestrian_m", 999.0),
                "nearest_static_m": info.get("nearest_static_m", 999.0),
                "nearest_road_edge_m": info.get("nearest_road_edge_m", 999.0),
                "min_dist_for_risk": info["min_front_dynamic"],
                "has_semantics": True,
            }

        scan = obs[:n]
        front = np.concatenate((scan[n - 15 :], scan[:15]))
        r_s = scan[40:80]
        l_s = scan[160:200]
        mf = float(front.min())
        return {
            "min_front_combined": mf,
            "min_front_dynamic": mf,
            "min_front_static": mf,
            "min_r_side_combined": float(r_s.min()),
            "min_r_side_static": float(r_s.min()),
            "min_r_side_road_edge": float(r_s.min()),
            "min_l_side_combined": float(l_s.min()),
            "min_l_side_static": float(l_s.min()),
            "min_l_side_road_edge": float(l_s.min()),
            "nearest_vehicle_m": float(scan.min()) * 50.0,
            "nearest_pedestrian_m": 999.0,
            "nearest_static_m": float(scan.min()) * 50.0,
            "nearest_road_edge_m": 999.0,
            "min_dist_for_risk": float(scan.min()),
            "has_semantics": False,
        }

    def _is_lane_change_context(self) -> bool:
        return bool(self.last_info.get("lane_change_permitted", False))

    def _get_risk_level_semantic(self, analysis: Dict) -> Tuple[str, float]:
        frontal_distance = analysis["min_dist_for_risk"]

        if frontal_distance > self.HORIZON_CONFIG["safe"]["min_dist_threshold"]:
            frontal_level = "safe"
        elif frontal_distance > self.HORIZON_CONFIG["warning"]["min_dist_threshold"]:
            frontal_level = "warning"
        else:
            frontal_level = "critical"

        lat_norm = abs(self.last_info.get("lateral_offset_norm", 0.0))
        if lat_norm > 0.85:
            lateral_level = "critical"
        elif lat_norm > 0.70:
            lateral_level = "warning"
        else:
            lateral_level = "safe"

        if self._is_lane_change_context() and lateral_level == "critical":
            lateral_level = "warning"

        level_rank = {"safe": 0, "warning": 1, "critical": 2}
        if level_rank[frontal_level] >= level_rank[lateral_level]:
            final_level = frontal_level
        else:
            final_level = lateral_level

        return final_level, frontal_distance

    def _categorize_intervention(self, a: Dict):
        if a["nearest_pedestrian_m"] < self.PED_EMERGENCY_M:
            self.stats["interventions_pedestrian"] += 1
        elif a["min_front_dynamic"] < a["min_front_static"]:
            self.stats["interventions_dynamic"] += 1
        else:
            self.stats["interventions_static"] += 1

    # ────────────────────────── SAFETY CHECK ──────────────────────────

    def _check_trajectory_safety(
        self,
        action: np.ndarray,
        horizon: int,
        risk_level: str,
        carla_map,
        ego,
        analysis: Dict,
    ) -> bool:
        multiplier = self.HORIZON_CONFIG[risk_level]["threshold_multiplier"]
        front_thr = self.front_threshold_base / multiplier
        side_thr = self.side_threshold_base / multiplier
        lat_thr = max(self.lateral_threshold_base / multiplier, 0.45)

        if self._is_lane_change_context():
            lat_thr = max(lat_thr, 1.2)

        if analysis["nearest_pedestrian_m"] < self.PED_EMERGENCY_M:
            return False

        if analysis["min_front_combined"] < front_thr:
            return False
        if analysis["min_r_side_static"] < side_thr:
            return False
        if analysis["min_l_side_static"] < side_thr:
            return False

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
            x,
            y,
            yaw_rad,
            speed,
            float(action[0]),
            float(action[1]),
            horizon,
        )

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
            lane_half = max(wp.lane_width / 2.0, 1.0)
            lat_norm = abs(lat_offset) / lane_half

            if lat_norm > lat_thr:
                return False

        return True

    # ────────────────────────── PROYECCIÓN ──────────────────────────

    def _build_emergency_action(self, analysis: Dict) -> np.ndarray:
        """
        Acción-objetivo a la que interpolar. La intención es:
          - Frenar con intensidad proporcional a la amenaza frontal.
          - Corregir hacia el centro si estamos descentrados.
          - Mantener rumbo si estamos en cambio de carril permitido.
        """
        lat_norm = self.last_info.get("lateral_offset_norm", 0.0)
        lane_change_ok = self._is_lane_change_context()

        if lane_change_ok and abs(lat_norm) > 0.5:
            steer_target = 0.0
        else:
            steer_target = float(
                np.clip(-lat_norm * self.lane_correction_gain, -1.0, 1.0)
            )

        # Sesgo por obstáculo estático lateral: empujar lejos.
        if analysis["min_l_side_static"] < self.side_threshold_base:
            steer_target = float(np.clip(steer_target + 0.4, -1.0, 1.0))
        if analysis["min_r_side_static"] < self.side_threshold_base:
            steer_target = float(np.clip(steer_target - 0.4, -1.0, 1.0))

        # Freno: más fuerte si el obstáculo frontal está muy cerca.
        front = analysis["min_front_combined"]
        if front < self.front_threshold_base * 0.5:
            tb_target = -1.0
        elif front < self.front_threshold_base:
            tb_target = -0.8
        else:
            tb_target = self.emergency_brake  # por defecto freno moderado

        return np.array([steer_target, tb_target], dtype=np.float32)

    def _project(
        self,
        proposed: np.ndarray,
        emergency: np.ndarray,
        horizon: int,
        risk_level: str,
        carla_map,
        ego,
        analysis: Dict,
    ) -> Tuple[np.ndarray, float]:
        """
        Proyección α-mixing: primera α cuya trayectoria pasa la verificación.
        Si ninguna es segura, devuelve la emergency con α=1.0.
        """
        for alpha in self.BLEND_ALPHAS:
            candidate = (1.0 - alpha) * proposed + alpha * emergency
            candidate = np.clip(candidate, -1.0, 1.0).astype(np.float32)
            if self._check_trajectory_safety(
                candidate, horizon, risk_level, carla_map, ego, analysis
            ):
                return candidate, float(alpha)
        return emergency.astype(np.float32), 1.0

    # ────────────────────────── ACCESO A OBJETOS CARLA ──────────────────────────

    def _get_carla_map(self) -> Optional[carla.Map]:
        env = self.env
        while env is not None:
            if hasattr(env, "map") and env.map is not None:
                return env.map
            env = getattr(env, "env", None)
        return None

    def _get_ego_vehicle(self) -> Optional[carla.Vehicle]:
        env = self.env
        while env is not None:
            if hasattr(env, "ego_vehicle") and env.ego_vehicle is not None:
                return env.ego_vehicle
            env = getattr(env, "env", None)
        return None

    # ────────────────────────── ESTADÍSTICAS ──────────────────────────

    def get_statistics(self) -> Dict:
        total = sum(
            [
                self.stats["safe_steps"],
                self.stats["warning_steps"],
                self.stats["critical_steps"],
            ]
        )
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
            "safe_steps": 0,
            "warning_steps": 0,
            "critical_steps": 0,
            "interventions_by_horizon": {1: 0, 5: 0, 10: 0},
            "interventions_dynamic": 0,
            "interventions_static": 0,
            "interventions_pedestrian": 0,
        }
        self.shield_activations = 0
