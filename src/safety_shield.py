"""
safety_shield.py - Basic Safety Shield (proyección continua)

PARADIGMA: MÍNIMA INTERFERENCIA POR PROYECCIÓN
  - Si la acción propuesta es segura → pass-through (shield_mask=0).
  - Si es insegura → se interpola con una acción-objetivo (`emergency`)
    mediante α∈{0.25, 0.5, 0.75, 1.0}; se devuelve la primera mezcla segura.
  - La intensidad α se expone como `shield_intensity` en info.

Con esta proyección la acción ejecutada queda siempre cerca de la propuesta
(no hay saltos discretos a [0, -1] o [±0.5, -0.5]) y nunca alcanza la
saturación post-tanh que rompía la estabilidad numérica del log_prob en PPO.

CAPA 1 — Obstáculos (LIDAR frontal + laterales).
CAPA 2 — Límites de carril (Waypoint API: lateral_offset_norm, heading).
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict


class CarlaSafetyShield(gym.Wrapper):
    """Basic shield con proyección hacia acción de emergencia."""

    BLEND_ALPHAS = (0.25, 0.5, 0.75, 1.0)
    SHIELD_MASK_THRESHOLD = 0.05  # α ≥ threshold ⇒ shield_mask = 1.0

    def __init__(
        self,
        env,
        num_lidar_rays: int = 240,
        front_threshold: float = 0.15,
        side_threshold: float = 0.04,
        lateral_threshold: float = 0.82,
        heading_threshold: float = 0.60,
        emergency_brake: float = -0.6,
        lane_correction_gain: float = 1.5,
    ):
        super().__init__(env)
        self.num_lidar_rays = num_lidar_rays
        self.front_threshold = front_threshold
        self.side_threshold = side_threshold
        self.lateral_threshold = lateral_threshold
        self.heading_threshold = heading_threshold
        self.emergency_brake = emergency_brake
        self.lane_correction_gain = lane_correction_gain

        self.shield_activations = 0
        self.last_obs = None
        self.last_info: Dict = {}

        self.intervention_stats = {
            "front": 0,
            "side_right": 0,
            "side_left": 0,
            "lane_right": 0,
            "lane_left": 0,
            "heading": 0,
        }

    # ────────────────────────── GYMNASIUM ──────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.last_info = info
        return obs, info

    def step(self, action: np.ndarray):
        lidar_scan = self._get_lidar(self.last_obs)
        lidar_analysis = self._analyze_lidar(lidar_scan)
        lat_norm = self.last_info.get("lateral_offset_norm", 0.0)
        head_norm = self.last_info.get("heading_error_norm", 0.0)

        proposed = np.asarray(action, dtype=np.float32).copy()

        if self._is_safe(proposed, lidar_analysis, lat_norm, head_norm):
            final_action = proposed
            alpha = 0.0
            reason = "safe"
        else:
            emergency = self._build_emergency_action(
                lidar_analysis, lat_norm, head_norm
            )
            final_action, alpha = self._project(
                proposed, emergency, lidar_analysis, lat_norm, head_norm
            )
            reason = self._classify_reason(lidar_analysis, lat_norm, head_norm)
            self._update_stats(reason)
            self.shield_activations += 1

        shield_activated = alpha >= self.SHIELD_MASK_THRESHOLD

        obs, reward, done, truncated, info = self.env.step(final_action)
        self.last_obs = obs
        self.last_info = info

        info.update(
            {
                "shield_activated": shield_activated,
                "shield_intensity": float(alpha),
                "shield_reason": reason,
                "min_front_dist": lidar_analysis["min_front"],
                "min_r_side_dist": lidar_analysis["min_r_side"],
                "min_l_side_dist": lidar_analysis["min_l_side"],
                "min_distance": lidar_analysis["min_dist"],
                "total_shield_activations": self.shield_activations,
                "executed_action": final_action,
                "proposed_action": proposed,
                "shield_modified_action": shield_activated,
            }
        )

        return obs, reward, done, truncated, info

    # ────────────────────────── LIDAR ──────────────────────────

    def _get_lidar(self, obs: np.ndarray) -> np.ndarray:
        return obs[: self.num_lidar_rays]

    def _analyze_lidar(self, scan: np.ndarray) -> Dict:
        n = self.num_lidar_rays
        front = np.concatenate((scan[n - 15 : n], scan[:15]))
        r_side = scan[40:80]
        l_side = scan[160:200]
        return {
            "min_front": float(np.min(front)),
            "min_r_side": float(np.min(r_side)),
            "min_l_side": float(np.min(l_side)),
            "min_dist": float(np.min(scan)),
        }

    # ────────────────────────── SAFETY CHECK ──────────────────────────

    def _is_safe(
        self,
        action: np.ndarray,
        analysis: Dict,
        lat_norm: float,
        head_norm: float,
    ) -> bool:
        tb = float(action[1])

        if analysis["min_front"] < self.front_threshold and tb > 0.0:
            return False
        if analysis["min_r_side"] < self.side_threshold:
            return False
        if analysis["min_l_side"] < self.side_threshold:
            return False
        if abs(lat_norm) > self.lateral_threshold:
            return False
        if abs(head_norm) > self.heading_threshold:
            return False
        return True

    def _classify_reason(
        self, analysis: Dict, lat_norm: float, head_norm: float
    ) -> str:
        if analysis["min_front"] < self.front_threshold:
            return "front_obstacle"
        if analysis["min_r_side"] < self.side_threshold:
            return "right_obstacle"
        if analysis["min_l_side"] < self.side_threshold:
            return "left_obstacle"
        if abs(lat_norm) > self.lateral_threshold:
            return "lane_boundary"
        if abs(head_norm) > self.heading_threshold:
            return "heading_error"
        return "safe"

    # ────────────────────────── PROYECCIÓN ──────────────────────────

    def _build_emergency_action(
        self, analysis: Dict, lat_norm: float, head_norm: float
    ) -> np.ndarray:
        steer_target = float(np.clip(-lat_norm * self.lane_correction_gain, -1.0, 1.0))

        if analysis["min_l_side"] < self.side_threshold:
            steer_target = float(np.clip(steer_target + 0.4, -1.0, 1.0))
        if analysis["min_r_side"] < self.side_threshold:
            steer_target = float(np.clip(steer_target - 0.4, -1.0, 1.0))

        if abs(head_norm) > self.heading_threshold:
            steer_target = float(np.clip(steer_target - 0.5 * head_norm, -1.0, 1.0))

        tb_target = self.emergency_brake
        if analysis["min_front"] < self.front_threshold * 0.5:
            tb_target = -1.0

        return np.array([steer_target, tb_target], dtype=np.float32)

    def _project(
        self,
        proposed: np.ndarray,
        emergency: np.ndarray,
        analysis: Dict,
        lat_norm: float,
        head_norm: float,
    ) -> Tuple[np.ndarray, float]:
        for alpha in self.BLEND_ALPHAS:
            candidate = (1.0 - alpha) * proposed + alpha * emergency
            candidate = np.clip(candidate, -1.0, 1.0).astype(np.float32)
            if self._is_safe(candidate, analysis, lat_norm, head_norm):
                return candidate, float(alpha)
        return emergency.astype(np.float32), 1.0

    def _update_stats(self, reason: str):
        if reason == "front_obstacle":
            self.intervention_stats["front"] += 1
        elif reason == "right_obstacle":
            self.intervention_stats["side_right"] += 1
        elif reason == "left_obstacle":
            self.intervention_stats["side_left"] += 1
        elif reason == "lane_boundary":
            if self.last_info.get("lateral_offset_norm", 0.0) > 0:
                self.intervention_stats["lane_right"] += 1
            else:
                self.intervention_stats["lane_left"] += 1
        elif reason == "heading_error":
            self.intervention_stats["heading"] += 1

    # ────────────────────────── ESTADÍSTICAS ──────────────────────────

    def get_statistics(self) -> Dict:
        return {
            "total_interventions": self.shield_activations,
            "interventions_by_reason": dict(self.intervention_stats),
            "safe_rate": 0.0,
            "warning_rate": 0.0,
            "critical_rate": 0.0,
            "interventions_dynamic": 0,
            "interventions_static": 0,
            "interventions_pedestrian": 0,
        }

    def reset_statistics(self):
        self.shield_activations = 0
        for k in self.intervention_stats:
            self.intervention_stats[k] = 0
