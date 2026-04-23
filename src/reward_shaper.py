"""
reward_shaper.py - Reward shaping desacoplado del shield.

DISEÑO: el reward recompensa la CONDUCTA del vehículo (velocidad, centramiento,
progreso, suavidad, evitación de bordes), no la presencia/ausencia del shield.
El agente no controla si el shield actúa; por tanto recompensar/penalizar
según la activación del shield introduce gradiente espurio y crea el óptimo
local "parar y dejar que el shield me corrija".

Componentes (10):
  + alive_bonus          (gated by speed_gate)
  + speed_reward         (Gaussiana centrada en velocidad objetivo)
  + lane_centering       (proximidad al centro del carril)
  + heading_alignment    (alineación angular con la carretera)
  + progress_bonus       (milestone cada 25 m)
  - smoothness_penalty   (|Δsteering|)
  - invasion_pen         (cruce no intencional de línea sólida)
  - road_penalty         (borde gradual o off_road puntual)
  - wrong_heading_pen    (heading > 90° respecto waypoint)
  - idle_penalty         (speed < min_moving) ← siempre activo en on_road
  - drift_penalty        (asimetría lateral con bordes cerca)

Durante `lane_change_permitted` AND |lateral_offset| > 0.5 se suprimen
invasion/road/drift para no castigar maniobras legítimas.
"""

import gymnasium as gym
import numpy as np
import math


class CarlaRewardShaper(gym.Wrapper):
    INTENTIONAL_STEER_THRESHOLD: float = 0.25
    PROGRESS_MILESTONE_M: float = 25.0

    def __init__(
        self,
        env,
        target_speed_kmh: float = 30.0,
        speed_weight: float = 0.10,
        smoothness_weight: float = 0.10,
        lane_centering_weight: float = 0.10,
        heading_alignment_weight: float = 0.04,
        lane_invasion_penalty: float = 0.25,
        off_road_penalty: float = 2.00,
        edge_warning_weight: float = 0.30,
        progress_bonus_weight: float = 0.30,
        wrong_heading_penalty: float = 0.50,
        speed_limit_margin: float = 0.05,
        idle_penalty_weight: float = 0.10,
        min_moving_speed_kmh: float = 5.0,
        speed_gate_full_kmh: float = 10.0,
        curvature_speed_scale: float = 0.4,
        lane_drift_penalty_weight: float = 0.08,
        alive_bonus: float = 0.15,
    ):
        super().__init__(env)

        self.target_speed_kmh = target_speed_kmh
        self.speed_weight = speed_weight
        self.smoothness_weight = smoothness_weight
        self.lane_centering_weight = lane_centering_weight
        self.heading_alignment_weight = heading_alignment_weight
        self.lane_invasion_penalty = lane_invasion_penalty
        self.off_road_penalty = off_road_penalty
        self.edge_warning_weight = edge_warning_weight
        self.progress_bonus_weight = progress_bonus_weight
        self.wrong_heading_penalty = wrong_heading_penalty
        self.speed_limit_margin = speed_limit_margin
        self.idle_penalty_weight = idle_penalty_weight
        self.min_moving_speed_kmh = min_moving_speed_kmh
        self.speed_gate_full_kmh = speed_gate_full_kmh
        self.curvature_speed_scale = curvature_speed_scale
        self.lane_drift_penalty_weight = lane_drift_penalty_weight
        self.alive_bonus = alive_bonus

        self._last_steering = 0.0
        self._last_milestone = 0.0

    def reset(self, **kwargs):
        self._last_steering = 0.0
        self._last_milestone = 0.0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray):
        obs, base_reward, done, truncated, info = self.env.step(action)

        executed_action = info.get("executed_action", action)
        current_steering = float(executed_action[0])

        speed_kmh = info.get("speed_kmh", 0.0)
        lateral_offset_norm = info.get("lateral_offset_norm", 0.0)
        heading_error_norm = info.get("heading_error_norm", 0.0)
        heading_error_deg = info.get("heading_error", 0.0)
        on_road = info.get("on_road", True)
        on_edge_warning = info.get("on_edge_warning", 0.0)
        lane_invasion = info.get("lane_invasion", False)
        total_distance = info.get("total_distance", 0.0)
        raw_limit = info.get("speed_limit_kmh", 0.0)
        dist_left_edge_norm = info.get("dist_left_edge_norm", 0.5)
        dist_right_edge_norm = info.get("dist_right_edge_norm", 0.5)
        road_curvature_norm = info.get("road_curvature_norm", 0.0)

        effective_limit = float(raw_limit) if raw_limit > 0.0 else self.target_speed_kmh

        if self.speed_gate_full_kmh > self.min_moving_speed_kmh:
            speed_gate = float(
                np.clip(
                    (speed_kmh - self.min_moving_speed_kmh)
                    / (self.speed_gate_full_kmh - self.min_moving_speed_kmh),
                    0.0,
                    1.0,
                )
            )
        else:
            speed_gate = float(
                np.clip(speed_kmh / max(self.speed_gate_full_kmh, 1.0), 0.0, 1.0)
            )

        # ── 1. Speed reward ─────────────────────────────────────────
        curvature_magnitude = abs(road_curvature_norm)
        curvature_factor = 1.0 - self.curvature_speed_scale * min(
            curvature_magnitude / 0.6, 1.0
        )
        curve_adjusted_limit = effective_limit * max(curvature_factor, 0.4)

        if on_road and speed_kmh > 0.5:
            speed_diff = abs(speed_kmh - curve_adjusted_limit)
            sigma = 0.35 * curve_adjusted_limit
            speed_reward = (
                math.exp(-(speed_diff**2) / (2.0 * sigma**2)) * self.speed_weight
            )
            speed_ceiling = effective_limit * (1.0 + self.speed_limit_margin)
            if speed_kmh > speed_ceiling:
                overspeed = (speed_kmh - speed_ceiling) / effective_limit
                speed_reward -= overspeed * self.speed_weight * 0.8
        else:
            speed_reward = 0.0

        # ── Detección de transición de carril ────────────────────────
        lane_change_permitted = info.get("lane_change_permitted", False)
        in_lane_transition = lane_change_permitted and abs(lateral_offset_norm) > 0.5

        # ── 2. Lane centering ────────────────────────────────────────
        min_edge_dist = min(dist_left_edge_norm, dist_right_edge_norm)
        centering_score = float(np.clip(min_edge_dist / 0.5, 0.0, 1.0))
        if in_lane_transition:
            lane_centering = 0.3 * self.lane_centering_weight
        else:
            lane_centering = (
                max(speed_gate, 0.3) * centering_score * self.lane_centering_weight
            )

        # ── 3. Heading alignment ─────────────────────────────────────
        heading_alignment = (
            speed_gate
            * math.exp(-(heading_error_norm**2) / (2.0 * 0.40**2))
            * self.heading_alignment_weight
        )

        # ── 4. Smoothness ────────────────────────────────────────────
        steering_diff = abs(current_steering - self._last_steering)
        smoothness_penalty = steering_diff * self.smoothness_weight

        # ── 5. Lane invasion ────────────────────────────────────────
        if in_lane_transition:
            invasion_pen = 0.0
        elif lane_invasion:
            intentional = abs(current_steering) >= self.INTENTIONAL_STEER_THRESHOLD
            if not intentional:
                invasion_severity = min(abs(lateral_offset_norm), 1.0)
                invasion_pen = self.lane_invasion_penalty * (
                    0.5 + 0.5 * invasion_severity
                )
            else:
                invasion_pen = 0.0
        else:
            invasion_pen = 0.0

        # ── 6. Edge / off-road ──────────────────────────────────────
        if not on_road:
            road_penalty = self.off_road_penalty
        elif in_lane_transition:
            road_penalty = 0.0
        else:
            critical_edge = min(dist_left_edge_norm, dist_right_edge_norm)
            edge_threshold = 0.4
            if critical_edge < edge_threshold:
                edge_proximity = (
                    (edge_threshold - critical_edge) / edge_threshold
                ) ** 2
                road_penalty = edge_proximity * self.edge_warning_weight
            elif on_edge_warning > 0.3:
                road_penalty = on_edge_warning * self.edge_warning_weight * 0.5
            else:
                road_penalty = 0.0

        # ── 7. Wrong heading ────────────────────────────────────────
        abs_heading_deg = abs(heading_error_deg)
        if abs_heading_deg > 90.0:
            wrong_heading_pen = (
                (abs_heading_deg - 90.0) / 90.0 * self.wrong_heading_penalty
            )
        else:
            wrong_heading_pen = 0.0

        # ── 8. Progress milestone ────────────────────────────────────
        milestone_crossed = (
            total_distance > 0
            and total_distance >= self._last_milestone + self.PROGRESS_MILESTONE_M
        )
        if milestone_crossed:
            progress_bonus = self.progress_bonus_weight
            self._last_milestone = (
                total_distance // self.PROGRESS_MILESTONE_M
            ) * self.PROGRESS_MILESTONE_M
        else:
            progress_bonus = 0.0

        # ── 9. Idle penalty (SIN grace period) ───────────────────────
        if speed_kmh < self.min_moving_speed_kmh and on_road:
            idle_fraction = 1.0 - speed_kmh / max(self.min_moving_speed_kmh, 1.0)
            idle_penalty = idle_fraction * self.idle_penalty_weight
        else:
            idle_penalty = 0.0

        # ── 10. Drift asimétrico ────────────────────────────────────
        edge_asymmetry = abs(dist_left_edge_norm - dist_right_edge_norm)
        if in_lane_transition:
            drift_penalty = 0.0
        elif edge_asymmetry > 0.3 and min_edge_dist < 0.35:
            drift_penalty = (edge_asymmetry - 0.3) * self.lane_drift_penalty_weight
        else:
            drift_penalty = 0.0

        # ── Alive bonus (gated by speed) ────────────────────────────
        alive_bonus_val = self.alive_bonus * speed_gate if on_road else 0.0

        # ── Suma final ──────────────────────────────────────────────
        shaped_reward = (
            base_reward
            + alive_bonus_val
            + speed_reward
            + lane_centering
            + heading_alignment
            + progress_bonus
            - smoothness_penalty
            - invasion_pen
            - road_penalty
            - wrong_heading_pen
            - idle_penalty
            - drift_penalty
        )

        self._last_steering = current_steering

        info.update(
            {
                "shaped_reward": shaped_reward,
                "raw_reward": base_reward,
                "alive_bonus": alive_bonus_val,
                "speed_bonus": speed_reward,
                "lane_center_bonus": lane_centering,
                "heading_bonus": heading_alignment,
                "smooth_penalty": smoothness_penalty,
                "invasion_penalty": invasion_pen,
                "road_penalty": road_penalty,
                "wrong_heading_pen": wrong_heading_pen,
                "progress_bonus": progress_bonus,
                "idle_penalty": idle_penalty,
                "drift_penalty": drift_penalty,
                "effective_speed_limit": effective_limit,
                "curve_adjusted_limit": curve_adjusted_limit,
                "centering_score": centering_score,
                "invasion_intentional": (
                    lane_invasion
                    and abs(current_steering) >= self.INTENTIONAL_STEER_THRESHOLD
                ),
                "in_lane_transition": in_lane_transition,
            }
        )

        return obs, shaped_reward, done, truncated, info
