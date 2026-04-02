import gymnasium as gym
import numpy as np
import math


class CarlaRewardShaper(gym.Wrapper):
    """
    Wrapper de reward shaping sobre CarlaEnv.

    Usa los datos del info dict (producidos por CarlaEnv._get_lane_features)
    que vienen directamente del Waypoint API de CARLA.

    Compatible con cualquier shield (se puede apilar encima o debajo).
    """

    INTENTIONAL_STEER_THRESHOLD: float = 0.25
    PROGRESS_MILESTONE_M: float = 25.0

    def __init__(
        self,
        env,
        target_speed_kmh: float = 30.0,
        speed_weight: float = 0.05,
        smoothness_weight: float = 0.10,
        lane_centering_weight: float = 0.15,
        heading_alignment_weight: float = 0.05,
        lane_invasion_penalty: float = 0.25,
        off_road_penalty: float = 2.00,
        edge_warning_weight: float = 0.30,
        progress_bonus_weight: float = 0.10,
        wrong_heading_penalty: float = 0.50,
        shield_intervention_penalty: float = 0.15,
    ):
        """
        Args:
            target_speed_kmh        : Velocidad objetivo (km/h)
            speed_weight            : Escala del bonus de velocidad
            smoothness_weight       : Escala de penalización de cambios bruscos
            lane_centering_weight   : Escala del bonus de centramiento en carril
            heading_alignment_weight: Escala del bonus de alineación angular
            lane_invasion_penalty   : Penalización por cruce de línea sólida
            off_road_penalty        : Penalización fuerte por salirse de carretera
            edge_warning_weight     : Penalización suave por acercarse al borde
            progress_bonus_weight   : Escala del bonus de progreso (milestone)
            wrong_heading_penalty   : Penalización por heading opuesto a waypoint
        """
        super().__init__(env)

        self.target_speed_kmh = target_speed_kmh
        self.speed_weight = speed_weight
        self.smoothness_weight = smoothness_weight
        self.lane_centering_weight = lane_centering_weight
        self.heading_alignment_weight = heading_alignment_weight
        self.lane_invasion_penalty = lane_invasion_penalty
        self.off_road_penalty = off_road_penalty
        self.edge_warning_weight = edge_warning_weight
        self.progress_bonus_weight    = progress_bonus_weight
        self.wrong_heading_penalty    = wrong_heading_penalty
        self.shield_intervention_penalty = shield_intervention_penalty


        self._last_steering     = 0.0
        self._last_milestone    = 0.0

    def reset(self, **kwargs):
        self._last_steering = 0.0
        self._last_milestone = 0.0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray):
        obs, base_reward, done, truncated, info = self.env.step(action)

        executed_action     = info.get("executed_action", action)
        current_steering    = float(executed_action[0])

        # ── Datos del Waypoint API (via CarlaEnv, exactos al cm) ──────
        speed_kmh           = info.get("speed_kmh", 0.0)
        lateral_offset_norm = info.get("lateral_offset_norm", 0.0)
        heading_error_norm  = info.get("heading_error_norm", 0.0)
        heading_error_deg   = info.get("heading_error", 0.0)
        on_road             = info.get("on_road", True)
        on_edge_warning     = info.get("on_edge_warning", 0.0)
        lane_invasion       = info.get("lane_invasion", False)
        total_distance      = info.get("total_distance", 0.0)

        # ── 1. Reward de velocidad ─────────────────────────────────────
        # Gaussiana centrada en target_speed_kmh (no penaliza por ir demasiado rápido
        # salvo que supere 2× el objetivo)
        if on_road and speed_kmh > 0.5:
            speed_diff = abs(speed_kmh - self.target_speed_kmh)
            # Sigma = 0.5 * target: a 1σ del objetivo el bonus vale ~0.6
            sigma = 0.5 * self.target_speed_kmh
            speed_reward = (
                math.exp(-(speed_diff**2) / (2.0 * sigma**2)) * self.speed_weight
            )
            if speed_kmh > self.target_speed_kmh * 2.0:
                overspeed_factor = (speed_kmh - self.target_speed_kmh * 2.0) / self.target_speed_kmh
                speed_reward -= overspeed_factor * self.speed_weight * 0.5
        else:
            speed_reward = 0.0

        # ── 2. Bonus de centramiento en carril ────────────────────────
        # Gaussiana: máximo en offset=0, sigma=0.35 del semi-ancho normalizado
        lane_centering = (
            math.exp(-(lateral_offset_norm**2) / (2.0 * 0.35**2)) *
            self.lane_centering_weight
        )

        # ── 3. Bonus de alineación angular ────────────────────────────
        heading_alignment = (
            math.exp(-(heading_error_norm**2) / (2.0 * 0.40**2)) *
            self.heading_alignment_weight
        )

        # ── 4. Penalización por steering brusco ───────────────────────
        steering_diff = abs(current_steering - self._last_steering)
        smoothness_penalty = steering_diff * self.smoothness_weight

        # ── 5. Penalización por invasión de carril (CARLA nativo) ─────
        # Distinguir invasión involuntaria (deriva) de maniobra intencional (cambio de carril).
        # El LaneInvasionSensor detecta cruces de líneas SÓLIDAS correctamente,
        # pero debe permitir cambios de carril activos.
        # Umbral steering 0.25 cubre cambios de carril estándar (acción[0] ≥ 0.25)
        # mientras identifica deriva involuntaria (acción[0] ≈ 0.0).
        if lane_invasion:
            intentional = abs(current_steering) >= self.INTENTIONAL_STEER_THRESHOLD
            if not intentional:
                invasion_severity = min(abs(lateral_offset_norm), 1.0)
                invasion_pen = self.lane_invasion_penalty * (0.5 + 0.5 * invasion_severity)
            else:
                invasion_pen = 0.0
        else:
            invasion_pen = 0.0

        # ── 6. Penalización por salirse de carretera ──────────────────
        if not on_road:
            road_penalty = self.off_road_penalty
        elif on_edge_warning > 0.3:
            # Penalización gradual al acercarse al borde
            road_penalty = on_edge_warning * self.edge_warning_weight
        else:
            road_penalty = 0.0

        # ── 7. Penalización por heading incorrecto (>90°) ───────────────
        abs_heading_deg = abs(heading_error_deg)
        if abs_heading_deg > 90.0:
            wrong_heading_pen = (
                (abs_heading_deg - 90.0) / 90.0 * self.wrong_heading_penalty
            )
        else:
            wrong_heading_pen = 0.0
        
        # ── 8. Bonus de progreso (milestone) ───────────────────────────
        milestone_crossed = (
            total_distance > 0 and
            total_distance >= self._last_milestone + self.PROGRESS_MILESTONE_M
        )
        if milestone_crossed:
            progress_bonus = self.progress_bonus_weight
            self._last_milestone = (
                (total_distance // self.PROGRESS_MILESTONE_M) * self.PROGRESS_MILESTONE_M
            )
        else:
            progress_bonus = 0.0
        
        shield_active    = info.get("shield_activated", info.get("shield_active", False))
        shield_intervention_pen = 0.0
        if shield_active and self.shield_intervention_penalty > 0.0:
            proposed_action = info.get("proposed_action", executed_action)
            action_divergence = float(np.linalg.norm(
                np.array(executed_action) - np.array(proposed_action)
            ))
            # Penalización proporcional a la divergencia (max divergence ≈ 2√2 ≈ 2.83)
            shield_intervention_pen = (
                action_divergence / 2.83 * self.shield_intervention_penalty
            )

        # ── Recompensa moldeada final ─────────────────────────────────
        shaped_reward = (
            base_reward
            + speed_reward
            + lane_centering
            + heading_alignment
            + progress_bonus
            - smoothness_penalty
            - invasion_pen
            - road_penalty
            - wrong_heading_pen
            - shield_intervention_pen
        )

        self._last_steering = current_steering

        info.update({
            "shaped_reward":    shaped_reward,
            "raw_reward":       base_reward,
            "speed_bonus":      speed_reward,
            "lane_center_bonus": lane_centering,
            "heading_bonus":    heading_alignment,
            "smooth_penalty":   smoothness_penalty,
            "invasion_penalty": invasion_pen,
            "road_penalty":     road_penalty,
            "wrong_heading_pen":    wrong_heading_pen,
            "progress_bonus":       progress_bonus,
            "shield_intervention_pen": shield_intervention_pen,
            "invasion_intentional": (
                lane_invasion and
                abs(current_steering) >= self.INTENTIONAL_STEER_THRESHOLD
            ),
        })

        return obs, shaped_reward, done, truncated, info
