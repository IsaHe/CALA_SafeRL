import gymnasium as gym
import numpy as np
import math


class CarlaRewardShaper(gym.Wrapper):
    """
    Wrapper de reward shaping sobre CarlaEnv (v2).

    Usa los datos del info dict (producidos por CarlaEnv._get_lane_features y_get_route_features), que vienen directamente del Waypoint API de CARLA.
    """

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
        shield_intervention_penalty: float = 0.05,
        speed_limit_margin: float = 0.05,
        idle_penalty_weight: float = 0.10,
        min_moving_speed_kmh: float = 5.0,
        speed_gate_full_kmh: float = 10.0,
        curvature_speed_scale: float = 0.4,
        lane_drift_penalty_weight: float = 0.08,
        alive_bonus: float = 0.15,
        shield_grace_duration: int = 40,
        shield_not_activated_bonus: float = 0.05,
    ):
        """
        Args:
            target_speed_kmh          : Velocidad objetivo (km/h)
            speed_weight              : Escala del bonus de velocidad
            smoothness_weight         : Escala de penalización de cambios bruscos
            lane_centering_weight     : Escala del bonus de centramiento en carril
            heading_alignment_weight  : Escala del bonus de alineación angular
            lane_invasion_penalty     : Penalización por cruce de línea sólida
            off_road_penalty          : Penalización fuerte por salirse de carretera
            edge_warning_weight       : Penalización gradual por proximidad al borde
            progress_bonus_weight     : Escala del bonus de progreso (milestone)
            wrong_heading_penalty     : Penalización por heading opuesto a waypoint
            speed_limit_margin        : Margen para considerar velocidad límite
            curvature_speed_scale     : Cuánto reduce la velocidad objetivo en curvas fuertes
                                        (0 = sin efecto, 1 = reduce hasta 0 en curva máxima)
            lane_drift_penalty_weight : Penalización por conducción sesgada hacia un borde
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
        self.progress_bonus_weight = progress_bonus_weight
        self.wrong_heading_penalty = wrong_heading_penalty
        self.shield_intervention_penalty = shield_intervention_penalty
        self.speed_limit_margin = speed_limit_margin
        self.idle_penalty_weight = idle_penalty_weight
        self.min_moving_speed_kmh = min_moving_speed_kmh
        self.speed_gate_full_kmh = speed_gate_full_kmh
        self.curvature_speed_scale = curvature_speed_scale
        self.lane_drift_penalty_weight = lane_drift_penalty_weight
        self.alive_bonus = alive_bonus
        self.shield_grace_duration = shield_grace_duration
        self.shield_not_activated_bonus = shield_not_activated_bonus

        self._last_steering = 0.0
        self._last_milestone = 0.0
        self._shield_grace_steps = 0

    def reset(self, **kwargs):
        self._last_steering = 0.0
        self._last_milestone = 0.0
        self._shield_grace_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray):
        obs, base_reward, done, truncated, info = self.env.step(action)

        executed_action = info.get("executed_action", action)
        current_steering = float(executed_action[0])

        # ── Datos del Waypoint API ────────────────────────────────────
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

        # ── 1. Reward de velocidad con factor de curvatura y penalización por exceso de velocidad respecto a el límite (con margen) ────────────────────────
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
            # Penalización por exceder el límite (más allá del margen permitido)
            speed_ceiling = effective_limit * (1.0 + self.speed_limit_margin)
            if speed_kmh > speed_ceiling:
                overspeed = (speed_kmh - speed_ceiling) / effective_limit
                speed_reward -= overspeed * self.speed_weight * 0.8
        else:
            speed_reward = 0.0

        # ── Detección de transición de carril ────────────────────────
        # Cuando las marcas viales permiten el cambio y el vehículo está
        # significativamente descentrado, suprimimos penalties que castigan
        # la posición inter-carril (edge, invasion, drift).
        lane_change_permitted = info.get("lane_change_permitted", False)
        in_lane_transition = lane_change_permitted and abs(lateral_offset_norm) > 0.5

        # ── 2. Bonus de centramiento basado en distancias a los bordes ─
        min_edge_dist = min(dist_left_edge_norm, dist_right_edge_norm)
        # Normalizar: min_edge=0.5 (centro exacto) → 1.0, min_edge→0 → 0.0
        centering_score = float(np.clip(min_edge_dist / 0.5, 0.0, 1.0))
        # Floor de 0.3 para que incluso parado reciba señal de centramiento.
        # Durante transición de carril, usar floor completo (no penalizar
        # la posición intermedia, el agente recuperará centramiento al llegar).
        if in_lane_transition:
            lane_centering = 0.3 * self.lane_centering_weight
        else:
            lane_centering = (
                max(speed_gate, 0.3) * centering_score * self.lane_centering_weight
            )

        # ── 3. Bonus de alineación angular ────────────────────────────
        heading_alignment = (
            speed_gate
            * math.exp(-(heading_error_norm**2) / (2.0 * 0.40**2))
            * self.heading_alignment_weight
        )

        # ── 4. Penalización por steering brusco ───────────────────────
        steering_diff = abs(current_steering - self._last_steering)
        smoothness_penalty = steering_diff * self.smoothness_weight

        # ── 5. Penalización por invasión de carril ─────
        # Suprimida durante transición de carril permitida (las marcas viales
        # permiten el cruce; penalizar aquí bloquearía la maniobra).
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

        # ── 6. Penalización de borde (gradual + cuadrática) ───────────
        # Suprimida durante transición de carril permitida: estar cerca del
        # borde del carril actual es inevitable al cambiar de carril.
        if not on_road:
            road_penalty = self.off_road_penalty
        elif in_lane_transition:
            road_penalty = 0.0
        else:
            # Usar el mínimo borde como indicador de proximidad
            critical_edge = min(dist_left_edge_norm, dist_right_edge_norm)
            edge_threshold = 0.4
            if critical_edge < edge_threshold:
                # Escala 0→1 cuadráticamente al acercarse al borde
                edge_proximity = (
                    (edge_threshold - critical_edge) / edge_threshold
                ) ** 2
                road_penalty = edge_proximity * self.edge_warning_weight
            elif on_edge_warning > 0.3:
                # Fallback a on_edge_warning si dist_edge no es confiable
                road_penalty = on_edge_warning * self.edge_warning_weight * 0.5
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

        # ── 9. Penalización por intervención del shield ───────────────
        shield_active = info.get("shield_activated", info.get("shield_active", False))
        shield_intervention_pen = 0.0
        shield_not_activated_bonus = 0.0
        proposed_action = info.get("proposed_action", executed_action)
        action_divergence = float(
            np.linalg.norm(np.array(executed_action) - np.array(proposed_action))
        )
        if shield_active and self.shield_intervention_penalty > 0.0:
            shield_intervention_pen = (
                action_divergence / 2.83 * self.shield_intervention_penalty
            )
        elif not shield_active and self.shield_not_activated_bonus > 0.0:
            # Bonus pequeño por no activar el shield (fomenta conducción segura sin depender del shield)
            shield_not_activated_bonus = (
                action_divergence / 2.83 * self.shield_not_activated_bonus
            )

        # ── 10. Penalización por vehículo parado ──────────────
        # Grace period: tras intervención del shield, suprimir idle_penalty
        # para dar tiempo al agente a recuperar velocidad.
        # IMPORTANTE: solo se activa si no está ya en grace period, para
        # evitar que activaciones continuas del shield lo reseteen
        # indefinidamente (lo que suprimiría idle_penalty para siempre).
        if shield_active and self._shield_grace_steps <= 0:
            self._shield_grace_steps = self.shield_grace_duration

        if (
            speed_kmh < self.min_moving_speed_kmh
            and on_road
            and self._shield_grace_steps <= 0
        ):
            idle_fraction = 1.0 - speed_kmh / max(self.min_moving_speed_kmh, 1.0)
            idle_penalty = idle_fraction * self.idle_penalty_weight
        else:
            idle_penalty = 0.0

        if self._shield_grace_steps > 0:
            self._shield_grace_steps -= 1

        # ── 11. Penalización por drift asimétrico ─────────────────────
        # Suprimida durante transición de carril (la asimetría es inherente
        # a la maniobra de cambio).
        edge_asymmetry = abs(dist_left_edge_norm - dist_right_edge_norm)
        if in_lane_transition:
            drift_penalty = 0.0
        elif edge_asymmetry > 0.3 and min_edge_dist < 0.35:
            drift_penalty = (edge_asymmetry - 0.3) * self.lane_drift_penalty_weight
        else:
            drift_penalty = 0.0

        # ── 12. Alive bonus (contrarresta acumulación de penalties) ────
        alive_bonus_val = self.alive_bonus if on_road else 0.0

        # ── Recompensa moldeada final ─────────────────────────────────
        shaped_reward = (
            base_reward
            + alive_bonus_val
            + speed_reward
            + lane_centering
            + heading_alignment
            + progress_bonus
            + shield_not_activated_bonus
            - smoothness_penalty
            - invasion_pen
            - road_penalty
            - wrong_heading_pen
            - shield_intervention_pen
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
                "shield_intervention_pen": shield_intervention_pen,
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
