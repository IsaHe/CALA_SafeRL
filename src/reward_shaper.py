"""
reward_shaper.py - Reward shaping monótono sin bolsillo de reposo.

DISEÑO (sesión 3):
  El paisaje de reward debe tener un **óptimo local NEGATIVO en reposo**
  para que el agente no encuentre refugio quedándose parado. La pendiente
  ∂R/∂v debe ser positiva desde el primer paso de movimiento.

  Señal tomada del LÍMITE DINÁMICO de carretera (`speed_limit_kmh` del
  Waypoint API) en vez de un target fijo: así el agente aprende a adaptar
  su velocidad al contexto.

Componentes (11 — sesión 4 añade acceleration_reward):
  + progress_reward      (lineal satura a 10 km/h; weight 0.30)
  + acceleration_reward  (clip(Δv, 0, 2) · weight 0.08; señal densa desde v=0)
  + speed_reward         (Gaussiana centrada en el límite dinámico)
  + lane_centering       (gated por has_moved_recently; off si está parado ≥5 pasos)
  + heading_alignment    (gated por has_moved_recently)
  + progress_milestone   (bonus puntual cada 25 m)
  - smoothness_penalty   (|Δsteering|)
  - invasion_pen         (cruce no intencional)
  - road_penalty         (borde gradual O off_road puntual, ajustado a 1.0)
  - wrong_heading_pen    (heading > 90°)
  - idle_penalty         (ESCALONADA · atenuada 0.3 si throttle>0.3 y v<2)
  - drift_penalty        (asimetría lateral con bordes cerca)

IDLE PENALTY ESCALONADA (pico 0.25):
    speed < 0.5 km/h          → −idle_weight              = −0.25
    0.5 ≤ speed < 2 km/h      → −idle_weight * 0.5        = −0.125
    2 ≤ speed < 5 km/h        → −idle_weight * 0.2        = −0.05
    speed ≥ 5 km/h            → 0

GATED CENTERING/HEADING:
  lane_centering y heading_alignment SÓLO se entregan si el agente se ha
  movido > IDLE_SPEED_THRESHOLD_KMH en alguno de los últimos
  MOVEMENT_WINDOW_STEPS pasos. Evita recompensar "parar centrado".

El shield NO interviene aquí. Durante `lane_change_permitted` AND
|lateral_offset| > 0.5 se suprimen invasion/road/drift para no castigar
maniobras legítimas.
"""

import gymnasium as gym
import numpy as np
import math
from collections import deque


class CarlaRewardShaper(gym.Wrapper):
    INTENTIONAL_STEER_THRESHOLD: float = 0.25
    PROGRESS_MILESTONE_M: float = 25.0
    IDLE_SPEED_THRESHOLD_KMH: float = 0.5
    MOVEMENT_WINDOW_STEPS: int = 5  # memoria para gate de centering/heading

    # Penalty hard por cruzar una línea sólida (detectado por LaneInvasionSensor,
    # que ya filtra a tipos sólidos). Desincentiva maniobras ilegales aunque
    # `lane_change_permitted` esté activo.
    SOLID_INVASION_PENALTY: float = 5.0

    # Coste pequeño por evento de cambio de carril (se detecta por cambio de
    # lane_id entre pasos). Con cooldown para no contar dobles triggers del
    # mismo evento físico. Combate el comportamiento errático de weaving.
    LANE_CHANGE_COST: float = 0.05
    LANE_CHANGE_COOLDOWN_STEPS: int = 20

    # Multiplicadores del idle escalonado (aplicados sobre idle_penalty_weight)
    IDLE_MULT_DEAD_STOP: float = 1.0  # speed < IDLE_SPEED_THRESHOLD_KMH
    IDLE_MULT_CRAWL: float = 0.5  # IDLE_SPEED_THRESHOLD_KMH ≤ speed < 2
    IDLE_MULT_SLOW: float = 0.2  # 2 ≤ speed < 5
    IDLE_TIER_MID_KMH: float = 2.0
    IDLE_TIER_HIGH_KMH: float = 5.0

    # Velocidad a la que satura el `progress_reward`. Se eligió 10 km/h (no
    # el límite dinámico) para amplificar ∂R/∂v en el tramo 0-10 km/h —
    # que es exactamente donde el agente estaba estancado (sesión 4).
    # Por encima de 10 km/h, `speed_reward` Gaussiana toma el relevo para
    # llevar al agente hasta el `effective_limit`.
    PROGRESS_SATURATION_KMH: float = 10.0

    # Atenuación de la idle_penalty cuando el agente comanda throttle>0.3
    # con speed<2 km/h (intentando arrancar — la física tarda en responder).
    IDLE_ACTION_ATTENUATION: float = 0.3
    IDLE_ACTION_THROTTLE_THRESHOLD: float = 0.3

    # Cap para delta_v en `acceleration_reward` (km/h por paso a 20 Hz).
    # Tesla Model 3 max ≈ 0.9 km/h/step, 2.0 deja margen ante glitches.
    ACCELERATION_DELTA_CAP_KMH: float = 2.0

    def __init__(
        self,
        env,
        target_speed_kmh: float = 30.0,  # fallback cuando no hay speed_limit válido
        speed_weight: float = 0.10,
        smoothness_weight: float = 0.10,
        lane_centering_weight: float = 0.15,
        heading_alignment_weight: float = 0.04,
        lane_invasion_penalty: float = 0.25,
        off_road_penalty: float = 1.00,  # bajado: el env ya penaliza −8 en base
        edge_warning_weight: float = 0.30,
        progress_bonus_weight: float = 0.30,
        wrong_heading_penalty: float = 0.50,
        speed_limit_margin: float = 0.05,
        idle_penalty_weight: float = 0.25,  # pico del escalón
        curvature_speed_scale: float = 0.4,
        lane_drift_penalty_weight: float = 0.08,
        progress_reward_weight: float = 0.30,
        acceleration_reward_weight: float = 0.08,
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
        self.curvature_speed_scale = curvature_speed_scale
        self.lane_drift_penalty_weight = lane_drift_penalty_weight
        self.progress_reward_weight = progress_reward_weight
        self.acceleration_reward_weight = acceleration_reward_weight

        self._last_steering = 0.0
        self._last_milestone = 0.0
        self._prev_speed_kmh = 0.0
        # Ventana móvil de velocidades para `has_moved_recently`.
        self._recent_speed_window: deque = deque(maxlen=self.MOVEMENT_WINDOW_STEPS)
        # Detección de cambio de carril por cambio de lane_id.
        self._last_lane_id = None
        self._lane_change_cooldown = 0

    def reset(self, **kwargs):
        self._last_steering = 0.0
        self._last_milestone = 0.0
        self._prev_speed_kmh = 0.0
        self._recent_speed_window.clear()
        self._last_lane_id = None
        self._lane_change_cooldown = 0
        return self.env.reset(**kwargs)

    @staticmethod
    def _idle_penalty_scaled(speed_kmh: float, weight: float) -> float:
        """Idle penalty escalonada (tiered) por velocidad."""
        if speed_kmh < CarlaRewardShaper.IDLE_SPEED_THRESHOLD_KMH:
            return weight * CarlaRewardShaper.IDLE_MULT_DEAD_STOP
        if speed_kmh < CarlaRewardShaper.IDLE_TIER_MID_KMH:
            return weight * CarlaRewardShaper.IDLE_MULT_CRAWL
        if speed_kmh < CarlaRewardShaper.IDLE_TIER_HIGH_KMH:
            return weight * CarlaRewardShaper.IDLE_MULT_SLOW
        return 0.0

    def _has_moved_recently(self) -> bool:
        """True si en alguno de los últimos pasos la velocidad fue > umbral idle."""
        if not self._recent_speed_window:
            return False
        return any(
            v >= self.IDLE_SPEED_THRESHOLD_KMH for v in self._recent_speed_window
        )

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

        # Límite efectivo: dinámico desde el Waypoint API cuando disponible,
        # fallback a `target_speed_kmh` sólo si el info no tiene señal.
        effective_limit = float(raw_limit) if raw_limit > 0.0 else self.target_speed_kmh

        # Actualizar ventana de velocidad reciente ANTES de computar gates.
        self._recent_speed_window.append(float(speed_kmh))
        has_moved = self._has_moved_recently()

        # ── 1. Progress reward DENSO con SATURACIÓN BAJA (sesión 4) ──
        # Satura a PROGRESS_SATURATION_KMH=10 km/h (no al effective_limit):
        # ∂R/∂v es 3× más pronunciado en 0-10 km/h que con saturación a 30+,
        # exactamente donde el agente estaba atascado en la run anterior.
        # Por encima de 10 km/h, `speed_reward` Gaussiana (§2) sigue
        # empujando hacia el `effective_limit`.
        if on_road:
            speed_ratio = float(
                np.clip(speed_kmh / self.PROGRESS_SATURATION_KMH, 0.0, 1.0)
            )
            progress_reward = speed_ratio * self.progress_reward_weight
        else:
            progress_reward = 0.0

        # ── 1b. Acceleration reward (señal desde primer km/h ganado) ─
        # Recompensa la transición dv>0, no sólo la velocidad absoluta.
        # Evita el "muro" inicial donde a v=0.2 km/h todavía todo pesa
        # contra el agente. Saturado a 2 km/h/step.
        if on_road:
            delta_v = speed_kmh - self._prev_speed_kmh
            acceleration_reward = (
                float(np.clip(delta_v, 0.0, self.ACCELERATION_DELTA_CAP_KMH))
                * self.acceleration_reward_weight
            )
        else:
            acceleration_reward = 0.0

        # ── 2. Speed reward Gaussiana (ajuste fino cerca del límite) ─
        curvature_magnitude = abs(road_curvature_norm)
        curvature_factor = 1.0 - self.curvature_speed_scale * min(
            curvature_magnitude / 0.6, 1.0
        )
        curve_adjusted_limit = effective_limit * max(curvature_factor, 0.4)

        if on_road and speed_kmh > 0.1:
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

        # ── 3. Lane centering (GATED por has_moved_recently) ─────────
        min_edge_dist = min(dist_left_edge_norm, dist_right_edge_norm)
        centering_score = float(np.clip(min_edge_dist / 0.5, 0.0, 1.0))
        if in_lane_transition:
            lane_centering = 0.3 * self.lane_centering_weight
        elif on_road and has_moved:
            lane_centering = centering_score * self.lane_centering_weight
        else:
            lane_centering = 0.0

        # ── 4. Heading alignment (GATED por has_moved_recently) ──────
        if on_road and has_moved:
            heading_alignment = (
                math.exp(-(heading_error_norm**2) / (2.0 * 0.40**2))
                * self.heading_alignment_weight
            )
        else:
            heading_alignment = 0.0

        # ── 5. Smoothness ────────────────────────────────────────────
        steering_diff = abs(current_steering - self._last_steering)
        smoothness_penalty = steering_diff * self.smoothness_weight

        # ── 6. Lane invasion ────────────────────────────────────────
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

        # ── 7. Edge / off-road (SHAPER) ─────────────────────────────
        # El CarlaEnv base aplica su propio `out_of_road_penalty` en
        # _compute_base_reward. Aquí mantenemos una penalty adicional más
        # suave (default 1.0) que cubre el tramo de borde gradual cuando
        # aún estás on_road pero cerca del límite.
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

        # ── 8. Wrong heading (>90°) ─────────────────────────────────
        abs_heading_deg = abs(heading_error_deg)
        if abs_heading_deg > 90.0:
            wrong_heading_pen = (
                (abs_heading_deg - 90.0) / 90.0 * self.wrong_heading_penalty
            )
        else:
            wrong_heading_pen = 0.0

        # ── 9. Progress milestone ────────────────────────────────────
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

        # ── 10. Idle penalty ESCALONADA (action-gated en tramos bajos) ─
        # Si el agente comanda throttle>0.3 con v<2 km/h (intentando
        # arrancar, la física aún no ha respondido) → atenuamos la penalty
        # al 30%. El agente "pasivo" parado sigue recibiendo 100%.
        if on_road:
            idle_penalty = self._idle_penalty_scaled(
                speed_kmh, self.idle_penalty_weight
            )
            executed_throttle = (
                float(executed_action[1]) if len(executed_action) > 1 else 0.0
            )
            if (
                speed_kmh < self.IDLE_TIER_MID_KMH
                and executed_throttle > self.IDLE_ACTION_THROTTLE_THRESHOLD
            ):
                idle_penalty *= self.IDLE_ACTION_ATTENUATION
        else:
            idle_penalty = 0.0

        # ── 11. Drift asimétrico ────────────────────────────────────
        edge_asymmetry = abs(dist_left_edge_norm - dist_right_edge_norm)
        if in_lane_transition:
            drift_penalty = 0.0
        elif edge_asymmetry > 0.3 and min_edge_dist < 0.35:
            drift_penalty = (edge_asymmetry - 0.3) * self.lane_drift_penalty_weight
        else:
            drift_penalty = 0.0

        # ── 12. Cruce de línea sólida (HARD penalty) ───────────────
        # El LaneInvasionSensor ya filtra a tipos sólidos. Esta penalty
        # se aplica aunque `lane_change_permitted=True` — cruzar una
        # sólida es ilegal incluso en zonas donde el waypoint permite
        # cambios (p. ej. salidas de autopista).
        solid_invasion_pen = (
            self.SOLID_INVASION_PENALTY if lane_invasion else 0.0
        )

        # ── 13. Coste por cambio de carril (debounced) ─────────────
        # Detectamos un cambio cuando `lane_id` cambia de un paso al
        # siguiente. Cooldown de LANE_CHANGE_COOLDOWN_STEPS para evitar
        # contar múltiples triggers del mismo evento físico.
        lane_id = info.get("lane_id", None)
        lane_change_event = False
        if (
            self._last_lane_id is not None
            and lane_id is not None
            and lane_id != self._last_lane_id
            and self._lane_change_cooldown == 0
        ):
            lane_change_event = True
            self._lane_change_cooldown = self.LANE_CHANGE_COOLDOWN_STEPS
        lane_change_cost = self.LANE_CHANGE_COST if lane_change_event else 0.0
        if self._lane_change_cooldown > 0:
            self._lane_change_cooldown -= 1
        if lane_id is not None:
            self._last_lane_id = lane_id

        # ── Suma final ──────────────────────────────────────────────
        shaped_reward = (
            base_reward
            + progress_reward
            + acceleration_reward
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
            - solid_invasion_pen
            - lane_change_cost
        )

        self._last_steering = current_steering
        self._prev_speed_kmh = float(speed_kmh)

        info.update(
            {
                "shaped_reward": shaped_reward,
                "raw_reward": base_reward,
                "alive_bonus": progress_reward,  # retrocompat con logger
                "progress_reward": progress_reward,
                "acceleration_reward": acceleration_reward,
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
                "solid_invasion_penalty": solid_invasion_pen,
                "lane_change_cost": lane_change_cost,
                "lane_change_event": lane_change_event,
                "effective_speed_limit": effective_limit,
                "curve_adjusted_limit": curve_adjusted_limit,
                "centering_score": centering_score,
                "has_moved_recently": has_moved,
                "invasion_intentional": (
                    lane_invasion
                    and abs(current_steering) >= self.INTENTIONAL_STEER_THRESHOLD
                ),
                "in_lane_transition": in_lane_transition,
            }
        )

        return obs, shaped_reward, done, truncated, info
