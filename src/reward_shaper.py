"""
reward_shaper.py - Reward shaping monótono sin bolsillo de reposo.

DISEÑO (sesión 3):
  El paisaje de reward debe tener un **óptimo local NEGATIVO en reposo**
  para que el agente no encuentre refugio quedándose parado. La pendiente
  ∂R/∂v debe ser positiva desde el primer paso de movimiento.

  Señal tomada del LÍMITE DINÁMICO de carretera (`speed_limit_kmh` del
  Waypoint API) en vez de un target fijo: así el agente aprende a adaptar
  su velocidad al contexto.

SOLID_INVASION DECAY GEOMÉTRICO (sesión 8 — fix varianza crítico):
  Antes: cada cruce de línea sólida pagaba -5.0 plano. Sin tope. En la
  run shieldIdleFix vimos un episodio con -185 (37 cruces) y en
  safetyGated -135 (27 cruces). Esos outliers dominaban el value_loss
  del PPO (tail mean 0.023-0.030; p99 4x el mean) y degradaban la
  estimación de V(s) en el resto del rollout.

  Ahora: el k-ésimo cruce del MISMO episodio paga 5.0 · 0.5^k. Suma
  asintótica = 10.0 (vs los -185 observados). Conserva el incentivo
  fuerte del primer cruce. Reset por episodio.

  El conteo de eventos crudos (sin penalty) se expone como
  `solid_invasion_event` en info → loggeado como
  `Reward/Components/Solid_Invasion_Events` para poder distinguir
  "1 cruce, -5.0" de "10 cruces, -9.99" desde las métricas.

MULTIPLICATIVE LANE-KEEP GATE (sesión 9 — fix offroad 73 %):
  La run roadEdgeDetection_adaptive_20260525-181247 cerró con 73 %
  offroad sostenido durante 307 episodios; análisis del log
  (`docs/diag_offroad_73pct.md`) identificó como causa estructural que
  el `safety_gate` cuadrático sólo recortaba cuando `min_edge < 0.5`,
  i.e. |lat_off_norm| > 0.5 (~0.88 m en un carril de 3.5 m). Para
  entonces el agente ya no puede recuperar: el ruido de steering (σ
  saturado a LOG_STD_MAX=-1.5) se come el margen.

  Sustituimos el gate cuadrático escalar por DOS factores lineales que
  se aplican como PRODUCTO sobre los positivos densos:

      centering_factor = max(1 - |lat_off_norm|, 0)
      heading_factor   = max(1 - |heading_err_deg| / 20, 0)
      lane_keep_mult   = centering_factor × heading_factor

      r_dense = (progress + accel + max(speed_bonus, 0)) × lane_keep_mult
              + min(speed_bonus, 0)    # overspeed sigue pasando íntegro

  Replica el diseño multiplicativo de bitsauce/Carla-ppo
  (`reward = speed_reward × centering × angle`) y CuRLA (arXiv
  2501.04982: `r = r_α · r_d · r_v + r_c`).

  Por qué lineal en vez de cuadrática:
    - cuadrática sólo presiona cerca del centro; en el rango carril-
      medio (lat_off≈0.5) era ineficaz.
    - lineal da gradiente CONSTANTE hacia el centro desde |lat_off|=0:
      cada 0.1 de drift cuesta 10 % del positivo denso. La política
      tiene señal incluso en pequeñas desviaciones.

  Por qué añadir heading_factor:
    - El gate antiguo sólo miraba posición. La política aprendió a
      "ir centrada pero cruzada" (serpenteo): el coche pasaba por el
      centro del carril con heading_err alto → progress reward pleno
      a pesar de ir hacia el borde. Heading_factor mata el reward
      cuando |heading| > 20° (umbral de bitsauce/CuRLA) — bloquea
      ese modo.

  Lane_centering y heading_alignment (componentes aditivos) también
  ADOPTAN los nuevos factores como score lineal en sustitución del
  centering_score y la Gaussiana antiguos. Esto refuerza la señal en
  la misma dirección sin doble-contar (uno paga por estar centrado,
  el otro penaliza no estarlo en el resto de positivos).

  Durante `in_lane_transition` el producto se piso (floor) a 0.3
  (`SAFETY_GATE_TRANSITION_FLOOR`); los componentes aditivos también
  se floorean a 0.3 × weight. Conserva la semántica de "no penalizar
  cambios de carril permitidos" del diseño antiguo.

  Backward compat: `info["safety_gate"]` se mantiene como alias de
  `lane_keep_multiplier` para que los dashboards históricos sigan
  graficando una serie equivalente.

Componentes (11 — sesión 4 añade acceleration_reward):
  + progress_reward      (lineal satura a 10 km/h; weight 0.30) · safety-gated
  + acceleration_reward  (clip(Δv, 0, 2) · weight 0.08; señal densa desde v=0) · safety-gated
  + speed_reward         (Gaussiana centrada en el límite dinámico) · safety-gated
  + lane_centering       (gated por has_moved_recently; off si está parado ≥5 pasos)
  + heading_alignment    (gated por has_moved_recently) · safety-gated
  + progress_milestone   (bonus puntual cada 25 m)
  - smoothness_penalty   (|Δsteering|)
  - invasion_pen         (cruce no intencional)
  - road_penalty         (borde gradual O off_road puntual, ajustado a 1.0)
  - wrong_heading_pen    (heading > 90°)
  - idle_penalty         (ESCALONADA · atenuada 0.3 si throttle>0.3 y v<2)
  - drift_penalty        (asimetría lateral con bordes cerca)

IDLE PENALTY ESCALONADA (pico 0.25):
    speed < 0.5 km/h          → -idle_weight              = -0.25
    0.5 ≤ speed < 2 km/h      → -idle_weight * 0.5        = -0.125
    2 ≤ speed < 5 km/h        → -idle_weight * 0.2        = -0.05
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
    #
    # DECAY GEOMÉTRICO (sesión 8 — fix varianza crítico):
    # Cada evento adicional dentro del mismo episodio paga
    # SOLID_INVASION_PENALTY · SOLID_INVASION_DECAY^N (N = eventos ya
    # contabilizados). Con DECAY=0.5 la suma asintótica de un episodio
    # converge a 2 · SOLID_INVASION_PENALTY = 10.0 (vs 185 observado en
    # la run shieldIdleFix y 135 en safetyGated, que dominaban el
    # value_loss del PPO). Reduce varianza terminal del crítico sin
    # eliminar el incentivo del primer cruce.
    SOLID_INVASION_PENALTY: float = 5.0
    SOLID_INVASION_DECAY: float = 0.5

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

    # Floor del `safety_gate` durante transiciones legales de carril.
    # Mismo valor que el multiplicador fijo que usa lane_centering durante
    # `in_lane_transition` — el agente sigue cobrando ~30% del shaping
    # denso para no penalizar cambios de carril permitidos.
    SAFETY_GATE_TRANSITION_FLOOR: float = 0.3

    def __init__(
        self,
        env,
        target_speed_kmh: float = 30.0,  # fallback cuando no hay speed_limit válido
        speed_weight: float = 0.10,
        smoothness_weight: float = 0.10,
        lane_centering_weight: float = 0.15,
        heading_alignment_weight: float = 0.04,
        lane_invasion_penalty: float = 0.25,
        off_road_penalty: float = 1.00,  # bajado: el env ya penaliza -8 en base
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
        # Contador acumulado de cruces de línea sólida en el episodio actual,
        # usado para aplicar el decay geométrico de SOLID_INVASION_PENALTY.
        self._solid_invasion_event_count = 0

    def reset(self, **kwargs):
        self._last_steering = 0.0
        self._last_milestone = 0.0
        self._prev_speed_kmh = 0.0
        self._recent_speed_window.clear()
        self._last_lane_id = None
        self._lane_change_cooldown = 0
        self._solid_invasion_event_count = 0
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

    @staticmethod
    def _lane_keep_factors(
        lateral_offset_norm: float,
        heading_error_deg: float,
    ) -> tuple:
        """Factores multiplicativos [0, 1] para gating lineal de positivos densos.

        Sustituye al `_safety_gate` cuadrático (ver docstring de cabecera).

        centering_factor = max(1 - |lat_off_norm|, 0)
            1 en el centro EXACTO del carril, 0 en el borde. Lineal.
        heading_factor   = max(1 - |heading_err_deg| / 20, 0)
            1 alineado con el carril, 0 a |20°| de desvío. Lineal.

        Estos factores se aplican como PRODUCTO sobre los positivos densos
        (progress, accel, speed_bonus_pos) y como SCORE LINEAL sobre los
        componentes aditivos lane_centering y heading_alignment.

        El umbral de 20° en heading_factor está tomado de
        bitsauce/Carla-ppo y CuRLA (arXiv 2501.04982). Por debajo de ese
        ángulo la corrección con steering normal es viable; por encima,
        el coche va cruzado y no debe cobrar reward de avance.

        Esta función NO aplica el floor de in_lane_transition: el caller
        lo hace al combinar los factores (sobre el producto y sobre cada
        componente aditivo por separado).
        """
        centering = max(1.0 - abs(float(lateral_offset_norm)), 0.0)
        heading = max(1.0 - abs(float(heading_error_deg)) / 20.0, 0.0)
        return centering, heading

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
        # heading_error_norm (Gaussiana) ya no se usa: heading_alignment
        # ahora es lineal en heading_factor (sesión 9). Sólo necesitamos los
        # grados absolutos.
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

        # ── Detección de transición de carril ────────────────────────
        # (computada ANTES de los positivos para poder pasar a _safety_gate)
        lane_change_permitted = info.get("lane_change_permitted", False)
        in_lane_transition = lane_change_permitted and abs(lateral_offset_norm) > 0.5

        # ── Lane-keep factors (sesión 9 — multiplicativo) ────────────
        # Reemplaza al safety_gate cuadrático con dos factores lineales
        # (centering, heading) que se aplican como PRODUCTO. Ver docstring
        # de cabecera del módulo para la justificación completa.
        centering_factor, heading_factor = self._lane_keep_factors(
            lateral_offset_norm, heading_error_deg
        )
        if in_lane_transition:
            # Floor explícito sobre el PRODUCTO: el coche puede ir cruzado
            # y descentrado durante una transición legal, y aún así cobra
            # el 30 % del pump positivo. Conserva la semántica antigua.
            lane_keep_mult = self.SAFETY_GATE_TRANSITION_FLOOR
        else:
            lane_keep_mult = centering_factor * heading_factor

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
            progress_reward = speed_ratio * self.progress_reward_weight * lane_keep_mult
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
                * lane_keep_mult
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
            # Sólo el bonus positivo se gatea por lane_keep_mult: el
            # sobrecoste por overspeed debe seguir penalizando aunque el
            # agente esté centrado.
            speed_reward = max(speed_reward, 0.0) * lane_keep_mult + min(
                speed_reward, 0.0
            )
        else:
            speed_reward = 0.0

        # ── 3. Lane centering (LINEAR en centering_factor, sesión 9) ──
        # Antes usaba `centering_score = min(min_edge/0.5, 1)` (plateau
        # en el centro). Ahora reutiliza el centering_factor lineal del
        # producto multiplicativo: presiona hacia el centro EXACTO sin
        # plateau, mismo gradiente que el factor multiplicativo.
        if in_lane_transition:
            lane_centering = (
                self.SAFETY_GATE_TRANSITION_FLOOR * self.lane_centering_weight
            )
        elif on_road and has_moved:
            lane_centering = centering_factor * self.lane_centering_weight
        else:
            lane_centering = 0.0

        # ── 4. Heading alignment (LINEAR en heading_factor, sesión 9) ─
        # Antes Gaussiana sobre heading_error_norm (muy plana: a 20° aún
        # paga ~96 % del bonus). Ahora lineal en heading_factor: a 20°
        # paga 0. Bloquea el modo "centrado pero cruzado" que la run
        # roadEdgeDetection mostraba como ruta hacia offroad.
        if in_lane_transition:
            heading_alignment = (
                self.SAFETY_GATE_TRANSITION_FLOOR * self.heading_alignment_weight
            )
        elif on_road and has_moved:
            heading_alignment = heading_factor * self.heading_alignment_weight
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

        # ── 10. Idle penalty ESCALONADA (action-gated + shield-attenuated) ─
        # Atenuación dual:
        #   (a) Si el agente comanda throttle>0.3 con v<2 km/h (arrancando,
        #       la física aún no ha respondido) → penalty al 30 %.
        #   (b) Si el shield está interviniendo con α>0, la idle_penalty se
        #       reescala por (1−α): el shield comandó el freno, el agente
        #       no debe ser penalizado por la acción que no eligió. Esto es
        #       simétrico con `mask_unshielded = 1−α` que PPOAgent aplica a
        #       policy/entropy loss, y elimina el shield-trap × idle_penalty
        #       observado en runs con `mean_α ≳ 0.4` y `corr(α,idle)=0.99`.
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
            shield_alpha = float(info.get("shield_intensity", 0.0))
            idle_penalty *= max(0.0, 1.0 - shield_alpha)
        else:
            idle_penalty = 0.0

        # ── 11. Drift asimétrico ────────────────────────────────────
        # `min_edge_dist` se calcula localmente: ya no se reutiliza desde
        # lane_centering (que ahora usa centering_factor lineal sobre
        # lat_off_norm, no la distancia al borde).
        min_edge_dist = min(dist_left_edge_norm, dist_right_edge_norm)
        edge_asymmetry = abs(dist_left_edge_norm - dist_right_edge_norm)
        if in_lane_transition:
            drift_penalty = 0.0
        elif edge_asymmetry > 0.3 and min_edge_dist < 0.35:
            drift_penalty = (edge_asymmetry - 0.3) * self.lane_drift_penalty_weight
        else:
            drift_penalty = 0.0

        # ── 12. Cruce de línea sólida (HARD penalty con decay) ─────
        # El LaneInvasionSensor ya filtra a tipos sólidos. Esta penalty
        # se aplica aunque `lane_change_permitted=True` — cruzar una
        # sólida es ilegal incluso en zonas donde el waypoint permite
        # cambios (p. ej. salidas de autopista).
        #
        # Decay geométrico (sesión 8): el k-ésimo cruce del MISMO
        # episodio paga SOLID_INVASION_PENALTY · SOLID_INVASION_DECAY^k
        # (k empieza en 0). Acota la varianza del retorno terminal sin
        # eliminar el incentivo del primer cruce.
        if lane_invasion:
            solid_invasion_pen = self.SOLID_INVASION_PENALTY * (
                self.SOLID_INVASION_DECAY**self._solid_invasion_event_count
            )
            self._solid_invasion_event_count += 1
            solid_invasion_event = True
        else:
            solid_invasion_pen = 0.0
            solid_invasion_event = False

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
                "solid_invasion_event": 1 if solid_invasion_event else 0,
                "lane_change_cost": lane_change_cost,
                "lane_change_event": lane_change_event,
                "effective_speed_limit": effective_limit,
                "curve_adjusted_limit": curve_adjusted_limit,
                # Centering_score legacy: ahora se publica centering_factor
                # (lineal en |lat_off|) en su lugar. Se conserva la clave
                # con el nuevo valor para que consumidores históricos no
                # rompan.
                "centering_score": centering_factor,
                "centering_factor": centering_factor,
                "heading_factor": heading_factor,
                "lane_keep_multiplier": lane_keep_mult,
                # Backward compat: dashboards históricos siguen graficando
                # `safety_gate`; lo aliasamos al nuevo lane_keep_mult que
                # juega el mismo papel semántico (escalar [0,1] de gating).
                "safety_gate": lane_keep_mult,
                "has_moved_recently": has_moved,
                "invasion_intentional": (
                    lane_invasion
                    and abs(current_steering) >= self.INTENTIONAL_STEER_THRESHOLD
                ),
                "in_lane_transition": in_lane_transition,
            }
        )

        return obs, shaped_reward, done, truncated, info
