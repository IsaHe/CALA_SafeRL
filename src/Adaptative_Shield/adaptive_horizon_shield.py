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
  - Recuperación de deadlock: si el coche lleva STALL_PATIENCE pasos parado y
    la situación es demostrablemente segura (frente despejado, sin peatón,
    margen al borde, lateral no crítico), la acción de emergencia pasa de
    frenar a un avance suave + steer hacia el carril, rompiendo el stall
    parado+heading que el brake-only emergency no podía resolver. No debilita
    ninguna respuesta de frenado ante peligro real.
"""

import gymnasium as gym
import numpy as np
import carla
import math
import copy
from typing import Tuple, Dict, Optional

from src.Adaptative_Shield.BicycleModel import BicycleModel


class CarlaAdaptiveHorizonShield(gym.Wrapper):
    """Shield adaptativo con proyección continua y BicycleModel."""

    HORIZON_CONFIG = {
        "safe": {
            "min_dist_threshold": 0.50,
            "horizon": 1,
            "threshold_multiplier": 1.0,
            "lateral_thr": 0.55,
        },
        "warning": {
            "min_dist_threshold": 0.20,
            "horizon": 5,
            "threshold_multiplier": 1.5,
            "lateral_thr": 0.40,
        },
        "critical": {
            "min_dist_threshold": 0.00,
            "horizon": 10,
            "threshold_multiplier": 2.0,
            "lateral_thr": 0.30,
        },
    }

    LATERAL_WARNING_OFFSET: float = 0.50
    LATERAL_CRITICAL_OFFSET: float = 0.70
    EDGE_GUARD_MIN_NORM: float = 0.15

    HEADING_WARNING_DEG: float = 10.0
    HEADING_CRITICAL_DEG: float = 20.0
    MAX_HEADING_DEV_RAD: float = 0.52
    MAX_LATERAL_DRIFT_M: float = 1.75
    LATERAL_RECOVERY_THROTTLE: float = -0.15
    EMERGENCY_STEER_CAP: float = 0.65

    STALL_SPEED_KMH: float = 1.5
    STALL_RECOVERY_EXIT_KMH: float = 5.0
    STALL_PATIENCE: int = 15
    STALL_RECOVERY_THROTTLE: float = 0.30

    PED_EMERGENCY_M: float = 4.0
    BLEND_ALPHAS = (0.25, 0.5, 0.75, 1.0)
    SHIELD_MASK_THRESHOLD = 0.05
    IN_LANE_SAFE_THRESHOLD: float = 0.3

    # Arco frontal ANCHO para vehículos: el cono estrecho ±FRONT_N=15 (±22.5°) del
    # processor PIERDE leads ligeramente fuera de eje (curvas / coche cercano que
    # subtiende >22.5°) — diagnosticado: el obs dinámico veía el lead (0.048) pero
    # min_front_dynamic=1.0, así que el shield no frenaba. Re-medimos el frente
    # dinámico desde el canal dinámico del OBS (obs[240:480], fresco y correcto)
    # sobre ±FRONT_WIDE_N bins. SÓLO dinámico (vehículos/peatones), nunca estático,
    # así que ensanchar NO dispara falsos positivos contra guardarraíles.
    FRONT_WIDE_N: int = 20  # ±30°
    DYN_CH_OFFSET: int = 240  # inicio del canal LIDAR dinámico en el obs (739-dim)
    # Umbral FIJO de freno para el arco ancho dinámico (NO dividido por el
    # multiplier de riesgo). Bug detectado: usar front_threshold_base/multiplier
    # ENCOGÍA el umbral al escalar el riesgo, así que el freno saltaba demasiado
    # tarde (~5 m). Con un umbral fijo ~0.25 (12.5 m ≈ rango de detección del LIDAR
    # sparse) el shield frena en cuanto ve el vehículo. Sólo dinámico -> no afecta
    # a estáticos.
    FRONT_WIDE_BRAKE_NORM: float = 0.25

    def __init__(
        self,
        env,
        num_lidar_rays: int = 240,
        front_threshold_base: float = 0.15,
        side_threshold_base: float = 0.02,
        lane_correction_gain: float = 0.5,
        heading_correction_gain: float = 1.5,
        emergency_brake: float = -0.6,
        meta_tunable: bool = False,
    ):
        super().__init__(env)

        self.num_lidar_rays = num_lidar_rays
        self.front_threshold_base = front_threshold_base
        self.side_threshold_base = side_threshold_base
        self.lane_correction_gain = lane_correction_gain
        self.heading_correction_gain = heading_correction_gain
        self.emergency_brake = emergency_brake

        self.bicycle_model = BicycleModel()
        self._calibrated_vehicle_id: Optional[int] = None

        self.last_obs: Optional[np.ndarray] = None
        self.last_info: Dict = {}
        self.shield_activations = 0
        self._stall_steps = 0
        self._last_emergency_was_recovery = False
        # Bypass total para el probe shield-OFF (mide si el agente conduce solo).
        self._bypass = False

        self.stats = {
            "safe_steps": 0,
            "warning_steps": 0,
            "critical_steps": 0,
            "interventions_dynamic": 0,
            "interventions_static": 0,
            "interventions_pedestrian": 0,
            "interventions_recovery": 0,
            "interventions_by_horizon": {1: 0, 5: 0, 10: 0},
        }

        self._permissiveness = 1.0
        if meta_tunable:
            self.set_permissiveness(1.0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.last_info = info
        self._calibrated_vehicle_id = None
        self._stall_steps = 0
        return obs, info

    def get_permissiveness(self) -> float:
        """Escalar [0,1] de permisividad actual del shield (1.0 = strict)."""
        return float(getattr(self, "_permissiveness", 1.0))

    def set_permissiveness(self, p: float) -> dict:
        """Aplica un escalar de permisividad [0,1] a los umbrales destetables.

        1.0 reproduce el shield strict de produccion; bajar p afloja el shield
        para destetar al agente. Mapea las claves planas/punteadas de
        ``src.meta.tunable_shield`` a atributos de instancia (deep-copiando
        HORIZON_CONFIG la primera vez para no mutar el dict de CLASE compartido
        entre instancias). Solo toca parametros destetables; los invariantes de
        seguridad (peaton, freno duro frontal, EDGE_GUARD, STALL_*) quedan
        intactos. Devuelve el dict de parametros aplicados.
        """
        from src.meta.tunable_shield import (
            clamp_permissiveness,
            permissiveness_to_params,
        )

        p = clamp_permissiveness(p)
        if "HORIZON_CONFIG" not in self.__dict__:
            self.HORIZON_CONFIG = copy.deepcopy(type(self).HORIZON_CONFIG)
        params = permissiveness_to_params(p)
        for name, value in params.items():
            if "." in name:
                _, level, key = name.split(".")
                self.HORIZON_CONFIG[level][key] = value
            else:
                setattr(self, name, value)
        self._permissiveness = p
        return params

    def set_bypass(self, flag: bool) -> None:
        """Activa/desactiva el bypass total del shield (para el probe shield-OFF).

        En bypass, ``step`` pasa la acción propuesta DIRECTAMENTE a CARLA sin
        ninguna verificación ni estadística — es el test puro de si el agente ha
        aprendido a conducir solo. Lo usa ``run_shield_off_probe`` en main_train.
        """
        self._bypass = bool(flag)

    def step(self, action: np.ndarray):
        if getattr(self, "_bypass", False):
            proposed = np.asarray(action, dtype=np.float32).copy()
            obs, reward, done, truncated, info = self.env.step(proposed)
            self.last_obs = obs
            self.last_info = info
            info.update(
                {
                    "shield_activated": False,
                    "shield_intensity": 0.0,
                    "executed_action": proposed,
                    "proposed_action": proposed,
                    "shield_bypassed": True,
                }
            )
            return obs, reward, done, truncated, info

        sem_analysis = self._analyze_semantic(self.last_obs, self.last_info)
        risk_level, _ = self._get_risk_level_semantic(sem_analysis)
        horizon = self.HORIZON_CONFIG[risk_level]["horizon"]
        self.stats[f"{risk_level}_steps"] += 1

        carla_map = self._get_carla_map()
        ego = self._get_ego_vehicle()
        self._calibrate_bicycle_model(ego)
        self._update_stall_counter(ego)

        proposed = np.asarray(action, dtype=np.float32).copy()

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
                if self._last_emergency_was_recovery:
                    self.stats["interventions_recovery"] += 1

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

    def _wide_front_dynamic(self, obs: np.ndarray) -> float:
        """Mínimo del canal LIDAR dinámico del OBS en el arco frontal ANCHO
        (±FRONT_WIDE_N). Usa el obs (fresco/correcto), no el escalar
        min_front_dynamic del info (que perdía leads fuera de eje). Sólo dinámico
        (vehículos/peatones) -> ensanchar no toca estáticos/guardarraíles."""
        n = self.num_lidar_rays
        off = self.DYN_CH_OFFSET
        if obs is None or len(obs) < off + n:
            return 1.0
        dyn = obs[off : off + n]
        w = self.FRONT_WIDE_N
        return float(min(dyn[n - w :].min(), dyn[:w].min()))

    def _analyze_semantic(self, obs: np.ndarray, info: Dict) -> Dict:
        n = self.num_lidar_rays
        if "min_front_dynamic" in info:
            wide = self._wide_front_dynamic(obs)
            # Riesgo frontal = el MÁS conservador entre el cono estrecho del info y
            # el arco ancho re-medido del obs (captura leads fuera de eje).
            risk_dist = min(info["min_front_dynamic"], wide)
            return {
                "min_front_combined": info["min_front_combined"],
                "min_front_dynamic": info["min_front_dynamic"],
                "min_front_dynamic_wide": wide,
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
                "min_dist_for_risk": risk_dist,
                "has_semantics": True,
            }

        scan = obs[:n]
        front = np.concatenate((scan[n - 15 :], scan[:15]))
        r_s = scan[40:80]
        l_s = scan[160:200]
        mf = float(front.min())
        wide = min(mf, self._wide_front_dynamic(obs))
        return {
            "min_front_combined": mf,
            "min_front_dynamic": mf,
            "min_front_dynamic_wide": wide,
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
            "min_dist_for_risk": wide,
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
        heading_abs = abs(self.last_info.get("heading_error", 0.0))
        if (
            lat_norm > self.LATERAL_CRITICAL_OFFSET
            or heading_abs > self.HEADING_CRITICAL_DEG
        ):
            lateral_level = "critical"
        elif (
            lat_norm > self.LATERAL_WARNING_OFFSET
            or heading_abs > self.HEADING_WARNING_DEG
        ):
            lateral_level = "warning"
        else:
            lateral_level = "safe"

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

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

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
        lat_thr = self.HORIZON_CONFIG[risk_level]["lateral_thr"]

        if analysis["nearest_pedestrian_m"] < self.PED_EMERGENCY_M:
            return False

        if analysis["min_front_combined"] < front_thr:
            return False

        # Vehículo dinámico cercano FUERA del cono estrecho (off-axis): el
        # min_front_combined estrecho lo perdía y el ego lo alcanzaba. El arco
        # ancho es sólo-dinámico, así que esto no frena por guardarraíles. Umbral
        # FIJO (no /multiplier) para frenar en cuanto se detecta, no al límite.
        if analysis.get("min_front_dynamic_wide", 1.0) < self.FRONT_WIDE_BRAKE_NORM:
            return False

        dist_left = self.last_info.get("dist_left_edge_norm", 1.0)
        dist_right = self.last_info.get("dist_right_edge_norm", 1.0)
        in_lane_safely = (
            not self._is_lane_change_context()
            and dist_left > self.IN_LANE_SAFE_THRESHOLD
            and dist_right > self.IN_LANE_SAFE_THRESHOLD
        )

        if not in_lane_safely:
            if analysis["min_r_side_static"] < side_thr:
                return False
            if analysis["min_l_side_static"] < side_thr:
                return False

        current_min_edge = min(
            self.last_info.get("dist_left_edge_norm", 1.0),
            self.last_info.get("dist_right_edge_norm", 1.0),
        )
        if current_min_edge < self.EDGE_GUARD_MIN_NORM:
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

        ref_wp = carla_map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        ref_right = ref_wp.transform.get_right_vector() if ref_wp is not None else None
        ref_x = transform.location.x
        ref_y = transform.location.y
        max_heading_dev = self.MAX_HEADING_DEV_RAD / multiplier
        max_drift = self.MAX_LATERAL_DRIFT_M / multiplier

        for px, py, pyaw in trajectory[1:]:
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
            if abs(lat_offset) / lane_half > lat_thr:
                return False

            lane_yaw = math.radians(wp.transform.rotation.yaw)
            if abs(self._wrap_angle(pyaw - lane_yaw)) > max_heading_dev:
                return False
            if ref_right is not None:
                ddx = px - ref_x
                ddy = py - ref_y
                if abs(ddx * ref_right.x + ddy * ref_right.y) > max_drift:
                    return False

        return True

    def _build_emergency_action(self, analysis: Dict) -> np.ndarray:
        self._last_emergency_was_recovery = False
        lat_norm = self.last_info.get("lateral_offset_norm", 0.0)
        heading_err_rad = math.radians(self.last_info.get("heading_error", 0.0))

        correction = (
            self.heading_correction_gain * heading_err_rad
            + self.lane_correction_gain * lat_norm
        )
        cap = self.EMERGENCY_STEER_CAP
        steer_target = float(np.clip(-correction, -cap, cap))

        if analysis["min_l_side_static"] < self.side_threshold_base:
            steer_target = float(np.clip(steer_target + 0.4, -1.0, 1.0))
        if analysis["min_r_side_static"] < self.side_threshold_base:
            steer_target = float(np.clip(steer_target - 0.4, -1.0, 1.0))

        front = analysis["min_front_combined"]
        if front < self.front_threshold_base * 0.5:
            tb_target = -1.0
        elif (
            front < self.front_threshold_base
            or abs(lat_norm) > self.LATERAL_CRITICAL_OFFSET
        ):
            tb_target = self.emergency_brake
        else:
            min_edge = min(
                self.last_info.get("dist_left_edge_norm", 1.0),
                self.last_info.get("dist_right_edge_norm", 1.0),
            )
            deadlock_recover = (
                self._stall_steps >= self.STALL_PATIENCE
                and min_edge >= self.EDGE_GUARD_MIN_NORM
                and analysis["nearest_pedestrian_m"] >= self.PED_EMERGENCY_M
            )
            self._last_emergency_was_recovery = deadlock_recover
            tb_target = (
                self.STALL_RECOVERY_THROTTLE
                if deadlock_recover
                else self.LATERAL_RECOVERY_THROTTLE
            )

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

    def _get_carla_map(self) -> Optional[carla.Map]:
        env = self.env
        while env is not None:
            if hasattr(env, "map") and env.map is not None:
                return env.map
            env = getattr(env, "env", None)
        return None

    def _calibrate_bicycle_model(self, ego: Optional[carla.Vehicle]) -> None:
        if ego is None:
            return
        try:
            vid = ego.id
        except Exception:
            return
        if vid == self._calibrated_vehicle_id:
            return
        try:
            physics = ego.get_physics_control()
            wheels = getattr(physics, "wheels", None)
            if not wheels:
                return
            max_angle_deg = max(float(w.max_steer_angle) for w in wheels)
            if max_angle_deg <= 0.0:
                return
            self.bicycle_model.set_max_steer_rad(math.radians(max_angle_deg))
            self._calibrated_vehicle_id = vid
        except Exception:
            return

    def _get_ego_vehicle(self) -> Optional[carla.Vehicle]:
        env = self.env
        while env is not None:
            if hasattr(env, "ego_vehicle") and env.ego_vehicle is not None:
                return env.ego_vehicle
            env = getattr(env, "env", None)
        return None

    def _update_stall_counter(self, ego: Optional[carla.Vehicle]) -> None:
        """Cuenta pasos consecutivos con el coche prácticamente parado.

        Alimenta la recuperación de deadlock en `_build_emergency_action`
        (ver constantes STALL_*). Usa la velocidad viva del ego; si no está
        disponible, cae a `speed_kmh` del último info.
        """
        speed_kmh = None
        if ego is not None:
            try:
                v = ego.get_velocity()
                speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            except Exception:
                speed_kmh = None
        if speed_kmh is None:
            speed_kmh = float(self.last_info.get("speed_kmh", 999.0))

        if speed_kmh < self.STALL_SPEED_KMH:
            self._stall_steps += 1
        elif speed_kmh >= self.STALL_RECOVERY_EXIT_KMH:
            self._stall_steps = 0

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
            "interventions_recovery": self.stats["interventions_recovery"],
            "recovery_activations": self.stats["interventions_recovery"],
            "recovery_rate": self.stats["interventions_recovery"] / total,
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
            "interventions_recovery": 0,
        }
        self.shield_activations = 0
