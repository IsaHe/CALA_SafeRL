"""
safety_shield.py - Basic Safety Shield nativo de CARLA

Reemplaza safety_shield_improved.py + lane_aware_basic_shield.py.

MEJORAS vs versión MetaDrive:
  1. Los límites de carril vienen del Waypoint API (exactos), no de la posición Y.
  2. La corrección de carril usa el offset lateral real en metros.
  3. Usa el sensor de lane invasion de CARLA para detección temprana.
  4. Principio de mínima interferencia: solo corrige lo estrictamente necesario.
  5. NO penaliza reward al activarse (el agente aprende sobre las acciones reales).

ARQUITECTURA (gym.Wrapper):
  CarlaSafetyShield
    └── CarlaRewardShaper
          └── CarlaEnv

El shield trabaja con:
  - obs[:num_lidar_rays] → scan LIDAR para detección de obstáculos
  - info["lateral_offset_norm"] → posición lateral exacta (Waypoint API)
  - info["heading_error_norm"]  → error angular exacto (Waypoint API)
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict


class CarlaSafetyShield(gym.Wrapper):
    """
    Shield de seguridad básico con dos capas de protección:

    CAPA 1 - Obstáculos (LIDAR):
      Detecta vehículos y objetos por delante/laterales.
      Corrige acelerando/frenando y ajustando steering.

    CAPA 2 - Límites de carril (Waypoint API CARLA):
      Detecta cuando el vehículo se acerca al borde o sale del carril.
      Corrige el steering para volver al centro.
      Esta capa es el diferencial clave vs MetaDrive: usa datos exactos.
    """

    def __init__(
        self,
        env,
        num_lidar_rays: int = 240,
        front_threshold: float = 0.15,
        side_threshold: float = 0.04,
        lateral_threshold: float = 0.82,  # fracción del semi-ancho → ~1.5m para carril 3.5m
        heading_threshold: float = 0.60,  # ~108° de error máximo antes de corregir
        k_steer: float = 2.5,
        k_brake: float = 8.0,
    ):
        """
        Args:
            front_threshold  : Distancia LIDAR normalizada bajo la cual frenar
            side_threshold   : Distancia LIDAR normalizada lateral de peligro
            lateral_threshold: |lateral_offset_norm| máximo antes de corregir carril
            heading_threshold: |heading_error_norm| máximo antes de corregir heading
            k_steer          : Ganancia de corrección de steering
            k_brake          : Ganancia de corrección de frenado
        """
        super().__init__(env)

        self.num_lidar_rays = num_lidar_rays
        self.front_threshold = front_threshold
        self.side_threshold = side_threshold
        self.lateral_threshold = lateral_threshold
        self.heading_threshold = heading_threshold
        self.k_steer = k_steer
        self.k_brake = k_brake

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

    # ══════════════════════════════════════════════════════════════════
    # GYMNASIUM API
    # ══════════════════════════════════════════════════════════════════

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        self.last_info = info
        return obs, info

    def step(self, action: np.ndarray):
        """Ejecuta un paso con protección de seguridad de dos capas."""
        lidar_scan = self._get_lidar(self.last_obs)
        lidar_analysis = self._analyze_lidar(lidar_scan)
        lat_norm = self.last_info.get("lateral_offset_norm", 0.0)
        head_norm = self.last_info.get("heading_error_norm", 0.0)

        # ── Verificación de seguridad ──────────────────────────────────
        obstacle_unsafe, obs_reason = self._check_obstacle(lidar_analysis)
        lane_unsafe, lane_reason = self._check_lane(lat_norm, head_norm)

        needs_shield = obstacle_unsafe or lane_unsafe

        if needs_shield:
            final_action = self._compute_correction(lidar_analysis, lat_norm, head_norm)
            self.shield_activations += 1
            reason = obs_reason if obstacle_unsafe else lane_reason
        else:
            final_action = action.copy()
            reason = "none"

        # ── Ejecutar en entorno ────────────────────────────────────────
        obs, reward, done, truncated, info = self.env.step(final_action)

        # Actualizar estado para el siguiente step
        self.last_obs = obs
        self.last_info = info

        # Enriquecer info
        info.update(
            {
                "shield_activated": needs_shield,
                "shield_reason": reason,
                "min_front_dist": lidar_analysis["min_front"],
                "min_r_side_dist": lidar_analysis["min_r_side"],
                "min_l_side_dist": lidar_analysis["min_l_side"],
                "min_distance": lidar_analysis["min_dist"],
                "total_shield_activations": self.shield_activations,
                "executed_action": final_action,
                "proposed_action": action,
                "shield_modified_action": needs_shield,
            }
        )

        return obs, reward, done, truncated, info

    # ══════════════════════════════════════════════════════════════════
    # ANÁLISIS LIDAR
    # ══════════════════════════════════════════════════════════════════

    def _get_lidar(self, obs: np.ndarray) -> np.ndarray:
        """Extrae el scan LIDAR de los primeros N elementos del obs."""
        return obs[: self.num_lidar_rays]

    def _analyze_lidar(self, scan: np.ndarray) -> Dict:
        """
        Analiza el scan en sectores angulares compatibles con MetaDrive.

        Con 240 rayos (1.5°/rayo):
          Frente      : ±22.5° → índices [-15:] + [:15]
          Lado derecho: 60-120° → índices [40:80]
          Lado izq.   : 240-300° → índices [160:200]
        """
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

    # ══════════════════════════════════════════════════════════════════
    # VERIFICACIÓN DE SEGURIDAD
    # ══════════════════════════════════════════════════════════════════

    def _check_obstacle(self, analysis: Dict) -> Tuple[bool, str]:
        """Verifica seguridad ante obstáculos con el scan LIDAR."""
        if analysis["min_front"] < self.front_threshold:
            return True, "front_obstacle"
        if analysis["min_r_side"] < self.side_threshold:
            return True, "right_obstacle"
        if analysis["min_l_side"] < self.side_threshold:
            return True, "left_obstacle"
        return False, "safe"

    def _check_lane(self, lat_norm: float, head_norm: float) -> Tuple[bool, str]:
        """
        Verifica seguridad de límites de carril.

        Usa datos del Waypoint API de CARLA (exactos al centímetro)
        en lugar de la heurística de posición Y de MetaDrive.
        """
        if abs(lat_norm) > self.lateral_threshold:
            return True, "lane_boundary"
        if abs(head_norm) > self.heading_threshold:
            return True, "heading_error"
        return False, "safe"

    # ══════════════════════════════════════════════════════════════════
    # CORRECCIÓN DE ACCIÓN
    # ══════════════════════════════════════════════════════════════════

    def _compute_correction(
        self,
        analysis: Dict,
        lat_norm: float,
        head_norm: float,
    ) -> np.ndarray:
        """
        Calcula la acción correctiva mínima necesaria.

        Prioridad:
          1. Frenar si hay obstáculo frontal
          2. Corregir posición lateral (Waypoint API)
          3. Corregir heading angular (Waypoint API)
          4. Evitar obstáculos laterales (LIDAR)
        """
        steering = 0.0
        throttle_brake = 0.0

        # ── 1. Frenazo frontal ─────────────────────────────────────────
        if analysis["min_front"] < self.front_threshold:
            danger = (
                self.front_threshold - analysis["min_front"]
            ) / self.front_threshold
            brake = -1.0 * (danger * self.k_brake)
            throttle_brake = float(np.clip(brake, -1.0, 0.0))
            self.intervention_stats["front"] += 1

        # ── 2. Corrección lateral (CARLA Waypoint API) ─────────────────
        # lat_norm > 0 → demasiado a la derecha → girar izquierda (steering negativo)
        if abs(lat_norm) > self.lateral_threshold * 0.75:
            lane_corr = -lat_norm * self.k_steer
            steering += lane_corr
            if lat_norm > 0:
                self.intervention_stats["lane_right"] += 1
            else:
                self.intervention_stats["lane_left"] += 1

        # ── 3. Corrección de heading (CARLA Waypoint API) ──────────────
        # head_norm > 0 → girado a la derecha respecto al carril → corregir a izquierda
        if abs(head_norm) > self.heading_threshold * 0.70:
            head_corr = -head_norm * self.k_steer * 0.45
            steering += head_corr
            self.intervention_stats["heading"] += 1

        # ── 4. Evasión de obstáculos laterales (LIDAR) ─────────────────
        if analysis["min_r_side"] < self.side_threshold:
            push = (self.side_threshold - analysis["min_r_side"]) * self.k_steer
            steering += push  # Empujar hacia izquierda
            self.intervention_stats["side_right"] += 1

        if analysis["min_l_side"] < self.side_threshold:
            push = (self.side_threshold - analysis["min_l_side"]) * self.k_steer
            steering -= push  # Empujar hacia derecha
            self.intervention_stats["side_left"] += 1

        return np.array(
            [
                float(np.clip(steering, -1.0, 1.0)),
                float(np.clip(throttle_brake, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
