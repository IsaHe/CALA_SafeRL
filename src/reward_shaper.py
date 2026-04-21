import gymnasium as gym
import numpy as np
import math


class CarlaRewardShaper(gym.Wrapper):
    """
    Two-level priority hierarchy reward shaper for CarlaEnv.

    Inspired by the lexicographic MORL architecture from Deshpande et al.
    (2021), "Navigation in Urban Environments amongst Pedestrians using
    Multi-Objective Deep RL". We emulate lexicographic dominance in a
    single scalar reward by clamping the efficiency/comfort contribution
    per step so that any triggered safety penalty mathematically
    outweighs the best possible efficiency reward in the same step.

    Priority hierarchy (6 core features total):
      Level 1 — Safety (dominant, unbounded negative):
        S1. Off-road / edge proximity
        S2. Lane invasion
        S3. Shield intervention
      Level 2 — Efficiency & comfort (subordinate, clipped to ±cap):
        E1. Speed tracking
        E2. Forward progress
        E3. Comfort (lane centering + steering smoothness)
    """

    PROGRESS_MILESTONE_M: float = 25.0
    INTENTIONAL_STEER_THRESHOLD: float = 0.25

    def __init__(
        self,
        env,
        target_speed_kmh: float = 30.0,
        # ── Level 2: Efficiency / Comfort (clipped to ±efficiency_cap) ─
        speed_weight: float = 0.10,
        progress_weight: float = 0.40,
        comfort_weight: float = 0.08,
        efficiency_cap: float = 0.20,
        # ── Level 1: Safety (every full-strength trigger > cap) ───────
        safety_edge_weight: float = 0.40,
        lane_invasion_penalty: float = 0.35,
        off_road_penalty: float = 2.00,
        off_road_penalty_k: float = 4.0,
        shield_intervention_penalty: float = 0.25,
        # ── Misc ──────────────────────────────────────────────────────
        speed_limit_margin: float = 0.05,
        curvature_speed_scale: float = 0.4,
        min_moving_speed_kmh: float = 5.0,
        speed_gate_full_kmh: float = 10.0,
        max_steps: int = 1000,
    ):
        """
        Args:
            target_speed_kmh            : Target cruising speed (km/h)
            speed_weight                : Scale of speed bonus (E1)
            progress_weight             : Milestone bonus amount (E2)
            comfort_weight              : Scale of centering/smoothness (E3)
            efficiency_cap              : Absolute cap on per-step Level-2 reward.
                                          Guarantees lexicographic dominance.
            safety_edge_weight          : Scale of graded edge proximity (S1)
            lane_invasion_penalty       : Full-strength lane invasion cost (S2)
            off_road_penalty            : Base off-road cost (S1)
            off_road_penalty_k          : Early-termination shame multiplier
            shield_intervention_penalty : Full-strength shield intervention cost (S3)
            curvature_speed_scale       : How much curves reduce target speed
        """
        super().__init__(env)

        self.target_speed_kmh = target_speed_kmh

        # Level 2 weights
        self.speed_weight = speed_weight
        self.progress_weight = progress_weight
        self.comfort_weight = comfort_weight
        self.efficiency_cap = efficiency_cap

        # Level 1 weights
        self.safety_edge_weight = safety_edge_weight
        self.lane_invasion_penalty = lane_invasion_penalty
        self.off_road_penalty = off_road_penalty
        self.off_road_penalty_k = off_road_penalty_k
        self.shield_intervention_penalty = shield_intervention_penalty

        # Misc
        self.speed_limit_margin = speed_limit_margin
        self.curvature_speed_scale = curvature_speed_scale
        self.min_moving_speed_kmh = min_moving_speed_kmh
        self.speed_gate_full_kmh = speed_gate_full_kmh
        self.max_steps = max_steps

        # ── Lexicographic dominance invariant ─────────────────────────
        # Efficiency reward is clipped to [-efficiency_cap, +efficiency_cap].
        # Every full-strength safety penalty weight strictly exceeds the
        # cap, so Level-1 ALWAYS dominates Level-2 within a single step:
        #     |efficiency_reward| ≤ efficiency_cap < min(safety_weights)
        # ⇒ no speed/progress/comfort combination can cancel a triggered
        #   safety penalty. (Off-road is catastrophic by construction.)
        assert self.lane_invasion_penalty > self.efficiency_cap
        assert self.safety_edge_weight > self.efficiency_cap
        assert self.shield_intervention_penalty > self.efficiency_cap
        assert self.off_road_penalty > self.efficiency_cap

        self._last_steering = 0.0
        self._last_milestone = 0.0
        self._current_step = 0

    def reset(self, **kwargs):
        self._last_steering = 0.0
        self._last_milestone = 0.0
        self._current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray):
        obs, base_reward, done, truncated, info = self.env.step(action)
        self._current_step += 1

        executed_action = info.get("executed_action", action)
        current_steering = float(executed_action[0])

        # ── Data from Waypoint API ────────────────────────────────────
        speed_kmh = info.get("speed_kmh", 0.0)
        lateral_offset_norm = info.get("lateral_offset_norm", 0.0)
        on_road = info.get("on_road", True)
        on_edge_warning = info.get("on_edge_warning", 0.0)
        lane_invasion = info.get("lane_invasion", False)
        total_distance = info.get("total_distance", 0.0)
        raw_limit = info.get("speed_limit_kmh", 0.0)
        dist_left_edge_norm = info.get("dist_left_edge_norm", 0.5)
        dist_right_edge_norm = info.get("dist_right_edge_norm", 0.5)
        road_curvature_norm = info.get("road_curvature_norm", 0.0)

        effective_limit = (
            float(raw_limit) if raw_limit > 0.0 else self.target_speed_kmh
        )

        # Speed gate — scales comfort signals to 0 at standstill.
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

        # Lane-change transition suppresses invasion and edge penalties
        # because crossing the lane marking is the intended behaviour.
        lane_change_permitted = info.get("lane_change_permitted", False)
        in_lane_transition = (
            lane_change_permitted and abs(lateral_offset_norm) > 0.5
        )

        # Shield signals
        shield_active = info.get(
            "shield_activated", info.get("shield_active", False)
        )
        proposed_action = info.get("proposed_action", executed_action)
        action_divergence = float(
            np.linalg.norm(np.array(executed_action) - np.array(proposed_action))
        )

        # ============================================================
        # LEVEL 1 — SAFETY (dominant)
        # ============================================================

        # ── S1. Off-road / edge proximity ────────────────────────────
        if not on_road:
            # Early off-road costs more than late off-road, removing the
            # incentive to end the episode quickly to dodge other costs.
            remaining_fraction = max(
                0.0, (self.max_steps - self._current_step) / self.max_steps
            )
            edge_penalty = self.off_road_penalty * (
                1.0 + self.off_road_penalty_k * remaining_fraction
            )
        elif in_lane_transition:
            edge_penalty = 0.0
        else:
            critical_edge = min(dist_left_edge_norm, dist_right_edge_norm)
            edge_threshold = 0.4
            if critical_edge < edge_threshold:
                edge_proximity = (
                    (edge_threshold - critical_edge) / edge_threshold
                ) ** 2
                edge_penalty = edge_proximity * self.safety_edge_weight
            elif on_edge_warning > 0.3:
                edge_penalty = on_edge_warning * self.safety_edge_weight * 0.5
            else:
                edge_penalty = 0.0

        # ── S2. Lane invasion ────────────────────────────────────────
        if in_lane_transition or not lane_invasion:
            invasion_pen = 0.0
        else:
            intentional = (
                abs(current_steering) >= self.INTENTIONAL_STEER_THRESHOLD
            )
            if not intentional:
                invasion_severity = min(abs(lateral_offset_norm), 1.0)
                invasion_pen = self.lane_invasion_penalty * (
                    0.5 + 0.5 * invasion_severity
                )
            else:
                invasion_pen = 0.0

        # ── S3. Shield intervention ──────────────────────────────────
        # action_divergence is ||executed - proposed||_2 for a 2D action
        # in [-1,1]^2, bounded by 2*sqrt(2) ≈ 2.83.
        if shield_active:
            shield_pen = (
                action_divergence / 2.83
            ) * self.shield_intervention_penalty
        else:
            shield_pen = 0.0

        safety_reward = -(edge_penalty + invasion_pen + shield_pen)

        # ============================================================
        # LEVEL 2 — EFFICIENCY / COMFORT (subordinate, clipped)
        # ============================================================

        # ── E1. Speed tracking (Gaussian around curvature-adjusted target) ─
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
            # Overspeed penalty stays inside the efficiency budget.
            speed_ceiling = effective_limit * (1.0 + self.speed_limit_margin)
            if speed_kmh > speed_ceiling:
                overspeed = (speed_kmh - speed_ceiling) / effective_limit
                speed_reward -= overspeed * self.speed_weight * 0.8
        else:
            speed_reward = 0.0

        # ── E2. Progress milestone (every PROGRESS_MILESTONE_M metres) ─
        milestone_crossed = (
            total_distance > 0
            and total_distance >= self._last_milestone + self.PROGRESS_MILESTONE_M
        )
        if milestone_crossed:
            progress_bonus = self.progress_weight
            self._last_milestone = (
                total_distance // self.PROGRESS_MILESTONE_M
            ) * self.PROGRESS_MILESTONE_M
        else:
            progress_bonus = 0.0

        # ── E3. Comfort (lane centering − steering smoothness) ───────
        # Smoothness is ZEROED on shield intervention: the jerk in that
        # step is produced by the shield's corrective action, not by the
        # agent's policy, so double-penalising is unfair and noisy.
        if shield_active:
            smoothness_penalty = 0.0
        else:
            steering_diff = abs(current_steering - self._last_steering)
            smoothness_penalty = steering_diff * self.comfort_weight

        min_edge_dist = min(dist_left_edge_norm, dist_right_edge_norm)
        centering_score = float(np.clip(min_edge_dist / 0.5, 0.0, 1.0))
        if in_lane_transition:
            centering_bonus = 0.3 * self.comfort_weight
        else:
            centering_bonus = (
                max(speed_gate, 0.3) * centering_score * self.comfort_weight
            )

        comfort_reward = centering_bonus - smoothness_penalty

        # ── Clip efficiency to enforce lexicographic dominance ────────
        # efficiency_reward ∈ [-efficiency_cap, +efficiency_cap], and
        # every triggered safety weight > efficiency_cap, so Level-1
        # strictly dominates Level-2 per step by construction.
        efficiency_raw = speed_reward + progress_bonus + comfort_reward
        efficiency_reward = float(
            np.clip(efficiency_raw, -self.efficiency_cap, self.efficiency_cap)
        )

        # ── Final shaped reward ──────────────────────────────────────
        shaped_reward = base_reward + safety_reward + efficiency_reward

        self._last_steering = current_steering

        info.update(
            {
                "shaped_reward": shaped_reward,
                "raw_reward": base_reward,
                # Level aggregates
                "safety_reward": safety_reward,
                "efficiency_reward": efficiency_reward,
                "efficiency_raw": efficiency_raw,
                # Level-1 components
                "edge_penalty": edge_penalty,
                "invasion_penalty": invasion_pen,
                "shield_pen": shield_pen,
                # Level-2 components
                "speed_bonus": speed_reward,
                "progress_bonus": progress_bonus,
                "comfort_reward": comfort_reward,
                # Diagnostics
                "effective_speed_limit": effective_limit,
                "curve_adjusted_limit": curve_adjusted_limit,
                "centering_score": centering_score,
                "in_lane_transition": in_lane_transition,
                "efficiency_clipped": efficiency_raw != efficiency_reward,
            }
        )

        return obs, shaped_reward, done, truncated, info
