"""
metrics.py - Safety Metrics para CARLA RL

Extiende las métricas del proyecto original con análisis específicos de CARLA:
  - lane_safety_metrics: usa offset lateral exacto del Waypoint API
  - speed_metrics: usa velocidades en km/h reales
  - carla_specific_metrics: invasiones de carril, impulsos de colisión

API compatible con main_eval_improved.py:
    SafetyMetrics.risk_level_distribution(all_infos)
    SafetyMetrics.shield_intervention_analysis(all_infos)
    SafetyMetrics.minimum_distance_distribution(all_infos)
    SafetyMetrics.hidden_unsafe_state_detection(all_episodes)
    SafetyMetrics.horizon_effectiveness(all_infos)
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict


class SafetyMetrics:
    """Métodos estáticos para cálculo de métricas de seguridad."""

    @staticmethod
    def risk_level_distribution(infos: List[Dict]) -> Dict[str, float]:
        """
        Distribución de niveles de riesgo a lo largo de un episodio.

        Args:
            infos: Lista de info dicts de cada step

        Returns:
            {'safe': float, 'warning': float, 'critical': float}
        """
        if not infos:
            return {"safe": 0.0, "warning": 0.0, "critical": 0.0}

        counts: Dict[str, int] = defaultdict(int)
        for info in infos:
            level = info.get("risk_level", "safe")
            counts[level] += 1

        total = len(infos)
        return {
            "safe":     counts["safe"]     / total,
            "warning":  counts["warning"]  / total,
            "critical": counts["critical"] / total,
        }

    @staticmethod
    def shield_intervention_analysis(infos: List[Dict]) -> Dict[str, Any]:
        """
        Análisis detallado de las intervenciones del shield.

        Retorna tasa de intervención y distribución por razón.
        """
        if not infos:
            return {
                "intervention_rate": 0.0,
                "total_interventions": 0,
                "total_steps": 0,
                "interventions_by_reason": {},
            }

        total        = len(infos)
        interventions = sum(
            1 for i in infos
            if i.get("shield_activated", i.get("shield_active", False))
        )

        reasons: Dict[str, int] = defaultdict(int)
        for info in infos:
            if info.get("shield_activated", info.get("shield_active", False)):
                reason = info.get("shield_reason", "unknown")
                reasons[reason] += 1

        return {
            "intervention_rate":         interventions / total,
            "total_interventions":       interventions,
            "total_steps":               total,
            "interventions_by_reason":   dict(reasons),
        }

    @staticmethod
    def minimum_distance_distribution(infos: List[Dict]) -> Dict[str, float]:
        """
        Estadísticas de distancia mínima al obstáculo más cercano.

        Usa 'min_distance' o 'min_front_dist' del info dict.
        """
        if not infos:
            return {
                "mean": 0.0, "std": 0.0, "min": 0.0,
                "max": 0.0, "median": 0.0,
                "below_0_15": 0.0, "below_0_30": 0.0,
            }

        distances = np.array([
            info.get("min_distance", info.get("min_front_dist", 1.0))
            for info in infos
        ])

        return {
            "mean":        float(np.mean(distances)),
            "std":         float(np.std(distances)),
            "min":         float(np.min(distances)),
            "max":         float(np.max(distances)),
            "median":      float(np.median(distances)),
            "below_0_15":  float(np.mean(distances < 0.15)),
            "below_0_30":  float(np.mean(distances < 0.30)),
        }

    @staticmethod
    def hidden_unsafe_state_detection(
        episodes: List[List[Dict]],
        horizon: int = 3,
    ) -> Dict[str, Any]:
        """
        Detecta 'hidden unsafe states': pasos donde el shield no intervino
        pero los siguientes N pasos tuvieron situaciones peligrosas.

        Basado en Paper 1: estados de los que no hay escape si el shield
        no interviene a tiempo.

        Args:
            episodes: Lista de episodios, cada uno como lista de info dicts
            horizon:  Ventana de búsqueda hacia adelante

        Returns:
            {'detected_count': int, 'total_checked': int, 'detection_rate': float}
        """
        detected_count = 0
        total_checked  = 0

        for episode_infos in episodes:
            for i, info in enumerate(episode_infos):
                # Solo analizar pasos donde el shield NO intervino
                if info.get("shield_activated", info.get("shield_active", False)):
                    continue

                future = episode_infos[i + 1 : i + horizon + 1]
                if not future:
                    continue

                total_checked += 1

                future_unsafe = any(
                    f.get("shield_activated", f.get("shield_active", False)) or
                    f.get("min_distance", 1.0) < 0.10 or
                    f.get("collision", False)
                    for f in future
                )

                if future_unsafe:
                    detected_count += 1

        return {
            "detected_count": detected_count,
            "total_checked":  total_checked,
            "detection_rate": detected_count / max(total_checked, 1),
        }

    @staticmethod
    def horizon_effectiveness(infos: List[Dict]) -> Dict[int, Dict]:
        """
        Efectividad del shield por horizonte de predicción.

        Retorna para cada horizonte: total_steps, intervention_count, intervention_rate.
        """
        horizon_data: Dict[int, Dict] = defaultdict(
            lambda: {"total_steps": 0, "intervention_count": 0}
        )

        for info in infos:
            horizon = info.get("horizon_used", 1)
            horizon_data[horizon]["total_steps"] += 1
            if info.get("shield_activated", info.get("shield_active", False)):
                horizon_data[horizon]["intervention_count"] += 1

        result = {}
        for h, data in horizon_data.items():
            total = data["total_steps"]
            count = data["intervention_count"]
            result[h] = {
                "total_steps":       total,
                "intervention_count": count,
                "intervention_rate":  count / max(total, 1),
            }

        return result

    # ══════════════════════════════════════════════════════════════════
    # MÉTRICAS ESPECÍFICAS DE CARLA
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def lane_safety_metrics(infos: List[Dict]) -> Dict[str, float]:
        """
        Métricas de seguridad de carril usando datos del Waypoint API de CARLA.

        Estas métricas son más precisas que las de MetaDrive porque usan
        offsets laterales reales en metros, no heurísticas.
        """
        if not infos:
            return {}

        # Offset lateral en metros
        lat_offsets = np.array([
            abs(info.get("lateral_offset", info.get("lateral_offset_norm", 0.0)))
            for info in infos
        ])

        # Error de heading en grados
        heading_errors = np.array([
            abs(info.get("heading_error", 0.0))
            for info in infos
        ])

        lane_invasions = sum(
            1 for info in infos if info.get("lane_invasion", False)
        )

        off_road_steps = sum(
            1 for info in infos if not info.get("on_road", True)
        )

        return {
            "mean_lateral_offset":       float(np.mean(lat_offsets)),
            "max_lateral_offset":        float(np.max(lat_offsets)),
            "std_lateral_offset":        float(np.std(lat_offsets)),
            "mean_heading_error_deg":    float(np.mean(heading_errors)),
            "max_heading_error_deg":     float(np.max(heading_errors)),
            "lane_invasion_rate":        lane_invasions / len(infos),
            "total_lane_invasions":      lane_invasions,
            "off_road_rate":             off_road_steps / len(infos),
            "total_off_road_steps":      off_road_steps,
        }

    @staticmethod
    def speed_metrics(infos: List[Dict]) -> Dict[str, float]:
        """
        Métricas de velocidad usando km/h reales de CARLA.
        """
        if not infos:
            return {}

        speeds = np.array([info.get("speed_kmh", 0.0) for info in infos])

        return {
            "mean_speed_kmh":   float(np.mean(speeds)),
            "max_speed_kmh":    float(np.max(speeds)),
            "std_speed_kmh":    float(np.std(speeds)),
            "pct_above_40kmh":  float(np.mean(speeds > 40.0)),
            "pct_below_5kmh":   float(np.mean(speeds < 5.0)),   # vehículo casi parado
        }

    @staticmethod
    def collision_analysis(infos: List[Dict]) -> Dict[str, Any]:
        """
        Análisis de colisiones con datos físicos de CARLA.
        """
        total_steps = len(infos)
        if total_steps == 0:
            return {}

        collisions = [info for info in infos if info.get("collision", False)]

        return {
            "total_collisions":  len(collisions),
            "collision_rate":    len(collisions) / total_steps,
            "episode_crashed":   len(collisions) > 0,
        }


class SafetyMetricsReporter:
    """Genera reportes de seguridad formateados."""

    @staticmethod
    def generate_report(
        all_infos: List[Dict],
        all_episodes: List[List[Dict]],
        shield_type: str = "none",
    ) -> str:
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("SAFETY METRICS REPORT — CARLA")
        lines.append("=" * 70)

        # Distribución de riesgo
        risk = SafetyMetrics.risk_level_distribution(all_infos)
        lines.append("\n📊 RISK DISTRIBUTION:")
        lines.append(f"  Safe:     {risk['safe']:.1%}")
        lines.append(f"  Warning:  {risk['warning']:.1%}")
        lines.append(f"  Critical: {risk['critical']:.1%}")

        # Shield
        if shield_type != "none":
            shield_a = SafetyMetrics.shield_intervention_analysis(all_infos)
            lines.append("\n🛡️ SHIELD INTERVENTIONS:")
            lines.append(f"  Rate:  {shield_a['intervention_rate']:.1%}")
            lines.append(f"  Total: {shield_a['total_interventions']}")
            if shield_a["interventions_by_reason"]:
                lines.append("  By reason:")
                for r, c in shield_a["interventions_by_reason"].items():
                    lines.append(f"    {r}: {c}")

        # Distancias
        dist = SafetyMetrics.minimum_distance_distribution(all_infos)
        lines.append("\n📏 MINIMUM DISTANCE STATS:")
        lines.append(f"  Mean: {dist['mean']:.3f}  |  Min: {dist['min']:.3f}")
        lines.append(f"  Below 0.15: {dist['below_0_15']:.1%}")

        # Lane safety (CARLA-específico)
        lane = SafetyMetrics.lane_safety_metrics(all_infos)
        if lane:
            lines.append("\n🛣️ LANE SAFETY (CARLA WAYPOINT API):")
            lines.append(f"  Mean lateral offset: {lane.get('mean_lateral_offset', 0):.3f}")
            lines.append(f"  Max lateral offset:  {lane.get('max_lateral_offset', 0):.3f}")
            lines.append(f"  Lane invasions:      {lane.get('total_lane_invasions', 0)}")
            lines.append(f"  Off-road rate:       {lane.get('off_road_rate', 0):.1%}")

        # Velocidad
        spd = SafetyMetrics.speed_metrics(all_infos)
        if spd:
            lines.append("\n🚗 SPEED METRICS:")
            lines.append(f"  Mean: {spd.get('mean_speed_kmh', 0):.1f} km/h")
            lines.append(f"  Max:  {spd.get('max_speed_kmh', 0):.1f} km/h")

        # Hidden unsafe states
        if all_episodes:
            hidden = SafetyMetrics.hidden_unsafe_state_detection(all_episodes)
            lines.append(f"\n⚠️ HIDDEN UNSAFE STATES: {hidden['detected_count']}"
                         f" / {hidden['total_checked']}"
                         f" ({hidden['detection_rate']:.1%})")

        # Horizon effectiveness
        if shield_type == "adaptive" and any(
            "horizon_used" in i for i in all_infos
        ):
            horizons = SafetyMetrics.horizon_effectiveness(all_infos)
            lines.append("\n🔭 HORIZON EFFECTIVENESS:")
            for h in sorted(horizons.keys()):
                d = horizons[h]
                lines.append(
                    f"  Horizon {h:>2}: {d['intervention_rate']:.1%} "
                    f"({d['intervention_count']}/{d['total_steps']})"
                )

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
