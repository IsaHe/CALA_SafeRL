from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


class SafetyMetrics:
    """Métodos estáticos para cálculo de métricas de seguridad."""

    @staticmethod
    def risk_level_distribution(infos: List[Dict]) -> Dict[str, float]:
        if not infos:
            return {"safe": 0.0, "warning": 0.0, "critical": 0.0}
        counts: Dict[str, int] = defaultdict(int)
        for info in infos:
            counts[info.get("risk_level", "safe")] += 1
        total = len(infos)
        return {k: counts[k] / total for k in ("safe", "warning", "critical")}

    @staticmethod
    def shield_intervention_analysis(infos: List[Dict]) -> Dict[str, Any]:
        if not infos:
            return {
                "intervention_rate": 0.0, "total_interventions": 0,
                "total_steps": 0, "interventions_by_reason": {},
            }
        total = len(infos)
        interventions = sum(
            1 for i in infos
            if i.get("shield_activated", i.get("shield_active", False))
        )
        reasons: Dict[str, int] = defaultdict(int)
        for info in infos:
            if info.get("shield_activated", info.get("shield_active", False)):
                reasons[info.get("shield_reason", "unknown")] += 1
        return {
            "intervention_rate":       interventions / total,
            "total_interventions":     interventions,
            "total_steps":             total,
            "interventions_by_reason": dict(reasons),
        }

    @staticmethod
    def minimum_distance_distribution(infos: List[Dict]) -> Dict[str, float]:
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
            "mean":       float(np.mean(distances)),
            "std":        float(np.std(distances)),
            "min":        float(np.min(distances)),
            "max":        float(np.max(distances)),
            "median":     float(np.median(distances)),
            "below_0_15": float(np.mean(distances < 0.15)),
            "below_0_30": float(np.mean(distances < 0.30)),
        }

    @staticmethod
    def hidden_unsafe_state_detection(
        episodes: List[List[Dict]],
        horizon: int = 3,
    ) -> Dict[str, Any]:
        detected = 0
        total_checked = 0
        for ep in episodes:
            for i, info in enumerate(ep):
                if info.get("shield_activated", info.get("shield_active", False)):
                    continue
                future = ep[i + 1: i + horizon + 1]
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
                    detected += 1
        return {
            "detected_count": detected,
            "total_checked":  total_checked,
            "detection_rate": detected / max(total_checked, 1),
        }

    @staticmethod
    def horizon_effectiveness(infos: List[Dict]) -> Dict[int, Dict]:
        data: Dict[int, Dict] = defaultdict(lambda: {"total_steps": 0, "intervention_count": 0})
        for info in infos:
            h = info.get("horizon_used", 1)
            data[h]["total_steps"] += 1
            if info.get("shield_activated", info.get("shield_active", False)):
                data[h]["intervention_count"] += 1
        return {
            h: {
                "total_steps":        d["total_steps"],
                "intervention_count": d["intervention_count"],
                "intervention_rate":  d["intervention_count"] / max(d["total_steps"], 1),
            }
            for h, d in data.items()
        }

    # ──────────────────────────────────────────────────────────────────
    # MÉTRICAS CARLA WAYPOINT API
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def lane_safety_metrics(infos: List[Dict]) -> Dict[str, float]:
        """
        Métricas de seguridad de carril usando datos del Waypoint API de CARLA.
        """
        if not infos:
            return {}

        lat_offsets    = np.array([abs(i.get("lateral_offset",
                                             i.get("lateral_offset_norm", 0.0)))
                                    for i in infos])
        heading_errors = np.array([abs(i.get("heading_error", 0.0)) for i in infos])
        lane_invasions = sum(1 for i in infos if i.get("lane_invasion", False))
        off_road_steps = sum(1 for i in infos if not i.get("on_road", True))

        return {
            "mean_lateral_offset":    float(np.mean(lat_offsets)),
            "max_lateral_offset":     float(np.max(lat_offsets)),
            "std_lateral_offset":     float(np.std(lat_offsets)),
            "pct_above_half_lane":    float(np.mean(lat_offsets > 0.5)),
            "mean_heading_error_deg": float(np.mean(heading_errors)),
            "max_heading_error_deg":  float(np.max(heading_errors)),
            "lane_invasion_rate":     lane_invasions / len(infos),
            "total_lane_invasions":   lane_invasions,
            "off_road_rate":          off_road_steps / len(infos),
            "total_off_road_steps":   off_road_steps,
        }

    @staticmethod
    def speed_metrics(infos: List[Dict]) -> Dict[str, float]:
        """Métricas de velocidad usando km/h reales de CARLA."""
        if not infos:
            return {}
        speeds = np.array([i.get("speed_kmh", 0.0) for i in infos])
        return {
            "mean_speed_kmh":  float(np.mean(speeds)),
            "max_speed_kmh":   float(np.max(speeds)),
            "std_speed_kmh":   float(np.std(speeds)),
            "pct_above_40kmh": float(np.mean(speeds > 40.0)),
            "pct_below_5kmh":  float(np.mean(speeds < 5.0)),
        }

    @staticmethod
    def speed_compliance_metrics(infos: List[Dict]) -> Dict[str, float]:
        """
        Cumplimiento del límite de velocidad dinámico (Waypoint API).

        Requiere que el info dict contenga 'speed_kmh' y 'speed_limit_kmh'
        (disponibles cuando se usa el CarlaEnv actualizado con speed limit).

        Returns:
          compliance_rate     : fracción de steps con speed ≤ limit * 1.05
          overspeed_5pct_rate : fracción de steps con speed > limit * 1.05
          overspeed_20pct_rate: fracción de steps con speed > limit * 1.20
          mean_speed_vs_limit : ratio medio speed/limit (1.0 = circulando al límite)
          mean_speed_limit_kmh: media del límite de la vía durante el episodio
        """
        if not infos:
            return {"compliance_rate": 1.0}

        valid = [(i["speed_kmh"], i["speed_limit_kmh"])
                 for i in infos
                 if "speed_kmh" in i and i.get("speed_limit_kmh", 0.0) > 0.0]

        if not valid:
            return {"compliance_rate": 1.0, "note": "no speed_limit_kmh data"}

        speeds, limits = zip(*valid)
        speeds  = np.array(speeds,  dtype=np.float32)
        limits  = np.array(limits,  dtype=np.float32)
        ratios  = speeds / np.maximum(limits, 1.0)

        return {
            "compliance_rate":      float(np.mean(ratios <= 1.05)),
            "overspeed_5pct_rate":  float(np.mean(ratios > 1.05)),
            "overspeed_20pct_rate": float(np.mean(ratios > 1.20)),
            "mean_speed_vs_limit":  float(np.mean(ratios)),
            "max_speed_vs_limit":   float(np.max(ratios)),
            "mean_speed_limit_kmh": float(np.mean(limits)),
        }

    @staticmethod
    def lane_edge_metrics(infos: List[Dict]) -> Dict[str, float]:
        """
        Métricas de proximidad a los bordes del carril (v2).

        Usa dist_left_edge_norm / dist_right_edge_norm del Waypoint API.
        Valor 1.0 = centro exacto del carril, 0.0 = en el borde.
        """
        if not infos:
            return {}

        left_dists  = np.array([i.get("dist_left_edge_norm",  0.5) for i in infos], dtype=np.float32)
        right_dists = np.array([i.get("dist_right_edge_norm", 0.5) for i in infos], dtype=np.float32)
        min_dists   = np.minimum(left_dists, right_dists)
        asymmetry   = np.abs(left_dists - right_dists)

        return {
            "mean_min_edge_dist":   float(np.mean(min_dists)),
            "min_min_edge_dist":    float(np.min(min_dists)),
            "mean_left_edge_dist":  float(np.mean(left_dists)),
            "mean_right_edge_dist": float(np.mean(right_dists)),
            "pct_below_0_3":        float(np.mean(min_dists < 0.30)),
            "pct_below_0_15":       float(np.mean(min_dists < 0.15)),
            "mean_edge_asymmetry":  float(np.mean(asymmetry)),
            "max_edge_asymmetry":   float(np.max(asymmetry)),
        }

    @staticmethod
    def semantic_lidar_metrics(infos: List[Dict]) -> Dict[str, Any]:
        """
        Métricas derivadas del LIDAR semántico (SemanticScanResult v2).

        Añadido en v2: nearest_road_edge_m y n_road_edge_pts para diagnóstico
        de detección de bordes de carretera por el sensor multi-canal.
        """
        if not infos:
            return {}

        veh_dists       = [i.get("nearest_vehicle_m",    999.0) for i in infos]
        ped_dists       = [i.get("nearest_pedestrian_m", 999.0) for i in infos]
        stat_dists      = [i.get("nearest_static_m",     999.0) for i in infos]
        road_edge_dists = [i.get("nearest_road_edge_m",  999.0) for i in infos]
        n               = len(infos)

        veh_real       = [d for d in veh_dists       if d < 999.0]
        ped_real       = [d for d in ped_dists       if d < 999.0]
        stat_real      = [d for d in stat_dists      if d < 999.0]
        road_edge_real = [d for d in road_edge_dists if d < 999.0]

        veh_critical_5m  = sum(1 for d in veh_dists if d < 5.0)
        veh_critical_10m = sum(1 for d in veh_dists if d < 10.0)
        ped_critical_4m  = sum(1 for d in ped_dists if d < 4.0)
        ped_critical_8m  = sum(1 for d in ped_dists if d < 8.0)

        road_edge_detection_rate = sum(1 for d in road_edge_dists if d < 999.0) / max(n, 1)
        road_edge_close_rate     = sum(1 for d in road_edge_dists if d < 5.0)   / max(n, 1)
        mean_road_edge_pts       = float(np.mean([i.get("n_road_edge_pts", 0) for i in infos]))

        tag_counts_total: Dict[int, int] = defaultdict(int)
        for info in infos:
            for tag, count in info.get("semantic_tag_counts", {}).items():
                tag_counts_total[tag] += count

        ep_had_pedestrian = any(d < 50.0 for d in ped_dists)
        ep_had_vehicles   = any(d < 50.0 for d in veh_dists)

        return {
            "mean_nearest_vehicle_m":    float(np.mean(veh_real))  if veh_real  else 999.0,
            "min_nearest_vehicle_m":     float(min(veh_dists)),
            "vehicle_critical_5m_rate":  veh_critical_5m  / n,
            "vehicle_critical_10m_rate": veh_critical_10m / n,

            "mean_nearest_pedestrian_m":   float(np.mean(ped_real)) if ped_real else 999.0,
            "min_nearest_pedestrian_m":    float(min(ped_dists)),
            "pedestrian_critical_4m_rate": ped_critical_4m / n,
            "pedestrian_critical_8m_rate": ped_critical_8m / n,

            "mean_nearest_static_m": float(np.mean(stat_real)) if stat_real else 999.0,
            "min_nearest_static_m":  float(min(stat_dists)),

            "mean_nearest_road_edge_m":  float(np.mean(road_edge_real)) if road_edge_real else 999.0,
            "min_nearest_road_edge_m":   float(min(road_edge_dists)),
            "road_edge_detection_rate":  road_edge_detection_rate,
            "road_edge_close_rate":      road_edge_close_rate,
            "mean_road_edge_pts":        mean_road_edge_pts,

            "ep_had_pedestrian": ep_had_pedestrian,
            "ep_had_vehicles":   ep_had_vehicles,
            "tag_counts":        dict(tag_counts_total),
        }

    @staticmethod
    def shield_semantic_analysis(infos: List[Dict]) -> Dict[str, Any]:
        """
        Desglose de intervenciones del shield por categoría semántica.

        Complementa shield_intervention_analysis con el origen de cada
        intervención (dinámica, estática, peatón), disponible cuando se usa
        CarlaAdaptiveHorizonShield con LIDAR semántico.

        Returns:
          total_interventions
          dynamic_interventions    : colisión inminente con vehículo
          static_interventions     : quitamiedos / muros laterales
          pedestrian_interventions : emergencia por peatón
          dynamic_rate, static_rate, pedestrian_rate : fracciones del total
          intervention_rate        : intervenciones / steps totales
        """
        if not infos:
            return {}

        total = len(infos)
        n_int      = sum(1 for i in infos if i.get("shield_activated", i.get("shield_active", False)))
        n_dynamic  = sum(1 for i in infos if i.get("interventions_dynamic", 0) > 0)
        n_static   = sum(1 for i in infos if i.get("interventions_static",  0) > 0)
        n_ped      = sum(1 for i in infos if i.get("interventions_pedestrian", 0) > 0)

        # También consultar los campos de nearest_*_m en steps de intervención
        veh_at_intervention  = [
            i.get("nearest_vehicle_m", 999.0)
            for i in infos
            if i.get("shield_activated", False) and i.get("nearest_vehicle_m", 999.0) < 999.0
        ]
        ped_at_intervention  = [
            i.get("nearest_pedestrian_m", 999.0)
            for i in infos
            if i.get("shield_activated", False) and i.get("nearest_pedestrian_m", 999.0) < 999.0
        ]

        return {
            "total_interventions":     n_int,
            "intervention_rate":       n_int / max(total, 1),
            "dynamic_interventions":   n_dynamic,
            "static_interventions":    n_static,
            "pedestrian_interventions":n_ped,
            "dynamic_rate":    n_dynamic / max(n_int, 1),
            "static_rate":     n_static  / max(n_int, 1),
            "pedestrian_rate": n_ped     / max(n_int, 1),
            "mean_vehicle_dist_at_intervention":
                float(np.mean(veh_at_intervention)) if veh_at_intervention else 999.0,
            "mean_pedestrian_dist_at_intervention":
                float(np.mean(ped_at_intervention)) if ped_at_intervention else 999.0,
        }

    @staticmethod
    def collision_analysis(infos: List[Dict]) -> Dict[str, Any]:
        """Análisis de colisiones con datos físicos de CARLA."""
        total = len(infos)
        if total == 0:
            return {}
        collisions = [i for i in infos if i.get("collision", False)]
        return {
            "total_collisions": len(collisions),
            "collision_rate":   len(collisions) / total,
            "episode_crashed":  len(collisions) > 0,
        }