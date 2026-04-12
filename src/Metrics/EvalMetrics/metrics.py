"""
metrics.py - Safety Metrics para CARLA RL
"""

from typing import Dict, List

from src.Metrics.EvalMetrics import SafetyMetrics

class SafetyMetricsReporter:
    """Genera reportes de seguridad formateados para consola y logs."""

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

        # ── Distribución de riesgo ────────────────────────────────────
        risk = SafetyMetrics.risk_level_distribution(all_infos)
        lines.append("\nRISK DISTRIBUTION:")
        lines.append(f"  Safe:     {risk['safe']:.1%}")
        lines.append(f"  Warning:  {risk['warning']:.1%}")
        lines.append(f"  Critical: {risk['critical']:.1%}")

        # ── Shield general ────────────────────────────────────────────
        if shield_type != "none":
            sa = SafetyMetrics.shield_intervention_analysis(all_infos)
            lines.append("\nSHIELD INTERVENTIONS:")
            lines.append(f"  Rate:  {sa['intervention_rate']:.1%}")
            lines.append(f"  Total: {sa['total_interventions']}")

        # ── Shield semántico ──────────────────────────────────────────
        sem_shield = SafetyMetrics.shield_semantic_analysis(all_infos)
        if sem_shield and sem_shield.get("total_interventions", 0) > 0:
            lines.append("\nSHIELD SEMANTIC BREAKDOWN:")
            n = sem_shield["total_interventions"]
            lines.append(f"  Dynamic  (vehicle): {sem_shield['dynamic_interventions']:>5}  ({sem_shield['dynamic_rate']:.1%})")
            lines.append(f"  Static   (guardrail): {sem_shield['static_interventions']:>5}  ({sem_shield['static_rate']:.1%})")
            lines.append(f"  Pedestrian:           {sem_shield['pedestrian_interventions']:>5}  ({sem_shield['pedestrian_rate']:.1%})")
            if sem_shield["mean_vehicle_dist_at_intervention"] < 999.0:
                lines.append(
                    f"  Mean vehicle dist at intervention: "
                    f"{sem_shield['mean_vehicle_dist_at_intervention']:.1f} m"
                )

        # ── Distancias ────────────────────────────────────────────────
        dist = SafetyMetrics.minimum_distance_distribution(all_infos)
        lines.append("\nMINIMUM DISTANCE STATS:")
        lines.append(f"  Mean: {dist['mean']:.3f}  |  Min: {dist['min']:.3f}")
        lines.append(f"  Below 0.15: {dist['below_0_15']:.1%}  |  Below 0.30: {dist['below_0_30']:.1%}")

        # ── LIDAR semántico ───────────────────────────────────────────
        sem = SafetyMetrics.semantic_lidar_metrics(all_infos)
        if sem:
            lines.append("\nSEMANTIC LIDAR DISTANCES:")
            mv = sem.get("mean_nearest_vehicle_m", 999.0)
            mp = sem.get("mean_nearest_pedestrian_m", 999.0)
            mr = sem.get("mean_nearest_road_edge_m", 999.0)
            lines.append(f"  Nearest vehicle    (mean): {mv:.1f} m" +
                         (" (no vehicles)" if mv >= 999.0 else ""))
            lines.append(f"  Nearest pedestrian (mean): {mp:.1f} m" +
                         (" (no pedestrians)" if mp >= 999.0 else ""))
            lines.append(f"  Nearest road edge  (mean): {mr:.1f} m" +
                         (" (not detected — check sensor FOV)" if mr >= 999.0 else ""))
            lines.append(f"  Vehicle   <5m rate:  {sem.get('vehicle_critical_5m_rate',  0):.1%}")
            lines.append(f"  Pedestrian <4m rate: {sem.get('pedestrian_critical_4m_rate',0):.1%}")
            lines.append(f"  Road edge detection: {sem.get('road_edge_detection_rate', 0):.1%}  "
                         f"close (<5m): {sem.get('road_edge_close_rate', 0):.1%}")

        # ── Lane safety ───────────────────────────────────────────────
        lane = SafetyMetrics.lane_safety_metrics(all_infos)
        if lane:
            lines.append("\nLANE SAFETY (Waypoint API):")
            lines.append(f"  Mean lateral offset: {lane.get('mean_lateral_offset', 0):.3f}")
            lines.append(f"  Max lateral offset:  {lane.get('max_lateral_offset',  0):.3f}")
            lines.append(f"  >half-lane rate:     {lane.get('pct_above_half_lane', 0):.1%}")
            lines.append(f"  Lane invasions:      {lane.get('total_lane_invasions', 0)}")
            lines.append(f"  Off-road rate:       {lane.get('off_road_rate', 0):.1%}")

        # ── Lane edge distances (v2) ──────────────────────────────────
        edge = SafetyMetrics.lane_edge_metrics(all_infos)
        if edge:
            lines.append("\nLANE EDGE PROXIMITY (v2 — Waypoint API):")
            lines.append(f"  Mean min-edge dist:  {edge.get('mean_min_edge_dist', 0):.3f}  "
                         f"(1.0=center, 0.0=edge)")
            lines.append(f"  Mean left  edge:     {edge.get('mean_left_edge_dist',  0):.3f}")
            lines.append(f"  Mean right edge:     {edge.get('mean_right_edge_dist', 0):.3f}")
            lines.append(f"  Alert zone  (<0.30): {edge.get('pct_below_0_3',  0):.1%}")
            lines.append(f"  Critical    (<0.15): {edge.get('pct_below_0_15', 0):.1%}")
            lines.append(f"  Mean asymmetry:      {edge.get('mean_edge_asymmetry', 0):.3f}  "
                         f"(0=centered, >0.3=drift)")

        # ── Velocidad ─────────────────────────────────────────────────
        spd = SafetyMetrics.speed_metrics(all_infos)
        if spd:
            lines.append("\nSPEED METRICS:")
            lines.append(f"  Mean: {spd.get('mean_speed_kmh', 0):.1f} km/h"
                         f"  |  Max: {spd.get('max_speed_kmh', 0):.1f} km/h")

        # ── Cumplimiento del límite de velocidad ──────────────────────
        comp = SafetyMetrics.speed_compliance_metrics(all_infos)
        if comp and "compliance_rate" in comp:
            lines.append("\nSPEED LIMIT COMPLIANCE:")
            lines.append(f"  Compliance rate:      {comp.get('compliance_rate', 0):.1%}")
            lines.append(f"  Overspeed >5%  rate:  {comp.get('overspeed_5pct_rate',  0):.1%}")
            lines.append(f"  Overspeed >20% rate:  {comp.get('overspeed_20pct_rate', 0):.1%}")
            lines.append(f"  Mean speed / limit:   {comp.get('mean_speed_vs_limit', 0):.2f}x")
            if "mean_speed_limit_kmh" in comp:
                lines.append(f"  Mean speed limit:     {comp['mean_speed_limit_kmh']:.1f} km/h")

        # ── Hidden unsafe states ──────────────────────────────────────
        if all_episodes:
            hidden = SafetyMetrics.hidden_unsafe_state_detection(all_episodes)
            lines.append(
                f"\nHIDDEN UNSAFE STATES: "
                f"{hidden['detected_count']} / {hidden['total_checked']} "
                f"({hidden['detection_rate']:.1%})"
            )

        # ── Horizon effectiveness ─────────────────────────────────────
        if shield_type == "adaptive" and any("horizon_used" in i for i in all_infos):
            horizons = SafetyMetrics.horizon_effectiveness(all_infos)
            lines.append("\nHORIZON EFFECTIVENESS:")
            for h in sorted(horizons.keys()):
                d = horizons[h]
                lines.append(
                    f"  Horizon {h:>2}: {d['intervention_rate']:.1%} "
                    f"({d['intervention_count']}/{d['total_steps']})"
                )

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)