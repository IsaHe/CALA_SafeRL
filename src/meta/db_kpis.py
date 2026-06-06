"""
db_kpis.py - Lee una metrics.sqlite de un run y construye un KpiSnapshot.

Usado por el Modo A (entre fases) del meta-controlador: cuando una fase termina,
se leen las ultimas ``window`` filas del eje 'episode' de su DB y se ensambla el
snapshot canonico. Reusa ``load_axis_frame`` de src/Metrics/live_metrics.py.

Los nombres de metrica espejean los que escribe main_train.py en el eje
'episode'. crash/offroad/success ya son rolling-100 en el bucle, asi que se toma
su ULTIMO valor; el resto se promedia sobre la ventana.
"""

from __future__ import annotations

from pathlib import Path

from src.meta.kpi_snapshot import KpiSnapshot, build_kpi_snapshot

# Metricas de las que se toma el ULTIMO valor (ya son rolling-100 en el loop).
_LAST_VALUE = {
    "crash_rate": "Training/Crash_Rate",
    "offroad_rate": "Training/Offroad_Rate",
    "success_rate": "Training/Success_Rate",
    "avg_reward_100": "Reward/Average_100_Episodes",
}
# Metricas que se promedian sobre la ventana.
_WINDOW_MEAN = {
    "stuck_rate": "Outcome/Stuck",
    "shielded_fraction": "Safety/Shield_Rate",
    "mean_speed_kmh": "CARLA/Mean_Speed_kmh",
}
# Promedios de ventana -> claves de shield_stats que espera build_kpi_snapshot.
_STATS_WINDOW_MEAN = {
    "interventions_dynamic": "Safety/Semantic/Dynamic_Interventions",
    "interventions_static": "Safety/Semantic/Static_Interventions",
    "interventions_pedestrian": "Safety/Semantic/Pedestrian_Interventions",
    "interventions_recovery": "Safety/Semantic/Recovery_Activations",
    "safe_rate": "Safety/Semantic/Safe_Step_Rate",
    "warning_rate": "Safety/Semantic/Warning_Step_Rate",
    "critical_rate": "Safety/Semantic/Critical_Step_Rate",
}


def read_kpi_snapshot_from_db(
    db_path,
    run_name: str,
    *,
    current_permissiveness: float,
    num_npc: int = 0,
    window: int = 100,
) -> KpiSnapshot | None:
    """Devuelve un KpiSnapshot del final del run, o None si la DB esta vacia."""
    from src.Metrics.live_metrics import load_axis_frame

    df = load_axis_frame(Path(db_path), run_name, "episode")
    if df is None or df.empty:
        return None

    tail = df.tail(window)

    def last_val(col: str, default: float = 0.0) -> float:
        if col in df.columns and not df[col].dropna().empty:
            return float(df[col].dropna().iloc[-1])
        return default

    def win_mean(col: str, default: float = 0.0) -> float:
        if col in tail.columns and not tail[col].dropna().empty:
            return float(tail[col].dropna().mean())
        return default

    shield_stats = {key: win_mean(col) for key, col in _STATS_WINDOW_MEAN.items()}
    episode = int(df["Step"].iloc[-1]) if "Step" in df.columns else len(df)

    return build_kpi_snapshot(
        episode=episode,
        current_permissiveness=current_permissiveness,
        crash_rate=last_val(_LAST_VALUE["crash_rate"]),
        offroad_rate=last_val(_LAST_VALUE["offroad_rate"]),
        success_rate=last_val(_LAST_VALUE["success_rate"]),
        stuck_rate=win_mean(_WINDOW_MEAN["stuck_rate"]),
        shielded_fraction=win_mean(_WINDOW_MEAN["shielded_fraction"]),
        window_episodes=min(window, len(df)),
        shield_stats=shield_stats,
        avg_reward_100=last_val(_LAST_VALUE["avg_reward_100"]),
        mean_speed_kmh=win_mean(_WINDOW_MEAN["mean_speed_kmh"]),
        num_npc=num_npc,
    )
