import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ollama
import json
from collections import defaultdict
import os
import importlib
import time
import math
import re
import hashlib

from src.Metrics.live_metrics import list_live_metric_dbs, load_datasets_from_sqlite


def trigger_autorefresh(interval_ms, key):
    # Streamlit 1.55 no expone st.autorefresh, así que usamos fragment + rerun.
    if hasattr(st, "fragment"):
        interval_seconds = max(float(interval_ms) / 1000.0, 0.25)
        state_key = f"__autorefresh_token_{key}"

        if state_key not in st.session_state:
            st.session_state[state_key] = time.monotonic()

        @st.fragment(run_every=interval_seconds)
        def _auto_refresh_fragment():
            last_tick = st.session_state.get(state_key, 0.0)
            now = time.monotonic()
            if now - last_tick >= interval_seconds * 0.9:
                st.session_state[state_key] = now
                st.rerun()

        _auto_refresh_fragment()
        return

    try:
        module = importlib.import_module("streamlit_autorefresh")
        st_autorefresh = getattr(module, "st_autorefresh")
        st_autorefresh(interval=interval_ms, key=key)
    except Exception:
        st.caption(
            "Auto-refresco no disponible en esta versión de Streamlit. Usa el botón Rerun o actualiza Streamlit."
        )


# Configuración de la página
st.set_page_config(
    page_title="RL Training Dashboard", layout="wide", initial_sidebar_state="collapsed"
)

# Estilos CSS personalizados para simular el aspecto del HTML
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E1E2E;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-label { font-size: 12px; color: #A0A0B0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #FFFFFF; }
    .insight-card {
        padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid;
        background-color: #1E1E2E;
    }
    .insight-good { border-left-color: #3ecf8e; }
    .insight-warn { border-left-color: #ff4f4f; }
    .insight-info { border-left-color: #4a9eff; }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("Dashboard de Entrenamiento RL - Agente Individual")
st.markdown(
    "Analiza la estabilidad, el rendimiento y el comportamiento de tu agente en CARLA."
)

dashboard_section = st.radio(
    "Apartado",
    options=["Análisis de run", "Comparación de runs"],
    horizontal=True,
)

# 1. Fuente de datos (solo análisis individual)
source_mode = "SQLite en tiempo real"
uploaded_files = []
live_runs = []
selected_live_run = None

if dashboard_section == "Análisis de run":
    source_mode = st.radio(
        "Fuente de datos",
        options=["SQLite en tiempo real", "CSV exportado"],
        horizontal=True,
    )

    if source_mode == "SQLite en tiempo real":
        control_a, control_b = st.columns([1, 1.4])
        with control_a:
            refresh_seconds = st.slider(
                "Refresco (segundos)", min_value=30, max_value=120, value=60, step=10
            )
            auto_refresh = st.checkbox("Auto-refresco", value=False)
            if auto_refresh:
                trigger_autorefresh(
                    interval_ms=refresh_seconds * 1000, key="live_metrics_refresh"
                )
        with control_b:
            live_runs = list_live_metric_dbs()
            live_run_names = [run_name for run_name, _ in live_runs]
            if live_run_names:
                selected_live_run = st.selectbox(
                    "Run en vivo",
                    options=live_run_names,
                    index=len(live_run_names) - 1,
                )
            else:
                st.info(
                    "No se encontraron bases de datos live. Inicia un entrenamiento nuevo para generar metrics.sqlite."
                )
    else:
        uploaded_files = st.file_uploader(
            "Sube los 3 CSV del run (episode, update y full)",
            type=["csv"],
            accept_multiple_files=True,
        )


def infer_dataset_kind(filename):
    name = (filename or "").lower()
    if name.endswith("_episode_data.csv"):
        return "episode"
    if name.endswith("_update_data.csv"):
        return "update"
    if name.endswith("_full_data.csv"):
        return "full"
    return "unknown"


def run_base_from_filename(filename):
    if not filename:
        return None
    suffixes = ["_episode_data.csv", "_update_data.csv", "_full_data.csv"]
    for suffix in suffixes:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return os.path.splitext(filename)[0]


def load_related_datasets(uploaded_name, uploaded_df):
    datasets = {}
    primary_kind = infer_dataset_kind(uploaded_name)
    if primary_kind in ("episode", "update", "full"):
        datasets[primary_kind] = uploaded_df
    else:
        datasets["uploaded"] = uploaded_df

    run_base = run_base_from_filename(uploaded_name)
    if not run_base:
        return datasets

    csv_dir = os.path.join("data", "csv")
    related_files = {
        "episode": f"{run_base}_episode_data.csv",
        "update": f"{run_base}_update_data.csv",
        "full": f"{run_base}_full_data.csv",
    }

    for kind, file_name in related_files.items():
        if kind in datasets:
            continue
        file_path = os.path.join(csv_dir, file_name)
        if os.path.exists(file_path):
            try:
                related_df = pd.read_csv(file_path)
                datasets[kind] = normalize_dataframe_columns(related_df)
            except Exception:
                pass

    if "full" not in datasets and "episode" in datasets and "update" in datasets:
        try:
            ep = datasets["episode"].copy()
            up = datasets["update"].copy()
            if "Step" in ep.columns and "Step" in up.columns:
                datasets["full"] = ep.merge(
                    up, on="Step", how="outer", suffixes=("", "_update")
                )
        except Exception:
            pass

    return datasets


def datasets_from_uploaded_files(uploaded_files_list):
    datasets = {}
    invalid_files = []
    duplicate_kinds = []
    run_bases = []

    for up_file in uploaded_files_list:
        kind = infer_dataset_kind(up_file.name)
        if kind not in ("episode", "update", "full"):
            invalid_files.append(up_file.name)
            continue

        run_base = run_base_from_filename(up_file.name)
        if run_base:
            run_bases.append(run_base)

        if kind in datasets:
            duplicate_kinds.append(kind)
            continue

        temp_df = pd.read_csv(up_file)
        temp_df.columns = [str(c).strip() for c in temp_df.columns]
        datasets[kind] = temp_df

    unique_run_bases = sorted(set(run_bases))
    if len(unique_run_bases) > 1:
        return {
            "datasets": {},
            "error": f"Los archivos pertenecen a runs distintos: {', '.join(unique_run_bases)}",
            "invalid_files": invalid_files,
            "duplicate_kinds": duplicate_kinds,
            "run_base": None,
        }

    run_base = unique_run_bases[0] if unique_run_bases else None
    return {
        "datasets": datasets,
        "error": None,
        "invalid_files": invalid_files,
        "duplicate_kinds": duplicate_kinds,
        "run_base": run_base,
    }


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def normalize_dataframe_columns(df):
    """Devuelve un DataFrame con nombres de columna normalizados (strip)."""
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(df)
    normalized = df.copy()
    normalized.columns = [str(c).strip() for c in normalized.columns]
    return normalized


def has_numeric_values(df, column_name):
    if column_name not in df.columns:
        return False
    return to_numeric(df[column_name]).notna().any()


def numeric_columns(df, exclude=None):
    exclude = set(exclude or [])
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if has_numeric_values(df, col):
            cols.append(col)
    return cols


def grouped_columns(columns):
    groups = defaultdict(list)
    for col in columns:
        group_name = col.split("/")[0] if "/" in col else "General"
        groups[group_name].append(col)
    return dict(groups)


def load_csv_run_datasets(run_base, csv_dir=os.path.join("data", "csv")):
    datasets = {}
    if not run_base:
        return datasets

    for kind in ("episode", "update", "full"):
        file_path = os.path.join(csv_dir, f"{run_base}_{kind}_data.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df.columns = [str(c).strip() for c in df.columns]
                datasets[kind] = df
            except Exception:
                pass

    if "full" not in datasets and "episode" in datasets and "update" in datasets:
        try:
            ep = datasets["episode"].copy()
            up = datasets["update"].copy()
            if "Step" in ep.columns and "Step" in up.columns:
                datasets["full"] = ep.merge(
                    up, on="Step", how="outer", suffixes=("", "_update")
                )
        except Exception:
            pass

    return datasets


def list_csv_run_bases(csv_dir=os.path.join("data", "csv")):
    if not os.path.exists(csv_dir):
        return []

    run_bases = []
    for file_name in os.listdir(csv_dir):
        if file_name.endswith("_full_data.csv"):
            run_base = run_base_from_filename(file_name)
            if run_base:
                run_bases.append(run_base)

    return sorted(set(run_bases))


# Función auxiliar para crear gráficas con media móvil
def plot_metric(
    df, x_col, y_col, title, rolling_window=30, color="#4a9eff", invert_good=False
):
    fig = go.Figure()
    y_series = to_numeric(df[y_col])

    # Línea cruda (transparente)
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=y_series,
            mode="lines",
            line=dict(color=color, width=1),
            opacity=0.3,
            name="Crudo",
        )
    )

    # Media móvil
    if rolling_window > 0:
        rolling_series = y_series.rolling(window=rolling_window, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=rolling_series,
                mode="lines",
                line=dict(color=color, width=2.5),
                name=f"Media ({rolling_window} ep)",
            )
        )

    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#333"),
        yaxis=dict(showgrid=True, gridcolor="#333"),
    )
    return fig


GENERATED_METRICS_BY_AXIS = {
    "episode": [
        "Reward/Raw_Episode",
        "Reward/Average_100_Episodes",
        "Reward/Components/Speed_Bonus",
        "Reward/Components/Acceleration_Reward",
        "Reward/Components/Lane_Centering",
        "Reward/Components/Heading_Alignment",
        "Reward/Components/Smooth_Penalty",
        "Reward/Components/Invasion_Penalty",
        "Reward/Components/Road_Penalty",
        "Reward/Components/Progress_Bonus",
        "Reward/Components/Idle_Penalty",
        "Reward/Components/Shield_Intensity_Mean",
        "Training/Success_Rate",
        "Training/Crash_Rate",
        "Training/Offroad_Rate",
        "Training/Episode_Length",
        "Training/Curriculum_NPC",
        "Safety/Shield_Activations",
        "Safety/Shield_Rate",
        "Safety/Min_Vehicle_Distance_m",
        "Safety/Min_Pedestrian_Distance_m",
        "Safety/Min_Front_Dynamic",
        "CARLA/Mean_Speed_kmh",
        "CARLA/Mean_Lateral_Offset_Norm",
        "CARLA/Mean_Heading_Error_deg",
        "CARLA/Total_Distance",
        "CARLA/Lane_Invasions_Ep",
        "CARLA/Collisions_Ep",
        "CARLA/Speed_Compliance_Rate",
        "CARLA/Mean_Speed_Limit_kmh",
        "CARLA/Mean_Dist_Left_Edge",
        "CARLA/Mean_Dist_Right_Edge",
        "CARLA/Min_Dist_Left_Edge",
        "CARLA/Min_Dist_Right_Edge",
        "CARLA/Mean_Road_Curvature",
        "CARLA/Mean_Road_Edge_LIDAR",
        "Outcome/Type",
        "Outcome/Stuck_Rate",
        "Outcome/Timeout",
        "Outcome/Crash",
        "Outcome/Stuck",
        "Outcome/Offroad",
        "Outcome/Success",
        "Outcome/Parked_Fraction",
        "Safety/Semantic/Dynamic_Interventions",
        "Safety/Semantic/Static_Interventions",
        "Safety/Semantic/Pedestrian_Interventions",
        "Safety/Semantic/Safe_Step_Rate",
        "Safety/Semantic/Warning_Step_Rate",
        "Safety/Semantic/Critical_Step_Rate",
    ],
    "update": [
        "Loss/Policy_Loss",
        "Loss/Value_Loss",
        "Loss/Grad_Norm",
        "Training/Entropy",
        "Training/Approx_KL",
        "Training/Epochs_Run",
        "Training/Epochs_Rejected",
        "Training/Shielded_Fraction",
        "Training/Learning_Rate",
    ],
}


def _is_integer_like_series(series, tol=1e-9):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return False
    return (clean - clean.round()).abs().max() <= tol


def classify_metric_kind(metric_name, series):
    name = str(metric_name or "").lower()
    clean = pd.to_numeric(series, errors="coerce").dropna()

    if metric_name == "Outcome/Type":
        return "outcome"
    if clean.empty:
        return "empty"

    min_value = float(clean.min())
    max_value = float(clean.max())
    unique_count = int(clean.nunique())
    integer_like = _is_integer_like_series(clean)

    if (
        "rate" in name
        or "compliance" in name
        or (min_value >= 0.0 and max_value <= 1.0)
    ):
        return "rate"
    if "learning_rate" in name or "approx_kl" in name or "entropy" in name:
        return "optimizer"
    if "loss" in name or "grad_norm" in name:
        return "loss"
    if (
        "collision" in name
        or "invasion" in name
        or "activation" in name
        or "npc" in name
        or "episode_length" in name
    ):
        return "count"
    if integer_like and unique_count <= 12:
        return "discrete"
    return "continuous"


def plot_metric_area(df, x_col, y_col, title, rolling_window=30, color="#3ecf8e"):
    fig = go.Figure()
    y_series = to_numeric(df[y_col])
    smoothed = y_series.rolling(window=max(rolling_window, 1), min_periods=1).mean()

    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=smoothed,
            mode="lines",
            fill="tozeroy",
            line=dict(color=color, width=2.5),
            name="Media móvil",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=y_series,
            mode="lines",
            line=dict(color=color, width=1),
            opacity=0.25,
            name="Crudo",
        )
    )

    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#333"),
        yaxis=dict(showgrid=True, gridcolor="#333"),
    )
    return fig


def plot_metric_histogram(series, title, color="#4a9eff"):
    numeric = to_numeric(series).dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=numeric,
            nbinsx=40,
            marker_color=color,
            opacity=0.85,
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#333"),
        yaxis=dict(showgrid=True, gridcolor="#333"),
    )
    return fig


def plot_metric_box(series, title, color="#f59e0b"):
    numeric = to_numeric(series).dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=numeric,
            boxpoints="outliers",
            marker=dict(color=color),
            line=dict(color=color),
            name="Distribución",
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#333"),
    )
    return fig


def plot_metric_discrete_bars(series, title, color="#8b5cf6"):
    values = to_numeric(series).dropna()
    counts = values.astype(int).value_counts().sort_index()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(v) for v in counts.index],
            y=counts.values,
            marker_color=color,
            text=counts.values,
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#333"),
    )
    return fig


def plot_outcome_pie(series, title):
    outcome_labels = {
        0: "timeout",
        1: "collision",
        2: "stuck",
        3: "out_of_road",
        4: "success",
    }
    values = to_numeric(series).dropna().astype(int)
    counts = values.value_counts().sort_index()
    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=[outcome_labels.get(v, f"outcome_{v}") for v in counts.index],
            values=counts.values,
            hole=0.35,
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_metric_visualizations(
    metric_df, full_df, metric_name, rolling_window=30, key_prefix=""
):
    series = (
        to_numeric(metric_df[metric_name])
        if metric_name in metric_df.columns
        else pd.Series(dtype=float)
    )
    if not series.notna().any():
        value_counts = (
            full_df[metric_name].astype(str).value_counts(dropna=False).head(12)
        )
        fig_cat = go.Figure()
        fig_cat.add_trace(
            go.Bar(
                x=value_counts.index.tolist(),
                y=value_counts.values.tolist(),
                marker_color="#a855f7",
            )
        )
        fig_cat.update_layout(
            title=f"{metric_name} (top valores)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#333"),
        )
        st.plotly_chart(
            fig_cat,
            width="stretch",
            key=chart_key(key_prefix, metric_name, "categorical"),
        )
        st.caption(
            f"Cobertura: {(full_df[metric_name].notna().mean() * 100 if len(full_df) else 0.0):.1f}%"
        )
        return

    kind = classify_metric_kind(metric_name, series)
    chart_a, chart_b = st.columns(2)

    with chart_a:
        if kind == "outcome":
            st.plotly_chart(
                plot_metric_discrete_bars(series, f"{metric_name} - distribución"),
                width="stretch",
                key=chart_key(key_prefix, metric_name, "primary", "outcome_bar"),
            )
        elif kind == "rate":
            st.plotly_chart(
                plot_metric_area(
                    metric_df,
                    "_step_x",
                    metric_name,
                    title=f"{metric_name} - evolución",
                    rolling_window=rolling_window,
                    color="#3ecf8e",
                ),
                width="stretch",
                key=chart_key(key_prefix, metric_name, "primary", "rate_area"),
            )
        elif kind == "discrete":
            st.plotly_chart(
                plot_metric_discrete_bars(series, f"{metric_name} - frecuencia"),
                width="stretch",
                key=chart_key(key_prefix, metric_name, "primary", "discrete_bar"),
            )
        else:
            primary_color = "#f59e0b" if kind in ("optimizer", "loss") else "#4a9eff"
            st.plotly_chart(
                plot_metric(
                    metric_df,
                    "_step_x",
                    metric_name,
                    title=f"{metric_name} - serie temporal",
                    rolling_window=rolling_window,
                    color=primary_color,
                ),
                width="stretch",
                key=chart_key(key_prefix, metric_name, "primary", "timeseries"),
            )

    with chart_b:
        if kind == "outcome":
            st.plotly_chart(
                plot_outcome_pie(series, f"{metric_name} - proporciones"),
                width="stretch",
                key=chart_key(key_prefix, metric_name, "secondary", "outcome_pie"),
            )
        elif kind in ("discrete", "count"):
            st.plotly_chart(
                plot_metric_discrete_bars(
                    series, f"{metric_name} - conteo de valores", color="#6366f1"
                ),
                width="stretch",
                key=chart_key(key_prefix, metric_name, "secondary", "count_bar"),
            )
        elif kind == "rate":
            st.plotly_chart(
                plot_metric_box(series, f"{metric_name} - dispersión", color="#22c55e"),
                width="stretch",
                key=chart_key(key_prefix, metric_name, "secondary", "rate_box"),
            )
        else:
            st.plotly_chart(
                plot_metric_histogram(series, f"{metric_name} - histograma"),
                width="stretch",
                key=chart_key(key_prefix, metric_name, "secondary", "histogram"),
            )

    st.caption(
        f"Cobertura numérica: {coverage_ratio(full_df, metric_name):.1f}% | Missing: {missing_ratio(full_df, metric_name):.1f}% | Tipo sugerido: {kind}"
    )


def plot_comparison_metric(
    df_a, df_b, label_a, label_b, x_col, y_col, title, rolling_window=30
):
    fig = go.Figure()
    run_specs = [
        (df_a, label_a, "#4a9eff"),
        (df_b, label_b, "#3ecf8e"),
    ]

    for frame, label, color in run_specs:
        if x_col not in frame.columns or y_col not in frame.columns:
            continue

        x_series = to_numeric(frame[x_col])
        if x_series.notna().sum() == 0:
            x_series = pd.Series(
                range(1, len(frame) + 1), index=frame.index, dtype=float
            )

        y_series = to_numeric(frame[y_col])
        if not y_series.notna().any():
            continue

        fig.add_trace(
            go.Scatter(
                x=x_series,
                y=y_series,
                mode="lines",
                line=dict(color=color, width=1),
                opacity=0.25,
                name=f"{label} crudo",
            )
        )

        if rolling_window > 0:
            fig.add_trace(
                go.Scatter(
                    x=x_series,
                    y=y_series.rolling(window=rolling_window, min_periods=1).mean(),
                    mode="lines",
                    line=dict(color=color, width=2.5),
                    name=f"{label} media",
                )
            )

    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#333"),
        yaxis=dict(showgrid=True, gridcolor="#333"),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def build_comparison_summary_rows(df, label):
    def metric_or_na(column_name, reducer="mean"):
        s = safe_series(df, column_name).dropna()
        if s.empty:
            return None
        if reducer == "last":
            return float(s.iloc[-1])
        if reducer == "max":
            return float(s.max())
        return float(s.mean())

    step_series = (
        to_numeric(df["Step"]) if "Step" in df.columns else pd.Series(dtype=float)
    )
    step_max = int(step_series.max()) if step_series.notna().any() else 0
    reward_mean = metric_or_na("Reward/Raw_Episode", "mean")
    success_rate = outcome_rate(df, 4.0) if "Outcome/Type" in df.columns else None
    collision_rate = outcome_rate(df, 1.0) if "Outcome/Type" in df.columns else None
    timeout_rate = outcome_rate(df, 0.0) if "Outcome/Type" in df.columns else None
    outroad_rate = outcome_rate(df, 3.0) if "Outcome/Type" in df.columns else None
    shield_rate = metric_or_na("Safety/Shield_Rate", "mean")

    rows = [
        {"Métrica": "Filas", label: len(df)},
        {"Métrica": "Step máximo", label: step_max},
        {"Métrica": "Reward medio", label: reward_mean},
        {"Métrica": "Éxito (Outcome=4)", label: success_rate},
        {"Métrica": "Colisión (Outcome=1)", label: collision_rate},
        {"Métrica": "Timeout (Outcome=0)", label: timeout_rate},
        {"Métrica": "Out-of-road (Outcome=3)", label: outroad_rate},
        {"Métrica": "Shield rate medio", label: shield_rate},
        {
            "Métrica": "Speed compliance media",
            label: metric_or_na("CARLA/Speed_Compliance_Rate", "mean"),
        },
        {
            "Métrica": "Distancia total media",
            label: metric_or_na("CARLA/Total_Distance", "mean"),
        },
        {"Métrica": "KL último", label: metric_or_na("Training/Approx_KL", "last")},
        {"Métrica": "Entropy última", label: metric_or_na("Training/Entropy", "last")},
        {"Métrica": "LR último", label: metric_or_na("Training/Learning_Rate", "last")},
        {"Métrica": "GradNorm último", label: metric_or_na("Loss/Grad_Norm", "last")},
    ]
    return rows


def missing_ratio(df, column_name):
    if column_name not in df.columns or len(df) == 0:
        return 100.0
    return (1.0 - df[column_name].notna().mean()) * 100


def coverage_ratio(df, column_name):
    if column_name not in df.columns or len(df) == 0:
        return 0.0
    return to_numeric(df[column_name]).notna().mean() * 100


def format_float(value, precision=3):
    if pd.isna(value):
        return "NA"
    return f"{value:.{precision}f}"


def chart_key(*parts):
    raw = "||".join(str(p) for p in parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"chart_{digest}"


def build_column_profile_table(df):
    rows = []
    for col in df.columns:
        numeric_series = to_numeric(df[col])
        numeric_valid = numeric_series.notna().sum()
        non_null = df[col].notna().sum()
        row = {
            "Columna": col,
            "Grupo": col.split("/")[0] if "/" in col else "General",
            "No nulos": int(non_null),
            "Cobertura %": round((non_null / len(df) * 100) if len(df) else 0.0, 2),
            "Tipo": "Numérica" if numeric_valid > 0 else "Categórica/Texto",
        }

        if numeric_valid > 0:
            row.update(
                {
                    "Media": round(numeric_series.mean(), 6),
                    "Std": round(numeric_series.std(), 6),
                    "Min": round(numeric_series.min(), 6),
                    "Max": round(numeric_series.max(), 6),
                    "Último": round(numeric_series.dropna().iloc[-1], 6)
                    if numeric_valid > 0
                    else None,
                }
            )
        else:
            mode_series = df[col].mode(dropna=True)
            row.update(
                {
                    "Media": None,
                    "Std": None,
                    "Min": None,
                    "Max": None,
                    "Último": None,
                    "Valores únicos": int(df[col].nunique(dropna=True)),
                    "Moda": mode_series.iloc[0] if not mode_series.empty else "NA",
                }
            )
        rows.append(row)

    profile_df = pd.DataFrame(rows)
    return profile_df.sort_values(["Grupo", "Columna"]).reset_index(drop=True)


def split_stats(df, cols):
    half_point = len(df) // 2
    first_half = df.iloc[:half_point] if half_point > 0 else df
    second_half = df.iloc[half_point:] if half_point > 0 else df

    rows = []
    for col in cols:
        c1 = safe_mean(first_half, col)
        c2 = safe_mean(second_half, col)
        rows.append((col, c1, c2, c2 - c1))
    return rows


def chunked(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def safe_series(df, column_name):
    """Devuelve una serie numérica segura o una serie vacía si la columna no existe."""
    if column_name not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column_name], errors="coerce")


def safe_mean(df, column_name):
    series = safe_series(df, column_name)
    return series.mean() if not series.empty else 0.0


def safe_last(df, column_name):
    series = safe_series(df, column_name).dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def outcome_rate(df, outcome_value):
    if "Outcome/Type" not in df.columns or len(df) == 0:
        return 0.0
    return (
        pd.to_numeric(df["Outcome/Type"], errors="coerce") == outcome_value
    ).mean() * 100


def split_mean_delta(first_half, second_half, column_name):
    first = safe_mean(first_half, column_name)
    second = safe_mean(second_half, column_name)
    return first, second, second - first


def data_coverage(df, columns):
    coverage = {}
    for col in columns:
        if col in df.columns and len(df) > 0:
            valid_ratio = pd.to_numeric(df[col], errors="coerce").notna().mean() * 100
            coverage[col] = valid_ratio
        else:
            coverage[col] = 0.0
    return coverage


def ppo_update_metric_report(update_df):
    """Resume métricas críticas del eje update para el análisis LLM."""
    required_metrics = [
        "Training/Approx_KL",
        "Training/Entropy",
        "Training/Learning_Rate",
        "Loss/Grad_Norm",
    ]

    if update_df is None or update_df.empty:
        return {
            "lines": ["- No disponible: no hay dataset update cargado."],
            "present_count": 0,
            "required_count": len(required_metrics),
        }

    lines = []
    present_count = 0

    for metric in required_metrics:
        series = safe_series(update_df, metric)
        valid = int(series.notna().sum())
        total = len(update_df)
        coverage = (valid / total * 100.0) if total > 0 else 0.0

        if valid == 0:
            lines.append(f"- {metric}: NO DISPONIBLE (cobertura 0.0%)")
            continue

        present_count += 1
        first_half = series.iloc[: max(len(series) // 2, 1)]
        second_half = series.iloc[len(series) // 2 :] if len(series) > 1 else series

        mean_total = series.mean()
        last_value = series.dropna().iloc[-1]
        first_mean = first_half.mean()
        second_mean = second_half.mean()
        trend = second_mean - first_mean

        lines.append(
            f"- {metric}: cobertura={coverage:.1f}%, media={mean_total:.6f}, "
            f"último={last_value:.6f}, min={series.min():.6f}, max={series.max():.6f}, "
            f"delta(2ª-1ª)={trend:+.6f}"
        )

    return {
        "lines": lines,
        "present_count": present_count,
        "required_count": len(required_metrics),
    }


def ppo_metric_snapshot(update_df):
    """Devuelve snapshot estructurado de métricas PPO críticas para el prompt del LLM."""
    update_df = normalize_dataframe_columns(update_df)

    metric_names = {
        "approx_kl": "Training/Approx_KL",
        "entropy": "Training/Entropy",
        "learning_rate": "Training/Learning_Rate",
        "grad_norm": "Loss/Grad_Norm",
    }

    snapshot = {}
    available_count = 0

    for key, col_name in metric_names.items():
        series = safe_series(update_df, col_name)
        valid = int(series.notna().sum()) if not series.empty else 0

        if valid == 0:
            snapshot[key] = {
                "column": col_name,
                "status": "missing",
                "coverage_pct": 0.0,
                "last": None,
                "mean": None,
                "min": None,
                "max": None,
            }
            continue

        numeric_values = series.dropna()
        total = len(update_df)
        last_value = (
            float(numeric_values.iloc[-1]) if not numeric_values.empty else None
        )
        mean_value = float(numeric_values.mean()) if not numeric_values.empty else None
        min_value = float(numeric_values.min()) if not numeric_values.empty else None
        max_value = float(numeric_values.max()) if not numeric_values.empty else None

        has_finite_stats = all(
            value is not None and math.isfinite(value)
            for value in (last_value, mean_value, min_value, max_value)
        )

        if not has_finite_stats:
            snapshot[key] = {
                "column": col_name,
                "status": "missing",
                "coverage_pct": round((valid / max(total, 1)) * 100.0, 2),
                "last": None,
                "mean": None,
                "min": None,
                "max": None,
                "reason": "invalid_numeric_values",
            }
            continue

        available_count += 1
        snapshot[key] = {
            "column": col_name,
            "status": "available",
            "coverage_pct": round((valid / max(total, 1)) * 100.0, 2),
            "last": last_value,
            "mean": mean_value,
            "min": min_value,
            "max": max_value,
        }

    snapshot["available_count"] = available_count
    snapshot["required_count"] = len(metric_names)
    return snapshot


def canonical_ppo_lines_from_snapshot(snapshot):
    label_map = {
        "approx_kl": "Approximate KL Divergence (approx_kl)",
        "entropy": "Entropy",
        "learning_rate": "Learning Rate",
        "grad_norm": "Gradient Norm (grad_norm)",
    }
    ordered_keys = ("approx_kl", "entropy", "learning_rate", "grad_norm")
    lines = []

    for key in ordered_keys:
        metric = snapshot.get(key, {})
        status = metric.get("status", "missing")
        label = label_map.get(key, key)

        if status == "available":
            line = (
                f"- {label}: status=available, "
                f"last={metric.get('last', 0.0):.6f}, "
                f"coverage_pct={metric.get('coverage_pct', 0.0):.2f}, "
                f"min={metric.get('min', 0.0):.6f}, max={metric.get('max', 0.0):.6f}"
            )
        else:
            reason = metric.get("reason", "not_present")
            line = (
                f"- {label}: status=missing, "
                f"coverage_pct={metric.get('coverage_pct', 0.0):.2f}, reason={reason}"
            )
        lines.append(line)

    return lines


def llm_has_ppo_placeholder_response(text):
    if not text:
        return False
    lowered = text.lower()
    placeholder_patterns = [
        r"value\s*unavailable",
        r"no\s*disponible",
        r"\[value\s*unavailable\]",
        r"\[no\s*disponible\]",
        r"coverage_pct\s*=\s*\[?value\s*unavailable\]?",
        r"last\s*=\s*\[?value\s*unavailable\]?",
        r"\b(?:approx_kl|entropy|learning_rate|grad_norm|last|coverage_pct)\b[\s:=\-]*[\"']?(?:value\s*unavailable|no\s*disponible)[\"']?",
    ]
    return any(re.search(pattern, lowered) for pattern in placeholder_patterns)


datasets = {}
run_base = None

if source_mode == "SQLite en tiempo real" and selected_live_run:
    live_run_map = dict(live_runs)
    selected_db = live_run_map.get(selected_live_run)
    if selected_db:
        datasets = load_datasets_from_sqlite(selected_db, selected_live_run)
        run_base = selected_live_run
        st.caption(f"Fuente en vivo: {selected_db}")
elif source_mode == "CSV exportado" and uploaded_files:
    upload_state = datasets_from_uploaded_files(uploaded_files)
    if upload_state["error"]:
        st.error(upload_state["error"])
        st.stop()

    datasets = upload_state["datasets"]
    if upload_state["invalid_files"]:
        st.warning(
            "Se ignoraron archivos que no siguen el formato esperado "
            "(_episode_data.csv, _update_data.csv, _full_data.csv): "
            + ", ".join(upload_state["invalid_files"])
        )
    if upload_state["duplicate_kinds"]:
        st.warning(
            "Hay datasets repetidos por tipo y se ignoraron duplicados: "
            + ", ".join(sorted(set(upload_state["duplicate_kinds"])))
        )

    run_base = upload_state["run_base"]
    if run_base and len(datasets) < 3:
        any_kind = next(iter(datasets.keys())) if datasets else None
        if any_kind is not None:
            datasets = load_related_datasets(
                f"{run_base}_{any_kind}_data.csv", datasets[any_kind]
            )

if dashboard_section == "Análisis de run" and datasets:
    datasets = {
        kind: normalize_dataframe_columns(frame) for kind, frame in datasets.items()
    }

if datasets:
    has_all_three = all(k in datasets for k in ("episode", "update", "full"))
    if has_all_three:
        st.success("Carga completa detectada: episode + update + full.")
    else:
        st.warning(
            "Para métricas completas sube los 3 archivos: episode_data.csv, update_data.csv y full_data.csv."
        )

    if not datasets:
        st.error(
            "No se pudo cargar ningún dataset válido. Revisa el nombre y formato de los archivos."
        )
        st.stop()

    dataset_options = []
    if "episode" in datasets:
        dataset_options.append("episode")
    if "update" in datasets:
        dataset_options.append("update")
    if "full" in datasets:
        dataset_options.append("full")
    if "uploaded" in datasets:
        dataset_options.append("uploaded")

    if not dataset_options:
        st.error("No hay datasets disponibles para visualizar.")
        st.stop()

    default_kind = "full" if "full" in dataset_options else dataset_options[0]
    if default_kind not in dataset_options:
        default_kind = dataset_options[0]

    selected_kind = st.selectbox(
        "Dataset para visualización principal",
        options=dataset_options,
        index=dataset_options.index(default_kind),
        format_func=lambda x: {
            "episode": "Episodio (recomendado para safety/outcomes)",
            "update": "Update PPO (loss/KL/entropy/lr)",
            "full": "Combinado completo",
            "uploaded": "Archivo subido",
        }.get(x, x),
    )
    df = datasets[selected_kind].copy()
    df.columns = [str(c).strip() for c in df.columns]

    available_info = []
    for kind in ["episode", "update", "full"]:
        if kind in datasets:
            available_info.append(kind)
    if available_info:
        st.caption(f"Datasets detectados para el run: {', '.join(available_info)}")

    if "Step" not in df.columns:
        st.error("El CSV no contiene la columna 'Step'.")
    else:
        step_series = to_numeric(df["Step"])
        if step_series.notna().sum() == 0:
            step_series = pd.Series(range(1, len(df) + 1), index=df.index, dtype=float)
        df["_step_x"] = step_series

        data_df = df.drop(columns=["_step_x"])
        profile_df = build_column_profile_table(data_df)
        all_groups = sorted(profile_df["Grupo"].unique().tolist())
        all_numeric_cols = numeric_columns(df, exclude=["_step_x"])

        st.markdown("---")

        control_col1, control_col2, control_col3 = st.columns([1.2, 1.2, 1])
        with control_col1:
            rolling_window = st.slider(
                "Ventana media móvil", min_value=0, max_value=200, value=30, step=5
            )
        with control_col2:
            groups_to_show = st.multiselect(
                "Grupos de métricas visibles",
                options=all_groups,
                default=all_groups,
            )
        with control_col3:
            max_points = st.number_input(
                "Máx. puntos por serie",
                min_value=500,
                max_value=100000,
                value=12000,
                step=500,
            )

        if len(df) > max_points:
            sampled_df = df.iloc[:: max(1, len(df) // max_points)].copy()
        else:
            sampled_df = df.copy()

        tab_resumen, tab_series, tab_datos, tab_llm = st.tabs(
            [
                "Resumen Ejecutivo",
                "Series por Grupo",
                "Cobertura y Tabla",
                "Analista IA",
            ]
        )

        with tab_resumen:
            st.subheader("Resumen Global")
            reward_series = safe_series(df, "Reward/Raw_Episode")
            outcome_series = safe_series(df, "Outcome/Type")
            valid_episode_rows = reward_series.notna().sum()

            # Para métricas PPO usamos el eje update cuando está disponible.
            ppo_df = normalize_dataframe_columns(datasets.get("update", pd.DataFrame()))

            if not ppo_df.empty and "Step" in ppo_df.columns:
                ppo_step_series = to_numeric(ppo_df["Step"])
                if ppo_step_series.notna().sum() == 0:
                    ppo_step_series = pd.Series(
                        range(1, len(ppo_df) + 1), index=ppo_df.index, dtype=float
                    )
                ppo_df["_step_x"] = ppo_step_series
            else:
                ppo_df = pd.DataFrame()

            success_rate = outcome_rate(df, 4.0)
            collision_rate = outcome_rate(df, 1.0)
            timeout_rate = outcome_rate(df, 0.0)
            outroad_rate = outcome_rate(df, 3.0)
            global_coverage = (
                profile_df["Cobertura %"].mean() if not profile_df.empty else 0.0
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Filas CSV</div><div class="metric-value">{len(df)}</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Filas de Episodio</div><div class="metric-value">{valid_episode_rows}</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Reward Medio</div><div class="metric-value">{safe_mean(df, "Reward/Raw_Episode"):.2f}</div></div>',
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Éxito (Outcome=4)</div><div class="metric-value" style="color: #3ecf8e;">{success_rate:.1f}%</div></div>',
                    unsafe_allow_html=True,
                )
            with c5:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Cobertura Global</div><div class="metric-value">{global_coverage:.1f}%</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Colisión (Outcome=1)", f"{collision_rate:.1f}%")
            with k2:
                st.metric("Out-of-road (Outcome=3)", f"{outroad_rate:.1f}%")
            with k3:
                st.metric("Timeout (Outcome=0)", f"{timeout_rate:.1f}%")
            with k4:
                st.metric(
                    "Step Máximo",
                    f"{int(step_series.max()) if step_series.notna().any() else 0}",
                )

            st.markdown("### Estabilidad PPO (Update)")

            kl_last = (
                safe_last(ppo_df, "Training/Approx_KL") if not ppo_df.empty else None
            )
            ent_last = (
                safe_last(ppo_df, "Training/Entropy") if not ppo_df.empty else None
            )
            lr_last = (
                safe_last(ppo_df, "Training/Learning_Rate")
                if not ppo_df.empty
                else None
            )
            grad_last = (
                safe_last(ppo_df, "Loss/Grad_Norm") if not ppo_df.empty else None
            )

            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Approx KL (último)</div><div class="metric-value">{format_float(kl_last, 6) if kl_last is not None else "NA"}</div></div>',
                    unsafe_allow_html=True,
                )
            with p2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Entropy (último)</div><div class="metric-value">{format_float(ent_last, 4) if ent_last is not None else "NA"}</div></div>',
                    unsafe_allow_html=True,
                )
            with p3:
                lr_value = f"{lr_last:.2e}" if lr_last is not None else "NA"
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Learning Rate (último)</div><div class="metric-value">{lr_value}</div></div>',
                    unsafe_allow_html=True,
                )
            with p4:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Grad Norm (último)</div><div class="metric-value">{format_float(grad_last, 4) if grad_last is not None else "NA"}</div></div>',
                    unsafe_allow_html=True,
                )

            if ppo_df.empty:
                st.info(
                    "No hay dataset update disponible para mostrar KL, Entropy, Learning Rate y Grad Norm."
                )
            else:
                if len(ppo_df) > max_points:
                    sampled_ppo_df = ppo_df.iloc[
                        :: max(1, len(ppo_df) // max_points)
                    ].copy()
                else:
                    sampled_ppo_df = ppo_df.copy()

                ppo_plot_specs = [
                    ("Training/Approx_KL", "#f59e0b"),
                    ("Training/Entropy", "#3ecf8e"),
                    ("Training/Learning_Rate", "#8b5cf6"),
                    ("Loss/Grad_Norm", "#ef4444"),
                ]
                available_ppo_metrics = [
                    spec
                    for spec in ppo_plot_specs
                    if spec[0] in sampled_ppo_df.columns
                    and to_numeric(sampled_ppo_df[spec[0]]).notna().any()
                ]

                if not available_ppo_metrics:
                    st.warning(
                        "El dataset update existe, pero no contiene valores numéricos válidos para KL/Entropy/LR/GradNorm."
                    )
                else:
                    for metric_pair in chunked(available_ppo_metrics, 2):
                        metric_cols = st.columns(len(metric_pair))
                        for idx, (metric_name, metric_color) in enumerate(metric_pair):
                            with metric_cols[idx]:
                                st.plotly_chart(
                                    plot_metric(
                                        sampled_ppo_df,
                                        "_step_x",
                                        metric_name,
                                        title=f"{metric_name} (update)",
                                        rolling_window=rolling_window,
                                        color=metric_color,
                                    ),
                                    width="stretch",
                                    key=chart_key("resumen", "ppo", metric_name, idx),
                                )
                                st.caption(
                                    f"Cobertura numérica: {coverage_ratio(ppo_df, metric_name):.1f}% | Missing: {missing_ratio(ppo_df, metric_name):.1f}%"
                                )

            if "Reward/Raw_Episode" in sampled_df.columns:
                st.plotly_chart(
                    plot_metric(
                        sampled_df,
                        "_step_x",
                        "Reward/Raw_Episode",
                        "Reward por episodio",
                        rolling_window=rolling_window,
                        color="#3ecf8e",
                    ),
                    width="stretch",
                    key=chart_key("resumen", "reward_raw_episode"),
                )

            risk_cols = [
                ("Training/Success_Rate", "#3ecf8e"),
                ("Training/Crash_Rate", "#ff4f4f"),
                ("Training/Offroad_Rate", "#f5a623"),
                ("Safety/Shield_Rate", "#4a9eff"),
            ]
            risk_cols = [col for col in risk_cols if col[0] in sampled_df.columns]
            if risk_cols:
                fig_risk = go.Figure()
                for col_name, color in risk_cols:
                    series = (
                        to_numeric(sampled_df[col_name])
                        .rolling(
                            rolling_window if rolling_window > 0 else 1, min_periods=1
                        )
                        .mean()
                    )
                    fig_risk.add_trace(
                        go.Scatter(
                            x=sampled_df["_step_x"],
                            y=series,
                            mode="lines",
                            name=col_name,
                            line=dict(color=color, width=2),
                        )
                    )
                fig_risk.update_layout(
                    title="Evolución de tasas críticas (media móvil)",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=True, gridcolor="#333"),
                    yaxis=dict(showgrid=True, gridcolor="#333"),
                    legend=dict(orientation="h", y=-0.25),
                )
                st.plotly_chart(
                    fig_risk, width="stretch", key=chart_key("resumen", "risk_rates")
                )

            if outcome_series.notna().any():
                outcome_counts = (
                    outcome_series.dropna().astype(int).value_counts().sort_index()
                )
                outcome_labels = {
                    0: "timeout",
                    1: "collision",
                    2: "stuck",
                    3: "out_of_road",
                    4: "success",
                }
                fig_outcome = go.Figure()
                fig_outcome.add_trace(
                    go.Bar(
                        x=[
                            outcome_labels.get(v, f"outcome_{v}")
                            for v in outcome_counts.index
                        ],
                        y=outcome_counts.values,
                        marker_color="#4a9eff",
                        text=outcome_counts.values,
                        textposition="outside",
                    )
                )
                fig_outcome.update_layout(
                    title="Distribución de Outcome/Type",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#333"),
                )
                st.plotly_chart(
                    fig_outcome,
                    width="stretch",
                    key=chart_key("resumen", "outcome_distribution"),
                )

        with tab_series:
            st.subheader("Todas las columnas representadas por grupo")
            st.caption(
                "Cada métrica se visualiza con una vista temporal o estructural y otra de distribución, para evitar análisis basado solo en líneas."
            )

            show_cross_axis_coverage = st.checkbox(
                "Incluir también métricas del resto de ejes (episode/update)",
                value=True,
                help="Activa la cobertura completa del run aunque el dataset principal sea otro.",
            )

            selected_groups = groups_to_show if groups_to_show else all_groups
            grouped = grouped_columns(data_df.columns.tolist())

            def _with_step(frame):
                temp = frame.copy()
                if "Step" in temp.columns:
                    step_vals = to_numeric(temp["Step"])
                    if step_vals.notna().sum() == 0:
                        step_vals = pd.Series(
                            range(1, len(temp) + 1), index=temp.index, dtype=float
                        )
                else:
                    step_vals = pd.Series(
                        range(1, len(temp) + 1), index=temp.index, dtype=float
                    )
                temp["_step_x"] = step_vals
                if len(temp) > max_points:
                    temp = temp.iloc[:: max(1, len(temp) // max_points)].copy()
                return temp

            axis_frames = [(selected_kind, _with_step(df), df)]
            if show_cross_axis_coverage:
                for extra_axis in ("episode", "update"):
                    if extra_axis in datasets and extra_axis != selected_kind:
                        axis_df = normalize_dataframe_columns(
                            datasets[extra_axis].copy()
                        )
                        axis_frames.append((extra_axis, _with_step(axis_df), axis_df))

            for axis_name, sampled_axis_df, axis_df_full in axis_frames:
                st.markdown(f"### Eje: {axis_name}")

                expected_for_axis = GENERATED_METRICS_BY_AXIS.get(axis_name, [])
                if expected_for_axis:
                    present_expected = [
                        m for m in expected_for_axis if m in axis_df_full.columns
                    ]
                    missing_expected = [
                        m for m in expected_for_axis if m not in axis_df_full.columns
                    ]
                    c_cov_1, c_cov_2, c_cov_3 = st.columns(3)
                    with c_cov_1:
                        st.metric("Métricas esperadas", len(expected_for_axis))
                    with c_cov_2:
                        st.metric("Presentes", len(present_expected))
                    with c_cov_3:
                        st.metric("Faltantes", len(missing_expected))
                    if missing_expected:
                        st.warning(
                            "Métricas esperadas no encontradas en este eje: "
                            + ", ".join(missing_expected)
                        )

                axis_data_df = (
                    axis_df_full.drop(columns=["_step_x"], errors="ignore")
                    if "_step_x" in axis_df_full.columns
                    else axis_df_full
                )
                axis_groups = grouped_columns(axis_data_df.columns.tolist())

                for group in selected_groups:
                    columns_in_group = axis_groups.get(group, [])
                    if not columns_in_group:
                        continue

                    with st.expander(
                        f"{group} ({len(columns_in_group)} columnas)", expanded=False
                    ):
                        for col_name in columns_in_group:
                            if col_name not in sampled_axis_df.columns:
                                continue
                            render_metric_visualizations(
                                sampled_axis_df,
                                axis_df_full,
                                col_name,
                                rolling_window=rolling_window,
                                key_prefix=f"series::{axis_name}::{group}::{col_name}",
                            )

        with tab_datos:
            st.subheader("Cobertura y calidad de columnas")
            st.dataframe(profile_df, width="stretch", height=420)

            if len(all_numeric_cols) >= 2:
                corr_df = pd.DataFrame(
                    {col: to_numeric(df[col]) for col in all_numeric_cols}
                ).corr()
                fig_corr = go.Figure(
                    data=go.Heatmap(
                        z=corr_df.values,
                        x=corr_df.columns,
                        y=corr_df.index,
                        colorscale="RdBu",
                        zmid=0,
                    )
                )
                fig_corr.update_layout(
                    title="Matriz de correlación (columnas numéricas)",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=600,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(
                    fig_corr,
                    width="stretch",
                    key=chart_key("datos", "correlation_matrix", selected_kind),
                )

            st.subheader("Tabla cruda (exploración)")
            show_columns = st.multiselect(
                "Selecciona columnas para explorar",
                options=data_df.columns.tolist(),
                default=data_df.columns.tolist()[:12],
                key="raw_table_columns",
            )

            row_start, row_end = st.slider(
                "Rango de filas",
                min_value=0,
                max_value=max(0, len(df) - 1),
                value=(0, min(200, max(0, len(df) - 1))),
                step=1,
            )
            if show_columns:
                st.dataframe(
                    data_df.loc[row_start:row_end, show_columns],
                    width="stretch",
                    height=350,
                )
            else:
                st.info("Selecciona al menos una columna para mostrar la tabla.")

        with tab_llm:
            st.subheader("Analista IA")
            modelo_seleccionado = st.text_input(
                "Modelo local de Ollama", value="gemma4:26b"
            )

            if st.button("Generar Insights de Entrenamiento"):
                with st.spinner(
                    f"Analizando datos con {modelo_seleccionado}... esto puede tardar unos segundos."
                ):
                    llm_df = datasets.get("full", df).copy()
                    llm_df.columns = [str(c).strip() for c in llm_df.columns]
                    llm_step_series = (
                        to_numeric(llm_df["Step"])
                        if "Step" in llm_df.columns
                        else pd.Series(dtype=float)
                    )
                    llm_numeric_cols = numeric_columns(llm_df, exclude=["Step"])
                    split_rows = split_stats(llm_df, llm_numeric_cols)

                    split_lines = []
                    for col_name, first_mean, second_mean, delta in split_rows:
                        split_lines.append(
                            f"- {col_name}: {format_float(first_mean, 4)} -> {format_float(second_mean, 4)} (delta {delta:+.4f})"
                        )

                    quality_lines = []
                    llm_profile_df = build_column_profile_table(llm_df)
                    for _, row in llm_profile_df.iterrows():
                        quality_lines.append(
                            f"- {row['Columna']}: cobertura={row['Cobertura %']:.1f}%, tipo={row['Tipo']}, media={row['Media'] if pd.notna(row['Media']) else 'NA'}, min={row['Min'] if pd.notna(row['Min']) else 'NA'}, max={row['Max'] if pd.notna(row['Max']) else 'NA'}"
                        )

                    outcome_counts_text = "No disponible"
                    if (
                        "Outcome/Type" in llm_df.columns
                        and to_numeric(llm_df["Outcome/Type"]).notna().any()
                    ):
                        out_counts = (
                            to_numeric(llm_df["Outcome/Type"])
                            .dropna()
                            .astype(int)
                            .value_counts()
                            .sort_index()
                        )
                        total_out = out_counts.sum() if out_counts.sum() > 0 else 1
                        outcome_counts_text = "\n".join(
                            [
                                f"- outcome={k}: {v} episodios ({(v / total_out) * 100:.1f}%)"
                                for k, v in out_counts.items()
                            ]
                        )

                    corr_lines = []
                    if (
                        "Reward/Raw_Episode" in llm_df.columns
                        and len(llm_numeric_cols) > 1
                    ):
                        corr_matrix = pd.DataFrame(
                            {c: to_numeric(llm_df[c]) for c in llm_numeric_cols}
                        ).corr()
                        if "Reward/Raw_Episode" in corr_matrix.columns:
                            reward_corr = (
                                corr_matrix["Reward/Raw_Episode"]
                                .dropna()
                                .drop(labels=["Reward/Raw_Episode"], errors="ignore")
                            )
                            top_corr = reward_corr.reindex(
                                reward_corr.abs().sort_values(ascending=False).index
                            ).head(8)
                            for k, v in top_corr.items():
                                corr_lines.append(
                                    f"- corr(Reward/Raw_Episode, {k}) = {v:+.3f}"
                                )

                    axis_info_lines = []
                    if "episode" in datasets:
                        axis_info_lines.append(
                            f"- Filas episode_data: {len(datasets['episode'])}"
                        )
                    if "update" in datasets:
                        axis_info_lines.append(
                            f"- Filas update_data: {len(datasets['update'])}"
                        )
                    axis_info_lines.append(
                        f"- Dataset base para prompt: {'full' if 'full' in datasets else selected_kind}"
                    )

                    axis_coverage_lines = []
                    for axis_name in ("episode", "update"):
                        expected = GENERATED_METRICS_BY_AXIS.get(axis_name, [])
                        axis_df = normalize_dataframe_columns(
                            datasets.get(axis_name, pd.DataFrame())
                        )
                        present = [m for m in expected if m in axis_df.columns]
                        missing = [m for m in expected if m not in axis_df.columns]
                        axis_coverage_lines.append(
                            f"- {axis_name}: esperadas={len(expected)}, presentes={len(present)}, faltantes={len(missing)}"
                        )
                        if missing:
                            axis_coverage_lines.append(
                                f"  faltantes_{axis_name}: {', '.join(missing)}"
                            )

                    coverage_priority_cols = [
                        "Reward/Raw_Episode",
                        "Training/Success_Rate",
                        "Training/Crash_Rate",
                        "Training/Offroad_Rate",
                        "Safety/Shield_Rate",
                        "Outcome/Type",
                        "CARLA/Speed_Compliance_Rate",
                        "CARLA/Total_Distance",
                        "Training/Approx_KL",
                        "Training/Entropy",
                        "Training/Learning_Rate",
                        "Loss/Grad_Norm",
                    ]
                    coverage_focus_lines = []
                    for c_name in coverage_priority_cols:
                        cov_val = coverage_ratio(llm_df, c_name)
                        coverage_focus_lines.append(
                            f"- {c_name}: cobertura_numerica={cov_val:.1f}%"
                        )

                    update_df_for_llm = normalize_dataframe_columns(
                        datasets.get("update", pd.DataFrame())
                    )
                    ppo_update_report = ppo_update_metric_report(update_df_for_llm)
                    ppo_snapshot = ppo_metric_snapshot(update_df_for_llm)
                    ppo_presence_line = (
                        f"- Métricas PPO críticas disponibles: "
                        f"{ppo_update_report['present_count']}/{ppo_update_report['required_count']}"
                    )

                    inconsistent_ppo_keys = []
                    for metric_key in (
                        "approx_kl",
                        "entropy",
                        "learning_rate",
                        "grad_norm",
                    ):
                        metric_info = ppo_snapshot.get(metric_key, {})
                        if metric_info.get("status") == "available" and (
                            metric_info.get("last") is None
                            or metric_info.get("coverage_pct") is None
                        ):
                            inconsistent_ppo_keys.append(metric_key)

                    if inconsistent_ppo_keys:
                        ppo_integrity_line = (
                            "- Integridad snapshot PPO: INCONSISTENTE en "
                            + ", ".join(inconsistent_ppo_keys)
                        )
                    else:
                        ppo_integrity_line = "- Integridad snapshot PPO: OK (sin contradicciones status/valores)"

                    ppo_snapshot_text = json.dumps(
                        ppo_snapshot, ensure_ascii=False, indent=2
                    )
                    ppo_canonical_lines = canonical_ppo_lines_from_snapshot(
                        ppo_snapshot
                    )

                    quality_focus_tokens = [
                        "Safety/",
                        "Training/",
                        "Outcome/",
                        "Reward/Raw_Episode",
                        "CARLA/Collisions_Ep",
                        "CARLA/Lane_Invasions_Ep",
                        "CARLA/Mean_Speed_kmh",
                        "CARLA/Speed_Compliance_Rate",
                        "CARLA/Total_Distance",
                    ]
                    quality_lines_reduced = [
                        line
                        for line in quality_lines
                        if any(token in line for token in quality_focus_tokens)
                    ]
                    if not quality_lines_reduced:
                        quality_lines_reduced = quality_lines[:20]

                    split_focus_metrics = [
                        "Reward/Raw_Episode",
                        "Training/Success_Rate",
                        "Training/Crash_Rate",
                        "Training/Offroad_Rate",
                        "Safety/Shield_Rate",
                        "CARLA/Speed_Compliance_Rate",
                        "CARLA/Total_Distance",
                    ]
                    split_lines_reduced = [
                        line
                        for line in split_lines
                        if any(
                            line.startswith(f"- {metric}:")
                            for metric in split_focus_metrics
                        )
                    ]
                    if not split_lines_reduced:
                        split_lines_reduced = split_lines[:20]

                    prompt_ppo = f"""
Eres un auditor de estabilidad PPO para RL en CARLA.

CONTEXTO:
- Dataset base: {selected_kind}
- Filas update_data: {len(update_df_for_llm)}
- Step máximo update: {int(to_numeric(update_df_for_llm["Step"]).max()) if ("Step" in update_df_for_llm.columns and len(update_df_for_llm) > 0 and to_numeric(update_df_for_llm["Step"]).notna().any()) else 0}

FUENTE AUTORITATIVA PPO:
{ppo_snapshot_text}

RESUMEN PPO:
{ppo_presence_line}
{"\n".join(ppo_update_report["lines"])}

COBERTURA DE MÉTRICAS EN EL RUN:
{"\n".join(axis_coverage_lines)}

SANITY CHECK:
{ppo_integrity_line}

BLOQUE PPO CANÓNICO (REPRODUCE LOS NÚMEROS LITERALMENTE):
{"\n".join(ppo_canonical_lines)}

REGLAS OBLIGATORIAS:
1. Debes incluir exactamente 4 bullets, uno por métrica: approx_kl, entropy, learning_rate, grad_norm.
2. Si status=available, cita last y coverage_pct numéricos (sin placeholders).
3. Si status=missing, explica que es faltante real de datos.
4. No inventes datos ni redondeos inconsistentes con el bloque canónico.

FORMATO:
- approx_kl: status=..., last=..., coverage_pct=..., lectura técnica breve.
- entropy: ...
- learning_rate: ...
- grad_norm: ...
"""

                    prompt_general = f"""
Eres un evaluador técnico de entrenamiento RL para conducción autónoma en CARLA con prioridad absoluta de seguridad.

CONTEXTO DEL EXPERIMENTO:
- Dataset cargado en dashboard: {selected_kind}
- Filas totales dataset base: {len(llm_df)}
- Step máximo: {int(llm_step_series.max()) if llm_step_series.notna().any() else 0}
- Convención Outcome/Type: 0=timeout, 1=collision, 2=stuck, 3=out_of_road, 4=success

EJES TEMPORALES DISPONIBLES:
{"\n".join(axis_info_lines)}

COBERTURA DE MÉTRICAS ESPERADAS POR EJE:
{"\n".join(axis_coverage_lines)}

COBERTURA EN MÉTRICAS CRÍTICAS:
{"\n".join(coverage_focus_lines)}

RESUMEN DE OUTCOMES:
{outcome_counts_text}

PERFIL RELEVANTE DE COLUMNAS (subset para ahorrar contexto):
{"\n".join(quality_lines_reduced)}

COMPARATIVA PRIMERA MITAD vs SEGUNDA MITAD (subset relevante):
{"\n".join(split_lines_reduced)}

CORRELACIONES RELEVANTES CON REWARD:
{"\n".join(corr_lines) if corr_lines else "- No disponible por falta de datos"}

NOTA: El diagnóstico numérico de PPO se genera en otra llamada dedicada. Aquí no repitas ni inventes valores PPO.

FORMATO DE RESPUESTA (Markdown):
- Diagnóstico de seguridad (4-7 bullets)
- Diagnóstico de comportamiento CARLA (3-6 bullets)
- Riesgo global del run: Bajo/Medio/Alto con justificación numérica breve
- Top 5 ajustes priorizados para la siguiente corrida (con impacto esperado y trade-off)
"""

                    try:
                        ppo_response = ollama.chat(
                            model=modelo_seleccionado,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "<|think|>Eres un auditor PPO estricto. Debes respetar datos autoritativos y evitar placeholders.",
                                },
                                {"role": "user", "content": prompt_ppo},
                            ],
                        )

                        general_response = ollama.chat(
                            model=modelo_seleccionado,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "<|think|>Eres un asistente experto en RL para CARLA enfocado en seguridad y comportamiento en pista.",
                                },
                                {"role": "user", "content": prompt_general},
                            ],
                        )

                        ppo_llm_text = ppo_response["message"]["content"]
                        general_llm_text = general_response["message"]["content"]
                        snapshot_has_available = (
                            ppo_snapshot.get("available_count", 0) > 0
                        )
                        llm_used_placeholders = llm_has_ppo_placeholder_response(
                            ppo_llm_text
                        )

                        if snapshot_has_available and llm_used_placeholders:
                            st.warning(
                                "La IA devolvió placeholders PPO (por ejemplo 'value unavailable') a pesar de que hay valores reales. "
                                "Se añade abajo un bloque PPO corregido desde el snapshot del dashboard."
                            )
                            corrected_section = (
                                "\n\n### Corrección automática PPO (fuente: snapshot dashboard)\n"
                                + "\n".join(ppo_canonical_lines)
                            )
                            ppo_llm_text = ppo_llm_text + corrected_section

                        st.info("**Conclusiones de la IA:**")
                        st.markdown("### PPO (bloque autoritativo dashboard)")
                        st.markdown("\n".join(ppo_canonical_lines))
                        st.markdown("### PPO (análisis LLM dedicado)")
                        st.markdown(ppo_llm_text)
                        st.markdown(
                            "### Seguridad y Comportamiento (análisis LLM dedicado)"
                        )
                        st.markdown(general_llm_text)
                        with st.expander("Ver prompt PPO enviado al LLM"):
                            st.code(prompt_ppo)
                        with st.expander("Ver prompt general enviado al LLM"):
                            st.code(prompt_general)

                    except Exception as e:
                        st.error(
                            f"Error al conectar con Ollama. ¿Te aseguraste de ejecutar 'ollama serve' o tener la app abierta? Detalle del error: {e}"
                        )
elif dashboard_section == "Análisis de run":
    if source_mode == "SQLite en tiempo real":
        st.info("Selecciona un run live para comenzar la visualización en tiempo real.")
    else:
        st.info("Sube al menos un CSV para cargar datos del run.")

if dashboard_section != "Comparación de runs":
    st.stop()

st.markdown("---")
st.header("Comparación visual de 2 entrenamientos")
st.caption(
    "Selecciona dos runs y superpone sus métricas para comparar rendimiento, estabilidad y seguridad."
)

comparison_source_mode = st.selectbox(
    "Fuente para la comparación",
    options=["SQLite en tiempo real", "CSV exportado"],
    index=0,
    key="comparison_source_mode",
)

if comparison_source_mode == "SQLite en tiempo real":
    comparison_runs = list_live_metric_dbs()
    comparison_run_names = [run_name for run_name, _ in comparison_runs]
    comparison_run_map = dict(comparison_runs)

    def comparison_loader(run_name):
        return load_datasets_from_sqlite(comparison_run_map[run_name], run_name)
else:
    comparison_run_names = list_csv_run_bases()

    def comparison_loader(run_base):
        return load_csv_run_datasets(run_base)


if len(comparison_run_names) < 2:
    if comparison_source_mode == "SQLite en tiempo real":
        st.info("Necesitas al menos 2 runs live activos para comparar.")
    else:
        st.info("No hay suficientes runs exportados en data/csv para comparar.")
else:
    compare_a_col, compare_b_col, compare_c_col = st.columns([1, 1, 1.1])
    with compare_a_col:
        compare_run_a = st.selectbox(
            "Run A",
            options=comparison_run_names,
            index=max(len(comparison_run_names) - 2, 0),
            key="compare_run_a",
        )

    compare_b_options = [
        run_name for run_name in comparison_run_names if run_name != compare_run_a
    ]
    with compare_b_col:
        compare_run_b = st.selectbox(
            "Run B",
            options=compare_b_options,
            index=0,
            key="compare_run_b",
        )

    with compare_c_col:
        comparison_rolling_window = st.slider(
            "Ventana media móvil",
            min_value=0,
            max_value=200,
            value=30,
            step=5,
            key="comparison_rolling_window",
        )

    compare_datasets_a = comparison_loader(compare_run_a)
    compare_datasets_b = comparison_loader(compare_run_b)

    common_kinds = [
        kind
        for kind in ("full", "episode", "update")
        if kind in compare_datasets_a and kind in compare_datasets_b
    ]
    if not common_kinds:
        st.warning("No hay un dataset común entre ambos runs para comparar.")
    else:
        compare_kind = st.selectbox(
            "Tipo de dataset a comparar",
            options=common_kinds,
            index=0,
            format_func=lambda kind: {
                "episode": "Episodio",
                "update": "Update PPO",
                "full": "Combinado completo",
            }.get(kind, kind),
            key="comparison_kind",
        )

        compare_df_a = compare_datasets_a[compare_kind].copy()
        compare_df_b = compare_datasets_b[compare_kind].copy()
        compare_df_a.columns = [str(c).strip() for c in compare_df_a.columns]
        compare_df_b.columns = [str(c).strip() for c in compare_df_b.columns]

        if "Step" not in compare_df_a.columns or "Step" not in compare_df_b.columns:
            st.error(
                "Ambos datasets deben incluir la columna 'Step' para poder compararlos."
            )
        else:
            compare_df_a["_step_x"] = to_numeric(compare_df_a["Step"])
            compare_df_b["_step_x"] = to_numeric(compare_df_b["Step"])
            if compare_df_a["_step_x"].notna().sum() == 0:
                compare_df_a["_step_x"] = pd.Series(
                    range(1, len(compare_df_a) + 1),
                    index=compare_df_a.index,
                    dtype=float,
                )
            if compare_df_b["_step_x"].notna().sum() == 0:
                compare_df_b["_step_x"] = pd.Series(
                    range(1, len(compare_df_b) + 1),
                    index=compare_df_b.index,
                    dtype=float,
                )

            shared_numeric_cols = sorted(
                set(numeric_columns(compare_df_a, exclude=["Step", "_step_x"]))
                & set(numeric_columns(compare_df_b, exclude=["Step", "_step_x"]))
            )

            if not shared_numeric_cols:
                st.warning(
                    "No hay columnas numéricas compartidas entre ambos runs para dibujar series superpuestas."
                )
            else:
                key_metrics = [
                    "Reward/Raw_Episode",
                    "Training/Success_Rate",
                    "Training/Crash_Rate",
                    "Training/Offroad_Rate",
                    "Safety/Shield_Rate",
                    "Training/Approx_KL",
                    "Training/Entropy",
                    "Training/Learning_Rate",
                ]
                default_compare_metrics = [
                    metric for metric in key_metrics if metric in shared_numeric_cols
                ]
                if not default_compare_metrics:
                    default_compare_metrics = shared_numeric_cols[:6]

                compare_metrics = st.multiselect(
                    "Métricas a superponer",
                    options=shared_numeric_cols,
                    default=default_compare_metrics,
                    key="comparison_metrics",
                )

                summary_a = pd.DataFrame(
                    build_comparison_summary_rows(compare_df_a, compare_run_a)
                )
                summary_b = pd.DataFrame(
                    build_comparison_summary_rows(compare_df_b, compare_run_b)
                )
                summary_df = summary_a.merge(summary_b, on="Métrica", how="outer")

                if (
                    compare_run_a in summary_df.columns
                    and compare_run_b in summary_df.columns
                ):
                    summary_df["Delta (B - A)"] = pd.to_numeric(
                        summary_df[compare_run_b], errors="coerce"
                    ) - pd.to_numeric(summary_df[compare_run_a], errors="coerce")

                st.subheader("Resumen rápido")
                st.dataframe(summary_df, width="stretch", height=320)

                st.subheader("Cobertura de métricas compartidas")
                coverage_rows = []
                for metric_name in shared_numeric_cols:
                    cov_a = coverage_ratio(compare_df_a, metric_name)
                    cov_b = coverage_ratio(compare_df_b, metric_name)
                    coverage_rows.append(
                        {
                            "Métrica": metric_name,
                            f"Cobertura {compare_run_a} (%)": cov_a,
                            f"Cobertura {compare_run_b} (%)": cov_b,
                            "Delta cobertura (B - A)": cov_b - cov_a,
                        }
                    )
                coverage_df = pd.DataFrame(coverage_rows)
                st.dataframe(coverage_df, width="stretch", height=260)

                if (
                    "Outcome/Type" in compare_df_a.columns
                    and "Outcome/Type" in compare_df_b.columns
                ):
                    labels = {
                        0: "timeout",
                        1: "collision",
                        2: "stuck",
                        3: "out_of_road",
                        4: "success",
                    }
                    out_a = (
                        to_numeric(compare_df_a["Outcome/Type"])
                        .dropna()
                        .astype(int)
                        .value_counts()
                        .sort_index()
                    )
                    out_b = (
                        to_numeric(compare_df_b["Outcome/Type"])
                        .dropna()
                        .astype(int)
                        .value_counts()
                        .sort_index()
                    )
                    all_keys = sorted(
                        set(out_a.index.tolist()) | set(out_b.index.tolist())
                    )
                    fig_out_cmp = go.Figure()
                    fig_out_cmp.add_trace(
                        go.Bar(
                            x=[labels.get(k, str(k)) for k in all_keys],
                            y=[int(out_a.get(k, 0)) for k in all_keys],
                            name=compare_run_a,
                            marker_color="#4a9eff",
                        )
                    )
                    fig_out_cmp.add_trace(
                        go.Bar(
                            x=[labels.get(k, str(k)) for k in all_keys],
                            y=[int(out_b.get(k, 0)) for k in all_keys],
                            name=compare_run_b,
                            marker_color="#3ecf8e",
                        )
                    )
                    fig_out_cmp.update_layout(
                        title="Comparación de outcomes por run",
                        barmode="group",
                        margin=dict(l=20, r=20, t=40, b=20),
                        height=320,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor="#333"),
                    )
                    st.plotly_chart(
                        fig_out_cmp,
                        width="stretch",
                        key=chart_key(
                            "comparison",
                            compare_run_a,
                            compare_run_b,
                            compare_kind,
                            "outcomes_bar",
                        ),
                    )

                st.subheader("Series superpuestas")
                if not compare_metrics:
                    st.info(
                        "Selecciona al menos una métrica para dibujar la comparación."
                    )
                else:
                    for metric_group in chunked(compare_metrics, 2):
                        metric_cols = st.columns(len(metric_group))
                        for idx, metric_name in enumerate(metric_group):
                            with metric_cols[idx]:
                                st.plotly_chart(
                                    plot_comparison_metric(
                                        compare_df_a,
                                        compare_df_b,
                                        compare_run_a,
                                        compare_run_b,
                                        "_step_x",
                                        metric_name,
                                        f"{metric_name} - {compare_run_a} vs {compare_run_b}",
                                        rolling_window=comparison_rolling_window,
                                    ),
                                    width="stretch",
                                    key=chart_key(
                                        "comparison",
                                        compare_run_a,
                                        compare_run_b,
                                        compare_kind,
                                        metric_name,
                                        idx,
                                    ),
                                )
