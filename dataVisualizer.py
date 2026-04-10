import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ollama
from collections import defaultdict
import os
import importlib

from src.live_metrics import list_live_metric_dbs, load_datasets_from_sqlite


def trigger_autorefresh(interval_ms, key):
    # Compatibilidad entre versiones de Streamlit y extensión opcional.
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=interval_ms, key=key)
        return

    try:
        module = importlib.import_module("streamlit_autorefresh")
        st_autorefresh = getattr(module, "st_autorefresh")
        st_autorefresh(interval=interval_ms, key=key)
    except Exception:
        st.caption("Auto-refresco no disponible en esta versión de Streamlit. Usa el botón Rerun o actualiza Streamlit.")

# Configuración de la página
st.set_page_config(page_title="RL Training Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Estilos CSS personalizados para simular el aspecto del HTML
st.markdown("""
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
""", unsafe_allow_html=True)

st.title("Dashboard de Entrenamiento RL - Agente Individual")
st.markdown("Analiza la estabilidad, el rendimiento y el comportamiento de tu agente en CARLA.")

# 1. Fuente de datos
source_mode = st.radio(
    "Fuente de datos",
    options=["SQLite en tiempo real", "CSV exportado"],
    horizontal=True,
)

uploaded_files = []
live_runs = []
selected_live_run = None

if source_mode == "SQLite en tiempo real":
    control_a, control_b = st.columns([1, 1.4])
    with control_a:
        refresh_seconds = st.slider("Refresco (segundos)", min_value=1, max_value=10, value=2, step=1)
        auto_refresh = st.checkbox("Auto-refresco", value=True)
        if auto_refresh:
            trigger_autorefresh(interval_ms=refresh_seconds * 1000, key="live_metrics_refresh")
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
            st.info("No se encontraron bases de datos live. Inicia un entrenamiento nuevo para generar metrics.sqlite.")
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
            return filename[:-len(suffix)]
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
                datasets[kind] = pd.read_csv(file_path)
            except Exception:
                pass

    if "full" not in datasets and "episode" in datasets and "update" in datasets:
        try:
            ep = datasets["episode"].copy()
            up = datasets["update"].copy()
            if "Step" in ep.columns and "Step" in up.columns:
                datasets["full"] = ep.merge(up, on="Step", how="outer", suffixes=("", "_update"))
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
    return pd.to_numeric(series, errors='coerce')


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
        group_name = col.split('/')[0] if '/' in col else 'General'
        groups[group_name].append(col)
    return dict(groups)

# Función auxiliar para crear gráficas con media móvil
def plot_metric(df, x_col, y_col, title, rolling_window=30, color="#4a9eff", invert_good=False):
    fig = go.Figure()
    y_series = to_numeric(df[y_col])
    
    # Línea cruda (transparente)
    fig.add_trace(go.Scatter(
        x=df[x_col], y=y_series,
        mode='lines', line=dict(color=color, width=1), opacity=0.3,
        name='Crudo'
    ))
    
    # Media móvil
    if rolling_window > 0:
        rolling_series = y_series.rolling(window=rolling_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df[x_col], y=rolling_series, 
            mode='lines', line=dict(color=color, width=2.5),
            name=f'Media ({rolling_window} ep)'
        ))

    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='#333'),
        yaxis=dict(showgrid=True, gridcolor='#333')
    )
    return fig


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


def build_column_profile_table(df):
    rows = []
    for col in df.columns:
        numeric_series = to_numeric(df[col])
        numeric_valid = numeric_series.notna().sum()
        non_null = df[col].notna().sum()
        row = {
            "Columna": col,
            "Grupo": col.split('/')[0] if '/' in col else 'General',
            "No nulos": int(non_null),
            "Cobertura %": round((non_null / len(df) * 100) if len(df) else 0.0, 2),
            "Tipo": "Numérica" if numeric_valid > 0 else "Categórica/Texto",
        }

        if numeric_valid > 0:
            row.update({
                "Media": round(numeric_series.mean(), 6),
                "Std": round(numeric_series.std(), 6),
                "Min": round(numeric_series.min(), 6),
                "Max": round(numeric_series.max(), 6),
                "Último": round(numeric_series.dropna().iloc[-1], 6) if numeric_valid > 0 else None,
            })
        else:
            mode_series = df[col].mode(dropna=True)
            row.update({
                "Media": None,
                "Std": None,
                "Min": None,
                "Max": None,
                "Último": None,
                "Valores únicos": int(df[col].nunique(dropna=True)),
                "Moda": mode_series.iloc[0] if not mode_series.empty else "NA",
            })
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
        yield items[i:i + chunk_size]


def safe_series(df, column_name):
    """Devuelve una serie numérica segura o una serie vacía si la columna no existe."""
    if column_name not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column_name], errors='coerce')


def safe_mean(df, column_name):
    series = safe_series(df, column_name)
    return series.mean() if not series.empty else 0.0


def outcome_rate(df, outcome_value):
    if 'Outcome/Type' not in df.columns or len(df) == 0:
        return 0.0
    return (pd.to_numeric(df['Outcome/Type'], errors='coerce') == outcome_value).mean() * 100


def split_mean_delta(first_half, second_half, column_name):
    first = safe_mean(first_half, column_name)
    second = safe_mean(second_half, column_name)
    return first, second, second - first


def data_coverage(df, columns):
    coverage = {}
    for col in columns:
        if col in df.columns and len(df) > 0:
            valid_ratio = pd.to_numeric(df[col], errors='coerce').notna().mean() * 100
            coverage[col] = valid_ratio
        else:
            coverage[col] = 0.0
    return coverage

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
            datasets = load_related_datasets(f"{run_base}_{any_kind}_data.csv", datasets[any_kind])

if datasets:

    has_all_three = all(k in datasets for k in ("episode", "update", "full"))
    if has_all_three:
        st.success("Carga completa detectada: episode + update + full.")
    else:
        st.warning("Para métricas completas sube los 3 archivos: episode_data.csv, update_data.csv y full_data.csv.")

    if not datasets:
        st.error("No se pudo cargar ningún dataset válido. Revisa el nombre y formato de los archivos.")
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

    
    if 'Step' not in df.columns:
        st.error("El CSV no contiene la columna 'Step'.")
    else:
        step_series = to_numeric(df['Step'])
        if step_series.notna().sum() == 0:
            step_series = pd.Series(range(1, len(df) + 1), index=df.index, dtype=float)
        df['_step_x'] = step_series

        data_df = df.drop(columns=['_step_x'])
        profile_df = build_column_profile_table(data_df)
        all_groups = sorted(profile_df['Grupo'].unique().tolist())
        all_numeric_cols = numeric_columns(df, exclude=['_step_x'])

        st.markdown("---")

        control_col1, control_col2, control_col3 = st.columns([1.2, 1.2, 1])
        with control_col1:
            rolling_window = st.slider("Ventana media móvil", min_value=0, max_value=200, value=30, step=5)
        with control_col2:
            groups_to_show = st.multiselect(
                "Grupos de métricas visibles",
                options=all_groups,
                default=all_groups,
            )
        with control_col3:
            max_points = st.number_input("Máx. puntos por serie", min_value=500, max_value=100000, value=12000, step=500)

        if len(df) > max_points:
            sampled_df = df.iloc[::max(1, len(df) // max_points)].copy()
        else:
            sampled_df = df.copy()

        tab_resumen, tab_series, tab_datos, tab_llm = st.tabs([
            "Resumen Ejecutivo",
            "Series por Grupo",
            "Cobertura y Tabla",
            "Analista IA",
        ])

        with tab_resumen:
            st.subheader("Resumen Global")
            reward_series = safe_series(df, 'Reward/Raw_Episode')
            outcome_series = safe_series(df, 'Outcome/Type')
            valid_episode_rows = reward_series.notna().sum()

            success_rate = outcome_rate(df, 4.0)
            collision_rate = outcome_rate(df, 1.0)
            timeout_rate = outcome_rate(df, 0.0)
            outroad_rate = outcome_rate(df, 3.0)
            global_coverage = profile_df['Cobertura %'].mean() if not profile_df.empty else 0.0

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Filas CSV</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Filas de Episodio</div><div class="metric-value">{valid_episode_rows}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Reward Medio</div><div class="metric-value">{safe_mean(df, "Reward/Raw_Episode"):.2f}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Éxito (Outcome=4)</div><div class="metric-value" style="color: #3ecf8e;">{success_rate:.1f}%</div></div>', unsafe_allow_html=True)
            with c5:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Cobertura Global</div><div class="metric-value">{global_coverage:.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Colisión (Outcome=1)", f"{collision_rate:.1f}%")
            with k2:
                st.metric("Out-of-road (Outcome=3)", f"{outroad_rate:.1f}%")
            with k3:
                st.metric("Timeout (Outcome=0)", f"{timeout_rate:.1f}%")
            with k4:
                st.metric("Step Máximo", f"{int(step_series.max()) if step_series.notna().any() else 0}")

            if 'Reward/Raw_Episode' in sampled_df.columns:
                st.plotly_chart(
                    plot_metric(sampled_df, '_step_x', 'Reward/Raw_Episode', "Reward por episodio", rolling_window=rolling_window, color="#3ecf8e"),
                    use_container_width=True,
                )

            risk_cols = [
                ('Training/Success_Rate', '#3ecf8e'),
                ('Training/Crash_Rate', '#ff4f4f'),
                ('Training/Offroad_Rate', '#f5a623'),
                ('Safety/Shield_Rate', '#4a9eff'),
            ]
            risk_cols = [col for col in risk_cols if col[0] in sampled_df.columns]
            if risk_cols:
                fig_risk = go.Figure()
                for col_name, color in risk_cols:
                    series = to_numeric(sampled_df[col_name]).rolling(rolling_window if rolling_window > 0 else 1, min_periods=1).mean()
                    fig_risk.add_trace(go.Scatter(
                        x=sampled_df['_step_x'],
                        y=series,
                        mode='lines',
                        name=col_name,
                        line=dict(color=color, width=2),
                    ))
                fig_risk.update_layout(
                    title='Evolución de tasas críticas (media móvil)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='#333'),
                    yaxis=dict(showgrid=True, gridcolor='#333'),
                    legend=dict(orientation='h', y=-0.25),
                )
                st.plotly_chart(fig_risk, use_container_width=True)

            if outcome_series.notna().any():
                outcome_counts = outcome_series.dropna().astype(int).value_counts().sort_index()
                outcome_labels = {
                    0: 'timeout',
                    1: 'collision',
                    2: 'stuck',
                    3: 'out_of_road',
                    4: 'success',
                }
                fig_outcome = go.Figure()
                fig_outcome.add_trace(go.Bar(
                    x=[outcome_labels.get(v, f'outcome_{v}') for v in outcome_counts.index],
                    y=outcome_counts.values,
                    marker_color='#4a9eff',
                    text=outcome_counts.values,
                    textposition='outside',
                ))
                fig_outcome.update_layout(
                    title='Distribución de Outcome/Type',
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=280,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#333'),
                )
                st.plotly_chart(fig_outcome, use_container_width=True)

        with tab_series:
            st.subheader("Todas las columnas representadas por grupo")
            st.caption("Cada columna se muestra como serie temporal (si es numérica) o distribución categórica (si es texto).")

            selected_groups = groups_to_show if groups_to_show else all_groups
            grouped = grouped_columns(data_df.columns.tolist())

            for group in selected_groups:
                columns_in_group = grouped.get(group, [])
                if not columns_in_group:
                    continue

                with st.expander(f"{group} ({len(columns_in_group)} columnas)", expanded=False):
                    for col_pair in chunked(columns_in_group, 2):
                        plot_cols = st.columns(len(col_pair))
                        for idx, col_name in enumerate(col_pair):
                            with plot_cols[idx]:
                                numeric_series = to_numeric(sampled_df[col_name]) if col_name in sampled_df.columns else pd.Series(dtype=float)
                                if numeric_series.notna().any():
                                    fig = plot_metric(
                                        sampled_df,
                                        '_step_x',
                                        col_name,
                                        title=col_name,
                                        rolling_window=rolling_window,
                                        color="#4a9eff",
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.caption(
                                        f"Cobertura numérica: {coverage_ratio(df, col_name):.1f}% | Missing: {missing_ratio(df, col_name):.1f}%"
                                    )
                                else:
                                    value_counts = sampled_df[col_name].astype(str).value_counts(dropna=False).head(10)
                                    fig_cat = go.Figure()
                                    fig_cat.add_trace(go.Bar(
                                        x=value_counts.index.tolist(),
                                        y=value_counts.values.tolist(),
                                        marker_color='#a855f7',
                                    ))
                                    fig_cat.update_layout(
                                        title=f"{col_name} (top valores)",
                                        margin=dict(l=20, r=20, t=40, b=20),
                                        height=250,
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        xaxis=dict(showgrid=False),
                                        yaxis=dict(showgrid=True, gridcolor='#333'),
                                    )
                                    st.plotly_chart(fig_cat, use_container_width=True)
                                    st.caption(f"Cobertura: {(df[col_name].notna().mean() * 100 if len(df) else 0.0):.1f}%")

        with tab_datos:
            st.subheader("Cobertura y calidad de columnas")
            st.dataframe(profile_df, use_container_width=True, height=420)

            if len(all_numeric_cols) >= 2:
                corr_df = pd.DataFrame({col: to_numeric(df[col]) for col in all_numeric_cols}).corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale='RdBu',
                    zmid=0,
                ))
                fig_corr.update_layout(
                    title='Matriz de correlación (columnas numéricas)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=600,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Tabla cruda (exploración)")
            show_columns = st.multiselect(
                "Selecciona columnas para explorar",
                options=data_df.columns.tolist(),
                default=data_df.columns.tolist()[:12],
                key='raw_table_columns',
            )

            row_start, row_end = st.slider(
                "Rango de filas",
                min_value=0,
                max_value=max(0, len(df) - 1),
                value=(0, min(200, max(0, len(df) - 1))),
                step=1,
            )
            if show_columns:
                st.dataframe(data_df.loc[row_start:row_end, show_columns], use_container_width=True, height=350)
            else:
                st.info("Selecciona al menos una columna para mostrar la tabla.")

        with tab_llm:
            st.subheader("Analista IA")
            modelo_seleccionado = st.text_input("Modelo local de Ollama", value='gemma3:12b')

            if st.button("Generar Insights de Entrenamiento"):
                with st.spinner(f"Analizando datos con {modelo_seleccionado}... esto puede tardar unos segundos."):
                    llm_df = datasets.get("full", df).copy()
                    llm_df.columns = [str(c).strip() for c in llm_df.columns]
                    llm_step_series = to_numeric(llm_df['Step']) if 'Step' in llm_df.columns else pd.Series(dtype=float)
                    llm_numeric_cols = numeric_columns(llm_df, exclude=['Step'])
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
                    if 'Outcome/Type' in llm_df.columns and to_numeric(llm_df['Outcome/Type']).notna().any():
                        out_counts = to_numeric(llm_df['Outcome/Type']).dropna().astype(int).value_counts().sort_index()
                        total_out = out_counts.sum() if out_counts.sum() > 0 else 1
                        outcome_counts_text = "\n".join([
                            f"- outcome={k}: {v} episodios ({(v / total_out) * 100:.1f}%)" for k, v in out_counts.items()
                        ])

                    corr_lines = []
                    if 'Reward/Raw_Episode' in llm_df.columns and len(llm_numeric_cols) > 1:
                        corr_matrix = pd.DataFrame({c: to_numeric(llm_df[c]) for c in llm_numeric_cols}).corr()
                        if 'Reward/Raw_Episode' in corr_matrix.columns:
                            reward_corr = corr_matrix['Reward/Raw_Episode'].dropna().drop(labels=['Reward/Raw_Episode'], errors='ignore')
                            top_corr = reward_corr.reindex(reward_corr.abs().sort_values(ascending=False).index).head(8)
                            for k, v in top_corr.items():
                                corr_lines.append(f"- corr(Reward/Raw_Episode, {k}) = {v:+.3f}")

                    axis_info_lines = []
                    if "episode" in datasets:
                        axis_info_lines.append(f"- Filas episode_data: {len(datasets['episode'])}")
                    if "update" in datasets:
                        axis_info_lines.append(f"- Filas update_data: {len(datasets['update'])}")
                    axis_info_lines.append(f"- Dataset base para prompt: {'full' if 'full' in datasets else selected_kind}")

                    prompt_estadistico = f"""
Eres un evaluador técnico de entrenamiento RL para conducción autónoma en CARLA con prioridad absoluta de seguridad.

CONTEXTO DEL EXPERIMENTO:
- Dataset cargado en dashboard: {selected_kind}
- Filas totales CSV (prompt): {len(llm_df)}
- Step máximo: {int(llm_step_series.max()) if llm_step_series.notna().any() else 0}
- Convención Outcome/Type: 0=timeout, 1=collision, 2=stuck, 3=out_of_road, 4=success

EJES TEMPORALES DISPONIBLES:
{"\n".join(axis_info_lines)}

RESUMEN DE OUTCOMES:
{outcome_counts_text}

PERFIL COMPLETO DE COLUMNAS (todas las columnas):
{"\n".join(quality_lines)}

COMPARATIVA PRIMERA MITAD vs SEGUNDA MITAD (todas las métricas numéricas):
{"\n".join(split_lines)}

CORRELACIONES RELEVANTES CON REWARD:
{"\n".join(corr_lines) if corr_lines else '- No disponible por falta de datos'}

REGLAS DE ANÁLISIS (OBLIGATORIAS):
1. Prioriza seguridad (collision, out_of_road, timeout, shield, invasiones de carril, distancias mínimas, cumplimiento de velocidad).
2. Evalúa estabilidad PPO (KL, entropy, losses, grad norm, learning rate).
3. Si una métrica clave tiene cobertura <70%, menciona impacto en fiabilidad.
4. No inventes causas: solo inferencias sustentadas por métricas.
5. Determina si el shield ayuda a seguridad o solo compensa una política débil.

FORMATO DE RESPUESTA (Markdown):
- Diagnóstico de seguridad (4-7 bullets)
- Diagnóstico de estabilidad PPO (3-6 bullets)
- Diagnóstico de comportamiento CARLA (3-6 bullets)
- Riesgo global del run: Bajo/Medio/Alto con justificación numérica breve
- Top 5 ajustes priorizados para la siguiente corrida (con impacto esperado y trade-off)
"""

                    try:
                        response = ollama.chat(model=modelo_seleccionado, messages=[
                            {
                                'role': 'system',
                                'content': 'Eres un asistente experto en RL para CARLA. Eres estricto con seguridad, estabilidad PPO y evidencia numérica.'
                            },
                            {
                                'role': 'user',
                                'content': prompt_estadistico
                            }
                        ])

                        st.info("**Conclusiones de la IA:**")
                        st.markdown(response['message']['content'])
                        with st.expander("Ver prompt enviado al LLM"):
                            st.code(prompt_estadistico)

                    except Exception as e:
                        st.error(f"Error al conectar con Ollama. ¿Te aseguraste de ejecutar 'ollama serve' o tener la app abierta? Detalle del error: {e}")
else:
    if source_mode == "SQLite en tiempo real":
        st.info("Selecciona un run live para comenzar la visualización en tiempo real.")
    else:
        st.info("Sube al menos un CSV para cargar datos del run.")