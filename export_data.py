import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Tags que en este proyecto siguen eje de timestep (actualización PPO)
UPDATE_TAG_PREFIXES = (
    "Loss/",
    "Training/Approx_KL",
    "Training/Entropy",
    "Training/Learning_Rate",
)


def _read_scalar_series(event_acc: EventAccumulator, tag: str) -> pd.DataFrame | None:
    events = event_acc.Scalars(tag)
    if not events:
        return None

    steps = [int(e.step) for e in events]
    values = [e.value for e in events]

    df_temp = pd.DataFrame({"Step": steps, tag: values})
    df_temp = df_temp.drop_duplicates(subset="Step", keep="last")
    df_temp = df_temp.sort_values("Step")
    df_temp.set_index("Step", inplace=True)
    return df_temp


def _classify_axis(tag: str, series_df: pd.DataFrame) -> str:
    # Regla explícita por prefijo para evitar ambigüedades.
    if tag.startswith(UPDATE_TAG_PREFIXES):
        return "update"

    # Heurística de respaldo para tags nuevos.
    steps = series_df.index.to_series().dropna().astype(int)
    if steps.empty:
        return "episode"

    max_step = int(steps.max())
    num_points = int(steps.nunique())

    # Series densas con step pequeño tienden a ser por episodio.
    if num_points > 0 and max_step <= 5000 and (num_points / max(max_step, 1)) > 0.35:
        return "episode"

    # Series más dispersas y de step alto suelen venir de timestep PPO.
    if max_step > 5000:
        return "update"

    return "episode"


def _concat_and_save(
    frames: list[pd.DataFrame], output_file: str
) -> pd.DataFrame | None:
    if not frames:
        return None

    df_final = pd.concat(frames, axis=1)
    df_final = df_final.sort_index()
    df_final.to_csv(output_file)
    return df_final


def extract_tensorboard_data(logdir: str):
    if not logdir:
        print(f"No se encontro log de entrenamiento {logdir}")
        return

    print(f"Procesando {logdir}...")

    run_name = os.path.basename(logdir)
    print(f"--> Exportando: {run_name}")

    # Evita recorte por defecto de TensorBoard (scalars=10000).
    event_acc = EventAccumulator(
        logdir,
        size_guidance={
            "scalars": 0,
            "tensors": 0,
            "images": 0,
            "audio": 0,
            "histograms": 0,
            "compressedHistograms": 0,
        },
    )
    event_acc.Reload()

    print(f"    Tags encontrados: {event_acc.Tags()}")

    tags = event_acc.Tags()["scalars"]

    if not tags:
        print("    (Sin métricas escalares encontradas)")
        return

    data_frames = []
    episode_frames = []
    update_frames = []
    axis_summary = {"episode": 0, "update": 0}

    for tag in tags:
        df_temp = _read_scalar_series(event_acc, tag)
        if df_temp is None:
            continue

        data_frames.append(df_temp)
        axis = _classify_axis(tag, df_temp)
        axis_summary[axis] += 1
        if axis == "update":
            update_frames.append(df_temp)
        else:
            episode_frames.append(df_temp)

    if data_frames:
        os.makedirs("data/csv", exist_ok=True)

        # Export combinado (compatibilidad hacia atrás)
        output_file_full = f"data/csv/{run_name}_full_data.csv"
        _concat_and_save(data_frames, output_file_full)
        print(f"    Guardado: {output_file_full}")

        # Export por eje temporal para evitar NA estructurales.
        output_file_ep = f"data/csv/{run_name}_episode_data.csv"
        ep_df = _concat_and_save(episode_frames, output_file_ep)
        if ep_df is not None:
            print(f"    Guardado: {output_file_ep}")

        output_file_up = f"data/csv/{run_name}_update_data.csv"
        up_df = _concat_and_save(update_frames, output_file_up)
        if up_df is not None:
            print(f"    Guardado: {output_file_up}")

        print(
            "    Resumen de tags por eje: "
            f"episode={axis_summary['episode']} | update={axis_summary['update']}"
        )


if __name__ == "__main__":
    # Ejemplo de uso: extraer datos de un run específico.
    log_directory = "runs/SemanticSensorAndRichMetrics2_adaptive_20260409-191428"  # Cambia esto por tu ruta real
    extract_tensorboard_data(log_directory)
