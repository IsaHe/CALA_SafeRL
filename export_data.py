import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_tensorboard_data(logdir: str):
    if not logdir:
        print(f"No se encontro log de entrenamiento {logdir}")
        return

    print(f"Procesando {logdir}...")

    run_name = os.path.basename(logdir)
    print(f"--> Exportando: {run_name}")

    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    print(f"    Tags encontrados: {event_acc.Tags()}")

    tags = event_acc.Tags()["scalars"]

    if not tags:
        print(f"    (Sin métricas escalares encontradas)")
        return

    data_frames = []

    for tag in tags:
        events = event_acc.Scalars(tag)

        steps = [e.step for e in events]
        values = [e.value for e in events]

        df_temp = pd.DataFrame({"Step": steps, tag: values})

        df_temp = df_temp.drop_duplicates(subset="Step", keep="last")
        df_temp.set_index("Step", inplace=True)

        data_frames.append(df_temp)

    if data_frames:
        df_final = pd.concat(data_frames, axis=1)

        output_file = f"data/csv/{run_name}_full_data.csv"
        df_final.to_csv(output_file)
        print(f"    Guardado: {output_file}")