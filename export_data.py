import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_tensorboard_data(root_logdir="./runs"):
    runs = [f.path for f in os.scandir(root_logdir) if f.is_dir()]

    if not runs:
        print(f"No se encontraron carpetas de entrenamiento en {root_logdir}")
        return

    print(f"Encontrados {len(runs)} entrenamientos. Procesando...")

    for run_path in runs:
        run_name = os.path.basename(run_path)
        print(f"--> Exportando: {run_name}")

        event_acc = EventAccumulator(run_path)
        event_acc.Reload()

        tags = event_acc.Tags()["scalars"]

        if not tags:
            print(f"    (Sin métricas escalares encontradas)")
            continue

        # Usaremos el 'step' como índice común
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

            output_file = f"{run_name}_full_data.csv"
            df_final.to_csv(output_file)
            print(f"    Guardado: {output_file}")


if __name__ == "__main__":
    extract_tensorboard_data()