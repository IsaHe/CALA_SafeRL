# CALA_SafeRL

Safe Reinforcement Learning (Safe RL) for autonomous driving in **CARLA**, using **PPO** plus an optional **Safety Shield** (basic or adaptive) and curriculum learning.  
This repo includes training + evaluation entrypoints, metrics logging, and utilities to run reference experiments.

> Main scripts:
> - `main_train.py` — train PPO in CARLA (optionally with a safety shield)
> - `main_eval.py` — evaluate a trained model (optionally with live dashboard)
> - `run_experiments.sh` — run baseline / basic shield / adaptive shield experiments

---

## Features

- **PPO agent** for continuous control (steering + throttle/brake).
- **Safety Shields**
  - `none`: standard PPO (no intervention)
  - `basic`: threshold-based interventions (LIDAR + waypoint/lane info)
  - `adaptive`: adaptive-horizon shield (Bicycle Model + waypoint/lane info)
- **Reward shaping** (lane centering, smoothness, progress, etc.) via `src/reward_shaper.py`
- **Curriculum learning** over NPC traffic density via `src/curriculumManager.py`
- **Metrics**
  - Training live metrics stored into SQLite (and usable from TensorBoard logs in `./runs`)
  - Evaluation safety report + optional matplotlib dashboard

---

## Repository structure (high level)

- `main_train.py` — training entrypoint (CARLA env → Shield → RewardShaper)
- `main_eval.py` — evaluation entrypoint + dashboard
- `run_experiments.sh` — scripted experiment runner
- `src/`
  - `CARLA/` — CARLA environment wrapper(s)
  - `PPO/` — PPO implementation
  - `Adaptative_Shield/` — adaptive shield implementation
  - `Metrics/` — training/eval metrics and reporters
  - `reward_shaper.py` — reward shaping wrapper
  - `safety_shield.py` — basic shield wrapper
  - `curriculumManager.py` — curriculum logic
- `tests/` — tests (if/when present)
- `docs/` — documentation assets
- `Memoria/` — LaTeX/Thesis-style report (project write-up)

---

## Requirements

You need:

1. **CARLA simulator** running (tested with CARLA **0.9.16** as referenced in `main_train.py`)
2. A Python environment with common scientific packages (NumPy, Matplotlib, etc.)
3. The CARLA Python API available to your Python environment (so `import carla` works)

> Note: exact dependency versions are not pinned in the repository root (no `requirements.txt` found at the root), so you may need to adapt to your setup.

---

## Start CARLA

Examples (from repo comments):

### Windows
Run CARLA first (example used in `main_train.py`):
```bash
CarlaUE4.exe -RenderOffScreen -carla-port=2000
```

### Linux (headless)
Example from `run_experiments.sh`:
```bash
./CarlaUE4.sh -RenderOffScreen -quality-level=Low
```

---

## Training

Train a model (default shield is `adaptive`):

```bash
python main_train.py --model_name my_model --shield_type adaptive
```

Common options:

- `--model_name` (required): base name for saving artifacts
- `--shield_type`: `none | basic | adaptive`
- `--host`, `--port`, `--tm_port`: CARLA connection
- `--map`: e.g. `Town04`
- `--num_npc`: number of NPC vehicles
- `--max_episodes`, `--max_steps`: training horizon

Models are saved under:
- `./data/models/`
- Checkpoints: `./data/models/<run_name>_checkpoints/`

Training runs/logs are saved under:
- `./runs/<run_name>/`

---

## Run reference experiments

The repo includes a convenience runner:

```bash
bash run_experiments.sh
```

Run only one experiment:

```bash
bash run_experiments.sh baseline
bash run_experiments.sh basic
bash run_experiments.sh adaptive
```

Environment variables supported by the script:
- `CARLA_HOST`, `CARLA_PORT`, `TM_PORT`
- `MAP`, `MAX_EPISODES`, `LR`

At the end, it prints suggested evaluation commands and TensorBoard hint.

---

## Evaluation

Evaluate a trained model:

```bash
python main_eval.py --model_name my_model_adaptive_final.pth --shield_type adaptive
```

Useful flags:
- `--episodes N`
- `--no_render` (collect metrics without CARLA rendering)
- `--no_dashboard` (disable matplotlib dashboard)
- `--deterministic` (disable action sampling)

The evaluation script prints per-episode outcomes (success/crash/off-road/timeout) and a final summary.  
If shield is enabled, it also reports interventions and generates a safety metrics report.

---

## Notes / Tips

- **Wrapper order matters** (used in training & enforced in eval):
  `CarlaEnv → Shield → RewardShaper`
- If you change sensors/observation dimensions, older trained models may become incompatible.
- For training curves:
  ```bash
  tensorboard --logdir ./runs
  ```

---

## License

No license file was detected in the repository root. If you intend others to use this code, consider adding a `LICENSE` file.

---

## Citation

If you use this repository in academic work, consider citing the associated report in `Memoria/` (and/or add a formal BibTeX entry here).
