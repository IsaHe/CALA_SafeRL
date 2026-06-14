"""
collect_annotation_rollouts.py - Vuelca observaciones de un checkpoint para la
anotacion de preferencias (fase 1). REQUIERE un servidor CARLA en marcha.

Reutiliza ``build_env`` de main_eval (misma cadena de wrappers que entrenamiento/
evaluacion) y la politica DETERMINISTA del checkpoint, con la normalizacion de
observaciones CONGELADA (igual que main_eval), y guarda las observaciones de 739
dim que ve el agente en un ``observations.npz`` (clave ``obs``, forma [N, 739]).

Para que el dataset contenga encuentros con vehiculos (el cuello de botella de la
evitacion de colisiones), correr con trafico denso (``--num_npc 60``) y, cuando la
fase 4 este lista, con el director de exposicion. Por ahora el trafico global da
los ~3% de pasos con encuentro que ``annotate_preferences.py`` luego sobremuestrea.

Ejemplo (con CARLA + venv TFG-RL activos):

    python tools/collect_annotation_rollouts.py \\
        --model_name learn_fix2_adaptive_final.pth \\
        --episodes 40 --num_npc 60 --shield_type adaptive \\
        --out data/preferences/observations.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_name", required=True, help="checkpoint en ./data/models o ruta"
    )
    p.add_argument("--out", default="data/preferences/observations.npz")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument(
        "--shield_type", default="adaptive", choices=["none", "basic", "adaptive"]
    )
    p.add_argument(
        "--stride", type=int, default=2, help="guarda 1 de cada `stride` pasos"
    )
    p.add_argument(
        "--max_obs", type=int, default=60000, help="cap del dataset (0=sin cap)"
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="politica estocastica (def: determinista)",
    )
    # ── flags que consume main_eval.build_env (defaults = main_eval) ──
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm_port", type=int, default=8000)
    p.add_argument("--map", type=str, default="Town04")
    p.add_argument("--num_npc", type=int, default=60)
    p.add_argument("--weather", type=str, default="ClearNoon")
    p.add_argument("--target_speed_kmh", type=float, default=30.0)
    p.add_argument("--success_distance", type=float, default=250.0)
    p.add_argument("--front_threshold", type=float, default=0.15)
    p.add_argument("--side_threshold", type=float, default=0.04)
    p.add_argument("--lateral_threshold", type=float, default=0.82)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--obs-norm", action=argparse.BooleanOptionalAction, default=True)
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Imports diferidos: requieren `carla`; mantienen el modulo importable sin sim.
    from main_eval import build_env
    from src.PPO.ppo_agent import PPOAgent

    model_path = Path("./data/models") / args.model_name
    if not model_path.exists():
        model_path = Path(args.model_name)
    if not model_path.exists():
        print(f"Model not found: {args.model_name}")
        sys.exit(1)

    env, _ = build_env(args, args.shield_type, render=False)
    agent = PPOAgent(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        normalize_obs=args.obs_norm,
    )
    agent.load(str(model_path))
    agent.policy.eval()
    agent._update_obs_stats = lambda *_a, **_k: (
        None
    )  # congelar obs-norm (como main_eval)

    deterministic = not args.stochastic
    collected: list[np.ndarray] = []
    cap = args.max_obs if args.max_obs > 0 else None

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            done = trunc = False
            step = 0
            while not (done or trunc):
                if step % args.stride == 0:
                    collected.append(np.asarray(obs, dtype=np.float32).copy())
                action, _, _, _ = agent.select_action(obs, deterministic=deterministic)
                obs, _, done, trunc, _ = env.step(action)
                step += 1
                if cap is not None and len(collected) >= cap:
                    break
            print(
                f"  ep {ep + 1}/{args.episodes}: {len(collected)} obs total", flush=True
            )
            if cap is not None and len(collected) >= cap:
                print(f"Reached max_obs={cap}; stopping.")
                break
    finally:
        env.close()

    obs_arr = np.asarray(collected, dtype=np.float32)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, obs=obs_arr)
    print(
        f"Saved {obs_arr.shape[0]} observations of dim {obs_arr.shape[1]} -> {out_path}"
    )


if __name__ == "__main__":
    main()
