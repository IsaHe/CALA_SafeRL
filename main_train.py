"""
main_train.py - Entrypoint de entrenamiento para CARLA Safe RL

Reemplaza main_train_improved_v2.py de MetaDrive.

DIFERENCIAS CLAVE vs versión MetaDrive:
  1. CarlaEnv en lugar de MetaDriveEnv — física real, Waypoint API, sensores nativos
  2. No hay LaneAwarenessWrapper: el info dict de CarlaEnv ya contiene
     lateral_offset_norm, heading_error_norm, on_road, etc. (Waypoint API exacto)
  3. El shield puede ser 'none', 'basic' (CarlaSafetyShield) o
     'adaptive' (CarlaAdaptiveHorizonShield con BicycleModel)
  4. TensorBoard logging idéntico al original para facilitar comparación
  5. Checkpoint y best-model con la misma lógica que la versión original
  6. El servidor CARLA debe estar corriendo antes de ejecutar este script

USO:
    # Arrancar servidor CARLA primero:
    #   ./CarlaUE4.sh -RenderOffScreen   (Linux sin pantalla)
    #   CarlaUE4.exe -quality-level=Low  (Windows, bajo consumo)

    python main_train.py --model_name mi_modelo --shield_type adaptive
    python main_train.py --model_name baseline --shield_type none
    python main_train.py --model_name basic_shield --shield_type basic --map Town01
"""

import argparse
import os
import time
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
import torch

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.carla_env import CarlaEnv
from src.reward_shaper import CarlaRewardShaper
from src.safety_shield import CarlaSafetyShield
from src.adaptive_horizon_shield import CarlaAdaptiveHorizonShield
from src.ppo_agent import PPOAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_train")


# ══════════════════════════════════════════════════════════════════════
# ARGUMENTOS
# ══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description="PPO Training con Safety Shield en CARLA")

    # Identificación del experimento
    p.add_argument("--model_name", type=str, required=True,
                   help="Nombre base del modelo a entrenar")
    p.add_argument("--shield_type", type=str,
                   choices=["none", "basic", "adaptive"], default="adaptive",
                   help="Tipo de safety shield")

    # Hiperparámetros PPO
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate inicial para PPO")
    p.add_argument("--max_episodes", type=int, default=2500,
                   help="Número máximo de episodios de entrenamiento")
    p.add_argument("--max_steps", type=int, default=1000,
                   help="Pasos máximos por episodio")
    p.add_argument("--update_timestep", type=int, default=2000,
                   help="Timesteps entre actualizaciones de política")

    # Parámetros del entorno CARLA
    p.add_argument("--host", type=str, default="localhost",
                   help="Host del servidor CARLA")
    p.add_argument("--port", type=int, default=2000,
                   help="Puerto del servidor CARLA")
    p.add_argument("--tm_port", type=int, default=8000,
                   help="Puerto del TrafficManager")
    p.add_argument("--map", type=str, default="Town04",
                   help="Mapa CARLA (Town01-Town07, Town10HD...)")
    p.add_argument("--num_npc", type=int, default=20,
                   help="Número de vehículos NPC gestionados por TrafficManager")
    p.add_argument("--weather", type=str, default="ClearNoon",
                   help="Preset de clima CARLA (ClearNoon, WetSunset, CloudyNight...)")
    p.add_argument("--target_speed_kmh", type=float, default=50.0,
                   help="Velocidad objetivo para el reward (km/h)")
    p.add_argument("--success_distance", type=float, default=250.0,
                   help="Metros a recorrer para considerar episodio exitoso")

    # Parámetros del shield
    p.add_argument("--front_threshold", type=float, default=0.15,
                   help="Umbral LIDAR frontal de seguridad [0-1 norm]")
    p.add_argument("--side_threshold", type=float, default=0.04,
                   help="Umbral LIDAR lateral de seguridad [0-1 norm]")
    p.add_argument("--lateral_threshold", type=float, default=0.82,
                   help="Fracción del semi-ancho de carril antes de corregir")

    # Reward shaping
    p.add_argument("--speed_weight", type=float, default=0.05,
                   help="Peso del bonus de velocidad")
    p.add_argument("--smoothness_weight", type=float, default=0.10,
                   help="Peso de la penalización por steering brusco")
    p.add_argument("--lane_centering_weight", type=float, default=0.15,
                   help="Peso del bonus de centramiento en carril")
    p.add_argument("--lane_invasion_penalty", type=float, default=0.25,
                   help="Penalización por invasión de carril (sensor CARLA)")
    p.add_argument("--off_road_penalty", type=float, default=2.00,
                   help="Penalización por salirse de carretera")

    # Checkpoints y logging
    p.add_argument("--ckpt_freq", type=int, default=200,
                   help="Frecuencia (en episodios) de guardado de checkpoints")
    p.add_argument("--seed", type=int, default=42,
                   help="Semilla para reproducibilidad")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL ENTORNO
# ══════════════════════════════════════════════════════════════════════

def build_env(args):
    """
    Construye la cadena de wrappers en orden correcto:
        CarlaEnv  →  CarlaRewardShaper  →  [Shield]
    """
    num_lidar_rays = 240

    # 1. Entorno base CARLA
    env = CarlaEnv(
        host=args.host,
        port=args.port,
        tm_port=args.tm_port,
        map_name=args.map,
        num_npc_vehicles=args.num_npc,
        weather=args.weather,
        synchronous=True,
        fixed_delta_seconds=0.05,
        num_lidar_rays=num_lidar_rays,
        lidar_range=50.0,
        max_episode_steps=args.max_steps,
        target_speed_kmh=args.target_speed_kmh,
        success_distance=args.success_distance,
        success_reward=30.0,
        out_of_road_penalty=10.0,
        crash_penalty=10.0,
        seed=args.seed,
    )

    # 2. Reward shaping con Waypoint API
    env = CarlaRewardShaper(
        env,
        target_speed_kmh=args.target_speed_kmh,
        speed_weight=args.speed_weight,
        smoothness_weight=args.smoothness_weight,
        lane_centering_weight=args.lane_centering_weight,
        lane_invasion_penalty=args.lane_invasion_penalty,
        off_road_penalty=args.off_road_penalty,
    )

    # 3. Shield (opcional)
    if args.shield_type == "basic":
        logger.info("🛡️  Shield: CarlaSafetyShield (LIDAR + Waypoint API)")
        env = CarlaSafetyShield(
            env,
            num_lidar_rays=num_lidar_rays,
            front_threshold=args.front_threshold,
            side_threshold=args.side_threshold,
            lateral_threshold=args.lateral_threshold,
        )
    elif args.shield_type == "adaptive":
        logger.info("🛡️  Shield: CarlaAdaptiveHorizonShield (BicycleModel + Waypoint API)")
        env = CarlaAdaptiveHorizonShield(
            env,
            num_lidar_rays=num_lidar_rays,
            front_threshold_base=args.front_threshold,
            side_threshold_base=args.side_threshold,
            lateral_threshold_base=args.lateral_threshold,
        )
    else:
        logger.info("⚠️  Sin shield — PPO estándar")

    return env, num_lidar_rays


# ══════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════

def train():
    args = get_args()

    timestamp    = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name     = f"{args.model_name}_{args.shield_type}_{timestamp}"
    log_dir      = f"./runs/{run_name}"
    models_dir   = Path("./data/models")
    ckpt_dir     = models_dir / f"{run_name}_checkpoints"

    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"Run: {run_name}")
    logger.info(f"TensorBoard: tensorboard --logdir {log_dir}")

    # Rutas de modelos
    final_model_path = models_dir / f"{args.model_name}_{args.shield_type}_final.pth"
    best_model_path  = models_dir / f"{args.model_name}_{args.shield_type}_best.pth"

    # Ventanas de métricas
    reward_window  = deque(maxlen=100)
    success_window = deque(maxlen=100)
    best_avg_reward = -float("inf")

    # ── Construir entorno ──────────────────────────────────────────────
    logger.info("Connecting to CARLA and building environment...")
    env, num_lidar_rays = build_env(args)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    logger.info(f"State dim: {state_dim} | Action dim: {action_dim}")

    # ── Agente PPO ─────────────────────────────────────────────────────
    agent = PPOAgent(state_dim, action_dim, lr=args.lr)

    memory = {
        "states": [], "actions": [], "log_probs": [],
        "rewards": [], "dones": [],
    }
    timestep       = 0
    avg_reward_100 = 0.0
    success_rate   = 0.0

    logger.info(f"Starting training for {args.max_episodes} episodes...\n")

    try:
        for episode in range(1, args.max_episodes + 1):

            # LR decay lineal → 0 al final del entrenamiento
            progress = (episode - 1) / args.max_episodes
            new_lr   = max(args.lr * (1.0 - progress), 1e-6)
            agent.set_lr(new_lr)

            obs, _ = env.reset()
            episode_reward       = 0.0
            ep_shield_activations = 0

            for step in range(args.max_steps):
                timestep += 1

                action, log_prob, _ = agent.select_action(obs)
                next_obs, reward, done, truncated, info = env.step(action)

                if info.get("shield_activated", info.get("shield_active", False)):
                    ep_shield_activations += 1

                memory["states"].append(obs)

                executed = info.get("executed_action", action)
                memory["actions"].append(executed)
                # Recalcular log_prob para la acción ejecutada:
                _, new_log_prob, _, _ = agent.policy.get_action_and_value(
                    torch.FloatTensor(obs).unsqueeze(0).to(agent.device),
                    torch.FloatTensor(executed).unsqueeze(0).to(agent.device)
                )
                memory["log_probs"].append(new_log_prob.cpu().item())

                memory["rewards"].append(reward)
                memory["dones"].append(done or truncated)

                obs             = next_obs
                episode_reward += reward

                # Actualización de política
                if timestep % args.update_timestep == 0:
                    train_metrics = agent.update(memory)
                    for key in memory:
                        memory[key] = []

                    writer.add_scalar("Loss/Policy_Loss",  train_metrics["policy_loss"],  episode)
                    writer.add_scalar("Loss/Value_Loss",   train_metrics["value_loss"],   episode)
                    writer.add_scalar("Training/Entropy",  train_metrics["entropy"],      episode)
                    writer.add_scalar("Training/Approx_KL", train_metrics["approx_kl"],  episode)

                if done or truncated:
                    break

            # ── Outcome del episodio ─────────────────────────────────
            is_success = int(info.get("arrive_dest", False))
            outcome    = 0
            if info.get("collision", False):
                outcome = 1
            elif info.get("out_of_road", False):
                outcome = 3
            elif info.get("arrive_dest", False):
                outcome = 4

            reward_window.append(episode_reward)
            success_window.append(is_success)
            avg_reward_100 = float(np.mean(reward_window))
            success_rate   = float(np.mean(success_window))

            # ── TensorBoard ──────────────────────────────────────────
            writer.add_scalar("Reward/Raw_Episode",          episode_reward,      episode)
            writer.add_scalar("Reward/Average_100_Episodes", avg_reward_100,      episode)
            writer.add_scalar("Training/Success_Rate",       success_rate,        episode)
            writer.add_scalar("Training/Learning_Rate",      new_lr,              episode)
            writer.add_scalar("Training/Episode_Length",     step,                episode)
            writer.add_scalar("Safety/Shield_Activations",   ep_shield_activations, episode)
            writer.add_scalar("Outcome/Type",                outcome,             episode)

            # Métricas CARLA específicas
            writer.add_scalar("CARLA/Speed_kmh",           info.get("speed_kmh", 0.0),      episode)
            writer.add_scalar("CARLA/Total_Distance",      info.get("total_distance", 0.0), episode)
            writer.add_scalar("CARLA/Lane_Invasions_Ep",   info.get("episode_lane_invasions", 0), episode)
            writer.add_scalar("CARLA/Collisions_Ep",       info.get("episode_collisions", 0), episode)
            writer.add_scalar("CARLA/Lateral_Offset_Norm", abs(info.get("lateral_offset_norm", 0.0)), episode)

            writer.flush()

            # ── Guardar mejor modelo (a partir del episodio 500) ─────
            if episode >= 500 and avg_reward_100 > best_avg_reward + 20:
                best_avg_reward = avg_reward_100
                agent.save(str(best_model_path))
                logger.info(
                    f"★  New best model saved at Ep {episode} "
                    f"(Avg100: {best_avg_reward:.1f})"
                )

            # ── Checkpoint periódico ──────────────────────────────────
            if episode % args.ckpt_freq == 0:
                ckpt_path = ckpt_dir / f"checkpoint_ep_{episode}.pth"
                agent.save(str(ckpt_path))
                logger.info(f"Checkpoint saved: {ckpt_path.name}")

            # ── Print progreso ────────────────────────────────────────
            if episode % 10 == 0:
                logger.info(
                    f"Ep {episode:>5} | R: {episode_reward:>8.1f} | "
                    f"Avg100: {avg_reward_100:>7.1f} | "
                    f"SuccRate: {success_rate:.2f} | "
                    f"Shield: {ep_shield_activations:>3} | "
                    f"Out: {outcome}"
                )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")

    finally:
        # Guardar modelo final
        agent.save(str(final_model_path))
        logger.info(f"Final model saved: {final_model_path}")

        env.close()
        writer.close()

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING FINISHED")
        logger.info("=" * 60)
        logger.info("To evaluate this model, run:")
        eval_cmd = (
            f"python main_eval.py "
            f"--model_name {final_model_path.name} "
            f"--shield_type {args.shield_type} "
            f"--map {args.map}"
        )
        logger.info(f"\n  {eval_cmd}\n")
        logger.info("=" * 60)


if __name__ == "__main__":
    train()
