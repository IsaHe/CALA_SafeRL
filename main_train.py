"""
USO:
    # Arrancar servidor CARLA primero:
    #   .\CARLA_0.9.16\CarlaUE4.exe -RenderOffScreen -carla-port=2000

    python main_train.py --model_name mi_modelo --shield_type adaptive
"""

import argparse
import os
import time
import logging
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.carla_env import CarlaEnv
from src.reward_shaper import CarlaRewardShaper
from src.safety_shield import CarlaSafetyShield
from src.adaptive_horizon_shield import CarlaAdaptiveHorizonShield
from src.ppo_agent import PPOAgent

import export_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_train")


# ARGUMENTOS

def get_args():
    p = argparse.ArgumentParser(description="PPO Training con Safety Shield en CARLA")

    # Identificativos y configuración general
    p.add_argument("--model_name", type=str, required=True,
                   help="Nombre base del modelo a entrenar")
    p.add_argument("--shield_type", type=str,
                   choices=["none", "basic", "adaptive"], default="adaptive",
                   help="Tipo de safety shield")

    # Configuración PPO
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate inicial para PPO")
    p.add_argument("--max_episodes", type=int, default=2500,
                   help="Número máximo de episodios de entrenamiento")
    p.add_argument("--max_steps", type=int, default=1000,
                   help="Pasos máximos por episodio")
    p.add_argument("--update_timestep", type=int, default=512,
                   help="Timesteps entre actualizaciones de política")
    p.add_argument("--k_epochs",         type=int,   default=10)
    p.add_argument("--entropy_coef",     type=float, default=0.01)
    p.add_argument("--value_loss_coef",  type=float, default=0.5,
                   help="Coeficiente para value loss.")

    # Parámetros del entorno
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
    p.add_argument("--shield_intervention_penalty", type=float, default=0.10,
                   help="Penalización por intervención del shield. "
                        "IMPORTANTE: usar > 0 para que la política aprenda a "
                        "evitar intervenciones. Default: 0.10")
    p.add_argument("--idle_penalty_weight", type=float, default=0.08,
                   help="Penalización por paso cuando speed < min_moving_speed. ")
    p.add_argument("--min_moving_speed_kmh", type=float, default=5.0,
                   help="Velocidad mínima para que lane_centering/heading tengan efecto.")

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
    Construye la cadena de wrappers:
        CarlaEnv  →  CarlaRewardShaper  →  Shield
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

    # 2. Shield (opcional)
    if args.shield_type == "basic":
        logger.info("Shield: CarlaSafetyShield (LIDAR + Waypoint API)")
        env = CarlaSafetyShield(
            env,
            num_lidar_rays=num_lidar_rays,
            front_threshold=args.front_threshold,
            side_threshold=args.side_threshold,
            lateral_threshold=args.lateral_threshold,
        )
    elif args.shield_type == "adaptive":
        logger.info("Shield: CarlaAdaptiveHorizonShield (BicycleModel + Waypoint API)")
        env = CarlaAdaptiveHorizonShield(
            env,
            num_lidar_rays=num_lidar_rays,
            front_threshold_base=args.front_threshold,
            side_threshold_base=args.side_threshold,
            lateral_threshold_base=args.lateral_threshold,
        )
    else:
        logger.info("Sin shield — PPO estándar")

    # 3. Reward shaping con Waypoint API
    env = CarlaRewardShaper(
        env,
        target_speed_kmh=args.target_speed_kmh,
        speed_weight=args.speed_weight,
        smoothness_weight=args.smoothness_weight,
        lane_centering_weight=args.lane_centering_weight,
        lane_invasion_penalty=args.lane_invasion_penalty,
        off_road_penalty=args.off_road_penalty,
        shield_intervention_penalty=args.shield_intervention_penalty,
        idle_penalty_weight=args.idle_penalty_weight,
        min_moving_speed_kmh=args.min_moving_speed_kmh,
    )



    return env, num_lidar_rays

# HELPERS DE MÉTRICAS DE EPISODIO

def _ep_mean(infos, key, default=0.0):
    vals = [i.get(key, default) for i in infos if key in i]
    return float(np.mean(vals)) if vals else default
 
def _ep_sum(infos, key, default=0.0):
    return float(sum(i.get(key, default) for i in infos))
 
def _ep_min(infos, key, default=1.0):
    vals = [i.get(key, default) for i in infos if key in i]
    return float(np.min(vals)) if vals else default
 
def _speed_compliance_rate(infos):
    """Fracción de steps donde speed <= speed_limit * 1.05."""
    n = compliant = 0
    for i in infos:
        limit = i.get("speed_limit_kmh", 0.0)
        if limit <= 0.0:
            continue
        n += 1
        if i.get("speed_kmh", 0.0) <= limit * 1.05:
            compliant += 1
    return compliant / max(n, 1)


# ENTRENAMIENTO

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
    reward_window = deque(maxlen=100)
    success_window = deque(maxlen=100)
    crash_window = deque(maxlen=100)
    offroad_window = deque(maxlen=100)
    best_avg_reward = -float("inf")

    # Construir entorno 
    logger.info("Connecting to CARLA and building environment...")
    env, num_lidar_rays = build_env(args)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    logger.info(f"State dim: {state_dim} | Action dim: {action_dim}")

    expected_updates = max(
        (args.max_episodes * args.max_steps) // args.update_timestep, 1
    )
    logger.info(f"Expected optimizer updates: ~{expected_updates}")

    # Agente PPO
    agent = PPOAgent(
        state_dim,
        action_dim,
        lr=args.lr,
        scheduler_t_max=expected_updates,
        k_epochs=args.k_epochs,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
    )

    memory = {
        "states": [],
        "actions":      [],
        "log_probs":    [],
        "rewards":      [],
        "dones":        [],
        "truncated":    [],
        "final_values": [],
    }
    timestep       = 0
    avg_reward_100 = 0.0
    success_rate   = 0.0

    logger.info(f"Starting training for {args.max_episodes} episodes...\n")

    try:
        for episode in range(1, args.max_episodes + 1):

            obs, _ = env.reset()
            episode_reward       = 0.0
            ep_shield_activations = 0
            ep_infos              = []

            for step in range(args.max_steps):
                timestep += 1

                action, log_prob, _ = agent.select_action(obs)
                next_obs, reward, done, truncated, info = env.step(action)
                ep_infos.append(info)

                executed_action = np.array(
                    info.get("executed_action", action), dtype=np.float32
                )
                executed_action = np.clip(
                    executed_action,
                    env.action_space.low,
                    env.action_space.high,
                )

                exec_log_prob = agent.evaluate_action(obs, executed_action)

                is_truncated = truncated and not done
                if is_truncated:
                    final_value = agent.compute_bootstrap_value(next_obs)
                else:
                    final_value = 0.0
 
                memory["states"].append(obs)
                memory["actions"].append(executed_action)
                memory["log_probs"].append(exec_log_prob)
                memory["rewards"].append(reward)
                memory["dones"].append(done)
                memory["truncated"].append(is_truncated)
                memory["final_values"].append(final_value)
                
                if info.get("shield_activated", info.get("shield_active", False)):
                    ep_shield_activations += 1

                obs = next_obs
                episode_reward += reward

                # Actualización de política
                if timestep % args.update_timestep == 0 and len(memory["states"]) > 0:
                    train_metrics = agent.update(memory)
                    for key in memory:
                        memory[key] = []
                
                    agent.step_scheduler()

                    writer.add_scalar("Loss/Policy_Loss", train_metrics["policy_loss"], timestep)
                    writer.add_scalar("Loss/Value_Loss", train_metrics["value_loss"], timestep)
                    writer.add_scalar("Loss/Grad_Norm", train_metrics["grad_norm"], timestep)
                    writer.add_scalar("Training/Entropy", train_metrics["entropy"], timestep)
                    writer.add_scalar("Training/Approx_KL", train_metrics["approx_kl"], timestep)
                    writer.add_scalar("Training/Learning_Rate", agent.get_lr(), timestep)

                if done or truncated:
                    break

            # ── Outcome del episodio ─────────────────────────────────
            is_success = int(info.get("arrive_dest", False))
            is_crash   = int(info.get("collision", False))
            is_offroad = int(info.get("out_of_road", False))

            outcome    = 0
            if info.get("collision", False):
                outcome = 1
            elif info.get("stuck", False):
                outcome = 2
            elif info.get("out_of_road", False):
                outcome = 3
            elif info.get("arrive_dest", False):
                outcome = 4

            reward_window.append(episode_reward)
            success_window.append(is_success)
            crash_window.append(is_crash)
            offroad_window.append(is_offroad)

            avg_reward_100 = float(np.mean(reward_window))
            success_rate = float(np.mean(success_window))
            crash_rate = float(np.mean(crash_window))
            offroad_rate = float(np.mean(offroad_window))

            # Ajuste de learning rate con scheduler
            ep_steps = len(ep_infos)

            # ── TensorBoard — Reward ──────────────────────────────────
            writer.add_scalar("Reward/Raw_Episode", episode_reward, episode)
            writer.add_scalar("Reward/Average_100_Episodes", avg_reward_100, episode)
 
            # Desglose de componentes de reward (media del episodio)
            writer.add_scalar("Reward/Components/Speed_Bonus",
                              _ep_mean(ep_infos, "speed_bonus"), episode)
            writer.add_scalar("Reward/Components/Lane_Centering",
                              _ep_mean(ep_infos, "lane_center_bonus"), episode)
            writer.add_scalar("Reward/Components/Heading_Alignment",
                              _ep_mean(ep_infos, "heading_bonus"), episode)
            writer.add_scalar("Reward/Components/Smooth_Penalty",
                              _ep_mean(ep_infos, "smooth_penalty"), episode)
            writer.add_scalar("Reward/Components/Invasion_Penalty",
                              _ep_mean(ep_infos, "invasion_penalty"), episode)
            writer.add_scalar("Reward/Components/Road_Penalty",
                              _ep_mean(ep_infos, "road_penalty"), episode)
            writer.add_scalar("Reward/Components/Progress_Bonus",
                              _ep_sum(ep_infos, "progress_bonus"), episode)
            writer.add_scalar("Reward/Components/Shield_Penalty",
                              _ep_mean(ep_infos, "shield_intervention_pen"), episode)
 
            # ── TensorBoard — Training ────────────────────────────────
            writer.add_scalar("Training/Success_Rate", success_rate, episode)
            writer.add_scalar("Training/Crash_Rate", crash_rate, episode)
            writer.add_scalar("Training/Offroad_Rate", offroad_rate, episode)
            writer.add_scalar("Training/Episode_Length", ep_steps, episode)
 
            # ── TensorBoard — Safety / Shield ─────────────────────────
            shield_rate = ep_shield_activations / max(ep_steps, 1)
            writer.add_scalar("Safety/Shield_Activations", ep_shield_activations, episode)
            writer.add_scalar("Safety/Shield_Rate", shield_rate, episode)
 
            # Desglose semántico del shield (si el shield lo expone)
            shield_wrapper = _get_shield(env)
            if shield_wrapper is not None:
                stats = shield_wrapper.get_statistics()
                writer.add_scalar("Safety/Semantic/Dynamic_Interventions",
                                  stats.get("interventions_dynamic", 0), episode)
                writer.add_scalar("Safety/Semantic/Static_Interventions",
                                  stats.get("interventions_static", 0), episode)
                writer.add_scalar("Safety/Semantic/Pedestrian_Interventions",
                                  stats.get("interventions_pedestrian", 0), episode)
                writer.add_scalar("Safety/Semantic/Safe_Step_Rate",
                                  stats.get("safe_rate", 0.0), episode)
                writer.add_scalar("Safety/Semantic/Warning_Step_Rate",
                                  stats.get("warning_rate", 0.0), episode)
                writer.add_scalar("Safety/Semantic/Critical_Step_Rate",
                                  stats.get("critical_rate", 0.0), episode)
                shield_wrapper.reset_statistics()
 
            # ── TensorBoard — CARLA / Entorno ─────────────────────────
            writer.add_scalar("CARLA/Mean_Speed_kmh",
                              _ep_mean(ep_infos, "speed_kmh"), episode)
            writer.add_scalar("CARLA/Mean_Lateral_Offset_Norm",
                              _ep_mean(ep_infos, "lateral_offset_norm"), episode)
            writer.add_scalar("CARLA/Mean_Heading_Error_deg",
                              _ep_mean(ep_infos, "heading_error"), episode)
            writer.add_scalar("CARLA/Total_Distance",
                              info.get("total_distance", 0.0), episode)
            writer.add_scalar("CARLA/Lane_Invasions_Ep",
                              info.get("episode_lane_invasions", 0), episode)
            writer.add_scalar("CARLA/Collisions_Ep",
                              info.get("episode_collisions", 0), episode)
            writer.add_scalar("CARLA/Speed_Compliance_Rate",
                              _speed_compliance_rate(ep_infos), episode)
            writer.add_scalar("CARLA/Mean_Speed_Limit_kmh",
                              _ep_mean(ep_infos, "speed_limit_kmh"), episode)
 
            # LIDAR semántico — distancias mínimas al obstáculo más peligroso
            min_veh_m = _ep_min(ep_infos, "nearest_vehicle_m",    default=999.0)
            min_ped_m = _ep_min(ep_infos, "nearest_pedestrian_m", default=999.0)
            writer.add_scalar("Safety/Min_Vehicle_Distance_m",
                              min_veh_m if min_veh_m < 999.0 else 50.0, episode)
            writer.add_scalar("Safety/Min_Pedestrian_Distance_m",
                              min_ped_m if min_ped_m < 999.0 else 50.0, episode)
            writer.add_scalar("Safety/Min_Front_Dynamic",
                              _ep_min(ep_infos, "min_front_dynamic"), episode)
 
            # Outcome detallado
            writer.add_scalar("Outcome/Type", outcome, episode)
            writer.add_scalar("Outcome/Stuck_Rate",
                              float(np.mean([int(i.get("stuck", False)) for i in ep_infos])),
                              episode)
 
            writer.flush()

            # ── Guardar mejor modelo (a partir del episodio 500) ─────
            if episode >= 500 and avg_reward_100 > best_avg_reward + 20:
                best_avg_reward = avg_reward_100
                agent.save(str(best_model_path))
                logger.info(
                    f"New best model saved at Ep {episode} "
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
                    f"Succ: {success_rate:.2f} | "
                    f"Shield: {ep_shield_activations}/{ep_steps} "
                    f"({shield_rate:.1%}) | "
                    f"Out: {['timeout','crash','stuck','offroad','success'][outcome]}"
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
        export_data.extract_tensorboard_data(log_dir)

def _get_shield(env):
    """Navega la cadena de wrappers para localizar el shield."""
    e = env
    while e is not None:
        if hasattr(e, "get_statistics") and hasattr(e, "reset_statistics"):
            return e
        e = getattr(e, "env", None)
    return None

if __name__ == "__main__":
    train()
