"""
USO:
    # Arrancar servidor CARLA primero:
    #   .\CARLA_0.9.16\CarlaUE4.exe -RenderOffScreen -carla-port=2000

    python main_train.py --model_name mi_modelo --shield_type adaptive
"""

import argparse
import logging
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from src.CARLA.Env.carla_env import CarlaEnv
from src.curriculumManager import CurriculumManager
from src.reward_shaper import CarlaRewardShaper
from src.safety_shield import CarlaSafetyShield
from src.Adaptative_Shield.adaptive_horizon_shield import CarlaAdaptiveHorizonShield
from src.PPO.ppo_agent import PPOAgent
from src.Metrics.live_metrics import LiveMetricsLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_train")


# ARGUMENTOS


def get_args():
    p = argparse.ArgumentParser(description="PPO Training con Safety Shield en CARLA")

    # Identificativos y configuración general
    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Nombre base del modelo a entrenar",
    )
    p.add_argument(
        "--shield_type",
        type=str,
        choices=["none", "basic", "adaptive"],
        default="adaptive",
        help="Tipo de safety shield",
    )

    # Configuración PPO
    p.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate inicial para PPO"
    )
    p.add_argument(
        "--max_episodes",
        type=int,
        default=2500,
        help="Número máximo de episodios de entrenamiento",
    )
    p.add_argument(
        "--max_steps", type=int, default=1000, help="Pasos máximos por episodio"
    )
    p.add_argument(
        "--update_timestep",
        type=int,
        default=2048,
        help="Timesteps entre actualizaciones de política",
    )
    p.add_argument("--k_epochs", type=int, default=10)
    p.add_argument("--entropy_coef", type=float, default=0.02)
    p.add_argument(
        "--value_loss_coef",
        type=float,
        default=0.25,
        help="Coeficiente para value loss.",
    )
    p.add_argument(
        "--kl_target",
        type=float,
        default=0.08,
        help="KL target para early-stop de epochs PPO. 0 desactiva el early-stop.",
    )
    p.add_argument(
        "--obs-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activar normalización online de observaciones. (Usa --no-obs-norm para desactivar)",
    )

    # Parámetros del entorno
    p.add_argument(
        "--host", type=str, default="localhost", help="Host del servidor CARLA"
    )
    p.add_argument("--port", type=int, default=2000, help="Puerto del servidor CARLA")
    p.add_argument(
        "--tm_port", type=int, default=8000, help="Puerto del TrafficManager"
    )
    p.add_argument(
        "--map",
        type=str,
        default="Town04",
        help="Mapa CARLA (Town01-Town07, Town10HD...)",
    )
    p.add_argument(
        "--num_npc",
        type=int,
        default=40,
        help="Número de vehículos NPC gestionados por TrafficManager",
    )
    p.add_argument(
        "--weather",
        type=str,
        default="ClearNoon",
        help="Preset de clima CARLA (ClearNoon, WetSunset, CloudyNight...)",
    )
    p.add_argument(
        "--target_speed_kmh",
        type=float,
        default=50.0,
        help="Velocidad objetivo para el reward (km/h)",
    )
    p.add_argument(
        "--success_distance",
        type=float,
        default=250.0,
        help="Metros a recorrer para considerar episodio exitoso",
    )

    # Curriculum de entrenamiento (basado en rendimiento)
    p.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Curriculum progresivo de 5 etapas con rollback automático: "
        "0 → 25%% → 50%% → 75%% → num_npc NPCs. "
        "Usa --no-curriculum para desactivar.",
    )

    # Parámetros del shield
    p.add_argument(
        "--front_threshold",
        type=float,
        default=0.15,
        help="Umbral LIDAR frontal de seguridad [0-1 norm]",
    )
    p.add_argument(
        "--side_threshold",
        type=float,
        default=0.04,
        help="Umbral LIDAR lateral de seguridad [0-1 norm]",
    )
    p.add_argument(
        "--lateral_threshold",
        type=float,
        default=0.65,
        help="Fracción del semi-ancho de carril antes de corregir (era 0.82)",
    )

    # Reward shaping
    # Level 2: Efficiency / Comfort weights
    p.add_argument(
        "--speed_weight", type=float, default=0.10, help="Peso del bonus de velocidad (E1)"
    )
    p.add_argument(
        "--comfort_weight",
        type=float,
        default=0.08,
        help="Peso combinado de centramiento y suavidad de dirección (E3)",
    )
    # Level 1: Safety weights (cada uno debe superar efficiency_cap=0.20)
    p.add_argument(
        "--lane_invasion_penalty",
        type=float,
        default=0.35,
        help="Penalización por invasión de carril (S2). Debe ser > efficiency_cap.",
    )
    p.add_argument(
        "--off_road_penalty",
        type=float,
        default=2.00,
        help="Penalización base por salirse de carretera (S1).",
    )
    p.add_argument(
        "--safety_edge_weight",
        type=float,
        default=0.40,
        help="Peso de penalización gradual por proximidad al borde (S1). Debe ser > efficiency_cap.",
    )
    p.add_argument(
        "--shield_intervention_penalty",
        type=float,
        default=0.25,
        help="Penalización por intervención del shield (S3). Debe ser > efficiency_cap.",
    )
    p.add_argument(
        "--min_moving_speed_kmh",
        type=float,
        default=5.0,
        help="Velocidad mínima para el speed gate.",
    )

    # Checkpoints y logging
    p.add_argument(
        "--ckpt_freq",
        type=int,
        default=200,
        help="Frecuencia (en episodios) de guardado de checkpoints",
    )
    p.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")

    return p.parse_args()


# CONSTRUCCIÓN DEL ENTORNO


def build_env(args, num_npc_override: int = None):
    """
    Construye la cadena de wrappers:
        CarlaEnv  →  CarlaRewardShaper  →  Shield
    Args:
        num_npc_override: si se pasa, anula args.num_npc (usado por el curriculum).
    """
    num_lidar_rays = 240
    num_npc = num_npc_override if num_npc_override is not None else args.num_npc

    # 1. Entorno base CARLA
    env = CarlaEnv(
        host=args.host,
        port=args.port,
        tm_port=args.tm_port,
        map_name=args.map,
        num_npc_vehicles=num_npc,
        weather=args.weather,
        synchronous=True,
        fixed_delta_seconds=0.05,
        num_lidar_rays=num_lidar_rays,
        lidar_range=50.0,
        max_episode_steps=args.max_steps,
        target_speed_kmh=args.target_speed_kmh,
        success_distance=args.success_distance,
        success_reward=30.0,
        out_of_road_penalty=20.0,
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

    # 3. Reward shaping — jerarquía lexicográfica de dos niveles
    env = CarlaRewardShaper(
        env,
        target_speed_kmh=args.target_speed_kmh,
        # Level 2: Efficiency / Comfort
        speed_weight=args.speed_weight,
        comfort_weight=args.comfort_weight,
        # Level 1: Safety
        lane_invasion_penalty=args.lane_invasion_penalty,
        off_road_penalty=args.off_road_penalty,
        safety_edge_weight=args.safety_edge_weight,
        shield_intervention_penalty=args.shield_intervention_penalty,
        min_moving_speed_kmh=args.min_moving_speed_kmh,
        max_steps=args.max_steps,
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

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.model_name}_{args.shield_type}_{timestamp}"
    log_dir = f"./runs/{run_name}"
    models_dir = Path("./data/models")
    ckpt_dir = models_dir / f"{run_name}_checkpoints"

    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_db_path = Path(log_dir) / "metrics.sqlite"
    live_metrics = LiveMetricsLogger(
        db_path=metrics_db_path,
        run_name=run_name,
        model_name=args.model_name,
        shield_type=args.shield_type,
        map_name=args.map,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        update_timestep=args.update_timestep,
    )
    logger.info(f"Run: {run_name}")
    logger.info(f"Live metrics DB: {metrics_db_path}")

    # Rutas de modelos
    final_model_path = models_dir / f"{args.model_name}_{args.shield_type}_final.pth"
    best_model_path = models_dir / f"{args.model_name}_{args.shield_type}_best.pth"

    # Ventanas de métricas
    reward_window = deque(maxlen=100)
    success_window = deque(maxlen=100)
    crash_window = deque(maxlen=100)
    offroad_window = deque(maxlen=100)
    best_avg_reward = -float("inf")

    # ── Curriculum manager ────────────────────────────────────────────────
    curriculum = CurriculumManager(
        max_npc=args.num_npc,
        enabled=args.curriculum,
        min_eps_per_stage=100,
        rollback_patience=50,
    )
    if args.curriculum:
        logger.info(
            f"[Curriculum] Activado | Etapas NPCs: {curriculum.stages} | "
            f"min_eps_por_etapa=100 | rollback_patience=50"
        )

    # Construir entorno
    logger.info("Connecting to CARLA and building environment...")
    initial_npc = curriculum.current_npc_count
    env, num_lidar_rays = build_env(args, num_npc_override=initial_npc)
    current_npc_count = initial_npc

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    logger.info(f"State dim: {state_dim} | Action dim: {action_dim}")

    expected_updates = max(
        (args.max_episodes * args.max_steps) // args.update_timestep, 1
    )
    logger.info(f"Expected optimizer updates: ~{expected_updates}")

    # Agente PPO
    kl_target = args.kl_target if args.kl_target > 0 else None
    normalize_obs = args.obs_norm
    agent = PPOAgent(
        state_dim,
        action_dim,
        lr=args.lr,
        scheduler_t_max=expected_updates,
        k_epochs=args.k_epochs,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        kl_target=kl_target,
        normalize_obs=normalize_obs,
    )
    logger.info(
        f"PPO: lr={args.lr} | update_timestep={args.update_timestep} | "
        f"kl_target={kl_target} | obs_norm={normalize_obs}"
    )

    memory = {
        "states": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "dones": [],
        "truncated": [],
        "final_values": [],
    }
    timestep = 0
    avg_reward_100 = 0.0
    success_rate = 0.0
    crash_rate = 0.0
    offroad_rate = 0.0
    run_status = "running"

    logger.info(f"Starting training for {args.max_episodes} episodes...\n")

    try:
        for episode in range(1, args.max_episodes + 1):
            obs, _ = env.reset()
            episode_reward = 0.0
            ep_shield_activations = 0
            ep_infos = []

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

                    live_metrics.log_metrics(
                        axis="update",
                        step=timestep,
                        metrics={
                            "Loss/Policy_Loss": train_metrics["policy_loss"],
                            "Loss/Value_Loss": train_metrics["value_loss"],
                            "Loss/Grad_Norm": train_metrics["grad_norm"],
                            "Training/Entropy": train_metrics["entropy"],
                            "Training/Approx_KL": train_metrics["approx_kl"],
                            "Training/Epochs_Run": train_metrics.get(
                                "epochs_run", args.k_epochs
                            ),
                            "Training/Learning_Rate": agent.get_lr(),
                        },
                    )

                if done or truncated:
                    break

            # ── Outcome del episodio ─────────────────────────────────
            is_success = int(info.get("arrive_dest", False))
            is_crash = int(info.get("collision", False))
            is_offroad = int(info.get("out_of_road", False))

            outcome = 0
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
            
            desired_npc, curriculum_event = curriculum.step(
                offroad_rate=offroad_rate,
                crash_rate=crash_rate,
                avg_reward=avg_reward_100,
            )
            if curriculum_event != "none":
                base_env = _get_base_env(env)
                old_npc = current_npc_count
                # Actualización no disruptiva: solo cambia el parámetro.
                # Toma efecto en el próximo env.reset() sin reconectar CARLA.
                base_env.num_npc_vehicles = desired_npc
                current_npc_count = desired_npc
                logger.info(
                    f"[Curriculum] Ep {episode} | {curriculum_event.upper()}: "
                    f"{old_npc} → {desired_npc} NPCs "
                    f"(crash={crash_rate:.2f}, offroad={offroad_rate:.2f}, "
                    f"r100={avg_reward_100:.1f}) | "
                    f"Etapa {curriculum.current_stage_idx + 1}/{len(curriculum.stages)}"
                )

            # Ajuste de learning rate con scheduler
            ep_steps = len(ep_infos)

            # ── Métricas de episodio (fuente unificada) ───────────────
            shield_rate = ep_shield_activations / max(ep_steps, 1)
            shield_semantic_metrics = {
                "Safety/Semantic/Dynamic_Interventions": 0.0,
                "Safety/Semantic/Static_Interventions": 0.0,
                "Safety/Semantic/Pedestrian_Interventions": 0.0,
                "Safety/Semantic/Safe_Step_Rate": 0.0,
                "Safety/Semantic/Warning_Step_Rate": 0.0,
                "Safety/Semantic/Critical_Step_Rate": 0.0,
            }

            # Desglose semántico del shield (si el shield lo expone)
            shield_wrapper = _get_shield(env)
            if shield_wrapper is not None:
                stats = shield_wrapper.get_statistics()
                shield_semantic_metrics = {
                    "Safety/Semantic/Dynamic_Interventions": stats.get(
                        "interventions_dynamic", 0.0
                    ),
                    "Safety/Semantic/Static_Interventions": stats.get(
                        "interventions_static", 0.0
                    ),
                    "Safety/Semantic/Pedestrian_Interventions": stats.get(
                        "interventions_pedestrian", 0.0
                    ),
                    "Safety/Semantic/Safe_Step_Rate": stats.get("safe_rate", 0.0),
                    "Safety/Semantic/Warning_Step_Rate": stats.get("warning_rate", 0.0),
                    "Safety/Semantic/Critical_Step_Rate": stats.get(
                        "critical_rate", 0.0
                    ),
                }
                shield_wrapper.reset_statistics()

            # LIDAR semántico — distancias mínimas al obstáculo más peligroso
            min_veh_m = _ep_min(ep_infos, "nearest_vehicle_m", default=999.0)
            min_ped_m = _ep_min(ep_infos, "nearest_pedestrian_m", default=999.0)

            live_metrics.log_metrics(
                axis="episode",
                step=episode,
                metrics={
                    # Reward
                    "Reward/Raw_Episode": episode_reward,
                    "Reward/Average_100_Episodes": avg_reward_100,
                    # Level aggregates
                    "Reward/Safety_Reward_Sum": _ep_sum(ep_infos, "safety_reward"),
                    "Reward/Efficiency_Reward_Sum": _ep_sum(ep_infos, "efficiency_reward"),
                    # Level 1: Safety components
                    "Reward/Safety/Edge_Penalty": _ep_mean(ep_infos, "edge_penalty"),
                    "Reward/Safety/Invasion_Penalty": _ep_mean(
                        ep_infos, "invasion_penalty"
                    ),
                    "Reward/Safety/Shield_Penalty": _ep_mean(ep_infos, "shield_pen"),
                    # Level 2: Efficiency components
                    "Reward/Efficiency/Speed_Bonus": _ep_mean(ep_infos, "speed_bonus"),
                    "Reward/Efficiency/Progress_Bonus": _ep_sum(
                        ep_infos, "progress_bonus"
                    ),
                    "Reward/Efficiency/Comfort_Reward": _ep_mean(
                        ep_infos, "comfort_reward"
                    ),
                    "Reward/Efficiency/Clipped_Rate": _ep_mean(
                        ep_infos, "efficiency_clipped"
                    ),
                    # Training
                    "Training/Success_Rate": success_rate,
                    "Training/Crash_Rate": crash_rate,
                    "Training/Offroad_Rate": offroad_rate,
                    "Training/Episode_Length": ep_steps,
                    "Training/Curriculum_NPC": current_npc_count,
                    # Safety
                    "Safety/Shield_Activations": ep_shield_activations,
                    "Safety/Shield_Rate": shield_rate,
                    "Safety/Min_Vehicle_Distance_m": min_veh_m
                    if min_veh_m < 999.0
                    else 50.0,
                    "Safety/Min_Pedestrian_Distance_m": min_ped_m
                    if min_ped_m < 999.0
                    else 50.0,
                    "Safety/Min_Front_Dynamic": _ep_min(ep_infos, "min_front_dynamic"),
                    # CARLA — métricas base
                    "CARLA/Mean_Speed_kmh": _ep_mean(ep_infos, "speed_kmh"),
                    "CARLA/Mean_Lateral_Offset_Norm": _ep_mean(
                        ep_infos, "lateral_offset_norm"
                    ),
                    "CARLA/Mean_Heading_Error_deg": _ep_mean(ep_infos, "heading_error"),
                    "CARLA/Total_Distance": info.get("total_distance", 0.0),
                    "CARLA/Lane_Invasions_Ep": info.get("episode_lane_invasions", 0),
                    "CARLA/Collisions_Ep": info.get("episode_collisions", 0),
                    "CARLA/Speed_Compliance_Rate": _speed_compliance_rate(ep_infos),
                    "CARLA/Mean_Speed_Limit_kmh": _ep_mean(ep_infos, "speed_limit_kmh"),
                    # CARLA — métricas nuevas de borde de carril
                    "CARLA/Mean_Dist_Left_Edge": _ep_mean(
                        ep_infos, "dist_left_edge_norm", default=0.5
                    ),
                    "CARLA/Mean_Dist_Right_Edge": _ep_mean(
                        ep_infos, "dist_right_edge_norm", default=0.5
                    ),
                    "CARLA/Min_Dist_Left_Edge": _ep_min(
                        ep_infos, "dist_left_edge_norm", default=0.5
                    ),
                    "CARLA/Min_Dist_Right_Edge": _ep_min(
                        ep_infos, "dist_right_edge_norm", default=0.5
                    ),
                    "CARLA/Mean_Road_Curvature": _ep_mean(
                        ep_infos, "road_curvature_norm", default=0.0
                    ),
                    "CARLA/Mean_Road_Edge_LIDAR": _ep_min(
                        ep_infos, "nearest_road_edge_m", default=999.0
                    ),
                    # Outcome
                    "Outcome/Type": outcome,
                    "Outcome/Stuck_Rate": float(
                        np.mean([int(i.get("stuck", False)) for i in ep_infos])
                    ),
                    # Semantic shield
                    **shield_semantic_metrics,
                },
            )

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
                    f"Out: {['timeout', 'crash', 'stuck', 'offroad', 'success'][outcome]}"
                    f" | NPCs: {current_npc_count}"
                )

        run_status = "finished"

    except KeyboardInterrupt:
        run_status = "interrupted"
        logger.info("Training interrupted by user.")

    except Exception:
        run_status = "failed"
        logger.exception("Training failed with an unexpected error.")
        raise

    finally:
        # Guardar modelo final
        agent.save(str(final_model_path))
        logger.info(f"Final model saved: {final_model_path}")

        env.close()
        live_metrics.close(status=run_status)

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


def _get_shield(env):
    """Navega la cadena de wrappers para localizar el shield."""
    e = env
    while e is not None:
        if hasattr(e, "get_statistics") and hasattr(e, "reset_statistics"):
            return e
        e = getattr(e, "env", None)
    return None


def _get_base_env(env):
    """
    Navega la cadena de wrappers Gymnasium hasta encontrar el CarlaEnv base.

    Se usa para actualizar num_npc_vehicles de forma no disruptiva durante
    las transiciones de curriculum, evitando reconstruir toda la cadena de
    wrappers (lo que causaba el deadlock de CARLA descrito en el Problema 1).
    """
    e = env
    while e is not None:
        if isinstance(e, CarlaEnv):
            return e
        e = getattr(e, "env", None)
    raise RuntimeError(
        "_get_base_env: CarlaEnv no encontrado en la cadena de wrappers."
    )


if __name__ == "__main__":
    train()
