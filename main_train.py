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
from src.meta.kpi_snapshot import build_kpi_snapshot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_train")


def _parse_spawn_indices(raw: str):
    """Parsea string CSV ('1,5,42') a list[int]. Permite vacío/None.

    Tolera espacios y trailing commas; rechaza cualquier token no-entero.
    """
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--spawn_point_indices: '{tok}' no es un entero válido"
            ) from exc
    return out if out else None


def get_args():
    p = argparse.ArgumentParser(description="PPO Training con Safety Shield en CARLA")

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
    p.add_argument(
        "--minibatch_size",
        type=int,
        default=64,
        help="Tamaño del minibatch dentro de cada epoch PPO (SB3 default: 64). "
        "Con update_timestep=2048 → 32 minibatches × k_epochs pasos de Adam.",
    )
    p.add_argument(
        "--entropy_coef",
        type=float,
        default=0.001,
        help="Initial entropy bonus coefficient. Decays linearly to "
        "--entropy_coef_min over --entropy_coef_decay_updates updates.",
    )
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
        default=30.0,
        help="Velocidad objetivo FALLBACK (km/h). Sólo se usa cuando el "
        "Waypoint API no devuelve speed_limit válido — el reward usa el "
        "límite dinámico real de la carretera por defecto.",
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
        default=0.02,
        help="Umbral LIDAR lateral de seguridad [0-1 norm] (Fix #6: bajado "
        "0.04→0.02 para reducir falsos positivos contra guardarraíles "
        "permanentes de Town04 a 1.5-2.5 m del carril, que enmascaraban "
        "22%% de los pasos del policy gradient sin amenaza real).",
    )
    p.add_argument(
        "--lateral_threshold",
        type=float,
        default=0.65,
        help="Fracción del semi-ancho de carril antes de corregir (era 0.82)",
    )
    p.add_argument(
        "--shield_permissiveness",
        type=float,
        default=1.0,
        help="Escalar [0,1] de permisividad del adaptive shield (1.0=strict "
        "baseline; <1.0 afloja TODOS los umbrales destetables a la vez vía "
        "shield.set_permissiveness, ver src/meta/tunable_shield.py). Lo gobierna "
        "el meta-controlador LLM en `train_curriculum.py --llm_tuner`. A 1.0 "
        "(default) el shield es idéntico al de producción.",
    )

    # Meta-controlador LLM (Modo B — intra-run). Ver src/meta/.
    p.add_argument(
        "--meta_tuner",
        action="store_true",
        help="Activa el meta-controlador LLM intra-run: cada --meta_interval "
        "episodios, Gemma/Ollama propone (en un hilo de fondo) la permisividad "
        "del adaptive shield, vetada por el SafetyGovernor, y se aplica en el "
        "siguiente boundary de episodio. Requiere --shield_type adaptive y "
        "`ollama serve`. No bloquea el bucle de entrenamiento.",
    )
    p.add_argument(
        "--meta_interval",
        type=int,
        default=100,
        help="Cada cuántos episodios el meta-controlador evalúa los KPIs y "
        "propone un ajuste de permisividad (default 100).",
    )
    p.add_argument(
        "--llm_model",
        type=str,
        default="gemma4:e4b",
        help="Tag del modelo Ollama para el meta-controlador (default: ligero).",
    )
    p.add_argument(
        "--llm_host",
        type=str,
        default=None,
        help="Host de Ollama (default: cliente local, http://localhost:11434).",
    )
    p.add_argument(
        "--meta_seed",
        type=int,
        default=42,
        help="Seed del LLM para determinismo del meta-controlador.",
    )
    p.add_argument(
        "--llm_gpu",
        action="store_true",
        help="Usa la GPU para el meta-LLM. POR DEFECTO corre en CPU (num_gpu=0) "
        "para NO competir por VRAM con CARLA+PPO — evita el error 'timed out "
        "waiting for llama-server to start' bajo contención de GPU. Actívalo "
        "solo si tu GPU tiene VRAM de sobra para CARLA + el modelo.",
    )
    p.add_argument(
        "--llm_keep_alive",
        type=str,
        default="0",
        help="keep_alive de Ollama para el meta-LLM. '0' descarga tras cada "
        "llamada (libera memoria); '10m' o '-1' lo mantiene cargado entre "
        "llamadas (más rápido, pero retiene memoria).",
    )
    p.add_argument(
        "--llm_timeout",
        type=float,
        default=600.0,
        help="Timeout (s) del cliente Ollama para el meta-LLM. Generoso por "
        "defecto porque la carga en frío de un modelo grande puede tardar.",
    )

    # Reward shaping
    p.add_argument(
        "--speed_weight", type=float, default=0.20, help="Peso del bonus de velocidad"
    )
    p.add_argument(
        "--smoothness_weight",
        type=float,
        default=0.10,
        help="Peso de la penalización por steering brusco",
    )
    p.add_argument(
        "--lane_centering_weight",
        type=float,
        default=0.30,
        help="Peso del bonus de centramiento en carril",
    )
    p.add_argument(
        "--lane_invasion_penalty",
        type=float,
        default=0.25,
        help="Penalización por invasión de carril (sensor CARLA)",
    )
    p.add_argument(
        "--off_road_penalty",
        type=float,
        default=1.00,
        help="Penalización adicional del shaper por salirse de carretera. "
        "El CarlaEnv base ya aplica una penalización fuerte.",
    )
    p.add_argument(
        "--idle_penalty_weight",
        type=float,
        default=0.25,
        help="Pico de la idle_penalty ESCALONADA. Se aplica a velocidades "
        "muy bajas; tramos 0.5-2 y 2-5 km/h la reducen progresivamente.",
    )
    p.add_argument(
        "--progress_reward_weight",
        type=float,
        default=0.20,
        help="Peso del progress_reward denso (satura a PROGRESS_SATURATION_KMH=10).",
    )
    p.add_argument(
        "--acceleration_reward_weight",
        type=float,
        default=0.08,
        help="Peso del acceleration_reward (clip(Δv, 0, 2) · weight). Señal "
        "densa desde el primer km/h ganado — complementa progress_reward en "
        "el tramo de arranque.",
    )
    p.add_argument(
        "--entropy_coef_min",
        type=float,
        default=0.0,
        help="Valor mínimo de entropy_coef tras el decay lineal. Bajado a "
        "0.0 para que la policy loss pueda dominar al final del training "
        "y converger a una política casi determinista.",
    )
    p.add_argument(
        "--entropy_coef_decay_updates",
        type=int,
        default=50,
        help="Number of PPO updates over which entropy_coef decays linearly "
        "from --entropy_coef down to --entropy_coef_min.",
    )

    # Checkpoints y logging
    p.add_argument(
        "--ckpt_freq",
        type=int,
        default=200,
        help="Frecuencia (en episodios) de guardado de checkpoints",
    )
    p.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    p.add_argument(
        "--spawn_point_indices",
        type=_parse_spawn_indices,
        default=None,
        help="Lista CSV de índices de spawn points a usar (ej. '1,5,42,118'). "
        "Si se especifica, CarlaEnv randomiza el spawn DENTRO de ese subset. "
        "Si vacío/no especificado, randomiza sobre todos los spawns del mapa. "
        "Útil para curriculum: P1 spawns fáciles (rectas), P2-P4 ampliando. "
        "Generar lista con `python profile_spawns.py --map Town04`.",
    )
    p.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Ruta a un checkpoint .pth previo para warm-start. Carga policy, "
        "obs_normalizer, ret_rms y returns_acc (formato v3). Si la ruta es "
        "relativa y no se encuentra, se busca también bajo ./data/models/. "
        "Útil para curricula multi-fase donde cada fase continúa el "
        "aprendizaje de la anterior (ver train_curriculum.py). NOTA: "
        "entropy_coef NO se persiste — reinicia al valor de --entropy_coef "
        "cada fase. Pasa --entropy_coef 0 en fases posteriores a la 1 si "
        "no quieres re-introducir presión upward sobre log_std.",
    )

    return p.parse_args()


def build_env(args, num_npc_override: int = None):
    """
    Construye la cadena de wrappers:
        CarlaEnv  →  Shield  →  CarlaRewardShaper
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
        success_reward=60.0,
        out_of_road_penalty=50.0,
        crash_penalty=10.0,
        seed=args.seed,
        spawn_point_indices=args.spawn_point_indices,
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
        # meta_tunable se activa si el meta-controlador intra-run está ON
        # (--meta_tuner) o si se pide una permisividad inicial <1.0. A 1.0 sin
        # meta_tuner el shield queda EXACTAMENTE como en producción.
        starts_loosened = args.shield_permissiveness < 1.0 - 1e-9
        use_meta = getattr(args, "meta_tuner", False) or starts_loosened
        env = CarlaAdaptiveHorizonShield(
            env,
            num_lidar_rays=num_lidar_rays,
            front_threshold_base=args.front_threshold,
            side_threshold_base=args.side_threshold,
            meta_tunable=use_meta,
        )
        if starts_loosened:
            env.set_permissiveness(args.shield_permissiveness)
            logger.info(
                f"[Shield] permissiveness inicial = "
                f"{args.shield_permissiveness:.3f} (1.0=strict, 0.0=weaned)"
            )
        elif use_meta:
            logger.info(
                "[Shield] meta-tunable ON; permissiveness arranca en 1.0 (strict)"
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
        idle_penalty_weight=args.idle_penalty_weight,
        progress_reward_weight=args.progress_reward_weight,
        acceleration_reward_weight=args.acceleration_reward_weight,
    )

    return env, num_lidar_rays


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

    final_model_path = models_dir / f"{args.model_name}_{args.shield_type}_final.pth"
    best_model_path = models_dir / f"{args.model_name}_{args.shield_type}_best.pth"

    reward_window = deque(maxlen=100)
    success_window = deque(maxlen=100)
    crash_window = deque(maxlen=100)
    offroad_window = deque(maxlen=100)
    stuck_window = deque(maxlen=100)
    shield_window = deque(maxlen=100)
    best_avg_reward = -float("inf")

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

    kl_target = args.kl_target if args.kl_target > 0 else None
    normalize_obs = args.obs_norm
    agent = PPOAgent(
        state_dim,
        action_dim,
        lr=args.lr,
        scheduler_t_max=expected_updates,
        k_epochs=args.k_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        entropy_coef_min=args.entropy_coef_min,
        entropy_coef_decay_updates=args.entropy_coef_decay_updates,
        value_loss_coef=args.value_loss_coef,
        kl_target=kl_target,
        normalize_obs=normalize_obs,
    )
    logger.info(
        f"PPO: lr={args.lr} | update_timestep={args.update_timestep} | "
        f"minibatch_size={args.minibatch_size} | kl_target={kl_target} | "
        f"obs_norm={normalize_obs}"
    )

    if args.load_model:
        load_path = Path(args.load_model)
        if not load_path.is_file():
            alt_path = models_dir / args.load_model
            if alt_path.is_file():
                load_path = alt_path
            else:
                logger.error(
                    f"[load_model] No se encontró checkpoint en "
                    f"'{args.load_model}' ni en '{alt_path}'. Abortando "
                    f"para evitar entrenar desde cero un curriculum."
                )
                raise FileNotFoundError(args.load_model)
        logger.info(f"[load_model] Cargando warm-start desde {load_path}")
        agent.load(str(load_path))
        logger.info(
            f"[load_model] OK. La política y normalizadores se cargaron. "
            f"entropy_coef se reinicia al valor pasado (={args.entropy_coef})."
        )

    # ── Meta-controlador LLM (Modo B — intra-run) ──────────────────────────
    meta_controller = None
    meta_runner = None
    if args.meta_tuner:
        meta_shield = _get_shield(env)
        if args.shield_type != "adaptive" or not hasattr(
            meta_shield, "set_permissiveness"
        ):
            logger.error(
                "--meta_tuner requiere --shield_type adaptive (shield destetable). "
                "Meta-controlador DESACTIVADO."
            )
        else:
            from src.meta.llm_tuner import LLMShieldTuner
            from src.meta.meta_controller import AsyncMetaRunner, MetaController
            from src.meta.safety_governor import GovernorConfig, SafetyGovernor

            _tuner = LLMShieldTuner(
                model=args.llm_model,
                host=args.llm_host,
                seed=args.meta_seed,
                num_gpu=None if args.llm_gpu else 0,
                keep_alive=args.llm_keep_alive,
                request_timeout=args.llm_timeout,
            )
            _device = "GPU" if args.llm_gpu else "CPU"
            logger.info(f"[meta] precargando {args.llm_model} en {_device}...")
            _ok, _msg = _tuner.warmup()
            logger.info(f"[meta] warmup {'OK' if _ok else 'FALLÓ'} ({_device}): {_msg}")
            if not _ok:
                logger.warning(
                    "[meta] El LLM no respondió en warmup; el meta-controlador "
                    "seguirá activo pero hará fallback no-op (shield strict) hasta "
                    "que Ollama responda. ¿Está `ollama serve` arriba y el modelo "
                    "pulled? Si es contención de GPU, este run ya usa CPU."
                )
            meta_controller = MetaController(
                _tuner,
                SafetyGovernor(GovernorConfig()),
                interval_episodes=args.meta_interval,
            )
            meta_runner = AsyncMetaRunner(meta_controller)
            logger.info(
                f"[meta] Modo B ACTIVO | model={args.llm_model} ({_device}) | "
                f"interval={args.meta_interval} eps | permissiveness arranca en "
                f"{meta_shield.get_permissiveness():.3f}"
            )

    memory = {
        "states": [],
        "raw_actions": [],
        "log_probs": [],
        "rewards": [],
        "dones": [],
        "truncated": [],
        "final_values": [],
        "shield_mask": [],
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

                action, raw_action, log_prob, _ = agent.select_action(obs)
                next_obs, reward, done, truncated, info = env.step(action)
                ep_infos.append(info)

                shield_activated = bool(
                    info.get("shield_activated", info.get("shield_active", False))
                )
                shield_alpha = float(
                    info.get("shield_intensity", 1.0 if shield_activated else 0.0)
                )

                executed_action = info.get("executed_action", action)
                if shield_alpha > 1e-4:
                    raw_action, log_prob = agent.evaluate_executed_action(
                        obs, executed_action
                    )

                is_truncated = truncated and not done
                if is_truncated:
                    final_value = agent.compute_bootstrap_value(next_obs)
                else:
                    final_value = 0.0

                memory["states"].append(obs)
                memory["raw_actions"].append(raw_action)
                memory["log_probs"].append(log_prob)
                memory["rewards"].append(reward)
                memory["dones"].append(done)
                memory["truncated"].append(is_truncated)
                memory["final_values"].append(final_value)

                memory["shield_mask"].append(shield_alpha)

                if shield_activated:
                    ep_shield_activations += 1

                obs = next_obs
                episode_reward += reward

                # Actualización de política
                if timestep % args.update_timestep == 0 and len(memory["states"]) > 0:
                    train_metrics = agent.update(memory)
                    for key in memory:
                        memory[key] = []

                    agent.step_scheduler()
                    agent.step_entropy_decay()

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
                            "Training/Epochs_Rejected": train_metrics.get(
                                "epochs_rejected", 0
                            ),
                            "Training/Shielded_Fraction": train_metrics.get(
                                "shielded_fraction", 0.0
                            ),
                            "Training/Mean_Shield_Alpha": train_metrics.get(
                                "mean_shield_alpha", 0.0
                            ),
                            "Training/Learning_Rate": agent.get_lr(),
                            "Training/Log_Std_Steering_Raw": train_metrics.get(
                                "log_std_steering_raw", 0.0
                            ),
                            "Training/Log_Std_Throttle_Raw": train_metrics.get(
                                "log_std_throttle_raw", 0.0
                            ),
                            "Training/Log_Std_Saturated_Fraction": train_metrics.get(
                                "log_std_saturated_fraction", 0.0
                            ),
                            "Training/Entropy_Coef_Current": train_metrics.get(
                                "entropy_coef_current", 0.0
                            ),
                        },
                    )

                if done or truncated:
                    break

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
            stuck_window.append(1 if outcome == 2 else 0)

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

            # Métricas de episodio (fuente unificada)
            shield_rate = ep_shield_activations / max(ep_steps, 1)
            shield_window.append(shield_rate)
            shield_semantic_metrics = {
                "Safety/Semantic/Dynamic_Interventions": 0.0,
                "Safety/Semantic/Static_Interventions": 0.0,
                "Safety/Semantic/Pedestrian_Interventions": 0.0,
                "Safety/Semantic/Recovery_Activations": 0.0,
                "Safety/Semantic/Safe_Step_Rate": 0.0,
                "Safety/Semantic/Warning_Step_Rate": 0.0,
                "Safety/Semantic/Critical_Step_Rate": 0.0,
                "Safety/Horizon/H1_Activations": 0.0,
                "Safety/Horizon/H5_Activations": 0.0,
                "Safety/Horizon/H10_Activations": 0.0,
            }

            # Desglose semántico del shield (si el shield lo expone)
            shield_wrapper = _get_shield(env)
            if shield_wrapper is not None:
                stats = shield_wrapper.get_statistics()
                _by_horizon = stats.get("interventions_by_horizon", {})
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
                    "Safety/Semantic/Recovery_Activations": stats.get(
                        "recovery_activations", 0.0
                    ),
                    "Safety/Semantic/Safe_Step_Rate": stats.get("safe_rate", 0.0),
                    "Safety/Semantic/Warning_Step_Rate": stats.get("warning_rate", 0.0),
                    "Safety/Semantic/Critical_Step_Rate": stats.get(
                        "critical_rate", 0.0
                    ),
                    "Safety/Horizon/H1_Activations": float(_by_horizon.get(1, 0)),
                    "Safety/Horizon/H5_Activations": float(_by_horizon.get(5, 0)),
                    "Safety/Horizon/H10_Activations": float(_by_horizon.get(10, 0)),
                }
                shield_wrapper.reset_statistics()

            # ── Meta-controlador LLM (Modo B): aplicar una decisión lista y,
            # cada N episodios, lanzar otra en background. Nunca bloquea el loop:
            # poll() devuelve sólo si el worker ya terminó; submit corre off-thread.
            if meta_runner is not None and shield_wrapper is not None:
                meta_decision = meta_runner.poll()
                if meta_decision is not None:
                    meta_controller.apply(shield_wrapper, meta_decision)
                    live_metrics.log_meta_decision(meta_decision.audit_record())
                    if meta_decision.changed:
                        logger.info(
                            f"[meta] Ep {episode}: permissiveness "
                            f"{meta_decision.snapshot.current_permissiveness:.3f} → "
                            f"{meta_decision.applied_permissiveness:.3f} | "
                            f"LLM({meta_decision.proposal.action}, "
                            f"ok={meta_decision.proposal.ok}) | "
                            f"governor={meta_decision.governor.verdict}"
                        )
                if meta_controller.should_run(episode) and not meta_runner.busy():
                    meta_snapshot = build_kpi_snapshot(
                        episode=episode,
                        current_permissiveness=shield_wrapper.get_permissiveness(),
                        crash_rate=crash_rate,
                        offroad_rate=offroad_rate,
                        stuck_rate=(
                            float(np.mean(stuck_window)) if stuck_window else 0.0
                        ),
                        success_rate=success_rate,
                        shielded_fraction=(
                            float(np.mean(shield_window)) if shield_window else 0.0
                        ),
                        window_episodes=len(crash_window),
                        shield_stats=stats,
                        avg_reward_100=avg_reward_100,
                        mean_speed_kmh=_ep_mean(ep_infos, "speed_kmh"),
                        num_npc=current_npc_count,
                    )
                    meta_runner.submit_if_idle(meta_snapshot, episode)

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
                    # Componentes del shaper
                    "Reward/Components/Alive_Bonus": _ep_sum(ep_infos, "alive_bonus"),
                    "Reward/Components/Acceleration_Reward": _ep_sum(
                        ep_infos, "acceleration_reward"
                    ),
                    "Reward/Components/Speed_Bonus": _ep_mean(ep_infos, "speed_bonus"),
                    "Reward/Components/Lane_Centering": _ep_mean(
                        ep_infos, "lane_center_bonus"
                    ),
                    "Reward/Components/Heading_Alignment": _ep_mean(
                        ep_infos, "heading_bonus"
                    ),
                    "Reward/Components/Smooth_Penalty": _ep_mean(
                        ep_infos, "smooth_penalty"
                    ),
                    "Reward/Components/Invasion_Penalty": _ep_mean(
                        ep_infos, "invasion_penalty"
                    ),
                    "Reward/Components/Road_Penalty": _ep_mean(
                        ep_infos, "road_penalty"
                    ),
                    "Reward/Components/Progress_Bonus": _ep_sum(
                        ep_infos, "progress_bonus"
                    ),
                    "Reward/Components/Idle_Penalty": _ep_sum(ep_infos, "idle_penalty"),
                    "Reward/Components/Shield_Intensity_Mean": _ep_mean(
                        ep_infos, "shield_intensity"
                    ),
                    "Reward/Components/Wrong_Heading_Pen": _ep_sum(
                        ep_infos, "wrong_heading_pen"
                    ),
                    "Reward/Components/Drift_Penalty": _ep_sum(
                        ep_infos, "drift_penalty"
                    ),
                    "Reward/Components/Solid_Invasion_Pen": _ep_sum(
                        ep_infos, "solid_invasion_penalty"
                    ),
                    "Reward/Components/Solid_Invasion_Events": _ep_sum(
                        ep_infos, "solid_invasion_event"
                    ),
                    "Reward/Components/Lane_Change_Cost": _ep_sum(
                        ep_infos, "lane_change_cost"
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
                    "Safety/Min_Vehicle_Distance_m": min_veh_m,
                    "Safety/Min_Pedestrian_Distance_m": min_ped_m,
                    "Safety/Vehicle_Detected": 1 if min_veh_m < 999.0 else 0,
                    "Safety/Pedestrian_Detected": 1 if min_ped_m < 999.0 else 0,
                    "Safety/Min_Front_Dynamic": _ep_min(ep_infos, "min_front_dynamic"),
                    "Safety/Min_Front_Combined": _ep_min(
                        ep_infos, "min_front_combined"
                    ),
                    "Safety/Min_Front_Static": _ep_min(ep_infos, "min_front_static"),
                    "Safety/Min_Side_L": _ep_min(ep_infos, "min_l_side_combined"),
                    "Safety/Min_Side_R": _ep_min(ep_infos, "min_r_side_combined"),
                    "Safety/Nearest_Static_m": _ep_min(
                        ep_infos, "nearest_static_m", default=999.0
                    ),
                    "Lidar/Pts_Per_Frame": _ep_mean(
                        ep_infos, "semantic_pts_per_frame", default=0.0
                    ),
                    "Lidar/Stale_Ratio": _ep_mean(
                        ep_infos, "semantic_stale_ratio", default=0.0
                    ),
                    "Lidar/Fresh_Rate": float(
                        np.mean(
                            [int(i.get("semantic_data_fresh", False)) for i in ep_infos]
                        )
                    ),
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
                    "CARLA/Mean_TTC_s": _ep_mean(ep_infos, "ttc_seconds"),
                    "CARLA/Min_TTC_s": _ep_min(ep_infos, "ttc_seconds", default=1e6),
                    "CARLA/Mean_Abs_Steering": float(
                        np.mean([abs(i.get("steering", 0.0)) for i in ep_infos])
                        if ep_infos
                        else 0.0
                    ),
                    "CARLA/Low_Speed_Fraction": _ep_mean(
                        ep_infos, "low_speed_fraction"
                    ),
                    "CARLA/Lane_Changes_Ep": float(
                        sum(int(i.get("lane_change_event", False)) for i in ep_infos)
                    ),
                    # Outcome
                    "Outcome/Type": outcome,
                    "Outcome/Stuck_Rate": float(
                        np.mean([int(i.get("stuck", False)) for i in ep_infos])
                    ),
                    "Outcome/Timeout": 1 if outcome == 0 else 0,
                    "Outcome/Crash": 1 if outcome == 1 else 0,
                    "Outcome/Stuck": 1 if outcome == 2 else 0,
                    "Outcome/Offroad": 1 if outcome == 3 else 0,
                    "Outcome/Success": 1 if outcome == 4 else 0,
                    # Fracción de pasos del episodio con speed < 1 km/h.
                    # Distingue "timeout conduciendo" de "timeout parado".
                    "Outcome/Parked_Fraction": float(
                        np.mean([int(i.get("speed_kmh", 0.0) < 1.0) for i in ep_infos])
                    ),
                    # Meta-controlador LLM (permisividad viva del shield)
                    "Meta/Shield_Permissiveness": (
                        shield_wrapper.get_permissiveness()
                        if shield_wrapper is not None
                        and hasattr(shield_wrapper, "get_permissiveness")
                        else 1.0
                    ),
                    # Semantic shield
                    **shield_semantic_metrics,
                },
            )

            if episode >= 500 and avg_reward_100 > best_avg_reward + 20:
                best_avg_reward = avg_reward_100
                agent.save(str(best_model_path))
                logger.info(
                    f"New best model saved at Ep {episode} "
                    f"(Avg100: {best_avg_reward:.1f})"
                )

            # Checkpoint periódico
            if episode % args.ckpt_freq == 0:
                ckpt_path = ckpt_dir / f"checkpoint_ep_{episode}.pth"
                agent.save(str(ckpt_path))
                logger.info(f"Checkpoint saved: {ckpt_path.name}")

            # Print progreso
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
        # Parar el worker del meta-controlador (no-op si no estaba activo).
        if meta_runner is not None:
            meta_runner.shutdown()

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
