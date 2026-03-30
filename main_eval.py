"""
main_eval.py - Entrypoint de evaluación para CARLA Safe RL

Reemplaza main_eval_improved.py de MetaDrive.

DIFERENCIAS CLAVE vs versión MetaDrive:
  1. Dashboard rediseñado para mostrar datos CARLA:
       - Offset lateral en metros reales (Waypoint API)
       - Speed en km/h real
       - Heading error en grados reales
       - Invasiones de carril (sensor nativo CARLA)
  2. El snapshot semántico al final del episodio incluye
     métricas específicas de CARLA (total_distance, lane_invasions, etc.)
  3. Informe final usa SafetyMetricsReporter (ampliado para CARLA)
  4. Modo determinista disponible: --deterministic

USO:
    # Con shield adaptativo (por defecto):
    python main_eval.py --model_name mi_modelo_adaptive_final.pth

    # Sin shield:
    python main_eval.py --model_name baseline_none_final.pth --shield_type none

    # Sin render (solo métricas):
    python main_eval.py --model_name mi_modelo.pth --no_render --episodes 20
"""

import argparse
import os
import math
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.carla_env import CarlaEnv
from src.reward_shaper import CarlaRewardShaper
from src.safety_shield import CarlaSafetyShield
from src.adaptive_horizon_shield import CarlaAdaptiveHorizonShield
from src.ppo_agent import PPOAgent
from src.metrics import SafetyMetrics, SafetyMetricsReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main_eval")


# ══════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════

class CarlaDashboard:
    """
    Dashboard matplotlib para monitorear el agente durante evaluación.

    Muestra datos reales de CARLA:
      - Polar plot del scan LIDAR
      - Velocidad actual vs objetivo
      - Offset lateral (metros reales, Waypoint API)
      - Heading error (grados reales, Waypoint API)
      - Contador de intervenciones del shield
    """

    def __init__(self, num_lidar_rays: int = 240, front_threshold: float = 0.15,
                 shield_type: str = "none"):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 7))
        self.fig.suptitle("CARLA Safe RL — Agent Dashboard", fontsize=13, y=0.98)
        gs = gridspec.GridSpec(2, 3, figure=self.fig, hspace=0.45, wspace=0.35)

        # ── LIDAR polar ────────────────────────────────────────────────
        self.ax_lidar = self.fig.add_subplot(gs[:, 0], projection="polar")
        self.num_lidar_rays = num_lidar_rays
        self.angles = np.linspace(0, 2 * np.pi, num_lidar_rays, endpoint=False)
        (self.lidar_line,) = self.ax_lidar.plot([], [], color="steelblue", linewidth=1.5)
        # Umbral de seguridad
        theta_c = np.linspace(0, 2 * np.pi, 200)
        self.ax_lidar.plot(theta_c, np.full(200, front_threshold),
                           color="red", linestyle="--", linewidth=1.2,
                           label=f"Threshold ({front_threshold:.2f})")
        self.ax_lidar.fill_between(theta_c, 0, front_threshold, color="red", alpha=0.07)
        self.ax_lidar.set_ylim(0, 1)
        self.ax_lidar.set_yticklabels([])
        self.ax_lidar.set_title("LIDAR scan", pad=14, fontsize=10)
        self.ax_lidar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=8)

        # ── Speed gauge ────────────────────────────────────────────────
        self.ax_speed = self.fig.add_subplot(gs[0, 1])
        self.ax_speed.set_title("Speed (km/h)", fontsize=10)
        self.ax_speed.set_xlim(0, 80)
        self.ax_speed.set_ylim(0, 1)
        self.ax_speed.set_yticks([])
        self.speed_bar = self.ax_speed.barh(0, 0, height=0.6, color="steelblue", align="center")
        self.ax_speed.axvline(30, color="green", linestyle="--", linewidth=1.2, label="Target")
        self.ax_speed.legend(fontsize=8)
        self.speed_text = self.ax_speed.text(40, 0, "0.0", va="center", fontsize=10)

        # ── Lateral offset ─────────────────────────────────────────────
        self.ax_lat = self.fig.add_subplot(gs[0, 2])
        self.ax_lat.set_title("Lateral offset (norm)", fontsize=10)
        self.ax_lat.set_xlim(-1.1, 1.1)
        self.ax_lat.set_ylim(0, 1)
        self.ax_lat.set_yticks([])
        self.ax_lat.axvline(0, color="gray", linewidth=0.8)
        self.ax_lat.axvline(0.82, color="orange", linestyle=":", linewidth=1.0)
        self.ax_lat.axvline(-0.82, color="orange", linestyle=":", linewidth=1.0)
        self.lat_marker = self.ax_lat.plot([0], [0.5], "D", color="steelblue", ms=10)[0]
        self.ax_lat.text(0, 0.15, "center", ha="center", fontsize=8, color="gray")

        # ── Agent info text ────────────────────────────────────────────
        self.ax_info = self.fig.add_subplot(gs[1, 1:])
        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(
            0.02, 0.95, "", transform=self.ax_info.transAxes,
            va="top", fontfamily="monospace", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#f0f4f8", alpha=0.8),
        )

        self.shield_type = shield_type
        plt.tight_layout()

    def update(self, obs: np.ndarray, action: np.ndarray, info: Dict,
               episode: int, step: int, total_shields: int):
        """Actualiza todos los paneles del dashboard."""

        # LIDAR
        lidar_data = obs[:self.num_lidar_rays]
        self.lidar_line.set_data(self.angles, lidar_data)

        # Speed bar
        speed_kmh = info.get("speed_kmh", 0.0)
        self.speed_bar[0].set_width(min(speed_kmh, 80.0))
        bar_color = "green" if abs(speed_kmh - 30.0) < 5.0 else (
            "orange" if abs(speed_kmh - 30.0) < 15.0 else "red"
        )
        self.speed_bar[0].set_color(bar_color)
        self.speed_text.set_text(f"{speed_kmh:.1f}")

        # Lateral offset
        lat_norm = info.get("lateral_offset_norm", 0.0)
        self.lat_marker.set_xdata([lat_norm])
        lat_color = "red" if abs(lat_norm) > 0.82 else (
            "orange" if abs(lat_norm) > 0.60 else "steelblue"
        )
        self.lat_marker.set_color(lat_color)

        # Info text
        heading_err = info.get("heading_error", 0.0)
        on_road     = info.get("on_road", True)
        risk        = info.get("risk_level", "—")
        shield_on   = info.get("shield_activated", info.get("shield_active", False))
        lat_m       = info.get("lateral_offset", 0.0)
        lane_inv    = info.get("episode_lane_invasions", 0)
        collisions  = info.get("episode_collisions", 0)
        dist        = info.get("total_distance", 0.0)
        min_dist    = info.get("min_distance", info.get("min_front_dist", 1.0))

        text = (
            f"Episode {episode} | Step {step}\n"
            f"{'─'*38}\n"
            f"Speed:          {speed_kmh:>6.1f} km/h\n"
            f"Lat offset:     {lat_m:>+6.3f} m  (norm {lat_norm:>+5.2f})\n"
            f"Heading error:  {heading_err:>+6.1f}°\n"
            f"On road:        {'YES' if on_road else 'NO ⚠️'}\n"
            f"Min LIDAR dist: {min_dist:>6.3f}\n"
            f"{'─'*38}\n"
            f"Shield type:    {self.shield_type.upper()}\n"
            f"Risk level:     {risk.upper()}\n"
            f"Shield active:  {'YES ⚡' if shield_on else 'no'}\n"
            f"Total shields:  {total_shields}\n"
            f"{'─'*38}\n"
            f"Total distance: {dist:>6.1f} m\n"
            f"Lane invasions: {lane_inv}\n"
            f"Collisions:     {collisions}\n"
            f"Steer: {action[0]:>+.3f}  |  Throttle/Brake: {action[1]:>+.3f}"
        )
        self.info_text.set_text(text)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.close("all")


# ══════════════════════════════════════════════════════════════════════
# ARGUMENTOS
# ══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(description="Evaluación del agente PPO en CARLA")

    p.add_argument("--model_name", type=str, required=True,
                   help="Nombre del archivo .pth del modelo")
    p.add_argument("--shield_type", type=str,
                   choices=["none", "basic", "adaptive"], default="adaptive")

    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm_port", type=int, default=8000)
    p.add_argument("--map", type=str, default="Town04")
    p.add_argument("--num_npc", type=int, default=20)
    p.add_argument("--weather", type=str, default="ClearNoon")
    p.add_argument("--target_speed_kmh", type=float, default=30.0)
    p.add_argument("--success_distance", type=float, default=250.0)

    p.add_argument("--front_threshold", type=float, default=0.15)
    p.add_argument("--side_threshold", type=float, default=0.04)
    p.add_argument("--lateral_threshold", type=float, default=0.82)

    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--no_render", action="store_true",
                   help="Deshabilitar renderizado (solo métricas)")
    p.add_argument("--no_dashboard", action="store_true",
                   help="Deshabilitar dashboard matplotlib")
    p.add_argument("--deterministic", action="store_true",
                   help="Política determinista (sin muestreo)")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL ENTORNO (idéntica lógica a main_train.py)
# ══════════════════════════════════════════════════════════════════════

def build_env(args, render: bool = True):
    """Construye la cadena de wrappers para evaluación."""
    num_lidar_rays = 240

    env = CarlaEnv(
        host=args.host,
        port=args.port,
        tm_port=args.tm_port,
        map_name=args.map,
        num_npc_vehicles=args.num_npc,
        weather=args.weather,
        render_mode="human" if render else None,
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
        seed=100,  # Semilla diferente a entrenamiento
    )

    # Reward shaping desactivado para evaluación pura (solo logging)
    env = CarlaRewardShaper(
        env,
        target_speed_kmh=args.target_speed_kmh,
        speed_weight=0.0,
        smoothness_weight=0.0,
        lane_centering_weight=0.0,
        lane_invasion_penalty=0.0,
        off_road_penalty=0.0,
    )

    if args.shield_type == "basic":
        logger.info("🛡️  Shield: CarlaSafetyShield")
        env = CarlaSafetyShield(
            env,
            num_lidar_rays=num_lidar_rays,
            front_threshold=args.front_threshold,
            side_threshold=args.side_threshold,
            lateral_threshold=args.lateral_threshold,
        )
    elif args.shield_type == "adaptive":
        logger.info("🛡️  Shield: CarlaAdaptiveHorizonShield")
        env = CarlaAdaptiveHorizonShield(
            env,
            num_lidar_rays=num_lidar_rays,
            front_threshold_base=args.front_threshold,
            side_threshold_base=args.side_threshold,
            lateral_threshold_base=args.lateral_threshold,
        )
    else:
        logger.info("⚠️  Sin shield")

    return env, num_lidar_rays


# ══════════════════════════════════════════════════════════════════════
# EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════

def evaluate():
    args = get_args()

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING TRAINED AGENT — CARLA")
    logger.info("=" * 70)
    logger.info(f"Model:   {args.model_name}")
    logger.info(f"Shield:  {args.shield_type}")
    logger.info(f"Map:     {args.map}")
    logger.info(f"Episodes:{args.episodes}")
    logger.info("=" * 70 + "\n")

    # ── Localizar modelo ───────────────────────────────────────────────
    model_path = Path("./data/models") / args.model_name
    if not model_path.exists():
        model_path = Path(args.model_name)
    if not model_path.exists():
        logger.error(f"Model not found: {args.model_name}")
        return

    logger.info(f"Loading model from: {model_path}")

    # ── Entorno ────────────────────────────────────────────────────────
    env, num_lidar_rays = build_env(args, render=not args.no_render)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ── Agente ─────────────────────────────────────────────────────────
    agent = PPOAgent(state_dim, action_dim)
    agent.load(str(model_path))
    agent.policy.eval()

    # ── Dashboard ──────────────────────────────────────────────────────
    show_dashboard = not args.no_dashboard
    dashboard = None
    if show_dashboard:
        dashboard = CarlaDashboard(
            num_lidar_rays=num_lidar_rays,
            front_threshold=args.front_threshold,
            shield_type=args.shield_type,
        )

    # ── Variables de evaluación ────────────────────────────────────────
    all_episodes: List[List[Dict]] = []
    all_infos:    List[Dict]       = []

    total_rewards  = []
    successes      = 0
    crashes        = 0
    timeouts       = 0
    total_shields  = 0

    header = f"{'Episode':<9} {'Reward':>8} {'Status':<22} {'Dist(m)':>8} {'Shields':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    try:
        for ep in range(1, args.episodes + 1):
            obs, _       = env.reset()
            ep_reward    = 0.0
            ep_infos: List[Dict] = []
            done         = False
            truncated    = False
            step         = 0

            while not (done or truncated) and step < args.max_steps:
                if args.deterministic:
                    action, _, _ = agent.select_action(obs, deterministic=True)
                else:
                    action, _, _ = agent.select_action(obs)

                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                step      += 1
                ep_infos.append(info)

                if not args.no_render:
                    env.render()

                if info.get("shield_activated", info.get("shield_active", False)):
                    total_shields += 1

                if dashboard is not None:
                    dashboard.update(obs, action, info, ep, step, total_shields)

            # ── Análisis del episodio ──────────────────────────────────
            outcome = "timeout"
            if info.get("collision", False):
                crashes += 1
                outcome = "crash 💥"
            elif info.get("arrive_dest", False):
                successes += 1
                outcome = "success ✅"
            elif info.get("out_of_road", False):
                crashes += 1
                outcome = "off-road ⚠️"
            else:
                timeouts += 1

            dist     = info.get("total_distance", 0.0)
            ep_shields = sum(
                1 for i in ep_infos
                if i.get("shield_activated", i.get("shield_active", False))
            )

            logger.info(
                f"Ep {ep:<6} {ep_reward:>8.2f}  {outcome:<22} "
                f"{dist:>8.1f}  {ep_shields:>8}"
            )

            total_rewards.append(ep_reward)
            all_episodes.append(ep_infos)
            all_infos.extend(ep_infos)

    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user.")

    finally:
        env.close()
        if dashboard:
            dashboard.close()

        # ── Resumen final ──────────────────────────────────────────────
        n = len(total_rewards)
        if n == 0:
            return

        avg_reward = float(np.mean(total_rewards))
        std_reward = float(np.std(total_rewards))

        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"\nModel:          {args.model_name}")
        logger.info(f"Shield:         {args.shield_type}")
        logger.info(f"Map:            {args.map}")
        logger.info(f"\nEpisodes:       {n}")
        logger.info(f"Avg reward:     {avg_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"Max reward:     {max(total_rewards):.2f}")
        logger.info(f"Min reward:     {min(total_rewards):.2f}")
        logger.info(f"\nSuccess rate:   {successes/n:.1%}  ({successes}/{n})")
        logger.info(f"Crash rate:     {crashes/n:.1%}  ({crashes}/{n})")
        logger.info(f"Timeout rate:   {timeouts/n:.1%}  ({timeouts}/{n})")

        if args.shield_type != "none":
            logger.info(f"\nTotal shield interventions: {total_shields}")
            logger.info(f"Avg per episode:            {total_shields/n:.1f}")

        # Reporte completo de métricas de seguridad
        if all_infos:
            report = SafetyMetricsReporter.generate_report(
                all_infos=all_infos,
                all_episodes=all_episodes,
                shield_type=args.shield_type,
            )
            logger.info(report)


if __name__ == "__main__":
    evaluate()
