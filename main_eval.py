"""
main_eval.py - Entrypoint de evaluación para CARLA Safe RL

USO:
    # Con shield adaptativo (por defecto):
    python main_eval.py --model_name mi_modelo_adaptive_final.pth

    # Sin shield:
    python main_eval.py --model_name baseline_none_final.pth --shield_type none

    # Sin render (solo métricas):
    python main_eval.py --model_name mi_modelo.pth --no_render --episodes 20
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from src.Adaptative_Shield.adaptive_horizon_shield import CarlaAdaptiveHorizonShield
from src.CARLA.Env.carla_env import CarlaEnv
from src.Metrics.EvalMetrics.metrics import SafetyMetricsReporter
from src.PPO.ppo_agent import PPOAgent
from src.reward_shaper import CarlaRewardShaper
from src.safety_shield import CarlaSafetyShield

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

    def __init__(
        self,
        num_lidar_rays: int = 240,
        front_threshold: float = 0.15,
        shield_type: str = "none",
        fallback_target_kmh: float = 30.0,
        lateral_threshold: float = 0.65,
    ):
        plt.ion()
        self.fig = plt.figure(figsize=(14, 7))
        self.fig.suptitle("CARLA Safe RL — Agent Dashboard", fontsize=13, y=0.98)
        gs = gridspec.GridSpec(2, 3, figure=self.fig, hspace=0.45, wspace=0.35)

        # ── LIDAR polar (combined + dynamic overlay) ───────────────────
        self.ax_lidar = self.fig.add_subplot(gs[:, 0], projection="polar")
        self.num_lidar_rays = num_lidar_rays
        self.angles = np.linspace(0, 2 * np.pi, num_lidar_rays, endpoint=False)
        self.ax_lidar.set_theta_zero_location("N")  # θ=0 = Norte = 12 o'clock = FRENTE
        self.ax_lidar.set_theta_direction(1)
        # combined scan LIDAR alto (techo): obstáculos + bordes de carretera
        (self.lidar_line,) = self.ax_lidar.plot(
            [], [], color="steelblue", linewidth=1.5, label="High combined"
        )
        # dynamic scan: vehículos y peatones
        (self.lidar_dynamic_line,) = self.ax_lidar.plot(
            [], [], color="darkorange", linewidth=1.2, linestyle="-", label="Dynamic"
        )
        # static scan: muros, quitamiedos, postes
        (self.lidar_static_line,) = self.ax_lidar.plot(
            [],
            [],
            color="mediumseagreen",
            linewidth=1.0,
            linestyle="--",
            label="Static",
        )
        # LIDAR bajo (parachoques delantero, range 30 m): guardarraíles bajos
        (self.lidar_low_line,) = self.ax_lidar.plot(
            [],
            [],
            color="purple",
            linewidth=1.3,
            linestyle="-",
            label="Low combined",
        )
        # Sub-scans semánticos del LIDAR alto que el procesador ya calcula
        # pero que hasta ahora no se exhibían:
        #   pedestrian_scan → solo peatones (alta prioridad de safety)
        #   road_edge_scan  → acera (8) + terreno (22): límite físico de
        #                     calzada. Útil para auditar el comportamiento
        #                     del shield en relación con el bordillo.
        (self.lidar_pedestrian_line,) = self.ax_lidar.plot(
            [],
            [],
            color="red",
            linewidth=1.4,
            linestyle="-",
            marker="x",
            markersize=4,
            label="Pedestrian",
        )
        (self.lidar_road_edge_line,) = self.ax_lidar.plot(
            [],
            [],
            color="goldenrod",
            linewidth=0.9,
            linestyle=":",
            label="Road edge",
        )
        # Indicador de frescura: punto en el centro del polar plot que se
        # vuelve verde cuando el LIDAR alto entregó un frame fresco para el
        # tick actual y rojo cuando no. Permite detectar de un vistazo
        # cualquier desincronía sensor-mundo durante la evaluación.
        (self.fresh_marker,) = self.ax_lidar.plot(
            [0.0], [0.0], marker="o", color="green", markersize=8, zorder=10
        )
        # Umbral de seguridad — wedge frontal de ±FRONT_N bins (≈ ±22.5°)
        # en lugar del antiguo círculo completo. El front_threshold solo
        # aplica al frente (FRONT_N=15 bins de 240 → ±22.5°). Pintarlo a
        # 360° era visualmente engañoso porque sugería un umbral activo
        # también lateral, cuando ahí actúa side_threshold (no dibujado).
        FRONT_N = 15
        front_half_angle = (FRONT_N / num_lidar_rays) * 2 * np.pi
        # En matplotlib polar con set_theta_zero_location("N") el ángulo 0
        # corresponde al frente y crece antihorario; un wedge centrado en
        # el frente se pinta entre -front_half_angle y +front_half_angle
        # (matplotlib normaliza ángulos negativos automáticamente).
        theta_w = np.linspace(-front_half_angle, front_half_angle, 100)
        self.ax_lidar.plot(
            theta_w,
            np.full_like(theta_w, front_threshold),
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"Front threshold ({front_threshold:.2f})",
        )
        self.ax_lidar.fill_between(
            theta_w, 0, front_threshold, color="red", alpha=0.12
        )
        self.ax_lidar.set_ylim(0, 1)
        # Ticks radiales en metros para que el usuario lea distancias reales
        # (antes el eje radial iba en [0,1] sin unidades: 0.075 parecía
        # "obstáculo a 7 cm" cuando en realidad era 3.75 m).
        self.ax_lidar.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
        self.ax_lidar.set_yticklabels(["5 m", "15 m", "25 m", "35 m", "45 m"])
        self.ax_lidar.set_title(
            "LIDAR polar (rango 50 m) — alto: combined/dyn/stat  ·  bajo: combined",
            pad=14,
            fontsize=9,
        )
        self.ax_lidar.legend(
            loc="lower center", bbox_to_anchor=(0.5, -0.25), fontsize=7, ncol=4
        )

        # ── Speed gauge ────────────────────────────────────────────────
        self.ax_speed = self.fig.add_subplot(gs[0, 1])
        self.ax_speed.set_title("Speed (km/h)", fontsize=10)
        self.ax_speed.set_xlim(0, 140)
        self.ax_speed.set_ylim(0, 1)
        self.ax_speed.set_yticks([])
        self.speed_bar = self.ax_speed.barh(
            0, 0, height=0.6, color="steelblue", align="center"
        )

        self._speed_target_line = self.ax_speed.axvline(
            fallback_target_kmh,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label="Limit",
        )

        self.ax_speed.legend(fontsize=8)
        self.speed_text = self.ax_speed.text(80, 0, "0.0 / --", va="center", fontsize=9)
        self._fallback_target_kmh = fallback_target_kmh

        # ── Lateral offset ─────────────────────────────────────────────
        self.ax_lat = self.fig.add_subplot(gs[0, 2])
        self.ax_lat.set_title("Lateral offset (norm)", fontsize=10)
        self.ax_lat.set_xlim(-1.1, 1.1)
        self.ax_lat.set_ylim(0, 1)
        self.ax_lat.set_yticks([])
        self.ax_lat.axvline(0, color="gray", linewidth=0.8)
        # Líneas de umbral lateral sincronizadas con el valor REAL configurado
        # del shield (`--lateral_threshold`). Antes se cableaban a ±0.82,
        # que era el default basic, pero el adaptive default es 0.65 — el
        # plot quedaba descalibrado tras los últimos ajustes de hyperparams.
        self._lateral_threshold = float(lateral_threshold)
        self.ax_lat.axvline(
            self._lateral_threshold,
            color="orange",
            linestyle=":",
            linewidth=1.0,
            label=f"lat th ({self._lateral_threshold:.2f})",
        )
        self.ax_lat.axvline(
            -self._lateral_threshold, color="orange", linestyle=":", linewidth=1.0
        )
        self.lat_marker = self.ax_lat.plot([0], [0.5], "D", color="steelblue", ms=10)[0]
        self.ax_lat.text(0, 0.15, "center", ha="center", fontsize=8, color="gray")

        # ── Agent info text ────────────────────────────────────────────
        self.ax_info = self.fig.add_subplot(gs[1, 1:])
        self.ax_info.axis("off")
        self.info_text = self.ax_info.text(
            0.02,
            0.95,
            "",
            transform=self.ax_info.transAxes,
            va="top",
            fontfamily="monospace",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#f0f4f8", alpha=0.8),
        )

        self.shield_type = shield_type
        plt.tight_layout()

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        info: Dict,
        episode: int,
        step: int,
        total_shields: int,
    ):
        """Actualiza todos los paneles del dashboard."""

        # LIDAR ALTO: combined/dynamic/static — LIDAR BAJO: combined del
        # sensor del parachoques (range 30 m, escalado al mismo eje radial
        # del LIDAR alto multiplicando por 30/50 = 0.6 para superponerlos
        # sin inducir falsa escala).
        n = self.num_lidar_rays
        lidar_combined = obs[:n]
        lidar_dynamic = obs[n : 2 * n]
        lidar_static = obs[2 * n : 3 * n]
        lidar_low = obs[3 * n : 4 * n]
        # Re-escalado del LIDAR bajo: un hit a 15 m (norm=0.5 en eje de 30 m)
        # se dibuja en radio 0.3 para coincidir con un hit a 15 m del alto.
        lidar_low_scaled = lidar_low * (30.0 / 50.0)
        self.lidar_line.set_data(self.angles, lidar_combined)
        self.lidar_dynamic_line.set_data(self.angles, lidar_dynamic)
        self.lidar_static_line.set_data(self.angles, lidar_static)
        # Enmascarado del LIDAR bajo: norm=1.0 = "no hit". Si pintamos esos
        # bins, aparece un anillo cosmético a radio 0.6 (= 30/50) en todo
        # el plot incluso sin obstáculos, y el usuario lo confunde con un
        # muro circular a 30 m. Solo dibujamos los bins con detección real.
        low_mask = lidar_low < 1.0
        if np.any(low_mask):
            self.lidar_low_line.set_data(
                self.angles[low_mask], lidar_low_scaled[low_mask]
            )
        else:
            # Sin detecciones — limpiamos la línea para no dejar el último
            # frame fantasma en pantalla.
            self.lidar_low_line.set_data([], [])

        # Sub-scans semánticos: peatón y borde de carretera. Se leen del
        # `info` (los puebla SemanticScanResult.to_info_dict). Aplicamos el
        # mismo enmascarado que el LIDAR bajo: solo dibujamos bins con hit
        # real para evitar líneas fantasma a radio 1.0.
        ped_scan = info.get("lidar_pedestrian_scan")
        if ped_scan is not None and len(ped_scan) == n:
            ped_arr = np.asarray(ped_scan)
            ped_mask = ped_arr < 1.0
            if np.any(ped_mask):
                self.lidar_pedestrian_line.set_data(
                    self.angles[ped_mask], ped_arr[ped_mask]
                )
            else:
                self.lidar_pedestrian_line.set_data([], [])
        edge_scan = info.get("lidar_road_edge_scan")
        if edge_scan is not None and len(edge_scan) == n:
            edge_arr = np.asarray(edge_scan)
            edge_mask = edge_arr < 1.0
            if np.any(edge_mask):
                self.lidar_road_edge_line.set_data(
                    self.angles[edge_mask], edge_arr[edge_mask]
                )
            else:
                self.lidar_road_edge_line.set_data([], [])

        # Speed bar
        speed_kmh = info.get("speed_kmh", 0.0)
        speed_limit = info.get("speed_limit_kmh", self._fallback_target_kmh)
        if speed_limit <= 0.0:
            speed_limit = self._fallback_target_kmh

        self._speed_target_line.set_xdata([speed_limit, speed_limit])

        self.speed_bar[0].set_width(min(speed_kmh, 140))

        # Color: verde ≤ límite, naranja hasta +20%, rojo por encima
        speed_ratio = speed_kmh / speed_limit if speed_limit > 0 else 1.0
        if speed_ratio <= 1.0:
            bar_color = "green"
        elif speed_ratio <= 1.2:
            bar_color = "orange"
        else:
            bar_color = "red"

        self.speed_bar[0].set_color(bar_color)
        self.speed_text.set_text(f"{speed_kmh:.1f} / {speed_limit:.0f}")

        # Lateral offset — colores referenciados al lateral_threshold
        # configurado del shield. Naranja a un 80% del umbral (early
        # warning), rojo al cruzarlo. Antes se usaban literales 0.82/0.60
        # cableados que no se actualizaban si cambiaba la config.
        lat_norm = info.get("lateral_offset_norm", 0.0)
        self.lat_marker.set_xdata([lat_norm])
        lt = self._lateral_threshold
        warn = 0.8 * lt
        lat_color = (
            "red"
            if abs(lat_norm) > lt
            else ("orange" if abs(lat_norm) > warn else "steelblue")
        )
        self.lat_marker.set_color(lat_color)

        # Info text
        heading_err = info.get("heading_error", 0.0)
        on_road = info.get("on_road", True)
        risk = info.get("risk_level", "—")
        shield_on = info.get("shield_activated", info.get("shield_active", False))
        lat_m = info.get("lateral_offset", 0.0)
        lane_inv = info.get("episode_lane_invasions", 0)
        collisions = info.get("episode_collisions", 0)
        dist = info.get("total_distance", 0.0)
        # `min_front_dist` está normalizado en [0,1] sobre el range 50 m.
        # Lo mostramos en METROS para evitar la confusión "0.075 = 7 cm?".
        min_dist_norm = info.get("min_distance", info.get("min_front_dist", 1.0))
        lidar_range_m = 50.0
        min_dist_m = min_dist_norm * lidar_range_m
        # LIDAR bajo (parachoques, range 30 m): distancia mínima frontal.
        low_min_norm = info.get("low_min_front_combined", 1.0)
        low_min_m = low_min_norm * 30.0

        # Frescura del LIDAR alto y bajo. Verde si el frame del sensor
        # cuadró con el world.tick() de este step; rojo en caso contrario.
        # El stale_ratio acumulado se imprime para detectar deriva.
        fresh_high = bool(info.get("semantic_data_fresh", True))
        fresh_low = bool(info.get("semantic_low_data_fresh", True))
        stale_ratio_high = float(info.get("semantic_stale_ratio", 0.0))
        stale_ratio_low = float(info.get("semantic_low_stale_ratio", 0.0))
        # Marker en el centro del polar plot. Combinamos alto y bajo:
        # verde si ambos frescos, naranja si solo uno, rojo si ninguno.
        if fresh_high and fresh_low:
            fresh_color = "green"
        elif fresh_high or fresh_low:
            fresh_color = "orange"
        else:
            fresh_color = "red"
        self.fresh_marker.set_color(fresh_color)

        text = (
            f"Episode {episode} | Step {step}\n"
            f"{'─' * 38}\n"
            f"Speed:          {speed_kmh:>6.1f} km/h\n"
            f"Lat offset:     {lat_m:>+6.3f} m  (norm {lat_norm:>+5.2f})\n"
            f"Heading error:  {heading_err:>+6.1f}°\n"
            f"On road:        {'YES' if on_road else 'NO ⚠️'}\n"
            f"Min LIDAR (hi): {min_dist_m:>6.2f} m  (norm {min_dist_norm:.3f})\n"
            f"Min LIDAR (lo): {low_min_m:>6.2f} m  (norm {low_min_norm:.3f})\n"
            f"LIDAR fresh:    hi={'Y' if fresh_high else 'N'} "
            f"(stale {stale_ratio_high:.2%})  "
            f"lo={'Y' if fresh_low else 'N'} (stale {stale_ratio_low:.2%})\n"
            f"{'─' * 38}\n"
            f"Shield type:    {self.shield_type.upper()}\n"
            f"Risk level:     {risk.upper()}\n"
            f"Shield active:  {'YES ⚡' if shield_on else 'no'}\n"
            f"Total shields:  {total_shields}\n"
            f"{'─' * 38}\n"
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

    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Nombre del archivo .pth del modelo",
    )
    p.add_argument(
        "--shield_type",
        type=str,
        choices=["none", "basic", "adaptive"],
        default="adaptive",
    )

    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm_port", type=int, default=8000)
    p.add_argument("--map", type=str, default="Town04")
    p.add_argument("--num_npc", type=int, default=20)
    p.add_argument("--weather", type=str, default="ClearNoon")
    p.add_argument("--target_speed_kmh", type=float, default=30.0)
    p.add_argument("--success_distance", type=float, default=250.0)
    p.add_argument(
        "--obs-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activar normalización online de observaciones. (Usa --no-obs-norm para desactivar)",
    )

    p.add_argument("--front_threshold", type=float, default=0.15)
    p.add_argument("--side_threshold", type=float, default=0.04)
    p.add_argument("--lateral_threshold", type=float, default=0.82)

    p.add_argument(
        "--idle_penalty_weight",
        type=float,
        default=0.25,
        help="Pico de la idle_penalty ESCALONADA (sincronizado con training).",
    )
    p.add_argument(
        "--progress_reward_weight",
        type=float,
        default=0.30,
        help="Peso del progress_reward (no afecta en eval: pesos de shaping "
        "se anulan a 0 para reportar el reward base de CarlaEnv).",
    )
    p.add_argument(
        "--acceleration_reward_weight",
        type=float,
        default=0.08,
        help="Peso del acceleration_reward (no afecta en eval; ver --progress_reward_weight).",
    )

    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument(
        "--no_render",
        action="store_true",
        help="Deshabilitar renderizado (solo métricas)",
    )
    p.add_argument(
        "--no_dashboard", action="store_true", help="Deshabilitar dashboard matplotlib"
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Política determinista (sin muestreo)",
    )

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
        # Sincronizado con main_train.py (sesión 5): 30.0.
        out_of_road_penalty=30.0,
        crash_penalty=10.0,
        seed=100,  # Semilla diferente a entrenamiento
    )

    # Wrapper order MUST match main_train.py: CarlaEnv → Shield → RewardShaper.
    # The shaper reads shield_activated / executed_action / proposed_action
    # from info to compute shield_pen and suppress smoothness on intervention.
    # Wrapping the shaper before the shield would leave those keys missing.
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

    # Zero out shaping weights so eval reports the pure base reward from
    # CarlaEnv. The shaper still sits in the chain to consume shield info
    # keys consistently with training. Todos los pesos añadidos en sesiones
    # 3-5 (progress_reward_weight, acceleration_reward_weight) también se
    # anulan a 0 — la intencionalidad es reportar el `raw_reward` base.
    env = CarlaRewardShaper(
        env,
        target_speed_kmh=args.target_speed_kmh,
        speed_weight=0.0,
        smoothness_weight=0.0,
        lane_centering_weight=0.0,
        lane_invasion_penalty=0.0,
        off_road_penalty=0.0,
        idle_penalty_weight=0.0,
        progress_reward_weight=0.0,
        acceleration_reward_weight=0.0,
    )

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

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ── Agente ─────────────────────────────────────────────────────────
    agent = PPOAgent(state_dim, action_dim, normalize_obs=args.obs_norm)
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
            fallback_target_kmh=args.target_speed_kmh,
            lateral_threshold=args.lateral_threshold,
        )

    # ── Variables de evaluación ────────────────────────────────────────
    all_episodes: List[List[Dict]] = []
    all_infos: List[Dict] = []

    total_rewards = []
    successes = 0
    crashes = 0
    timeouts = 0
    total_shields = 0

    header = (
        f"{'Episode':<9} {'Reward':>8} {'Status':<22} {'Dist(m)':>8} {'Shields':>8}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_infos: List[Dict] = []
            done = False
            truncated = False
            step = 0

            while not (done or truncated) and step < args.max_steps:
                if args.deterministic:
                    action, _, _, _ = agent.select_action(obs, deterministic=True)
                else:
                    action, _, _, _ = agent.select_action(obs)

                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                step += 1
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

            dist = info.get("total_distance", 0.0)
            ep_shields = sum(
                1
                for i in ep_infos
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
        logger.info(f"\nSuccess rate:   {successes / n:.1%}  ({successes}/{n})")
        logger.info(f"Crash rate:     {crashes / n:.1%}  ({crashes}/{n})")
        logger.info(f"Timeout rate:   {timeouts / n:.1%}  ({timeouts}/{n})")

        if args.shield_type != "none":
            logger.info(f"\nTotal shield interventions: {total_shields}")
            logger.info(f"Avg per episode:            {total_shields / n:.1f}")

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
