"""
main_eval.py - Entrypoint de evaluación para CARLA Safe RL

Por defecto la evaluación es HEADLESS (sin render ni dashboard) y
DETERMINISTA (acción media, = política desplegada). La normalización de
observaciones se CONGELA. Cada episodio usa `seed + ep`, idéntico entre
shields, para comparaciones reproducibles.

USO:
    # Eval rápida con shield adaptativo (headless, determinista):
    python main_eval.py --model_name mi_modelo_adaptive_final.pth --episodes 50

    # Ablación de dependencia del shield en UNA orden (mismos escenarios):
    python main_eval.py --model_name mi_modelo.pth \
        --shield_type none adaptive --episodes 50 --out ablation.json

    # Interactivo (cámara CARLA + dashboard matplotlib, un solo shield):
    python main_eval.py --model_name mi_modelo.pth --render --dashboard

    # Política estocástica (muestreo) en vez de la determinista por defecto:
    python main_eval.py --model_name mi_modelo.pth --stochastic
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.Adaptative_Shield.adaptive_horizon_shield import CarlaAdaptiveHorizonShield
from src.CARLA.Env.carla_env import CarlaEnv
from src.Metrics.EvalMetrics.metrics import SafetyMetricsReporter
from src.PPO.ppo_agent import PPOAgent
from src.reward_shaper import CarlaRewardShaper
from src.safety_shield import CarlaSafetyShield

BEV_GROUPS = [
    # Vehículos (Car, Truck, Bus, Train, Motorcycle, Bicycle)
    ("Vehicle", frozenset({14, 15, 16, 17, 18, 19}), "#cc0000", "o", 18),
    # Peatones (Pedestrian + Rider sobre vehículo)
    ("Pedestrian", frozenset({12, 13}), "#ff00ff", "X", 36),
    # Obstáculos estáticos altos (Building, Wall, Fence, Pole, TrafficLight,
    # TrafficSign, Vegetation, Static, Bridge, GuardRail)
    (
        "Static",
        frozenset({3, 4, 5, 6, 7, 8, 9, 20, 26, 28}),
        "#5b6f80",
        ".",
        6,
    ),
    # Bordes/transiciones de la calzada (SideWalks, Terrain, Ground, RailTrack)
    ("RoadEdge", frozenset({2, 10, 25, 27}), "#ffa500", "s", 9),
    # Tag 21 = Dynamic (props que se mueven pero no son vehículo/peatón)
    ("Dynamic", frozenset({21}), "#ffd700", "D", 10),
    # Catch-all: tags inesperados (Other, Water, Unlabeled, RoadLine si llega
    # a colarse aquí pese al canal aparte). Si esta categoría tiene puntos
    # consistentemente, hay tags no agrupados que conviene revisar.
    ("Other", None, "#888888", ".", 5),
]

# Marcas de carril obtenidas del Waypoint API (no del LIDAR semántico,
# que en CARLA NO emite tag 24 RoadLine — ver issues #455 y #3638). Los
# arrays vienen pobladas directamente desde info["lane_marking_*"] en
# frame del sensor.
#
# Cada entrada del BEV: (label, info_key_x, info_key_y, color,
#                        marker, size, linestyle).
# Las líneas sólidas (no cruzables) se dibujan en blanco continuo y las
# discontinuas (cambio permitido) en blanco con guiones.
BEV_LANE_MARKINGS = [
    ("Solid (no cross)", "lane_marking_left_solid", "#b8007a", "_", 22, "-"),
    ("Solid (no cross)", "lane_marking_right_solid", "#b8007a", "_", 22, "-"),
    ("Dashed (allowed)", "lane_marking_left_dashed", "#ffd84d", "_", 18, "--"),
    ("Dashed (allowed)", "lane_marking_right_dashed", "#ffd84d", "_", 18, "--"),
]

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

    Panel principal: BEV (bird's eye view) point map del LIDAR semántico.
    En lugar de pintar el scan ya bin-eado en 240 valores, pintamos cada
    punto individual post-filtros con código de color por categoría
    semántica. Esto permite verificar visualmente si el procesador está
    perdiendo, agrupando o filtrando hits incorrectamente — algo
    imposible de auditar con el polar plot bin-eado anterior.

    Paneles secundarios: speed gauge, offset lateral e info text.

    Referencias visuales (todas en metros, frame del sensor):
      - Eje X de pantalla = lateral del coche (positivo = derecha)
      - Eje Y de pantalla = longitudinal (positivo = adelante)
      - Origen (0, 0) = ego vehicle
      - Anillos de distancia a 10/25/50 m
      - Wedge rojo = zona donde el front_threshold del shield activa
      - Conversión: pantalla_x = y_carla, pantalla_y = x_carla
        (UE LH → BEV con frente arriba)
    """

    LIDAR_RANGE_M = 50.0  # alcance del LIDAR alto

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

        # ── BEV point map (sustituye al polar plot anterior) ───────────
        self.ax_lidar = self.fig.add_subplot(gs[:, 0])
        self.num_lidar_rays = num_lidar_rays
        self._front_threshold = float(front_threshold)
        rng = self.LIDAR_RANGE_M
        self.ax_lidar.set_xlim(-rng, rng)
        self.ax_lidar.set_ylim(-rng, rng)
        self.ax_lidar.set_aspect("equal", adjustable="box")
        self.ax_lidar.set_xlabel("Lateral (m)  →  derecha", fontsize=8)
        self.ax_lidar.set_ylabel("Longitudinal (m)  →  adelante", fontsize=8)
        self.ax_lidar.set_title(
            "LIDAR semántico — BEV point map (post-filtros ego + altura)",
            pad=10,
            fontsize=10,
        )
        self.ax_lidar.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        self.ax_lidar.axhline(0, color="gray", linewidth=0.4, alpha=0.6)
        self.ax_lidar.axvline(0, color="gray", linewidth=0.4, alpha=0.6)

        # Anillos de distancia (10, 25, 50 m) — referencia visual muy sutil
        # para no confundirlos con artefactos del LIDAR (los rayos del
        # canal inferior chocan con el suelo a distancias fijas y forman
        # circunferencias). Por eso los pintamos en gris claro y a baja
        # opacidad, con la etiqueta a un lado en vez de arriba.
        for r_m in (10.0, 25.0, 50.0):
            ring = mpatches.Circle(
                (0, 0),
                r_m,
                fill=False,
                edgecolor="#cccccc",
                linewidth=0.4,
                linestyle=(0, (1, 4)),  # punteado más espaciado
                alpha=0.35,
                zorder=1,
            )
            self.ax_lidar.add_patch(ring)
            self.ax_lidar.text(
                r_m + 0.5,
                0.5,
                f"{int(r_m)} m",
                fontsize=6,
                color="#999999",
                ha="left",
                alpha=0.7,
            )

        # Wedge del front_threshold del shield: si front_threshold=0.15 y
        # range=50 m, la zona crítica es un sector frontal (±FRONT_N bins
        # ≈ ±22.5°) hasta 7.5 m. Lo pintamos como cuña roja transparente
        # para que se vea cuándo entra un punto al perímetro de seguridad.
        FRONT_N = 15
        half_angle_deg = (FRONT_N / num_lidar_rays) * 360.0
        threshold_radius = self._front_threshold * rng
        # En BEV "frente arriba", el sector frontal se pinta entre 90°-half
        # y 90°+half (matplotlib usa el convenio matemático: 0° a la
        # derecha, ángulos antihorarios).
        wedge = mpatches.Wedge(
            center=(0.0, 0.0),
            r=threshold_radius,
            theta1=90.0 - half_angle_deg,
            theta2=90.0 + half_angle_deg,
            facecolor="red",
            alpha=0.18,
            edgecolor="red",
            linewidth=1.0,
            linestyle="--",
            label=f"Front threshold ({front_threshold:.2f}={threshold_radius:.1f} m)",
        )
        self.ax_lidar.add_patch(wedge)

        # Ego vehicle: rectángulo aproximado del Tesla Model 3
        # (longitud ≈ 4.7 m, ancho ≈ 1.85 m, centrado en el origen).
        ego_len = 4.7
        ego_wid = 1.85
        ego_rect = mpatches.Rectangle(
            (-ego_wid / 2, -ego_len / 2 + 1.5),  # +1.5 para que el morro
            ego_wid,  # cuadre con el sensor alto
            ego_len,
            facecolor="steelblue",
            edgecolor="white",
            alpha=0.85,
            linewidth=1.0,
            zorder=5,
        )
        self.ax_lidar.add_patch(ego_rect)
        # Triangulito que indica el sentido de marcha
        self.ax_lidar.plot(
            [0],
            [3.0],
            marker="^",
            color="white",
            markersize=8,
            zorder=6,
            markeredgecolor="black",
        )

        # Capa de fondo: marcas de carril (líneas sólidas y discontinuas)
        # obtenidas del Waypoint API en `info["lane_marking_*"]`. NO se
        # usa el LIDAR semántico — CARLA no emite tag 24 (RoadLine)
        # porque las marcas son texturas sobre el mesh del Road, no un
        # mesh aparte (Issues #455 y #3638). El waypoint API en cambio
        # expone la posición exacta desde el OpenDRIVE del mapa.
        # Para evitar duplicar entradas en la leyenda (sólida izq y dcha
        # tienen el mismo label), solo añadimos `label=...` la primera
        # vez que aparece cada label único.
        self._lane_markings: Dict[str, plt.Artist] = {}
        seen_labels = set()
        for label, info_key, color, marker, size, _ls in BEV_LANE_MARKINGS:
            legend_label = label if label not in seen_labels else None
            seen_labels.add(label)
            sc = self.ax_lidar.scatter(
                [],
                [],
                s=size,
                c=color,
                marker=marker,
                label=legend_label,
                alpha=0.85,
                edgecolors="none",
                zorder=2,
            )
            self._lane_markings[info_key] = sc

        # Scatter por categoría semántica del LIDAR alto.
        self._lidar_scatters: Dict[str, plt.Artist] = {}
        for label, _tags, color, marker, size in BEV_GROUPS:
            sc = self.ax_lidar.scatter(
                [],
                [],
                s=size,
                c=color,
                marker=marker,
                label=label,
                alpha=0.9,
                edgecolors="none",
                zorder=4,
            )
            self._lidar_scatters[label] = sc

        # NOTA: el LIDAR bajo (z=0.5 m, range 30 m) se eliminó del sistema
        # tras verificar que era totalmente redundante con el alto (todo
        # lo que veía también lo alcanzaba el sensor de techo). OBS_DIM
        # bajó de 979 → 739; los modelos previos no son compatibles.

        # Indicador de frescura: punto en la esquina superior izquierda.
        # Verde = LIDAR fresco en el tick actual. Rojo = stale.
        # Permite detectar de un vistazo desincronías sensor-mundo
        # durante la evaluación.
        self.fresh_marker = self.ax_lidar.scatter(
            [-rng + 4],
            [rng - 4],
            s=110,
            c="green",
            marker="o",
            edgecolors="black",
            linewidths=1.0,
            zorder=10,
        )
        self.ax_lidar.text(
            -rng + 8,
            rng - 4,
            "fresh",
            fontsize=7,
            color="black",
            va="center",
        )

        self.ax_lidar.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            fontsize=7,
            ncol=4,
            framealpha=0.85,
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

        # ── BEV point map (LIDAR alto) ─────────────────────────────────
        # Leemos los puntos crudos post-filtros desde el info dict. Los
        # puebla SemanticScanResult.to_info_dict() — los mismos puntos
        # que usa el procesador para construir los scans bin-eados, así
        # que cualquier obstáculo que aparezca aquí está siendo "visto"
        # por el agente. Si no aparece, el filtro lo está descartando.
        #
        # Conversión LH → BEV con frente arriba:
        #   pantalla_x = +y_carla  (derecha CARLA → derecha pantalla)
        #   pantalla_y = +x_carla  (adelante CARLA → arriba pantalla)
        pts_x = info.get("lidar_points_x")
        pts_y = info.get("lidar_points_y")
        pts_tag = info.get("lidar_points_tag")
        if pts_x is not None and pts_y is not None and pts_tag is not None:
            screen_x = np.asarray(pts_y, dtype=np.float32)  # lateral
            screen_y = np.asarray(pts_x, dtype=np.float32)  # longitudinal
            tag_arr = np.asarray(pts_tag, dtype=np.uint32)
            # Asignamos cada punto a su grupo y refrescamos el scatter.
            assigned = np.zeros(len(tag_arr), dtype=bool)
            for label, tags, _color, _marker, _size in BEV_GROUPS:
                if tags is None:
                    # Grupo "Other" agarra los que no han sido asignados.
                    mask = ~assigned
                else:
                    mask = np.isin(tag_arr, list(tags))
                    assigned |= mask
                if np.any(mask):
                    coords = np.column_stack((screen_x[mask], screen_y[mask]))
                else:
                    coords = np.empty((0, 2), dtype=np.float32)
                self._lidar_scatters[label].set_offsets(coords)
        else:
            # Sin puntos crudos disponibles — vaciar todos los scatters
            # para no dejar el último frame fantasma en pantalla.
            empty = np.empty((0, 2), dtype=np.float32)
            for sc in self._lidar_scatters.values():
                sc.set_offsets(empty)

        # ── Marcas de carril (waypoint API) ───────────────────────────
        # Las cuatro entradas vienen pobladas en frame del sensor (UE LH:
        # x=adelante, y=derecha) y aquí solo las convertimos al frame del
        # plot (pantalla_x = y_carla, pantalla_y = x_carla).
        empty = np.empty((0, 2), dtype=np.float32)
        for _label, info_key, _c, _m, _s, _ls in BEV_LANE_MARKINGS:
            xs = info.get(f"{info_key}_x")
            ys = info.get(f"{info_key}_y")
            if xs is not None and ys is not None and len(xs) > 0:
                screen_x = np.asarray(ys, dtype=np.float32)
                screen_y = np.asarray(xs, dtype=np.float32)
                coords = np.column_stack((screen_x, screen_y))
            else:
                coords = empty
            self._lane_markings[info_key].set_offsets(coords)

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

        # Frescura del LIDAR. Verde si el frame del sensor cuadró con el
        # world.tick() de este step; rojo en caso contrario. El stale_ratio
        # acumulado se imprime para detectar deriva.
        fresh = bool(info.get("semantic_data_fresh", True))
        stale_ratio = float(info.get("semantic_stale_ratio", 0.0))
        # Marker de frescura en la esquina del BEV. set_color funciona
        # tanto en Line2D como en PathCollection.
        self.fresh_marker.set_color("green" if fresh else "red")

        # Conteos de puntos por categoría post-filtros. Si estos números
        # son 0 o muy bajos en un escenario donde sí hay obstáculos, el
        # procesador los está descartando antes de bin-ear — es el
        # síntoma más claro de un bug en filtros o tabla de tags.
        n_veh = int(info.get("n_vehicle_pts", 0))
        n_ped = int(info.get("n_pedestrian_pts", 0))
        n_stat = int(info.get("n_static_pts", 0))
        n_edge = int(info.get("n_road_edge_pts", 0))
        n_pts = int(info.get("semantic_pts_per_frame", 0))

        text = (
            f"Episode {episode} | Step {step}\n"
            f"{'─' * 42}\n"
            f"Speed:          {speed_kmh:>6.1f} km/h\n"
            f"Lat offset:     {lat_m:>+6.3f} m  (norm {lat_norm:>+5.2f})\n"
            f"Heading error:  {heading_err:>+6.1f}°\n"
            f"On road:        {'YES' if on_road else 'NO ⚠️'}\n"
            f"Min LIDAR:      {min_dist_m:>6.2f} m  (norm {min_dist_norm:.3f})\n"
            f"LIDAR fresh:    {'Y' if fresh else 'N'}  "
            f"(stale {stale_ratio:.2%})\n"
            f"Pts/frame:      {n_pts:>4d}\n"
            f"By tag:         veh={n_veh:>3d} ped={n_ped:>3d} "
            f"stat={n_stat:>3d} edge={n_edge:>3d}\n"
            f"{'─' * 42}\n"
            f"Shield type:    {self.shield_type.upper()}\n"
            f"Risk level:     {risk.upper()}\n"
            f"Shield active:  {'YES ⚡' if shield_on else 'no'}\n"
            f"Total shields:  {total_shields}\n"
            f"{'─' * 42}\n"
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
    p = argparse.ArgumentParser(
        description="Evaluación del agente PPO en CARLA. Por defecto corre "
        "HEADLESS y DETERMINISTA (alineado con el despliegue real). Acepta "
        "varios --shield_type para una ablación controlada en una sola orden."
    )

    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Nombre del archivo .pth del modelo",
    )
    p.add_argument(
        "--shield_type",
        type=str,
        nargs="+",
        choices=["none", "basic", "adaptive"],
        default=["adaptive"],
        help="Uno o varios shields. Con varios (p.ej. '--shield_type none "
        "adaptive') corre cada uno con LOS MISMOS escenarios por episodio "
        "(misma semilla) y los compara — ablación de dependencia del shield.",
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
        help="Activar normalización de observaciones. (Usa --no-obs-norm para desactivar)",
    )

    p.add_argument("--front_threshold", type=float, default=0.15)
    p.add_argument("--side_threshold", type=float, default=0.04)
    p.add_argument("--lateral_threshold", type=float, default=0.82)

    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Semilla base de evaluacion. Cada episodio usa seed+ep, IDENTICO "
        "entre shields, para que la comparacion vea exactamente los mismos "
        "escenarios (spawn + NPCs). 100 != semilla de training (42).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Ruta opcional .json donde guardar el resumen de resultados.",
    )

    # ── Visualización: OFF por defecto (es el cuello de botella) ────────
    p.add_argument(
        "--render",
        action="store_true",
        help="Mostrar la cámara espectadora de CARLA (lento). Por defecto off.",
    )
    p.add_argument(
        "--dashboard",
        action="store_true",
        help="Mostrar el dashboard matplotlib (redibuja por paso, lento). Off por defecto.",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Muestrear la política. Por defecto la eval es DETERMINISTA "
        "(acción media), que es la política realmente desplegada.",
    )

    # ── Flags legados (no-op, se mantienen por compatibilidad) ──────────
    # Headless y determinista ya son el comportamiento por defecto, así que
    # --no_render / --no_dashboard / --deterministic no hacen nada; se
    # aceptan para que las órdenes/documentación antiguas no fallen.
    p.add_argument("--no_render", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no_dashboard", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--deterministic", action="store_true", help=argparse.SUPPRESS)
    # Pesos de shaping: en eval todos se anulan a 0 (se reporta el reward base
    # de CarlaEnv), así que estos no tienen efecto. Se aceptan por compat.
    p.add_argument(
        "--idle_penalty_weight", type=float, default=0.25, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--progress_reward_weight", type=float, default=0.30, help=argparse.SUPPRESS
    )
    p.add_argument(
        "--acceleration_reward_weight", type=float, default=0.08, help=argparse.SUPPRESS
    )

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL ENTORNO (idéntica lógica a main_train.py)
# ══════════════════════════════════════════════════════════════════════


def build_env(args, shield_type: str, render: bool = False):
    """Construye la cadena de wrappers para evaluación con un shield dado."""
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
        out_of_road_penalty=10.0,
        crash_penalty=10.0,
        seed=args.seed,  # Semilla diferente a entrenamiento (42)
    )

    # Wrapper order MUST match main_train.py: CarlaEnv → Shield → RewardShaper.
    # The shaper reads shield_activated / executed_action / proposed_action
    # from info to compute shield_pen and suppress smoothness on intervention.
    # Wrapping the shaper before the shield would leave those keys missing.
    if shield_type == "basic":
        logger.info("🛡️  Shield: CarlaSafetyShield")
        env = CarlaSafetyShield(
            env,
            num_lidar_rays=num_lidar_rays,
            front_threshold=args.front_threshold,
            side_threshold=args.side_threshold,
            lateral_threshold=args.lateral_threshold,
        )
    elif shield_type == "adaptive":
        logger.info("🛡️  Shield: CarlaAdaptiveHorizonShield")
        env = CarlaAdaptiveHorizonShield(
            env,
            num_lidar_rays=num_lidar_rays,
            front_threshold_base=args.front_threshold,
            side_threshold_base=args.side_threshold,
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

# Taxonomía de outcomes ALINEADA con el training (`Outcome/Type`):
# success / crash / offroad / stuck / timeout como categorías DISTINTAS.
# (La versión anterior contaba offroad como crash y stuck como timeout.)
_OUTCOME_LABELS = {
    "success": "success ✅",
    "crash": "crash 💥",
    "offroad": "off-road ⚠️",
    "stuck": "stuck 🐢",
    "timeout": "timeout ⏱",
}


def _classify_outcome(info: Dict) -> str:
    """Clasifica el outcome terminal de un episodio igual que CarlaEnv."""
    if info.get("collision", False) or info.get("crash_vehicle", False):
        return "crash"
    if info.get("out_of_road", False):
        return "offroad"
    if info.get("arrive_dest", False):
        return "success"
    if info.get("stuck", False):
        return "stuck"
    return "timeout"


def _run_shield_eval(
    env,
    agent,
    args,
    shield_type: str,
    deterministic: bool,
    render: bool,
    dashboard: Optional["CarlaDashboard"],
) -> Dict:
    """Corre `args.episodes` episodios para un shield y devuelve el resumen.

    Cada episodio se reinicia con `seed = args.seed + ep`, idéntico entre
    shields, de modo que la comparación ve exactamente el mismo escenario.
    """
    counts = {k: 0 for k in _OUTCOME_LABELS}
    total_rewards: List[float] = []
    distances: List[float] = []
    total_shields = 0
    all_episodes: List[List[Dict]] = []
    all_infos: List[Dict] = []

    logger.info("\n" + "─" * 70)
    logger.info(f"Shield = {shield_type.upper()}  |  {args.episodes} episodes")
    logger.info("─" * 70)
    header = (
        f"{'Episode':<9} {'Reward':>8} {'Status':<14} {'Dist(m)':>8} {'Shields':>8}"
    )
    logger.info(header)

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        ep_infos: List[Dict] = []
        done = truncated = False
        step = 0
        info: Dict = {}

        while not (done or truncated) and step < args.max_steps:
            action, _, _, _ = agent.select_action(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            ep_infos.append(info)

            if render:
                env.render()
            if info.get("shield_activated", info.get("shield_active", False)):
                total_shields += 1
            if dashboard is not None:
                dashboard.update(obs, action, info, ep, step, total_shields)

        outcome = _classify_outcome(info)
        counts[outcome] += 1
        dist = info.get("total_distance", 0.0)
        distances.append(dist)
        ep_shields = sum(
            1
            for i in ep_infos
            if i.get("shield_activated", i.get("shield_active", False))
        )
        logger.info(
            f"Ep {ep:<6} {ep_reward:>8.2f}  {_OUTCOME_LABELS[outcome]:<14} "
            f"{dist:>8.1f}  {ep_shields:>8}"
        )

        total_rewards.append(ep_reward)
        all_episodes.append(ep_infos)
        all_infos.extend(ep_infos)

    n = max(len(total_rewards), 1)
    return {
        "shield_type": shield_type,
        "episodes": len(total_rewards),
        "success_rate": counts["success"] / n,
        "crash_rate": counts["crash"] / n,
        "offroad_rate": counts["offroad"] / n,
        "stuck_rate": counts["stuck"] / n,
        "timeout_rate": counts["timeout"] / n,
        "counts": counts,
        "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "std_reward": float(np.std(total_rewards)) if total_rewards else 0.0,
        "avg_distance": float(np.mean(distances)) if distances else 0.0,
        "total_shields": total_shields,
        "shields_per_ep": total_shields / n,
        "_all_infos": all_infos,
        "_all_episodes": all_episodes,
    }


def _print_comparison(results: List[Dict]):
    """Tabla de comparación entre shields (modo ablación)."""
    logger.info("\n" + "=" * 88)
    logger.info("SHIELD ABLATION — same scenarios per episode (seed-aligned)")
    logger.info("=" * 88)
    hdr = (
        f"{'shield':<10}{'success':>9}{'crash':>8}{'offroad':>9}{'stuck':>8}"
        f"{'timeout':>9}{'reward':>10}{'shlds/ep':>10}{'dist(m)':>9}"
    )
    logger.info(hdr)
    logger.info("-" * len(hdr))
    for r in results:
        logger.info(
            f"{r['shield_type']:<10}"
            f"{r['success_rate']:>8.1%} "
            f"{r['crash_rate']:>7.1%} "
            f"{r['offroad_rate']:>8.1%} "
            f"{r['stuck_rate']:>7.1%} "
            f"{r['timeout_rate']:>8.1%} "
            f"{r['avg_reward']:>9.1f} "
            f"{r['shields_per_ep']:>9.1f} "
            f"{r['avg_distance']:>8.1f}"
        )
    logger.info("=" * 88)
    # Interpretación directa de dependencia: el salto de crash+offroad entre
    # 'none' y un shield = cuánta seguridad aporta el shield (= dependencia).
    by_type = {r["shield_type"]: r for r in results}
    if "none" in by_type and len(results) > 1:
        none = by_type["none"]
        none_unsafe = none["crash_rate"] + none["offroad_rate"]
        for r in results:
            if r["shield_type"] == "none":
                continue
            shielded_unsafe = r["crash_rate"] + r["offroad_rate"]
            logger.info(
                f"Dependencia ({r['shield_type']} vs none): "
                f"crash+offroad {none_unsafe:.1%} → {shielded_unsafe:.1%} "
                f"(el shield evita {none_unsafe - shielded_unsafe:+.1%} de catástrofe)"
            )
        logger.info("=" * 88)


def evaluate():
    args = get_args()

    shield_types: List[str] = list(dict.fromkeys(args.shield_type))  # dedup, keep order
    multi = len(shield_types) > 1
    deterministic = not args.stochastic
    # Visualización solo tiene sentido con un único shield; en ablación se
    # fuerza headless (es una operación por lotes y matplotlib es el cuello
    # de botella principal).
    render = bool(args.render) and not multi
    show_dashboard = bool(args.dashboard) and not multi
    if multi and (args.render or args.dashboard):
        logger.info("Modo ablación (varios shields): render/dashboard desactivados.")

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATING TRAINED AGENT — CARLA")
    logger.info("=" * 70)
    logger.info(f"Model:    {args.model_name}")
    logger.info(f"Shields:  {', '.join(shield_types)}")
    logger.info(f"Map:      {args.map}  |  NPCs: {args.num_npc}")
    logger.info(f"Episodes: {args.episodes} (seed base {args.seed})")
    logger.info(f"Policy:   {'deterministic' if deterministic else 'stochastic'}")
    logger.info("=" * 70)

    # ── Localizar modelo ───────────────────────────────────────────────
    model_path = Path("./data/models") / args.model_name
    if not model_path.exists():
        model_path = Path(args.model_name)
    if not model_path.exists():
        logger.error(f"Model not found: {args.model_name}")
        return
    logger.info(f"Loading model from: {model_path}")

    agent: Optional[PPOAgent] = None
    results: List[Dict] = []

    try:
        for shield_type in shield_types:
            env, num_lidar_rays = build_env(args, shield_type, render=render)
            dashboard = None
            try:
                if agent is None:
                    agent = PPOAgent(
                        env.observation_space.shape[0],
                        env.action_space.shape[0],
                        normalize_obs=args.obs_norm,
                    )
                    agent.load(str(model_path))
                    agent.policy.eval()
                    # CONGELAR la normalización de observaciones en eval: el
                    # `select_action` normal actualiza el RunningMeanStd cada
                    # paso, lo que haría derivar la normalización respecto a
                    # las stats con las que se entrenó. En despliegue las stats
                    # están fijas, así que las congelamos para una eval fiel.
                    agent._update_obs_stats = lambda *_a, **_k: None

                if show_dashboard:
                    dashboard = CarlaDashboard(
                        num_lidar_rays=num_lidar_rays,
                        front_threshold=args.front_threshold,
                        shield_type=shield_type,
                        fallback_target_kmh=args.target_speed_kmh,
                        lateral_threshold=args.lateral_threshold,
                    )

                res = _run_shield_eval(
                    env, agent, args, shield_type, deterministic, render, dashboard
                )
                results.append(res)
                _print_shield_summary(res, args, detailed=not multi)
            except KeyboardInterrupt:
                logger.info("\nEvaluation interrupted by user.")
                break
            finally:
                env.close()
                if dashboard is not None:
                    dashboard.close()
    finally:
        if multi and len(results) > 1:
            _print_comparison(results)
        if args.out and results:
            _save_results(args, results)


def _print_shield_summary(res: Dict, args, detailed: bool):
    """Resumen por shield (+ reporte de seguridad detallado si detailed)."""
    n = res["episodes"]
    if n == 0:
        return
    c = res["counts"]
    logger.info("\n" + "=" * 70)
    logger.info(f"SUMMARY — shield={res['shield_type']}  ({n} episodes)")
    logger.info("=" * 70)
    logger.info(f"Avg reward:   {res['avg_reward']:.2f} ± {res['std_reward']:.2f}")
    logger.info(f"Avg distance: {res['avg_distance']:.1f} m")
    logger.info(f"Success:      {res['success_rate']:.1%}  ({c['success']}/{n})")
    logger.info(f"Crash:        {res['crash_rate']:.1%}  ({c['crash']}/{n})")
    logger.info(f"Off-road:     {res['offroad_rate']:.1%}  ({c['offroad']}/{n})")
    logger.info(f"Stuck:        {res['stuck_rate']:.1%}  ({c['stuck']}/{n})")
    logger.info(f"Timeout:      {res['timeout_rate']:.1%}  ({c['timeout']}/{n})")
    if res["shield_type"] != "none":
        logger.info(
            f"Shield interventions: {res['total_shields']} "
            f"({res['shields_per_ep']:.1f}/ep)"
        )

    if detailed and res["_all_infos"]:
        report = SafetyMetricsReporter.generate_report(
            all_infos=res["_all_infos"],
            all_episodes=res["_all_episodes"],
            shield_type=res["shield_type"],
        )
        logger.info(report)


def _save_results(args, results: List[Dict]):
    """Guarda un resumen serializable (sin los infos crudos) en JSON."""
    payload = {
        "model": args.model_name,
        "map": args.map,
        "num_npc": args.num_npc,
        "episodes": args.episodes,
        "seed": args.seed,
        "success_distance": args.success_distance,
        "deterministic": not args.stochastic,
        "results": [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in results
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    evaluate()
