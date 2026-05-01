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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.Adaptative_Shield.adaptive_horizon_shield import CarlaAdaptiveHorizonShield
from src.CARLA.Env.carla_env import CarlaEnv
from src.Metrics.EvalMetrics.metrics import SafetyMetricsReporter
from src.PPO.ppo_agent import PPOAgent
from src.reward_shaper import CarlaRewardShaper
from src.safety_shield import CarlaSafetyShield


# Mapa de categorías semánticas para el BEV point map.
# Cada entrada: (label, set de tags, color hex, marker, tamaño)
#
# IMPORTANTE — tabla CARLA 0.9.14+ (alineada con CityScapes). La numeración
# antigua (0.9.10-0.9.13) era totalmente diferente y mezclaba categorías:
# tag 4 era Pedestrian (ahora Wall), tag 10 era Vehicles (ahora Terrain),
# etc. Si el dashboard está conectado a un servidor 0.9.14+ pero usa la
# tabla vieja, las paredes salen como peatones y los coches no aparecen
# como dinámicos. Verificado en evaluación real.
# Ref: https://carla.readthedocs.io/en/latest/ref_sensors/
#
# Las categorías replican las del SemanticLidarProcessor:
#   VEHICLE_TAGS    = {14 Car, 15 Truck, 16 Bus, 17 Train, 18 Motorcycle, 19 Bicycle}
#   PEDESTRIAN_TAGS = {12 Pedestrian, 13 Rider}
#   STATIC_OBS_TAGS = {3, 4, 5, 6, 7, 8, 9, 20, 26, 28}
#   ROAD_EDGE_TAGS  = {2 SideWalks, 10 Terrain, 25 Ground, 27 RailTrack}
# Cualquier tag fuera de estos grupos cae en "Other" — útil para detectar
# tags que el procesador está ignorando (p. ej. 22 Other, 23 Water).
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

# Capa de carretera para fondo del BEV: Roads (1) y RoadLine (24).
# Se pintan en color claro y NUNCA cuentan como obstáculo. Provienen del
# canal `lidar_road_points_*` que el procesador captura ANTES del filtro
# de altura.
BEV_ROAD_GROUPS = [
    ("Road", frozenset({1}), "#3a3a3a", ".", 3),
    ("RoadLine", frozenset({24}), "#ffff66", ".", 5),
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

        # Anillos de distancia (10, 25, 50 m) — referencia visual.
        for r_m in (10.0, 25.0, 50.0):
            ring = mpatches.Circle(
                (0, 0),
                r_m,
                fill=False,
                edgecolor="gray",
                linewidth=0.6,
                linestyle=":",
                alpha=0.6,
            )
            self.ax_lidar.add_patch(ring)
            self.ax_lidar.text(
                0, r_m + 0.5, f"{int(r_m)} m",
                fontsize=6, color="gray", ha="center", alpha=0.7,
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
            ego_wid,                              # cuadre con el sensor alto
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
            [0], [3.0], marker="^", color="white",
            markersize=8, zorder=6, markeredgecolor="black",
        )

        # Capa de fondo: carretera + marcas (Roads, RoadLine). Se dibuja
        # ANTES de las categorías de obstáculos para quedar como fondo.
        # Los puntos vienen del canal road_points_* del procesador, que
        # los captura PRE-filtro-altura porque están a nivel del suelo.
        self._road_scatters: Dict[str, plt.Artist] = {}
        for label, _tags, color, marker, size in BEV_ROAD_GROUPS:
            sc = self.ax_lidar.scatter(
                [], [],
                s=size, c=color, marker=marker,
                label=label, alpha=0.5, edgecolors="none", zorder=2,
            )
            self._road_scatters[label] = sc

        # Scatter por categoría semántica del LIDAR alto.
        self._lidar_scatters: Dict[str, plt.Artist] = {}
        for label, _tags, color, marker, size in BEV_GROUPS:
            sc = self.ax_lidar.scatter(
                [], [],
                s=size, c=color, marker=marker,
                label=label, alpha=0.9, edgecolors="none", zorder=4,
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
            [-rng + 4], [rng - 4],
            s=110, c="green", marker="o", edgecolors="black",
            linewidths=1.0, zorder=10,
        )
        self.ax_lidar.text(
            -rng + 8, rng - 4, "fresh",
            fontsize=7, color="black", va="center",
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
                    coords = np.column_stack(
                        (screen_x[mask], screen_y[mask])
                    )
                else:
                    coords = np.empty((0, 2), dtype=np.float32)
                self._lidar_scatters[label].set_offsets(coords)
        else:
            # Sin puntos crudos disponibles — vaciar todos los scatters
            # para no dejar el último frame fantasma en pantalla.
            empty = np.empty((0, 2), dtype=np.float32)
            for sc in self._lidar_scatters.values():
                sc.set_offsets(empty)

        # ── Capa de carretera (Roads + RoadLine) ──────────────────────
        # Estos puntos no son obstáculo: el procesador los captura ANTES
        # del filtro de altura específicamente para visualización. Sirven
        # como fondo del BEV — al ver las marcas blancas el usuario puede
        # auditar la posición lateral del coche en su carril.
        road_x = info.get("lidar_road_points_x")
        road_y = info.get("lidar_road_points_y")
        road_tag = info.get("lidar_road_points_tag")
        if road_x is not None and road_y is not None and road_tag is not None:
            r_screen_x = np.asarray(road_y, dtype=np.float32)
            r_screen_y = np.asarray(road_x, dtype=np.float32)
            r_tag = np.asarray(road_tag, dtype=np.uint32)
            for label, tags, _color, _marker, _size in BEV_ROAD_GROUPS:
                mask = np.isin(r_tag, list(tags))
                if np.any(mask):
                    coords = np.column_stack(
                        (r_screen_x[mask], r_screen_y[mask])
                    )
                else:
                    coords = np.empty((0, 2), dtype=np.float32)
                self._road_scatters[label].set_offsets(coords)
        else:
            empty = np.empty((0, 2), dtype=np.float32)
            for sc in self._road_scatters.values():
                sc.set_offsets(empty)

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
