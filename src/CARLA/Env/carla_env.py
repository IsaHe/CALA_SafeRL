"""
carla_env.py - CARLA Gymnasium Environment for Safe RL

LAYOUT DE OBSERVACIÓN (739 dimensiones):
  obs[0:240]    → LIDAR ALTO combinado (techo z=1.0, range 50 m, 3 ch).
                  Cobertura larga distancia: vehículos, muros, obstáculos altos.
  obs[240:480]  → LIDAR ALTO dinámico (vehículos + peatones).
  obs[480:720]  → LIDAR ALTO estático (muros, quitamiedos, postes).
  obs[720:728]  → Lane features extendidas (8 dims):
                    [0] lateral_offset_norm    — posición en carril [-1,1]
                    [1] heading_error_norm      — alineación de heading [-1,1]
                    [2] on_edge_warning         — proximidad al borde [0,1]
                    [3] lane_width_norm         — anchura de carril normalizada [0,1]
                    [4] dist_left_edge_norm     — distancia al borde izquierdo [0,1]
                    [5] dist_right_edge_norm    — distancia al borde derecho [0,1]
                    [6] lane_change_left        — cambio de carril permitido a izq. {0,1}
                    [7] road_curvature_norm     — curvatura próxima normalizada [-1,1]
  obs[728:732]  → Lane marking type (4 dims, binarias):
                    [0] solid_left   — borde izquierdo es línea sólida
                    [1] solid_right  — borde derecho es línea sólida
                    [2] dashed_left  — borde izquierdo es línea discontinua
                    [3] dashed_right — borde derecho es línea discontinua
                  Permite al agente distinguir cruces ilegales (sólido) de
                  cambios de carril permitidos (discontinuo).
  obs[732:734]  → Vehicle state (speed_norm, steering)
  obs[734:739]  → Route info extendida (5 dims):
                    [0] next_wp_angle_norm       — ángulo al wp a 5 m
                    [1] wp_angle_20m_norm         — ángulo al wp a 20 m
                    [2] progress_norm             — progreso del episodio
                    [3] speed_limit_norm          — límite de velocidad normalizado
                    [4] speed_ratio               — speed/limit (>1 si excede límite)


ACCIÓN (2 dimensiones continuas):
  action[0] → steering       [-1.0, 1.0]
  action[1] → throttle_brake [-1.0, 1.0]  (>0=gas, <0=freno)
"""

import gymnasium as gym
import numpy as np
import carla
import random
import time
import math
import logging
import cv2
from collections import deque
from typing import Optional, Tuple, Dict, List, Sequence

from src.CARLA.Sensors.carla_sensors import SensorManager

logger = logging.getLogger(__name__)


class CarlaEnv(gym.Env):
    """
    Entorno CARLA Gymnasium para Safe RL en conducción autónoma.

    Parámetros clave:
        host / port         : Dirección del servidor CARLA (default localhost:2000)
        map_name            : Town04 (autopista), Town01/02/03 (ciudad), Town05 (cruce grande)
        num_npc_vehicles    : NPCs gestionados por TrafficManager
        synchronous         : True para reproducibilidad (necesario para RL)
        fixed_delta_seconds : Paso de simulación (0.05s = 20 Hz)
        num_lidar_rays      : Rayos LIDAR horizontales (compatible con shields)
        success_distance    : Metros a recorrer para considerar éxito
        target_speed_kmh    : Velocidad objetivo para reward de velocidad
    """

    metadata = {"render_modes": ["human"]}

    # Constantes de observación
    LIDAR_DIM = 240  # combined scan
    DYNAMIC_DIM = 240  # dynamic-only scan
    STATIC_DIM = 240  # static-only scan
    LANE_DIM = 8
    LANE_MARKING_DIM = 4  # solid_left, solid_right, dashed_left, dashed_right
    VEHICLE_DIM = 2
    ROUTE_DIM = 5
    OBS_DIM = (
        LIDAR_DIM
        + DYNAMIC_DIM
        + STATIC_DIM
        + LANE_DIM
        + LANE_MARKING_DIM
        + VEHICLE_DIM
        + ROUTE_DIM
    )  # 739

    MAX_SPEED_LIMIT_KMH: float = 130.0

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        tm_port: int = 8000,
        timeout: float = 20.0,
        map_name: str = "Town04",
        num_npc_vehicles: int = 20,
        weather: str = "ClearNoon",
        render_mode: Optional[str] = None,
        synchronous: bool = True,
        fixed_delta_seconds: float = 0.05,
        num_lidar_rays: int = 240,
        lidar_range: float = 50.0,
        lidar_height_filter: float = 0.5,
        max_episode_steps: int = 1000,
        target_speed_kmh: float = 30.0,
        success_distance: float = 250.0,
        success_reward: float = 30.0,
        out_of_road_penalty: float = 30.0,
        stuck_window_size: int = 200,
        stuck_threshold_fraction: float = 0.90,
        stuck_speed_kmh: float = 1.0,
        crash_penalty: float = 10.0,
        seed: int = 42,
        spawn_point_idx: Optional[int] = None,
        spawn_point_indices: Optional[Sequence[int]] = None,
        route_npc_count: int = 0,
    ):
        super().__init__()

        # Configuración
        self.host = host
        self.port = port
        self.tm_port = tm_port
        self.timeout = timeout
        self.map_name = map_name
        self.num_npc_vehicles = num_npc_vehicles
        # Trafico de EXPOSICION inyectado en la ruta del ego (curriculum de
        # colision). 0 = comportamiento EXACTO previo. Mutable entre episodios:
        # el TunableRouteExposure / main_train lo reescriben y el siguiente
        # reset() lo consume. ``last_route_npcs_spawned`` es la verdad de campo
        # (requested != spawned siempre distinguible).
        self.route_npc_count = int(route_npc_count)
        self.last_route_npcs_spawned = 0
        self.weather = weather
        self.render_mode = render_mode
        self.synchronous = synchronous
        self.fixed_delta_seconds = fixed_delta_seconds
        self.num_lidar_rays = num_lidar_rays
        self.lidar_range = lidar_range
        self.lidar_height_filter = lidar_height_filter
        self.max_episode_steps = max_episode_steps
        self.target_speed_kmh = target_speed_kmh
        self.success_distance = success_distance
        self.success_reward = success_reward
        self.out_of_road_penalty = out_of_road_penalty
        self.crash_penalty = crash_penalty
        self.stuck_window_size = stuck_window_size
        self.stuck_threshold_fraction = stuck_threshold_fraction
        self.stuck_speed_kmh = stuck_speed_kmh
        self.base_seed = seed
        self.spawn_point_idx = spawn_point_idx
        self.spawn_point_indices: Optional[List[int]] = (
            list(spawn_point_indices) if spawn_point_indices else None
        )

        obs_low = np.concatenate(
            [
                np.zeros(
                    self.LIDAR_DIM + self.DYNAMIC_DIM + self.STATIC_DIM,
                    dtype=np.float32,
                ),
                np.full(self.LANE_DIM, -1.0, dtype=np.float32),
                np.zeros(self.LANE_MARKING_DIM, dtype=np.float32),
                np.full(self.VEHICLE_DIM + self.ROUTE_DIM, -1.0, dtype=np.float32),
            ]
        )
        obs_high = np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.map: Optional[carla.Map] = None
        self.ego_vehicle: Optional[carla.Vehicle] = None
        # Transform de spawn del ego (pose EXACTA, conocida sin necesidad de tick).
        # En modo síncrono get_transform() es estable solo tras world.tick(), y el
        # tráfico en ruta se coloca ANTES del primer tick del reset, así que usa
        # esta pose en vez de leer una transform potencialmente en el origen.
        self._ego_spawn_transform: Optional[carla.Transform] = None
        self.sensor_manager: Optional[SensorManager] = None
        self.npc_vehicles = []
        self._tm: Optional[carla.TrafficManager] = None

        # Estado episodio
        self.step_count = 0
        self.total_distance = 0.0
        self._last_location: Optional[carla.Location] = None
        self._last_tick_frame: Optional[int] = None
        self._low_speed_window: deque = deque(maxlen=self.stuck_window_size)
        self.episode_collisions = 0
        self.episode_lane_invasions = 0
        self.last_obs: Optional[np.ndarray] = None
        self.last_info: Dict = {}

        # Variables para renderizado
        self.camera_sensor: Optional[carla.Sensor] = None
        self.current_image: Optional[np.ndarray] = None
        self._cv2_window_created = False

        #  Conectar
        self._connect()

    def _connect(self):
        """Conecta con el servidor CARLA y carga el mapa."""
        logger.info(f"Connecting to CARLA at {self.host}:{self.port} ...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)

        self.world = self.client.load_world(self.map_name)
        self.map = self.world.get_map()
        logger.info(f"Loaded map: {self.map_name}")

        if self.synchronous:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.fixed_delta_seconds
            settings.no_rendering_mode = self.render_mode is None
            self.world.apply_settings(settings)

        weather_attr = getattr(
            carla.WeatherParameters, self.weather, carla.WeatherParameters.ClearNoon
        )
        self.world.set_weather(weather_attr)

        self._tm = self.client.get_trafficmanager(self.tm_port)
        self._tm.set_synchronous_mode(self.synchronous)
        self._tm.set_global_distance_to_leading_vehicle(2.5)
        self._tm.set_random_device_seed(self.base_seed)

    # GYMNASIUM API

    def reset(self, *, seed=None, options=None):
        """Reinicia el entorno para un nuevo episodio."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._cleanup()
        self._spawn_ego_vehicle()
        self._spawn_npc_vehicles()
        self._spawn_route_traffic()

        self.sensor_manager = SensorManager(
            self.world,
            self.ego_vehicle,
            num_lidar_rays=self.num_lidar_rays,
            lidar_range=self.lidar_range,
            height_filter=self.lidar_height_filter,
        )
        self.sensor_manager.update_ego_id(self.ego_vehicle.id)

        if self.render_mode == "human":
            bp_lib = self.world.get_blueprint_library()
            camera_bp = bp_lib.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", "640")
            camera_bp.set_attribute("image_size_y", "480")
            camera_bp.set_attribute("fov", "90")

            camera_transform = carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-8.0)
            )

            self.camera_sensor = self.world.spawn_actor(
                camera_bp,
                camera_transform,
                attach_to=self.ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
            self.camera_sensor.listen(self._parse_image)

        self.step_count = 0
        self.total_distance = 0.0
        self._low_speed_window.clear()
        self.episode_collisions = 0
        self.episode_lane_invasions = 0
        self._current_speed_limit = self.target_speed_kmh

        loc = self.ego_vehicle.get_location()
        self._last_location = carla.Location(loc.x, loc.y, loc.z)

        if self.synchronous:
            last_frame = None
            for _ in range(3):
                last_frame = self.world.tick()
            self._last_tick_frame = int(last_frame) if last_frame is not None else None
        else:
            time.sleep(0.15)
            self._last_tick_frame = None

        obs, info = self._build_observation()
        self.last_obs = obs
        self.last_info = info
        return obs, info

    def step(self, action: np.ndarray):
        """Ejecuta un paso de simulación."""
        control = self._action_to_control(action)
        self.ego_vehicle.apply_control(control)

        if self.synchronous:
            tick_frame = self.world.tick()
            self._last_tick_frame = int(tick_frame) if tick_frame is not None else None

        self.step_count += 1

        current_loc = self.ego_vehicle.get_location()
        if self._last_location is not None:
            step_dist = current_loc.distance(self._last_location)
            if step_dist < 5.0:
                self.total_distance += step_dist
        self._last_location = carla.Location(
            current_loc.x, current_loc.y, current_loc.z
        )

        obs, info = self._build_observation()
        self.last_obs = obs
        self.last_info = info

        reward = self._compute_base_reward(action, info)

        done, truncated = self._check_termination(info)

        info.update(
            {
                "step": self.step_count,
                "total_distance": self.total_distance,
                "episode_collisions": self.episode_collisions,
                "episode_lane_invasions": self.episode_lane_invasions,
            }
        )

        return obs, reward, done, truncated, info

    def close(self):
        """Limpia recursos y restaura modo asíncrono."""
        self._cleanup()
        if self.synchronous and self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                settings.no_rendering_mode = False
                self.world.apply_settings(settings)
            except Exception:
                pass

    def render(self):
        """Muestra la vista en 3ra persona en una ventana de OpenCV."""
        if self.render_mode == "human" and self.current_image is not None:
            if not self._cv2_window_created:
                cv2.namedWindow("CARLA Ego View", cv2.WINDOW_AUTOSIZE)
                self._cv2_window_created = True

            cv2.imshow("CARLA Ego View", self.current_image)
            cv2.waitKey(1)

    def _build_observation(self) -> Tuple[np.ndarray, Dict]:
        """
        Construye el vector de observación completo desde sensores y API CARLA.

        Retorna obs (739,) e info enriquecido con datos CARLA para los shields.
        """
        expected_frame = self._last_tick_frame if self.synchronous else None
        sem = self.sensor_manager.get_semantic_result(expected_frame=expected_frame)
        sem_status = self.sensor_manager.get_semantic_status()
        lidar_combined = sem.combined
        lidar_dynamic = sem.dynamic
        lidar_static = sem.static

        raw_limit = self.ego_vehicle.get_speed_limit()
        if raw_limit > 0.0:
            self._current_speed_limit = float(raw_limit)
        speed_limit_kmh = self._current_speed_limit

        lane_features, lane_info = self._get_lane_features()
        vehicle_state = self._get_vehicle_state(speed_limit_kmh)
        route_features = self._get_route_features(speed_limit_kmh)

        lidar_end = self.LIDAR_DIM + self.DYNAMIC_DIM + self.STATIC_DIM
        obs = np.concatenate(
            [
                lidar_combined,
                lidar_dynamic,
                lidar_static,
                lane_features,
                vehicle_state,
                route_features,
            ],
            dtype=np.float32,
        )
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs[:lidar_end] = np.clip(obs[:lidar_end], 0.0, 1.0)
        obs[lidar_end:] = np.clip(obs[lidar_end:], -1.0, 1.0)

        collision = self.sensor_manager.get_collision()
        lane_invasion = self.sensor_manager.get_lane_invasion()

        if collision:
            self.episode_collisions += 1
        if lane_invasion:
            self.episode_lane_invasions += 1

        v = self.ego_vehicle.get_velocity()
        speed_ms = math.sqrt(v.x**2 + v.y**2)
        speed_kmh = speed_ms * 3.6

        self._low_speed_window.append(speed_kmh < self.stuck_speed_kmh)

        min_front_norm = sem.min_front_combined
        min_front_m = min_front_norm * self.lidar_range
        ttc_s = (min_front_m / speed_ms) if speed_ms > 0.5 else 1e6

        info: Dict = {}

        info["collision"] = collision
        info["lane_invasion"] = lane_invasion

        info["lateral_offset"] = lane_info.get("lateral_offset", 0.0)
        info["lateral_offset_norm"] = lane_info.get("lateral_offset_norm", 0.0)
        info["heading_error"] = lane_info.get("heading_error_deg", 0.0)
        info["heading_error_norm"] = lane_info.get("heading_error_norm", 0.0)
        info["lane_width"] = lane_info.get("lane_width", 3.5)
        info["on_road"] = lane_info.get("on_road", True)
        info["on_edge_warning"] = lane_info.get("on_edge_warning", 0.0)
        info["dist_left_edge_norm"] = lane_info.get("dist_left_edge_norm", 0.5)
        info["dist_right_edge_norm"] = lane_info.get("dist_right_edge_norm", 0.5)
        info["lane_change_left"] = lane_info.get("lane_change_left", False)
        info["lane_change_right"] = lane_info.get("lane_change_right", False)
        info["lane_change_permitted"] = lane_info.get("lane_change_permitted", False)
        info["road_curvature_norm"] = lane_info.get("road_curvature_norm", 0.0)
        info["waypoint"] = lane_info.get("waypoint")
        info["lane_id"] = lane_info.get("lane_id", 0)
        info["road_id"] = lane_info.get("road_id", 0)
        info["solid_left"] = lane_info.get("solid_left", False)
        info["solid_right"] = lane_info.get("solid_right", False)
        info["dashed_left"] = lane_info.get("dashed_left", False)
        info["dashed_right"] = lane_info.get("dashed_right", False)

        info["speed_kmh"] = speed_kmh
        info["speed_ms"] = speed_ms
        info["steering"] = float(self.ego_vehicle.get_control().steer)
        info["speed_limit_kmh"] = speed_limit_kmh
        info["speed_limit_norm"] = float(
            np.clip(speed_limit_kmh / self.MAX_SPEED_LIMIT_KMH, 0.0, 1.0)
        )
        info["semantic_data_fresh"] = bool(sem_status.get("fresh", 0))
        info["semantic_stale_reads"] = int(sem_status.get("stale_reads", 0))
        info["semantic_fresh_reads"] = int(sem_status.get("fresh_reads", 0))
        info["semantic_last_frame"] = int(sem_status.get("last_frame", -1))
        info["semantic_pts_per_frame"] = int(sem_status.get("pts_per_frame", 0))

        total_alto = info["semantic_fresh_reads"] + info["semantic_stale_reads"]
        info["semantic_stale_ratio"] = (
            info["semantic_stale_reads"] / total_alto if total_alto > 0 else 0.0
        )

        info["world_tick_frame"] = self._last_tick_frame

        info["ttc_seconds"] = ttc_s

        if len(self._low_speed_window) > 0:
            info["low_speed_fraction"] = sum(self._low_speed_window) / len(
                self._low_speed_window
            )
        else:
            info["low_speed_fraction"] = 0.0

        info["total_distance"] = self.total_distance
        info["success_distance"] = self.success_distance

        info.update(sem.to_info_dict())

        lane_markings = self._sample_lane_markings()
        for k, v in lane_markings.items():
            info[f"lane_marking_{k}"] = v

        return obs.astype(np.float32), info

    _SOLID_MARKING_TYPES = frozenset(
        {
            carla.LaneMarkingType.Solid,
            carla.LaneMarkingType.SolidSolid,
            carla.LaneMarkingType.SolidBroken,
            carla.LaneMarkingType.BrokenSolid,
        }
    )
    _DASHED_MARKING_TYPES = frozenset(
        {
            carla.LaneMarkingType.Broken,
            carla.LaneMarkingType.BrokenBroken,
        }
    )

    _LANE_SAMPLE_RANGE_M = 40.0
    _LANE_SAMPLE_STEP_M = 2.0

    def _sample_lane_markings(self) -> Dict[str, np.ndarray]:
        """
        Samplea las marcas de carril visibles alrededor del ego usando el
        Waypoint API. Las marcas se devuelven en el frame del SENSOR
        (UE LH: x=adelante, y=derecha) para que el dashboard las plotee
        sin transformaciones extra.

        Por qué Waypoint API y NO LIDAR: en CARLA las RoadLines son una
        textura sobre el mesh del Road, no un mesh aparte. El LIDAR
        semántico nunca emite tag 24 (verificado en CARLA Issues #455 y
        #3638). El waypoint API en cambio expone la posición exacta de
        las líneas en el OpenDRIVE del mapa.

        Returns:
            Dict con cuatro arrays float32 en frame ego:
              left_solid_x/y   → puntos de las líneas sólidas a la izq.
              left_dashed_x/y  → puntos de las líneas discontinuas a la izq.
              right_solid_x/y  → puntos de las líneas sólidas a la dcha.
              right_dashed_x/y → puntos de las líneas discontinuas a la dcha.
            Si no hay carril válido (ego fuera de calzada), todos vacíos.
        """
        empty = np.zeros(0, dtype=np.float32)
        out = {
            "left_solid_x": empty.copy(),
            "left_solid_y": empty.copy(),
            "left_dashed_x": empty.copy(),
            "left_dashed_y": empty.copy(),
            "right_solid_x": empty.copy(),
            "right_solid_y": empty.copy(),
            "right_dashed_x": empty.copy(),
            "right_dashed_y": empty.copy(),
        }

        ego_tf = self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_yaw_rad = math.radians(ego_tf.rotation.yaw)
        cos_y = math.cos(ego_yaw_rad)
        sin_y = math.sin(ego_yaw_rad)

        wp = self.map.get_waypoint(
            ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if wp is None:
            return out

        step = self._LANE_SAMPLE_STEP_M
        n_steps = int(self._LANE_SAMPLE_RANGE_M / step)
        sampled_wps = [wp]
        cur = wp
        for _ in range(n_steps):
            nxt = cur.next(step)
            if not nxt:
                break
            cur = nxt[0]
            sampled_wps.append(cur)
        cur = wp
        for _ in range(n_steps):
            prv = cur.previous(step)
            if not prv:
                break
            cur = prv[0]
            sampled_wps.append(cur)

        left_solid, left_dashed = [], []
        right_solid, right_dashed = [], []

        for swp in sampled_wps:
            wp_tf = swp.transform
            wp_loc = wp_tf.location
            half_w = max(swp.lane_width, 2.0) / 2.0
            right_vec = wp_tf.get_right_vector()

            left_world = (
                wp_loc.x - right_vec.x * half_w,
                wp_loc.y - right_vec.y * half_w,
            )
            right_world = (
                wp_loc.x + right_vec.x * half_w,
                wp_loc.y + right_vec.y * half_w,
            )

            for marking_world, bucket_solid, bucket_dashed, marking_type in (
                (
                    left_world,
                    left_solid,
                    left_dashed,
                    swp.left_lane_marking.type,
                ),
                (
                    right_world,
                    right_solid,
                    right_dashed,
                    swp.right_lane_marking.type,
                ),
            ):
                dx = marking_world[0] - ego_loc.x
                dy = marking_world[1] - ego_loc.y
                x_ego = dx * cos_y + dy * sin_y
                y_ego = -dx * sin_y + dy * cos_y
                if abs(x_ego) > self.lidar_range or abs(y_ego) > self.lidar_range:
                    continue
                if marking_type in self._SOLID_MARKING_TYPES:
                    bucket_solid.append((x_ego, y_ego))
                elif marking_type in self._DASHED_MARKING_TYPES:
                    bucket_dashed.append((x_ego, y_ego))

        def to_arrays(pairs):
            if not pairs:
                return empty.copy(), empty.copy()
            arr = np.asarray(pairs, dtype=np.float32)
            return arr[:, 0], arr[:, 1]

        out["left_solid_x"], out["left_solid_y"] = to_arrays(left_solid)
        out["left_dashed_x"], out["left_dashed_y"] = to_arrays(left_dashed)
        out["right_solid_x"], out["right_solid_y"] = to_arrays(right_solid)
        out["right_dashed_x"], out["right_dashed_y"] = to_arrays(right_dashed)
        return out

    def _get_lane_features(self) -> Tuple[np.ndarray, Dict]:
        """
        Extrae características de carril usando el Waypoint API de CARLA.

        Retorna 12 features (8 base + 4 nuevas de tipo de marca):
          [0..7]   features base (offset, heading, edge, lane_change_left, ...)
          [8]  solid_left   — borde izquierdo es línea sólida (no cruzable)
          [9]  solid_right  — borde derecho es línea sólida
          [10] dashed_left  — borde izquierdo es línea discontinua (cruzable)
          [11] dashed_right — borde derecho es línea discontinua

        El agente necesita los 4 flags de tipo de marca para distinguir cruces
        ilegales de maniobras legales: el sensor de invasión filtra a sólidas
        pero esa señal sólo llega como evento puntual; aquí va en el estado.
        """
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_loc = vehicle_transform.location

        waypoint = self.map.get_waypoint(
            vehicle_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if waypoint is None:
            features = np.zeros(self.LANE_DIM + self.LANE_MARKING_DIM, dtype=np.float32)
            features[2] = 1.0
            features[3] = 0.5
            return features, {
                "lateral_offset": 0.0,
                "lateral_offset_norm": 0.0,
                "heading_error_deg": 0.0,
                "heading_error_norm": 0.0,
                "on_road": False,
                "on_edge_warning": 1.0,
                "lane_width": 3.5,
                "dist_left_edge_norm": 0.0,
                "dist_right_edge_norm": 0.0,
                "lane_change_left": False,
                "lane_change_right": False,
                "lane_change_permitted": False,
                "road_curvature_norm": 0.0,
                "lane_id": 0,
                "road_id": 0,
                "solid_left": False,
                "solid_right": False,
                "dashed_left": False,
                "dashed_right": False,
            }

        wp_transform = waypoint.transform
        lane_width = max(waypoint.lane_width, 2.0)
        half_width = lane_width / 2.0

        wp_right = wp_transform.get_right_vector()
        diff = vehicle_loc - wp_transform.location
        lateral_offset = diff.x * wp_right.x + diff.y * wp_right.y
        lateral_offset_norm = float(np.clip(lateral_offset / half_width, -1.0, 1.0))

        vehicle_yaw = vehicle_transform.rotation.yaw
        lane_yaw = wp_transform.rotation.yaw
        heading_error_deg = vehicle_yaw - lane_yaw
        heading_error_deg = ((heading_error_deg + 180.0) % 360.0) - 180.0
        heading_error_norm = float(np.clip(heading_error_deg / 180.0, -1.0, 1.0))

        dist_to_edge = 1.0 - abs(lateral_offset_norm)
        edge_threshold = 0.3
        on_edge_warning = float(
            np.clip((edge_threshold - dist_to_edge) / edge_threshold, 0.0, 1.0)
            if dist_to_edge < edge_threshold
            else 0.0
        )

        lane_width_norm = float(np.clip(lane_width / 4.5, 0.0, 1.0))

        nearest_driving_wp = self.map.get_waypoint(
            vehicle_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if nearest_driving_wp is None:
            on_road = False
        else:
            wp_loc = nearest_driving_wp.transform.location
            dx = vehicle_loc.x - wp_loc.x
            dy = vehicle_loc.y - wp_loc.y
            dist_to_nearest_lane = math.sqrt(dx * dx + dy * dy)
            on_road_threshold = (nearest_driving_wp.lane_width / 2.0) * 1.15
            on_road = dist_to_nearest_lane <= on_road_threshold

        dist_left_edge_norm = float(
            np.clip((half_width - lateral_offset) / half_width, 0.0, 1.0)
        )
        dist_right_edge_norm = float(
            np.clip((half_width + lateral_offset) / half_width, 0.0, 1.0)
        )

        lc = waypoint.lane_change
        lane_change_left = float(lc in (carla.LaneChange.Left, carla.LaneChange.Both))
        lane_change_right = float(lc in (carla.LaneChange.Right, carla.LaneChange.Both))
        lane_change_permitted = lc != carla.LaneChange.NONE

        left_marking_type = waypoint.left_lane_marking.type
        right_marking_type = waypoint.right_lane_marking.type
        solid_left = left_marking_type in self._SOLID_MARKING_TYPES
        solid_right = right_marking_type in self._SOLID_MARKING_TYPES
        dashed_left = left_marking_type in self._DASHED_MARKING_TYPES
        dashed_right = right_marking_type in self._DASHED_MARKING_TYPES

        road_curvature_norm = 0.0
        next_wps_10 = waypoint.next(10.0)
        if next_wps_10:
            wp10_yaw = next_wps_10[0].transform.rotation.yaw
            curv_deg = wp10_yaw - lane_yaw
            curv_deg = ((curv_deg + 180.0) % 360.0) - 180.0
            road_curvature_norm = float(np.clip(curv_deg / 45.0, -1.0, 1.0))

        features = np.array(
            [
                lateral_offset_norm,
                heading_error_norm,
                on_edge_warning,
                lane_width_norm,
                dist_left_edge_norm,
                dist_right_edge_norm,
                lane_change_left,
                road_curvature_norm,
                float(solid_left),
                float(solid_right),
                float(dashed_left),
                float(dashed_right),
            ],
            dtype=np.float32,
        )

        info = {
            "lateral_offset": float(lateral_offset),
            "lateral_offset_norm": lateral_offset_norm,
            "heading_error_deg": float(heading_error_deg),
            "heading_error_norm": heading_error_norm,
            "on_road": bool(on_road),
            "on_edge_warning": on_edge_warning,
            "lane_width": float(lane_width),
            "dist_left_edge_norm": dist_left_edge_norm,
            "dist_right_edge_norm": dist_right_edge_norm,
            "lane_change_left": bool(lane_change_left),
            "lane_change_right": bool(lane_change_right),
            "lane_change_permitted": bool(lane_change_permitted),
            "road_curvature_norm": road_curvature_norm,
            "waypoint": waypoint,
            "lane_id": int(waypoint.lane_id),
            "road_id": int(waypoint.road_id),
            "solid_left": bool(solid_left),
            "solid_right": bool(solid_right),
            "dashed_left": bool(dashed_left),
            "dashed_right": bool(dashed_right),
        }

        return features, info

    def _get_vehicle_state(self, speed_limit_kmh: float) -> np.ndarray:
        """Retorna estado normalizado del vehículo: [speed_norm, steering]."""
        v = self.ego_vehicle.get_velocity()
        speed_ms = math.sqrt(v.x**2 + v.y**2)
        speed_kmh = speed_ms * 3.6
        norm_ref = max(speed_limit_kmh * 1.5, 10.0)
        speed_norm = float(np.clip(speed_kmh / norm_ref, 0.0, 1.0))
        steering = float(np.clip(self.ego_vehicle.get_control().steer, -1.0, 1.0))
        return np.array([speed_norm, steering], dtype=np.float32)

    def _get_route_features(self, speed_limit_kmh: float) -> np.ndarray:
        """
        Retorna información de ruta:
          [0] angle_to_wp_5m_norm   — ángulo al siguiente waypoint a 5 m
          [1] angle_to_wp_20m_norm  — ángulo al waypoint a 20 m (anticipa curvas)
          [2] progress_norm         — progreso del episodio [0,1]
          [3] speed_limit_norm      — límite de velocidad normalizado [0,1]
          [4] speed_ratio           — speed/limit, >1 si excede el límite

        El ángulo a 20 m permite al agente anticipar curvas con suficiente antelación
        para ajustar velocidad y posición lateral antes de entrar en la curva.
        speed_ratio da al agente contexto sobre si va demasiado rápido para el límite actual.
        """
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_loc = vehicle_transform.location

        waypoint = self.map.get_waypoint(vehicle_loc, project_to_road=True)

        speed_limit_norm = float(
            np.clip(speed_limit_kmh / self.MAX_SPEED_LIMIT_KMH, 0.0, 1.0)
        )

        v = self.ego_vehicle.get_velocity()
        speed_ms = math.sqrt(v.x**2 + v.y**2)
        speed_kmh_now = speed_ms * 3.6
        speed_ratio = float(
            np.clip(speed_kmh_now / max(speed_limit_kmh, 1.0), 0.0, 2.0)
        )

        progress_norm = float(
            np.clip(self.total_distance / self.success_distance, 0.0, 1.0)
        )

        if waypoint is None:
            return np.array(
                [0.0, 0.0, progress_norm, speed_limit_norm, speed_ratio],
                dtype=np.float32,
            )

        vehicle_yaw = vehicle_transform.rotation.yaw

        angle_5m_norm = 0.0
        next_wps_5 = waypoint.next(5.0)
        if next_wps_5:
            diff_yaw = next_wps_5[0].transform.rotation.yaw - vehicle_yaw
            diff_yaw = ((diff_yaw + 180.0) % 360.0) - 180.0
            angle_5m_norm = float(np.clip(diff_yaw / 180.0, -1.0, 1.0))

        angle_20m_norm = 0.0
        next_wps_20 = waypoint.next(20.0)
        if next_wps_20:
            diff_yaw = next_wps_20[0].transform.rotation.yaw - vehicle_yaw
            diff_yaw = ((diff_yaw + 180.0) % 360.0) - 180.0
            angle_20m_norm = float(np.clip(diff_yaw / 180.0, -1.0, 1.0))

        return np.array(
            [
                angle_5m_norm,
                angle_20m_norm,
                progress_norm,
                speed_limit_norm,
                speed_ratio,
            ],
            dtype=np.float32,
        )

    def _action_to_control(self, action: np.ndarray) -> carla.VehicleControl:
        """Convierte acción normalizada [-1,1]² a VehicleControl de CARLA."""
        steering = float(np.clip(action[0], -1.0, 1.0))
        tb = float(np.clip(action[1], -1.0, 1.0))

        if tb >= 0.0:
            throttle = float(tb)
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(-tb)

        return carla.VehicleControl(
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            steer=steering,
            brake=float(np.clip(brake, 0.0, 1.0)),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )

    def _compute_base_reward(self, action: np.ndarray, info: Dict) -> float:
        """
        Recompensa base.
        Usa forward_speed_ms (dot velocity x heading) en lugar de |v|
        para no recompensar la marcha atrás.
        """
        t = self.ego_vehicle.get_transform()
        v = self.ego_vehicle.get_velocity()
        yaw = math.radians(t.rotation.yaw)
        fwd = v.x * math.cos(yaw) + v.y * math.sin(yaw)
        fwd = max(fwd, 0.0)

        reward = fwd * self.fixed_delta_seconds * 0.3

        if info.get("collision", False):
            reward -= self.crash_penalty

        if not info.get("on_road", True):
            reward -= self.out_of_road_penalty

        if self.total_distance >= self.success_distance:
            reward += self.success_reward

        return float(reward)

    def _parse_image(self, image):
        """Convierte la imagen raw de CARLA a un array numpy (BGR)."""
        if self.render_mode != "human":
            return

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self.current_image = array[:, :, :3]

    # TERMINACIÓN

    def _check_termination(self, info: Dict) -> Tuple[bool, bool]:
        """Verifica condiciones de terminación del episodio."""

        # Colisión
        if info.get("collision", False):
            info["crash_vehicle"] = True
            return True, False

        if not info.get("on_road", True):
            info["out_of_road"] = True
            return True, False

        if self.total_distance >= self.success_distance:
            info["arrive_dest"] = True
            return True, False

        if (
            len(self._low_speed_window) >= self.stuck_window_size
            and sum(self._low_speed_window)
            >= self.stuck_threshold_fraction * self.stuck_window_size
        ):
            info["stuck"] = True
            return False, True

        if self.step_count >= self.max_episode_steps:
            return False, True

        return False, False

    def _spawn_ego_vehicle(self):
        """Spawna el vehículo ego en un punto de spawn válido."""
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("role_name", "hero")

        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"No spawn points found in map {self.map_name}")

        if self.spawn_point_idx is not None and 0 <= self.spawn_point_idx < len(
            spawn_points
        ):
            candidates = [spawn_points[self.spawn_point_idx]]
        elif self.spawn_point_indices:
            valid = [
                spawn_points[i]
                for i in self.spawn_point_indices
                if 0 <= i < len(spawn_points)
            ]
            if not valid:
                logger.warning(
                    f"spawn_point_indices={self.spawn_point_indices} no contiene "
                    f"índices válidos (mapa tiene {len(spawn_points)} spawns). "
                    f"Fallback a random sobre todos."
                )
                candidates = list(spawn_points)
            else:
                candidates = valid
            random.shuffle(candidates)
        else:
            candidates = list(spawn_points)
            random.shuffle(candidates)

        for sp in candidates:
            actor = self.world.try_spawn_actor(vehicle_bp, sp)
            if actor is not None:
                self.ego_vehicle = actor
                self._ego_spawn_transform = sp  # pose exacta para el route traffic
                self.ego_vehicle.set_autopilot(False)
                self.ego_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=0.3)
                )
                return

        raise RuntimeError("Failed to spawn ego vehicle after trying all spawn points")

    def _spawn_npc_vehicles(self):
        """Spawna vehículos NPC usando TrafficManager de CARLA."""
        if self.num_npc_vehicles == 0:
            return

        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = [
            bp
            for bp in bp_lib.filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels")) == 4
        ]

        spawn_points = list(self.map.get_spawn_points())
        random.shuffle(spawn_points)
        ego_loc = self.ego_vehicle.get_location()

        spawned = 0
        for sp in spawn_points:
            if spawned >= self.num_npc_vehicles:
                break
            if sp.location.distance(ego_loc) < 25.0:
                continue
            bp = random.choice(vehicle_bps)
            npc = self.world.try_spawn_actor(bp, sp)
            if npc is not None:
                npc.set_autopilot(True, self._tm.get_port())
                self._tm.vehicle_percentage_speed_difference(
                    npc, random.uniform(-20, 10)
                )
                self._tm.distance_to_leading_vehicle(npc, random.uniform(1.5, 4.0))
                self.npc_vehicles.append(npc)
                spawned += 1

    # Trafico de exposicion en ruta (curriculum de colision frontal).
    ROUTE_TRAFFIC_SPAWN_Z = 1.0  # lift sobre el waypoint (los map spawns usan +0.6;
    #   un lift menor hace que el bbox clipe el mesh y try_spawn falle en silencio).
    ROUTE_TRAFFIC_GAP_M = (40.0, 60.0)  # separacion entre leads consecutivos.
    #   Trade-off medido (diag 2026-06-14): leads CERCA (20-35 m) → 83% crash contra
    #   guardarrail al esquivar; leads LEJOS (50-70 m) → 50%. Más distancia = más
    #   margen de reacción, así que se usa 40-60 m. (El LIDAR semántico sparse sólo
    #   detecta vehículos a <~15 m, pero el problema dominante NO es la detección
    #   sino que el agente no sabe ceder/seguir y se va al guardarrail.)
    ROUTE_TRAFFIC_SPEED_KMH = (40.0, 48.0)  # velocidad ABSOLUTA de los leads.
    #   Diag 2026-06-15: leads a 10-20 km/h vs un ego que crucea a ~55-60 km/h daban
    #   una velocidad de cierre de ~45 km/h; con detección LIDAR de vehículos sólo a
    #   ~11 m eso es <1 s de aviso -> alcance POR DETRÁS inevitable (los "crashes
    #   contra guardarrail" eran en realidad alcances al lead, mal atribuidos).
    #   40-48 km/h son más lentos que el ego (hay encuentro) pero con cierre moderado
    #   ~10-20 km/h (~2-3 s de margen): el encuentro es EVITABLE y por tanto aprendible.
    # NOTA: los leads se colocan SIEMPRE en el carril del ego (in-lane). El antiguo
    #   35% en carril adyacente quedaba FUERA del cono frontal del shield (±22.5°) y
    #   del obs frontal -> el ego los rozaba sin "verlos" de frente (sideswipe).

    def _spawn_route_traffic(self):
        """Inyecta ``route_npc_count`` leads lentos en la ruta del ego.

        Camina hacia delante por la cadena de waypoints del ego colocando
        vehiculos IN-LANE (en el carril del ego) a `ROUTE_TRAFFIC_GAP_M`, a
        velocidad moderada para que el ego los alcance de frente con margen. Se
        anaden a ``npc_vehicles`` (limpieza existente). Defensivo: logs a INFO si
        spawnea menos de lo pedido; ``last_route_npcs_spawned`` = verdad de campo.
        """
        self.last_route_npcs_spawned = 0
        requested = int(getattr(self, "route_npc_count", 0))
        if requested <= 0 or self.ego_vehicle is None:
            return

        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = [
            bp
            for bp in bp_lib.filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels")) == 4
        ]

        # Pose del ego: usa la transform de SPAWN (exacta y disponible sin tick).
        # En modo síncrono, get_transform() devuelve el origen hasta el primer
        # world.tick() — y este método corre ANTES del tick del reset, así que leer
        # el actor colocaba el tráfico respecto al ORIGEN del mapa (síntoma: leads a
        # cientos de metros y con bearings detrás del ego).
        ego_tf = self._ego_spawn_transform or self.ego_vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_fwd = ego_tf.get_forward_vector()
        wp = self.map.get_waypoint(ego_loc, project_to_road=True)
        if wp is None:
            logger.info("[route_traffic] ego waypoint None; 0 leads spawned")
            return

        # CARLA docs: next()/previous() recorren la lane en la dirección OpenDRIVE
        # (lane_id positivo = sentido OPUESTO a la geometría), NO en la del ego. Si
        # caminásemos siempre con next() los leads pueden caer DETRÁS del ego (lo
        # observado: bearings ~±160°). Elegimos next/previous según el signo del
        # producto escalar entre el forward del ego y el de la lane, para avanzar
        # SIEMPRE en el sentido de marcha real del ego.
        lane_fwd = wp.transform.get_forward_vector()
        use_next = (
            ego_fwd.x * lane_fwd.x + ego_fwd.y * lane_fwd.y + ego_fwd.z * lane_fwd.z
        ) >= 0.0

        def _advance(w, dist):
            chain = w.next(dist) if use_next else w.previous(dist)
            return chain[0] if chain else None

        spawned = 0
        for k in range(requested):
            gap = random.uniform(*self.ROUTE_TRAFFIC_GAP_M)
            wp = _advance(wp, gap)
            if wp is None:
                break  # fin de la ruta alcanzable

            place_wp = wp  # SIEMPRE in-lane (sin adyacentes; ver nota en constantes)

            # Guard: nunca spawnear DETRÁS del ego (topología en bucle de Town04
            # puede curvar la ruta de vuelta). dot(ego→lead, ego_fwd) > 0 ⇒ delante.
            lead_loc = place_wp.transform.location
            ahead = (
                (lead_loc.x - ego_loc.x) * ego_fwd.x
                + (lead_loc.y - ego_loc.y) * ego_fwd.y
                + (lead_loc.z - ego_loc.z) * ego_fwd.z
            )
            if ahead <= 0.0:
                continue  # detrás: lo saltamos (seguimos avanzando la cadena)

            tf = place_wp.transform
            tf.location.z += self.ROUTE_TRAFFIC_SPAWN_Z
            npc = self.world.try_spawn_actor(random.choice(vehicle_bps), tf)
            if npc is None:
                continue  # ocupado / clip; seguimos al siguiente waypoint
            npc.set_autopilot(True, self._tm.get_port())
            self._tm.auto_lane_change(npc, False)
            self._tm.set_desired_speed(
                npc, float(random.uniform(*self.ROUTE_TRAFFIC_SPEED_KMH))
            )
            self.npc_vehicles.append(npc)
            spawned += 1

        self.last_route_npcs_spawned = spawned
        if spawned < requested:
            logger.info(
                f"[route_traffic] requested {requested} leads, spawned {spawned} "
                f"(occupied spawns / end of route)"
            )

    def _cleanup(self):
        """Destruye todos los actores del episodio anterior."""
        if self.camera_sensor is not None and self.camera_sensor.is_alive:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None
        self.current_image = None

        if self.sensor_manager is not None:
            self.sensor_manager.destroy()
            self.sensor_manager = None

        actors_to_destroy = [npc for npc in self.npc_vehicles if npc.is_alive]
        if actors_to_destroy:
            self.client.apply_batch_sync(
                [carla.command.DestroyActor(a) for a in actors_to_destroy],
                True,
            )
        self.npc_vehicles.clear()

        if self.ego_vehicle is not None and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

        if self.synchronous and self.world is not None:
            try:
                self.world.tick()
            except Exception:
                pass
