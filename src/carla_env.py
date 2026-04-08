"""
carla_env.py - CARLA Gymnasium Environment for Safe RL

LAYOUT DE OBSERVACIÓN (249 dimensiones):
  obs[0:240]   → LIDAR scan normalizado (240 rayos, [0,1], 1=libre, ~0=obstáculo cercano)
  obs[240:244] → Lane features (lateral_offset_norm, heading_error_norm, on_edge_warn, lane_width_norm)
  obs[244:246] → Vehicle state (speed_norm, steering)
  obs[246:249] → Route info (next_wp_angle_norm, progress_norm)

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
from typing import Optional, Tuple, Dict

from src.carla_sensors import SensorManager

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

    # ── Constantes de observación ─────────────────────────────────────
    LIDAR_DIM      = 240
    LANE_DIM       = 4
    VEHICLE_DIM    = 2
    ROUTE_DIM      = 3
    OBS_DIM        = LIDAR_DIM + LANE_DIM + VEHICLE_DIM + ROUTE_DIM  # 249

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
        out_of_road_penalty: float = 10.0,
        crash_penalty: float = 10.0,
        seed: int = 42,
        spawn_point_idx: Optional[int] = None,
    ):
        super().__init__()

        # ── Config ────────────────────────────────────────────────────
        self.host               = host
        self.port               = port
        self.tm_port            = tm_port
        self.timeout            = timeout
        self.map_name           = map_name
        self.num_npc_vehicles   = num_npc_vehicles
        self.weather            = weather
        self.render_mode        = render_mode
        self.synchronous        = synchronous
        self.fixed_delta_seconds = fixed_delta_seconds
        self.num_lidar_rays     = num_lidar_rays
        self.lidar_range        = lidar_range
        self.lidar_height_filter = lidar_height_filter
        self.max_episode_steps  = max_episode_steps
        self.target_speed_kmh   = target_speed_kmh
        self.success_distance   = success_distance
        self.success_reward     = success_reward
        self.out_of_road_penalty = out_of_road_penalty
        self.crash_penalty      = crash_penalty
        self.base_seed          = seed
        self.spawn_point_idx    = spawn_point_idx

        # ── Gymnasium spaces ──────────────────────────────────────────
        obs_low  = np.concatenate([
            np.zeros(self.LIDAR_DIM, dtype=np.float32),
            np.full(self.LANE_DIM + self.VEHICLE_DIM + self.ROUTE_DIM,
                    -1.0, dtype=np.float32),
        ])
        obs_high = np.ones(self.OBS_DIM, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ── Estado CARLA ───────────────────────────────────────────────
        self.client: Optional[carla.Client]  = None
        self.world:  Optional[carla.World]   = None
        self.map:    Optional[carla.Map]     = None
        self.ego_vehicle: Optional[carla.Vehicle] = None
        self.sensor_manager: Optional[SensorManager] = None
        self.npc_vehicles = []
        self._tm: Optional[carla.TrafficManager] = None

        # ── Estado episodio ────────────────────────────────────────────
        self.step_count = 0
        self.total_distance = 0.0
        self._last_location: Optional[carla.Location] = None
        self._consecutive_stopped = 0
        self.episode_collisions = 0
        self.episode_lane_invasions = 0
        self.last_obs: Optional[np.ndarray] = None
        self.last_info: Dict = {}

        # ── Variables para renderizado ─────────────────────────
        self.camera_sensor: Optional[carla.Sensor] = None
        self.current_image: Optional[np.ndarray] = None
        self._cv2_window_created = False

        # ── Conectar ───────────────────────────────────────────────────
        self._connect()

    # ══════════════════════════════════════════════════════════════════
    # CONEXIÓN Y CONFIGURACIÓN
    # ══════════════════════════════════════════════════════════════════

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
            settings.no_rendering_mode = (self.render_mode is None)
            self.world.apply_settings(settings)

        # Clima
        weather_attr = getattr(
            carla.WeatherParameters, self.weather,
            carla.WeatherParameters.ClearNoon
        )
        self.world.set_weather(weather_attr)

        # TrafficManager
        self._tm = self.client.get_trafficmanager(self.tm_port)
        self._tm.set_synchronous_mode(self.synchronous)
        self._tm.set_global_distance_to_leading_vehicle(2.5)
        self._tm.set_random_device_seed(self.base_seed)

    # ══════════════════════════════════════════════════════════════════
    # GYMNASIUM API
    # ══════════════════════════════════════════════════════════════════

    def reset(self, *, seed=None, options=None):
        """Reinicia el entorno para un nuevo episodio."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._cleanup()
        self._spawn_ego_vehicle()
        self._spawn_npc_vehicles()

        self.sensor_manager = SensorManager(
            self.world,
            self.ego_vehicle,
            num_lidar_rays = self.num_lidar_rays,
            lidar_range = self.lidar_range,
            height_filter = self.lidar_height_filter,
        )

        if self.render_mode == "human":
            bp_lib = self.world.get_blueprint_library()
            camera_bp = bp_lib.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '640') # Resolución (ajústalo si quieres)
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '90')
            
            # Posición en 3ra persona: 5.5m atrás, 2.5m arriba, inclinada 8 grados abajo
            camera_transform = carla.Transform(
                carla.Location(x=-5.5, z=2.5), 
                carla.Rotation(pitch=-8.0)
            )
            
            self.camera_sensor = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.ego_vehicle
            )
            self.camera_sensor.listen(self._parse_image)

        # Reiniciar estado
        self.step_count = 0
        self.total_distance = 0.0
        self._consecutive_stopped = 0
        self.episode_collisions = 0
        self.episode_lane_invasions = 0
        self._current_speed_limit = self.target_speed_kmh

        loc = self.ego_vehicle.get_location()
        self._last_location = carla.Location(loc.x, loc.y, loc.z)

        # Tick inicial para poblar sensores
        if self.synchronous:
            for _ in range(3):
                self.world.tick()
        else:
            time.sleep(0.15)

        obs, info = self._build_observation()
        self.last_obs = obs
        self.last_info = info
        return obs, info

    def step(self, action: np.ndarray):
        """Ejecuta un paso de simulación."""
        # Aplicar control
        control = self._action_to_control(action)
        self.ego_vehicle.apply_control(control)

        # Avanzar simulación
        if self.synchronous:
            self.world.tick()

        self.step_count += 1

        # Actualizar distancia recorrida
        current_loc = self.ego_vehicle.get_location()
        if self._last_location is not None:
            step_dist = current_loc.distance(self._last_location)
            # Filtro: distancias muy grandes indican teleporte (error)
            if step_dist < 5.0:
                self.total_distance += step_dist
        self._last_location = carla.Location(current_loc.x, current_loc.y, current_loc.z)

        # Construir observación
        obs, info = self._build_observation()
        self.last_obs = obs
        self.last_info = info

        # Recompensa base
        reward = self._compute_base_reward(action, info)

        # Terminación
        done, truncated = self._check_termination(info)

        # Info adicional
        info.update({
            "step": self.step_count,
            "total_distance": self.total_distance,
            "episode_collisions": self.episode_collisions,
            "episode_lane_invasions": self.episode_lane_invasions,
        })

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
            cv2.waitKey(1) # Necesario para que OpenCV refresque la GUI

    # ══════════════════════════════════════════════════════════════════
    # OBSERVACIÓN
    # ══════════════════════════════════════════════════════════════════

    def _build_observation(self) -> Tuple[np.ndarray, Dict]:
        """
        Construye el vector de observación completo desde sensores y API CARLA.

        Retorna obs (248,) e info enriquecido con datos CARLA para los shields.
        """
        # LIDAR scan (240,)
        sem = self.sensor_manager.get_semantic_result()
        lidar_scan = sem.combined

        # Límite de velocidad dinámico
        raw_limit = self.ego_vehicle.get_speed_limit()
        if raw_limit > 0.0:
            self._current_speed_limit = float(raw_limit)
        speed_limit_kmh = self._current_speed_limit

        lane_features, lane_info = self._get_lane_features()
        vehicle_state = self._get_vehicle_state(speed_limit_kmh)
        route_features = self._get_route_features(speed_limit_kmh)

        obs = np.concatenate(
            [lidar_scan, lane_features, vehicle_state, route_features],
            dtype=np.float32
        )
        obs = np.clip(obs, -1.0, 1.0)

        # Eventos de sensores
        collision = self.sensor_manager.get_collision()
        lane_invasion = self.sensor_manager.get_lane_invasion()

        if collision:
            self.episode_collisions += 1
        if lane_invasion:
            self.episode_lane_invasions += 1

        # Velocidad actual
        v = self.ego_vehicle.get_velocity()
        speed_ms = math.sqrt(v.x**2 + v.y**2)
        speed_kmh = speed_ms * 3.6

        # Detectar vehículo parado
        if speed_kmh < 1.0:
            self._consecutive_stopped += 1
        else:
            self._consecutive_stopped = 0

        # TTC usando combined scan (frente)
        min_front_norm = sem.min_front_combined
        min_front_m    = min_front_norm * self.lidar_range
        ttc_s = (min_front_m / speed_ms) if speed_ms > 0.5 else float("inf")
        
        info: Dict = {}
 
        # — Eventos —
        info["collision"]     = collision
        info["lane_invasion"] = lane_invasion
 
        # — Lane (Waypoint API) —
        info["lateral_offset"]      = lane_info.get("lateral_offset", 0.0)
        info["lateral_offset_norm"] = lane_info.get("lateral_offset_norm", 0.0)
        info["heading_error"]       = lane_info.get("heading_error_deg", 0.0)
        info["heading_error_norm"]  = lane_info.get("heading_error_norm", 0.0)
        info["lane_width"]          = lane_info.get("lane_width", 3.5)
        info["on_road"]             = lane_info.get("on_road", True)
        info["on_edge_warning"]     = lane_info.get("on_edge_warning", 0.0)
        info["waypoint"]            = lane_info.get("waypoint")
 
        # — Vehículo —
        info["speed_kmh"]          = speed_kmh
        info["speed_ms"]           = speed_ms
        info["steering"]           = float(self.ego_vehicle.get_control().steer)
        info["speed_limit_kmh"]    = speed_limit_kmh
        info["speed_limit_norm"]   = float(np.clip(speed_limit_kmh / self.MAX_SPEED_LIMIT_KMH, 0.0, 1.0))
 
        # — TTC —
        info["ttc_seconds"]        = ttc_s
        info["consecutive_stopped"]= self._consecutive_stopped
 
        # — Progreso —
        info["total_distance"]     = self.total_distance
        info["success_distance"]   = self.success_distance
 
        # — LIDAR semántico completo (to_info_dict puebla todos los campos) —
        info.update(sem.to_info_dict())
 
        return obs.astype(np.float32), info

    def _get_lane_features(self) -> Tuple[np.ndarray, Dict]:
        """
        Extrae características de carril usando el Waypoint API de CARLA.
        """
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_loc = vehicle_transform.location

        # Waypoint más cercano en un carril de conducción
        waypoint = self.map.get_waypoint(
            vehicle_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if waypoint is None:
            # Completamente fuera de carretera
            features = np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float32)
            return features, {
                "lateral_offset": 0.0,
                "lateral_offset_norm": 0.0,
                "heading_error_deg": 0.0,
                "heading_error_norm": 0.0,
                "on_road": False,
                "on_edge_warning": 1.0,
                "lane_width": 3.5,
            }

        wp_transform = waypoint.transform
        lane_width = max(waypoint.lane_width, 2.0)  # mínimo 2m
        half_width = lane_width / 2.0

        # Offset lateral (metros, positivo = derecha del carril)
        # Proyectamos el vector vehículo→waypoint sobre el vector lateral del waypoint
        wp_right = wp_transform.get_right_vector()
        diff = vehicle_loc - wp_transform.location
        lateral_offset = diff.x * wp_right.x + diff.y * wp_right.y
        lateral_offset_norm = float(np.clip(lateral_offset / half_width, -1.0, 1.0))

        # Error de heading (grados)
        vehicle_yaw = vehicle_transform.rotation.yaw
        lane_yaw = wp_transform.rotation.yaw
        heading_error_deg = vehicle_yaw - lane_yaw
        heading_error_deg = ((heading_error_deg + 180.0) % 360.0) - 180.0
        heading_error_norm = float(np.clip(heading_error_deg / 180.0, -1.0, 1.0))

        # On-edge warning
        dist_to_edge = 1.0 - abs(lateral_offset_norm)
        edge_threshold = 0.3
        on_edge_warning = float(
            np.clip((edge_threshold - dist_to_edge) / edge_threshold, 0.0, 1.0)
            if dist_to_edge < edge_threshold else 0.0
        )

        # Lane width normalizado
        lane_width_norm = float(np.clip(lane_width / 4.5, 0.0, 1.0))

        # On-road check
        # Verificamos si el vehículo está en un carril válido (no sólo cerca)
        road_waypoint = self.map.get_waypoint(
            vehicle_loc,
            project_to_road=False,  # False = solo si está ON carril
        )
        on_road = (
            road_waypoint is not None and
            road_waypoint.lane_type == carla.LaneType.Driving
        )

        features = np.array([
            lateral_offset_norm,
            heading_error_norm,
            on_edge_warning,
            lane_width_norm,
        ], dtype=np.float32)

        info = {
            "lateral_offset": float(lateral_offset),
            "lateral_offset_norm": lateral_offset_norm,
            "heading_error_deg": float(heading_error_deg),
            "heading_error_norm": heading_error_norm,
            "on_road": bool(on_road),
            "on_edge_warning": on_edge_warning,
            "lane_width": float(lane_width),
            "waypoint": waypoint,  # objeto CARLA para uso en shields
        }

        return features, info

    def _get_vehicle_state(self, speed_limit_kmh: float) -> np.ndarray:
        """Retorna estado normalizado del vehículo: [speed_norm, steering]."""
        v = self.ego_vehicle.get_velocity()
        speed_ms = math.sqrt(v.x**2 + v.y**2)
        speed_kmh = speed_ms * 3.6
        # Normalizar contra 150% del límite dinámico para headroom natural
        norm_ref = max(speed_limit_kmh * 1.5, 10.0)  # mínimo 10 km/h para evitar NaN
        speed_norm = float(np.clip(speed_kmh / norm_ref, 0.0, 1.0))
        steering = float(np.clip(self.ego_vehicle.get_control().steer, -1.0, 1.0))
        return np.array([speed_norm, steering], dtype=np.float32)

    def _get_route_features(self, speed_limit_kmh: float) -> np.ndarray:
        """
        Retorna información de ruta: [angle_to_next_wp_norm, progress_norm].

        Usa el Waypoint API para obtener el siguiente waypoint 5m adelante,
        lo que guía al agente a seguir la carretera.
        """
        vehicle_transform = self.ego_vehicle.get_transform()
        vehicle_loc = vehicle_transform.location

        waypoint = self.map.get_waypoint(vehicle_loc, project_to_road=True)

        speed_limit_norm = float(np.clip(speed_limit_kmh / self.MAX_SPEED_LIMIT_KMH, 0.0, 1.0))


        if waypoint is None:
            return np.array([0.0, 0.0, speed_limit_norm], dtype=np.float32)
 
        next_wps = waypoint.next(5.0)
        if not next_wps:
            return np.array([0.0, 0.0, speed_limit_norm], dtype=np.float32)

        next_wp = next_wps[0]

        # Ángulo entre heading actual y dirección del siguiente waypoint
        next_yaw = next_wp.transform.rotation.yaw
        vehicle_yaw = vehicle_transform.rotation.yaw
        angle_diff = next_yaw - vehicle_yaw
        angle_diff = ((angle_diff + 180.0) % 360.0) - 180.0
        angle_norm = float(np.clip(angle_diff / 180.0, -1.0, 1.0))

        # Progreso del episodio
        progress_norm = float(np.clip(self.total_distance / self.success_distance, 0.0, 1.0))

        return np.array([angle_norm, progress_norm, speed_limit_norm], dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════
    # CONTROL Y RECOMPENSA BASE
    # ══════════════════════════════════════════════════════════════════

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
        Usa forward_speed_ms (dot velocity × heading) en lugar de |v|
        para no recompensar la marcha atrás.
        """
        t   = self.ego_vehicle.get_transform()
        v = self.ego_vehicle.get_velocity()
        yaw = math.radians(t.rotation.yaw)
        fwd = v.x * math.cos(yaw) + v.y * math.sin(yaw)
        fwd = max(fwd, 0.0)

        # Recompensa por avanzar (proporcional a velocidad hacia adelante)
        reward  = fwd * self.fixed_delta_seconds * 0.3

        # Penalización por colisión
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
        # CARLA devuelve BGRA, OpenCV usa BGR. Quitamos el canal alpha (A).
        self.current_image = array[:, :, :3]

    # ══════════════════════════════════════════════════════════════════
    # TERMINACIÓN
    # ══════════════════════════════════════════════════════════════════

    def _check_termination(self, info: Dict) -> Tuple[bool, bool]:
        """Verifica condiciones de terminación del episodio."""

        # Colisión
        if info.get("collision", False):
            info["crash_vehicle"] = True
            return True, False

        # Fuera de carretera
        if not info.get("on_road", True):
            # Doble verificación con el API
            wp = self.map.get_waypoint(
                self.ego_vehicle.get_location(),
                project_to_road=False,
            )
            if wp is None or wp.lane_type != carla.LaneType.Driving:
                info["out_of_road"] = True
                return True, False

        # Éxito: distancia completada
        if self.total_distance >= self.success_distance:
            info["arrive_dest"] = True
            return True, False

        # Vehículo parado demasiado tiempo (>15s = 300 steps a 20Hz)
        if self._consecutive_stopped > 300:
            info["stuck"] = True
            return False, True

        # Timeout
        if self.step_count >= self.max_episode_steps:
            return False, True

        return False, False

    # ══════════════════════════════════════════════════════════════════
    # SPAWN Y LIMPIEZA
    # ══════════════════════════════════════════════════════════════════

    def _spawn_ego_vehicle(self):
        """Spawna el vehículo ego en un punto de spawn válido."""
        bp_lib = self.world.get_blueprint_library()
        # Tesla Model 3: vehículo compacto, parámetros bien calibrados en CARLA
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle_bp.set_attribute("role_name", "hero")

        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"No spawn points found in map {self.map_name}")

        if (self.spawn_point_idx is not None and
                0 <= self.spawn_point_idx < len(spawn_points)):
            candidates = [spawn_points[self.spawn_point_idx]]
        else:
            candidates = list(spawn_points)
            random.shuffle(candidates)

        for sp in candidates:
            actor = self.world.try_spawn_actor(vehicle_bp, sp)
            if actor is not None:
                self.ego_vehicle = actor
                self.ego_vehicle.set_autopilot(False)
                # Control inicial: freno suave para que no deslice
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
        # Solo vehículos de 4 ruedas (excluir motos/bicis para simplificar)
        vehicle_bps = [
            bp for bp in bp_lib.filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels")) == 4
        ]

        spawn_points = list(self.map.get_spawn_points())
        random.shuffle(spawn_points)
        ego_loc = self.ego_vehicle.get_location()

        spawned = 0
        for sp in spawn_points:
            if spawned >= self.num_npc_vehicles:
                break
            # No spawnar demasiado cerca del ego
            if sp.location.distance(ego_loc) < 25.0:
                continue
            bp = random.choice(vehicle_bps)
            npc = self.world.try_spawn_actor(bp, sp)
            if npc is not None:
                npc.set_autopilot(True, self._tm.get_port())
                # Variedad de comportamientos NPC
                self._tm.vehicle_percentage_speed_difference(
                    npc, random.uniform(-20, 10)
                )
                self._tm.distance_to_leading_vehicle(
                    npc, random.uniform(1.5, 4.0)
                )
                self.npc_vehicles.append(npc)
                spawned += 1

        logger.info(f"Spawned {spawned} NPC vehicles")

    def _cleanup(self):
        """Destruye todos los actores del episodio anterior."""
        if self.camera_sensor is not None and self.camera_sensor.is_alive:
            self.camera_sensor.stop() # Detener el listen primero
            self.camera_sensor.destroy()
            self.camera_sensor = None
        self.current_image = None

        # Sensores primero (para evitar callbacks en actores muertos)
        if self.sensor_manager is not None:
            self.sensor_manager.destroy()
            self.sensor_manager = None

        # NPCs
        actors_to_destroy = [
            npc for npc in self.npc_vehicles if npc.is_alive
        ]
        if actors_to_destroy:
            # Destrucción en batch es más eficiente
            self.client.apply_batch_sync(
                [carla.command.DestroyActor(a) for a in actors_to_destroy],
                True,
            )
        self.npc_vehicles.clear()

        # Ego
        if self.ego_vehicle is not None and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

        # Tick para procesar destrucciones
        if self.synchronous and self.world is not None:
            try:
                self.world.tick()
            except Exception:
                pass
