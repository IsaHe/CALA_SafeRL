"""
debug_route_spawn.py - Validación EN VIVO del tráfico de exposición (fase 4).
REQUIERE un servidor CARLA en marcha. No entrena, no usa LLM ni agente.

Construye un CarlaEnv con ``route_npc`` leads en ruta (y opcionalmente NPCs
globales), resetea, y comprueba que los leads:
  - SE SPAWNEAN (last_route_npcs_spawned == route_npc),
  - están DELANTE del ego (bearing ≈ 0°, no ±180°),
  - van LENTOS y en km/h (no 36-72 km/h, que indicaría unidades m/s),
  - son VISIBLES al LIDAR dinámico (front_dynamic_m baja al avanzar).

Firmas de fallo (memoria):
  spawneados=0          → mecánica de spawn (z-lift / waypoints) rota.
  bearings ±180°        → bug de dirección de waypoints (se spawnean detrás).
  velocidades 36-72 km/h→ set_desired_speed interpretó m/s; dividir por 3.6.

    python debug_route_spawn.py --route_npc 4 --num_npc 0 --steps 200 --episodes 3
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from src.CARLA.Env.carla_env import CarlaEnv
from src.meta.scene_caption import extract_scene_features


def _bearing_deg(ego, other) -> float:
    et = ego.get_transform()
    el = et.location
    ol = other.get_location()
    ang_world = math.degrees(math.atan2(ol.y - el.y, ol.x - el.x))
    rel = (ang_world - et.rotation.yaw + 180.0) % 360.0 - 180.0
    return rel  # ≈0 delante, ±180 detrás


def _follow_action(env) -> np.ndarray:
    """Pure-pursuit mínimo: el ego SIGUE su carril (steer hacia el waypoint a 6 m
    en su sentido de marcha) para no salirse en las curvas de Town04 y poder
    ALCANZAR los leads lentos. Sin esto, un throttle recto se sale del carril
    antes de encontrarlos y la validación daría un falso negativo."""
    tf = env.ego_vehicle.get_transform()
    loc, fwd = tf.location, tf.get_forward_vector()
    wp = env.map.get_waypoint(loc, project_to_road=True)
    if wp is None:
        return np.array([0.0, 0.3], dtype=np.float32)
    lane_fwd = wp.transform.get_forward_vector()
    use_next = (fwd.x * lane_fwd.x + fwd.y * lane_fwd.y) >= 0.0
    chain = wp.next(6.0) if use_next else wp.previous(6.0)
    if not chain:
        return np.array([0.0, 0.3], dtype=np.float32)
    tgt = chain[0].transform.location
    desired = math.degrees(math.atan2(tgt.y - loc.y, tgt.x - loc.x))
    err = (desired - tf.rotation.yaw + 180.0) % 360.0 - 180.0
    # Throttle alto: con 0.4 el ego se estancaba (~0.1 km/h) y nunca alcanzaba a
    # los leads. Reducimos el gas sólo si hay que girar fuerte (curvas cerradas).
    steer = float(np.clip(err / 30.0, -1.0, 1.0))
    throttle = 0.75 if abs(steer) < 0.5 else 0.5
    return np.array([steer, throttle], dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route_npc", type=int, default=4)
    p.add_argument("--num_npc", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm_port", type=int, default=8000)
    p.add_argument("--map", default="Town04")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--lidar_channels", type=int, default=3)
    args = p.parse_args()

    env = CarlaEnv(
        host=args.host,
        port=args.port,
        tm_port=args.tm_port,
        map_name=args.map,
        num_npc_vehicles=args.num_npc,
        synchronous=True,
        fixed_delta_seconds=0.05,
        max_episode_steps=args.steps,
        seed=args.seed,
        route_npc_count=args.route_npc,
        lidar_channels=args.lidar_channels,
    )

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            spawned = env.last_route_npcs_spawned
            leads = env.npc_vehicles[-spawned:] if spawned else []
            print(f"\n=== Episode {ep + 1} ===")
            print(f"requested={args.route_npc}  spawned={spawned}")
            if spawned:
                env.world.tick() if hasattr(env.world, "tick") else None
                for k, lead in enumerate(leads):
                    if not lead.is_alive:
                        continue
                    d = env.ego_vehicle.get_location().distance(lead.get_location())
                    b = _bearing_deg(env.ego_vehicle, lead)
                    v = lead.get_velocity()
                    spd = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
                    print(
                        f"  lead {k}: dist={d:5.1f} m  bearing={b:+6.1f}°  speed={spd:4.1f} km/h"
                    )

            min_front = float("inf")
            detected_steps = 0
            for _ in range(args.steps):
                obs, _, done, trunc, _ = env.step(_follow_action(env))
                fd = extract_scene_features(obs).front_dynamic_m
                if np.isfinite(fd):
                    min_front = min(min_front, fd)
                    detected_steps += 1
                if done or trunc:
                    break

            print(
                f"  min front_dynamic over episode = "
                f"{min_front if np.isfinite(min_front) else float('nan'):.1f} m  "
                f"({detected_steps} steps with a vehicle in the front cone)"
            )
            verdict = (
                "SPAWN FAIL (0 leads)"
                if spawned == 0
                else "OK"
                if (detected_steps > 0 and np.isfinite(min_front))
                else "leads spawned but never seen by LIDAR — check bearing/speed above"
            )
            print(f"  VERDICT: {verdict}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
