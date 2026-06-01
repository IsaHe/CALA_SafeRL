import argparse
import math
from collections import deque
from types import SimpleNamespace

import carla
import numpy as np

from main_eval import build_env
from src.PPO.ppo_agent import PPOAgent
from src.Adaptative_Shield.BicycleModel import BicycleModel


def find_attr(env, name):
    e = env
    while e is not None:
        if hasattr(e, name) and getattr(e, name) is not None:
            return getattr(e, name)
        e = getattr(e, "env", None)
    return None


def lateral_in_frame(wp, px, py):
    right = wp.transform.get_right_vector()
    dx = px - wp.transform.location.x
    dy = py - wp.transform.location.y
    return dx * right.x + dy * right.y


def calibrate(model, ego):
    physics = ego.get_physics_control()
    wheels = physics.wheels[:2]
    nominal = math.radians(max(float(w.max_steer_angle) for w in wheels))
    curve = getattr(physics, "steering_curve", None)
    pts = (
        sorted((float(p.x), float(np.clip(p.y, 0.0, 1.0))) for p in curve)
        if curve
        else [(0.0, 1.0)]
    )
    model.set_max_steer_rad(nominal)
    return nominal, pts


def curve_mult(pts, speed_ms):
    if speed_ms <= pts[0][0]:
        return pts[0][1]
    if speed_ms >= pts[-1][0]:
        return pts[-1][1]
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        if speed_ms <= x1:
            span = max(x1 - x0, 1e-6)
            return y0 + (y1 - y0) * (speed_ms - x0) / span
    return pts[-1][1]


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--shield_type", type=str, default="adaptive")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--use_curve", action="store_true")
    cli = ap.parse_args()

    args = SimpleNamespace(
        host="localhost",
        port=2000,
        tm_port=8000,
        map="Town04",
        num_npc=0,
        weather="ClearNoon",
        target_speed_kmh=30.0,
        success_distance=250.0,
        max_steps=1000,
        shield_type=cli.shield_type,
        front_threshold=0.15,
        side_threshold=0.02,
        lateral_threshold=0.65,
        idle_penalty_weight=0.0,
        progress_reward_weight=0.0,
        acceleration_reward_weight=0.0,
    )

    env, _ = build_env(args, render=True)
    carla_map = find_attr(env, "map")

    agent = None
    if cli.model:
        agent = PPOAgent(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            normalize_obs=True,
        )
        agent.load(cli.model)
        agent.policy.eval()

    probe = BicycleModel()
    nominal, pts = None, None
    window = deque(maxlen=20)

    for ep in range(1, cli.episodes + 1):
        obs, _ = env.reset()
        ego = find_attr(env, "ego_vehicle")
        nominal, pts = calibrate(probe, ego)
        window.clear()
        done = truncated = False

        while not (done or truncated):
            if agent is not None:
                action, _, _, _ = agent.select_action(obs, deterministic=True)
            else:
                action = np.array(
                    [np.random.uniform(-0.3, 0.3), 0.66], dtype=np.float32
                )

            tf0 = ego.get_transform()
            vel0 = ego.get_velocity()
            x0, y0 = tf0.location.x, tf0.location.y
            yaw0 = math.radians(tf0.rotation.yaw)
            speed0 = math.sqrt(vel0.x**2 + vel0.y**2)
            wp0 = carla_map.get_waypoint(
                tf0.location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            lat0 = lateral_in_frame(wp0, x0, y0)
            lane_tag = f"{wp0.road_id}:{wp0.lane_id}" if wp0 is not None else "-:-"

            obs, _, done, truncated, info = env.step(action)
            env.render()

            executed = np.asarray(info.get("executed_action", action), dtype=np.float32)
            alpha = float(info.get("shield_intensity", 0.0))
            risk = info.get("risk_level", "-")

            if cli.use_curve:
                probe.set_max_steer_rad(nominal * curve_mult(pts, speed0))
            else:
                probe.set_max_steer_rad(nominal)
            traj = probe.predict_trajectory(
                x0, y0, yaw0, speed0, float(executed[0]), float(executed[1]), 1
            )
            px, py, _ = traj[-1]
            lat_pred = lateral_in_frame(wp0, px, py)

            tf1 = ego.get_transform()
            lat_real = lateral_in_frame(wp0, tf1.location.x, tf1.location.y)

            d_pred = lat_pred - lat0
            d_real = lat_real - lat0
            fresh = bool(info.get("semantic_data_fresh", True))
            row = (
                f"spd={speed0 * 3.6:5.1f} lat={info.get('lateral_offset_norm', 0.0):+5.2f} "
                f"hdg={info.get('heading_error', 0.0):+6.1f} steer={executed[0]:+5.2f} "
                f"a={alpha:4.2f} risk={risk:<8} lane={lane_tag:>7} dLatPred={d_pred:+5.3f} dLatReal={d_real:+5.3f} "
                f"fresh={'Y' if fresh else 'N'}"
            )
            window.append(row)
            print(row)

            if info.get("out_of_road", False):
                print("\n=== OFFROAD POST-MORTEM (last 20 steps) ===")
                for r in window:
                    print(r)
                print("===========================================\n")

    env.close()


if __name__ == "__main__":
    run()
