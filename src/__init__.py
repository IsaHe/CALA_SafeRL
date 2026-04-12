"""
src/ - Safe RL for Autonomous Driving in CARLA

Exports principales:
    CarlaEnv                    → entorno Gymnasium sobre CARLA
    CarlaRewardShaper           → reward shaping con Waypoint API
    CarlaSafetyShield           → shield básico (LIDAR + carril)
    CarlaAdaptiveHorizonShield  → shield adaptativo con BicycleModel
    PPOAgent                    → agente PPO (sin cambios respecto al original)
    SafetyMetrics               → métricas de seguridad
    SafetyMetricsReporter       → generador de reportes
"""

from src.CARLA.Env.carla_env import CarlaEnv
from src.CARLA.Sensors.carla_sensors import SensorManager
from src.reward_shaper import CarlaRewardShaper
from src.safety_shield import CarlaSafetyShield
from src.Adaptative_Shield.adaptive_horizon_shield import CarlaAdaptiveHorizonShield
from src.PPO.ppo_agent import PPOAgent
from src.Metrics.EvalMetrics.metrics import SafetyMetrics, SafetyMetricsReporter

__all__ = [
    "CarlaEnv",
    "SensorManager",
    "CarlaRewardShaper",
    "CarlaSafetyShield",
    "CarlaAdaptiveHorizonShield",
    "PPOAgent",
    "SafetyMetrics",
    "SafetyMetricsReporter",
]
