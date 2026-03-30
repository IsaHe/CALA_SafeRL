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

from src.carla_env import CarlaEnv
from src.carla_sensors import SensorManager
from src.reward_shaper import CarlaRewardShaper
from src.safety_shield import CarlaSafetyShield
from src.adaptive_horizon_shield import CarlaAdaptiveHorizonShield
from src.ppo_agent import PPOAgent
from src.metrics import SafetyMetrics, SafetyMetricsReporter

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
