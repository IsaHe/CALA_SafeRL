"""
utils.py - Utilidades para CARLA Safe RL

Reemplaza el utils.py de MetaDrive con funciones específicas de CARLA:
  - ModelManager     : gestión de modelos y checkpoints (sin cambios de API)
  - ExperimentAnalyzer: comparación entre shields con métricas CARLA
  - CarlaServerManager: helper para arrancar/detener el servidor CARLA
  - ConfigurationTemplate: plantillas listas para usar con main_train.py
"""

import os
import json
import torch
import subprocess
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# MODEL MANAGER (API compatible con versión MetaDrive)
# ══════════════════════════════════════════════════════════════════════


class ModelManager:
    """Gestión de modelos y checkpoints — misma API que la versión original."""

    MODELS_DIR = "./data/models"

    @staticmethod
    def list_models() -> Dict[str, List[str]]:
        """Lista todos los modelos disponibles agrupados por tipo de shield."""
        models: Dict[str, List[str]] = {"none": [], "basic": [], "adaptive": []}

        if not os.path.exists(ModelManager.MODELS_DIR):
            return models

        for f in os.listdir(ModelManager.MODELS_DIR):
            if not f.endswith(".pth"):
                continue
            if "none" in f:
                models["none"].append(f)
            elif "basic" in f:
                models["basic"].append(f)
            elif "adaptive" in f:
                models["adaptive"].append(f)

        return models

    @staticmethod
    def get_latest_model(shield_type: str = "adaptive") -> Optional[str]:
        """Retorna el nombre del modelo final más reciente para un shield type."""
        if not os.path.exists(ModelManager.MODELS_DIR):
            logger.warning(f"No models directory at {ModelManager.MODELS_DIR}")
            return None

        candidates = []
        for f in os.listdir(ModelManager.MODELS_DIR):
            if f.endswith(".pth") and shield_type in f and "final" in f:
                fp = os.path.join(ModelManager.MODELS_DIR, f)
                candidates.append((os.path.getmtime(fp), f))

        if not candidates:
            logger.warning(f"No {shield_type} final models found")
            return None

        candidates.sort(reverse=True)
        return candidates[0][1]

    @staticmethod
    def get_best_model(shield_type: str = "adaptive") -> Optional[str]:
        """Retorna el mejor modelo guardado para un shield type."""
        if not os.path.exists(ModelManager.MODELS_DIR):
            return None

        candidates = []
        for f in os.listdir(ModelManager.MODELS_DIR):
            if f.endswith(".pth") and shield_type in f and "best" in f:
                fp = os.path.join(ModelManager.MODELS_DIR, f)
                candidates.append((os.path.getmtime(fp), f))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    @staticmethod
    def get_checkpoint_progression(
        shield_type: str,
        model_prefix: str = "",
    ) -> List[str]:
        """Retorna los checkpoints ordenados por episodio."""
        if not os.path.exists(ModelManager.MODELS_DIR):
            return []

        checkpoints = []
        for root, dirs, files in os.walk(ModelManager.MODELS_DIR):
            for f in files:
                if "checkpoint" in f and shield_type in root:
                    if not model_prefix or model_prefix in root:
                        checkpoints.append(os.path.join(root, f))

        def ep_from_path(p: str) -> int:
            try:
                return int(p.split("checkpoint_ep_")[1].split(".pth")[0])
            except (IndexError, ValueError):
                return 0

        checkpoints.sort(key=ep_from_path)
        return checkpoints

    @staticmethod
    def print_model_info(model_path: str):
        """Imprime información de un archivo de modelo.

        Compatible con el nuevo formato de checkpoint (v2) que almacena
        un dict con claves 'policy' y 'obs_normalizer', y con el formato
        legacy que solo contenía el state_dict de la política.
        """
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return

        size_mb = os.path.getsize(model_path) / 1024 / 1024
        raw = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False,
        )

        # Detectar formato
        if isinstance(raw, dict) and "policy" in raw:
            state_dict = raw["policy"]
            has_norm = raw.get("obs_normalizer") is not None
            fmt = "v2 (policy + obs_normalizer)"
        else:
            state_dict = raw
            has_norm = False
            fmt = "legacy (solo policy)"

        print(f"\nMODEL: {os.path.basename(model_path)}")
        print("=" * 60)
        print(f"Formato:  {fmt}")
        print(f"Tamaño:   {size_mb:.2f} MB")
        print(f"Normaliz: {'incluido' if has_norm else 'no incluido'}")
        total_params = 0
        for name, tensor in state_dict.items():
            params = int(np.prod(tensor.shape))
            total_params += params
            print(f"  {name:<40} {str(tuple(tensor.shape)):<20} {params:>8,}")
        print(f"\nTotal parameters: {total_params:,}")
        print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════
# CARLA SERVER MANAGER
# ══════════════════════════════════════════════════════════════════════


class CarlaServerManager:
    """
    Helper para gestionar el servidor CARLA desde Python.

    Permite arrancar y detener el servidor automáticamente,
    útil para pipelines de CI o experimentos desatendidos.
    """

    def __init__(
        self,
        carla_root: str,
        host: str = "localhost",
        port: int = 2000,
        quality: str = "Low",
        offscreen: bool = True,
    ):
        """
        Args:
            carla_root  : Directorio raíz de CARLA (contiene CarlaUE4.sh / CarlaUE4.exe)
            host        : Host para esperar conexión
            port        : Puerto TCP de CARLA
            quality     : 'Low', 'Medium', 'Epic'
            offscreen   : True para modo headless (Linux sin pantalla)
        """
        self.carla_root = Path(carla_root)
        self.host = host
        self.port = port
        self.quality = quality
        self.offscreen = offscreen
        self._process: Optional[subprocess.Popen] = None

    def start(self, wait_timeout: float = 60.0) -> bool:
        """Arranca el servidor CARLA y espera hasta que esté disponible."""
        import platform

        if platform.system() == "Windows":
            carla_bin = self.carla_root / "CarlaUE4.exe"
        else:
            carla_bin = self.carla_root / "CarlaUE4.sh"

        if not carla_bin.exists():
            logger.error(f"CARLA binary not found: {carla_bin}")
            return False

        cmd = [str(carla_bin), f"-quality-level={self.quality}"]
        if self.offscreen and platform.system() != "Windows":
            cmd.append("-RenderOffScreen")

        logger.info(f"Starting CARLA: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return self.wait_for_connection(timeout=wait_timeout)

    def wait_for_connection(self, timeout: float = 60.0) -> bool:
        """Espera hasta que el servidor CARLA acepte conexiones."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                import carla

                client = carla.Client(self.host, self.port)
                client.set_timeout(3.0)
                client.get_server_version()
                logger.info(f"CARLA server ready at {self.host}:{self.port}")
                return True
            except Exception:
                time.sleep(2.0)

        logger.error("CARLA server did not start in time")
        return False

    def stop(self):
        """Detiene el servidor CARLA."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            logger.info("CARLA server stopped")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT ANALYZER
# ══════════════════════════════════════════════════════════════════════


class ExperimentAnalyzer:
    """Análisis comparativo entre experimentos con diferentes shields."""

    @staticmethod
    def compare_shields():
        """Imprime un resumen comparativo de los modelos disponibles."""
        models = ModelManager.list_models()

        print("\n" + "=" * 70)
        print("MODEL COMPARISON — CARLA Safe RL")
        print("=" * 70)

        for shield_type in ["none", "basic", "adaptive"]:
            files = models[shield_type]
            latest = ModelManager.get_latest_model(shield_type)
            best = ModelManager.get_best_model(shield_type)

            print(f"\n🛡️  {shield_type.upper():<10}  ({len(files)} models)")
            if latest:
                size = (
                    os.path.getsize(os.path.join(ModelManager.MODELS_DIR, latest))
                    / 1024
                    / 1024
                )
                print(f"   Latest (final): {latest}  [{size:.2f} MB]")
            if best:
                print(f"   Best:           {best}")

        print("\n" + "=" * 70 + "\n")

    @staticmethod
    def calculate_training_metrics(rewards: List[float]) -> Dict[str, float]:
        """Métricas estadísticas sobre una lista de rewards de episodios."""
        if not rewards:
            return {}
        arr = np.array(rewards)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
        }


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION TEMPLATES
# ══════════════════════════════════════════════════════════════════════


class ConfigurationTemplate:
    """Plantillas de configuración probadas para distintos escenarios CARLA.

    Actualizadas en v2 para reflejar los nuevos defaults:
      - lr: 1e-4 (era 2-3e-4) — más estable con obs normalización
      - lateral_threshold: 0.65 (era 0.82) — shield más conservador
      - update_timestep: 2048 (era 512) — buffer más grande, menos varianza
      - curriculum: activado donde corresponde para aprendizaje progresivo
    """

    # Sin shield — benchmark puro
    BASELINE = {
        "shield_type": "none",
        "max_episodes": 2500,
        "lr": 1e-4,
        "update_timestep": 2048,
        "map": "Town04",
        "num_npc": 20,
        "target_speed_kmh": 30.0,
        "success_distance": 250.0,
        "speed_weight": 0.05,
        "smoothness_weight": 0.10,
        "curriculum": True,
    }

    # Shield básico conservador (Town01-03: ciudad con más curvas)
    SAFE_URBAN = {
        "shield_type": "basic",
        "max_episodes": 2500,
        "lr": 1e-4,
        "update_timestep": 2048,
        "map": "Town03",
        "num_npc": 30,
        "target_speed_kmh": 25.0,
        "success_distance": 200.0,
        "front_threshold": 0.20,
        "side_threshold": 0.06,
        "lateral_threshold": 0.65,
        "speed_weight": 0.03,
        "smoothness_weight": 0.15,
        "curriculum": True,
    }

    # Shield adaptativo balanceado (Town04: autopista, referencia principal)
    SAFE_HIGHWAY = {
        "shield_type": "adaptive",
        "max_episodes": 2500,
        "lr": 1e-4,
        "update_timestep": 2048,
        "map": "Town04",
        "num_npc": 20,
        "target_speed_kmh": 40.0,
        "success_distance": 400.0,
        "front_threshold": 0.15,
        "side_threshold": 0.04,
        "lateral_threshold": 0.65,
        "speed_weight": 0.05,
        "smoothness_weight": 0.10,
        "curriculum": True,
    }

    # Shield adaptativo con mayor velocidad objetivo
    SAFE_AGGRESSIVE = {
        "shield_type": "adaptive",
        "max_episodes": 2500,
        "lr": 1e-4,
        "update_timestep": 2048,
        "map": "Town04",
        "num_npc": 30,
        "target_speed_kmh": 50.0,
        "success_distance": 500.0,
        "front_threshold": 0.10,
        "side_threshold": 0.03,
        "lateral_threshold": 0.65,
        "speed_weight": 0.08,
        "smoothness_weight": 0.08,
        "curriculum": True,
        "curriculum_phase1_eps": 300,
        "curriculum_phase2_eps": 1000,
    }

    @staticmethod
    def get_command(config_name: str, model_name: str) -> Optional[str]:
        """Genera el comando completo de main_train.py para una plantilla."""
        configs = {
            "baseline": ConfigurationTemplate.BASELINE,
            "safe_urban": ConfigurationTemplate.SAFE_URBAN,
            "safe_highway": ConfigurationTemplate.SAFE_HIGHWAY,
            "safe_aggressive": ConfigurationTemplate.SAFE_AGGRESSIVE,
        }

        cfg = configs.get(config_name)
        if cfg is None:
            print(f"❌ Unknown config: {config_name}")
            print(f"   Available: {list(configs.keys())}")
            return None

        parts = ["python main_train.py", f"    --model_name {model_name}"]
        for k, v in cfg.items():
            parts.append(f"    --{k} {v}")

        return " \\\n".join(parts)

    @staticmethod
    def print_all_commands(model_prefix: str = "my_model"):
        """Imprime los comandos para todas las plantillas disponibles."""
        templates = ["baseline", "safe_urban", "safe_highway", "safe_aggressive"]
        for t in templates:
            cmd = ConfigurationTemplate.get_command(t, f"{model_prefix}_{t}")
            print(f"\n# {t.upper()}")
            print(cmd)
            print()


# ══════════════════════════════════════════════════════════════════════
# PERSISTENCIA DE CONFIGURACIONES
# ══════════════════════════════════════════════════════════════════════


def save_experimental_config(filename: str, config: Dict):
    """Guarda una configuración de experimento en JSON."""
    with open(filename, "w") as f:
        json.dump(config, f, indent=4)
    print(f"✅ Config saved to {filename}")


def load_experimental_config(filename: str) -> Optional[Dict]:
    """Carga una configuración de experimento desde JSON."""
    try:
        with open(filename, "r") as f:
            config = json.load(f)
        print(f"✅ Config loaded from {filename}")
        return config
    except FileNotFoundError:
        print(f"❌ Config file not found: {filename}")
        return None


# ══════════════════════════════════════════════════════════════════════
# QUICK START GUIDE
# ══════════════════════════════════════════════════════════════════════


def print_quick_start_guide():
    print("""
╔════════════════════════════════════════════════════════════════╗
║      CARLA SAFE RL — QUICK START GUIDE  (v2)                   ║
╚════════════════════════════════════════════════════════════════╝

0. ARRANCAR SERVIDOR CARLA:

   Linux (headless):
   $ ./CarlaUE4.sh -RenderOffScreen -quality-level=Low

   Windows:
   > CarlaUE4.exe -quality-level=Low

1. ENTRENAMIENTO (configuración recomendada v2):

   Baseline sin shield + curriculum:
   $ python main_train.py --model_name baseline --shield_type none \\
       --curriculum --lr 1e-4 --update_timestep 2048

   Shield adaptativo (configuración completa recomendada):
   $ python main_train.py --model_name adaptive --shield_type adaptive \\
       --curriculum --lr 1e-4 --update_timestep 2048 \\
       --lateral_threshold 0.65 --kl_target 0.05

   Desactivar normalización de obs (para debug):
   $ python main_train.py --model_name debug --shield_type adaptive \\
       --no_obs_norm

2. EVALUACION:

   $ python main_eval.py --model_name adaptive_shield_adaptive_final.pth

   Sin render (solo métricas):
   $ python main_eval.py --model_name ... --no_render --episodes 20

3. MONITORIZACION:

   TensorBoard:
   $ tensorboard --logdir ./runs

4. GESTION DE MODELOS:

   from utils import ModelManager
   ModelManager.list_models()
   latest = ModelManager.get_latest_model('adaptive')
   ModelManager.print_model_info(f'./data/models/{latest}')

5. COMPARACION DE SHIELDS:

   from utils import ExperimentAnalyzer
   ExperimentAnalyzer.compare_shields()

6. CONFIGURACIONES PREDEFINIDAS:

   from utils import ConfigurationTemplate
   ConfigurationTemplate.print_all_commands('mi_proyecto')

╔════════════════════════════════════════════════════════════════╗
║  CAMBIOS V2 (por qué estos flags importan):                    ║
║  --curriculum        : empieza sin NPCs, aprende lane-keeping  ║
║  --lr 1e-4           : más estable con obs normalización       ║
║  --update_timestep 2048 : reduce varianza del buffer PPO       ║
║  --lateral_threshold 0.65 : shield interviene antes del borde  ║
║  --kl_target 0.05    : evita actualizaciones destructivas      ║
╠════════════════════════════════════════════════════════════════╣
║  MAPAS RECOMENDADOS:                                           ║
║  Town04 -> autopista (benchmark principal, empezar aqui)       ║
║  Town01/02/03 -> ciudad (mas complejo)                         ║
║  Town05 -> cruce grande (maxima dificultad)                    ║
╚════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    print_quick_start_guide()
    print("\nModelos disponibles:")
    ExperimentAnalyzer.compare_shields()
    print("\nPlantillas de configuración:")
    ConfigurationTemplate.print_all_commands("demo")
