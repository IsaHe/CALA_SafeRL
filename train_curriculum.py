"""
train_curriculum.py - Curriculum multi-fase para PPO + Adaptive Shield en CARLA

Diseño
======
4 fases que rampean simultáneamente tres dimensiones, cada una continuando el
aprendizaje de la anterior vía --load_model:

    Fase | success_distance | num_npc | lr     | entropy_coef | episodios
    -----+------------------+---------+--------+--------------+----------
      1  |   100 m          |    0    | 1e-4   |    0.001     |   2000
      2  |   150 m          |   10    | 5e-5   |    0.0       |   2000
      3  |   200 m          |   20    | 2e-5   |    0.0       |   2000
      4  |   250 m          |   40    | 1e-5   |    0.0       |   2000

Uso
===
    # Pre-requisito: servidor CARLA corriendo + venv activado
    python train_curriculum.py --base_name myrun

    # Reanudar desde una fase específica (útil si la 2 se cortó)
    python train_curriculum.py --base_name myrun --start_phase 2

    # Override budget (rápido para iterar el diseño)
    python train_curriculum.py --base_name quicktest --episodes_per_phase 500

Salida
======
- ./data/models/{base_name}_p{i}_adaptive_final.pth  por fase
- ./runs/{base_name}_p{i}_adaptive_<timestamp>/      con metrics.sqlite
- Cada fase carga el _final.pth de la anterior vía --load_model
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] curriculum: %(message)s",
)
logger = logging.getLogger("curriculum")


@dataclass(frozen=True)
class Phase:
    """Configuración de una fase del curriculum.

    spawn_tier: clave del perfil de spawns. Valores válidos:
        "easy"           → solo spawns clasificados easy
        "easy+medium"    → easy ∪ medium
        "all"            → todos los spawns del mapa (None → CarlaEnv random)

    El perfil se genera con `profile_spawns.py --map <map>` y se guarda en
    `spawn_profiles/<map>.json`. Si no existe, el curriculum cae a None
    (random sobre todos los spawns) con un aviso.
    """

    idx: int
    success_distance: float
    num_npc: int
    lr: float
    entropy_coef: float
    episodes: int
    spawn_tier: str
    description: str


# Diseño del curriculum. Se referencia por --start_phase / --end_phase.
# spawn_tier rampea de "easy" (rectas curadas) a "all" (todos los spawns del
# mapa) para suavizar la curva de aprendizaje geométrica complementando el
# ramp de success_distance y num_npc.
DEFAULT_PHASES: list[Phase] = [
    Phase(
        idx=1,
        success_distance=100.0,
        num_npc=0,
        lr=1e-4,
        entropy_coef=0.001,
        spawn_tier="easy",
        episodes=300,
        description="Foundations — lane-keep 100 m en rectas, sin tráfico",
    ),
    Phase(
        idx=2,
        success_distance=150.0,
        num_npc=10,
        lr=5e-5,
        entropy_coef=0.0,
        episodes=400,
        spawn_tier="easy+medium",
        description="Extension — 150 m + curvas suaves, tráfico ligero",
    ),
    Phase(
        idx=3,
        success_distance=200.0,
        num_npc=20,
        lr=2e-5,
        entropy_coef=0.0,
        spawn_tier="all",
        episodes=700,
        description="Consolidation — 200 m + todos los spawns, tráfico moderado",
    ),
    Phase(
        idx=4,
        success_distance=250.0,
        num_npc=40,
        lr=1e-5,
        entropy_coef=0.0,
        episodes=1000,
        spawn_tier="all",
        description="Final — 250 m + tráfico denso, política preservada (lr fino)",
    ),
]


def load_spawn_profile(map_name: str, profile_path: Path | None) -> dict | None:
    """Carga el JSON producido por profile_spawns.py.

    Si no se encuentra, devuelve None y deja que el caller decida si
    continuar (con random spawn) o abortar.
    """
    resolved = (
        profile_path
        if profile_path is not None
        else Path("./spawn_profiles") / f"{map_name}.json"
    )
    if not resolved.is_file():
        return None
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"No se pudo leer {resolved}: {exc}")
        return None


def resolve_spawn_indices(profile: dict | None, tier: str) -> list[int] | None:
    """Convierte una clave de tier (ej. 'easy+medium') a una lista de
    índices de spawn points usando el perfil cargado.

    Devuelve None si:
      - profile es None (no hay perfil disponible)
      - tier es 'all' o '*' (no filtrar — CarlaEnv random sobre todos)

    Para tiers compuestos ('easy+medium'), une los conjuntos.
    """
    if profile is None:
        return None
    tier_norm = tier.strip().lower()
    if tier_norm in {"all", "*", ""}:
        return None
    tiers_data = profile.get("tiers", {})
    indices: set[int] = set()
    for key in tier_norm.split("+"):
        key = key.strip()
        bucket = tiers_data.get(key, [])
        indices.update(int(i) for i in bucket)
    if not indices:
        logger.warning(
            f"Tier '{tier}' no produjo índices en el perfil. Fallback a random."
        )
        return None
    return sorted(indices)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Curriculum multi-fase para PPO + Adaptive Shield en CARLA"
    )
    p.add_argument(
        "--base_name",
        type=str,
        required=True,
        help="Nombre base para los modelos. Cada fase guarda como "
        "<base_name>_p{i}_adaptive_final.pth en ./data/models/",
    )
    p.add_argument(
        "--episodes_per_phase",
        type=int,
        default=2000,
        help="Episodios por fase. Default 2000 → presupuesto total 8000 eps "
        "(~8-12 h de wall-clock).",
    )
    p.add_argument(
        "--start_phase",
        type=int,
        default=1,
        help="Fase desde la que arrancar (1-4). Si > 1, carga el _final.pth "
        "de la fase anterior. Útil para reanudar tras una interrupción.",
    )
    p.add_argument(
        "--end_phase",
        type=int,
        default=4,
        help="Fase final inclusive (1-4). Para terminar antes y revisar "
        "métricas intermedias.",
    )
    p.add_argument(
        "--shield_type",
        type=str,
        choices=["none", "basic", "adaptive"],
        default="adaptive",
        help="Tipo de shield para todas las fases.",
    )
    p.add_argument(
        "--map",
        type=str,
        default="Town04",
        help="Mapa CARLA. Se mantiene constante entre fases.",
    )
    p.add_argument(
        "--host", type=str, default="localhost", help="Host del servidor CARLA"
    )
    p.add_argument("--port", type=int, default=2000, help="Puerto del servidor CARLA")
    p.add_argument(
        "--tm_port", type=int, default=8000, help="Puerto del TrafficManager"
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Steps máximos por episodio (común a todas las fases).",
    )
    p.add_argument(
        "--ckpt_freq",
        type=int,
        default=200,
        help="Frecuencia de checkpoint intra-fase (eps).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Imprime los comandos de cada fase sin ejecutarlos. Útil para "
        "verificar el diseño antes de comprometer 8h de entrenamiento.",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Intérprete Python (default: el actual). Útil si el venv difiere.",
    )
    p.add_argument(
        "--spawn_profile",
        type=str,
        default=None,
        help="Ruta al JSON producido por profile_spawns.py. Default: "
        "./spawn_profiles/<map>.json. Si no existe, se usa random sobre "
        "todos los spawns (con aviso).",
    )
    return p.parse_args()


def build_cmd(
    phase: Phase,
    args: argparse.Namespace,
    load_model_path: Path | None,
    spawn_indices: list[int] | None,
) -> list[str]:
    """Construye el argv para una invocación a main_train.py."""
    phase_model_name = f"{args.base_name}_p{phase.idx}"
    cmd: list[str] = [
        args.python,
        "main_train.py",
        "--model_name",
        phase_model_name,
        "--shield_type",
        args.shield_type,
        "--map",
        args.map,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tm_port",
        str(args.tm_port),
        "--max_episodes",
        str(phase.episodes),
        "--max_steps",
        str(args.max_steps),
        "--ckpt_freq",
        str(args.ckpt_freq),
        # Dimensiones del curriculum
        "--success_distance",
        str(phase.success_distance),
        "--num_npc",
        str(phase.num_npc),
        "--lr",
        str(phase.lr),
        "--entropy_coef",
        str(phase.entropy_coef),
        # Sin CurriculumManager interno: el orquestador controla NPCs y spawn
        # explícitamente por fase.
        "--no-curriculum",
    ]
    if load_model_path is not None:
        cmd += ["--load_model", str(load_model_path)]
    if spawn_indices:
        cmd += ["--spawn_point_indices", ",".join(str(i) for i in spawn_indices)]
    return cmd


def run_phase(
    phase: Phase,
    args: argparse.Namespace,
    load_model_path: Path | None,
    spawn_indices: list[int] | None,
) -> Path:
    """Ejecuta una fase del curriculum y devuelve la ruta del modelo final.

    Aborta si el proceso de entrenamiento sale con código distinto de 0.
    """
    cmd = build_cmd(phase, args, load_model_path, spawn_indices)
    out_model = (
        Path("./data/models")
        / f"{args.base_name}_p{phase.idx}_{args.shield_type}_final.pth"
    )

    logger.info("═" * 78)
    logger.info(f"▶ FASE {phase.idx}: {phase.description}")
    logger.info(
        f"  success_distance={phase.success_distance} m | "
        f"num_npc={phase.num_npc} | lr={phase.lr} | "
        f"entropy_coef={phase.entropy_coef} | episodios={phase.episodes}"
    )
    if spawn_indices is not None:
        logger.info(
            f"  spawn_tier='{phase.spawn_tier}' ({len(spawn_indices)} spawns: "
            f"{spawn_indices[:10]}{'…' if len(spawn_indices) > 10 else ''})"
        )
    else:
        logger.info(
            f"  spawn_tier='{phase.spawn_tier}' → SIN filtro (random sobre todos)"
        )
    if load_model_path:
        logger.info(f"  warm-start ← {load_model_path}")
    else:
        logger.info("  warm-start ← (none, cold start)")
    logger.info(f"  output → {out_model}")
    logger.info("─" * 78)
    logger.info("Comando:\n  " + " \\\n    ".join(cmd))
    logger.info("─" * 78)

    if args.dry_run:
        logger.info("[dry-run] skipping execution")
        return out_model

    t_start = time.time()
    # Streaming stdout/stderr en tiempo real (sin capturar) para ver progreso
    # del entrenamiento. cwd es el directorio del script (raíz del repo).
    cwd = Path(__file__).resolve().parent
    result = subprocess.run(cmd, cwd=cwd)
    dur_min = (time.time() - t_start) / 60.0

    if result.returncode != 0:
        logger.error(
            f"❌ Fase {phase.idx} falló (exit code {result.returncode}) "
            f"tras {dur_min:.1f} min. Abortando curriculum."
        )
        sys.exit(result.returncode)

    if not out_model.is_file():
        logger.error(
            f"❌ Fase {phase.idx} completó con exit 0 pero NO se encontró "
            f"el modelo final esperado en {out_model}. ¿Renombraste algo?"
        )
        sys.exit(2)

    logger.info(f"✅ Fase {phase.idx} OK en {dur_min:.1f} min → {out_model}")
    return out_model


def main():
    args = get_args()

    if args.start_phase < 1 or args.start_phase > len(DEFAULT_PHASES):
        logger.error(f"--start_phase debe estar en [1, {len(DEFAULT_PHASES)}]")
        sys.exit(1)
    if args.end_phase < args.start_phase or args.end_phase > len(DEFAULT_PHASES):
        logger.error(
            f"--end_phase debe estar en [{args.start_phase}, {len(DEFAULT_PHASES)}]"
        )
        sys.exit(1)

    phases_to_run = DEFAULT_PHASES[args.start_phase - 1 : args.end_phase]

    # Resolver la ruta de carga inicial: si arrancamos en fase >1, cargamos
    # el _final.pth de la fase anterior. Si arrancamos en fase 1, cold start.
    if args.start_phase > 1:
        prev_phase_idx = args.start_phase - 1
        load_path = (
            Path("./data/models")
            / f"{args.base_name}_p{prev_phase_idx}_{args.shield_type}_final.pth"
        )
        if not load_path.is_file():
            logger.error(
                f"--start_phase={args.start_phase} requiere checkpoint en "
                f"{load_path}, pero no existe. Verifica que la fase {prev_phase_idx} "
                f"se completó. Para empezar de cero, usa --start_phase 1."
            )
            sys.exit(1)
    else:
        load_path = None

    # Cargar perfil de spawns para resolver el tier de cada fase. Si no
    # existe, avisar pero seguir (CarlaEnv hará random sobre todos los
    # spawns del mapa, equivalente al comportamiento pre-curriculum).
    spawn_profile_path = Path(args.spawn_profile) if args.spawn_profile else None
    spawn_profile = load_spawn_profile(args.map, spawn_profile_path)
    if spawn_profile is None:
        logger.warning(
            f"No se encontró perfil de spawns para mapa '{args.map}'. "
            f"Genera uno con `python profile_spawns.py --map {args.map}` "
            f"para activar el curriculum por geometría. Continuando con "
            f"spawn random sobre TODOS los puntos del mapa en TODAS las fases."
        )

    logger.info("CURRICULUM CONFIGURADO")
    logger.info(f"  base_name             = {args.base_name}")
    logger.info(f"  fases                 = {args.start_phase} → {args.end_phase}")
    logger.info(f"  episodios por fase    = {args.episodes_per_phase}")
    logger.info(
        f"  presupuesto total     = {args.episodes_per_phase * len(phases_to_run)} eps"
    )
    logger.info(f"  shield                = {args.shield_type}")
    logger.info(f"  mapa                  = {args.map}")
    logger.info(f"  CARLA                 = {args.host}:{args.port}")
    if spawn_profile is not None:
        counts = spawn_profile.get("tier_counts", {})
        logger.info(
            f"  spawn profile         = {counts.get('easy', 0)} easy / "
            f"{counts.get('medium', 0)} medium / {counts.get('hard', 0)} hard"
        )
    if args.dry_run:
        logger.info("  *** DRY RUN — no se ejecutará entrenamiento ***")

    t_global = time.time()
    for phase in phases_to_run:
        spawn_indices = resolve_spawn_indices(spawn_profile, phase.spawn_tier)
        load_path = run_phase(phase, args, load_path, spawn_indices)

    dur_h = (time.time() - t_global) / 3600.0
    logger.info("═" * 78)
    logger.info(f"✅ CURRICULUM COMPLETO en {dur_h:.2f} h")
    logger.info(f"   Modelo final: {load_path}")
    logger.info("═" * 78)
    logger.info("Para evaluar el modelo final:")
    logger.info(
        f"   python main_eval.py --model_name {load_path.name} "
        f"--shield_type {args.shield_type} --map {args.map}"
    )
    logger.info("Para inspeccionar curvas:")
    logger.info("   streamlit run dataVisualizer.py")


if __name__ == "__main__":
    main()
