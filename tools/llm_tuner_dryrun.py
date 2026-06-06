"""
llm_tuner_dryrun.py - Verificacion standalone del pipeline meta SIN CARLA.

Alimenta un KpiSnapshot sintetico al LLMShieldTuner + SafetyGovernor + el mapeo
de permisividad de tunable_shield, e imprime la decision resuelta. Por defecto usa
un chat_fn STUB (corre sin Ollama). Con --live llama a la Gemma real via Ollama
para verificar el round-trip y la decodificacion restringida por esquema.

USO:
    # Offline (stub, sin Ollama):
    python tools/llm_tuner_dryrun.py

    # Live (requiere `ollama serve` y el modelo pulled):
    python tools/llm_tuner_dryrun.py --live --model gemma2:2b

    # Forzar un escenario (debe disparar force_tighten):
    python tools/llm_tuner_dryrun.py --crash 0.30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.meta.kpi_snapshot import build_kpi_snapshot  # noqa: E402
from src.meta.llm_tuner import LLMShieldTuner  # noqa: E402
from src.meta.meta_controller import MetaController  # noqa: E402
from src.meta.safety_governor import GovernorConfig, SafetyGovernor  # noqa: E402
from src.meta.tunable_shield import permissiveness_to_params  # noqa: E402


def _stub_chat(*, model, messages, format, options, keep_alive):
    """Mimica ollama.chat: JSON que cumple el esquema, deterministico."""
    return {
        "message": {
            "content": json.dumps(
                {
                    "target_permissiveness": 0.9,
                    "action": "loosen",
                    "reasoning": "stub: low crash/offroad, high shielded_fraction.",
                }
            )
        }
    }


def get_args():
    p = argparse.ArgumentParser(
        description="Dry-run del meta-controlador LLM (sin CARLA)"
    )
    p.add_argument("--live", action="store_true", help="Llamar a Ollama real")
    p.add_argument("--model", default="gemma4:e4b", help="Tag del modelo Ollama")
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Forzar inferencia en CPU (num_gpu=0), como en runs con CARLA.",
    )
    p.add_argument("--episode", type=int, default=300)
    p.add_argument("--permissiveness", type=float, default=1.0)
    p.add_argument("--crash", type=float, default=0.02)
    p.add_argument("--offroad", type=float, default=0.05)
    p.add_argument("--shielded", type=float, default=0.6)
    p.add_argument(
        "--since-change",
        type=int,
        default=200,
        help="Episodios desde el ultimo cambio (para evaluar el dwell)",
    )
    return p.parse_args()


def main():
    args = get_args()
    snapshot = build_kpi_snapshot(
        episode=args.episode,
        current_permissiveness=args.permissiveness,
        crash_rate=args.crash,
        offroad_rate=args.offroad,
        stuck_rate=0.05,
        success_rate=0.7,
        shielded_fraction=args.shielded,
        shield_stats={
            "interventions_dynamic": 12,
            "interventions_static": 30,
            "interventions_pedestrian": 0,
            "interventions_recovery": 2,
            "safe_rate": 0.8,
            "warning_rate": 0.15,
            "critical_rate": 0.05,
        },
        avg_reward_100=120.0,
        mean_speed_kmh=22.0,
        num_npc=20,
    )

    chat_fn = None if args.live else _stub_chat
    tuner = LLMShieldTuner(
        model=args.model,
        chat_fn=chat_fn,
        num_gpu=0 if args.cpu else None,
    )
    if args.live:
        ok, msg = tuner.warmup()
        print(f"warmup {'OK' if ok else 'FAILED'} ({'CPU' if args.cpu else 'GPU'}): {msg}")
    governor = SafetyGovernor(GovernorConfig())
    controller = MetaController(tuner, governor, interval_episodes=100)
    # Simula que el ultimo cambio fue hace `since_change` episodios.
    controller._last_change_episode = args.episode - args.since_change

    print("=" * 72)
    print("KPI SNAPSHOT")
    print(snapshot.to_json())
    print("=" * 72)

    decision = controller.decide(snapshot, args.episode)

    print(f"LLM ok            : {decision.proposal.ok}")
    print(f"LLM action        : {decision.proposal.action}")
    print(f"LLM target        : {decision.proposal.target_permissiveness:.4f}")
    print(f"LLM reasoning     : {decision.proposal.reasoning}")
    print("-" * 72)
    print(f"Governor verdict  : {decision.governor.verdict}")
    print(f"Governor reason   : {decision.governor.reason}")
    print("-" * 72)
    print(f"current p         : {snapshot.current_permissiveness:.4f}")
    print(
        f"APPLIED p         : {decision.applied_permissiveness:.4f}  "
        f"(changed={decision.changed})"
    )
    print("-" * 72)
    print("Resolved thresholds @ applied p:")
    resolved = permissiveness_to_params(decision.applied_permissiveness)
    for name, value in resolved.items():
        print(f"  {name:42s} = {value:.4f}")
    print("=" * 72)
    if not args.live:
        print("NOTE: ran with STUB chat_fn. Pass --live to call real Gemma via Ollama.")


if __name__ == "__main__":
    main()
