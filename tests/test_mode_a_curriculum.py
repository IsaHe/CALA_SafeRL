"""
test_mode_a_curriculum.py - Tests del Modo A (entre fases) del meta-controlador.

Sin CARLA ni Ollama: construye una metrics.sqlite sintetica con LiveMetricsLogger
y verifica que db_kpis la lee bien, que build_cmd de train_curriculum inyecta
--shield_permissiveness, y la cadena DB -> snapshot -> governor end-to-end.
"""

import argparse
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Metrics.live_metrics import LiveMetricsLogger  # noqa: E402
from src.meta.db_kpis import read_kpi_snapshot_from_db  # noqa: E402


def _make_db(path, n=5, crash=0.05, offroad=0.08, shielded=0.5):
    lg = LiveMetricsLogger(
        db_path=path,
        run_name="r",
        model_name="m",
        shield_type="adaptive",
        map_name="Town04",
        max_episodes=10,
        max_steps=100,
        update_timestep=2048,
    )
    for ep in range(1, n + 1):
        lg.log_metrics(
            axis="episode",
            step=ep,
            metrics={
                "Training/Crash_Rate": crash,
                "Training/Offroad_Rate": offroad,
                "Training/Success_Rate": 0.6,
                "Outcome/Stuck": 0.0,
                "Safety/Shield_Rate": shielded,
                "CARLA/Mean_Speed_kmh": 20.0,
                "Safety/Semantic/Dynamic_Interventions": 10.0,
                "Safety/Semantic/Static_Interventions": 25.0,
                "Safety/Semantic/Safe_Step_Rate": 0.8,
                "Reward/Average_100_Episodes": 100.0,
            },
        )
    lg.close(status="finished")


def test_read_kpi_snapshot_from_db(tmp_path):
    db = tmp_path / "metrics.sqlite"
    _make_db(db)
    snap = read_kpi_snapshot_from_db(db, "r", current_permissiveness=0.9, num_npc=10)
    assert snap is not None
    assert snap.crash_rate == pytest.approx(0.05)
    assert snap.offroad_rate == pytest.approx(0.08)
    assert snap.success_rate == pytest.approx(0.6)
    assert snap.shielded_fraction == pytest.approx(0.5)
    assert snap.intervention_dynamic == pytest.approx(10.0)
    assert snap.intervention_static == pytest.approx(25.0)
    assert snap.safe_rate == pytest.approx(0.8)
    assert snap.current_permissiveness == pytest.approx(0.9)
    assert snap.num_npc == 10
    assert snap.avg_reward_100 == pytest.approx(100.0)


def test_read_kpi_snapshot_empty_db_returns_none(tmp_path):
    db = tmp_path / "metrics.sqlite"
    lg = LiveMetricsLogger(
        db_path=db,
        run_name="r",
        model_name="m",
        shield_type="adaptive",
        map_name="Town04",
        max_episodes=10,
        max_steps=100,
        update_timestep=2048,
    )
    lg.close(status="finished")  # sin episodios
    assert read_kpi_snapshot_from_db(db, "r", current_permissiveness=1.0) is None


def test_build_cmd_includes_shield_permissiveness():
    import train_curriculum as tc

    args = argparse.Namespace(
        base_name="x",
        shield_type="adaptive",
        map="Town04",
        host="localhost",
        port=2000,
        tm_port=8000,
        max_steps=1000,
        ckpt_freq=200,
        python="python",
    )
    cmd = tc.build_cmd(tc.DEFAULT_PHASES[0], args, None, None, 0.85)
    assert "--shield_permissiveness" in cmd
    assert cmd[cmd.index("--shield_permissiveness") + 1] == "0.8500"
    # Default 1.0 cuando no se pasa.
    cmd_default = tc.build_cmd(tc.DEFAULT_PHASES[0], args, None, None)
    assert cmd_default[cmd_default.index("--shield_permissiveness") + 1] == "1.0000"


def test_end_to_end_db_to_decision(tmp_path):
    """Cadena Modo A completa sin CARLA ni Ollama: DB -> snapshot -> governor."""
    from src.meta.llm_tuner import LLMShieldTuner
    from src.meta.meta_controller import MetaController
    from src.meta.safety_governor import GovernorConfig, SafetyGovernor

    db = tmp_path / "metrics.sqlite"
    _make_db(db, crash=0.0, offroad=0.0, shielded=0.7)  # seguro -> aflojar
    snap = read_kpi_snapshot_from_db(db, "r", current_permissiveness=1.0, num_npc=10)

    def _chat(*, model, messages, format, options, keep_alive):
        return {
            "message": {
                "content": json.dumps(
                    {
                        "target_permissiveness": 0.85,
                        "action": "loosen",
                        "reasoning": "safe",
                    }
                )
            }
        }

    ctrl = MetaController(
        LLMShieldTuner(chat_fn=_chat),
        SafetyGovernor(GovernorConfig()),
    )
    dec = ctrl.decide(snap, episode=300)  # dwell satisfecho
    assert dec.governor.verdict == "loosen"
    assert dec.applied_permissiveness == pytest.approx(0.9)  # 1.0 - max_step
    assert dec.changed is True
