"""
test_mode_b_intrarun.py - Tests del Modo B (intra-run) del meta-controlador.

Sin CARLA ni Ollama: el AsyncMetaRunner (worker single-flight en background) y la
persistencia meta_decisions de LiveMetricsLogger.
"""

import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Metrics.live_metrics import (  # noqa: E402
    LiveMetricsLogger,
    load_meta_decisions,
)
from src.meta.kpi_snapshot import build_kpi_snapshot  # noqa: E402
from src.meta.llm_tuner import LLMShieldTuner  # noqa: E402
from src.meta.meta_controller import AsyncMetaRunner, MetaController  # noqa: E402
from src.meta.safety_governor import GovernorConfig, SafetyGovernor  # noqa: E402


def _snapshot(p=1.0, crash=0.0, offroad=0.0):
    return build_kpi_snapshot(
        episode=100,
        current_permissiveness=p,
        crash_rate=crash,
        offroad_rate=offroad,
        stuck_rate=0.0,
        success_rate=0.8,
        shielded_fraction=0.7,
    )


def _chat_loosen(*, model, messages, format, options, keep_alive):
    return {
        "message": {
            "content": json.dumps(
                {"target_permissiveness": 0.8, "action": "loosen", "reasoning": "safe"}
            )
        }
    }


def _runner(chat_fn):
    ctrl = MetaController(
        LLMShieldTuner(chat_fn=chat_fn),
        SafetyGovernor(GovernorConfig()),
        interval_episodes=100,
    )
    return AsyncMetaRunner(ctrl)


def _wait(runner, timeout=5.0):
    t0 = time.time()
    while runner.busy() and time.time() - t0 < timeout:
        time.sleep(0.005)


def test_async_runner_submit_poll():
    runner = _runner(_chat_loosen)
    try:
        assert runner.submit_if_idle(_snapshot(), 100) is True
        _wait(runner)
        dec = runner.poll()
        assert dec is not None
        assert dec.governor.verdict == "loosen"
        assert dec.applied_permissiveness == pytest.approx(0.9)  # bounded from 1.0
        assert runner.poll() is None  # ya consumida
    finally:
        runner.shutdown()


def test_async_runner_single_flight():
    # chat_fn que bloquea hasta soltarlo, para observar el estado busy.
    gate = threading.Event()

    def _slow_chat(*, model, messages, format, options, keep_alive):
        gate.wait(2.0)
        return {
            "message": {
                "content": json.dumps(
                    {
                        "target_permissiveness": 0.9,
                        "action": "hold",
                        "reasoning": "x",
                    }
                )
            }
        }

    runner = _runner(_slow_chat)
    try:
        assert runner.submit_if_idle(_snapshot(), 100) is True
        assert runner.busy() is True
        # ocupado -> segundo submit rechazado (single-flight).
        assert runner.submit_if_idle(_snapshot(), 200) is False
        gate.set()
        _wait(runner)
        assert runner.poll() is not None
    finally:
        gate.set()
        runner.shutdown()


def test_log_and_load_meta_decision(tmp_path):
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
    ctrl = MetaController(
        LLMShieldTuner(chat_fn=_chat_loosen),
        SafetyGovernor(GovernorConfig()),
        interval_episodes=100,
    )
    dec = ctrl.decide(_snapshot(), episode=100)
    lg.log_meta_decision(dec.audit_record())
    lg.close(status="finished")

    df = load_meta_decisions(db, "r")
    assert len(df) == 1
    row = df.iloc[0]
    assert int(row["episode"]) == 100
    assert row["governor_verdict"] == "loosen"
    assert row["applied_permissiveness"] == pytest.approx(0.9)
    assert int(row["llm_ok"]) == 1


def test_load_meta_decisions_absent_table_returns_empty(tmp_path):
    # DB sin tabla meta_decisions (simula run antiguo): devuelve df vacio.
    import sqlite3

    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE foo (x INTEGER)")
    conn.close()
    assert load_meta_decisions(db, "r").empty
