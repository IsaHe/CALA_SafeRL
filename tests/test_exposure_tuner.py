"""
test_exposure_tuner.py - Curriculum de EXPOSICIÓN a tráfico (fase 4).

Sin CARLA ni Ollama: mapeo permisividad↔nº de leads, el adaptador
TunableRouteExposure escribe en el env base, el SafetyGovernor CORTA tráfico ante
un breach de crash, y el LLMExposureTuner usa su prompt propio + fallback no-op.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meta.kpi_snapshot import build_kpi_snapshot  # noqa: E402
from src.meta.llm_exposure_tuner import (  # noqa: E402
    EXPOSURE_SYSTEM_PROMPT,
    LLMExposureTuner,
)
from src.meta.llm_tuner import SYSTEM_PROMPT  # noqa: E402
from src.meta.safety_governor import GovernorConfig, SafetyGovernor  # noqa: E402
from src.meta.tunable_exposure import (  # noqa: E402
    TunableRouteExposure,
    exposure_to_route_npcs,
)


def _snapshot(**ov):
    base = dict(
        episode=300,
        current_permissiveness=0.5,
        crash_rate=0.05,
        offroad_rate=0.05,
        stuck_rate=0.0,
        success_rate=0.6,
        shielded_fraction=0.5,
        route_npc_count=3,
        encounter_rate=0.1,
    )
    base.update(ov)
    return build_kpi_snapshot(**base)


class _FakeBaseEnv:
    def __init__(self):
        self.route_npc_count = -1  # será sobrescrito por el adaptador en __init__


def _chat_returning(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)

    def _chat(*, model, messages, format, options, keep_alive):
        return {"message": {"content": content, "_system": messages[0]["content"]}}

    return _chat


# ── mapeo p ↔ nº de leads ─────────────────────────────────────────────────────


def test_exposure_mapping_endpoints_and_monotone():
    assert exposure_to_route_npcs(1.0, 6) == 0  # strict = sin tráfico
    assert exposure_to_route_npcs(0.0, 6) == 6  # permisivo = tope
    vals = [exposure_to_route_npcs(p, 6) for p in (1.0, 0.75, 0.5, 0.25, 0.0)]
    assert vals == sorted(vals)  # menos p ⇒ más tráfico
    assert all(0 <= v <= 6 for v in vals)


# ── adaptador ──────────────────────────────────────────────────────────────────


def test_adapter_syncs_env_and_round_trips():
    base = _FakeBaseEnv()
    adapter = TunableRouteExposure(base, max_route_npcs=6)
    # init: p=1.0 ⇒ 0 leads escritos en el env.
    assert adapter.get_permissiveness() == 1.0
    assert base.route_npc_count == 0

    adapter.set_permissiveness(0.0)  # máxima exposición
    assert base.route_npc_count == 6
    assert adapter.route_npc_count == 6

    adapter.set_permissiveness(0.5)
    assert base.route_npc_count == 3


# ── governor: un breach de crash CORTA tráfico ────────────────────────────────


def test_governor_crash_breach_cuts_traffic():
    gov = SafetyGovernor(GovernorConfig(crash_ceiling=0.20, crash_target=0.12))
    snap = _snapshot(crash_rate=0.30)  # > ceiling
    d = gov.govern(current_p=0.4, proposed_p=0.0, kpi=snap, episodes_since_change=999)
    assert d.verdict == "force_tighten"
    assert d.applied_permissiveness > 0.4  # p sube
    # p mayor ⇒ MENOS leads: el tráfico se corta.
    assert exposure_to_route_npcs(d.applied_permissiveness, 6) < exposure_to_route_npcs(
        0.4, 6
    )


# ── LLMExposureTuner ──────────────────────────────────────────────────────────


def test_exposure_tuner_uses_its_own_prompt():
    t = LLMExposureTuner(chat_fn=_chat_returning({}))
    assert t._system_prompt() == EXPOSURE_SYSTEM_PROMPT
    assert t._system_prompt() != SYSTEM_PROMPT


def test_exposure_tuner_valid_json_applied():
    t = LLMExposureTuner(
        chat_fn=_chat_returning(
            {"target_permissiveness": 0.3, "action": "loosen", "reasoning": "low enc"}
        )
    )
    prop = t.propose(_snapshot(current_permissiveness=0.5))
    assert prop.ok is True
    assert prop.target_permissiveness == pytest.approx(0.3)
    assert prop.action == "loosen"


def test_exposure_tuner_prompt_includes_encounter_rate():
    captured = {}

    def _chat(*, model, messages, format, options, keep_alive):
        captured["user"] = messages[1]["content"]
        return {
            "message": {
                "content": json.dumps(
                    {"target_permissiveness": 0.5, "action": "hold", "reasoning": "x"}
                )
            }
        }

    LLMExposureTuner(chat_fn=_chat).propose(_snapshot(encounter_rate=0.123))
    assert "encounter_rate" in captured["user"]
    assert "0.123" in captured["user"]


def test_exposure_tuner_malformed_falls_back():
    t = LLMExposureTuner(chat_fn=_chat_returning("not json {"))
    prop = t.propose(_snapshot(current_permissiveness=0.5))
    assert prop.ok is False
    assert prop.action == "hold"
    assert prop.target_permissiveness == pytest.approx(0.5)


# ── KpiSnapshot lleva los campos de exposición ────────────────────────────────


def test_snapshot_carries_exposure_fields():
    snap = _snapshot(route_npc_count=4, encounter_rate=0.25)
    assert snap.route_npc_count == 4
    assert snap.encounter_rate == pytest.approx(0.25)
    d = json.loads(snap.to_json())
    assert d["route_npc_count"] == 4
    assert d["encounter_rate"] == pytest.approx(0.25)
