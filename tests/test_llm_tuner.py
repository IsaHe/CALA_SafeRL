"""
test_llm_tuner.py - Tests del pipeline del meta-controlador LLM (Phase 0).

Verifican, SIN CARLA ni servidor Ollama, las garantias del destete del shield:
  - tunable_shield: mapeo permisividad->umbrales acotado y monotono.
  - kpi_snapshot:   ensamblado correcto desde valores del bucle.
  - safety_governor: suelo de seguridad, ratchet, paso acotado, dwell.
  - llm_tuner:      JSON valido / clamp / fallback no-op (ollama.chat inyectado).
  - meta_controller: cadencia, decide end-to-end, apply sobre shield falso.

La unica seccion que requiere `carla` (integracion real del shield) usa
importorskip, igual que test_shield_deadlock_recovery.py.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meta.kpi_snapshot import KpiSnapshot, build_kpi_snapshot  # noqa: E402
from src.meta.llm_tuner import (  # noqa: E402
    LLMShieldTuner,
    ShieldTuningResponse,
    build_user_prompt,
)
from src.meta.meta_controller import MetaController  # noqa: E402
from src.meta.safety_governor import GovernorConfig, SafetyGovernor  # noqa: E402
from src.meta.tunable_shield import (  # noqa: E402
    WEANABLE_ENDPOINTS,
    clamp_permissiveness,
    permissiveness_to_params,
)


# ── helpers ────────────────────────────────────────────────────────────────


def _snapshot(**overrides) -> KpiSnapshot:
    base = dict(
        episode=300,
        current_permissiveness=1.0,
        crash_rate=0.02,
        offroad_rate=0.04,
        stuck_rate=0.03,
        success_rate=0.7,
        shielded_fraction=0.6,
    )
    base.update(overrides)
    return build_kpi_snapshot(**base)


def _chat_returning(payload):
    """Devuelve un chat_fn que responde `payload` (str cruda o dict->json)."""
    content = payload if isinstance(payload, str) else json.dumps(payload)

    def _chat(*, model, messages, format, options, keep_alive):
        return {"message": {"content": content}}

    return _chat


# ── tunable_shield ──────────────────────────────────────────────────────────


def test_permissiveness_endpoints_strict_and_permissive():
    strict = permissiveness_to_params(1.0)
    permissive = permissiveness_to_params(0.0)
    for name, (s, p) in WEANABLE_ENDPOINTS.items():
        assert strict[name] == pytest.approx(s)
        assert permissive[name] == pytest.approx(p)


def test_permissiveness_midpoint_is_average():
    mid = permissiveness_to_params(0.5)
    for name, (s, p) in WEANABLE_ENDPOINTS.items():
        assert mid[name] == pytest.approx((s + p) / 2.0)


def test_permissiveness_clamped_outside_unit_interval():
    assert clamp_permissiveness(1.7) == 1.0
    assert clamp_permissiveness(-0.3) == 0.0
    # p>1 resuelve igual que p=1 (strict); p<0 igual que p=0 (permissive).
    assert permissiveness_to_params(5.0) == pytest.approx(permissiveness_to_params(1.0))
    assert permissiveness_to_params(-5.0) == pytest.approx(
        permissiveness_to_params(0.0)
    )


def test_front_threshold_loosens_monotonically_and_bounded():
    # front_threshold_base baja con la permisividad (menor = mas permisivo) y
    # nunca sale del intervalo [0.08, 0.15].
    vals = [
        permissiveness_to_params(p)["front_threshold_base"]
        for p in (1.0, 0.75, 0.5, 0.25, 0.0)
    ]
    assert vals == sorted(vals, reverse=True)
    assert all(0.08 - 1e-9 <= v <= 0.15 + 1e-9 for v in vals)


def test_lateral_offset_loosens_upward_and_bounded():
    # LATERAL_CRITICAL_OFFSET sube con la permisividad (mayor = mas permisivo).
    vals = [
        permissiveness_to_params(p)["LATERAL_CRITICAL_OFFSET"] for p in (1.0, 0.5, 0.0)
    ]
    assert vals == sorted(vals)
    assert all(0.70 - 1e-9 <= v <= 0.90 + 1e-9 for v in vals)


# ── kpi_snapshot ────────────────────────────────────────────────────────────


def test_build_kpi_snapshot_reads_shield_stats():
    snap = _snapshot(
        shield_stats={
            "interventions_dynamic": 11,
            "interventions_static": 22,
            "interventions_pedestrian": 1,
            "recovery_activations": 4,
            "safe_rate": 0.7,
            "warning_rate": 0.2,
            "critical_rate": 0.1,
        }
    )
    assert snap.intervention_dynamic == 11
    assert snap.intervention_static == 22
    assert snap.intervention_pedestrian == 1
    assert snap.intervention_recovery == 4  # cae a recovery_activations
    assert snap.safe_rate == pytest.approx(0.7)


def test_build_kpi_snapshot_defaults_without_stats():
    snap = _snapshot()  # sin shield_stats
    assert snap.intervention_dynamic == 0.0
    assert snap.critical_rate == 0.0
    # round-trip json
    d = json.loads(snap.to_json())
    assert d["crash_rate"] == pytest.approx(snap.crash_rate)


# ── safety_governor ─────────────────────────────────────────────────────────


def _gov() -> SafetyGovernor:
    return SafetyGovernor(GovernorConfig())


def test_governor_force_tighten_on_crash_breach_overrides_loosen():
    gov = _gov()
    snap = _snapshot(crash_rate=0.30)  # > ceiling 0.10
    # incluso si el LLM pide aflojar al maximo:
    d = gov.govern(current_p=0.8, proposed_p=0.0, kpi=snap, episodes_since_change=999)
    assert d.verdict == "force_tighten"
    assert d.applied_permissiveness == pytest.approx(min(1.0, 0.8 + 0.20))


def test_governor_force_tighten_on_offroad_breach():
    gov = _gov()
    snap = _snapshot(offroad_rate=0.30)  # > ceiling 0.15
    d = gov.govern(current_p=0.9, proposed_p=0.5, kpi=snap, episodes_since_change=999)
    assert d.verdict == "force_tighten"
    assert d.applied_permissiveness > 0.9


def test_governor_holds_during_dwell():
    gov = _gov()
    snap = _snapshot(crash_rate=0.0, offroad_rate=0.0)
    d = gov.govern(current_p=0.8, proposed_p=0.6, kpi=snap, episodes_since_change=10)
    assert d.verdict == "hold_dwell"
    assert d.applied_permissiveness == pytest.approx(0.8)


def test_governor_holds_when_not_under_targets():
    gov = _gov()
    # crash bajo techo (0.10) pero sobre target (0.05) -> no aflojar.
    snap = _snapshot(crash_rate=0.07, offroad_rate=0.0)
    d = gov.govern(current_p=0.8, proposed_p=0.6, kpi=snap, episodes_since_change=999)
    assert d.verdict == "hold_unsafe"
    assert d.applied_permissiveness == pytest.approx(0.8)


def test_governor_loosens_one_bounded_step_when_safe():
    gov = _gov()
    snap = _snapshot(crash_rate=0.0, offroad_rate=0.0)
    # LLM pide saltar a 0.0 pero el paso esta acotado a max_step=0.10.
    d = gov.govern(current_p=0.8, proposed_p=0.0, kpi=snap, episodes_since_change=999)
    assert d.verdict == "loosen"
    assert d.applied_permissiveness == pytest.approx(0.7)  # 0.8 - 0.10


def test_governor_tighten_allowed_immediately_without_dwell():
    gov = _gov()
    snap = _snapshot(crash_rate=0.0, offroad_rate=0.0)
    d = gov.govern(current_p=0.7, proposed_p=0.95, kpi=snap, episodes_since_change=1)
    assert d.verdict == "tighten"
    assert d.applied_permissiveness == pytest.approx(0.8)  # 0.7 + max_step 0.10


def test_governor_never_loosens_past_evidence_then_steps_down():
    gov = _gov()
    p = 1.0
    # fase insegura (crash sobre target): p se mantiene.
    for _ in range(3):
        d = gov.govern(p, 0.0, _snapshot(crash_rate=0.09), 999)
        assert d.applied_permissiveness == pytest.approx(p)
        p = d.applied_permissiveness
    # fase segura: baja un paso acotado por decision.
    d = gov.govern(p, 0.0, _snapshot(crash_rate=0.0, offroad_rate=0.0), 999)
    assert d.applied_permissiveness == pytest.approx(0.9)


# ── llm_tuner ───────────────────────────────────────────────────────────────


def test_tuner_valid_json_applied():
    tuner = LLMShieldTuner(
        chat_fn=_chat_returning(
            {"target_permissiveness": 0.85, "action": "loosen", "reasoning": "ok"}
        )
    )
    prop = tuner.propose(_snapshot(current_permissiveness=0.95))
    assert prop.ok is True
    assert prop.target_permissiveness == pytest.approx(0.85)
    assert prop.action == "loosen"


def test_tuner_out_of_range_target_clamped():
    tuner = LLMShieldTuner(
        chat_fn=_chat_returning(
            {"target_permissiveness": 1.7, "action": "tighten", "reasoning": "x"}
        )
    )
    prop = tuner.propose(_snapshot())
    assert prop.ok is True
    assert prop.target_permissiveness == 1.0  # clamped


def test_tuner_malformed_json_falls_back_to_hold():
    tuner = LLMShieldTuner(chat_fn=_chat_returning("not json at all {"))
    prop = tuner.propose(_snapshot(current_permissiveness=0.8))
    assert prop.ok is False
    assert prop.action == "hold"
    assert prop.target_permissiveness == pytest.approx(0.8)  # no-op = current


def test_tuner_missing_key_falls_back():
    tuner = LLMShieldTuner(
        chat_fn=_chat_returning({"action": "loosen", "reasoning": "no target"})
    )
    prop = tuner.propose(_snapshot(current_permissiveness=0.8))
    assert prop.ok is False
    assert prop.target_permissiveness == pytest.approx(0.8)


def test_tuner_extra_key_rejected_by_forbid():
    tuner = LLMShieldTuner(
        chat_fn=_chat_returning(
            {
                "target_permissiveness": 0.5,
                "action": "loosen",
                "reasoning": "x",
                "evil": 1,
            }
        )
    )
    prop = tuner.propose(_snapshot(current_permissiveness=0.8))
    assert prop.ok is False
    assert prop.target_permissiveness == pytest.approx(0.8)


def test_tuner_wrong_type_falls_back():
    tuner = LLMShieldTuner(
        chat_fn=_chat_returning(
            {"target_permissiveness": "high", "action": "loosen", "reasoning": "x"}
        )
    )
    prop = tuner.propose(_snapshot(current_permissiveness=0.8))
    assert prop.ok is False


def test_tuner_connection_error_falls_back():
    def _boom(**kwargs):
        raise ConnectionError("ollama down")

    tuner = LLMShieldTuner(chat_fn=_boom)
    prop = tuner.propose(_snapshot(current_permissiveness=0.8))
    assert prop.ok is False
    assert prop.action == "hold"
    assert prop.target_permissiveness == pytest.approx(0.8)


def test_tuner_passes_deterministic_options_and_schema():
    captured = {}

    def _chat(*, model, messages, format, options, keep_alive):
        captured.update(options=options, keep_alive=keep_alive, fmt=format)
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

    tuner = LLMShieldTuner(seed=123, chat_fn=_chat)
    tuner.propose(_snapshot())
    assert captured["options"]["temperature"] == 0.0
    assert captured["options"]["seed"] == 123
    assert captured["keep_alive"] == 0
    # format es el JSON schema del contrato (decodificacion restringida).
    assert captured["fmt"] == ShieldTuningResponse.model_json_schema()


def test_tuner_num_gpu_in_options_when_set():
    captured = {}

    def _chat(*, model, messages, format, options, keep_alive):
        captured["options"] = dict(options)
        return {
            "message": {
                "content": json.dumps(
                    {"target_permissiveness": 0.9, "action": "hold", "reasoning": "x"}
                )
            }
        }

    # num_gpu=0 (CPU) se propaga a options.
    LLMShieldTuner(num_gpu=0, chat_fn=_chat).propose(_snapshot())
    assert captured["options"]["num_gpu"] == 0
    # Default (None) NO añade num_gpu (deja decidir a Ollama).
    captured.clear()
    LLMShieldTuner(chat_fn=_chat).propose(_snapshot())
    assert "num_gpu" not in captured["options"]


def test_tuner_warmup_ok_and_failure():
    ok_tuner = LLMShieldTuner(
        chat_fn=_chat_returning(
            {"target_permissiveness": 1.0, "action": "hold", "reasoning": "x"}
        )
    )
    ok, _ = ok_tuner.warmup()
    assert ok is True

    def _boom(**kwargs):
        raise ConnectionError("down")

    bad, bmsg = LLMShieldTuner(chat_fn=_boom).warmup()
    assert bad is False
    assert "ConnectionError" in bmsg


def test_user_prompt_contains_kpis():
    snap = _snapshot(crash_rate=0.123)
    text = build_user_prompt(snap)
    assert "permissiveness" in text.lower()
    assert "0.123" in text


# ── meta_controller ─────────────────────────────────────────────────────────


class _FakeShield:
    """Shield duck-typed para testear el controlador sin CARLA."""

    def __init__(self, p=1.0):
        self._p = p
        self.applied = []

    def get_permissiveness(self):
        return self._p

    def set_permissiveness(self, p):
        self._p = p
        self.applied.append(p)
        return {"front_threshold_base": p}


def _controller(chat_payload, **gov_overrides):
    tuner = LLMShieldTuner(chat_fn=_chat_returning(chat_payload))
    gov = SafetyGovernor(GovernorConfig(**gov_overrides))
    return MetaController(tuner, gov, interval_episodes=100)


def test_controller_should_run_cadence():
    ctrl = _controller(
        {"target_permissiveness": 1.0, "action": "hold", "reasoning": "x"}
    )
    assert ctrl.should_run(100) is True
    assert ctrl.should_run(250) is False
    assert ctrl.should_run(0) is False


def test_controller_decide_applies_governed_value():
    ctrl = _controller(
        {"target_permissiveness": 0.0, "action": "loosen", "reasoning": "safe"}
    )
    snap = _snapshot(current_permissiveness=1.0, crash_rate=0.0, offroad_rate=0.0)
    dec = ctrl.decide(snap, episode=300)  # since_change=300 >= dwell 100
    assert dec.governor.verdict == "loosen"
    assert dec.applied_permissiveness == pytest.approx(0.9)  # paso acotado
    assert dec.changed is True


def test_controller_apply_mutates_fake_shield():
    ctrl = _controller(
        {"target_permissiveness": 0.0, "action": "loosen", "reasoning": "safe"}
    )
    shield = _FakeShield(p=1.0)
    snap = _snapshot(current_permissiveness=1.0, crash_rate=0.0, offroad_rate=0.0)
    dec = ctrl.decide(snap, episode=300)
    ctrl.apply(shield, dec)
    assert shield.get_permissiveness() == pytest.approx(0.9)


def test_controller_dwell_tracks_last_change():
    ctrl = _controller(
        {"target_permissiveness": 0.0, "action": "loosen", "reasoning": "safe"}
    )
    snap = _snapshot(current_permissiveness=1.0, crash_rate=0.0, offroad_rate=0.0)
    d1 = ctrl.decide(snap, episode=300)
    assert d1.changed is True
    # justo despues, otra decision con since-change pequeno -> hold_dwell.
    snap2 = _snapshot(
        current_permissiveness=d1.applied_permissiveness,
        crash_rate=0.0,
        offroad_rate=0.0,
    )
    d2 = ctrl.decide(snap2, episode=320)  # since=20 < 100
    assert d2.governor.verdict == "hold_dwell"
    assert d2.changed is False


def test_controller_audit_record_has_keys():
    ctrl = _controller(
        {"target_permissiveness": 1.0, "action": "hold", "reasoning": "x"}
    )
    dec = ctrl.decide(_snapshot(), episode=100)
    rec = dec.audit_record()
    for k in (
        "episode",
        "applied_permissiveness",
        "governor_verdict",
        "llm_action",
        "crash_rate",
        "kpi_json",
    ):
        assert k in rec


# ── integracion real del shield (requiere `carla` importable) ───────────────


def test_shield_set_permissiveness_applies_endpoints():
    pytest.importorskip("carla")
    from src.Adaptative_Shield.adaptive_horizon_shield import (
        CarlaAdaptiveHorizonShield as Shield,
    )

    # object.__new__ evita necesitar un servidor CARLA (igual que el test de
    # deadlock). set_permissiveness solo usa self.__dict__, copy y tunable_shield.
    sh = object.__new__(Shield)
    sh._permissiveness = 1.0

    sh.set_permissiveness(1.0)  # strict = baseline de produccion
    assert sh.front_threshold_base == pytest.approx(0.15)
    assert sh.HORIZON_CONFIG["warning"]["threshold_multiplier"] == pytest.approx(1.5)
    assert sh.LATERAL_CRITICAL_OFFSET == pytest.approx(0.70)

    sh.set_permissiveness(0.0)  # permissive = maximo destete
    assert sh.front_threshold_base == pytest.approx(0.08)
    assert sh.HORIZON_CONFIG["warning"]["threshold_multiplier"] == pytest.approx(1.0)
    assert sh.LATERAL_CRITICAL_OFFSET == pytest.approx(0.90)

    # NO muta el dict de CLASE compartido entre instancias.
    assert Shield.HORIZON_CONFIG["warning"]["threshold_multiplier"] == pytest.approx(
        1.5
    )
    assert sh.get_permissiveness() == pytest.approx(0.0)
