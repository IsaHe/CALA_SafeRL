"""
test_preference_annotator.py - Anotador de preferencias de seguridad (fase 1).

Verifican, SIN servidor Ollama (ollama.chat inyectado), las mismas garantias que
el tuner del shield pero para una salida de PREFERENCIA:
  - JSON valido -> safer/confidence aplicados; confidence clampada,
  - malformed / clave faltante / extra (forbid) / tipo erroneo / safer fuera de
    {0,1} / conexion -> fallback ABSTENCION (ok=False, safer=-1),
  - opciones deterministas + schema restringido pasados a ollama,
  - annotate_symmetric: acepta si (A,B) y (B,A) coinciden, se abstiene si no.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meta.preference_annotator import (  # noqa: E402
    PreferenceAnnotator,
    PreferenceResponse,
)


def _chat_returning(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)

    def _chat(*, model, messages, format, options, keep_alive):
        return {"message": {"content": content}}

    return _chat


def _chat_sequence(payloads):
    """chat_fn que devuelve payloads sucesivos (para annotate_symmetric)."""
    contents = [p if isinstance(p, str) else json.dumps(p) for p in payloads]
    calls = {"i": 0}

    def _chat(*, model, messages, format, options, keep_alive):
        c = contents[min(calls["i"], len(contents) - 1)]
        calls["i"] += 1
        return {"message": {"content": c}}

    return _chat


# ── annotate (direccional) ───────────────────────────────────────────────────


def test_valid_preference_a_safer():
    ann = PreferenceAnnotator(
        chat_fn=_chat_returning({"safer": 0, "confidence": 0.9, "reasoning": "A clear"})
    )
    lbl = ann.annotate("cap A", "cap B")
    assert lbl.ok is True
    assert lbl.safer == 0
    assert lbl.confidence == pytest.approx(0.9)


def test_valid_preference_b_safer():
    ann = PreferenceAnnotator(
        chat_fn=_chat_returning({"safer": 1, "confidence": 0.6, "reasoning": "B"})
    )
    assert ann.annotate("a", "b").safer == 1


def test_confidence_clamped():
    ann = PreferenceAnnotator(
        chat_fn=_chat_returning({"safer": 0, "confidence": 1.7, "reasoning": "x"})
    )
    lbl = ann.annotate("a", "b")
    assert lbl.ok is True
    assert lbl.confidence == 1.0


def test_malformed_json_abstains():
    ann = PreferenceAnnotator(chat_fn=_chat_returning("not json {"))
    lbl = ann.annotate("a", "b")
    assert lbl.ok is False
    assert lbl.safer == -1


def test_missing_key_abstains():
    ann = PreferenceAnnotator(
        chat_fn=_chat_returning({"confidence": 0.5, "reasoning": "no safer"})
    )
    assert ann.annotate("a", "b").ok is False


def test_extra_key_rejected_by_forbid():
    ann = PreferenceAnnotator(
        chat_fn=_chat_returning(
            {"safer": 0, "confidence": 0.5, "reasoning": "x", "evil": 1}
        )
    )
    assert ann.annotate("a", "b").ok is False


def test_safer_out_of_domain_abstains():
    # safer=2 no esta en Literal[0,1] -> validacion falla -> abstencion.
    ann = PreferenceAnnotator(
        chat_fn=_chat_returning({"safer": 2, "confidence": 0.5, "reasoning": "x"})
    )
    assert ann.annotate("a", "b").ok is False


def test_connection_error_abstains():
    def _boom(**kwargs):
        raise ConnectionError("ollama down")

    lbl = PreferenceAnnotator(chat_fn=_boom).annotate("a", "b")
    assert lbl.ok is False
    assert lbl.safer == -1


def test_passes_deterministic_options_and_schema():
    captured = {}

    def _chat(*, model, messages, format, options, keep_alive):
        captured.update(options=options, fmt=format, keep_alive=keep_alive)
        return {
            "message": {
                "content": json.dumps({"safer": 0, "confidence": 0.5, "reasoning": "x"})
            }
        }

    PreferenceAnnotator(seed=7, chat_fn=_chat).annotate("a", "b")
    assert captured["options"]["temperature"] == 0.0
    assert captured["options"]["seed"] == 7
    assert captured["keep_alive"] == 0
    assert captured["fmt"] == PreferenceResponse.model_json_schema()


# ── annotate_symmetric (anti sesgo de posicion) ──────────────────────────────


def test_symmetric_consistent_a_safer():
    # (A,B) dice A (safer=0); (B,A) dice A (safer=1, porque A va segundo).
    ann = PreferenceAnnotator(
        chat_fn=_chat_sequence(
            [
                {"safer": 0, "confidence": 0.8, "reasoning": "A"},
                {"safer": 1, "confidence": 0.6, "reasoning": "A again"},
            ]
        )
    )
    lbl = ann.annotate_symmetric("cap A", "cap B")
    assert lbl.ok is True
    assert lbl.safer == 0
    assert lbl.confidence == pytest.approx(0.7)  # media


def test_symmetric_consistent_b_safer():
    # (A,B) dice B (safer=1); (B,A) dice B (safer=0).
    ann = PreferenceAnnotator(
        chat_fn=_chat_sequence(
            [
                {"safer": 1, "confidence": 0.9, "reasoning": "B"},
                {"safer": 0, "confidence": 0.9, "reasoning": "B again"},
            ]
        )
    )
    lbl = ann.annotate_symmetric("cap A", "cap B")
    assert lbl.ok is True
    assert lbl.safer == 1


def test_symmetric_position_bias_disagreement_abstains():
    # (A,B) dice A (0) pero (B,A) tambien dice "primero" (0) => incoherente.
    ann = PreferenceAnnotator(
        chat_fn=_chat_sequence(
            [
                {"safer": 0, "confidence": 0.9, "reasoning": "first"},
                {"safer": 0, "confidence": 0.9, "reasoning": "first"},
            ]
        )
    )
    lbl = ann.annotate_symmetric("cap A", "cap B")
    assert lbl.ok is False
    assert lbl.safer == -1


def test_symmetric_abstains_if_one_direction_fails():
    ann = PreferenceAnnotator(
        chat_fn=_chat_sequence(
            [{"safer": 0, "confidence": 0.9, "reasoning": "ok"}, "garbage {"]
        )
    )
    assert ann.annotate_symmetric("a", "b").ok is False
