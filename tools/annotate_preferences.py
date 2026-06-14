"""
annotate_preferences.py - Anotacion OFFLINE de preferencias de seguridad (fase 1).

Lee un dataset de observaciones (``observations.npz`` con un array ``obs`` de
forma [N, 739], producido por ``tools/collect_annotation_rollouts.py``), forma
pares de escenas, le pide a un LLM local (via ``PreferenceAnnotator``) cual de
cada par es la MAS SEGURA, y guarda:

  - ``preferences.npz``  : obs_a [M,739], obs_b [M,739], safer [M], confidence [M]
                           (M <= n_pairs; los pares con abstencion o baja confianza
                           se descartan). Es el dataset que la fase 2 entrena con
                           perdida Bradley-Terry.
  - ``annotations.jsonl``: auditoria legible (captions + safer + reasoning) de TODOS
                           los pares, incluidos los descartados.

Para que la preferencia mida COLISION (no "conduce bonito"):
  - solo se emparejan frames EN MOVIMIENTO (``--moving_only``), de modo que el
    riesgo de colision no se confunda con el estado 'parado' (un coche parado con
    la via despejada tiene riesgo de colision casi nulo, pero un anotador debil
    tiende a marcarlo como inseguro);
  - se SOBREMUESTREAN los pares con un obstaculo dinamico EN TRAYECTORIA (cono
    frontal, ``is_front_encounter``), que es la senal que de verdad informa la
    evitacion frontal -- un vehiculo de lado/atras no choca.

100% offline: no necesita CARLA. Necesita Ollama solo para la anotacion real;
las funciones puras (``select_pairs``, ``is_front_encounter``, ``run_annotation``)
son testeables con un anotador mock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meta.preference_annotator import PreferenceAnnotator  # noqa: E402
from src.meta.scene_caption import caption_from_obs, extract_scene_features  # noqa: E402


def is_encounter(obs: np.ndarray, max_m: float = 35.0) -> bool:
    """True si hay un obstaculo dinamico (vehiculo/peaton) a <= max_m en CUALQUIER
    direccion. Util para reportar exposicion; NO para sobremuestrear colision."""
    f = extract_scene_features(obs)
    return bool(np.isfinite(f.nearest_dynamic_m) and f.nearest_dynamic_m <= max_m)


def is_front_encounter(obs: np.ndarray, max_m: float = 25.0) -> bool:
    """True si hay un obstaculo dinamico EN TRAYECTORIA (cono frontal) a <= max_m.

    Es la senal relevante para la colision: un vehiculo de lado/atras no choca. Se
    usa para sobremuestrear los pares que de verdad informan la evitacion frontal.
    """
    f = extract_scene_features(obs)
    return bool(np.isfinite(f.front_dynamic_m) and f.front_dynamic_m <= max_m)


def scene_speed_kmh(obs: np.ndarray) -> float:
    return extract_scene_features(obs).speed_kmh


def select_pairs(
    eligible_idx: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
    oversample_idx: np.ndarray,
    oversample: float = 0.5,
) -> list[tuple[int, int]]:
    """Muestrea ``n_pairs`` pares (i, j) distintos con indices GLOBALES.

    Ambos endpoints se toman de ``eligible_idx`` (p.ej. solo frames en movimiento,
    para que el riesgo de colision no se confunda con el estado 'parado'). Con
    probabilidad ``oversample`` el primer endpoint se fuerza a ``oversample_idx``
    (encuentros frontales), enriqueciendo la region rara que importa.
    """
    eligible = np.asarray(eligible_idx)
    if eligible.size < 2:
        return []
    os_idx = np.asarray(oversample_idx)
    has_os = os_idx.size > 0
    pairs: list[tuple[int, int]] = []
    for _ in range(n_pairs):
        if has_os and rng.random() < oversample:
            i = int(rng.choice(os_idx))
        else:
            i = int(rng.choice(eligible))
        j = int(rng.choice(eligible))
        while j == i:
            j = int(rng.choice(eligible))
        pairs.append((i, j))
    return pairs


def run_annotation(
    obs: np.ndarray,
    pairs: list[tuple[int, int]],
    annotator: PreferenceAnnotator,
    *,
    symmetric: bool = True,
    min_confidence: float = 0.0,
    progress_every: int = 0,
) -> tuple[dict, list[dict]]:
    """Anota cada par y devuelve (dataset, registros de auditoria).

    Los pares con abstencion (``ok=False``) o ``confidence < min_confidence`` se
    omiten del dataset pero SI aparecen en la auditoria.
    """
    obs_a: list[np.ndarray] = []
    obs_b: list[np.ndarray] = []
    safer: list[int] = []
    confidence: list[float] = []
    audit: list[dict] = []

    for k, (i, j) in enumerate(pairs):
        cap_a = caption_from_obs(obs[i])
        cap_b = caption_from_obs(obs[j])
        lbl = (
            annotator.annotate_symmetric(cap_a, cap_b)
            if symmetric
            else annotator.annotate(cap_a, cap_b)
        )
        audit.append(
            {
                "i": int(i),
                "j": int(j),
                "caption_a": cap_a,
                "caption_b": cap_b,
                "safer": int(lbl.safer),
                "confidence": float(lbl.confidence),
                "ok": bool(lbl.ok),
                "reasoning": lbl.reasoning,
            }
        )
        if progress_every and (k + 1) % progress_every == 0:
            kept = len(safer)
            print(f"  annotated {k + 1}/{len(pairs)} pairs, kept {kept}", flush=True)

        if (not lbl.ok) or lbl.confidence < min_confidence:
            continue
        obs_a.append(obs[i])
        obs_b.append(obs[j])
        safer.append(int(lbl.safer))
        confidence.append(float(lbl.confidence))

    dataset = {
        "obs_a": np.asarray(obs_a, dtype=np.float32).reshape(-1, obs.shape[1]),
        "obs_b": np.asarray(obs_b, dtype=np.float32).reshape(-1, obs.shape[1]),
        "safer": np.asarray(safer, dtype=np.int8),
        "confidence": np.asarray(confidence, dtype=np.float32),
    }
    return dataset, audit


def _load_observations(path: Path) -> np.ndarray:
    data = np.load(path)
    key = "obs" if "obs" in data.files else data.files[0]
    obs = np.asarray(data[key], dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError(f"expected obs of shape [N, D], got {obs.shape}")
    return obs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", required=True, help="observations.npz de entrada")
    p.add_argument("--out", default="data/preferences/preferences.npz")
    p.add_argument("--audit", default="data/preferences/annotations.jsonl")
    p.add_argument("--n_pairs", type=int, default=4000)
    p.add_argument("--oversample", type=float, default=0.5)
    p.add_argument(
        "--encounter_max_m",
        type=float,
        default=25.0,
        help="umbral (m) de obstaculo EN TRAYECTORIA para sobremuestrear colision",
    )
    p.add_argument(
        "--moving_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="solo emparejar frames en movimiento (evita confundir 'parado' con seguro)",
    )
    p.add_argument("--min_speed_kmh", type=float, default=5.0)
    p.add_argument("--min_confidence", type=float, default=0.0)
    p.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--limit", type=int, default=0, help="cap de observaciones (0=todas)"
    )
    p.add_argument("--seed", type=int, default=42)
    # LLM (offline: GPU libre, sin contencion con CARLA).
    p.add_argument("--model", default="gemma4:e4b")
    p.add_argument("--host", default=None)
    p.add_argument("--num_gpu", type=int, default=None)
    p.add_argument("--keep_alive", default=0)
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args()

    obs = _load_observations(Path(args.observations))
    if args.limit and obs.shape[0] > args.limit:
        rng0 = np.random.default_rng(args.seed)
        obs = obs[rng0.choice(obs.shape[0], args.limit, replace=False)]
    print(f"Loaded {obs.shape[0]} observations of dim {obs.shape[1]}")

    # Features por frame una sola vez (velocidad + encuentro frontal en trayectoria).
    feats = [extract_scene_features(o) for o in obs]
    speeds = np.array([f.speed_kmh for f in feats])
    front_enc = np.array(
        [
            np.isfinite(f.front_dynamic_m) and f.front_dynamic_m <= args.encounter_max_m
            for f in feats
        ]
    )

    if args.moving_only:
        eligible_idx = np.flatnonzero(speeds >= args.min_speed_kmh)
    else:
        eligible_idx = np.arange(obs.shape[0])
    oversample_idx = np.intersect1d(np.flatnonzero(front_enc), eligible_idx)
    print(
        f"Eligible endpoints (moving_only={args.moving_only}, "
        f">={args.min_speed_kmh} km/h): {eligible_idx.size}/{obs.shape[0]}\n"
        f"In-path front encounters (<{args.encounter_max_m:.0f}m) among eligible: "
        f"{oversample_idx.size}"
    )
    if eligible_idx.size < 2:
        print("Aborting: fewer than 2 eligible frames to pair.")
        sys.exit(1)

    annotator = PreferenceAnnotator(
        model=args.model,
        host=args.host,
        seed=args.seed,
        num_gpu=args.num_gpu,
        keep_alive=args.keep_alive,
        request_timeout=args.timeout,
    )
    ok, msg = annotator.warmup()
    print(f"LLM warmup: {'OK' if ok else 'FAILED'} ({args.model}): {msg}")
    if not ok:
        print(
            "Aborting: Ollama not reachable. Is `ollama serve` up and the model pulled?"
        )
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    pairs = select_pairs(
        eligible_idx, args.n_pairs, rng, oversample_idx, args.oversample
    )
    print(f"Annotating {len(pairs)} pairs (symmetric={args.symmetric})...")

    dataset, audit = run_annotation(
        obs,
        pairs,
        annotator,
        symmetric=args.symmetric,
        min_confidence=args.min_confidence,
        progress_every=max(1, len(pairs) // 20),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset)
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as fh:
        for rec in audit:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    kept = dataset["safer"].size
    abstained = len(pairs) - kept
    print(
        f"Done. Kept {kept}/{len(pairs)} pairs ({abstained} abstained/low-conf).\n"
        f"  preferences -> {out_path}\n  audit -> {audit_path}"
    )


if __name__ == "__main__":
    main()
