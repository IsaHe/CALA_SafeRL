"""
prepare_pairs.py - Prepara pares de captions para anotacion EXTERNA (sin LLM local).

Para usar un anotador FUERTE (p.ej. Claude via subagentes) en vez del Gemma-4B
local (que abstuvo el 85% por sesgo de posicion), separamos la SELECCION de pares
(deterministica, sin LLM) de la ANOTACION. Este script:

  1. carga observations.npz,
  2. selecciona pares (solo frames en movimiento; sobremuestrea obstaculos EN
     TRAYECTORIA, reutilizando la logica de annotate_preferences),
  3. RANDOMIZA el orden A/B de cada par y lo REGISTRA (idx_a/idx_b) -> anula el
     sesgo de posicion en agregado sin doblar las llamadas,
  4. escribe data/preferences/pairs.jsonl y lo trocea en chunks/ para repartir
     entre N anotadores en paralelo.

Cada linea de pairs.jsonl / chunk: {pair_id, idx_a, idx_b, caption_a, caption_b}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meta.scene_caption import caption_from_obs, extract_scene_features  # noqa: E402
from tools.annotate_preferences import select_pairs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observations", default="data/preferences/observations.npz")
    ap.add_argument("--out_dir", default="data/preferences")
    ap.add_argument("--n_pairs", type=int, default=600)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--oversample", type=float, default=0.7)
    ap.add_argument("--encounter_max_m", type=float, default=25.0)
    ap.add_argument("--min_speed_kmh", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    obs = np.load(args.observations)["obs"]
    feats = [extract_scene_features(o) for o in obs]
    speeds = np.array([f.speed_kmh for f in feats])
    front = np.array(
        [
            np.isfinite(f.front_dynamic_m) and f.front_dynamic_m <= args.encounter_max_m
            for f in feats
        ]
    )
    eligible = np.flatnonzero(speeds >= args.min_speed_kmh)
    oversample_idx = np.intersect1d(np.flatnonzero(front), eligible)
    print(
        f"obs={obs.shape[0]}  eligible(moving)={eligible.size}  "
        f"in-path-encounter endpoints={oversample_idx.size}"
    )

    rng = np.random.default_rng(args.seed)
    raw_pairs = select_pairs(
        eligible, args.n_pairs, rng, oversample_idx, args.oversample
    )

    recs = []
    for pid, (i, j) in enumerate(raw_pairs):
        if rng.random() < 0.5:  # randomiza que endpoint se muestra como A
            i, j = j, i
        recs.append(
            {
                "pair_id": pid,
                "idx_a": int(i),
                "idx_b": int(j),
                "caption_a": caption_from_obs(obs[i]),
                "caption_b": caption_from_obs(obs[j]),
            }
        )

    out_dir = Path(args.out_dir)
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)
    with (out_dir / "pairs.jsonl").open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    sz = (len(recs) + args.chunks - 1) // args.chunks
    chunk_paths = []
    for c in range(args.chunks):
        part = recs[c * sz : (c + 1) * sz]
        if not part:
            break
        path = out_dir / "chunks" / f"pairs_{c:03d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        chunk_paths.append((str(path), len(part)))

    print(f"wrote {len(recs)} pairs -> {out_dir / 'pairs.jsonl'}")
    for p, n in chunk_paths:
        print(f"  chunk {p}: {n} pairs")


if __name__ == "__main__":
    main()
