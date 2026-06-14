"""
merge_pair_labels.py - Une pairs.jsonl + labels_*.jsonl en el dataset de preferencias.

Reconstruye el mismo formato que ``annotate_preferences.py`` a partir de las
etiquetas producidas por anotadores externos (subagentes Claude), usando los
indices globales (idx_a/idx_b) registrados por ``prepare_pairs.py`` para mapear
cada par a sus observaciones.

  - ``preferences.npz``  : obs_a, obs_b, safer, confidence (solo pares ok).
  - ``annotations.jsonl``: auditoria (i, j, caption_a/b, safer, confidence, ok,
                           reasoning) compatible con ``validate_preferences.py``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observations", default="data/preferences/observations.npz")
    ap.add_argument("--pairs", default="data/preferences/pairs.jsonl")
    ap.add_argument("--labels_glob", default="data/preferences/chunks/labels_*.jsonl")
    ap.add_argument("--out", default="data/preferences/preferences.npz")
    ap.add_argument("--audit", default="data/preferences/annotations.jsonl")
    ap.add_argument("--min_confidence", type=float, default=0.0)
    args = ap.parse_args()

    obs = np.load(args.observations)["obs"]
    pairs = {int(r["pair_id"]): r for r in _load_jsonl(args.pairs)}

    labels: dict[int, dict] = {}
    label_files = sorted(glob.glob(args.labels_glob))
    for path in label_files:
        for d in _load_jsonl(path):
            labels[int(d["pair_id"])] = d
    print(f"pairs={len(pairs)}  label files={len(label_files)}  labels={len(labels)}")

    obs_a, obs_b, safer, conf, audit = [], [], [], [], []
    for pid, pr in sorted(pairs.items()):
        lab = labels.get(pid)
        ok = lab is not None and int(lab.get("safer", -1)) in (0, 1)
        s = int(lab["safer"]) if ok else -1
        c = float(lab.get("confidence", 0.0)) if ok else 0.0
        c = min(1.0, max(0.0, c))
        audit.append(
            {
                "i": int(pr["idx_a"]),
                "j": int(pr["idx_b"]),
                "caption_a": pr["caption_a"],
                "caption_b": pr["caption_b"],
                "safer": s,
                "confidence": c,
                "ok": bool(ok),
                "reasoning": (lab.get("reasoning", "") if lab else "missing label"),
            }
        )
        if ok and c >= args.min_confidence:
            obs_a.append(obs[pr["idx_a"]])
            obs_b.append(obs[pr["idx_b"]])
            safer.append(s)
            conf.append(c)

    dataset = {
        "obs_a": np.asarray(obs_a, dtype=np.float32).reshape(-1, obs.shape[1]),
        "obs_b": np.asarray(obs_b, dtype=np.float32).reshape(-1, obs.shape[1]),
        "safer": np.asarray(safer, dtype=np.int8),
        "confidence": np.asarray(conf, dtype=np.float32),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **dataset)
    with Path(args.audit).open("w", encoding="utf-8") as fh:
        for rec in audit:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    missing = sum(1 for r in audit if not r["ok"])
    print(
        f"kept {dataset['safer'].size}/{len(pairs)} pairs "
        f"({missing} missing/invalid).\n  -> {args.out}\n  -> {args.audit}"
    )


if __name__ == "__main__":
    main()
