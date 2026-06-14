"""
validate_preferences.py - Validacion del dataset de preferencias (fase 1).

Comprueba integridad (observations.npz / preferences.npz) y, sobre todo, la
CALIDAD SEMANTICA de las etiquetas: que la preferencia del LLM siga el riesgo de
COLISION (obstaculo en trayectoria) y no confunda 'parado/bien centrado' con
seguro. Reutilizable tras cada re-anotacion para ver si los arreglos funcionaron.

Solo numpy + scene_caption (sin CARLA, sin Ollama).

    python tools/validate_preferences.py [--dir data/preferences]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.meta.scene_caption import extract_scene_features  # noqa: E402


def _section(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _fd(f) -> float:
    return f.front_dynamic_m if np.isfinite(f.front_dynamic_m) else 999.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/preferences")
    ap.add_argument(
        "--gap", type=float, default=8.0, help="m de asimetria frontal 'clara'"
    )
    args = ap.parse_args()
    d = Path(args.dir)

    # ── A. observations.npz ──────────────────────────────────────────────────
    _section("A. observations.npz")
    obs = np.load(d / "observations.npz")["obs"]
    print(f"shape={obs.shape} dtype={obs.dtype}")
    print(
        f"LIDAR[:, :720]  min={obs[:, :720].min():.3f} max={obs[:, :720].max():.3f} (esp. [0,1])"
    )
    print(
        f"feats[:,720:]   min={obs[:, 720:].min():.3f} max={obs[:, 720:].max():.3f} (esp. [-1,1])"
    )
    print(f"NaN/Inf: {np.isnan(obs).any()} / {np.isinf(obs).any()}")
    feats = [extract_scene_features(o) for o in obs]
    speeds = np.array([f.speed_kmh for f in feats])
    fronts = np.array([_fd(f) for f in feats])
    front_enc = (fronts <= 35) & (fronts < 999)
    print(
        f"speed km/h: mean={speeds.mean():.1f} p50={np.median(speeds):.1f} max={speeds.max():.1f}"
    )
    print(f"moving frames (>5 km/h): {100 * (speeds > 5).mean():.1f}%")
    print(
        f"in-path front encounters (<35m): {front_enc.sum()}/{len(obs)} = {100 * front_enc.mean():.1f}%"
    )

    # ── B. preferences.npz ───────────────────────────────────────────────────
    _section("B. preferences.npz")
    pref = np.load(d / "preferences.npz")
    safer, conf = pref["safer"], pref["confidence"]
    print(f"obs_a={pref['obs_a'].shape} safer={safer.shape}")
    print(
        f"safer values: {dict(zip(*np.unique(safer, return_counts=True)))} (esp. solo 0/1)"
    )
    if safer.size:
        print(
            f"balance: A={100 * (safer == 0).mean():.1f}%  B={100 * (safer == 1).mean():.1f}%"
        )
        print(
            f"confidence: min={conf.min():.2f} mean={conf.mean():.2f} max={conf.max():.2f}"
        )

    # ── C. annotations.jsonl ─────────────────────────────────────────────────
    _section("C. annotations.jsonl")
    recs = [
        json.loads(line) for line in (d / "annotations.jsonl").open(encoding="utf-8")
    ]
    ok = [r for r in recs if r["ok"]]
    posbias = [r for r in recs if not r["ok"] and "position-bias" in r["reasoning"]]
    n = max(1, len(recs))
    print(
        f"total={len(recs)}  ok={len(ok)} ({100 * len(ok) / n:.1f}%)  "
        f"position-bias abstain={len(posbias)} ({100 * len(posbias) / n:.1f}%)"
    )

    # ── D. SEMANTIC: ¿la preferencia sigue el riesgo de colision? ────────────
    _section("D. Semantic quality (the decisive checks)")

    def feat(idx):
        return extract_scene_features(obs[idx])

    # D1. asimetria frontal clara -> mas despejado debe ser mas seguro.
    agree = disagree = 0
    for r in ok:
        fi, fj = _fd(feat(r["i"])), _fd(feat(r["j"]))
        if abs(fi - fj) < args.gap:
            continue
        should = 0 if fi > fj else 1
        agree += int(r["safer"] == should)
        disagree += int(r["safer"] != should)
    dec = agree + disagree
    print(
        f"D1. clear front asymmetry (>{args.gap:.0f}m): {dec} pairs; "
        f"'more clearance=safer' agreement: {100 * agree / max(1, dec):.0f}%  "
        f"(>50% good; <50% means the signal is INVERTED)"
    )

    # D2. ¿confunde 'parado' con seguro? (esperado: bajo, parado != peligro)
    stop_pref = stop_tot = 0
    for r in ok:
        i_stop = feat(r["i"]).speed_kmh < 1.0
        j_stop = feat(r["j"]).speed_kmh < 1.0
        if i_stop ^ j_stop:
            stop_tot += 1
            stop_pref += int(r["safer"] == (0 if i_stop else 1))
    print(
        f"D2. stopped-vs-moving pairs: {stop_tot}; LLM picks STOPPED as safer: "
        f"{100 * stop_pref / max(1, stop_tot):.0f}%  (a high value here = the "
        f"stopped confound; with --moving_only this should be ~0 pairs)"
    )

    # D3. subconjunto LIMPIO: obstaculo en via & movil vs via libre & movil.
    clean = []
    for r in ok:
        fi, fj = feat(r["i"]), feat(r["j"])
        blk_i = _fd(fi) < 12 and fi.speed_kmh > 5
        blk_j = _fd(fj) < 12 and fj.speed_kmh > 5
        clr_i = (not np.isfinite(fi.nearest_dynamic_m)) and fi.speed_kmh > 5
        clr_j = (not np.isfinite(fj.nearest_dynamic_m)) and fj.speed_kmh > 5
        if blk_i and clr_j:
            clean.append(r["safer"] == 1)
        elif blk_j and clr_i:
            clean.append(r["safer"] == 0)
    print(
        f"D3. CLEAN subset (in-path obstacle&moving vs clear&moving): n={len(clean)}; "
        f"LLM picks the clear road as safer: "
        f"{100 * (sum(clean) / len(clean)) if clean else float('nan'):.0f}%  "
        f"(this is the headline number; want >>50%)"
    )

    # ── E. eyeball samples ───────────────────────────────────────────────────
    _section("E. Sample high-confidence in-path-encounter pairs")
    shown = 0
    for r in ok:
        if r["confidence"] < 0.7:
            continue
        if (
            "directly ahead" not in r["caption_a"]
            and "directly ahead" not in r["caption_b"]
        ):
            continue
        print(
            f"\n[i={r['i']} j={r['j']} safer={'A' if r['safer'] == 0 else 'B'} conf={r['confidence']:.2f}]"
        )
        print(f"  A: {r['caption_a']}")
        print(f"  B: {r['caption_b']}")
        print(f"  reason: {r['reasoning']}")
        shown += 1
        if shown >= 4:
            break


if __name__ == "__main__":
    main()
