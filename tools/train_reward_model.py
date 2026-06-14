"""
train_reward_model.py - Entrena el reward model r_phi (fase 2) por preferencias.

Lee ``preferences.npz`` (obs_a, obs_b, safer, confidence), entrena con perdida
Bradley-Terry y corre tres GATES de validacion antes de guardar:
  1. ranking accuracy en validacion (held-out) -- debe ser >> 0.5.
  2. monotonicidad: r_phi debe CRECER al alejarse un obstaculo frontal.
  3. resumen del campo de reward (sanity de rango/signo).

Guarda ``data/models/reward_model.pth`` y, si hay matplotlib, un PNG del campo.
100% offline (no CARLA, no Ollama).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.imitation.reward_model import (  # noqa: E402
    RewardModel,
    save_reward_model,
    train_bradley_terry,
)
from src.meta.scene_caption import extract_scene_features  # noqa: E402

# Las gates de monotonicidad/sanity se evaluan SOBRE OBSERVACIONES REALES del
# dataset (in-distribution). Un probe sintetico que toca un solo bin del LIDAR
# es OOD (un vehiculo real ilumina varios bins del cono frontal + lane features)
# y da una senal enganosa: el modelo solo es valido donde fue entrenado.

_BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 35), (35, 1e9)]


def _front_distances(obs_pool: np.ndarray) -> np.ndarray:
    out = np.empty(obs_pool.shape[0], dtype=np.float32)
    for k, o in enumerate(obs_pool):
        fd = extract_scene_features(o).front_dynamic_m
        out[k] = fd if np.isfinite(fd) else 999.0
    return out


def distance_buckets(
    model: RewardModel, obs_pool: np.ndarray, fd: np.ndarray
) -> list[tuple[str, int, float]]:
    """r_phi medio por bucket de distancia frontal REAL. Debe CRECER con la dist."""
    r = np.asarray(model.predict_np(obs_pool)).reshape(-1)
    rows = []
    for lo, hi in _BUCKETS:
        mask = (fd >= lo) & (fd < hi)
        label = f"{lo:.0f}-{hi:.0f}m" if hi < 1e8 else f">{lo:.0f}m/clear"
        rows.append(
            (
                label,
                int(mask.sum()),
                float(r[mask].mean()) if mask.any() else float("nan"),
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preferences", default="data/preferences/preferences.npz")
    ap.add_argument("--out", default="data/models/reward_model.pth")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    data = np.load(args.preferences)
    dataset = {
        k: data[k] for k in ("obs_a", "obs_b", "safer", "confidence") if k in data
    }
    n = dataset["obs_a"].shape[0]
    print(f"Loaded {n} preference pairs from {args.preferences} (device={args.device})")
    if n < 10:
        print("Too few pairs to train a meaningful reward model; aborting.")
        sys.exit(1)

    model = RewardModel(hidden=args.hidden)
    history = train_bradley_terry(
        model,
        dataset,
        epochs=args.epochs,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        device=args.device,
        verbose=True,
    )

    model.eval()
    # Pool de observaciones REALES del dataset para validar in-distribution.
    obs_pool = np.concatenate([dataset["obs_a"], dataset["obs_b"]], axis=0)
    fd = _front_distances(obs_pool)

    # ── GATE 1: ranking accuracy (held-out) ───────────────────────────────────
    val_acc = history["val_acc"][-1] if history["val_acc"] else float("nan")
    print(f"\n[GATE 1] held-out ranking accuracy: {val_acc:.3f}  (want >> 0.5)")
    # Ranking sobre los pares de ALTA confianza (los 'ties' de baja confianza son
    # ~50/50 por diseno y bajan la media); informativo, in-sample.
    conf = dataset.get("confidence")
    if conf is not None and len(conf):
        r_a = np.asarray(model.predict_np(dataset["obs_a"])).reshape(-1)
        r_b = np.asarray(model.predict_np(dataset["obs_b"])).reshape(-1)
        pred = (r_b > r_a).astype(np.int8)
        hi = np.asarray(conf) >= 0.8
        if hi.any():
            acc_hi = float((pred[hi] == dataset["safer"][hi]).mean())
            print(
                f"          high-confidence (>=0.8) pair ranking: {acc_hi:.3f} "
                f"(n={int(hi.sum())}, in-sample)"
            )

    # ── GATE 2: monotonicity on REAL obs (r_phi should rise with front distance)
    rows = distance_buckets(model, obs_pool, fd)
    print("[GATE 2] mean r_phi by REAL front-distance bucket (should INCREASE):")
    means = []
    for label, cnt, mean_r in rows:
        print(f"    {label:>11}: n={cnt:5d}  mean r_phi={mean_r:+.3f}")
        if cnt > 0:
            means.append(mean_r)
    increasing = all(b >= a - 1e-3 for a, b in zip(means, means[1:]))
    print(f"    monotonic non-decreasing across non-empty buckets: {increasing}")

    # ── GATE 3: clear road scores safer than a close in-path lead (real obs) ──
    close_mask, clear_mask = fd < 8.0, fd > 35.0
    r_all = np.asarray(model.predict_np(obs_pool)).reshape(-1)
    clear_m = float(r_all[clear_mask].mean()) if clear_mask.any() else float("nan")
    close_m = float(r_all[close_mask].mean()) if close_mask.any() else float("nan")
    print(
        f"[GATE 3] mean r_phi clear(>35m)={clear_m:+.3f} vs close(<8m)={close_m:+.3f}  "
        f"(clear should be HIGHER: {clear_m > close_m})"
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_reward_model(
        model,
        args.out,
        meta={
            "val_acc": val_acc,
            "n_pairs": int(n),
            "epochs": args.epochs,
            "monotonic_real": bool(increasing),
        },
    )
    print(f"\nSaved reward model -> {args.out}")

    if args.plot:
        _try_plot(obs_pool, fd, r_all, Path(args.out).with_suffix(".field.png"))


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _try_plot(obs_pool: np.ndarray, fd: np.ndarray, r: np.ndarray, path: Path) -> None:
    """r_phi vs distancia frontal REAL (in-distribution): scatter + media por bin."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(plot skipped: {exc})")
        return
    vis = fd <= 50.0  # solo donde hay obstaculo frontal medible
    d = fd[vis]
    rr = np.asarray(r).reshape(-1)[vis]
    plt.figure(figsize=(7, 4))
    plt.scatter(d, rr, s=6, alpha=0.3, label="real obs")
    edges = np.linspace(0, 50, 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = [
        rr[(d >= lo) & (d < hi)].mean() if ((d >= lo) & (d < hi)).any() else np.nan
        for lo, hi in zip(edges[:-1], edges[1:])
    ]
    plt.plot(centers, means, "r-o", lw=2, label="bin mean")
    plt.xlabel("front in-path obstacle distance (m)")
    plt.ylabel("r_phi (higher = safer)")
    plt.title("Learned safety reward vs front distance (real observations)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"Saved reward curve -> {path}")


if __name__ == "__main__":
    main()
