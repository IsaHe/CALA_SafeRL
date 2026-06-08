"""
scripts/plot_training.py  —  Training-dynamics figures from a SQLite metrics DB.

Reads the metrics.sqlite produced by main_train.py and generates:
  1. Episode reward + rolling success/crash/offroad rates over training.
  2. Shield-off probe learning curve  (Eval/ShieldOff_* metrics).
  3. PPO diagnostics: log_std, KL, entropy, grad norm.
  4. LLM permissiveness vs shield-activation rate  (from meta_decisions table).

Usage:
    # Single run:
    python scripts/plot_training.py \
        --db runs/FullSendC1_p4_adaptive_20260602-203653/metrics.sqlite \
        --out_dir Memoria/imgs

    # Multiple runs overlaid (e.g., weantest phases):
    python scripts/plot_training.py \
        --db runs/weantest_p1_adaptive_*/metrics.sqlite \
              runs/weantest_p2_adaptive_*/metrics.sqlite \
              runs/weantest_p3_adaptive_*/metrics.sqlite \
              runs/weantest_p4_adaptive_*/metrics.sqlite \
        --label "Phase 1" "Phase 2" "Phase 3" "Phase 4" \
        --out_dir Memoria/imgs --prefix training_weantest

    # Shield-off probe only (weaned model):
    python scripts/plot_training.py \
        --db runs/DepReduce_p4_adaptive_*/metrics.sqlite \
        --probe_only --out_dir Memoria/imgs

    # LLM negative plot:
    python scripts/plot_training.py \
        --db runs/FinalAssistLLM_adaptive_*/metrics.sqlite \
        --llm_only --out_dir Memoria/imgs
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

PALETTE = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db", "#9b59b6",
           "#1abc9c", "#f39c12", "#e91e63"]


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def _query_metric(conn: sqlite3.Connection, metric_name: str,
                  axis: str = "episode") -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for one metric from metric_events."""
    cur = conn.execute(
        "SELECT step, value FROM metric_events "
        "WHERE metric_name=? AND axis=? ORDER BY step",
        (metric_name, axis),
    )
    rows = cur.fetchall()
    if not rows:
        return np.array([]), np.array([])
    steps, vals = zip(*rows)
    return np.array(steps), np.array(vals, dtype=float)


def _rolling(y: np.ndarray, w: int = 100) -> np.ndarray:
    """Centered rolling mean with a window of w."""
    if len(y) == 0:
        return y
    kernel = np.ones(w) / w
    pad = w // 2
    y_padded = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(y_padded, kernel, mode="valid")[: len(y)]


def _open(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path, check_same_thread=False)


# ── Figure 1: reward + outcome rates ──────────────────────────────────────────

def plot_reward_outcomes(dbs: list[sqlite3.Connection], labels: list[str],
                         out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=False)
    ax_rew, ax_succ, ax_crash, ax_shield = axes.flat

    for conn, label, color in zip(dbs, labels, PALETTE):
        ep_r, r = _query_metric(conn, "Reward/Raw_Episode")
        ep_s, s = _query_metric(conn, "Training/Success_Rate")
        ep_c, c = _query_metric(conn, "Training/Crash_Rate")
        ep_o, o = _query_metric(conn, "Training/Offroad_Rate")
        ep_sh, sh = _query_metric(conn, "Safety/Shield_Rate")

        if len(ep_r):
            ax_rew.plot(ep_r, _rolling(r, 50), color=color, lw=1.4, label=label)
        if len(ep_s):
            ax_succ.plot(ep_s, _rolling(s, 50), color=color, lw=1.4, label=label)
        if len(ep_c):
            ax_crash.plot(ep_c, _rolling(c, 50), color=color, lw=1.4, label=label)
            ax_crash.plot(ep_o, _rolling(o, 50), color=color, lw=1.0,
                          linestyle="--", alpha=0.7)
        if len(ep_sh):
            ax_shield.plot(ep_sh, _rolling(sh, 50), color=color, lw=1.4, label=label)

    for ax, title, ylabel in [
        (ax_rew,    "Episode reward",     "Reward"),
        (ax_succ,   "Success rate",       "Rate"),
        (ax_crash,  "Crash rate (—) / Off-road (--)", "Rate"),
        (ax_shield, "Shield activation rate", "Fraction"),
    ]:
        ax.set_title(title, pad=6)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episode")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        if len(labels) > 1:
            ax.legend(fontsize=8, framealpha=0.8)

    fig.tight_layout(pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ── Figure 2: shield-off probe ────────────────────────────────────────────────

def plot_shield_off_probe(dbs: list[sqlite3.Connection], labels: list[str],
                          out_path: Path) -> None:
    """
    Plots the shield-off probe metrics (Eval/ShieldOff_*) — the 'true' learning
    curve showing whether the agent can drive without the shield.
    """
    # check at least one db has probe data
    has_data = any(
        len(_query_metric(c, "Eval/ShieldOff_Success")[0]) > 0 for c in dbs
    )
    if not has_data:
        print("No Eval/ShieldOff_* data found — skipping shield-off probe plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    keys   = ["Eval/ShieldOff_Success", "Eval/ShieldOff_Crash", "Eval/ShieldOff_Offroad"]
    titles = ["Shield-off success rate", "Shield-off crash rate", "Shield-off off-road rate"]

    for conn, label, color in zip(dbs, labels, PALETTE):
        for ax, key, title in zip(axes, keys, titles):
            ep, vals = _query_metric(conn, key)
            if len(ep) == 0:
                continue
            ax.scatter(ep, vals, s=18, color=color, alpha=0.55, zorder=3)
            if len(ep) >= 5:
                ax.plot(ep, _rolling(vals, min(len(vals), 5)), color=color,
                        lw=1.6, label=label)

    for ax, title in zip(axes, titles):
        ax.set_title(title, pad=6)
        ax.set_xlabel("Episode")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linestyle=":", alpha=0.3)
        if len(labels) > 1:
            ax.legend(fontsize=8, framealpha=0.8)

    axes[0].set_ylabel("Rate")
    fig.suptitle("Shield-off probe — true policy competence without the shield",
                 fontsize=11, y=1.02)
    fig.tight_layout(pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ── Figure 3: PPO diagnostics ─────────────────────────────────────────────────

def plot_ppo_diagnostics(dbs: list[sqlite3.Connection], labels: list[str],
                         out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
    metrics = [
        ("Training/Log_Std_Steering_Raw",         "log σ steering (raw)"),
        ("Training/Log_Std_Throttle_Raw",          "log σ throttle (raw)"),
        ("Training/Log_Std_Saturated_Fraction",    "Saturated log_std fraction"),
        ("Training/Approx_KL",                     "Approx KL"),
        ("Training/Entropy",                       "Entropy"),
        ("Loss/Grad_Norm",                         "Gradient norm"),
    ]
    for conn, label, color in zip(dbs, labels, PALETTE):
        for ax, (key, title) in zip(axes.flat, metrics):
            step, val = _query_metric(conn, key, axis="update")
            if len(step) == 0:
                continue
            ax.plot(step, val, color=color, lw=1.0, alpha=0.8, label=label)

    for ax, (_, title) in zip(axes.flat, metrics):
        ax.set_title(title, pad=5)
        ax.set_xlabel("Timestep")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linestyle=":", alpha=0.3)
        if len(labels) > 1:
            ax.legend(fontsize=8)

    # Saturation reference lines for log_std
    LOG_STD_MIN, LOG_STD_MAX = -3.0, -1.2
    for row in axes:
        for ax in row[:2]:
            ax.axhline(LOG_STD_MAX, color="red",  linestyle="--", lw=0.8,
                       alpha=0.6, label="LOG_STD_MAX")
            ax.axhline(LOG_STD_MIN, color="blue", linestyle="--", lw=0.8,
                       alpha=0.6, label="LOG_STD_MIN")

    fig.tight_layout(pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ── Figure 4: LLM permissiveness vs shield activation (negative result) ────────

def plot_llm_negative(conn: sqlite3.Connection, out_path: Path) -> None:
    """
    Two-axis time-series:  permissiveness (left) and shield_rate (right).
    Also shows pearson correlation between the two.
    """
    # meta_decisions table stores permissiveness
    cur = conn.execute(
        "SELECT episode, applied_permissiveness, shielded_fraction, crash_rate "
        "FROM meta_decisions ORDER BY episode"
    )
    rows = cur.fetchall()
    if not rows:
        print("No meta_decisions data — skipping LLM plot.")
        return

    eps, perm, sh_frac, cr_rate = (np.array(col, dtype=float)
                                    for col in zip(*rows))

    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    ax2 = ax1.twinx()

    color_p = "#3498db"
    color_s = "#e74c3c"

    l1, = ax1.plot(eps, perm,    color=color_p, lw=2.0, label="Permissiveness $p$")
    l2, = ax2.plot(eps, sh_frac, color=color_s, lw=1.5, linestyle="--",
                   label="Shield activation rate")

    # Pearson correlation annotation
    if len(perm) > 1:
        rho = float(np.corrcoef(perm, sh_frac)[0, 1])
        ax1.text(0.02, 0.97,
                 f"Pearson $\\rho$(perm, shield rate) = {rho:+.2f}",
                 transform=ax1.transAxes, fontsize=9, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#cccccc", alpha=0.9))

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Permissiveness $p$", color=color_p)
    ax2.set_ylabel("Shield activation rate", color=color_s)
    ax1.tick_params(axis="y", colors=color_p)
    ax2.tick_params(axis="y", colors=color_s)

    handles = [l1, l2]
    ax1.legend(handles=handles, loc="lower left", framealpha=0.85)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax1.grid(axis="x", linestyle=":", alpha=0.3)

    fig.suptitle(
        "LLM meta-controller: loosening the shield does not reduce reliance",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ── CLI ─────────────────────────────────────────────────────────────────────

def _glob_or_literal(pattern: str) -> list[str]:
    """Expand glob patterns or return the literal path if it exists."""
    from glob import glob
    matches = sorted(glob(pattern))
    return matches if matches else [pattern]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", nargs="+", required=True,
                   help="Path(s) to metrics.sqlite (glob patterns accepted)")
    p.add_argument("--label", nargs="*", default=None,
                   help="Labels for each DB (auto-inferred from dir name if omitted)")
    p.add_argument("--out_dir", default="Memoria/imgs")
    p.add_argument("--prefix", default="training",
                   help="Output filename prefix (default: training)")
    p.add_argument("--probe_only", action="store_true",
                   help="Only generate the shield-off probe figure")
    p.add_argument("--llm_only", action="store_true",
                   help="Only generate the LLM negative-result figure")
    p.add_argument("--ppo_only", action="store_true",
                   help="Only generate the PPO diagnostics figure")
    args = p.parse_args()

    # Expand glob patterns
    db_paths = []
    for pat in args.db:
        db_paths.extend(_glob_or_literal(pat))
    db_paths = [p2 for p2 in db_paths if Path(p2).exists()]

    if not db_paths:
        print("ERROR: no SQLite databases found. Check --db paths.")
        return

    labels = args.label or [Path(p2).parent.name for p2 in db_paths]
    labels = labels[: len(db_paths)]  # trim if too many

    print(f"Opening {len(db_paths)} database(s):")
    for lbl, pth in zip(labels, db_paths):
        print(f"  [{lbl}] {pth}")

    dbs = [_open(pth) for pth in db_paths]
    out_dir = Path(args.out_dir)
    prefix  = args.prefix

    if args.llm_only:
        plot_llm_negative(dbs[0], out_dir / "llm_negative.pdf")
    elif args.probe_only:
        plot_shield_off_probe(dbs, labels, out_dir / "shield_off_probe.pdf")
    elif args.ppo_only:
        plot_ppo_diagnostics(dbs, labels, out_dir / f"{prefix}_ppo_diag.pdf")
    else:
        plot_reward_outcomes(dbs, labels, out_dir / f"{prefix}_curves.pdf")
        plot_shield_off_probe(dbs, labels, out_dir / "shield_off_probe.pdf")
        plot_ppo_diagnostics(dbs, labels, out_dir / f"{prefix}_ppo_diag.pdf")
        if args.llm_only or any(
            len(_query_metric(c, "Eval/ShieldOff_Success")[0]) == 0
            and conn.execute("SELECT count(*) FROM meta_decisions").fetchone()[0] > 0
            for c, conn in zip(dbs, dbs)
        ):
            plot_llm_negative(dbs[0], out_dir / "llm_negative.pdf")

    for conn in dbs:
        conn.close()


if __name__ == "__main__":
    main()
