"""
scripts/plot_weaned_comparison.py  —  Side-by-side comparison of baseline vs weaned agent.

Reads:
  data/results/baseline_3way_ablation.json  (none / basic / adaptive on FullSendC1_p4)
  data/results/weaned_3way_ablation.json    (none / basic / adaptive on learn_fix2)

Generates:
  Memoria/imgs/weaned_vs_baseline.pdf — 2-row × 3-col figure for Chapter 7.

Usage:
    python scripts/plot_weaned_comparison.py [--out_dir Memoria/imgs]
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

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

OUTCOME_ORDER  = ["success", "crash", "offroad", "stuck", "timeout"]
OUTCOME_LABELS = ["Success", "Crash", "Off-road", "Stuck", "Timeout"]
OUTCOME_COLORS = ["#2ecc71", "#e74c3c", "#e67e22", "#9b59b6", "#95a5a6"]
SHIELD_DISPLAY = {"none": "No shield", "basic": "Basic", "adaptive": "Adaptive"}

ROOT    = Path(__file__).resolve().parent.parent
JSON_B  = ROOT / "data/results/baseline_3way_ablation.json"
JSON_W  = ROOT / "data/results/weaned_3way_ablation.json"


def _load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["results"]


def _label(s: str) -> str:
    return SHIELD_DISPLAY.get(s, s)


def _pct(ax):
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))


def _trim(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)


def plot_comparison(results_b: list[dict], results_w: list[dict], out_path: Path) -> None:
    """
    2-row × 3-column figure:
      Row 0 = Baseline model (FullSendC1_p4)
      Row 1 = Weaned model   (learn_fix2)
    Columns: outcome distribution | crash+offroad | avg distance + shields/ep
    """
    rows = [
        (results_b, "Baseline (\\texttt{FullSendC1\\_p4})"),
        (results_w, "Weaned (\\texttt{learn\\_fix2})"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for row_idx, (results, row_title) in enumerate(rows):
        n      = len(results)
        x      = np.arange(n)
        labels = [_label(r["shield_type"]) for r in results]
        w      = 0.32

        # ── col 0: stacked outcome bars ───────────────────────────────────────
        ax = axes[row_idx][0]
        btm = np.zeros(n)
        for key, lbl, col in zip(OUTCOME_ORDER, OUTCOME_LABELS, OUTCOME_COLORS):
            vals = np.array([r[f"{key}_rate"] for r in results])
            bars = ax.bar(x, vals, bottom=btm, color=col, label=lbl,
                          width=0.55, edgecolor="white", linewidth=0.4)
            for bar, v in zip(bars, vals):
                if v > 0.05:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:.0%}",
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold",
                    )
            btm += vals
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        _pct(ax)
        ax.set_ylabel(row_title, fontsize=9, labelpad=4)
        if row_idx == 0:
            ax.set_title("Outcome distribution")
            ax.legend(
                loc="lower center", bbox_to_anchor=(0.5, -0.55),
                ncol=3, fontsize=7.5, framealpha=0.85,
            )
        _trim(ax)

        # ── col 1: crash + offroad grouped ────────────────────────────────────
        ax = axes[row_idx][1]
        crash   = np.array([r["crash_rate"]   for r in results])
        offroad = np.array([r["offroad_rate"] for r in results])
        ax.bar(x - w / 2, crash,   w, label="Crash",    color="#e74c3c", edgecolor="white")
        ax.bar(x + w / 2, offroad, w, label="Off-road", color="#e67e22", edgecolor="white")
        for xi, c, o in zip(x, crash, offroad):
            if c > 0.01:
                ax.text(xi - w / 2, c + 0.01, f"{c:.0%}", ha="center", fontsize=8)
            if o > 0.01:
                ax.text(xi + w / 2, o + 0.01, f"{o:.0%}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        _pct(ax)
        if row_idx == 0:
            ax.set_title("Crash and off-road rates")
            ax.legend(fontsize=8, framealpha=0.85)
        _trim(ax)

        # ── col 2: avg distance + shields/ep ─────────────────────────────────
        ax  = axes[row_idx][2]
        ax2 = ax.twinx()
        dist  = np.array([r.get("avg_distance", 0) for r in results])
        shlds = np.array([r.get("shields_per_ep", 0) for r in results])
        c_d, c_s = "#1abc9c", "#e67e22"
        b1 = ax.bar(x - w / 2, dist,  w, color=c_d, edgecolor="white", label="Avg dist (m)")
        b2 = ax2.bar(x + w / 2, shlds, w, color=c_s, edgecolor="white", label="Shields/ep")
        ax.set_ylabel("Distance (m)", color=c_d, fontsize=9)
        ax2.set_ylabel("Shields/ep",  color=c_s, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if row_idx == 0:
            ax.set_title("Distance & shield rate")
            ax.legend(handles=[b1, b2], labels=["Avg dist (m)", "Shields/ep"],
                      fontsize=7.5, framealpha=0.85)
        ax.spines[["top"]].set_visible(False)
        ax2.spines[["top"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    fig.suptitle(
        "Baseline vs weaned agent — 50 episodes, seed-aligned, 60 NPCs",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="Memoria/imgs",
                   help="Output directory (default: Memoria/imgs)")
    args = p.parse_args()

    if not JSON_B.exists():
        print(f"ERROR: baseline JSON not found at {JSON_B}")
        return
    if not JSON_W.exists():
        print(f"ERROR: weaned JSON not found at {JSON_W}")
        print("Run first:")
        print(
            "  python main_eval.py --model_name learn_fix2_adaptive_final.pth "
            "--shield_type none basic adaptive --episodes 50 "
            "--out data/results/weaned_3way_ablation.json"
        )
        return

    results_b = _load(JSON_B)
    results_w = _load(JSON_W)

    out_dir = Path(args.out_dir)
    plot_comparison(results_b, results_w, out_dir / "weaned_vs_baseline.pdf")


if __name__ == "__main__":
    main()
