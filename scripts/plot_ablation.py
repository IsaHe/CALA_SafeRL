"""
scripts/plot_ablation.py  —  Shield-ablation comparison figures for the thesis.

Reads the JSON output produced by:
    python main_eval.py --model_name <model> \
        --shield_type none adaptive --episodes 50 --out ablation.json

Generates two publication-quality figures:
  1. Stacked bar chart of outcome rates per shield configuration.
  2. Side-by-side unsafe-rate / shields-per-episode comparison.

Usage:
    python scripts/plot_ablation.py --input ablation.json \
        [--out_dir Memoria/imgs] [--prefix shield_ablation]
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
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
        "pdf.fonttype": 42,   # embed fonts for LaTeX compatibility
        "ps.fonttype": 42,
    }
)

OUTCOME_ORDER   = ["success", "crash", "offroad", "stuck", "timeout"]
OUTCOME_LABELS  = ["Success", "Crash", "Off-road", "Stuck", "Timeout"]
OUTCOME_COLORS  = ["#2ecc71", "#e74c3c", "#e67e22", "#9b59b6", "#95a5a6"]

SHIELD_DISPLAY  = {"none": "No shield", "basic": "Basic shield",
                   "adaptive": "Adaptive shield"}


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _label(s: str) -> str:
    return SHIELD_DISPLAY.get(s, s)


# ── Figure 1: stacked outcome bars ────────────────────────────────────────────

def plot_outcome_bars(results: list[dict], out_path: Path) -> None:
    n = len(results)
    x = np.arange(n)
    labels = [_label(r["shield_type"]) for r in results]

    fig, ax = plt.subplots(figsize=(max(4, 2.2 * n), 3.6))

    bottoms = np.zeros(n)
    for key, label, color in zip(OUTCOME_ORDER, OUTCOME_LABELS, OUTCOME_COLORS):
        vals = np.array([r[f"{key}_rate"] for r in results])
        bars = ax.bar(x, vals, bottom=bottoms, color=color,
                      label=label, width=0.55, edgecolor="white", linewidth=0.4)
        # annotate inside bar if wide enough
        for bar, v in zip(bars, vals):
            if v > 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.0%}",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold",
                )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Episode fraction")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=len(OUTCOME_ORDER), framealpha=0.85,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ── Figure 2: unsafe rate + shields/ep comparison ─────────────────────────────

def plot_safety_comparison(results: list[dict], out_path: Path) -> None:
    n = len(results)
    labels = [_label(r["shield_type"]) for r in results]
    x = np.arange(n)
    w = 0.35

    crash   = np.array([r["crash_rate"]   for r in results])
    offroad = np.array([r["offroad_rate"] for r in results])
    unsafe  = crash + offroad
    shld_ep = np.array([r["shields_per_ep"] for r in results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    # Panel A: crash and off-road stacked
    bar1 = ax1.bar(x - w / 2, crash,   w, label="Crash",    color="#e74c3c", edgecolor="white")
    bar2 = ax1.bar(x + w / 2, offroad, w, label="Off-road", color="#e67e22", edgecolor="white")
    for bar, vals in [(bar1, crash), (bar2, offroad)]:
        for b, v in zip(bar, vals):
            if v > 0.02:
                ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                         f"{v:.0%}", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Episode fraction")
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax1.set_title("(a) Crash and off-road rates")
    ax1.legend(framealpha=0.8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    # Panel B: shields per episode
    bar3 = ax2.bar(x, shld_ep, 0.55, color="#3498db", edgecolor="white")
    for b, v in zip(bar3, shld_ep):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                 f"{v:.1f}", ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Shield interventions / episode")
    ax2.set_title("(b) Shield activation rate")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout(pad=1.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close(fig)


# ── Figure 3: combined ablation overview (matches thesis fig:ablation) ─────────

def plot_combined(results: list[dict], out_path: Path) -> None:
    """
    3-panel figure with corrected legend placement below the charts.
    Matches the clean layout of Memoria/imgs/shield_ablation.pdf.
    """
    n = len(results)
    labels = [_label(r["shield_type"]) for r in results]
    x = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))

    # ── panel 0: stacked outcome bars ────────────────────────────────────
    ax = axes[0]
    bottoms = np.zeros(n)
    for key, label, color in zip(OUTCOME_ORDER, OUTCOME_LABELS, OUTCOME_COLORS):
        vals = np.array([r[f"{key}_rate"] for r in results])
        ax.bar(x, vals, bottom=bottoms, color=color, label=label,
               width=0.55, edgecolor="white", linewidth=0.4)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_title("Outcome distribution")
    # FIX: Place legend below x-axis, aligned to the upper edge of the legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=7.5,
        framealpha=0.85,
        borderaxespad=0,
    )
    ax.spines[["top", "right"]].set_visible(False)

    # ── panel 1: success vs unsafe gap ───────────────────────────────────
    ax = axes[1]
    succ   = np.array([r["success_rate"] for r in results])
    unsafe = np.array([r["crash_rate"] + r["offroad_rate"] for r in results])
    w = 0.32
    ax.bar(x - w / 2, succ,   w, label="Success",       color="#2ecc71", edgecolor="white")
    ax.bar(x + w / 2, unsafe, w, label="Crash+off-road", color="#e74c3c", edgecolor="white")
    for bars, vals in [((x - w / 2, succ), succ), ((x + w / 2, unsafe), unsafe)]:
        for xi, v in zip(bars[0], bars[1]):
            ax.text(xi, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_title("Success vs. unsafe")
    # FIX: Also place this legend outside and below, aligning with panel 0
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=7.5,
        framealpha=0.85,
        borderaxespad=0,
    )
    ax.spines[["top", "right"]].set_visible(False)

    # ── panel 2: distance & shields/ep ───────────────────────────────────
    ax = axes[2]
    dist  = np.array([r["avg_distance"] for r in results])
    shlds = np.array([r["shields_per_ep"] for r in results])

    color_d = "#1abc9c"
    color_s = "#e67e22"
    ax2_twin = ax.twinx()
    b1 = ax.bar(x - w / 2, dist,  w, color=color_d, edgecolor="white",
                label="Avg distance (m)")
    b2 = ax2_twin.bar(x + w / 2, shlds, w, color=color_s, edgecolor="white",
                      label="Shields/ep")
    ax.set_ylabel("Distance (m)", color=color_d, fontsize=9)
    ax2_twin.set_ylabel("Shields/ep", color=color_s, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Distance & shield rate")
    
    # FIX: Clean unified legend handles for outside placement
    # First, collect handles and labels from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2_twin.get_legend_handles_labels()
    lines = h1 + h2
    labels_to_show = l1 + l2
    
    # Place unified legend outside and below, aligning with the other panels
    # Use upper center and -0.15 for consistent spacing
    ax.legend(
        handles=lines,
        labels=labels_to_show,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=7.5,
        framealpha=0.85,
        borderaxespad=0,
    )

    ax.spines[["top"]].set_visible(False)
    ax2_twin.spines[["top"]].set_visible(False)

    fig.suptitle("Shield ablation — weaned agent (50 episodes, seed-aligned)",
                 fontsize=11, y=1.01)
    
    # FIX: Use rect=[0, 0.08, 1, 1] to manually create a large bottom margin
    # This reserves 8% of the figure height at the bottom for the legends.
    fig.tight_layout(pad=1.5, rect=[0, 0.08, 1, 1])
    fig.subplots_adjust(bottom=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    print(f"Saved fixed figure -> {out_path}")
    plt.close(fig)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Path to ablation JSON produced by main_eval.py --out")
    p.add_argument("--out_dir", default="Memoria/imgs",
                   help="Output directory for figures (default: Memoria/imgs)")
    p.add_argument("--prefix", default="shield_ablation",
                   help="Filename prefix (default: shield_ablation)")
    args = p.parse_args()

    data = _load_json(args.input)
    results = data["results"]
    if not results:
        print("No results found in JSON.")
        return

    out_dir = Path(args.out_dir)
    prefix  = args.prefix

    print(f"\nLoaded {len(results)} shield configuration(s) from '{args.input}':")
    for r in results:
        print(f"  {_label(r['shield_type']):<20} "
              f"success={r['success_rate']:.1%}  "
              f"crash={r['crash_rate']:.1%}  "
              f"offroad={r['offroad_rate']:.1%}  "
              f"shields/ep={r['shields_per_ep']:.1f}")

    plot_outcome_bars(results,    out_dir / f"{prefix}_outcomes.pdf")
    plot_safety_comparison(results, out_dir / f"{prefix}_safety.pdf")
    plot_combined(results,         out_dir / f"{prefix}.pdf")


if __name__ == "__main__":
    main()
