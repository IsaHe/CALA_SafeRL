"""
scripts/gen_motif_figures.py  --  Generate the Chapter 8 figures
(LLM-derived reward + exposure) for the thesis.

Run once (no CARLA required, but needs the run SQLite DBs, the two 16-channel
ablation JSONs, observations_16ch.npz and reward_model_16ch.pth):

    python scripts/gen_motif_figures.py

Outputs (Memoria/imgs/):
    motif_encounter.pdf      encounter coverage vs training episode (base16 vs motif16)
    motif_ablation.pdf       shield-OFF / shield-ON outcome bars (base16 vs motif16)
    motif_reward_field.pdf   r_phi as a function of front clearance (monotonicity)
"""

import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent
IMGS = ROOT / "Memoria" / "imgs"
IMGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

DB_BASE = ROOT / "runs/base16_adaptive_20260615-085604/metrics.sqlite"
DB_MOTIF = ROOT / "runs/motif16_c3_adaptive_20260615-171209/metrics.sqlite"
JSON_BASE = ROOT / "base16_ablation.json"
JSON_MOTIF = ROOT / "motif16_ablation.json"
OBS_16 = ROOT / "data/preferences/observations_16ch.npz"
RM_16 = ROOT / "data/models/reward_model_16ch.pth"

# palette (shared with gen_thesis_figures.py)
C_BASE = "#7f8c8d"   # control (grey)
C_MOTIF = "#8e44ad"  # method (purple)
C_GREEN = "#2ecc71"
C_RED = "#e74c3c"
C_ORANGE = "#e67e22"


def qry(conn, metric, axis="episode"):
    rows = conn.execute(
        "SELECT step, value FROM metric_events WHERE metric_name=? AND axis=?"
        " ORDER BY step", (metric, axis)).fetchall()
    if not rows:
        return np.array([]), np.array([])
    steps, vals = zip(*rows)
    return np.array(steps), np.array(vals, dtype=float)


def roll(y, w=100):
    if len(y) < 2:
        return y
    k = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(y, (pad, pad), "edge"), k, "valid")[:len(y)]


def trim(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1 - encounter coverage over training
# ═══════════════════════════════════════════════════════════════════════════
def fig_encounter():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for db, label, color in [
        (DB_BASE, "base16 (global traffic only)", C_BASE),
        (DB_MOTIF, "motif16 (+ exposure director)", C_MOTIF),
    ]:
        conn = sqlite3.connect(str(db))
        ep, er = qry(conn, "Safety/Encounter_Rate")
        conn.close()
        if len(ep):
            ax.plot(ep, roll(er, 100), color=color, lw=1.8, label=label)
    ax.axhline(0.03, color=C_RED, ls="--", lw=1.2)
    ax.text(ax.get_xlim()[1], 0.045, "3-channel LIDAR ceiling (~3%)",
            color=C_RED, ha="right", va="bottom", fontsize=8.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Encounter rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Vehicle-encounter coverage during training")
    ax.legend(loc="center right")
    trim(ax)
    fig.tight_layout()
    fig.savefig(IMGS / "motif_encounter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote motif_encounter.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2 - ablation outcome bars (shield-OFF / shield-ON)
# ═══════════════════════════════════════════════════════════════════════════
def _result(path, shield):
    d = json.load(open(path))
    for r in d["results"]:
        if r["shield_type"] == shield:
            return r
    raise KeyError(shield)


def fig_ablation():
    cats = ["success", "crash", "offroad", "stuck"]
    labels = ["Success", "Crash", "Off-road", "Stuck"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, shield, title in [
        (axes[0], "none", "Shield OFF (deployed policy)"),
        (axes[1], "adaptive", "Shield ON"),
    ]:
        rb = _result(JSON_BASE, shield)
        rm = _result(JSON_MOTIF, shield)
        x = np.arange(len(cats))
        w = 0.38
        vb = [rb[f"{c}_rate"] for c in cats]
        vm = [rm[f"{c}_rate"] for c in cats]
        ax.bar(x - w / 2, vb, w, color=C_BASE, label="base16 (control)")
        ax.bar(x + w / 2, vm, w, color=C_MOTIF, label="motif16 (method)")
        for xi, (a, b) in enumerate(zip(vb, vm)):
            ax.text(xi - w / 2, a + 0.01, f"{a:.0%}", ha="center", fontsize=7.5)
            ax.text(xi + w / 2, b + 0.01, f"{b:.0%}", ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        trim(ax)
    axes[0].set_ylabel("Episode rate")
    axes[0].legend(loc="upper right")
    fig.suptitle("Outcome distribution at 60 NPCs (50 seed-aligned episodes)", y=1.02)
    fig.tight_layout()
    fig.savefig(IMGS / "motif_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote motif_ablation.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 - reward model monotonicity (r_phi vs front clearance)
# ═══════════════════════════════════════════════════════════════════════════
def fig_reward_field():
    from src.imitation.reward_model import load_reward_model
    from src.meta.scene_caption import extract_scene_features

    obs = np.load(OBS_16)["obs"].astype(np.float32)
    model = load_reward_model(str(RM_16))
    r = model.predict_np(obs)

    front = np.array([extract_scene_features(o).front_dynamic_m for o in obs])
    front = np.clip(front, 0, 50)

    edges = np.array([0, 5, 10, 15, 20, 30, 50.0001])
    centers, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (front >= lo) & (front < hi)
        if m.sum() >= 5:
            centers.append((lo + min(hi, 50)) / 2)
            means.append(r[m].mean())
            stds.append(r[m].std())
    centers, means, stds = map(np.array, (centers, means, stds))

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.errorbar(centers, means, yerr=stds, fmt="o-", color=C_MOTIF,
                ecolor=C_MOTIF, elinewidth=1, capsize=3, lw=1.8, ms=5)
    ax.set_xlabel("Nearest in-path vehicle distance (m)")
    ax.set_ylabel(r"Learned reward $r_\phi$")
    ax.set_title(r"$r_\phi$ increases monotonically with front clearance")
    trim(ax)
    fig.tight_layout()
    fig.savefig(IMGS / "motif_reward_field.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote motif_reward_field.pdf")


if __name__ == "__main__":
    fig_encounter()
    fig_ablation()
    fig_reward_field()
    print("done ->", IMGS)
