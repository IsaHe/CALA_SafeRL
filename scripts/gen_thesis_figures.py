"""
scripts/gen_thesis_figures.py  --  Generate ALL Chapter 6 figures for the thesis.

Run once (no CARLA required):
    python scripts/gen_thesis_figures.py
"""

import json
import sqlite3
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

ROOT   = Path(__file__).resolve().parent.parent
IMGS   = ROOT / "Memoria" / "imgs"
IMGS.mkdir(parents=True, exist_ok=True)

# ── paths ──────────────────────────────────────────────────────────────────────
DB_MAIN   = ROOT / "runs/learn_baseline_adaptive_20260607-150805/metrics.sqlite"
DB_FIX2   = ROOT / "runs/learn_fix2_adaptive_20260607-232554/metrics.sqlite"
JSON_3WAY = ROOT / "data/results/baseline_3way_ablation.json"
JSON_FIX2 = ROOT / "data/results/learn_fix2_ablation.json"

# ── helpers ────────────────────────────────────────────────────────────────────

def qry(conn, metric, axis="episode", last=None):
    if last:
        rows = conn.execute(
            "SELECT step, value FROM metric_events WHERE metric_name=? AND axis=?"
            " ORDER BY step DESC LIMIT ?", (metric, axis, last)).fetchall()
        rows = list(reversed(rows))
    else:
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


def pct(ax):
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))


def trim(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 – Main adaptive-shield training curves (learn_baseline)
# ══════════════════════════════════════════════════════════════════════════════

def fig_training_main():
    conn = sqlite3.connect(str(DB_MAIN))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    (ax_r, ax_sc), (ax_sh, ax_npc) = axes

    # Panel A – rolling reward
    ep, r = qry(conn, "Reward/Raw_Episode")
    ax_r.plot(ep, roll(r, 50), color="#2980b9", lw=1.6, label="50-ep rolling mean")
    ax_r.fill_between(ep, roll(r, 50) - roll(np.abs(r - roll(r, 50)), 50),
                      roll(r, 50) + roll(np.abs(r - roll(r, 50)), 50),
                      alpha=0.15, color="#2980b9")
    ax_r.set_title("Episode reward"); ax_r.set_ylabel("Reward"); ax_r.set_xlabel("Episode")
    ax_r.legend(); trim(ax_r)

    # Panel B – safety rates
    for metric, label, color in [
        ("Training/Success_Rate", "Success",  "#2ecc71"),
        ("Training/Crash_Rate",   "Crash",    "#e74c3c"),
        ("Training/Offroad_Rate", "Off-road", "#e67e22"),
    ]:
        ep, v = qry(conn, metric)
        if len(ep):
            ax_sc.plot(ep, v, lw=1.3, label=label, color=color, alpha=0.9)
    ax_sc.set_title("Safety rates (100-ep rolling)"); ax_sc.set_ylabel("Rate")
    ax_sc.set_xlabel("Episode"); pct(ax_sc); ax_sc.legend(); trim(ax_sc)

    # Panel C – shield activation rate
    ep, sh = qry(conn, "Safety/Shield_Rate")
    if len(ep):
        ax_sh.plot(ep, roll(sh, 50), color="#8e44ad", lw=1.4)
    ax_sh.set_title("Shield activation rate"); ax_sh.set_ylabel("Fraction of steps")
    ax_sh.set_xlabel("Episode"); pct(ax_sh); trim(ax_sh)

    # Panel D – curriculum NPC count
    ep, npc = qry(conn, "Training/Curriculum_NPC")
    if len(ep):
        ax_npc.step(ep, npc, color="#16a085", lw=1.6, where="post")
    ax_npc.set_title("Curriculum NPC count"); ax_npc.set_ylabel("Active NPCs")
    ax_npc.set_xlabel("Episode"); trim(ax_npc)
    ax_npc.grid(axis="y", linestyle=":", alpha=0.35)

    fig.suptitle(
        "Adaptive-shield training (\\texttt{learn\\_baseline}, 1035 episodes, 60 NPCs)",
        fontsize=11, y=1.01)
    fig.tight_layout(pad=2.0)
    out = IMGS / "training_main.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 – Risk level & intervention breakdown (learn_baseline)
# ══════════════════════════════════════════════════════════════════════════════

def fig_risk_interventions():
    conn = sqlite3.connect(str(DB_MAIN))
    fig, (ax_risk, ax_int) = plt.subplots(1, 2, figsize=(9, 3.8))

    # Risk levels over training
    for metric, label, color in [
        ("Safety/Semantic/Safe_Step_Rate",     "Safe",     "#2ecc71"),
        ("Safety/Semantic/Warning_Step_Rate",  "Warning",  "#f39c12"),
        ("Safety/Semantic/Critical_Step_Rate", "Critical", "#e74c3c"),
    ]:
        ep, v = qry(conn, metric)
        if len(ep):
            ax_risk.plot(ep, roll(v, 50), lw=1.4, label=label, color=color)
    ax_risk.set_title("Risk-level distribution (adaptive shield)")
    ax_risk.set_ylabel("Fraction of steps"); ax_risk.set_xlabel("Episode")
    pct(ax_risk); ax_risk.legend(); trim(ax_risk)

    # Intervention breakdown (last 100 eps)
    cats   = ["Dynamic", "Static", "Pedestrian"]
    keys   = ["Safety/Semantic/Dynamic_Interventions",
              "Safety/Semantic/Static_Interventions",
              "Safety/Semantic/Pedestrian_Interventions"]
    colors = ["#3498db", "#95a5a6", "#e91e63"]
    means  = []
    for k in keys:
        _, v = qry(conn, k, last=100)
        means.append(float(np.mean(v)) if len(v) else 0.0)
    bars = ax_int.bar(cats, means, color=colors, edgecolor="white", width=0.55)
    for b, m in zip(bars, means):
        ax_int.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{m:.1f}", ha="center", fontsize=9)
    ax_int.set_title("Interventions per episode by category\n(last 100 episodes)")
    ax_int.set_ylabel("Mean interventions / episode"); trim(ax_int)
    ax_int.grid(axis="y", linestyle=":", alpha=0.35)

    fig.tight_layout(pad=2.0)
    out = IMGS / "training_risk.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 – 3-way shield comparison (FullSendC1_p4 under none/basic/adaptive)
# ══════════════════════════════════════════════════════════════════════════════

OUTCOME_ORDER  = ["success", "crash", "offroad", "stuck", "timeout"]
OUTCOME_LABELS = ["Success", "Crash", "Off-road", "Stuck", "Timeout"]
OUTCOME_COLORS = ["#2ecc71", "#e74c3c", "#e67e22", "#9b59b6", "#95a5a6"]
SHIELD_DISPLAY = {"none": "No shield", "basic": "Basic", "adaptive": "Adaptive"}


def fig_3way_comparison():
    with open(JSON_3WAY, encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]

    n      = len(results)
    x      = np.arange(n)
    labels = [SHIELD_DISPLAY.get(r["shield_type"], r["shield_type"]) for r in results]
    w      = 0.30

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    # Panel A – stacked outcomes
    ax = axes[0]
    btm = np.zeros(n)
    for key, lbl, col in zip(OUTCOME_ORDER, OUTCOME_LABELS, OUTCOME_COLORS):
        vals = np.array([r[f"{key}_rate"] for r in results])
        ax.bar(x, vals, bottom=btm, color=col, label=lbl, width=0.55,
               edgecolor="white", linewidth=0.4)
        btm += vals
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); pct(ax)
    ax.set_title("Outcome distribution"); trim(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3,
              fontsize=8, framealpha=0.85)

    # Panel B – crash + offroad grouped
    ax = axes[1]
    crash   = np.array([r["crash_rate"]   for r in results])
    offroad = np.array([r["offroad_rate"] for r in results])
    ax.bar(x - w / 2, crash,   w, label="Crash",    color="#e74c3c", edgecolor="white")
    ax.bar(x + w / 2, offroad, w, label="Off-road", color="#e67e22", edgecolor="white")
    for xi, c, o in zip(x, crash, offroad):
        ax.text(xi - w / 2, c + 0.01, f"{c:.0%}", ha="center", fontsize=8)
        ax.text(xi + w / 2, o + 0.01, f"{o:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels); pct(ax)
    ax.set_title("Crash and off-road rates")
    ax.legend(framealpha=0.85); trim(ax)

    # Panel C – shields/ep
    ax = axes[2]
    shld = np.array([r["shields_per_ep"] for r in results])
    bars = ax.bar(x, shld, 0.55, color="#3498db", edgecolor="white")
    for b, v in zip(bars, shld):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Shield interventions / episode"); trim(ax)
    ax.set_ylabel("Interventions")
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    fig.suptitle(
        "Shield comparison -- same model (\\texttt{FullSendC1\\_p4}), 50 episodes, 20 NPCs",
        fontsize=10, y=1.02)
    fig.tight_layout(pad=2.0)
    out = IMGS / "shield_comparison_3way.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 – Curriculum NPC trajectory (standalone for chapter)
# ══════════════════════════════════════════════════════════════════════════════

def fig_curriculum():
    conn = sqlite3.connect(str(DB_MAIN))
    ep, npc = qry(conn, "Training/Curriculum_NPC")
    ep_r, r = qry(conn, "Reward/Raw_Episode")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    # NPC count
    ax1.step(ep, npc, color="#16a085", lw=1.8, where="post", label="NPC count")
    ax1.set_ylabel("Active NPCs"); ax1.set_ylim(-2, max(npc) * 1.15)
    ax1.legend(loc="upper left"); trim(ax1)
    ax1.set_title("Curriculum progression and episode reward over training")

    # Rolling reward
    ax2.plot(ep_r, roll(r, 50), color="#2980b9", lw=1.4, label="50-ep rolling reward")
    ax2.set_ylabel("Reward"); ax2.set_xlabel("Episode"); ax2.legend(loc="upper left")
    trim(ax2)

    fig.tight_layout(pad=1.5)
    out = IMGS / "curriculum_npc.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 – PPO diagnostics (log_std, KL, entropy, grad norm)
# ══════════════════════════════════════════════════════════════════════════════

def fig_ppo_diagnostics():
    conn = sqlite3.connect(str(DB_MAIN))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    metrics_upd = [
        ("Training/Log_Std_Steering_Raw",      "log sigma (steering)",       "#8e44ad"),
        ("Training/Log_Std_Throttle_Raw",       "log sigma (throttle)",       "#2980b9"),
        ("Training/Log_Std_Saturated_Fraction", "Saturated fraction",         "#e74c3c"),
        ("Training/Approx_KL",                  "Approx KL",                  "#e67e22"),
        ("Training/Entropy",                    "Policy entropy",             "#27ae60"),
        ("Loss/Grad_Norm",                      "Gradient norm",              "#7f8c8d"),
    ]
    for ax, (key, title, color) in zip(axes.flat, metrics_upd):
        step, val = qry(conn, key, axis="update")
        if len(step):
            ax.plot(step, val, color=color, lw=0.8, alpha=0.7)
        ax.set_title(title); ax.set_xlabel("Timestep"); trim(ax)

    # Mark log_std bounds
    LOG_MIN, LOG_MAX = -3.0, -1.2
    for ax in axes.flat[:2]:
        ax.axhline(LOG_MAX, color="red",  linestyle="--", lw=0.9,
                   alpha=0.7, label="MAX")
        ax.axhline(LOG_MIN, color="blue", linestyle="--", lw=0.9,
                   alpha=0.7, label="MIN")
        ax.legend(fontsize=8)

    fig.suptitle("PPO update diagnostics (\\texttt{learn\\_baseline})",
                 fontsize=11, y=1.01)
    fig.tight_layout(pad=2.0)
    out = IMGS / "ppo_diagnostics.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 – Shield-off probe comparison (baseline vs weaned)
# ══════════════════════════════════════════════════════════════════════════════

def fig_probe_comparison():
    conn_base = sqlite3.connect(str(DB_MAIN))
    conn_fix2 = sqlite3.connect(str(DB_FIX2))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    pairs = [
        ("Eval/ShieldOff_Success",  "Shield-off success"),
        ("Eval/ShieldOff_Crash",    "Shield-off crash"),
    ]
    for (conn, label, color) in [
        (conn_base, "Pre-weaning (learn\\_baseline)", "#e74c3c"),
        (conn_fix2, "Post-weaning (learn\\_fix2)",    "#2ecc71"),
    ]:
        ep_s, s = qry(conn, "Eval/ShieldOff_Success")
        ep_c, c = qry(conn, "Eval/ShieldOff_Crash")
        ep_sh, sh = qry(conn, "Safety/Shield_Rate")

        if len(ep_s):
            axes[0].scatter(ep_s, s, s=25, color=color, alpha=0.65, zorder=3)
            if len(ep_s) > 3:
                axes[0].plot(ep_s, roll(s, min(len(s), 5)), color=color, lw=1.8, label=label)
        if len(ep_c):
            axes[1].scatter(ep_c, c, s=25, color=color, alpha=0.65, zorder=3)
            if len(ep_c) > 3:
                axes[1].plot(ep_c, roll(c, min(len(c), 5)), color=color, lw=1.8, label=label)
        if len(ep_sh):
            axes[2].plot(ep_sh, roll(sh, 50), color=color, lw=1.5, label=label)

    for ax, title in zip(axes, ["Shield-off success rate",
                                  "Shield-off crash rate",
                                  "Shield activation rate (training)"]):
        ax.set_title(title); ax.set_xlabel("Episode"); pct(ax); trim(ax)
        ax.legend(fontsize=8, framealpha=0.85)

    axes[0].set_ylabel("Rate")
    fig.suptitle("Shield-off probe: pre-weaning vs post-weaning",
                 fontsize=11, y=1.02)
    fig.tight_layout(pad=1.5)
    out = IMGS / "shield_off_probe.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)
    conn_base.close()
    conn_fix2.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 – LLM negative result (FinalAssistLLM run)
# ══════════════════════════════════════════════════════════════════════════════

def fig_llm_negative():
    llm_dbs = sorted(
        (ROOT / "runs").glob("FinalAssistLLM_adaptive_*/metrics.sqlite"))
    if not llm_dbs:
        print("No FinalAssistLLM run found -- skipping llm_negative.")
        return
    conn = sqlite3.connect(str(llm_dbs[-1]))
    cur = conn.execute(
        "SELECT episode, applied_permissiveness, shielded_fraction, crash_rate "
        "FROM meta_decisions ORDER BY episode")
    rows = cur.fetchall()
    if not rows:
        print("No meta_decisions rows -- skipping llm_negative.")
        conn.close()
        return

    eps, perm, sh, cr = (np.array(col, dtype=float) for col in zip(*rows))
    rho_sh = float(np.corrcoef(perm, sh)[0, 1]) if len(perm) > 1 else 0.0
    rho_cr = float(np.corrcoef(perm, cr)[0, 1]) if len(perm) > 1 else 0.0

    fig, ax1 = plt.subplots(figsize=(8, 3.8))
    ax2 = ax1.twinx()

    c_p, c_s, c_c = "#2980b9", "#e74c3c", "#f39c12"
    l1, = ax1.plot(eps, perm, color=c_p, lw=2.2, label="Permissiveness $p$")
    l2, = ax2.plot(eps, sh,   color=c_s, lw=1.6, linestyle="--",
                   label="Shield activation rate")
    l3, = ax2.plot(eps, cr,   color=c_c, lw=1.2, linestyle=":",
                   label="Crash rate", alpha=0.8)

    ax1.text(0.02, 0.97,
             f"$\\rho$(perm, shield) $\\approx$ {rho_sh:+.2f}\n"
             f"$\\rho$(perm, crash) $\\approx$ {rho_cr:+.2f}",
             transform=ax1.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#cccccc", alpha=0.9))

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Permissiveness $p$", color=c_p)
    ax2.set_ylabel("Rate", color=c_s)
    ax1.tick_params(axis="y", colors=c_p)
    ax2.tick_params(axis="y", colors=c_s)

    ax1.legend(handles=[l1, l2, l3], loc="lower left", framealpha=0.85)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax1.grid(axis="x", linestyle=":", alpha=0.3)

    fig.suptitle(
        "LLM meta-controller: loosening the shield does not reduce reliance",
        fontsize=10, y=1.02)
    fig.tight_layout()
    out = IMGS / "llm_negative.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 – Throttle-collapse comparison (steering-only vs steering+throttle BC)
# ══════════════════════════════════════════════════════════════════════════════

def fig_throttle_collapse():
    """
    Compare shield-off success + mean speed for: baseline (no BC),
    learn_baseline (steering-only BC), and learn_fix2 (steering BC, more training).
    Uses the probe metrics logged to SQLite.
    """
    sources = [
        (DB_MAIN,  "Steering BC only (learn\\_baseline)",  "#2ecc71"),
        (DB_FIX2,  "Steering BC + more training (learn\\_fix2)", "#3498db"),
    ]

    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(9, 3.8))

    for db_path, label, color in sources:
        conn = sqlite3.connect(str(db_path))
        ep_s, s = qry(conn, "Eval/ShieldOff_Success")
        ep_r, r = qry(conn, "CARLA/Mean_Speed_kmh")
        if len(ep_s):
            ax_s.scatter(ep_s, s, s=22, color=color, alpha=0.6, zorder=3)
            ax_s.plot(ep_s, roll(s, min(len(s), 5)), color=color, lw=1.8, label=label)
        if len(ep_r):
            ax_r.plot(ep_r, roll(r, 50), color=color, lw=1.5, label=label)
        conn.close()

    ax_s.set_title("Shield-off success (probe)"); pct(ax_s)
    ax_s.set_xlabel("Episode"); ax_s.set_ylabel("Success rate")
    ax_s.legend(fontsize=8); trim(ax_s)

    ax_r.set_title("Mean speed during training")
    ax_r.set_xlabel("Episode"); ax_r.set_ylabel("km/h")
    ax_r.legend(fontsize=8); trim(ax_r)

    fig.suptitle("BC teacher effect: probe success and speed over training",
                 fontsize=10, y=1.02)
    fig.tight_layout(pad=1.5)
    out = IMGS / "throttle_collapse.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating thesis figures...")
    fig_training_main()
    fig_risk_interventions()
    fig_3way_comparison()
    fig_curriculum()
    fig_ppo_diagnostics()
    fig_probe_comparison()
    fig_llm_negative()
    fig_throttle_collapse()
    print("All figures generated.")
