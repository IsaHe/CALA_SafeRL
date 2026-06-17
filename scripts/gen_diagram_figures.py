"""
scripts/gen_diagram_figures.py -- Generate the conceptual/diagram figures of the thesis
as clean vector PDFs (no CARLA, no training data required).

Outputs (Memoria/imgs/*.pdf, which pdflatex prefers over the old *.png):
    ppo_surrogate      (Fig 4.1)   bicycle_model     (Fig 4.2)
    wrapper_chain      (Fig 6.1)   lidar_channels    (Fig 6.2)
    actor_critic_arch  (Fig 6.3)   reward_overview   (Fig 6.4)
    risk_levels        (Fig 6.5)   curriculum_stages (Fig 6.6)

Run:
    python scripts/gen_diagram_figures.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

matplotlib.rcParams.update({
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
})

IMGS = Path(__file__).resolve().parent.parent / "Memoria" / "imgs"
IMGS.mkdir(parents=True, exist_ok=True)

# palette ---------------------------------------------------------------------
C_ACTION = "#e67e22"   # downward action path
C_OBS = "#2980b9"      # upward obs/reward path
C_BOX = "#eef2f6"
C_EDGE = "#34495e"
C_GREEN = "#27ae60"
C_AMBER = "#f39c12"
C_RED = "#e74c3c"
C_GREY = "#95a5a6"


def _save(fig, name):
    out = IMGS / f"{name}.pdf"
    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def _box(ax, x, y, w, h, text, fc=C_BOX, ec=C_EDGE, fs=10, bold=False, tc="black"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2, clip_on=False))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, zorder=3)


# =============================================================================
# Fig 4.1 -- PPO clipped surrogate objective
# =============================================================================
def fig_ppo_surrogate():
    eps = 0.2
    r = np.linspace(0, 2, 400)
    A_pos, A_neg = 1.0, -1.0
    clip = np.clip(r, 1 - eps, 1 + eps)
    L_pos = np.minimum(r * A_pos, clip * A_pos)
    L_neg = np.minimum(r * A_neg, clip * A_neg)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(r, L_pos, color="#2980b9", lw=2.2, label=r"$\hat{A}_t > 0$  (advantage positive)")
    ax.plot(r, L_neg, color="#c0392b", lw=2.2, ls="--",
            label=r"$\hat{A}_t < 0$  (advantage negative)")

    for xc in (1 - eps, 1 + eps):
        ax.axvline(xc, color="#7f8c8d", ls=":", lw=1.0)
    ax.axvline(1.0, color="#bdc3c7", ls="-", lw=0.8)
    ax.axhline(0.0, color="#bdc3c7", lw=0.8)

    # clip markers
    ax.plot([1 + eps], [1 + eps], "o", color="#2980b9", ms=5, zorder=5)
    ax.plot([1 - eps], [-(1 - eps)], "o", color="#c0392b", ms=5, zorder=5)
    ax.annotate("clipped", xy=(1.6, 1.2), xytext=(1.45, 1.62), color="#2980b9",
                fontsize=9, ha="center")
    ax.annotate("clipped", xy=(0.4, -0.8), xytext=(0.42, -1.35), color="#c0392b",
                fontsize=9, ha="center")

    ax.set_xticks([0, 0.8, 1.0, 1.2, 2.0])
    ax.set_xticklabels(["0", r"$1-\epsilon$", "1", r"$1+\epsilon$", "2"])
    ax.set_xlabel(r"importance-sampling ratio $r_t(\theta)$")
    ax.set_ylabel(r"$\mathcal{L}^{\mathrm{CLIP}}$ contribution")
    ax.set_xlim(0, 2)
    ax.set_ylim(-1.7, 1.8)
    ax.text(0.815, -1.62, r"$\epsilon=0.2$", fontsize=9, color="#7f8c8d")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, "ppo_surrogate")


# =============================================================================
# Fig 4.2 -- Kinematic bicycle model
# =============================================================================
def fig_bicycle_model():
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.set_aspect("equal")
    ax.axis("off")

    L = 4.2                      # drawn wheelbase
    theta = np.deg2rad(18)       # heading
    delta = np.deg2rad(28)       # steering
    rx, ry = 0.0, 0.0            # rear axle
    fx = rx + L * np.cos(theta)
    fy = ry + L * np.sin(theta)

    # chassis
    ax.plot([rx, fx], [ry, fy], color=C_EDGE, lw=2.2, zorder=3)

    def wheel(cx, cy, ang, c=C_EDGE):
        ww, wl = 0.28, 1.1
        dx, dy = wl / 2 * np.cos(ang), wl / 2 * np.sin(ang)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=c, lw=6,
                solid_capstyle="round", zorder=4)

    wheel(rx, ry, theta)                 # rear wheel (aligned with body)
    wheel(fx, fy, theta + delta, "#c0392b")  # front wheel (steered)

    # heading arrow
    hx, hy = 2.6 * np.cos(theta), 2.6 * np.sin(theta)
    ax.add_patch(FancyArrowPatch((fx, fy), (fx + hx, fy + hy),
                 arrowstyle="-|>", mutation_scale=16, color="#2980b9", lw=1.6, zorder=3))
    ax.text(fx + hx + 0.15, fy + hy + 0.15, r"$v$", color="#2980b9", fontsize=12)

    # predicted arc
    R = L / np.tan(delta)
    cxc = rx - R * np.sin(theta)
    cyc = ry + R * np.cos(theta)
    phis = np.linspace(0, 0.9, 60)
    ax_arc = cxc + R * np.sin(theta + phis)
    ay_arc = cyc - R * np.cos(theta + phis)
    ax.plot(ax_arc, ay_arc, color=C_GREEN, lw=1.8, ls="--", zorder=2)
    ax.text(ax_arc[-1] + 0.1, ay_arc[-1], "predicted\ntrajectory",
            color=C_GREEN, fontsize=8.5, va="center")

    # heading angle theta from horizontal at rear axle
    ax.plot([rx, rx + 2.0], [ry, ry], color="#bdc3c7", lw=1.0, zorder=1)
    tt = np.linspace(0, theta, 30)
    ax.plot(rx + 0.9 * np.cos(tt), ry + 0.9 * np.sin(tt), color=C_GREY, lw=1.0)
    ax.text(rx + 1.05, ry + 0.18, r"$\theta$", fontsize=12)

    # steering angle delta at front wheel (relative to body line)
    body_dir = theta
    dd = np.linspace(body_dir, body_dir + delta, 30)
    ax.plot(fx + 0.95 * np.cos(dd), fy + 0.95 * np.sin(dd), color="#c0392b", lw=1.0)
    ax.text(fx + 1.05 * np.cos(body_dir + delta / 2),
            fy + 1.05 * np.sin(body_dir + delta / 2) + 0.12, r"$\delta$",
            fontsize=12, color="#c0392b")

    # rear axle point + label (x, y)
    ax.plot(rx, ry, "o", color=C_EDGE, ms=6, zorder=5)
    ax.text(rx - 0.15, ry - 0.45, r"$(x,\,y)$", fontsize=11, ha="center")

    # wheelbase L bracket
    ax.annotate("", xy=(fx, fy - 0.9), xytext=(rx, ry - 0.9),
                arrowprops=dict(arrowstyle="<->", color=C_GREY, lw=1.0))
    ax.text((rx + fx) / 2, (ry + fy) / 2 - 1.15, r"$L = 2.87\,$m",
            fontsize=10, ha="center", color=C_GREY)

    ax.set_xlim(-1.2, 8.0)
    ax.set_ylim(-2.0, 4.2)
    _save(fig, "bicycle_model")


# =============================================================================
# Fig 6.1 -- Wrapper chain
# =============================================================================
def fig_wrapper_chain():
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)

    layers = [
        ("PPOAgent", "produces action $a_t$; receives shaped obs and reward", "#dfe9f3"),
        ("CarlaRewardShaper", "outermost wrapper: shapes the reward $\\tilde{r}_t$", "#eaf6ec"),
        ("Safety Shield", "none / basic / adaptive: may replace $a_t$ with $a_t^{\\star}$", "#fdebd0"),
        ("CarlaEnv", "gymnasium.Env wrapping the CARLA Python API", "#eef2f6"),
        ("CARLA Simulator", "CarlaUE4.exe (synchronous, 20 Hz)", "#eaecee"),
    ]
    ys = np.linspace(10.6, 1.4, len(layers))
    w, h = 5.4, 1.5
    cx = 5.0
    for (title, sub, fc), y in zip(layers, ys):
        ax.add_patch(FancyBboxPatch((cx - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     linewidth=1.3, edgecolor=C_EDGE, facecolor=fc, zorder=2))
        ax.text(cx, y + 0.22, title, ha="center", va="center", fontsize=11.5,
                fontweight="bold", zorder=3)
        ax.text(cx, y - 0.34, sub, ha="center", va="center", fontsize=8.3,
                color="#555", zorder=3)

    # arrows: action path (down, left side) and obs/reward path (up, right side)
    xL = cx - w / 2 - 0.55
    xR = cx + w / 2 + 0.55
    for i in range(len(layers) - 1):
        y0, y1 = ys[i] - h / 2, ys[i + 1] + h / 2
        ax.add_patch(FancyArrowPatch((xL, y0 - 0.02), (xL, y1 + 0.02),
                     arrowstyle="-|>", mutation_scale=15, color=C_ACTION, lw=2.0,
                     connectionstyle="arc3,rad=0"))
        ax.add_patch(FancyArrowPatch((xR, y1 + 0.02), (xR, y0 - 0.02),
                     arrowstyle="-|>", mutation_scale=15, color=C_OBS, lw=2.0,
                     connectionstyle="arc3,rad=0"))
    ax.text(xL - 0.18, 6.0, "action  $a_t$", color=C_ACTION, fontsize=9.5,
            rotation=90, ha="center", va="center", fontweight="bold")
    ax.text(xR + 0.18, 6.0, "observation / reward", color=C_OBS, fontsize=9.5,
            rotation=90, ha="center", va="center", fontweight="bold")
    _save(fig, "wrapper_chain")


# =============================================================================
# Fig 6.2 -- Three semantic LIDAR channels (bird's-eye view)
# =============================================================================
def fig_lidar_channels():
    """Top-down view of one LIDAR scan split into the three semantic channels.

    The same scene (a slower lead vehicle ahead, a pedestrian near the right edge,
    guardrails on both sides) is shown three times. Each panel keeps only the
    returns of one channel; the combined channel is the per-ray nearest hit. A
    top-down scene is used instead of a polar range plot because the latter draws
    an obstacle ahead as a counter-intuitive inward notch.
    """
    MAX_R = 26.0
    RAIL_X = 7.0
    RAIL_Y = (-6.0, 30.0)
    CAR = (-1.0, 1.0, 11.0, 15.5)          # lead vehicle: xmin, xmax, ymin, ymax
    PED_C, PED_R = (5.3, 7.5), 0.55         # pedestrian near the right edge

    def hit_vline(dx, dy, x0, yr):
        if abs(dx) < 1e-9:
            return np.inf
        t = x0 / dx
        if t <= 1e-6:
            return np.inf
        return t if yr[0] <= t * dy <= yr[1] else np.inf

    def hit_box(dx, dy, box):
        xmin, xmax, ymin, ymax = box
        tmin, tmax = -np.inf, np.inf
        for d, lo, hi in ((dx, xmin, xmax), (dy, ymin, ymax)):
            if abs(d) < 1e-9:
                if not (lo <= 0 <= hi):
                    return np.inf
            else:
                t1, t2 = lo / d, hi / d
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmax < max(tmin, 1e-6):
            return np.inf
        return tmin if tmin > 1e-6 else tmax

    def hit_circle(dx, dy, c, r):
        b = -2.0 * (dx * c[0] + dy * c[1])
        cc = c[0] ** 2 + c[1] ** 2 - r ** 2
        disc = b * b - 4 * cc
        if disc < 0:
            return np.inf
        s = np.sqrt(disc)
        t = (-b - s) / 2.0
        if t <= 1e-6:
            t = (-b + s) / 2.0
        return t if t > 1e-6 else np.inf

    angles = np.deg2rad(np.arange(0, 360, 6))

    def cast(angle, see_static, see_dynamic):
        dx, dy = np.sin(angle), np.cos(angle)   # 0 deg = forward (+y)
        best_t, kind = MAX_R, "free"
        if see_static:
            for x0 in (-RAIL_X, RAIL_X):
                t = hit_vline(dx, dy, x0, RAIL_Y)
                if t < best_t:
                    best_t, kind = t, "static"
        if see_dynamic:
            for t in (hit_box(dx, dy, CAR), hit_circle(dx, dy, PED_C, PED_R)):
                if t < best_t:
                    best_t, kind = t, "dynamic"
        return dx * best_t, dy * best_t, kind

    C_DYN, C_STAT = "#e67e22", "#34495e"
    panels = [
        ("Combined  (obs[0:240])\nnearest of all obstacles", True, True),
        ("Dynamic  (obs[240:480])\nvehicles + pedestrians", False, True),
        ("Static  (obs[480:720])\nbarriers + guardrails", True, False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.8))
    for ax, (title, see_s, see_d) in zip(axes, panels):
        ax.add_patch(Rectangle((-RAIL_X, RAIL_Y[0]), 2 * RAIL_X,
                     RAIL_Y[1] - RAIL_Y[0], facecolor="#eceff1", edgecolor="none", zorder=0))
        for lane_x in (-RAIL_X / 3.0, RAIL_X / 3.0):
            ax.plot([lane_x, lane_x], [RAIL_Y[0], RAIL_Y[1]], color="#b0bec5",
                    lw=1.0, ls=(0, (6, 6)), zorder=1)

        # rays
        for a in angles:
            ex, ey, kind = cast(a, see_s, see_d)
            if kind == "free":
                ax.plot([0, ex], [0, ey], color="#90a4ae", lw=0.5, alpha=0.30, zorder=2)
            else:
                col = C_DYN if kind == "dynamic" else C_STAT
                ax.plot([0, ex], [0, ey], color=col, lw=0.8, alpha=0.55, zorder=3)
                ax.plot(ex, ey, marker="o", ms=2.6, color=col, zorder=4)

        # guardrails
        rail_col = C_STAT if see_s else "#cfd8dc"
        rail_lw = 3.2 if see_s else 1.4
        for x0 in (-RAIL_X, RAIL_X):
            ax.plot([x0, x0], RAIL_Y, color=rail_col, lw=rail_lw, zorder=3,
                    solid_capstyle="butt")

        # lead vehicle
        seen_d = see_d
        ax.add_patch(Rectangle((CAR[0], CAR[2]), CAR[1] - CAR[0], CAR[3] - CAR[2],
                     facecolor=(C_DYN if seen_d else "#d7dde0"),
                     edgecolor=(C_STAT if seen_d else "#b0bec5"),
                     lw=1.2, zorder=5, alpha=0.95 if seen_d else 0.6))
        # pedestrian
        ax.add_patch(plt.Circle(PED_C, PED_R, facecolor=(C_DYN if seen_d else "#d7dde0"),
                     edgecolor=(C_STAT if seen_d else "#b0bec5"), lw=1.0,
                     zorder=5, alpha=0.95 if seen_d else 0.6))

        # ego (always)
        ax.add_patch(Rectangle((-1.0, -2.25), 2.0, 4.5, facecolor="#2c3e50",
                     edgecolor="black", lw=1.0, zorder=6))
        ax.plot(0, 2.7, marker="^", color="#2c3e50", ms=9, zorder=6)

        if see_s and see_d:   # label only the combined panel
            ax.annotate("lead\nvehicle", (0, 13.2), color="white", fontsize=7.5,
                        ha="center", va="center", zorder=7, fontweight="bold")
            ax.annotate("pedestrian", (PED_C[0], PED_C[1] - 1.4), color=C_STAT,
                        fontsize=7.5, ha="center", va="top", zorder=7)
            ax.annotate("guardrail", (-RAIL_X, 26.5), color="white", fontsize=7.5,
                        ha="center", va="center", rotation=90, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc=C_STAT, ec="none"))
            ax.annotate("ego", (0, 0), color="white", fontsize=7.5, ha="center",
                        va="center", zorder=7, fontweight="bold")

        ax.set_xlim(-11, 11)
        ax.set_ylim(-7, 30)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9.5, pad=8)

    handles = [
        Line2D([0], [0], color=C_DYN, lw=2.4, marker="o", ms=4,
               label="dynamic return (vehicle / pedestrian)"),
        Line2D([0], [0], color=C_STAT, lw=2.4, marker="o", ms=4,
               label="static return (guardrail)"),
        Line2D([0], [0], color="#90a4ae", lw=1.2, alpha=0.6,
               label="free ray (no return within 50 m)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("One LIDAR scan, split into three semantic channels "
                 "(bird's-eye view, ego facing up)", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, "lidar_channels")


# =============================================================================
# Fig 6.3 -- Actor-Critic architecture
# =============================================================================
def fig_actor_critic_arch():
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)

    def arrow(p0, p1, c=C_EDGE):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                     mutation_scale=14, color=c, lw=1.4, zorder=1, clip_on=False))

    _box(ax, 1.6, 6.6, 2.4, 1.0, "obs[0:720]\nLIDAR (720)", fc="#dfe9f3", fs=9)
    _box(ax, 1.6, 2.6, 2.4, 1.0, "obs[720:739]\nvector (19)", fc="#dfe9f3", fs=9)
    _box(ax, 5.0, 6.6, 2.6, 1.2,
         "LIDAR encoder\n$720\\!\\to\\!256\\!\\to\\!64$", fc="#eaf6ec", fs=9)
    _box(ax, 7.9, 4.6, 1.5, 1.0, "concat\n(83)", fc="#fdebd0", fs=9)
    _box(ax, 10.4, 4.6, 2.6, 1.2,
         "shared trunk\n$83\\!\\to\\!256\\!\\to\\!256$", fc="#eaf6ec", fs=9)
    _box(ax, 13.0, 6.7, 2.0, 1.5,
         "actor head\n$\\to 2$ mean\n$+\\,\\log\\sigma$", fc="#f5e6f7", fs=8.5)
    _box(ax, 13.0, 2.5, 2.0, 1.1, "critic head\n$\\to V(s)$", fc="#f5e6f7", fs=8.5)

    arrow((2.8, 6.6), (3.7, 6.6))
    arrow((6.3, 6.6), (7.5, 5.1))
    arrow((2.8, 2.6), (7.5, 4.1))
    arrow((8.65, 4.6), (9.1, 4.6))
    arrow((11.7, 4.9), (12.0, 6.3))
    arrow((11.7, 4.3), (12.0, 2.7))

    ax.text(7.9, 3.5, r"$\phi(s_t)\in\mathbb{R}^{83}$", fontsize=9, ha="center",
            color="#555")
    ax.text(13.0, 5.4, r"TanhNormal$(\mu,\sigma)$", fontsize=8, ha="center",
            color="#555")
    ax.text(7.0, 8.4, "Actor-Critic network (shared representation, separate heads)",
            fontsize=11, ha="center", fontweight="bold")
    ax.set_xlim(-0.4, 14.7)
    ax.set_ylim(0, 9)
    _save(fig, "actor_critic_arch")


# =============================================================================
# Fig 6.4 -- Reward components overview
# =============================================================================
def fig_reward_overview():
    bonuses = ["Alive bonus (speed-gated)", "Speed bonus (Gaussian)",
               "Acceleration reward", "Lane centring", "Heading alignment",
               "Progress milestone (25 m)"]
    penalties = ["Smoothness (steering $\\Delta$)", "Solid-line invasion (cap 6)",
                 "Edge proximity (quadratic)", "Wrong-heading", "Drift",
                 "Idle (after grace)", "Lane-change cost", "Shield reliance"]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.text(2.6, 9.4, "Bonuses  (+)", fontsize=12, fontweight="bold",
            color=C_GREEN, ha="center")
    ax.text(7.4, 9.4, "Penalties  (-)", fontsize=12, fontweight="bold",
            color=C_RED, ha="center")

    def stack(items, x, color, fc):
        y = 8.6
        for it in items:
            ax.add_patch(FancyBboxPatch((x - 2.1, y - 0.36), 4.2, 0.62,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         linewidth=1.0, edgecolor=color, facecolor=fc, zorder=2))
            ax.text(x, y - 0.05, it, ha="center", va="center", fontsize=8.6)
            y -= 0.78

    stack(bonuses, 2.6, C_GREEN, "#eafaf0")
    stack(penalties, 7.4, C_RED, "#fdecea")

    # base reward strip at the bottom
    ax.add_patch(FancyBboxPatch((1.0, 0.5), 8.0, 0.95,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 linewidth=1.2, edgecolor=C_EDGE, facecolor="#eef2f6", zorder=2))
    ax.text(5.0, 0.97, r"sparse base reward:  success $+30$    crash $-10$    off-road $-30$",
            ha="center", va="center", fontsize=9.5, fontweight="bold")
    _save(fig, "reward_overview")


# =============================================================================
# Fig 6.5 -- Risk-level classification grid
# =============================================================================
def fig_risk_levels():
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # worst-of-two: rows = frontal (safe/warn/crit), cols = lateral (safe/warn/crit)
    levels = [["safe", "warning", "critical"],
              ["warning", "warning", "critical"],
              ["critical", "critical", "critical"]]
    colmap = {"safe": ("#eafaf0", C_GREEN), "warning": ("#fef5e7", C_AMBER),
              "critical": ("#fdecea", C_RED)}
    hmap = {"safe": "$H{=}1,\\ \\kappa{=}1.0$", "warning": "$H{=}5,\\ \\kappa{=}1.5$",
            "critical": "$H{=}10,\\ \\kappa{=}2.0$"}

    x0, y0, cell = 2.4, 1.8, 2.2
    col_labels = ["$<0.50$", "$0.50$–$0.70$", "$>0.70$"]   # |lateral offset|
    row_labels = ["$>0.50$", "$0.20$–$0.50$", "$\\leq 0.20$"]  # front dyn LIDAR
    for i in range(3):       # rows (frontal)
        for j in range(3):   # cols (lateral)
            lv = levels[i][j]
            fc, ec = colmap[lv]
            yy = y0 + (2 - i) * cell
            xx = x0 + j * cell
            ax.add_patch(Rectangle((xx, yy), cell, cell, facecolor=fc,
                         edgecolor=ec, lw=1.5, zorder=2))
            ax.text(xx + cell / 2, yy + cell / 2 + 0.28, lv, ha="center",
                    va="center", fontsize=10.5, fontweight="bold", color=ec)
            ax.text(xx + cell / 2, yy + cell / 2 - 0.42, hmap[lv], ha="center",
                    va="center", fontsize=8.2, color="#555")

    for j, cl in enumerate(col_labels):
        ax.text(x0 + j * cell + cell / 2, y0 + 3 * cell + 0.3, cl, ha="center",
                fontsize=9)
    for i, rl in enumerate(row_labels):
        ax.text(x0 - 0.3, y0 + (2 - i) * cell + cell / 2, rl, ha="right",
                va="center", fontsize=9)
    ax.text(x0 + 1.5 * cell, y0 + 3 * cell + 0.95,
            r"lateral-position criterion  $|\ell_t|$", ha="center", fontsize=10,
            fontweight="bold")
    ax.text(x0 - 1.55, y0 + 1.5 * cell,
            "frontal-distance\ncriterion  $d^{\\mathrm{dyn}}_t$", ha="center",
            va="center", fontsize=10, rotation=90, fontweight="bold")
    ax.set_xlim(0, x0 + 3 * cell + 0.5)
    ax.set_ylim(0, y0 + 3 * cell + 1.5)
    _save(fig, "risk_levels")


# =============================================================================
# Fig 6.6 -- Curriculum state machine
# =============================================================================
def fig_curriculum_stages():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.axis("off")
    xs = np.linspace(1.4, 12.6, 5)
    y = 3.0
    rad = 0.72
    npcs = [0, 15, 30, 45, 60]
    advance = [r"offroad $<20\%$" + "\n" + r"$\wedge\ R>0$",
               r"crash $<25\%$" + "\n" + r"$\wedge$ offroad $<15\%$",
               r"crash $<15\%$", r"crash $<10\%$"]

    for i, (x, npc) in enumerate(zip(xs, npcs)):
        ax.add_patch(plt.Circle((x, y), rad, facecolor="#dfe9f3",
                     edgecolor=C_EDGE, lw=1.6, zorder=3))
        ax.text(x, y + 0.13, f"Stage {i}", ha="center", va="center",
                fontsize=9.5, fontweight="bold", zorder=4)
        ax.text(x, y - 0.28, f"{npc} NPCs", ha="center", va="center",
                fontsize=8.5, color="#555", zorder=4)

    for i in range(4):
        x0, x1 = xs[i] + rad, xs[i + 1] - rad
        # forward (advance) arrow, arched above
        ax.add_patch(FancyArrowPatch((x0, y + 0.28), (x1, y + 0.28),
                     connectionstyle="arc3,rad=-0.32", arrowstyle="-|>",
                     mutation_scale=14, color=C_GREEN, lw=1.6, zorder=2))
        ax.text((x0 + x1) / 2, y + 1.18, advance[i], ha="center", va="center",
                fontsize=7.6, color=C_GREEN)
        # backward (rollback) arrow, arched below
        ax.add_patch(FancyArrowPatch((x1, y - 0.28), (x0, y - 0.28),
                     connectionstyle="arc3,rad=-0.32", arrowstyle="-|>",
                     mutation_scale=12, color=C_RED, lw=1.3, ls="--", zorder=2))

    ax.text(7.0, 0.55, "backward arrows: rollback after 50 collapse episodes "
            "(decay-weighted); $\\geq 100$ episodes per stage before any transition",
            ha="center", fontsize=8.2, color=C_RED)
    ax.text(7.0, 4.7, "Performance-driven NPC curriculum",
            ha="center", fontsize=11.5, fontweight="bold")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    _save(fig, "curriculum_stages")


# =============================================================================
# Fig (new) -- GAE lambda weighting (complements the GAE equation)
# =============================================================================
def fig_gae_lambda():
    gamma = 0.99
    L = np.arange(0, 21)
    series = [
        (0.0, C_RED, r"$\lambda=0$  (one-step TD: low variance, biased)"),
        (0.95, C_GREEN, r"$\lambda=0.95$  (used here: bias-variance balance)"),
        (1.0, C_OBS, r"$\lambda=1$  (Monte Carlo: unbiased, high variance)"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for lam, c, lab in series:
        w = (gamma * lam) ** L
        if lam == 0.0:
            w = np.zeros_like(L, dtype=float)
            w[0] = 1.0
        ax.plot(L, w, marker="o", ms=4, color=c, lw=1.9, label=lab)
    ax.set_xlabel(r"look-ahead offset $\ell$ (steps into the future)")
    ax.set_ylabel(r"weight on TD residual $\delta^V_{t+\ell}$:  $(\gamma\lambda)^{\ell}$")
    ax.set_title("GAE: exponential weighting of one-step TD residuals",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    _save(fig, "gae_lambda")


# =============================================================================
# Fig (new) -- log_std tanh-squash parameterisation (complements policy squash)
# =============================================================================
def fig_log_std_squash():
    z = np.linspace(-4, 4, 600)
    mid, span = -2.1, 0.9
    sigma_new = np.exp(mid + span * np.tanh(z))         # bounded [0.05, 0.30]
    sigma_old = np.exp(np.clip(z, -3.0, -0.7))          # old straight-through clamp

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.plot(z, sigma_old, color=C_RED, lw=2.0, ls="--",
            label="old straight-through clamp (flat beyond cap)")
    ax.plot(z, sigma_new, color=C_GREEN, lw=2.3,
            label="tanh-squashed parameterisation (smooth)")

    ax.axhline(0.4966, color=C_RED, lw=0.9, ls=":", alpha=0.8)
    ax.text(-3.9, 0.51, r"$\sigma=0.497$ frozen (old failure)", color=C_RED, fontsize=8)
    ax.axhline(0.30, color=C_GREEN, lw=0.9, ls=":", alpha=0.8)
    ax.text(-3.9, 0.315, r"$\sigma_{\max}=0.30$ (new ceiling)", color=C_GREEN, fontsize=8)
    ax.axhline(0.05, color=C_GREY, lw=0.9, ls=":", alpha=0.8)
    ax.text(-3.9, 0.062, r"$\sigma_{\min}=0.05$", color="#555", fontsize=8)

    z_init = float(np.arctanh((-1.5 - mid) / span))
    ax.plot([z_init], [np.exp(-1.5)], marker="D", ms=8, color=C_EDGE, zorder=5)
    ax.annotate(r"init $\sigma\approx0.22$", (z_init, np.exp(-1.5)),
                textcoords="offset points", xytext=(10, 6), fontsize=8.5)

    ax.set_xlabel(r"raw trainable parameter $z$  (actor log-std)")
    ax.set_ylabel(r"policy standard deviation $\sigma$")
    ax.set_title("Bounded log-std stops exploration noise from freezing",
                 fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, 0.56)
    fig.tight_layout()
    _save(fig, "log_std_squash")


# =============================================================================
# Fig (new) -- alpha-blend minimal-interference projection
# =============================================================================
def fig_alpha_blend():
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    a_prop = np.array([0.15, 0.65])     # steering, throttle/brake
    a_em = np.array([-0.45, -0.55])
    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]

    unsafe = Polygon([[-1.05, 0.15], [1.05, 0.15], [1.05, 1.05], [-1.05, 1.05]],
                     closed=True, facecolor="#fdecea", edgecolor=C_RED,
                     lw=1.0, alpha=0.7, zorder=1)
    ax.add_patch(unsafe)
    ax.text(0.55, 0.86, "unsafe:\npredicted to\nleave lane", color=C_RED,
            fontsize=8.5, ha="center", va="center")

    pts = np.array([(1 - a) * a_prop + a * a_em for a in alphas])
    ax.plot(pts[:, 0], pts[:, 1], color=C_EDGE, lw=1.4, zorder=2)
    safe_idx = next(i for i, p in enumerate(pts) if p[1] < 0.15)
    for i, (a, p) in enumerate(zip(alphas, pts)):
        if i == 0:
            ax.plot(*p, marker="o", ms=11, color=C_OBS, zorder=4)
            ax.annotate(r"$a^{\mathrm{prop}}$ ($\alpha{=}0$)", p,
                        textcoords="offset points", xytext=(10, 6), fontsize=9, color=C_OBS)
        elif i == len(alphas) - 1:
            ax.plot(*p, marker="s", ms=11, color=C_RED, zorder=4)
            ax.annotate(r"$a^{\mathrm{em}}$ ($\alpha{=}1$)", p,
                        textcoords="offset points", xytext=(10, -6), fontsize=9, color=C_RED)
        else:
            col = C_GREEN if i == safe_idx else C_GREY
            ax.plot(*p, marker="o", ms=8, color=col, zorder=4)
            ax.annotate(rf"$\alpha{{=}}{a}$", p, textcoords="offset points",
                        xytext=(8, 7), fontsize=8, color=col)
    ax.annotate("first safe blend\n(executed)", pts[safe_idx],
                textcoords="offset points", xytext=(-92, -8), fontsize=8.5,
                color=C_GREEN,
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2))

    ax.axhline(0, color="#bbb", lw=0.6)
    ax.axvline(0, color="#bbb", lw=0.6)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"steering  $a^{(1)}$")
    ax.set_ylabel(r"throttle / brake  $a^{(2)}$")
    ax.set_title("Minimal-interference projection:\nsearch $\\alpha$ until the blend passes the check",
                 fontsize=10.5, fontweight="bold")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "alpha_blend")


# =============================================================================
# Fig (new) -- Bradley-Terry preference model (complements the BT loss)
# =============================================================================
def fig_bradley_terry():
    d = np.linspace(-6, 6, 400)
    p = 1.0 / (1.0 + np.exp(-d))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(d, p, color=C_OBS, lw=2.4)
    ax.axhline(0.5, color="#bbb", lw=0.8, ls=":")
    ax.axvline(0.0, color="#bbb", lw=0.8, ls=":")
    ax.fill_between(d, 0.5, p, where=(d >= 0), color="#eafaf0", alpha=0.8)
    ax.fill_between(d, p, 0.5, where=(d < 0), color="#fdecea", alpha=0.8)
    ax.annotate(r"$s_a$ judged safer", (3.2, 0.9), fontsize=9, color=C_GREEN, ha="center")
    ax.annotate(r"$s_b$ judged safer", (-3.2, 0.1), fontsize=9, color=C_RED, ha="center")
    ax.set_xlabel(r"reward gap  $r_\phi(s_a) - r_\phi(s_b)$")
    ax.set_ylabel(r"$P(s_a \succ s_b) = \sigma\!\left(r_\phi(s_a)-r_\phi(s_b)\right)$")
    ax.set_title("Bradley-Terry preference distilled from the LLM judgements",
                 fontsize=10.5, fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "bradley_terry")


# =============================================================================
# Fig (new) -- lane-keep gate heatmap (replaces the gate equation)
# =============================================================================
def fig_lane_keep_gate():
    ell = np.linspace(0, 1, 200)        # |normalised lateral offset|
    psi = np.linspace(0, 30, 200)       # |heading error| in degrees
    E, P = np.meshgrid(ell, psi)
    mu = np.clip(1 - E, 0, None) * np.clip(1 - P / 20.0, 0, None)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    im = ax.pcolormesh(E, P, mu, cmap="YlGn", shading="auto", vmin=0, vmax=1)
    cs = ax.contour(E, P, mu, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                    colors="#333", linewidths=0.7)
    ax.clabel(cs, fmt="%.1f", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"lane-keep gate  $\mu_t \in [0,1]$")
    ax.axhline(20, color=C_RED, lw=1.3, ls="--")
    ax.text(0.5, 21.0, r"$|\psi_t|\geq 20^\circ \Rightarrow \mu_t=0$", color=C_RED,
            fontsize=8.5, ha="center")
    ax.set_xlabel(r"$|\ell_t|$  (normalised lateral offset)")
    ax.set_ylabel(r"$|\psi_t|$  (heading error, degrees)")
    ax.set_title("Lane-keep gate: forward bonuses survive only when\ncentred AND aligned",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    _save(fig, "lane_keep_gate")


# =============================================================================
# Fig (new) -- permissiveness ramp (replaces the interpolation equation)
# =============================================================================
def fig_permissiveness_ramp():
    p = np.linspace(0, 1, 100)
    thr = [
        (r"front clearance $\tau_f$", 0.15, 0.05, C_RED),
        (r"lateral tolerance $\tau_\ell$", 0.55, 0.80, C_OBS),
        (r"side clearance $\tau_s$", 0.04, 0.01, C_AMBER),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    for name, strict, perm, c in thr:
        tau = strict + (1 - p) * (perm - strict)
        ax.plot(p, tau, color=c, lw=2.3, label=name)
        ax.plot([1], [strict], marker="o", ms=6, color=c)
        ax.plot([0], [perm], marker="s", ms=6, color=c)
    ax.invert_xaxis()   # p=1 (strict) on the left, p=0 (weaned) on the right
    ax.set_ylim(0, 0.9)
    ax.axvline(1.0, color="#bbb", lw=0.8, ls=":")
    ax.axvline(0.0, color="#bbb", lw=0.8, ls=":")
    # endpoint captions placed ABOVE the axes (axis-fraction y) so they never
    # collide with the plotted lines, the y-axis or the right border
    xtrans = ax.get_xaxis_transform()
    ax.text(1.0, 1.02, "$p=1$: production\nshield (strict)", transform=xtrans,
            ha="left", va="bottom", fontsize=8.5, color=C_EDGE)
    ax.text(0.0, 1.02, "$p=0$: maximally\nweaned (permissive)", transform=xtrans,
            ha="right", va="bottom", fontsize=8.5, color=C_EDGE)
    ax.set_xlabel(r"permissiveness  $p$")
    ax.set_ylabel(r"effective threshold  $\tau_i(p)$")
    # legend BELOW the axis, horizontal, so it never overlaps the lines
    ax.legend(frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "permissiveness_ramp")


if __name__ == "__main__":
    print("Generating diagram figures...")
    fig_ppo_surrogate()
    fig_bicycle_model()
    fig_wrapper_chain()
    fig_lidar_channels()
    fig_actor_critic_arch()
    fig_reward_overview()
    fig_risk_levels()
    fig_curriculum_stages()
    fig_gae_lambda()
    fig_log_std_squash()
    fig_alpha_blend()
    fig_bradley_terry()
    fig_lane_keep_gate()
    fig_permissiveness_ramp()
    print("Done.")
