"""
scripts/gen_gantt.py -- Generate Figure 2.1: project Gantt chart as a vector PDF.

Timeline: September 2025 to June 2026 (10 months).
Milestones: M1 Sep, M2 Nov, M3 Jan, M4 Mar, M5 May, M6 Jun.

Run:
    python scripts/gen_gantt.py
Output: Memoria/imgs/gantt_chart.pdf
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
})

IMGS = Path(__file__).resolve().parent.parent / "Memoria" / "imgs"
IMGS.mkdir(parents=True, exist_ok=True)

# ── timeline ──────────────────────────────────────────────────────────────────
MONTHS = ["Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
# month index 0-based → x position = index + 0.5 (bar centre)

# ── work packages ─────────────────────────────────────────────────────────────
# (label, start_month_idx, end_month_idx)  -- both inclusive, 0-based
WPS = [
    ("WP0  Project management",   0, 9),
    ("WP1  Research & SOTA",      0, 2),
    ("WP2  Environment & wrappers", 1, 3),
    ("WP3  PPO agent",            2, 4),
    ("WP4  Safety shields",       3, 5),
    ("WP5  Reward shaper",        3, 4),
    ("WP6  Curriculum manager",   4, 5),
    ("WP7  Metrics & dashboard",  4, 5),
    ("WP8  Experimentation",      5, 7),
    ("WP9  Dependence study",     7, 8),
    ("WP10 Thesis writing",       8, 9),
]

# ── milestones: (label, month_idx) ────────────────────────────────────────────
MILESTONES = [
    ("M1", 0),   # Sep 2025
    ("M2", 2),   # Nov 2025
    ("M3", 4),   # Jan 2026
    ("M4", 6),   # Mar 2026
    ("M5", 8),   # May 2026
    ("M6", 9),   # Jun 2026
]

N_WP = len(WPS)
N_ROWS = N_WP + 1  # +1 for the milestone row

BAR_H = 0.55
BAR_FC = "#444444"
BAR_EC = "#222222"
MS_FC  = "#111111"   # milestone diamond fill
ROW_H  = 1.0         # vertical spacing per row (y of row i = N_ROWS - 1 - i)

FIG_W = 13.0
FIG_H = 0.55 * N_ROWS + 1.4   # compact height

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(-0.05, 10.05)
ax.set_ylim(-1.5, N_ROWS + 0.1)
ax.set_aspect("auto")
ax.axis("off")

# ── year header ───────────────────────────────────────────────────────────────
HEADER_Y = N_ROWS - 0.05
ax.text(2.0, HEADER_Y + 0.62, "2025",
        ha="center", va="center", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="#d0d8e4", ec="#7a8fa6", lw=0.8))
ax.text(7.0, HEADER_Y + 0.62, "2026",
        ha="center", va="center", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="#d0d8e4", ec="#7a8fa6", lw=0.8))

# year separator vertical line at x=4
ax.axvline(4, color="#7a8fa6", lw=1.0, ls="--",
           ymin=0.0, ymax=1.0, alpha=0.45)

# ── month column header ───────────────────────────────────────────────────────
for i, m in enumerate(MONTHS):
    ax.text(i + 0.5, HEADER_Y + 0.12, m,
            ha="center", va="center", fontsize=9, color="#333333")

# ── alternating column shading ────────────────────────────────────────────────
for i in range(10):
    if i % 2 == 0:
        ax.axvspan(i, i + 1, ymin=0.0, ymax=1.0, color="#f0f0f0", alpha=0.55, zorder=0)

# grid lines
for i in range(11):
    ax.axvline(i, color="#cccccc", lw=0.5, zorder=1)

# ── work package bars ─────────────────────────────────────────────────────────
for row, (label, s, e) in enumerate(WPS):
    y = N_ROWS - 1 - row
    x0 = s
    width = e - s + 1
    ax.barh(y, width, left=x0, height=BAR_H,
            color=BAR_FC, edgecolor=BAR_EC, linewidth=0.6, zorder=3)
    ax.text(-0.12, y, label,
            ha="right", va="center", fontsize=8.5, color="#222222")

# ── milestone row (bottom) ────────────────────────────────────────────────────
ms_y = -0.55

# label on the left
ax.text(-0.12, ms_y, "Milestones",
        ha="right", va="center", fontsize=8.5, color="#222222", style="italic")

for label, idx in MILESTONES:
    x = idx + 0.5   # centre of the month column
    # diamond marker
    ax.plot(x, ms_y, marker="D", markersize=9,
            color=MS_FC, markeredgecolor="#666666", markeredgewidth=0.5, zorder=5)
    ax.text(x, ms_y - 0.45, label,
            ha="center", va="center", fontsize=8, color="#222222", style="italic")

# horizontal dashed guide line for milestone row
ax.axhline(ms_y, color="#cccccc", lw=0.5, ls=":", zorder=1)

# ── outer frame ───────────────────────────────────────────────────────────────
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(pad=0.3)

out = IMGS / "gantt_chart.pdf"
fig.savefig(str(out), bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {out}")
