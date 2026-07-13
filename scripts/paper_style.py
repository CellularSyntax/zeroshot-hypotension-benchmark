"""Shared Nature-Medicine-style plotting system for the TiRex-2 / VitalDB paper figures.

Everything visual is centralised here so all figures share one look:
  * typography (Arial/Helvetica, small sizes), hairline axes, no top/right spines
  * a fixed, colourblind-safe palette with FIXED semantic meaning across every figure
    (M1 = teal, M0 = amber, persistence = grey, Kapral = purple, Zhu = red)
  * helpers: panel_letter(), finish_axes(), save_fig(), horizon parsing, foil constants

Import into scripts/paper_figures.py; do not put data logic here.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ── typography ────────────────────────────────────────────────────────────────
# Prefer Arial/Helvetica (Nature house style); fall back to the best sans available.
_PREFERRED = ["Arial", "Helvetica", "Helvetica Neue", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"]
_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
SANS = next((f for f in _PREFERRED if f in _AVAIL), "DejaVu Sans")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [SANS],
    "pdf.fonttype": 42, "ps.fonttype": 42,          # editable text in Illustrator
    "svg.fonttype": "none",
    "font.size": 7,
    "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "axes.grid": False,
})

# ── palette (fixed semantics everywhere) ──────────────────────────────────────
# Colours are drawn from the seaborn "colorblind" qualitative palette (Wong 2011,
# Nature Methods) — a fixed, CVD-safe mapping with FIXED semantic meaning across
# every figure. TiRex-2 (M1) is the focal series (deep blue), rendered with the
# heaviest visual weight; comparators take distinct, well-separated hues.
C = {
    "M1":        "#0173B2",  # deep blue — TiRex-2 / with-covariate (focal model)
    "M1_light":  "#8EC1E0",  # light blue — forecast interval fills
    "M0":        "#DE8F05",  # orange — target-only (covariate ablation control, Fig 2 only)
    "M0_light":  "#F2C878",
    "persist":   "#949494",  # grey  — persistence baseline
    "transition":"#08519C",  # dark blue  — transition windows (sequential within-model)
    "steady":    "#9ECAE1",  # light blue — steady windows (sequential within-model)
    "kapral":    "#7B3FA0",  # purple diamond — Kapral 2024 (TFT), external literature point
    "zhu":       "#C0392B",  # red square     — Zhu 2026 (Transformer), external literature point
    "rate":      "#56B4E9",  # sky blue — RATE covariate arm (Fig 2 only; unused legacy key)
    "pressor":   "#ECE133",  # yellow   — phenylephrine arm (Fig 2 only; unused legacy key)
    "ink":       "#222222",
    "grid":      "#DDDDDD",
    "event":     "#C0392B",  # alarm hue (MAP<65 threshold) — reserved, not a data series
}

# ── external foil numbers (from notes/RELATED_WORK.md; also in comparison tables)
# Hypotension AUROC. Kapral: (internal, external). Zhu: external only.
KAPRAL_AUROC = {5: (0.909, 0.903), 7: (0.88, 0.867)}
KAPRAL_AUROC_OVERALL = (0.933, 0.919)          # horizon-averaged internal/external
ZHU_AUROC = {5: 0.904, 10: 0.892, 15: 0.882}

QLEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Q_MED, Q_LO, Q_HI = 4, 0, 8                     # median, 10%, 90% indices

# Column widths (mm -> inches) for Nature Medicine.
MM = 1 / 25.4
W1 = 89 * MM      # single column
W15 = 120 * MM    # 1.5 column
W2 = 183 * MM     # double column

FIG_DIR = "outputs/figs/paper"
TAB_DIR = "results/tables"


def horizons_sorted(per_horizon: dict) -> list[int]:
    """['1min','3min',...] -> [1,3,...] sorted ascending."""
    return sorted(int(k.replace("min", "")) for k in per_horizon)


def hkey(h: int) -> str:
    return f"{h}min"


def panel_letter(ax, letter: str, dx: float = -0.16, dy: float = 1.06):
    """Bold lowercase panel tag in axes-fraction coords (Nature style)."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="right", family="sans-serif")


# House style: panels carry only their bold letter (a, b, c…); the caption
# describes each panel. `title=` args are retained at call sites as inline
# documentation of panel content, but are not rendered while this flag is False.
SHOW_PANEL_TITLES = False


def finish(ax, xlabel=None, ylabel=None, title=None, ygrid=False):
    # Plain white background — no gridlines anywhere (paper house style).
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title and SHOW_PANEL_TITLES:
        ax.set_title(title, loc="center")
    return ax


def save_fig(fig, name: str):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{FIG_DIR}/{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {FIG_DIR}/{name}.pdf + .png + .svg", flush=True)
