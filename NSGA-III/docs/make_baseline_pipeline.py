"""Render the conceptual ITC 2019 NSGA-III baseline pipeline for the README."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
STEM = OUT / "nsga3-baseline-pipeline"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

INK = "#21374B"
BLUE = "#245F92"
PALE_BLUE = "#EAF3FA"
PALE_GREY = "#F2F5F7"
ORANGE = "#D89243"
PALE_ORANGE = "#FFF5E9"


def box(ax, x, y, w, h, label, *, fill=PALE_BLUE, edge=BLUE, size=10):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.12",
            linewidth=1.35,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=INK, fontsize=size, linespacing=1.22)


def arrow(ax, start, end, *, color=BLUE, rad=0, width=1.55):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=12,
            linewidth=width, color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


fig, ax = plt.subplots(figsize=(13, 7.1), dpi=300)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.1)
ax.axis("off")

ax.text(0.55, 6.75, "ITC 2019 NSGA-III baseline", color=INK,
        fontsize=17, fontweight="bold", va="center")
ax.text(0.55, 6.38,
        "Each XML instance is solved independently  |  no LLM controller",
        color="#597083", fontsize=10, va="center")

# Input and encoding.
ax.text(0.55, 5.92, "INPUT & ENCODING", color=BLUE, fontsize=9,
        fontweight="bold", va="center")
box(ax, 0.55, 4.94, 3.32, 0.72, "ITC 2019 instance (XML)",
    fill=PALE_GREY, edge="#8297A7")
box(ax, 4.85, 4.94, 3.32, 0.72, "Parse classes, domains & constraints")
box(ax, 9.16, 4.94, 3.32, 0.72,
    "Encode choices + initialize\npopulation")
arrow(ax, (3.89, 5.30), (4.83, 5.30))
arrow(ax, (8.19, 5.30), (9.14, 5.30))

# Search loop. The enclosing band avoids suggesting that a single operator
# can act independently of the rest of the solver.
ax.add_patch(
    FancyBboxPatch(
        (0.38, 2.28), 12.24, 2.22,
        boxstyle="round,pad=0.015,rounding_size=0.15",
        linewidth=1.4, edgecolor=BLUE, facecolor="#F8FBFD",
    )
)
ax.text(0.62, 4.20, "NSGA-III SEARCH", color=BLUE, fontsize=9,
        fontweight="bold", va="center")
ax.text(12.35, 4.20, "repeat for configured generations",
        color="#597083", fontsize=8.5, va="center", ha="right")
box(ax, 0.67, 3.03, 2.55, 0.83,
    "Create offspring\n(crossover + mutation)", size=9.7)
box(ax, 3.69, 3.03, 2.55, 0.83, "Bounded hard-\nconstraint repair", size=9.7)
box(ax, 6.70, 3.03, 2.55, 0.83,
    "Evaluate: 4 objectives\n+ hard violations", size=9.7)
box(ax, 9.71, 3.03, 2.55, 0.83,
    "NSGA-III survival\n(reference directions)", size=9.7)
for start, end in [((3.24, 3.44), (3.67, 3.44)),
                   ((6.26, 3.44), (6.68, 3.44)),
                   ((9.27, 3.44), (9.69, 3.44))]:
    arrow(ax, start, end)
arrow(ax, (11.0, 3.01), (11.0, 2.66), width=1.25)
arrow(ax, (10.96, 2.59), (1.95, 2.59), width=1.25)
arrow(ax, (1.95, 2.63), (1.95, 3.01), width=1.25)

# The vertical arrows link phases, not individual operators.
arrow(ax, (6.50, 4.91), (6.50, 4.53), color=ORANGE)
arrow(ax, (6.50, 2.26), (6.50, 1.94), color=ORANGE)

ax.text(0.55, 1.83, "FINAL DECISION & ARTIFACTS", color=BLUE,
        fontsize=9, fontweight="bold", va="center")
box(ax, 1.02, 0.72, 4.70, 0.83,
    "Choose final candidate:\nfewest hard violations, then lowest total penalty",
    fill=PALE_ORANGE, edge=ORANGE, size=9.5)
box(ax, 7.27, 0.72, 4.70, 0.83,
    "Always: result.json + choices.npz\nIf feasible: solution.xml",
    fill=PALE_ORANGE, edge=ORANGE, size=9.5)
arrow(ax, (5.75, 1.13), (7.24, 1.13), color=ORANGE)

fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.015)
fig.savefig(STEM.with_suffix(".svg"), facecolor="white")
fig.savefig(STEM.with_suffix(".pdf"), facecolor="white")
fig.savefig(STEM.with_suffix(".png"), dpi=300, facecolor="white")
plt.close(fig)
