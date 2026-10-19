"""Render the baseline/improved NSGA-III pipeline schematic."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT = Path(__file__).resolve().parent
NAVY = "#25425D"
BLUE_FILL = "#F1F5F8"
ORANGE = "#AD5B22"
ORANGE_FILL = "#FFF3E9"
INK = "#183044"
MUTED = "#586A79"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def box(ax, x, y, width, height, label, *, changed=False):
    edge = ORANGE if changed else NAVY
    fill = ORANGE_FILL if changed else BLUE_FILL
    ax.add_patch(Rectangle((x, y), width, height, linewidth=1.35,
                           edgecolor=edge, facecolor=fill, zorder=1))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
            color=INK, fontsize=10.3, weight="normal", linespacing=1.22, zorder=2)


def flow_arrows(ax, xs, width, y, color):
    for left, right in zip(xs[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch(
            (left + width + 0.003, y), (right - 0.003, y),
            arrowstyle="-|>", mutation_scale=19, linewidth=1.55,
            color=color, shrinkA=0, shrinkB=0, zorder=3,
        ))


def main():
    fig = plt.figure(figsize=(13.4, 5.2), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.09, 0.94, "NSGA-III solver pipeline: baseline to improved copy",
            fontsize=17.2, weight="bold", color=INK, va="center")
    ax.text(0.09, 0.865, "ORIGINAL BASELINE", fontsize=10.5,
            weight="bold", color=NAVY, va="center")
    ax.text(0.09, 0.535, "IMPROVED COPY", fontsize=10.5,
            weight="bold", color=ORANGE, va="center")

    xs = [0.09, 0.27, 0.45, 0.63, 0.81]
    width = 0.145
    height = 0.215
    baseline = [
        "ITC 2019 XML\nTime + room\nchromosome",
        "Greedy room\nplacement +\n1-class repair",
        "Crossover +\nmutation +\nfull re-pack",
        "4 soft objectives\n+ hard constraint\nNSGA-III survival",
        "Best observed\ncandidate\nXML if hard = 0",
    ]
    improved = [
        baseline[0],
        "Distribution-aware\nplacement +\n1-class repair",
        "Crossover +\nmutation + local\n1/2-class repair",
        baseline[3],
        baseline[4],
    ]
    for i, x in enumerate(xs):
        box(ax, x, 0.615, width, height, baseline[i])
        box(ax, x, 0.285, width, height, improved[i], changed=i in (1, 2))

    flow_arrows(ax, xs, width, 0.615 + height / 2, NAVY)
    flow_arrows(ax, xs, width, 0.285 + height / 2, ORANGE)

    ax.text(0.09, 0.135, "SEPARATE CHECK", fontsize=10.5,
            weight="bold", color=ORANGE, va="center")
    ax.add_patch(FancyArrowPatch((0.242, 0.135), (0.277, 0.135),
                                 arrowstyle="-|>", mutation_scale=18,
                                 linewidth=1.5, color=ORANGE, zorder=3))
    ax.add_patch(Rectangle((0.286, 0.075), 0.67, 0.12, linewidth=1.3,
                           linestyle=(0, (4, 3)), edgecolor=ORANGE,
                           facecolor="#FFF9F3", zorder=1))
    ax.text(0.621, 0.135,
            "Controller operator: accept a strict hard decrease even if still infeasible\nNot used in the matched solver runs",
            ha="center", va="center", fontsize=10.1, color=MUTED,
            linespacing=1.3, zorder=2)

    fig.savefig(OUT / "solver_pipeline.png", dpi=600,
                facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / "solver_pipeline.pdf", facecolor="white",
                bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / "solver_pipeline.svg", facecolor="white",
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
