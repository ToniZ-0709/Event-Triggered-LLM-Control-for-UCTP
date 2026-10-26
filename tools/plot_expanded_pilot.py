"""Plot saved endpoints as one 3x3 Matplotlib comparison per solver seed.

Contract: compare methods within each instance, hard first and soft on an
exact hard tie, including positive-hard ties as requested for this view.
Archetype: quantitative grid, all nine instances retained per seed.
Each mark is one archived endpoint, not a mean; no uncertainty is estimated.
This is a separate descriptive view: it does not change the campaign report's
conservative positive-hard tie policy or its matched-budget win counts.
Export: screen-readable 16.8 x 11.5 inch PDF/SVG and 300 dpi PNG, editable text.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("baseline", "single", "multi")
LABELS = {"baseline": "Baseline", "single": "Single", "multi": "Multi"}
COLORS = {"baseline": "#87929F", "single": "#277DA8", "multi": "#D48B32"}
INK = "#263342"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": "#C5CBD1",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": "#56616D",
    "ytick.color": INK,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def endpoint_key(record: dict) -> tuple[int, int]:
    """Requested lexicographic order; do not use a weighted scalar mixture."""
    return int(record["hard"]), int(record["internal_total"])


def load_rows(report: dict, seed: int) -> list[dict]:
    selected = [row for row in report["instances"] if row["seed"] == seed]
    by_name = {row["instance"]: row for row in selected}
    expected = report["expected_instances"]
    if len(selected) != 9 or len(by_name) != 9 or set(by_name) != set(expected):
        raise ValueError(f"Expected exactly nine distinct instances for seed {seed}")
    rows = [by_name[instance] for instance in expected]
    for row in rows:
        if set(row["methods"]) != set(METHODS):
            raise ValueError("Missing method in saved comparison")
        for record in row["methods"].values():
            hard, soft = endpoint_key(record)
            if min(hard, soft) < 0:
                raise ValueError("Negative hard/soft score")
    return rows


def apply_followup(report: dict, followup: dict, seeds: list[int]) -> tuple[dict, list[dict]]:
    """Replace complete matched triples in the display only; retain both source reports."""
    view = deepcopy(report)
    replacements = []
    for row in followup["instances"]:
        if row["seed"] not in seeds:
            continue
        records = row["methods"]
        if (set(records) != set(METHODS) or not row.get("evaluation_matched_triple") or
                len({record["evaluations"] for record in records.values()}) != 1 or
                any(not record.get("evaluation_cap_reached") or record.get("api_failures")
                    for record in records.values())):
            raise ValueError("Follow-up must contain a completed, API-eligible, evaluation-matched triple")
        matches = [index for index, old in enumerate(view["instances"])
                   if (old["seed"], old["instance"]) == (row["seed"], row["instance"])]
        if len(matches) != 1:
            raise ValueError("Follow-up must replace exactly one existing instance/seed triple")
        replacement = deepcopy(row)
        replacement["display_source"] = "no_timecap_followup"
        view["instances"][matches[0]] = replacement
        replacements.append({"seed": row["seed"], "instance": row["instance"],
                             "evaluations": next(iter(records.values()))["evaluations"],
                             "runs": row.get("runs", {})})
    if not replacements:
        raise ValueError("No complete follow-up triple exists for the requested seeds")
    return view, replacements


def seed_summary(rows: list[dict]) -> dict:
    matched = [row for row in rows if row["evaluation_matched_triple"]]
    pairwise = {}
    for left, right in (("baseline", "single"), ("baseline", "multi"), ("single", "multi")):
        counts = Counter()
        for row in matched:
            a, b = endpoint_key(row["methods"][left]), endpoint_key(row["methods"][right])
            counts["better" if b < a else "worse" if b > a else "equal"] += 1
        pairwise[f"{right}_vs_{left}"] = {key: counts[key] for key in ("better", "worse", "equal")}
    return {
        "instances": len(rows), "evaluation_matched_instances": len(matched),
        "quality_rule": "Descriptive lexicographic (hard, soft), including positive-hard ties.",
        "quality_counts": pairwise,
        "method_totals_completed_endpoints": {method: {
            "feasible_internal": sum(row["methods"][method]["hard"] == 0 for row in rows),
            **{key: sum(row["methods"][method].get(key, 0) for row in rows)
               for key in ("runtime_seconds", "api_calls", "total_native_tokens")}}
            for method in METHODS},
        "cost_scope": "Displayed completed endpoints only; failed attempts and unknown request usage remain in source campaign reports.",
    }


def draw_panel(ax, row: dict) -> dict:
    records = row["methods"]
    hard_values = [int(records[method]["hard"]) for method in METHODS]
    hard_tied = len(set(hard_values)) == 1
    metric = "soft" if hard_tied else "hard"
    values = [int(records[method]["internal_total"] if hard_tied else records[method]["hard"])
              for method in METHODS]
    matched = bool(row["evaluation_matched_triple"])
    best_key = min(endpoint_key(records[method]) for method in METHODS)
    best = [method for method in METHODS if endpoint_key(records[method]) == best_key]
    # Time-censored unequal-evaluation endpoints are displayed, not awarded a win.
    highlighted = set(best) if matched else set()

    title = row["instance"] + (" *" if not matched else "")
    ax.text(0, 1.10, title, transform=ax.transAxes,
            fontweight="bold", fontsize=12)

    maximum = max(max(values), 1)
    # Keep numeric labels in a reserved area beyond the bars, never over data.
    ax.set_xlim(0, maximum * (1.55 if hard_tied else 2.10))
    ax.set_ylim(2.65, -0.65)
    bars = ax.barh(range(3), values, height=0.48,
                   color=[COLORS[method] for method in METHODS],
                   edgecolor=[INK if method in highlighted else "none" for method in METHODS],
                   linewidth=1.6, zorder=3)
    ax.set_yticks(range(3), [LABELS[method] for method in METHODS])
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=3, pad=4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    # No full-height grid: it would cross endpoint labels beyond the bars.
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Soft (H = {hard_values[0]}) ↓" if hard_tied
                  else "Hard ↓", labelpad=6)

    for y, method, value, bar, tick in zip(range(3), METHODS, values, bars,
                                         ax.get_yticklabels()):
        record = records[method]
        if hard_tied:
            label = f"{value:,}"
        else:
            label = f"H {int(record['hard'])}  |  S {int(record['internal_total']):,}"
        if method in highlighted:
            tick.set_fontweight("bold")
        if value == 0:
            ax.plot(0, y, marker="o", markersize=4.5, color=COLORS[method],
                    clip_on=False, zorder=4)
        ax.annotate(label, (bar.get_width(), y), xytext=(7, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=10, fontweight="bold" if method in highlighted else "normal")

    return {
        "panel": row["instance"], "instance": row["instance"], "seed": row["seed"],
        "metric_shown": metric, "hard_tied": hard_tied,
        "evaluation_matched": matched,
        "highlighted_methods": sorted(highlighted),
        "descriptive_lexicographic_minima": best,
        "source": row.get("display_source", "original_campaign"),
        "observations": [{"method": method, "hard": int(records[method]["hard"]),
                          "soft": int(records[method]["internal_total"]),
                          "evaluations": int(records[method]["evaluations"])}
                         for method in METHODS],
        "uncertainty": "None: one endpoint per method for this instance and seed.",
    }


def build_figure(report: dict, seed: int, output: Path, qa: Path,
                 require_matplotlib_panel_alignment) -> dict:
    rows = load_rows(report, seed)
    fig, axes = plt.subplots(3, 3, figsize=(16.8, 11.5))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.855,
                        wspace=0.43, hspace=0.57)
    fig.suptitle(f"ITC 2019 | Seed {seed}",
                 x=0.075, y=0.965, ha="left", fontsize=21, fontweight="bold")
    fig.text(0.075, 0.920,
             "Hard first, soft on ties  |  ↓ Better  |  Outline = best",
             ha="left", fontsize=11, color="#56616D")
    panels = [draw_panel(ax, row) for ax, row in zip(axes.flat, rows)]
    footer = "H = hard; S = soft. Internal scores."
    if any(not row["evaluation_matched_triple"] for row in rows):
        footer += "  * 20-min cap, unequal evaluations; no winner marked."
    if any(row.get("display_source") == "no_timecap_followup" for row in rows):
        footer += "  tg: no time cap, 1,200 evaluations."
    fig.text(0.075, 0.025, footer,
             fontsize=10.5, color="#56616D")
    fig.canvas.draw()
    name = f"seed_{seed}_comparison"
    require_matplotlib_panel_alignment(
        fig, json_out=qa / f"{name}.alignment.json",
        overlay_svg=qa / f"{name}.alignment.svg",
        require_panel_labels=False, tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(output / f"{name}.pdf", facecolor="white")
    fig.savefig(output / f"{name}.svg", facecolor="white")
    fig.savefig(output / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)
    return {"seed": seed, "panels": panels, "summary": seed_summary(rows)}


def retained_followup_sources(previous: dict, current: dict | None) -> list[dict]:
    """Keep provenance for follow-up endpoints when only one seed is redrawn."""
    sources = deepcopy(previous.get("followup_sources", []))
    legacy = previous.get("followup_source")
    if legacy and not any(item.get("sha256") == legacy.get("sha256") for item in sources):
        sources.append(deepcopy(legacy))
    if current:
        existing = next((item for item in sources
                         if item.get("sha256") == current.get("sha256")), None)
        if existing is None:
            sources.append(deepcopy(current))
        else:
            replacements = {(item["seed"], item["instance"]): item
                            for item in existing.get("replacements", [])}
            replacements.update({(item["seed"], item["instance"]): item
                                 for item in current.get("replacements", [])})
            existing["replacements"] = list(replacements.values())
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=ROOT / "solver_checks/expanded_pilot_comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/expanded_pilot")
    parser.add_argument("--seeds", type=int, nargs="+", help="Only redraw these seeds; preserve the others")
    parser.add_argument("--followup-report", type=Path,
                        help="Display completed matched follow-up triples instead of their old capped endpoints")
    parser.add_argument("--qa-output", type=Path, default=ROOT / "tmp/expanded_pilot_figures")
    parser.add_argument("--figure-audit-scripts", type=Path,
                        default=Path.home() / ".codex/skills/nature-figure/scripts")
    args = parser.parse_args()
    sys.path.insert(0, str(args.figure_audit_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    report_bytes = args.report.read_bytes()
    report = json.loads(report_bytes)
    seeds = args.seeds if args.seeds is not None else report["seeds"]
    if len(seeds) != len(set(seeds)) or not set(seeds) <= set(report["seeds"]):
        parser.error("Choose distinct seeds present in the original report")
    previous_path = args.output / "plot_manifest.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    if set(seeds) != set(report["seeds"]):
        if previous.get("source_sha256") != hashlib.sha256(report_bytes).hexdigest():
            parser.error("Partial redraw requires an existing manifest from the same original report")
        if not set(report["seeds"]) - set(seeds) <= {figure["seed"] for figure in previous["figures"]}:
            parser.error("Partial redraw cannot discard another seed's source data")
    replacements = []
    followup_source = None
    if args.followup_report is not None:
        followup_bytes = args.followup_report.read_bytes()
        report, replacements = apply_followup(report, json.loads(followup_bytes), seeds)
        followup_source = {"sha256": hashlib.sha256(followup_bytes).hexdigest(),
                           "replacements": replacements}
    args.output.mkdir(parents=True, exist_ok=True)
    args.qa_output.mkdir(parents=True, exist_ok=True)
    figures_by_seed = {figure["seed"]: figure for figure in previous.get("figures", [])}
    for seed in seeds:
        figures_by_seed[seed] = build_figure(report, int(seed), args.output, args.qa_output,
                                             require_matplotlib_panel_alignment)
    figures = [figures_by_seed[seed] for seed in report["seeds"]]
    with (args.output / "comparison_source.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("seed", "instance", "method", "hard", "soft", "evaluations",
                  "evaluation_matched", "metric_shown")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for figure in figures:
            for panel in figure["panels"]:
                for observation in panel["observations"]:
                    writer.writerow({"seed": figure["seed"], "instance": panel["instance"],
                                     **observation,
                                     "evaluation_matched": panel["evaluation_matched"],
                                     "metric_shown": panel["metric_shown"]})
    manifest = {"source_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "rule": "Minimize (hard, soft), including tied positive hard as requested.",
                "qa_notes": {
                    "width": "Large screen-readable grid requested, not journal-column output.",
                    "uncertainty": "54 displayed endpoints, one per instance/seed/method; no aggregation across seeds. Explicit complete follow-up triples replace old censored endpoints in this view only.",
                    "layout": "Repeated panels are the nine requested instance strata, with common method colors.",
                    "presentation": "Compact labels; panel scales differ; H > 0 remains infeasible regardless of soft tie-break.",
                },
                "original_comparison_report_modified": False, "figures": figures,
                "redrawn_seeds": seeds, "followup_source": followup_source,
                "followup_sources": retained_followup_sources(previous, followup_source)}
    (args.output / "plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"redrawn_seeds": seeds, "figures": len(figures), "panels": sum(len(f["panels"]) for f in figures),
                      "observations": 54, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
