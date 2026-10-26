"""Render per-instance, per-seed endpoint, convergence and resource figures.

Contract: describe one archived run per method, without estimating uncertainty
or attributing nearby progress to an intervention. Quantitative grids separate
endpoint quality, progress per time/evaluation, and completed-run resources.
PDF/SVG text stays editable; PNG previews use 300 dpi. Source CSV/JSON accompanies
every figure set. Existing nine-panel figures and campaign reports are inputs only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    from .search_result_data import METHODS, load_run, shared_ranges, campaign_costs
except ImportError:
    from search_result_data import METHODS, load_run, shared_ranges, campaign_costs

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"baseline": "#87929F", "single": "#277DA8", "multi": "#D48B32"}
LABELS = {"baseline": "Baseline", "single": "Single", "multi": "Inspector–Planner"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False})


def save_figure(fig, path, title, footer, require_matplotlib_panel_alignment):
    fig.subplots_adjust(left=.10, right=.97, top=.78, bottom=.34, wspace=.48)
    fig.suptitle(title, x=.10, y=.96, ha="left", fontsize=14)
    fig.text(.10, .045, footer, ha="left", va="bottom", fontsize=8)
    for label, ax in zip("abcdef", fig.axes):
        ax.set_label(label)
        ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-24, 22),
                    textcoords="offset points", fontweight="bold", fontsize=11)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=path.with_suffix(".alignment.json"),
              tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def bars(ax, runs, field, title, xlabel, annotate):
    maximum = max((r["endpoint"].get(field) or 0 for r in runs.values()), default=1) or 1
    ax.set_xlim(0, maximum*1.7)
    ax.set_ylim(len(runs)-.4, -.6)
    for y, (method, run) in enumerate(runs.items()):
        value = run["endpoint"].get(field)
        if value is None:
            ax.text(0, y, "Unknown", va="center")
            continue
        ax.barh(y, value, height=.5, color=COLORS[method])
        if not value:
            ax.plot(0, y, "o", color=COLORS[method], clip_on=False)
        ax.annotate(annotate(method, run), (value,y), xytext=(6,0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(range(len(runs)), [LABELS[m] for m in runs])
    ax.set_title(title, loc="left", pad=18, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))


def final_quality(runs, folder, title, note, alignment):
    fig, axes = plt.subplots(1,2,figsize=(13,5.8))
    bars(axes[0], runs, "hard", "Final hard violations", "Hard violations ↓",
         lambda m,r: f"H {r['endpoint']['hard']}\nEval {r['endpoint'].get('evaluations', '?')}")
    bars(axes[1], runs, "internal_total", "Weighted soft total (descriptive)", "Weighted soft total ↓",
         lambda m,r: f"{r['endpoint']['internal_total']:,}\nH {r['endpoint']['hard']}")
    save_figure(fig, folder/"final_quality", title, note+"\nSoft comparisons at positive hard ties are descriptive; all H > 0 remain infeasible.", alignment)


def convergence(runs, folder, title, note, alignment):
    feasible = any(any(p["hard"] == 0 for p in r["anytime"]) for r in runs.values())
    fig, axes = plt.subplots(1,3 if feasible else 2, figsize=(18 if feasible else 13,5.8))
    for method, run in runs.items():
        points = [p for p in run["anytime"] if p["hard"] is not None]
        for ax, x in zip(axes[:2], ("seconds", "evaluations")):
            if not points:
                continue
            ax.step([p[x] for p in points], [p["hard"] for p in points], where="post",
                    color=COLORS[method], label=LABELS[method])
            ax.plot(points[-1][x], points[-1]["hard"], "s", color=COLORS[method], ms=4)
            for review in run["reviews"]:
                if review[x] is not None and review["hard"] is not None:
                    ax.plot(review[x], review["hard"], "*" if review["admitted"] else "|",
                            ms=9, color=COLORS[method])
        if feasible:
            points = [p for p in points if p["hard"] == 0 and p["internal_total"] is not None]
            if points:
                axes[2].step([p["evaluations"] for p in points], [p["internal_total"] for p in points],
                             where="post", color=COLORS[method])
                axes[2].plot(points[-1]["evaluations"], points[-1]["internal_total"], "s", color=COLORS[method], ms=4)
    for ax, xlabel, subtitle in zip(axes, ("Elapsed wall-clock seconds", "Logical full evaluations", "Logical full evaluations"),
                                   ("Progress per wall time", "Progress per evaluation", "Quality after feasibility")):
        ax.set_title(subtitle, loc="left", pad=18, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0)
        ax.set_ylabel("Incumbent weighted soft" if ax is axes[-1] and feasible else "Incumbent hard violations")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[0].legend(loc="upper right", fontsize=8)
    save_figure(fig, folder/"convergence", title,
        note+"\nReview: vertical tick; admitted: star; final observation: square. Timing does not establish causal attribution.\nCurves retain initialization time and observed endpoints; comparisons use shared observed ranges (manifest).", alignment)


def resources(runs, folder, title, note, alignment, split_runtime=False):
    fig, axes = plt.subplots(1,2,figsize=(13,5.8))
    bars(axes[0], runs, "runtime_seconds", "Completed endpoint runtime", "Recorded runtime (seconds)",
         lambda m,r: f"{r['endpoint'].get('runtime_seconds', 0):,.0f} s\nCalls {r['endpoint'].get('api_calls','?')}")
    if split_runtime:
        axes[0].clear()
        for y, (method, run) in enumerate(runs.items()):
            total = run["endpoint"].get("runtime_seconds")
            api = run["endpoint"].get("api_latency_seconds")
            if total is None or api is None or not 0 <= api <= total:
                raise ValueError("Cannot split unknown, negative or overlapping API/runtime totals")
            axes[0].barh(y, api, color=COLORS[method], hatch="///", label="API handling (incl. pacing)" if y == 0 else None)
            axes[0].barh(y, total-api, left=api, color=COLORS[method], alpha=.5, label="All remaining work" if y == 0 else None)
        axes[0].set_yticks(range(len(runs)), [LABELS[m] for m in runs])
        axes[0].invert_yaxis()
        axes[0].set_title("Runtime decomposition", loc="left", pad=18, fontsize=11)
        axes[0].set_xlabel("Recorded runtime (seconds)")
        axes[0].legend(fontsize=7, loc="upper right")
    for method, run in runs.items():
        reviews = [r for r in run["reviews"] if r["seconds"] is not None]
        if method == "baseline" and run["anytime"]:
            axes[1].plot([run["anytime"][0]["seconds"], run["anytime"][-1]["seconds"]], [0,0], color=COLORS[method], label=LABELS[method])
        elif reviews:
            known = all(r["usage_complete"] for r in reviews)
            axes[1].step([r["seconds"] for r in reviews], [r["cumulative_reported_tokens"] for r in reviews],
                         where="post", color=COLORS[method], linestyle="-" if known else "--",
                         label=LABELS[method]+(" (lower bound)" if not known else ""))
            axes[1].plot(reviews[-1]["seconds"], reviews[-1]["cumulative_reported_tokens"], "s", color=COLORS[method], ms=4)
    axes[1].set_title("Reported native tokens at review checkpoints", loc="left", pad=18, fontsize=10)
    axes[1].set_xlabel("Elapsed wall-clock seconds")
    axes[1].set_ylabel("Cumulative reported native tokens")
    axes[1].legend(fontsize=8)
    save_figure(fig, folder/"resources", title, note+
        "\nRuntime includes API waiting; remaining work is not a pure-solver timer. No monetary cost is inferred.\nUsage uses provider totals once; missing usage stays unknown. Legacy timing resolves to post-review generations.", alignment)


def extras(runs, folder, title, note, alignment, requested):
    reasons = {}
    if "reviews" in requested:
        fig, axes = plt.subplots(1,2,figsize=(13,5.8))
        for method, run in runs.items():
            if not run["reviews"]:
                continue
            for ax, key in zip(axes,("reported_tokens","latency_seconds")):
                reviews = run["reviews"]
                ax.plot([r["generation"] for r in reviews], [r[key] for r in reviews], "o-", color=COLORS[method], label=LABELS[method])
        for ax, ylabel in zip(axes,("Reported tokens (known portion)","Request handling seconds, incl. pacing")):
            ax.set_xlabel("Review generation")
            ax.set_ylabel(ylabel)
        axes[0].legend(fontsize=8)
        save_figure(fig, folder/"per_review", title, note+"\nUnknown request usage contributes no known tokens; it is not zero usage.", alignment)
    if "diversity" in requested:
        fig, axes = plt.subplots(1,2,figsize=(13,5.8))
        for method, run in runs.items():
            for ax,key in zip(axes,("hard","placement_diversity")):
                points = [p for p in run["anytime"] if p.get(key) is not None]
                ax.step([p["evaluations"] for p in points], [p[key] for p in points], where="post", color=COLORS[method], label=LABELS[method])
        for ax,ylabel in zip(axes,("Incumbent hard violations","Placement diversity")):
            ax.set_xlabel("Logical full evaluations")
            ax.set_ylabel(ylabel)
        axes[0].legend(fontsize=8)
        save_figure(fig, folder/"diversity", title, note+"\nDiversity is a population metric; alignment alone does not establish a cause of stagnation.", alignment)
    if "operators" in requested:
        operators = sorted({r["operator"] for run in runs.values() for r in run["reviews"] if r["operator"]})
        fig, axes = plt.subplots(1,2,figsize=(13,5.8))
        operator_colors = {op: plt.get_cmap("tab10")(j) for j,op in enumerate(operators)}
        for ax, method in zip(axes,("single","multi")):
            if method not in runs:
                continue
            for review in runs[method]["reviews"]:
                if review["operator"] not in operators:
                    continue
                y = operators.index(review["operator"])
                marker = ("." if review["operator"] == "no_op" else "o" if review["selected"] else
                          "s" if review["admitted"] else "x")
                ax.plot(review["generation"], y, marker, color=operator_colors[review["operator"]])
            ax.set_yticks(range(len(operators)), operators)
            ax.set_title(LABELS[method], loc="left", pad=18)
            ax.set_xlabel("Review generation")
        save_figure(fig, folder/"operators", title, note+"\nCircle: selected; square: rejected by survival; cross: no admitted candidate; dot: no-op.\nSource and incumbent improvement are separate fields in reviews.csv.", alignment)
    if "objectives" in requested:
        available = any(p["weighted_objectives"] is not None for run in runs.values() for p in run["anytime"])
        if not available:
            reasons["objectives"] = "No incumbent objective vectors saved; population minima cannot substitute."
        else:
            fig, axes = plt.subplots(1,4,figsize=(20,5.8))
            for method, run in runs.items():
                points = [p for p in run["anytime"] if p["weighted_objectives"] is not None]
                for j,ax in enumerate(axes):
                    ax.step([p["evaluations"] for p in points], [p["weighted_objectives"][j] for p in points], where="post", color=COLORS[method], label=LABELS[method])
            for ax,name in zip(axes,("Time","Room","Distribution","Student")):
                ax.set_title(name, loc="left", pad=18)
                ax.set_xlabel("Logical full evaluations")
                ax.set_ylabel("Incumbent weighted component")
            axes[0].legend(fontsize=8)
            save_figure(fig, folder/"objectives", title, note+"\nAll components belong to the same archived timetable at each checkpoint.", alignment)
    return reasons


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def render_row(report, row, output, alignment, extra=(), split_runtime=False):
    runs = {m: load_run(row["runs"][m], row["instance"], m, endpoint)
            for m,endpoint in row["methods"].items() if m in row.get("runs", {})}
    if not runs:
        raise ValueError("No saved run paths for requested comparison")
    caps = {r["settings"].get("time_limit_seconds") for r in runs.values()}
    if len(caps) > 1:
        raise ValueError("Time-capped and no-time-cap endpoints require separate figures")
    folder = output / f"seed_{row['seed']}" / row["instance"]
    folder.mkdir(parents=True, exist_ok=True)
    ranges = shared_ranges(runs)
    warnings = [f"{m}: {w}" for m,r in runs.items() for w in r["warnings"]]
    if len(runs) != len(METHODS):
        warnings.append("Incomplete method triple")
    source_sets = [r["settings"].get("solver_sources_sha256") for r in runs.values()]
    if any(s != source_sets[0] for s in source_sets[1:]):
        warnings.append("Source snapshots differ: speed/quality attribution is unmatched")
    configs = [r["settings"].get("config") for r in runs.values()]
    if any(c != configs[0] for c in configs[1:]):
        warnings.append("Solver configurations differ: unmatched comparison")
    matched = bool(row.get("evaluation_matched_triple")) and not warnings
    note = f"Seed {row['seed']}; one run/method; internal scores; " + ("matched endpoint evaluations." if matched else "unmatched or incomplete endpoints.")
    statuses = [f"{LABELS[m]}: {'time-censored' if r['endpoint'].get('time_cap_reached') else r['endpoint'].get('completion_status', 'completed endpoint')}"
                for m,r in runs.items()]
    note += "\n" + "; ".join(statuses)
    costs = campaign_costs(report, row["seed"], row["instance"])
    failed = sum(a.get("completion_status") != "completed" for a in costs["published_attempts"])
    note += f"\nEndpoint costs shown; campaign has {failed} incomplete + {len(costs['unpublished_attempts'])} unpublished attempts; failed-request usage unknown."
    title = f"ITC 2019 | {row['instance']} | Seed {row['seed']}"
    final_quality(runs, folder, title, note, alignment)
    convergence(runs, folder, title, note, alignment)
    resources(runs, folder, title, note, alignment, split_runtime)
    skipped = extras(runs, folder, title, note, alignment, extra)
    for method,run in runs.items():
        write_csv(folder/f"{method}_anytime.csv", run["anytime"])
        write_csv(folder/f"{method}_reviews.csv", run["reviews"])
    endpoint_fields = ("hard","internal_total","evaluations","runtime_seconds","api_calls","total_native_tokens")
    write_csv(folder/"endpoints.csv", [{"method":m,**{k:r["endpoint"].get(k) for k in endpoint_fields}} for m,r in runs.items()])
    manifest = {"instance": row["instance"], "seed": row["seed"], "endpoint_evaluation_matched": matched,
        "shared_observed_ranges": ranges, "time_limit_seconds": next(iter(caps)),
        "cost_scope_figures": "displayed_completed_or_incomplete_endpoints_only",
        "campaign_costs": costs, "warnings": warnings, "skipped_optional_figures": skipped,
        "hardware_comparability": "Recorded hardware conditions are not available in legacy manifests; speed is descriptive.",
        "uncertainty": "One run per method/seed; no aggregates, confidence interval or significance claim.",
        "ranges": "Full observed curves retained; cross-method comparisons use shared_observed_ranges only.",
        "sources": {m:{"trace_files":r["trace_files"], "settings":r["settings"], "endpoint":r["endpoint"]} for m,r in runs.items()},
        "figures": sorted(p.name for p in folder.glob("*.pdf"))}
    (folder/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Separate output folder, never the old nine-panel folder")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--instances", nargs="+")
    parser.add_argument("--extra", nargs="*", choices=("reviews","diversity","operators","objectives"), default=[])
    parser.add_argument("--split-runtime", action="store_true")
    parser.add_argument("--figure-audit-scripts", type=Path, default=Path.home()/".codex/skills/nature-figure/scripts")
    args = parser.parse_args()
    if args.output.resolve() == (ROOT/"output/expanded_pilot").resolve():
        parser.error("Use a separate diagnostics output folder")
    sys.path.insert(0, str(args.figure_audit_scripts))
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        try:
            from .figure_alignment import require_matplotlib_panel_alignment
        except ImportError:
            from figure_alignment import require_matplotlib_panel_alignment
    report_bytes = args.report.read_bytes()
    report = json.loads(report_bytes)
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    previous_path = args.output/"manifest.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    if previous and previous.get("source_sha256") != report_hash:
        parser.error("This output belongs to another report snapshot; choose a fresh output folder")
    rows = [r for r in report["instances"] if (args.seeds is None or r["seed"] in args.seeds)
            and (args.instances is None or r["instance"] in args.instances)]
    if not rows:
        parser.error("No matching instance/seed rows")
    manifests = [render_row(report,r,args.output,require_matplotlib_panel_alignment,args.extra,args.split_runtime) for r in rows]
    indexed_rows = {(r["instance"],r["seed"]):r for r in previous.get("rows", [])}
    indexed_rows.update({(m["instance"],m["seed"]):{"instance":m["instance"], "seed":m["seed"],
                                                   "warnings":m["warnings"]} for m in manifests})
    summary = {"source_report": str(args.report.resolve()), "source_sha256": report_hash,
               "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "matplotlib_version": matplotlib.__version__,
               "rows": list(indexed_rows.values())}
    (args.output/"manifest.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"rows":len(manifests), "output":str(args.output)}))


if __name__ == "__main__":
    main()
