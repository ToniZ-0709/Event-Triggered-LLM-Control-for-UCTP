"""Plot archived Phase 1 results using Matplotlib."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .phase1_data import ARCHIVE, METHODS, archived_runs, read_jsonl, report_rows
except ImportError:
    from phase1_data import ARCHIVE, METHODS, archived_runs, read_jsonl, report_rows

COLORS = {"baseline": "#6b7280", "single": "#2563eb", "multi": "#d97706"}
LABELS = {"baseline": "Baseline", "single": "Single", "multi": "Multi"}


def plot(root: Path, output: Path, seed: int, instance: str, graph_types=("final_quality", "convergence", "resources")):
    candidates = [(campaign, row) for campaign, row in report_rows(root)
                  if row["seed"] == seed and row["instance"] == instance]
    if len(candidates) != 1:
        raise ValueError(f"Expected one archived comparison for {instance}, seed {seed}; got {len(candidates)}")
    campaign, row = candidates[0]
    archive = archived_runs(root)
    paths = {}
    for method in METHODS:
        key = (campaign, seed, instance, method)
        if key not in archive:
            raise ValueError(f"Missing archived run: {key}")
        paths[method] = archive[key][0]
    traces = {}
    reviews = {}
    if "convergence" in graph_types:
        for method, folder in paths.items():
            child = folder / instance
            traces[method] = read_jsonl(child / "anytime_trace.jsonl")
            reviews[method] = read_jsonl(child / "controller_trace.jsonl")
            if not traces[method] or (method != "baseline" and not reviews[method]):
                raise ValueError(f"Missing trace for {method}: {child}")
    output.mkdir(parents=True, exist_ok=True)
    note = "No global time cap" if campaign == "tg_no_timecap" else "Original time-capped campaign"
    title = f"{instance} | seed {seed} | {note}"

    if "final_quality" in graph_types:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        feasible = all(row["methods"][m]["hard"] == 0 for m in METHODS)
        field = "internal_total" if feasible else "hard"
        values = [row["methods"][m][field] for m in METHODS]
        bars = ax.barh([LABELS[m] for m in METHODS], values, color=[COLORS[m] for m in METHODS])
        ax.bar_label(bars, padding=4)
        ax.invert_yaxis()
        ax.set_xlabel("Internal soft total (all feasible)" if feasible else "Hard violations (lower is better)")
        ax.set_title(title)
        fig.savefig(output / "final_quality.png", dpi=180)
        plt.close(fig)

    if "convergence" in graph_types:
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
        for method in METHODS:
            trace = traces[method]
            for column, key in enumerate(("evaluations", "elapsed_seconds", "generation")):
                ax = axes[0, column]
                points = [p for p in trace if p.get(key) is not None and p.get("best_observed_hard") is not None]
                if points:
                    ax.step([p[key] for p in points], [p["best_observed_hard"] for p in points],
                            where="post", label=LABELS[method], color=COLORS[method])
                feasible_points = [p for p in points if p["best_observed_hard"] == 0
                                   and p.get("best_observed_internal_total") is not None]
                if feasible_points:
                    axes[1, column].step([p[key] for p in feasible_points],
                                         [p["best_observed_internal_total"] for p in feasible_points],
                                         where="post", color=COLORS[method])
                by_gen = {p["generation"]: p for p in trace}
                marks = [by_gen[e["generation"]] for e in reviews[method] if e.get("generation") in by_gen]
                marks = [p for p in marks if p.get(key) is not None and p.get("best_observed_hard") is not None]
                if marks:
                    ax.scatter([p[key] for p in marks], [p["best_observed_hard"] for p in marks],
                               marker="|", color=COLORS[method], s=70)
                soft_marks = [p for p in marks if p["best_observed_hard"] == 0
                              and p.get("best_observed_internal_total") is not None]
                if soft_marks:
                    axes[1, column].scatter([p[key] for p in soft_marks],
                        [p["best_observed_internal_total"] for p in soft_marks],
                        marker="|", color=COLORS[method], s=70)
        for column, label in enumerate(("Evaluations", "Elapsed seconds", "Generation")):
            axes[0, column].set_xlabel(label)
            axes[0, column].set_ylabel("Incumbent hard violations")
            axes[1, column].set_xlabel(label)
            axes[1, column].set_ylabel("Soft total after feasibility")
        axes[0, 0].legend()
        fig.suptitle(title + " | ticks: review-generation checkpoints, not causal effect")
        fig.savefig(output / "convergence.png", dpi=180)
        plt.close(fig)

    if "resources" in graph_types:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for ax, field, label in zip(axes, ("runtime_seconds", "api_calls", "total_native_tokens"),
                                     ("Runtime (s)", "API calls", "Native tokens")):
            values = [row["methods"][m].get(field) for m in METHODS]
            if any(v is None for v in values):
                raise ValueError(f"Missing {field}; do not substitute zero")
            bars = ax.bar([LABELS[m] for m in METHODS], values, color=[COLORS[m] for m in METHODS])
            ax.bar_label(bars, padding=3, fmt="%.0f")
            ax.set_ylabel(label)
            ax.margins(y=.18)
        fig.suptitle(title)
        fig.savefig(output / "resources.png", dpi=180)
        plt.close(fig)


def select_comparisons(root: Path, instances=None, seeds=None):
    rows = report_rows(root)
    available_instances = {row["instance"] for _, row in rows}
    available_seeds = {row["seed"] for _, row in rows}
    unknown_instances = set(instances or ()) - available_instances
    unknown_seeds = set(seeds or ()) - available_seeds
    if unknown_instances or unknown_seeds:
        raise ValueError(f"Unknown instances: {sorted(unknown_instances)}, seeds: {sorted(unknown_seeds)}")
    selected = [(row["instance"], row["seed"]) for _, row in rows
                if (not instances or row["instance"] in instances) and (not seeds or row["seed"] in seeds)]
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate instance and seed across reports")
    if not selected:
        raise ValueError("No matching instance and seed")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", nargs="+", help="One or more instance IDs, default: all")
    parser.add_argument("--seed", nargs="+", type=int, help="One or more seeds, default: all")
    parser.add_argument("--graph", nargs="+", choices=("final_quality", "convergence", "resources"),
                        default=("final_quality", "convergence", "resources"), help="Graph types, default: all")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, help="Output folder, default: archive/graphs")
    args = parser.parse_args()
    if any(not name or any(not (c.isascii() and (c.isalnum() or c in "-_")) for c in name)
           for name in args.instance or ()):
        parser.error("Instance must contain only letters, digits, hyphens or underscores")
    archive = args.archive.resolve()
    graphs = (archive / "graphs").resolve()
    base = (args.output or graphs).resolve()
    if base.is_relative_to(archive) and not base.is_relative_to(graphs):
        parser.error("Inside the archive, write figures only under graphs/")
    try:
        selected = select_comparisons(archive, args.instance, args.seed)
        for instance, seed in selected:
            output = (base / instance / f"seed_{seed}").resolve()
            if not output.is_relative_to(base):
                raise ValueError("Output path outside selected directory")
            plot(archive, output, seed, instance, args.graph)
            print(f"Saved {instance}, seed {seed}: {output}")
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
