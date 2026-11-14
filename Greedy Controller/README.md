# Greedy Controller

The Greedy Controller is an offline, deterministic-policy local search baseline that operates without LLM or external API calls (`requires_llm = False`, `provider = "none"`, `api_calls = 0`). It runs on top of the shared ITC 2019 NSGA-III solver (`NSGA-III/main.py`) and serves as a non-LLM reference policy alongside the Random, Single-Agent, and Multi-Agent controllers.

---

## 1. Control Policy

Implemented by `GreedyController` in `Greedy Controller/controller.py` (overriding `ControllerCallback._review`), the controller makes all decisions locally and deterministically from the current population and candidate regions, requiring no Gemini SDK, no `GEMINI_API_KEY`, and no Free Tier confirmation.

### Candidate Region Ranking (`WHERE`)

At each triggered review, `GreedyController` filters `regions` to candidates with at least one legal operator (`eligible = [r for r in regions if r.available_operators]`) and computes the shared review probe budget:

```text
budget = min(FIXED_EXECUTION_BUDGET, self.current_operator_limit)
```

where `FIXED_EXECUTION_BUDGET = 10` (`NSGA-III/itc2019/controller.py`) and `self.current_operator_limit` is `max_operator_evaluations` (default `24`) capped by `remaining_evaluations - 1` (after deducting region-building scoring calls and reserving 1 evaluation for the potential injection recheck).

Early `no_op` exits occur before any operator probing when:
- `eligible` is empty or `budget < 1` (`decision_status = "no_legal_action"`)
- Every eligible region has `hard_contribution == 0`, `soft_contribution == 0`, and empty `unlocalized_penalties` (`decision_status = "no_region_penalty"`)

Otherwise, eligible regions are ranked relative to their source individual `algorithm.pop[int(region.solution_id)]` (`hard` from `G[0]` and weighted objective sum `total = sum(F)`):
- If any eligible region has `hard_contribution > 0`, regions are ordered by minimizing `(-region.hard_contribution, -region.soft_contribution, hard, total, region.region_id)`.
- If all eligible regions have `hard_contribution == 0`, regions are ordered by minimizing `(-region.soft_contribution, total, region.region_id)`.

The single top-ranked candidate region is selected for probing.

### Sequential Operator Probing (`HOW`)

Legal operators on the chosen region are ordered canonically according to `OPERATORS = ("move", "swap", "destroy_repair")` (`NSGA-III/itc2019/operators.py`):

1. The shared `budget` is split across the `k` legal operators via `base, extra = divmod(budget, k)`, giving operator `i` a quota of `base + int(i < extra)` evaluations (for example, quotas `4, 3, 3` when `budget = 10` and all 3 operators are available).
2. Each operator is probed sequentially via `probe_operator(algorithm, instance, region, op, quota, self.rng, self.deadline)` against the unchanged source individual. Probing stops early if `quota == 0` or `self.deadline` is reached.
3. Operator proposals use `self.rng` (seeded per instance and solver seed via SHA-256 of `"{instance.name}:{seed}:operators"`), while region ranking, budget division, and tie-breaking are deterministic.
4. Probe evaluations do not modify `algorithm.pop`. Every evaluation is charged to `algorithm.evaluator.n_eval` and recorded under `role["probes"]` (`operator_id`, `quota`, `evaluations`, `best_trial`, `candidate_quality`).

### Improvement Criterion and `no_op` Fallback

Inside `probe_operator`, candidate admissibility (`_admissible_candidate`) is judged lexicographically as `(hard, total)` against the chosen region's source/base solution `(source_hard, source_total)`:
- When `source_hard > 0`, a trial is admissible only if `trial_hard < source_hard` (strict hard-violation reduction).
- When `source_hard == 0`, a trial is admissible only if `trial_hard == 0` and `trial_total < source_total` (preserves feasibility and strictly lowers total weighted soft penalty).

Across all probed operators on the selected region, `GreedyController` picks the operator whose best admissible candidate has the lexicographically smallest `probe.candidate_quality = (hard, total)` (maximizing immediate improvement over the shared source solution). Exact ties in `(hard, total)` retain the earlier operator in `OPERATORS`. Meanwhile, the solver tracks the global run incumbent separately via `best_observed_lexicographic_hard_then_total`.

If no probed operator produces an admissible improving candidate, `GreedyController` falls back to `ActionPlan(None, None, None, "no_op", 0, self.settings.review_interval)` with `decision_status = "no_improving_candidate"`, leaving `algorithm.pop` unchanged. In `ControllerCallback.report()`, this counts as an intentional `policy_no_op` (`fallback_no_op = 0`).

### Review Interval (`WHEN`) and Evaluation Accounting

- **Next review interval**: `next_review_interval` is always locked to `self.settings.review_interval` (default `10` generations) for both active interventions and `no_op` decisions. Event triggers (`stagnation` or `low_diversity`) can still initiate an earlier review subject to `min_review_interval`.
- **Probe and recheck accounting**: All evaluations spent across operator probes are summed into `probe_evaluations` and charged to `algorithm.evaluator.n_eval` even when Greedy falls back to `no_op`. When an improving candidate is returned (`decision_status = "policy_decision"`), `ControllerCallback.notify()` calls `inject_operator_candidate(algorithm, review.probe)` instead of re-running `apply_operator`. `inject_operator_candidate` performs `1` `score_state` verification check (`scoring_rechecks = 1`, added to `algorithm.evaluator.n_eval` and `controller.operator_scoring_rechecks`) before passing the single candidate to `algorithm.survival.do(...)`.

---

## 2. Event Trigger & State Input

`GreedyController` inherits the shared event-trigger gate and Schema v1 `StatePacket` pipeline from `ControllerCallback` (`NSGA-III/itc2019/controller.py`).

### Review Trigger Rules

Checked at the end of each generation (after routine solver repairs and incumbent updates), a review triggers only when all preconditions hold:
- `generation >= warmup_generations` (default `5`)
- `generation - last_review >= min_review_interval` (default `3`)
- `remaining_evaluations is None or remaining_evaluations >= 4` (otherwise increments `budget_skipped_reviews` and skips)
- `remaining() is None or remaining() > 0.1` seconds (when a time limit is active)
- `len(self.events) < max_reviews` (default `12`)

When eligible, a review fires if at least one trigger condition is satisfied:
- `deadline`: `generation >= next_review` (starts at `warmup_generations`, then `generation + next_review_interval`, bounded by `[min_review_interval, max_review_interval]` = `[3, 20]`)
- `stagnation`: `stagnant_generations >= stagnation_window` (default `5`) and not yet handled for the current stagnation streak (resets when `_progress_signal(best_observed)` advances)
- `low_diversity`: `placement_diversity <= diversity_threshold` (default `0.20`) and not yet handled until `placement_diversity > diversity_recovery_threshold` (default `0.30`)

### StatePacket & Region Inputs

When a review triggers, `RegionBuilder.build` (`NSGA-III/itc2019/regions.py`) constructs up to `max_regions` (default `6`) conflict-coupled `RegionCandidate` objects of up to `region_size` (default `5`) classes across at most 2 distinct representative solutions. `ControllerCallback._packet` assembles the Schema v1 `StatePacket` containing `schema_version`, `instance`, `global` population metrics (`population_metrics`), `recent_best_quality`, public `candidate_regions`, `recent_outcomes`, and `allowed_actions`.

The complete `StatePacket` is logged to `controller_trace.jsonl` under `"state"` for cross-pipeline consistency. `GreedyController._review` reads `regions` (`solution_id`, `region_id`, `boundary_id`, `class_indices`, `hard_contribution`, `soft_contribution`, `unlocalized_penalties`, `available_operators`), `algorithm.pop` (`F`, `G`, `X`), `self.current_operator_limit`, `self.deadline`, `self.rng`, and `self.settings.review_interval`.

---

## 3. Setup and CLI Execution

All commands below run from the repository root (`Experiment/`). Python 3.10 or later is required.

Install the shared baseline dependencies:

```cmd
python -m pip install -r NSGA-III/requirements.txt
```

Inspect an instance without running optimization:

```cmd
python "Greedy Controller/main.py" --instance wbg-fal10 --inspect
```

Preview pilot commands and per-instance budgets with `--dry-run`:

```cmd
python run_pilot.py --pipeline greedy --instance muni-fsps-spr17 --seed 17 --dry-run
```

Run optimization through the pilot runner (defaults to `profiles/pilot.json`, omit `--instance` to run all profile instances, or pass `--profile profiles/expanded_pilot.json` and `--output DIR`):

```cmd
python run_pilot.py --pipeline greedy --instance muni-fsps-spr17 --seed 17
```

Run the direct entrypoint directly (see `python "Greedy Controller/main.py" --help` for all solver and controller flags):

```cmd
python "Greedy Controller/main.py" --instance muni-fsps-spr17 --seed 17 --population 20 --generations 200 --evaluations 1200 --time-limit-seconds 630
```

Live progress flags on `Greedy Controller/main.py`:
- `--quiet`: suppress live progress output on `stderr` while retaining `run.log`
- `--progress-interval N`: emit generation progress every `N` generations (default `1`, set `0` to log reviews and repairs only)
- `--progress-log PATH`: append live progress lines to an additional log file path

---

## 4. Output Artifacts

Direct runs write under `Greedy Controller/results_itc2019/<run_id>/` by default (override with `--output`). Running via `run_pilot.py --output DIR` writes under `DIR/greedy/<run_id>/` (without `--output`, it writes to `Greedy Controller/results_itc2019/<run_id>/`).

Run-level files under `<run_id>/`:
- `run_config.json`: run ID, resolved solver/repair/controller settings (`provider: "none"`, `model: null`, `max_api_calls: 0`, `fixed_execution_budget: 10`, no `llm` section), XML SHA-256 hashes, dependency versions, `controller_sources_sha256`, and `completion_status`
- `summary.json`: JSON array summarizing all instances in the run, including final score, runtime, `evaluation_breakdown`, `score_engine`, `monitor`, and `controller` counters (`probe_evaluations`, `policy_no_op`, `fallback_no_op`, `operator_scoring_rechecks`, `region_scoring_calls`)

Per-instance files under `<run_id>/<instance>/`:
- `result.json`: single-instance summary record matching its entry in `summary.json`
- `anytime_trace.jsonl`: generation-by-generation search metrics, stage timings, and incumbent quality
- `controller_trace.jsonl`: per-review trace including `generation`, `trigger`, `state` (`StatePacket`), `roles` (with the `"greedy"` record and its `"probes"` list of `operator_id`, `quota`, `evaluations`, `best_trial`, `candidate_quality`), `decision_status`, `validation`, `action`, `outcome`, `population_after`, `incumbent_after`, and `probe_evaluations`
- `choices.npz`: compressed NumPy archive storing `time_indices` and `room_indices` of the final best chromosome
- `run.log`: human-readable live progress and completion log
- `solution.xml` (with `expected.json`): exported only when the final solution is internally feasible (`score.hard == 0`) with valid student sectioning (official ITC 2019 validation is a separate step)

See the [repository README](../README.md) for pipeline comparisons, profiles, and pilot workflows.
