# Random Controller

Offline uniform-random baseline controller for the shared ITC 2019 NSGA-III solver. It uses the same solver configuration, built-in repairs, review gate, region builder, and bounded operators as the LLM and greedy controllers while making zero LLM or API calls.

See the [repository README](../README.md) for benchmark profiles, pipeline comparisons, and multi-worker batch execution.

---

## 1. Purpose and Control Policy

`RandomController` (`Random Controller/controller.py`, inheriting from `ControllerCallback` in `NSGA-III/itc2019/controller.py`) sets `mode = "random"` and `requires_llm = False`. Under `ControllerSettings`, it requires `provider = "none"`, `model = None`, and `max_api_calls = 0`, and executes through `Random Controller/main.py` using the shared CLI in `NSGA-III/main.py` (`pipeline_name = "NSGA-III + Random"`).

At each triggered review, `RandomController.decide(packet)` applies the following policy:

- **Uniform action space:** Collects every `(region, operator)` pair across `packet["candidate_regions"]` and `region["available_operators"]` in order (`move`, `destroy_repair` when the region has movable classes, and `swap` when compatible swap pairs exist), plus one `no_op` alternative. When at least one pair exists and `budget > 0`, it draws `chosen` uniformly over `0 .. len(pairs)` (`len(pairs) + 1` equal-probability options):
  - `0 <= chosen < len(pairs)`: selects that `(region, operator)` pair with `status = "policy_decision"` and `validation = "valid_action"`.
  - `chosen == len(pairs)`: selects `no_op` (`execution_budget = 0`) with `status = "policy_decision"` and `validation = "valid_no_op"`.
  - Empty `pairs` or `budget <= 0`: falls back to `no_op` (`execution_budget = 0`) with `status = "no_legal_action"` and `validation = "valid_no_op"`.
- **Fixed execution budget:** Sets `execution_budget = min(FIXED_EXECUTION_BUDGET, self.current_operator_limit)`, where `FIXED_EXECUTION_BUDGET = 10`. Before `decide`, the review gate initializes `self.current_operator_limit` to `max_operator_evaluations` (default `24`) and, when `--evaluations` is configured, caps it to `min(max_operator_evaluations, remaining_evaluations - 1)` after subtracting `RegionBuilder` scoring calls (reserving one evaluation for a candidate injection recheck). When an action other than `no_op` is chosen, `apply_operator` charges all probe evaluations to the solver evaluator and performs one additional `scoring_rechecks` call whenever an admissible candidate is found and rechecked before NSGA-III survival.
- **Locked review interval:** Always sets `next_review_interval` to `settings.review_interval` (default `10` generations, configurable via `--review-interval` within `[min_review_interval, max_review_interval]`).
- **RNG isolation:** `RandomController` draws policy choices from a dedicated `policy_rng` (`numpy.random.default_rng` seeded from the SHA-256 digest of `"{instance.name}:{seed}:controller-policy"`). This leaves the pymoo solver RNG (`--seed`), the shared operator RNG (`"{instance.name}:{seed}:operators"`), and the built-in repair RNGs undisturbed across pipelines.
- **Zero LLM or API calls:** Records `api_requests: 0` and `latency_seconds: 0.0` in each role entry, never imports or initializes LLM provider clients, and never triggers API failure exits.

---

## 2. Event Trigger and State Input

### Shared Review Trigger Rules

`RandomController` inherits the exact review gate implemented in `ControllerCallback._trigger` and `ControllerCallback.notify` (`NSGA-III/itc2019/controller.py`):

1. **Hysteresis resets (checked every generation):**
   - If `stagnant_generations < stagnation_window` (default `5`), the stagnation latch resets so a later plateau can trigger again.
   - If `placement_diversity > diversity_recovery_threshold` (default `0.30`), the low-diversity latch resets so a later collapse can trigger again.
2. **Minimum gating conditions:** No review can fire while `generation < warmup_generations` (default `5`) or `generation - last_review < min_review_interval` (default `3`, with `last_review` initialized to `-min_review_interval`).
3. **Trigger conditions (evaluated once the gates open):**
   - `deadline`: `generation >= next_review` (`next_review` starts at `warmup_generations` and advances to `generation + next_review_interval` after each review, so every review schedules the next deadline `review_interval` generations later by default).
   - `stagnation`: `stagnant_generations >= stagnation_window` (default `5`) and not yet handled since the last progress change, where progress tracks `("hard", incumbent_hard)` while infeasible and `("feasible_total", incumbent_internal_total)` once `incumbent_hard == 0`.
   - `low_diversity`: `placement_diversity <= diversity_threshold` (default `0.20`) and not yet handled since diversity exceeded `diversity_recovery_threshold` (default `0.30`).
4. **Execution limits:** Even when triggered, a review is skipped if `remaining_evaluations < 4` (recorded in `budget_skipped_reviews`), if remaining wall-clock time is `<= 0.1s`, or if `max_reviews` (default `12`) has already been reached.

### `StatePacket` Fields Consumed

Before calling `decide(packet)`, `ControllerCallback` builds a full `StatePacket` containing:

- `instance` and `global` (non-redundant population and budget metrics including `incumbent_hard`, `incumbent_internal_total`, `stagnant_generations`, `remaining_evaluations` adjusted for region scoring, `reviews_used`, `max_reviews`, and optional `remaining_seconds`).
- `solutions`: deduplicated per-solution scores (`hard`, `total`, optional `unlocalized_penalties`) and class summaries (`classes` keyed by `class_id`).
- `candidate_regions`: up to `max_regions` (default `6`) connected neighborhoods of up to `region_size` (default `5`) classes built around the highest-penalty classes across up to two distinct representative solutions.
- `recent_outcomes`: last `recent_outcomes` (default `5`) compacted review outcomes.
- `allowed_actions`: `execution_budget` bounds `[1, current_operator_limit]`, and `review_interval` bounds `[min_review_interval, max_review_interval]`.

`RandomController.decide` consumes **only** `packet["candidate_regions"]` (specifically `available_operators`, `solution_id`, `region_id`, and `boundary_id` on each candidate region), together with `self.current_operator_limit` and `self.settings.review_interval`. It ignores `global`, `solutions`, `recent_outcomes`, `allowed_actions`, and region penalty or class summaries, while still writing the complete `StatePacket` to `controller_trace.jsonl` so traces match the schema of the other controller pipelines.

---

## 3. Setup and CLI Execution

Commands below run from the `Experiment/` directory (Python 3.10 or later).

### Installation

Install only the baseline solver dependencies (no LLM provider client or API key is required):

```cmd
python -m pip install -r NSGA-III/requirements.txt
```

### Inspect and Dry Run

Inspect an ITC 2019 XML instance using the direct entrypoint without running optimization:

```cmd
python "Random Controller/main.py" --instance wbg-fal10 --inspect
```

Preview the exact solver command and budget without running it:

```cmd
python run_experiment.py --pipeline random --instance muni-fsps-spr17 --seed 17 --dry-run
```

### Running Optimization

Run via `run_experiment.py` using the default experiment settings (`settings/experiment.json`, default `--seed 17`):

```cmd
python run_experiment.py --pipeline random --instance muni-fsps-spr17 --seed 17
```

Omit `--instance` to run all 12 instances in [settings/experiment.json](../settings/experiment.json), pass `--output ExperimentResult` to write under a shared campaign folder, or `--workers N --live` to execute multiple offline jobs in parallel.

Run directly via `Random Controller/main.py` when specifying custom solver, repair, or review-gate flags:

```cmd
python "Random Controller/main.py" --instance muni-fsps-spr17 --seed 17 --population 20 --generations 200 --evaluations 1800
```

Direct CLI defaults inherit from [settings/experiment.json](../settings/experiment.json): `--population 20`, `--generations 200`, `--evaluations 1800`, `--partitions 3`, `--seed 17`, `--mutation-rate 0.35`, `--mutation-fraction 0.01`, `--crossover-rate 0.7`, `--dataset Dataset_ITC2019`, `--output "Random Controller/results_itc2019"`, `--review-interval 10`, `--warmup-generations 5`, `--stagnation-window 5`, `--min-review-interval 3`, `--max-review-interval 20`, `--diversity-threshold 0.20`, `--diversity-recovery-threshold 0.30`, `--max-regions 6`, `--region-size 5`, `--max-operator-evaluations 24`, and `--max-reviews 12`. `--stop-on-feasible`, `--manifest FILE`, `--all`, and `--repair-*` options are also supported. Run `python "Random Controller/main.py" --help` for the full option list.

### Live Progress Flags

When invoking `Random Controller/main.py` directly, you can control live logging with:

- `--quiet`: suppress live progress output on `stderr` while still writing each instance's `run.log`.
- `--progress-interval N`: emit generation lines every `N` generations (default `1`). Set `0` to log only `[start]`, `[done]`, review decisions/outcomes, and repair events.
- `--progress-log PATH`: optional additional file path where live progress lines are appended alongside each instance's `run.log`.

---

## 4. Output Artifacts

By default, direct runs write to `Random Controller/results_itc2019/<run_id>/`. When run with `run_experiment.py --output DIR`, results are written to `DIR/random/<run_id>/`. Each `<run_id>` encodes the instance count, population, generations, partitions, seed, and a 16-character SHA-256 fingerprint of all solver, repair, controller, and source-file settings.

Run-level files under `<output>/<run_id>/`:

- `run_config.json`: `run_id`, resolved solver/repair/controller settings (`fixed_execution_budget: 10`), Python and library versions, XML and source SHA-256 hashes, completion timestamp, and `completion_status` (`"completed"`).
- `summary.json`: JSON list aggregating every per-instance `result.json` summary in the run.

Per-instance files under `<output>/<run_id>/<instance>/`:

- `result.json`: runtime, evaluation breakdown (`objective_and_monitor`, `routine_repair`, `final_verification`), `score_engine`, `sectioning_fallback`, `stage_timing`, final score, internal feasibility status (`official_status: "not_run"`), `monitor` stats, and `controller` stats (`mode: "random"`, `reviews`, `budget_skipped_reviews`, `no_op`, `policy_no_op`, `fallback_no_op`, `interventions`, `selected_interventions`, `region_scoring_calls`, `operator_scoring_rechecks`, and `api_calls: 0`).
- `anytime_trace.jsonl`: generation-by-generation `population_metrics`, cumulative stage timings, and incumbent quality (`best_observed_hard`, `best_observed_internal_total`).
- `controller_trace.jsonl`: one JSON line per review containing `generation`, `trigger`, full `state` (`StatePacket`), `roles` (`role: "random"`, `policy: "uniform_random"`, `valid_pairs`, `chosen_index`, `api_requests: 0`, `latency_seconds: 0.0`, `status`), `decision_status`, `validation`, `action`, `outcome`, `population_after`, and `incumbent_after`.
- `choices.npz`: compressed NumPy archive storing `time_indices` and `room_indices` of the final incumbent solution.
- `run.log`: human-readable progress log (`[start]`, `[repair @ gen ...]`, `[gen ...]`, `[review #...]`, `decision:`, `outcome:`, and `[done]`).
- `solution.xml` (plus `expected.json`): exported only when the final incumbent is internally feasible (`score.hard == 0`) with valid student sectioning. Official ITC 2019 validator verification is a separate step.
