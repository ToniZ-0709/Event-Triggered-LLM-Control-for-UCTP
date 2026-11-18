# NSGA-III Baseline

The baseline runs the shared ITC 2019 NSGA-III solver with built-in construction, student sectioning, offspring hard repair, and periodic CP-SAT and plateau repairs. It serves as both the standalone baseline pipeline (`baseline`) and the shared evolutionary substrate reused by every controller pipeline, making zero LLM or external API calls.

See the [repository README](../README.md) for cross-pipeline setup, profile summaries, and runner workflows.

---

## 1. Solver Architecture and Pipeline

The solver optimizes a complete class placement for one independent ITC 2019 XML instance per job in [itc2019/nsga3.py](itc2019/nsga3.py):

1. **Chromosome Encoding and Objectives (`PlacementProblem`)**:
   - Decision vector length is `2 * n` integer genes for `n` classes: the first `n` entries select each class's time option index, and the next `n` entries select its room option index (`-1` when `no_room` is true).
   - Minimizes four weighted ITC 2019 soft objectives (`F = [time, room, distribution, student]` multiplied by `instance.weights`) subject to one inequality constraint (`G = [score.hard]`), where `score.hard = room_conflicts + room_unavailable + hard_distributions + unassigned_requests`.
   - Student sectioning inside [itc2019/scoring.py](itc2019/scoring.py) uses a deterministic capacity-aware greedy/beam decoder (`_greedy_section_students`) with an OR-Tools CP-SAT reallocation fallback (`repair_sectioning`) when any requested course cannot initially be assigned.
   - `ScoreEngine` maintains a bounded 16-entry LRU cache per instance, while still counting logical score requests toward the evaluation budget so cached re-checks do not silently expand search effort.

2. **Initialization (`PlacementSampling`)**:
   - Builds each initial individual via `ConstructionIndex` in [itc2019/construction.py](itc2019/construction.py) using bounded forward-checking over time/room domains filtered by hard `SameRoom`/`SameStart` restrictions and conflict lookahead.
   - Applies `_improve_hard` (`--initial-repair-steps`, default `12`, and `--initial-pair-trials`, default `3`) to resolve single-class and bounded two-class hard conflicts before generation 1.

3. **Reference-Direction Selection and Variation (`BudgetedNSGA3`)**:
   - Uses 4D Das-Dennis reference directions from `--partitions` (default `3`), with an optional scaled inner layer (`--inner-partitions`, `--inner-scale`). Population size must be at least the number of reference directions.
   - Parent selection uses `_seeded_cv_tournament`, routing all infeasible-tie comparisons through the solver RNG (avoiding `pymoo` 0.6.2's unseeded tie branch).
   - Variation applies `ClassCrossover` (uniform class-wise swap of `(time, room)` pairs at `--crossover-rate`), `ClassMutation` (re-sampling time and room genes for `max(1, round(n * mutation_fraction))` mutable classes at `--mutation-rate`), and `RoomRepair` (`_improve_hard` for `--offspring-repair-steps` and `--offspring-pair-trials`).
   - `BudgetedNSGA3._infill()` shrinks the final offspring batch size to `remaining_scores()` so a run does not overshoot `--evaluations`.

4. **Anytime Incumbent and Termination**:
   - `Incumbent` records the lexicographically best `(hard, total)` candidate across every full `score_state` evaluation, ensuring low-total feasible solutions discarded by reference-direction survival are never lost.
   - Runs terminate on the earliest of `--generations` (`MaximumGenerationTermination`), `--time-limit-seconds` (`TimeBasedTermination`), `--evaluations` (`FullEvaluationTermination` over `evaluator.n_eval + repair_scoring_calls`), or `--stop-on-feasible` (`FeasibleSolutionTermination`). By default (`--stop-on-feasible` omitted), the solver continues optimizing soft cost after reaching feasibility.

---

## 2. Periodic Routine and CP-SAT Repairs

At the end of each generation, `BaselineMonitor.notify` in [itc2019/controller.py](itc2019/controller.py) runs up to three bounded intensification hooks on the best current schedule when at least 8 evaluations remain in budget:

| Repair stage | Cadence & gate | Mechanism & budget | Source |
| :--- | :--- | :--- | :--- |
| **Coordinated distribution repair** | Every `5` generations when `hard > 0`, `hard_distributions > 0`, `hard_distributions >= room_conflicts`, and `> 10s` remain | Seeds OR-Tools CP-SAT `distribution_neighborhood_repair` (up to `4` scoring calls) or up to `3` rounds of `_joint_improve_hard` (`8` pair trials) within `5.0s` | [itc2019/neighborhood.py](itc2019/neighborhood.py) |
| **Room plateau search** | Every `5` generations when `room_conflicts + room_unavailable >= 4` and `hard > near_feasible_hard` | Neutral/slack local search with tabu list (`max_steps=2000`, `max_restarts=100`, `max_changes=8`, `hard_slack=2`, up to `16` scoring calls, `5.0s` limit) | [itc2019/room_search.py](itc2019/room_search.py) |
| **Near-feasible room-chain repair** | Every `--repair-interval` (default `5`) generations when room conflicts/unavailable exist and `hard <= --repair-near-feasible-hard` (default `12`) | `adaptive_room_repair` widens OR-Tools CP-SAT neighborhoods across `(16, 16)`, `(32, 32)`, and `(64, 64)` classes/options, then falls back to beam search `_room_chain_repair` (`max_depth=6`, `beam_width=12`, `hard_slack=2`) | [itc2019/neighborhood.py](itc2019/neighborhood.py), [itc2019/repair_policy.py](itc2019/repair_policy.py) |

Near-feasible room-chain repairs are additionally gated by `FailureBackoff` in [itc2019/repair_policy.py](itc2019/repair_policy.py), keyed by `(source_hard, sorted(conflict_signatures))`:

- If a signature improves (`selected` or `rejected_by_selection`), its cooldown resets to `--repair-failure-cooldown` (default `5` generations).
- Repeated non-improving attempts on the same signature double the cooldown exponentially (`initial * 2^(failures - 1)`) up to `--repair-max-failure-cooldown` (default `20` generations), avoiding wasted CP-SAT calls on stuck states and incrementing `room_repair_backoff_skips`.

---

## 3. Shared Controller and Review Gate Hooks

[main.py](main.py) exports `main(controller_factory=..., pipeline_root=..., pipeline_name=..., controller_sources=...)` so all controller pipelines share the identical solver loop, repair budget, CLI options, and artifact writer:

- `ControllerCallback` subclasses `BaselineMonitor` in [itc2019/controller.py](itc2019/controller.py), running the three periodic repairs first and then forwarding `room_chain_checked=True`, `room_plateau_checked=True`, `coordinated_checked=True` to `super().notify(...)` so repairs never execute twice in one generation.
- When a review triggers (`deadline`, `stagnation`, or `low_diversity`), `RegionBuilder` constructs conflict/cost regions into `StatePacket`, validates the returned `ActionPlan`, executes `move`, `swap`, `destroy_repair`, or `no_op`, and feeds any improving candidate into NSGA-III environmental survival.

---

## 4. Setup and Running

Install the baseline dependencies from the `Experiment/` repository root:

```cmd
python -m pip install -r NSGA-III/requirements.txt
```

Inspect an instance without running optimization:

```cmd
python NSGA-III/main.py --instance wbg-fal10 --inspect
```

Preview or run the calibrated pilot profile via `run_pilot.py` from `Experiment/`:

```powershell
python run_pilot.py --pipeline baseline --instance muni-fsps-spr17 --seed 17 --dry-run
python run_pilot.py --pipeline baseline --instance muni-fsps-spr17 --seed 17
```

Omit `--instance` to run all instances defined in the selected profile ([profiles/pilot.json](../profiles/pilot.json) by default, or pass `--profile profiles/phase1.json` or `--profile profiles/phase2.json`).

### Direct CLI Options

Direct invocations of `python NSGA-III/main.py` default to `population=20`, `generations=20`, `partitions=3`, `seed=42`, `mutation-rate=0.2`, `mutation-fraction=0.005`, and `crossover-rate=0.7` (whereas `run_pilot.py` applies profile presets such as `generations=200`, `mutation-rate=0.35`, and `mutation-fraction=0.01`). Inspect all flags with `python NSGA-III/main.py --help`.

| Option group | Flags | Notes |
| :--- | :--- | :--- |
| **Instance selection** | `--dataset`, `--instance`, `--manifest`, `--all`, `--inspect` | Default dataset is `Dataset_ITC2019/`. Repeat `--instance` or supply a newline manifest file |
| **Search & termination** | `--population`, `--generations`, `--partitions`, `--inner-partitions`, `--inner-scale`, `--seed`, `--evaluations`, `--time-limit-seconds`, `--stop-on-feasible` | `--evaluations` caps `n_eval + repair_scoring_calls`. `--stop-on-feasible` exits after any generation produces `hard == 0` |
| **Variation & offspring repair** | `--mutation-rate`, `--mutation-fraction`, `--crossover-rate`, `--initial-repair-steps`, `--initial-pair-trials`, `--offspring-repair-steps`, `--offspring-pair-trials` | Controls genetic variation and local hard-repair intensity |
| **Shared CP-SAT repair** | `--repair-interval`, `--repair-near-feasible-hard`, `--repair-neighborhood-classes`, `--repair-neighborhood-options`, `--repair-neighborhood-seconds`, `--repair-neighborhood-deterministic-limit`, `--repair-fallback-seconds`, `--repair-fallback-evaluations`, `--repair-failure-cooldown`, `--repair-max-failure-cooldown` | Controls periodic CP-SAT neighborhood size, runtime, fallback beam budget, and `FailureBackoff` cooldowns |
| **Live progress logging** | `--progress-interval N`, `--progress-log PATH`, `--quiet` | Default `--progress-interval 1` logs every generation (`0` logs `[start]`, repairs, and `[done]` only). `--quiet` mutes `stderr` while keeping `run.log` and optional `--progress-log` |
| **Output & XML metadata** | `--output`, `--author`, `--institution`, `--country` | Default `--output` is `NSGA-III/results_itc2019` |

Example direct run stopping as soon as feasibility is reached and writing an extra progress log:

```powershell
python NSGA-III/main.py --instance wbg-fal10 --generations 100 --evaluations 1200 --stop-on-feasible --progress-interval 5 --progress-log wbg-progress.log
```

---

## 5. Output Artifacts and Verification Caveat

Direct runs write to `NSGA-III/results_itc2019/<run_id>/`, while `run_pilot.py --output DIR` writes to `DIR/baseline/<run_id>/`. Runs stage into a temporary directory and atomically publish only when settings match or the target directory does not yet exist (`run_id` includes a SHA-256 fingerprint of instance hashes, solver/repair settings, and environment/source hashes).

- `run_config.json`: Run ID, complete solver/repair parameters, instance XML SHA-256 hashes, Python/dependency versions, and source hashes.
- `summary.json`: Array of per-instance summary records for all instances in the run.
- `<instance>/result.json`: Final score, feasibility flag, evaluation breakdown (`objective_and_monitor`, `routine_repair`, `final_verification`), `score_engine` cache statistics, `stage_timing`, and monitor repair counts.
- `<instance>/anytime_trace.jsonl`: Generation-by-generation metrics (`best_hard`, `best_internal_total`, `best_observed_hard`, `best_observed_internal_total`, `placement_diversity`, `reference_direction_coverage`, `objective_distribution`).
- `<instance>/choices.npz`: Compressed `time_indices` and `room_indices` chromosome arrays for the best observed solution.
- `<instance>/run.log`: Human-readable progress log (`[start]`, `[repair @ gen ...]`, `[gen ...]`, `[done]`).
- `<instance>/solution.xml` and `<instance>/expected.json`: Exported only when `score.hard == 0` and `validate_sectioning` passes.

**Official validation caveat**: `solution.xml` export establishes internal feasibility under the solver's evaluator and deterministic student sectioning decoder. Both `result.json` (`validation.official_status`) and `expected.json` (`official_status`) record `"not_run"` by default. Independent verification against the official ITC 2019 validator must be performed separately (see [tools/validate_itc2019.py](../tools/validate_itc2019.py)).
