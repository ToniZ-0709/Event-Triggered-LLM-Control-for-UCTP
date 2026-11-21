# Single-Agent Controller

The Single-Agent Controller integrates one bounded LLM API decision per triggered review into the shared [NSGA-III baseline](../NSGA-III/README.md) solver across Google Gemini Developer API and Cloudflare Workers AI. At each review, a single structured JSON call decides **WHERE** to intervene (selecting a candidate region), **HOW** to intervene (selecting an operator and evaluation budget), and **WHEN** to schedule the next review.

See the [repository README](../README.md) for cross-pipeline setup, profile comparisons, and multi-worker execution.

---

## 1. Architecture and Control Flow

`SingleAgentController` in [controller.py](controller.py) subclasses `ControllerCallback` in [../NSGA-III/itc2019/controller.py](../NSGA-III/itc2019/controller.py). The solver continues to own population evolution, routine offspring repair, periodic CP-SAT and plateau repairs, incumbent archiving, and all chromosome mutations.

```text
Generation end -> BaselineMonitor repairs & incumbent sync
  -> Evaluate review triggers (deadline | stagnation | low_diversity)
  -> Check gates (budget >= 4, API active, time > 0.1s, reviews < max_reviews)
  -> RegionBuilder constructs candidate regions + StatePacket
  -> One LLM API call (role="single_agent", strict JSON schema / prompt_json)
     |-- api_* error (not deadline_skipped) -> api_disabled, force termination, exit 3
     '-- JSON validated against StatePacket IDs, operators, budgets, intervals
         |-- valid action -> probe operator within execution_budget -> recheck & merge best into NSGA-III survival
         '-- no_op / invalid decision -> apply no_op, schedule next review
```

### Event-Triggered Review Conditions

`ControllerCallback._trigger` evaluates review triggers at the end of each generation after shared `BaselineMonitor` repairs finish:

- **Warmup and spacing**: Reviews never trigger before `--warmup-generations` (default `5`) or sooner than `--min-review-interval` (default `3`) generations after the previous review.
- **Trigger conditions**:
  1. `deadline`: `generation >= next_review` (initialized to `warmup_generations`, then updated after each review to `generation + next_review_interval`).
  2. `stagnation`: `stagnant_generations >= --stagnation-window` (default `5`) when not already handled for the current stagnation streak. Resets whenever `best_observed` `(hard, total)` progress changes or `stagnant_generations` drops below the window.
  3. `low_diversity`: `placement_diversity <= --diversity-threshold` (default `0.20`), suppressed once handled until `placement_diversity` recovers above `--diversity-recovery-threshold` (default `0.30`).
- **Pre-call gates**: A triggered review proceeds only if `remaining_evaluations >= 4` (otherwise `budget_skipped_reviews` increments), `not api_disabled`, `api_calls < --max-api-calls`, `remaining_seconds > 0.1`, and `reviews < --max-reviews` (default `12`).

### StatePacket

When a review proceeds, `RegionBuilder.build` in [../NSGA-III/itc2019/regions.py](../NSGA-III/itc2019/regions.py) selects up to two diverse representative individuals (the lexicographically best `(hard, total)` schedule and the farthest eligible schedule in normalized objective and decision space) and grows up to `--max-regions` (`6`) local subgraphs of up to `--region-size` (`5`) connected classes. Region scoring calls are charged to the solver evaluation budget, and `current_operator_limit` is capped at `min(--max-operator-evaluations, remaining_evaluations - 1)`.

The serialized `StatePacket` contains:

- `instance`: Instance name
- `global`: Non-redundant population and budget metrics (`generation`, optional `remaining_seconds`, `evaluation_budget` + `remaining_evaluations` when budgeted or `evaluations` otherwise, `population_size`, `feasible_count`, optional `feasible_nondominated_count` when `feasible_count > 0`, `decision_diversity`, `placement_diversity`, optional `reference_direction_coverage`, `objective_distribution`, `incumbent_hard`, `incumbent_internal_total`, optional `best_hard`/`best_internal_total` fallback only if absent from `solutions`, `stagnant_generations`, `reviews_used`, `max_reviews`)
- `solutions`: Per-solution map keyed by `solution_id` containing `hard`, `total`, optional `unlocalized_penalties`, and deduplicated `classes` (keyed by `class_id` with `current_time`, `time_options`, `room_options`, `current_time_penalty`, and `current_room_id`/`current_room_penalty` when `room_options > 0`)
- `candidate_regions`: List of public region descriptors (`solution_id`, `region_id`, `boundary_id`, `class_ids`, `boundary_class_ids` truncated at 32 with `boundary_classes_omitted` when needed, `hard_contribution`, `soft_contribution`, `available_operators`, and `constraint_details` with non-overlapping `in_region_class_ids`/`outside_region_class_ids` in `violation_groups` plus conditional `params`, `penalty`, and `current_violations`)
- `recent_outcomes`: Last `5` compacted intervention summaries (`generation`, `operator_id`, `validation`, plus non-null/non-zero `region_id`, `boundary_id`, `class_ids`, `outcome`, `attempts`, `evaluations`, `source_hard`, `source_total`, `best_trial_hard`, `best_trial_total`, `best_hard_change`, `best_internal_total_change`)
- `allowed_actions`: `execution_budget` (`[1, current_operator_limit]`) and `review_interval` (`[min_review_interval, max_review_interval]`)

### Strict JSON Output and Action Validation

`DecisionClient.request` (shared across [../providers/gemini.py](../providers/gemini.py) and [../providers/cloudflare.py](../providers/cloudflare.py)) enforces structured JSON output (`json_schema` on Gemini and `cloudflare_llama`; `prompt_json` embedding `SCHEMAS["single_agent"]` in system prompt on `cloudflare_mistral`) matching `SCHEMAS["single_agent"]`:

```json
{
  "solution_id": "string or null",
  "region_id": "string or null",
  "boundary_id": "string or null",
  "operator_id": "no_op | move | swap | destroy_repair",
  "execution_budget": 12,
  "next_review_interval": 10
}
```

1. **Syntactic checks (`validate_shape`)**: Rejects duplicate JSON keys, non-finite constants, missing/extra keys (`additionalProperties: false`), wrong primitive types, or enum violations (`invalid_json`, `invalid_schema`).
2. **Semantic solver checks (`ControllerCallback._validate`)**:
   - `next_review_interval` must be an integer within `[--min-review-interval, --max-review-interval]` (default `[3, 20]`).
   - For `no_op`: `solution_id`, `region_id`, `boundary_id` must all be `null` and `execution_budget` must equal `0` (`valid_no_op`).
   - For active operators (`move`, `swap`, `destroy_repair`): `execution_budget` must be an integer in `[1, current_operator_limit]`, `(solution_id, region_id, boundary_id)` must match one entry in `candidate_regions`, and `operator_id` must appear in that region's `available_operators` (`valid_action`).
   - A well-formed response that fails semantic validation (or a non-API `model_refusal` / `incomplete_response`) falls back for that single review to `no_op` (`execution_budget=0`, `next_review_interval=--review-interval`) and records the validation reason (`missing_decision`, `invalid_review_interval`, `invalid_no_op`, `unknown_operator`, `invalid_operator_budget`, `operator_unavailable_in_region`, `unknown_region_or_boundary`) in `controller_trace.jsonl`.

### Action Execution and Scoring

Valid interventions run through `apply_operator` in [../NSGA-III/itc2019/operators.py](../NSGA-III/itc2019/operators.py), scoped strictly to `region.class_indices` of `region.solution_id`:

- `move`: Tries ranked time and room combinations for one class at a time inside the region.
- `swap`: Coordinates two classes inside the region across direct room/time exchanges, two-class eviction hand-offs (one class takes the partner's room or overlapping slot while the partner moves to its best free option), and coupled conflict-pair moves.
- `destroy_repair`: First runs OR-Tools CP-SAT `room_neighborhood_repair` over `region.class_indices` with the outside timetable fixed (bounded by deterministic CP-SAT time and any active run deadline, allowing soft-cost improvement when already feasible), then sequentially rebuilds up to 3 mutable region classes under the remaining `execution_budget`.
- **Admissibility and survival**: When `source_hard > 0`, a trial candidate is admissible if `trial_hard < source_hard` or `(trial_hard == source_hard and trial_total < source_total)`. Once feasible (`source_hard == 0`), it must preserve `trial_hard == 0` and improve at least one weighted objective. At most one best admissible candidate is rechecked with `score_state` (`+1` `operator_scoring_rechecks`) and passed to `TimetableSurvival` (ranking unique chromosomes first and breaking infeasible `CV` ties by lower total soft cost), updating `Incumbent` if improved.

---

## 2. Provider Settings and Fail-Closed Policy

All model profiles and transport defaults live centrally under [../settings/](../settings/) (`experiment.json` and `models/*.json`). Requests are served via `httpx` and `.env` quota pools in [../providers/gemini.py](../providers/gemini.py) (`https://generativelanguage.googleapis.com/v1beta`, Pacific daily quota reset) and [../providers/cloudflare.py](../providers/cloudflare.py) (`https://api.cloudflare.com/client/v4/accounts`, UTC daily quota reset on error `3036`):

| Setting | CLI flag / environment variable | Default / allowed values |
| :--- | :--- | :--- |
| **Model profiles (`settings/models/`)** | `--model` / `LLM_MODEL` / `GEMINI_MODEL` | `gemini_hydrogen` / `gemini-4-hydrogen` (default), `gemini_flash_lite` / `gemini-3.5-flash-lite`, `cloudflare_llama` / `@cf/meta/llama-3.3-70b-instruct-fp8-fast`, `cloudflare_mistral` / `@cf/mistralai/mistral-small-3.1-24b-instruct` |
| **Free Tier confirmation** | `--confirm-free-tier` / `GEMINI_FREE_TIER_CONFIRMED=1` / `CLOUDFLARE_FREE_TIER_CONFIRMED=1` | Required before any API client is created |
| **Credentials** | `.env` `GEMINI_API_KEY_001...` (or `GEMINI_API_KEY` env var); `.env` `CLOUDFLARE_ACCOUNT_ID_001...` + `CLOUDFLARE_API_TOKEN_001...` | Read locally (never pass on CLI or commit, redacted in logs) |
| **Thinking level** | `--thinking-level` / `LLM_THINKING_LEVEL` / `GEMINI_THINKING_LEVEL` | Gemini only: `low` (default), `medium`, `high` (`GEMINI_THINKING_LEVEL` applies only to Gemini; Cloudflare profiles reject explicit `--thinking-level`) |
| **Request timeout** | `--api-timeout-seconds` | `60.0` seconds (from `settings/experiment.json`) |
| **Retry & cooldown policy** | `providers/credential_pool.py` | Transient retry + `60s` slot cooldown after `3` consecutive transient errors; daily quota rotation across `.env` slots (`tmp/gemini/` vs `tmp/cloudflare/`) |
| **Pacing & payload caps** | `--api-min-interval-seconds`, `--api-max-input-characters`, `--max-output-tokens`, `--max-api-calls` | Defaults: `10.0s`, `60000` chars, `4096` tokens including thinking, `100` calls direct (`12` via `run_experiment.py`) |

Note on centralized settings: [settings/experiment.json](../settings/experiment.json) selects `gemini_hydrogen` by default and shares all solver/repair/controller/LLM caps across both direct CLI runs and `run_experiment.py`.

**Fail-closed API and quota policy**:

- Non-retryable billing (`402` or Cloudflare error `5035`) and configuration/auth errors (`400`/`401`/`403`/`404` except Llama per-review schema mismatch `invalid_schema`) are **never retried**, immediately set `client.blocked_reason`, and have **no silent model or provider fallback**.
- When all credential slots hit daily quota and wall-clock deadline expires without waiting out reset, `request` returns `api_deadline_skipped`. Any fatal `api_*` failure during a run sets `api_disabled = True`, forces solver termination (`force_termination = True`), halts any further instances in a multi-instance CLI invocation, saves artifacts with `completion_status: "incomplete_api_failure"` and `comparison_eligible: false`, and exits with **exit code `3`**.

---

## 3. Setup and Running

Run all commands from the `Experiment/` repository root:

```cmd
python -m pip install -r "Single-Agent Controller/requirements.txt"
python tools/gemini_preflight.py
python tools/gemini_preflight.py --model cloudflare_llama
```

Verify key presence, Free Tier confirmation, and one live `no_op` test call before running optimization:

```cmd
python tools/gemini_preflight.py --live --confirm-free-tier
python tools/gemini_preflight.py --live --confirm-free-tier --model cloudflare_llama
```

Preview or run the experiment profile via `run_experiment.py`:

```powershell
python run_experiment.py --pipeline single --instance muni-fsps-spr17 --seed 17 --dry-run
python run_experiment.py --pipeline single --instance muni-fsps-spr17 --seed 17 --confirm-free-tier
```

Override the model (or thinking level on Gemini profiles) if needed:

```powershell
python run_experiment.py --pipeline single --instance pu-cs-fal07 --seed 17 --model gemini_flash_lite --thinking-level medium --confirm-free-tier
python run_experiment.py --pipeline single --instance pu-cs-fal07 --seed 17 --model cloudflare_mistral --confirm-free-tier
```

### Direct CLI and Live Logging Options

Run `python "Single-Agent Controller/main.py" --help` to see all inherited baseline flags plus controller and LLM flags (`--review-interval`, `--warmup-generations`, `--stagnation-window`, `--min-review-interval`, `--max-review-interval`, `--diversity-threshold`, `--diversity-recovery-threshold`, `--max-regions`, `--region-size`, `--max-operator-evaluations`, `--max-reviews`, `--provider`, `--model`, `--thinking-level`, `--confirm-free-tier`, `--api-min-interval-seconds`, `--api-max-input-characters`, `--api-timeout-seconds`, `--max-output-tokens`, `--max-api-calls`).

Live progress and logging flags:

- `--progress-interval N`: Emit generation progress every `N` generations (default `1`, `0` logs `[start]`, repairs, reviews, and `[done]` only).
- `--progress-log PATH`: Append live progress lines to an additional log file.
- `--quiet`: Suppress `stderr` progress output while retaining `<instance>/run.log` and optional `--progress-log`.

Example direct invocation:

```powershell
python "Single-Agent Controller/main.py" --instance wbg-fal10 --generations 50 --evaluations 1200 --thinking-level low --confirm-free-tier --progress-interval 5 --progress-log single-wbg.log
```

---

## 4. Output Artifacts

Direct runs save to `Single-Agent Controller/results_itc2019/<run_id>/`, while `run_experiment.py --output DIR` saves to `DIR/single/<run_id>/`.

- `run_config.json`: Run ID, full solver/repair/controller/LLM metadata (no API keys), instance and source SHA-256 hashes, and `completion_status` (`"completed"` or `"incomplete_api_failure"`).
- `summary.json`: Array of per-instance summaries.
- `<instance>/result.json`: Final score, feasibility, evaluation breakdown, `score_engine` stats, `stage_timing`, `monitor` metrics, and `controller` metrics (`api_calls`, token counts including `thinking_tokens`, `api_latency_seconds`, `api_pacing_wait_seconds`, `reviews`, `no_op`, `model_no_op`, `fallback_no_op`, `interventions`, `selected_interventions`, `completion_status`, `comparison_eligible`).
- `<instance>/controller_trace.jsonl`: One JSON line per triggered review recording `generation`, `trigger`, full `state` (`StatePacket`), `roles` (single LLM telemetry/token/status record), `decision_status`, `validation`, `action`, `outcome`, `population_after`, and `incumbent_after`.
- `<instance>/anytime_trace.jsonl`: Generation-level search trajectory and incumbent quality.
- `<instance>/choices.npz`: Compressed `time_indices` and `room_indices` arrays for the best observed solution.
- `<instance>/run.log`: Human-readable live progress including `[review #...]` decision and outcome lines.
- `<instance>/solution.xml` and `<instance>/expected.json`: Exported when `score.hard == 0` and internal sectioning passes (`official_status` remains `"not_run"` until checked separately via [../tools/validate_itc2019.py](../tools/validate_itc2019.py)).
