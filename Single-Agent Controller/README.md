# Single-Agent Controller

The Single-Agent Controller integrates one bounded Google Gemini Developer API decision per triggered review into the shared [NSGA-III baseline](../NSGA-III/README.md) solver. At each review, a single structured JSON call decides **WHERE** to intervene (selecting a candidate region), **HOW** to intervene (selecting an operator and evaluation budget), and **WHEN** to schedule the next review.

See the [repository README](../README.md) for cross-pipeline setup, profile comparisons, and multi-worker execution.

---

## 1. Architecture and Control Flow

`SingleAgentController` in [controller.py](controller.py) subclasses `ControllerCallback` in [../NSGA-III/itc2019/controller.py](../NSGA-III/itc2019/controller.py). The solver continues to own population evolution, routine offspring repair, periodic CP-SAT and plateau repairs, incumbent archiving, and all chromosome mutations.

```text
Generation end -> BaselineMonitor repairs & incumbent sync
  -> Evaluate review triggers (deadline | stagnation | low_diversity)
  -> Check gates (budget >= 4, API active, time > 0.1s, reviews < max_reviews)
  -> RegionBuilder constructs candidate regions + StatePacket
  -> One Gemini API call (role="single_agent", strict JSON schema)
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

- `schema_version`: `1`
- `instance`: Instance name
- `global`: Population and anytime metrics (`generation`, `elapsed_seconds`, `remaining_seconds`, `evaluations`, `evaluation_budget`, `remaining_evaluations`, `population_size`, `feasible_count`, `best_hard`, `best_internal_total`, `best_feasible_internal_total`, `incumbent_hard`, `incumbent_internal_total`, `progress_mode`, `stagnant_generations`, `feasible_nondominated_count`, `decision_diversity`, `placement_diversity`, `reference_direction_coverage`, `objective_distribution`)
- `recent_best_quality`: Last `stagnation_window` `(hard, internal_total)` tuples
- `candidate_regions`: List of public region descriptors (`solution_id`, `region_id`, `boundary_id`, `class_ids`, `boundary_class_ids` truncated at 32 with `boundary_class_count` and `boundary_truncated`, `constraint_ids`, `hard_contribution`, `soft_contribution`, `unlocalized_penalties`, `class_summaries`, `movable_class_count`, `compatible_swap_pairs`, `available_operators`)
- `recent_outcomes`: Last `5` intervention summaries (`generation`, `region_id`, `operator_id`, `validation`, `outcome`, `attempts`, `evaluations`, `elapsed_seconds`, `best_hard_change`, `best_internal_total_change`)
- `allowed_actions`: `operators` (`["move", "swap", "destroy_repair", "no_op"]`), `execution_budget` (`[1, current_operator_limit]`), and `review_interval` (`[min_review_interval, max_review_interval]`)

### Strict JSON Output and Action Validation

`GeminiDecisionClient.request` in [../NSGA-III/itc2019/llm.py](../NSGA-III/itc2019/llm.py) requests structured JSON (`response_mime_type="application/json"`, `response_json_schema=SCHEMAS["single_agent"]`, `candidate_count=1`, `automatic_continuation=False`, `automatic_function_calling` disabled, `include_thoughts=False`):

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

- `move`: Re-samples one movable class's time or room gene inside the region.
- `swap`: Exchanges compatible `(time, room)` assignments between two classes inside the region.
- `destroy_repair`: First runs OR-Tools CP-SAT `room_neighborhood_repair` over `region.class_indices` with the outside timetable fixed (up to `3.0s` or remaining deadline, allowing soft-cost improvement when already feasible), then sequentially rebuilds up to 3 mutable region classes under the remaining `execution_budget`.
- **Admissibility and survival**: When `source_hard > 0`, a trial candidate is admissible only if `trial_hard < source_hard`. Once feasible (`source_hard == 0`), it must preserve `trial_hard == 0` and reduce `trial_total < source_total`. At most one best admissible candidate is rechecked with `score_state` (`+1` `operator_scoring_rechecks`) and passed to NSGA-III `survival.do`, updating `Incumbent` if improved.

---

## 2. Gemini Free Tier Settings and Fail-Closed Policy

All API transport rules live in [../NSGA-III/itc2019/llm.py](../NSGA-III/itc2019/llm.py) using the `google-genai` SDK against `https://generativelanguage.googleapis.com` (`v1beta`, `vertexai=False`):

| Setting | CLI flag / environment variable | Default / allowed values |
| :--- | :--- | :--- |
| **Allowed models (`FREE_TIER_MODELS`)** | `--model` / `GEMINI_MODEL` | `gemini-4-hydrogen` (default), `gemini-3.5-flash-lite` |
| **Free Tier confirmation** | `--confirm-free-tier` / `GEMINI_FREE_TIER_CONFIRMED=1` | Required before any API client is created |
| **API key** | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Read from environment only (never pass as a CLI argument, redacted in logs, conflicting dual keys rejected) |
| **Thinking level** | `--thinking-level` / `GEMINI_THINKING_LEVEL` | `minimal`, `low` (default), `medium`, `high` |
| **Request timeout** | `--api-timeout-seconds` | `60.0` seconds (`DEFAULT_TIMEOUT_SECONDS = 60.0`) |
| **SDK retries (`RETRY_ATTEMPTS`)** | Internal `HttpRetryOptions` | `attempts=3` on transient HTTP codes `(408, 500, 502, 503, 504)` only |
| **Pacing & payload caps** | `--api-min-interval-seconds`, `--api-max-input-characters`, `--max-output-tokens`, `--max-api-calls` | Defaults: `10.0s`, `60000` chars, `1024` tokens, `100` calls direct (`12` via `run_pilot.py`) |

Note on profiles: [profiles/pilot.json](../profiles/pilot.json) and [profiles/phase1.json](../profiles/phase1.json) configure `gemini-3.5-flash-lite`, while [profiles/phase2.json](../profiles/phase2.json) configures `gemini-4-hydrogen`. All three set `thinking_level: "low"` and `max_reviews: 12`.

**Fail-closed API and quota policy**:

- HTTP `429` (`api_quota_exhausted`), `402` (`api_billing_required`), and `400`/`401`/`403`/`404` (`api_configuration_error`) are **never retried**, immediately set `client.blocked_reason`, and have **no silent model or provider fallback** (no OpenAI or paid fallback).
- Any `api_*` failure during a run (except `api_deadline_skipped`) sets `api_disabled = True`, forces solver termination (`force_termination = True`), halts any further instances in a multi-instance CLI invocation, saves artifacts with `completion_status: "incomplete_api_failure"` and `comparison_eligible: false`, and exits with **exit code `3`**.

---

## 3. Setup and Running

Run all commands from the `Experiment/` repository root:

```cmd
python -m pip install -r "Single-Agent Controller/requirements.txt"
python tools/gemini_preflight.py
```

Verify key presence, Free Tier confirmation, and one live `no_op` test call before running optimization:

```cmd
python tools/gemini_preflight.py --live --confirm-free-tier
```

Preview or run the pilot profile via `run_pilot.py`:

```powershell
python run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --dry-run
python run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --confirm-free-tier
```

Override the model or thinking level if needed:

```powershell
python run_pilot.py --profile profiles/phase1.json --pipeline single --instance pu-cs-fal07 --seed 17 --model gemini-4-hydrogen --thinking-level medium --confirm-free-tier
```

### Direct CLI and Live Logging Options

Run `python "Single-Agent Controller/main.py" --help` to see all inherited baseline flags plus controller and Gemini flags (`--review-interval`, `--warmup-generations`, `--stagnation-window`, `--min-review-interval`, `--max-review-interval`, `--diversity-threshold`, `--diversity-recovery-threshold`, `--max-regions`, `--region-size`, `--max-operator-evaluations`, `--max-reviews`, `--model`, `--thinking-level`, `--confirm-free-tier`, `--api-min-interval-seconds`, `--api-max-input-characters`, `--api-timeout-seconds`, `--max-output-tokens`, `--max-api-calls`).

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

Direct runs save to `Single-Agent Controller/results_itc2019/<run_id>/`, while `run_pilot.py --output DIR` saves to `DIR/single/<run_id>/`.

- `run_config.json`: Run ID, full solver/repair/controller/LLM metadata (no API keys), instance and source SHA-256 hashes, and `completion_status` (`"completed"` or `"incomplete_api_failure"`).
- `summary.json`: Array of per-instance summaries.
- `<instance>/result.json`: Final score, feasibility, evaluation breakdown, `score_engine` stats, `stage_timing`, `monitor` metrics, and `controller` metrics (`api_calls`, token counts including `thinking_tokens`, `api_latency_seconds`, `api_pacing_wait_seconds`, `reviews`, `no_op`, `model_no_op`, `fallback_no_op`, `interventions`, `selected_interventions`, `completion_status`, `comparison_eligible`).
- `<instance>/controller_trace.jsonl`: One JSON line per triggered review recording `generation`, `trigger`, full `state` (`StatePacket`), `roles` (single Gemini telemetry/token/status record), `decision_status`, `validation`, `action`, `outcome`, `population_after`, and `incumbent_after`.
- `<instance>/anytime_trace.jsonl`: Generation-level search trajectory and incumbent quality.
- `<instance>/choices.npz`: Compressed `time_indices` and `room_indices` arrays for the best observed solution.
- `<instance>/run.log`: Human-readable live progress including `[review #...]` decision and outcome lines.
- `<instance>/solution.xml` and `<instance>/expected.json`: Exported when `score.hard == 0` and internal sectioning passes (`official_status` remains `"not_run"` until checked separately via [../tools/validate_itc2019.py](../tools/validate_itc2019.py)).
