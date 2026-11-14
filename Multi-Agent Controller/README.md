# Inspector-Planner Controller (Multi-Agent)

The Multi-Agent Controller decomposes each event-triggered review into a sequential two-stage **Inspector -> Planner** Gemini pipeline integrated with the shared [NSGA-III baseline](../NSGA-III/README.md) solver. **Inspector** selects **WHERE** to intervene (`candidate_regions` target and relevant constraints), and only after target validation and resource checks does **Planner** decide **HOW** (operator and evaluation budget) and **WHEN** (next review interval).

See the [repository README](../README.md) for cross-pipeline setup, profile comparisons, and multi-worker execution.

---

## 1. Sequential Inspector -> Planner Architecture

`MultiAgentController` in [controller.py](controller.py) subclasses `ControllerCallback` in [../NSGA-III/itc2019/controller.py](../NSGA-III/itc2019/controller.py). As in the other pipelines, the shared solver owns population evolution, offspring repair, periodic CP-SAT and plateau repairs, incumbent archiving, and chromosome updates.

```text
Generation end -> BaselineMonitor repairs & incumbent sync
  -> Evaluate review triggers (deadline | stagnation | low_diversity)
  -> Check gates (budget >= 4, API active, time > 0.1s, reviews < max_reviews)
  -> RegionBuilder builds candidate regions + StatePacket (Schema v1)
  -> Stage 1: Gemini Inspector (role="inspector", input=StatePacket)
     |-- api_* error -> api_disabled, force termination, exit 3
     |-- null IDs -> early stop ("inspector_no_target"), fallback no_op (0 Planner calls)
     |-- invalid target/constraints -> early stop, fallback no_op (0 Planner calls)
     |-- deadline <= 0.1s or call budget reached -> early stop, fallback no_op (0 Planner calls)
     '-- Stage 2: Gemini Planner (role="planner", input={"state": StatePacket, "diagnosis": Inspector JSON})
         |-- api_* error -> api_disabled, force termination, exit 3
         '-- bind Inspector target + Planner operator/budget/interval -> _validate & apply_operator
```

### Event-Triggered Review Conditions and StatePacket (Schema v1)

Review triggers and `StatePacket` construction are identical to the Single-Agent controller for fair comparison:

- **Warmup and spacing**: No reviews before `--warmup-generations` (default `5`) or within `--min-review-interval` (default `3`) generations of `last_review`.
- **Trigger reasons**: `deadline` (`generation >= next_review`), `stagnation` (`stagnant_generations >= --stagnation-window`, default `5`, resetting on incumbent progress), and `low_diversity` (`placement_diversity <= --diversity-threshold`, default `0.20`, re-arming above `--diversity-recovery-threshold`, default `0.30`).
- **Pre-review gates**: Requires `remaining_evaluations >= 4` (else increments `budget_skipped_reviews`), `not api_disabled`, `api_calls < --max-api-calls`, `remaining_seconds > 0.1`, and `reviews < --max-reviews` (default `12`).
- **`StatePacket` v1 contents**: Built by `RegionBuilder.build` in [../NSGA-III/itc2019/regions.py](../NSGA-III/itc2019/regions.py) across up to 2 diverse representative solutions and up to `--max-regions` (`6`) connected subgraphs of up to `--region-size` (`5`) classes, containing `schema_version` (`1`), `instance`, `global` metrics, `recent_best_quality`, `candidate_regions` (`solution_id`, `region_id`, `boundary_id`, `class_ids`, `boundary_class_ids` truncated at 32, `constraint_ids`, `hard_contribution`, `soft_contribution`, `unlocalized_penalties`, `class_summaries`, `movable_class_count`, `compatible_swap_pairs`, `available_operators`), `recent_outcomes`, and `allowed_actions`.

### Stage 1: Inspector Schema, Validation, and Early Termination

`client.request("inspector", packet, self.remaining())` enforces `SCHEMAS["inspector"]` (`additionalProperties: false`):

```json
{
  "solution_id": "string or null",
  "region_id": "string or null",
  "boundary_id": "string or null",
  "relevant_constraint_ids": ["string"]
}
```

Before invoking Planner, `MultiAgentController.decide` validates the Inspector output and checks remaining wall-clock and API budgets. Any of the following conditions ends the review early **without spending a Planner API call**:

1. **Inspector API or response failure (`inspector_<status>`)**: If `raw is None` (such as an `api_*` error, `model_refusal`, `incomplete_response`, `invalid_json`, or `invalid_schema`), Planner is skipped. Any `api_*` status (except `api_deadline_skipped`) triggers fail-closed termination (exit code `3`).
2. **Intentional abstention (`inspector_no_target`)**: When `(solution_id, region_id, boundary_id) == (None, None, None)`, the review returns `_fallback()` (`no_op`, `execution_budget=0`, `next_review_interval=--review-interval`) and is counted as an intentional `model_no_op`.
3. **Invalid region target (`invalid_inspector_target`)**: `(solution_id, region_id, boundary_id)` must exactly match one entry in `packet["candidate_regions"]`. Otherwise the review ends with `no_op` (`fallback_no_op`).
4. **Invalid constraint diagnosis (`invalid_inspector_constraints`)**: Every item in `relevant_constraint_ids` must be a string present in the matched region's `constraint_ids`. Otherwise the review ends with `no_op` (`fallback_no_op`).
5. **Pre-Planner time guard (`deadline_before_planner`)**: If `remaining_seconds <= 0.1`, Planner is skipped and the review returns `_fallback()` (`no_op`).
6. **Pre-Planner API call budget guard (`call_budget_before_planner`)**: If `self.api_calls + len(records) >= self.settings.max_api_calls`, Planner is skipped and the review returns `_fallback()` (`no_op`).

### Stage 2: Planner Payload, Output Schema, and Execution

If Inspector validation and pre-Planner guards pass, Planner receives `{"state": packet, "diagnosis": raw}` via `client.request("planner", planner_payload, self.remaining())`, enforcing `SCHEMAS["planner"]` (`additionalProperties: false`):

```json
{
  "operator_id": "no_op | move | swap | destroy_repair",
  "execution_budget": 12,
  "next_review_interval": 10
}
```

- **Target binding**: Planner cannot alter Inspector's target. If `operator_id == "no_op"`, target IDs are cleared to `(None, None, None)`. Otherwise the validated Inspector `(solution_id, region_id, boundary_id)` tuple is bound directly into `ActionPlan`.
- **Planner failures**: If Planner returns `None`, status is `planner_<status>`. If keys/types cannot construct `ActionPlan`, status is `malformed_planner_action`.
- **Shared validation and execution**: `ControllerCallback._validate` checks `next_review_interval` in `[--min-review-interval, --max-review-interval]` (default `[3, 20]`), `execution_budget` (`0` for `no_op`, or `1..current_operator_limit` capped by `min(--max-operator-evaluations, remaining_evaluations - 1)`), and verifies `operator_id` is in the matched region's `available_operators`.
- **Operator execution and admissibility**: Valid actions run `apply_operator` (`move`, `swap`, or `destroy_repair` combining region-scoped OR-Tools CP-SAT `room_neighborhood_repair` with up to 3 sequential class rebuilds). Before feasibility (`source_hard > 0`), only candidates with `trial_hard < source_hard` are admissible. After feasibility (`source_hard == 0`), candidates must keep `trial_hard == 0` and achieve `trial_total < source_total`. At most one best candidate is rechecked via `score_state` and passed to NSGA-III survival.

---

## 2. Gemini Free Tier Settings and Fail-Closed Policy

Both Inspector and Planner share the same `GeminiDecisionClient` instance in [../NSGA-III/itc2019/llm.py](../NSGA-III/itc2019/llm.py) (`https://generativelanguage.googleapis.com`, `v1beta`, `vertexai=False`, `candidate_count=1`, `automatic_continuation=False`, `automatic_function_calling` disabled, `include_thoughts=False`):

| Setting | CLI flag / environment variable | Default / allowed values |
| :--- | :--- | :--- |
| **Allowed models (`FREE_TIER_MODELS`)** | `--model` / `GEMINI_MODEL` | `gemini-3.8-flash` (default), `gemini-3.5-flash-lite`, `gemini-4-hydrogen` |
| **Free Tier confirmation** | `--confirm-free-tier` / `GEMINI_FREE_TIER_CONFIRMED=1` | Required before instantiating the API client |
| **API key** | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Environment only (never CLI args, redacted in errors, mismatched dual keys rejected) |
| **Thinking level** | `--thinking-level` / `GEMINI_THINKING_LEVEL` | `minimal`, `low` (default), `medium`, `high` |
| **Request timeout** | `--api-timeout-seconds` | `60.0` seconds (`DEFAULT_TIMEOUT_SECONDS = 60.0`) |
| **SDK retries (`RETRY_ATTEMPTS`)** | Internal `HttpRetryOptions` | `attempts=3` on transient HTTP codes `(408, 500, 502, 503, 504)` only |
| **Pacing & payload caps** | `--api-min-interval-seconds`, `--api-max-input-characters`, `--max-output-tokens`, `--max-api-calls` | Defaults: `10.0s`, `60000` chars, `1024` tokens, `100` calls direct (`24` via `run_pilot.py` for `max_reviews=12`) |

Note on pacing and call budgets: Because each full review invokes Inspector and then Planner sequentially, `--api-min-interval-seconds` (default `10.0s`) paces the gap between Inspector and Planner as well as between consecutive reviews. In `run_pilot.py`, `--max-api-calls` is automatically set to `max_reviews * 2` (`24` when `max_reviews = 12`). Profiles [profiles/pilot.json](../profiles/pilot.json) (`gemini-3.8-flash`) and [profiles/expanded_pilot.json](../profiles/expanded_pilot.json) (`gemini-3.5-flash-lite`) both use `thinking_level: "low"`.

**Fail-closed API and quota policy**:

- HTTP `429` (`api_quota_exhausted`), `402` (`api_billing_required`), and `400`/`401`/`403`/`404` (`api_configuration_error`) are **never retried**, immediately set `client.blocked_reason`, and have **no silent model or provider fallback**.
- Any `api_*` failure on either Inspector or Planner (except `api_deadline_skipped`) sets `api_disabled = True`, forces solver termination (`force_termination = True`), halts remaining instances, saves partial artifacts with `completion_status: "incomplete_api_failure"` and `comparison_eligible: false`, and exits with **exit code `3`**.

---

## 3. Setup and Running

Run all commands from the `Experiment/` repository root:

```cmd
python -m pip install -r "Multi-Agent Controller/requirements.txt"
python tools/gemini_preflight.py
```

Verify Free Tier readiness with one live synthetic `no_op` check before running optimization:

```cmd
python tools/gemini_preflight.py --live --confirm-free-tier
```

Preview or run the pilot profile via `run_pilot.py`:

```powershell
python run_pilot.py --pipeline multi --instance muni-fsps-spr17 --seed 17 --dry-run
python run_pilot.py --pipeline multi --instance muni-fsps-spr17 --seed 17 --confirm-free-tier
```

Run with an alternative profile, model, or thinking level:

```powershell
python run_pilot.py --profile profiles/expanded_pilot.json --pipeline multi --instance pu-cs-fal07 --seed 17 --model gemini-3.8-flash --thinking-level medium --confirm-free-tier
```

### Direct CLI and Live Logging Options

Run `python "Multi-Agent Controller/main.py" --help` to list all inherited baseline, controller, and Gemini options (`--review-interval`, `--warmup-generations`, `--stagnation-window`, `--min-review-interval`, `--max-review-interval`, `--diversity-threshold`, `--diversity-recovery-threshold`, `--max-regions`, `--region-size`, `--max-operator-evaluations`, `--max-reviews`, `--model`, `--thinking-level`, `--confirm-free-tier`, `--api-min-interval-seconds`, `--api-max-input-characters`, `--api-timeout-seconds`, `--max-output-tokens`, `--max-api-calls`).

Live progress and logging flags:

- `--progress-interval N`: Emit generation progress every `N` generations (default `1`, `0` logs `[start]`, repairs, reviews, and `[done]` only).
- `--progress-log PATH`: Append live progress lines to an additional log file.
- `--quiet`: Suppress `stderr` progress output while keeping `<instance>/run.log` and optional `--progress-log`.

Example direct invocation:

```powershell
python "Multi-Agent Controller/main.py" --instance wbg-fal10 --generations 50 --evaluations 1200 --thinking-level low --confirm-free-tier --progress-interval 5 --progress-log multi-wbg.log
```

---

## 4. Output Artifacts

Direct runs save to `Multi-Agent Controller/results_itc2019/<run_id>/`, while `run_pilot.py --output DIR` saves to `DIR/multi/<run_id>/`.

- `run_config.json`: Run ID, full solver/repair/controller/LLM metadata (no API keys), source and instance SHA-256 hashes, and `completion_status` (`"completed"` or `"incomplete_api_failure"`).
- `summary.json`: Array of per-instance summaries.
- `<instance>/result.json`: Final score, feasibility, evaluation breakdown, `score_engine` stats, `stage_timing`, `monitor` metrics, and `controller` metrics (`api_calls`, cumulative token counts including `thinking_tokens`, `api_latency_seconds`, `api_pacing_wait_seconds`, `reviews`, `no_op`, `model_no_op`, `fallback_no_op`, `interventions`, `selected_interventions`, `completion_status`, `comparison_eligible`).
- `<instance>/controller_trace.jsonl`: One JSON line per review containing `generation`, `trigger`, `state` (`StatePacket`), `roles` (one `inspector` record on early stop, or sequential `inspector` and `planner` records), `decision_status`, `validation`, `action`, `outcome`, `population_after`, and `incumbent_after`.
- `<instance>/anytime_trace.jsonl`: Generation-level search trajectory and incumbent quality.
- `<instance>/choices.npz`: Compressed `time_indices` and `room_indices` arrays for the best observed solution.
- `<instance>/run.log`: Human-readable live progress including `[review #...]` lines showing both `inspector[...]` and `planner[...]` summaries.
- `<instance>/solution.xml` and `<instance>/expected.json`: Exported when `score.hard == 0` and internal sectioning passes (`official_status` stays `"not_run"` until verified separately via [../tools/validate_itc2019.py](../tools/validate_itc2019.py)).
