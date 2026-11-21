# Inspector-Planner Controller (Multi-Agent)

The Multi-Agent Controller decomposes each event-triggered review into a sequential two-stage **Inspector -> Planner** LLM pipeline integrated with the shared [NSGA-III baseline](../NSGA-III/README.md) solver across Google Gemini Developer API and Cloudflare Workers AI. **Inspector** selects **WHERE** to intervene (`candidate_regions` target and relevant constraints), and only after target validation and resource checks does **Planner** decide **HOW** (operator and evaluation budget) and **WHEN** (next review interval).

See the [repository README](../README.md) for cross-pipeline setup, profile comparisons, and multi-worker execution.

---

## 1. Sequential Inspector -> Planner Architecture

`MultiAgentController` in [controller.py](controller.py) subclasses `ControllerCallback` in [../NSGA-III/itc2019/controller.py](../NSGA-III/itc2019/controller.py). As in the other pipelines, the shared solver owns population evolution, offspring repair, periodic CP-SAT and plateau repairs, incumbent archiving, and chromosome updates.

```text
Generation end -> BaselineMonitor repairs & incumbent sync
  -> Evaluate review triggers (deadline | stagnation | low_diversity)
  -> Check gates (budget >= 4, API active, time > 0.1s, reviews < max_reviews)
  -> RegionBuilder builds candidate regions + StatePacket
  -> Stage 1: LLM Inspector (role="inspector", input=StatePacket)
     |-- api_* error -> api_disabled, force termination, exit 3
     |-- null IDs -> early stop ("inspector_no_target"), fallback no_op (0 Planner calls)
     |-- invalid target/constraints -> early stop, fallback no_op (0 Planner calls)
     |-- deadline <= 0.1s or call budget reached -> early stop, fallback no_op (0 Planner calls)
     '-- Stage 2: LLM Planner (role="planner", input={"state": StatePacket, "diagnosis": Inspector JSON})
         |-- api_* error -> api_disabled, force termination, exit 3
         '-- bind Inspector target + Planner operator/budget/interval -> _validate & apply_operator
```

### Event-Triggered Review Conditions and StatePacket

Review triggers and `StatePacket` construction are identical to the Single-Agent controller for fair comparison:

- **Warmup and spacing**: No reviews before `--warmup-generations` (default `5`) or within `--min-review-interval` (default `3`) generations of `last_review`.
- **Trigger reasons**: `deadline` (`generation >= next_review`), `stagnation` (`stagnant_generations >= --stagnation-window`, default `5`, resetting on incumbent progress), and `low_diversity` (`placement_diversity <= --diversity-threshold`, default `0.20`, re-arming above `--diversity-recovery-threshold`, default `0.30`).
- **Pre-review gates**: Requires `remaining_evaluations >= 4` (else increments `budget_skipped_reviews`), `not api_disabled`, `api_calls < --max-api-calls`, `remaining_seconds > 0.1`, and `reviews < --max-reviews` (default `12`).
- **`StatePacket` contents**: Built by `RegionBuilder.build` in [../NSGA-III/itc2019/regions.py](../NSGA-III/itc2019/regions.py) across up to 2 diverse representative solutions and up to `--max-regions` (`6`) connected subgraphs of up to `--region-size` (`5`) classes, containing `instance`, non-redundant `global` metrics, deduplicated `solutions` (`hard`, `total`, optional `unlocalized_penalties`, `classes`), `candidate_regions` (`solution_id`, `region_id`, `boundary_id`, `class_ids`, `boundary_class_ids` truncated at 32, `hard_contribution`, `soft_contribution`, `available_operators`, `constraint_details`), `recent_outcomes`, and `allowed_actions`.

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
4. **Invalid constraint diagnosis (`invalid_inspector_constraints`)**: Every item in `relevant_constraint_ids` must be a string present in the matched region's `constraint_details` IDs. Otherwise the review ends with `no_op` (`fallback_no_op`).
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
- **Operator execution and admissibility**: Valid actions run `apply_operator`. `move` ranks time and room combinations for one class at a time. `swap` coordinates two classes inside the region across direct room/time exchanges, two-class eviction hand-offs, and coupled conflict-pair moves. `destroy_repair` combines joint CP-SAT repair with up to 3 sequential class rebuilds. Before feasibility (`source_hard > 0`), admissible trials reduce hard violations or reduce total soft cost at the same hard count. After feasibility, admissible trials stay feasible and improve at least one weighted objective. At most one best candidate is rechecked via `score_state` and passed to `TimetableSurvival` (ranking unique chromosomes first and breaking infeasible `CV` ties by lower total soft cost).

---

## 2. Provider Settings and Fail-Closed Policy

Both Inspector and Planner share one client created from [../settings/](../settings/) (`experiment.json` + `models/*.json`) via [../providers/gemini.py](../providers/gemini.py) (`https://generativelanguage.googleapis.com/v1beta`, Pacific daily reset) or [../providers/cloudflare.py](../providers/cloudflare.py) (`https://api.cloudflare.com/client/v4/accounts`, UTC daily reset on error `3036`), using `httpx` and `.env` quota tracking (`json_schema` on Gemini and `cloudflare_llama`; `prompt_json` on `cloudflare_mistral`):

| Setting | CLI flag / environment variable | Default / allowed values |
| :--- | :--- | :--- |
| **Model profiles (`settings/models/`)** | `--model` / `LLM_MODEL` / `GEMINI_MODEL` | `gemini_hydrogen` / `gemini-4-hydrogen` (default), `gemini_flash_lite` / `gemini-3.5-flash-lite`, `cloudflare_llama` / `@cf/meta/llama-3.3-70b-instruct-fp8-fast`, `cloudflare_mistral` / `@cf/mistralai/mistral-small-3.1-24b-instruct` |
| **Free Tier confirmation** | `--confirm-free-tier` / `GEMINI_FREE_TIER_CONFIRMED=1` / `CLOUDFLARE_FREE_TIER_CONFIRMED=1` | Required before instantiating the API client |
| **Credentials** | `.env` `GEMINI_API_KEY_001...` (or `GEMINI_API_KEY` env var); `.env` `CLOUDFLARE_ACCOUNT_ID_001...` + `CLOUDFLARE_API_TOKEN_001...` | Read locally (never CLI args, redacted in errors) |
| **Thinking level** | `--thinking-level` / `LLM_THINKING_LEVEL` / `GEMINI_THINKING_LEVEL` | Gemini only: `low` (default), `medium`, `high` (`GEMINI_THINKING_LEVEL` applies only to Gemini; Cloudflare profiles reject explicit `--thinking-level`) |
| **Request timeout** | `--api-timeout-seconds` | `60.0` seconds (from `settings/experiment.json`) |
| **Retry & cooldown policy** | `providers/credential_pool.py` | Transient retry + `60s` slot cooldown after `3` consecutive transient errors; daily quota rotation across `.env` slots (`tmp/gemini/` vs `tmp/cloudflare/`) |
| **Pacing & payload caps** | `--api-min-interval-seconds`, `--api-max-input-characters`, `--max-output-tokens`, `--max-api-calls` | Defaults: `10.0s`, `60000` chars, `4096` tokens including thinking, `100` calls direct (`24` via `run_experiment.py` for `max_reviews=12`) |

Note on pacing and centralized settings: Because each full review invokes Inspector and then Planner sequentially, `--api-min-interval-seconds` (default `10.0s`) paces the gap between Inspector and Planner as well as between consecutive reviews. In `run_experiment.py`, `--max-api-calls` is automatically set to `max_reviews * 2` (`24` when `max_reviews = 12`). [settings/experiment.json](../settings/experiment.json) configures `gemini_hydrogen` with `thinking_level: "low"` by default.

**Fail-closed API and quota policy**:

- Non-retryable billing (`402` or Cloudflare error `5035`) and configuration/auth errors (`400`/`401`/`403`/`404` except Llama per-review schema mismatch `invalid_schema`) are **never retried**, immediately set `client.blocked_reason`, and have **no silent model or provider fallback**.
- When all slots hit daily quota and wall-clock deadline expires without waiting out reset, `request` returns `api_deadline_skipped`. Any fatal `api_*` failure on either Inspector or Planner sets `api_disabled = True`, forces solver termination (`force_termination = True`), halts remaining instances, saves partial artifacts with `completion_status: "incomplete_api_failure"` and `comparison_eligible: false`, and exits with **exit code `3`**.

---

## 3. Setup and Running

Run all commands from the `Experiment/` repository root:

```cmd
python -m pip install -r "Multi-Agent Controller/requirements.txt"
python tools/gemini_preflight.py
python tools/gemini_preflight.py --model cloudflare_llama
```

Verify Free Tier readiness with one live synthetic `no_op` check before running optimization:

```cmd
python tools/gemini_preflight.py --live --confirm-free-tier
python tools/gemini_preflight.py --live --confirm-free-tier --model cloudflare_llama
```

Preview or run the experiment profile via `run_experiment.py`:

```powershell
python run_experiment.py --pipeline multi --instance muni-fsps-spr17 --seed 17 --dry-run
python run_experiment.py --pipeline multi --instance muni-fsps-spr17 --seed 17 --confirm-free-tier
```

Run with an alternative model (or thinking level on Gemini profiles):

```powershell
python run_experiment.py --pipeline multi --instance pu-cs-fal07 --seed 17 --model gemini_flash_lite --thinking-level medium --confirm-free-tier
python run_experiment.py --pipeline multi --instance pu-cs-fal07 --seed 17 --model cloudflare_mistral --confirm-free-tier
```

### Direct CLI and Live Logging Options

Run `python "Multi-Agent Controller/main.py" --help` to list all inherited baseline, controller, and LLM options (`--review-interval`, `--warmup-generations`, `--stagnation-window`, `--min-review-interval`, `--max-review-interval`, `--diversity-threshold`, `--diversity-recovery-threshold`, `--max-regions`, `--region-size`, `--max-operator-evaluations`, `--max-reviews`, `--provider`, `--model`, `--thinking-level`, `--confirm-free-tier`, `--api-min-interval-seconds`, `--api-max-input-characters`, `--api-timeout-seconds`, `--max-output-tokens`, `--max-api-calls`).

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

Direct runs save to `Multi-Agent Controller/results_itc2019/<run_id>/`, while `run_experiment.py --output DIR` saves to `DIR/multi/<run_id>/`.

- `run_config.json`: Run ID, full solver/repair/controller/LLM metadata (no API keys), source and instance SHA-256 hashes, and `completion_status` (`"completed"` or `"incomplete_api_failure"`).
- `summary.json`: Array of per-instance summaries.
- `<instance>/result.json`: Final score, feasibility, evaluation breakdown, `score_engine` stats, `stage_timing`, `monitor` metrics, and `controller` metrics (`api_calls`, cumulative token counts including `thinking_tokens`, `api_latency_seconds`, `api_pacing_wait_seconds`, `reviews`, `no_op`, `model_no_op`, `fallback_no_op`, `interventions`, `selected_interventions`, `completion_status`, `comparison_eligible`).
- `<instance>/controller_trace.jsonl`: One JSON line per review containing `generation`, `trigger`, `state` (`StatePacket`), `roles` (one `inspector` record on early stop, or sequential `inspector` and `planner` records), `decision_status`, `validation`, `action`, `outcome`, `population_after`, and `incumbent_after`.
- `<instance>/anytime_trace.jsonl`: Generation-level search trajectory and incumbent quality.
- `<instance>/choices.npz`: Compressed `time_indices` and `room_indices` arrays for the best observed solution.
- `<instance>/run.log`: Human-readable live progress including `[review #...]` lines showing both `inspector[...]` and `planner[...]` summaries.
- `<instance>/solution.xml` and `<instance>/expected.json`: Exported when `score.hard == 0` and internal sectioning passes (`official_status` stays `"not_run"` until verified separately via [../tools/validate_itc2019.py](../tools/validate_itc2019.py)).
