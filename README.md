# Event-Triggered LLM Control for University Course Timetabling

An event-triggered Large Language Model (LLM) control framework integrated with an NSGA-III evolutionary solver for university course timetabling on the ITC 2019 benchmark (`Dataset_ITC2019/`).

---

## 1. Architecture and Pipelines

All five pipelines share the same chromosome encoding, constraint evaluation engine, student sectioning decoder, incumbent tracking, and built-in CP-SAT and neighborhood repair routines (`NSGA-III/itc2019/`).

| Pipeline | CLI value | Control architecture | Technical reference |
| :--- | :--- | :--- | :--- |
| **Baseline** | `baseline` | NSGA-III with built-in CP-SAT and neighborhood repairs (`0` controller reviews) | [NSGA-III Baseline](NSGA-III/README.md) |
| **Random** | `random` | Offline uniform sampling over legal `(region, operator)` pairs plus `no_op` (`0` API calls) | [Random Controller](Random%20Controller/README.md) |
| **Greedy** | `greedy` | Offline penalty-ranked region selection and budgeted operator probing (`0` API calls) | [Greedy Controller](Greedy%20Controller/README.md) |
| **Single-agent** | `single` | One Gemini call per review selects WHERE, HOW, and WHEN from `StatePacket` | [Single-Agent Controller](Single-Agent%20Controller/README.md) |
| **Multi-agent** | `multi` | Sequential Gemini Inspector (WHERE) and Planner (HOW, WHEN) calls per review | [Multi-Agent Controller](Multi-Agent%20Controller/README.md) |

### Shared Controller Contract

The four controller pipelines (`random`, `greedy`, `single`, `multi`) inherit `ControllerCallback` (`NSGA-III/itc2019/controller.py`) and share the same event-triggered review loop:

- **Review triggers (WHEN):** After a 5-generation warmup (`--warmup-generations`) and subject to a minimum gap of 3 generations (`--min-review-interval`), a review fires on a scheduled deadline (default every 10 generations, bounded in `[3, 20]`), search stagnation (5 generations without hard or feasible-soft progress), or placement diversity dropping to `0.20` (re-armed above `0.30`). Reviews stop once `max_reviews` (default `12`) is reached or fewer than 4 evaluations remain.
- **Candidate regions (WHERE) and operators (HOW):** `RegionBuilder` extracts up to `max_regions` (`6`) local conflict regions of up to `region_size` (`5`) classes from up to 2 representative individuals. Valid operators are `move`, `swap`, `destroy_repair`, and `no_op`.
- **Execution budgets:** `Random` and `Greedy` fix `execution_budget` at `min(10, current_operator_limit)` and lock `next_review_interval` to `review_interval` (`10`). `Single-agent` and `Multi-agent` choose `execution_budget` in `[1, current_operator_limit]` (capped by `max_operator_evaluations`, default `24`) and `next_review_interval` in `[3, 20]`.
- **Solver validation:** Every decision is validated against active region IDs, legal operators, and integer bounds before any operator executes or modifies the population.

---

## 2. Setup and Prerequisites

### Requirements

- Python `3.10` or later
- Google Gemini API key (`GEMINI_API_KEY`) for the `single` and `multi` pipelines only

Install the shared solver dependencies for `baseline`, `random`, and `greedy`:

```powershell
python -m pip install -r NSGA-III/requirements.txt
```

Install the Gemini controller dependencies (`google-genai` plus solver dependencies) for `single` and `multi`:

```powershell
python -m pip install -r "Single-Agent Controller/requirements.txt"
python -m pip install -r "Multi-Agent Controller/requirements.txt"
```

### Gemini Free Tier Configuration and Preflight

Only published Gemini Free Tier text models are permitted by `NSGA-III/itc2019/llm.py` (`FREE_TIER_MODELS`):

- `gemini-4-hydrogen` (default)
- `gemini-3.5-flash-lite`

Set `GEMINI_API_KEY` locally in your environment (never pass a key as a CLI argument or commit it, and avoid setting a conflicting `GOOGLE_API_KEY`). Because an API key alone does not expose its project billing tier, confirm in Google AI Studio that the project is on the Free Tier and either pass `--confirm-free-tier` or set `GEMINI_FREE_TIER_CONFIRMED=1`.

The local campaign runner reads numbered Gemini keys from the repository-root `.env` file (`GEMINI_API_KEY_001`, `GEMINI_API_KEY_002`, and so on). Keep existing slots in order and append new ones using the next number so the quota ledger can keep tracking the same keys. Direct CLI runs read `GEMINI_API_KEY` from the process environment.

Run the offline preflight check (zero network calls) from the repository root (`Experiment/`):

```powershell
python tools/gemini_preflight.py
```

Run one bounded synthetic `no_op` live check after confirming the Free Tier project:

```powershell
python tools/gemini_preflight.py --live --confirm-free-tier --model gemini-4-hydrogen --thinking-level low --timeout-seconds 60
```

`--thinking-level` accepts `minimal`, `low` (default), `medium`, or `high`, and can also be set via `GEMINI_THINKING_LEVEL`. `--model` can also be set via `GEMINI_MODEL`.

### API Timeout, Retries, and Fail-Closed Policy

- **Request timeout and pacing:** Each Gemini call uses a `60s` default timeout (`--api-timeout-seconds`), a `10.0s` minimum interval between request starts (`--api-min-interval-seconds`), and a `60,000`-character input cap (`--api-max-input-characters`).
- **Transient retries:** The SDK retries up to `attempts=3` times only on transient timeouts and HTTP 5xx status codes (`408`, `500`, `502`, `503`, `504`).
- **Fail-closed exit code `3`:** Quota exhaustion (`429`), billing-required (`402`), configuration or auth errors (`400`, `401`, `403`, `404`), and unrecovered API errors do not fall back to another provider or model. The controller disables further API calls, stops the run, saves partial results marked `completion_status: "incomplete_api_failure"`, and exits with code `3`.

---

## 3. Running Experiments

Run all commands from the repository root (`Experiment/`). Add `--dry-run` with `run_pilot.py` to print the resolved solver commands and budgets before running.

### Execution Profiles

| Profile | Instances | LLM model in profile | Purpose |
| :--- | :--- | :--- | :--- |
| [profiles/pilot.json](profiles/pilot.json) | 6 instances (`muni-fi-spr16`, `muni-fsps-spr17`, `yach-fal17`, `nbi-spr18`, `mary-fal18`, `bet-spr18`) | `gemini-3.5-flash-lite` | Pilot profile with uniform `population=20`, `partitions=3`, default repair, and measured Phase 1 eval/time caps |
| [profiles/phase1.json](profiles/phase1.json) | 8 instances  (`muni-fi-spr16`, `muni-fsps-spr17`, `yach-fal17`, `nbi-spr18`, `mary-fal18`, `bet-spr18`, `pu-cs-fal07`, `lums-spr18`) | `gemini-3.5-flash-lite` | Phase 1 benchmark profile (`population=20`, `partitions=3`, default repair) |
| [profiles/phase2.json](profiles/phase2.json) | 18 instances | `gemini-4-hydrogen`, thinking `low` | Main experiment with all five pipelines, seeds 17, 42 and 73, uniform `population=20`, `partitions=3`, default repair, up to 1,800 evaluations and 200 generations per run, and no global time cap |

### Running Stage 2A from PowerShell

Run the diagnostic slice for `bet-spr18`, `muni-fsps-spr17`, and `iku-spr18` with seed 17 and all five pipelines:

```powershell
.\tools\run_stage2a.ps1 -DryRun
.\tools\run_stage2a.ps1 -ConfirmFreeTier
```

The second command streams progress in the terminal and prints each job's `stderr.log` path. Results, per-job logs, and Hydrogen HTTP counts are saved under a timestamped directory in `tmp/stage2a/`. The default is one worker. Use `-Workers 3` to overlap offline jobs with a single Gemini job. Gemini jobs use the campaign's `.env` pool and shared quota tracker.

### Previewing Commands (`--dry-run`)

Preview all five pipelines on an instance:

```powershell
python run_pilot.py --pipeline baseline --pipeline random --pipeline greedy --pipeline single --pipeline multi --instance muni-fsps-spr17 --seed 17 --dry-run
```

Preview a Phase 1 profile run:

```powershell
python run_pilot.py --profile profiles/phase1.json --pipeline single --instance pu-cs-fal07 --seed 17 --dry-run
```

Preview the 90 Phase 2 jobs for one seed. Run the command separately for each
of the three seeds when preparing the campaign. The explicit model and
thinking flags take precedence over any environment overrides.

```powershell
python run_pilot.py --profile profiles/phase2.json --pipeline baseline --pipeline random --pipeline greedy --pipeline single --pipeline multi --seed 17 --model gemini-4-hydrogen --thinking-level low --output ExperimentResult/phase2 --dry-run
```

### Running Offline and Gemini Pipelines via `run_pilot.py`

Run offline pipelines (`baseline`, `random`, `greedy`) with an explicit output directory:

```powershell
python run_pilot.py --profile profiles/phase2.json --pipeline baseline --instance muni-fsps-spr17 --seed 17 --output ExperimentResult/phase2
python run_pilot.py --profile profiles/phase2.json --pipeline random --instance muni-fsps-spr17 --seed 17 --output ExperimentResult/phase2
python run_pilot.py --profile profiles/phase2.json --pipeline greedy --instance muni-fsps-spr17 --seed 17 --output ExperimentResult/phase2
```

Run Gemini pipelines (`single`, `multi`) with `--confirm-free-tier` (optionally overriding `--model` and `--thinking-level`):

```powershell
python run_pilot.py --profile profiles/phase2.json --pipeline single --instance muni-fsps-spr17 --seed 17 --confirm-free-tier --model gemini-4-hydrogen --thinking-level low --output ExperimentResult/phase2
python run_pilot.py --profile profiles/phase2.json --pipeline multi --instance muni-fsps-spr17 --seed 17 --confirm-free-tier --model gemini-4-hydrogen --thinking-level low --output ExperimentResult/phase2
```

Omit `--instance` to run every instance in the selected `--profile`. When `--output` is omitted, each pipeline writes directly to its own `results_itc2019/` folder rather than `<output>/<pipeline>/`. `run_pilot.py` sets `max_api_calls` to `max_reviews` (`12`) for `single` and `2 * max_reviews` (`24`) for `multi`.

### Parallel Workers and Live Logs

`run_pilot.py` defaults to `--workers 1` (sequential). Set `--workers N` (`N >= 2`) to run jobs concurrently and add `--live` to stream worker progress lines (`[pipeline/instance] ...`) to terminal `stderr`:

```powershell
python run_pilot.py --workers 3 --live --pipeline baseline --pipeline random --pipeline greedy --instance muni-fsps-spr17 --seed 17 --output ExperimentResult/phase2
```

At most one Gemini job (`single` or `multi`) runs at a time per batch while offline jobs fill remaining worker slots. If any job returns a non-zero exit code, no new jobs are started, active jobs are waited on, and `run_pilot.py` exits with that failed job's code.

When `--workers 2` or higher is used:

- Batch logs are written to `<output>/pilot_logs/batch-*/` (defaulting to `ExperimentResult/phase2/pilot_logs/batch-*/` when `--output` is omitted), containing `stderr.log` (progress and errors) and `stdout.log` (JSON summary output).
- The runner prints `[start] pipeline/instance -> <stderr.log path>` and `[exit <code>] pipeline/instance`.
- Follow one job's log in a separate PowerShell window using the path from its `[start]` line:

```powershell
Get-Content -LiteralPath "<stderr.log path from the start line>" -Tail 20 -Wait
```

### Direct Entrypoints and Progress Flags

Each pipeline can also be invoked directly (`NSGA-III/main.py`, `"Random Controller/main.py"`, `"Greedy Controller/main.py"`, `"Single-Agent Controller/main.py"`, `"Multi-Agent Controller/main.py"`). Pass `--manifest pilot_instances.txt` to select the 6 pilot instances (or pass repeated `--instance` flags; use `run_pilot.py --profile profiles/phase2.json` to run all 18 Phase 2 instances with the Phase 2 solver and controller settings). Direct entrypoints also accept `--inspect`, `--all`, `--stop-on-feasible`, `--time-limit-seconds`, and live logging flags:

- `--progress-interval N`: emit generation progress every `N` generations (`0` logs reviews and repairs only, default `1`)
- `--progress-log PATH`: append progress lines to an additional log file
- `--quiet`: suppress terminal `stderr` progress while keeping `<instance>/run.log`

Example direct inspection and run:

```powershell
python NSGA-III/main.py --instance wbg-fal10 --inspect
python "Random Controller/main.py" --instance muni-fsps-spr17 --seed 17 --generations 200 --evaluations 1200 --time-limit-seconds 630 --progress-interval 5
```

Direct `main.py` entrypoints default `--seed` to `42` and `--generations` to `20`, whereas `run_pilot.py` defaults `--seed` to `17` and applies full per-instance budgets from the selected profile.

---

## 4. Output Artifacts and Pipeline Comparison

### Run Output Layout

With `run_pilot.py --output DIR`, results are published under `DIR/<pipeline>/<run_id>/`. Without `--output`, each pipeline writes to its own `results_itc2019/<run_id>/` folder (for example `NSGA-III/results_itc2019/<run_id>/`).

Run-level files under `<run_id>/`:

- `run_config.json`: Hardware and environment versions, solver, repair, controller, and LLM settings, dataset XML SHA-256 hashes, source file SHA-256 hashes, and `completion_status` (`"completed"` or `"incomplete_api_failure"`).
- `summary.json`: JSON array summarizing all instances executed in the run (consumed by `compare_pipelines.py`).

Per-instance files under `<run_id>/<instance>/`:

- `result.json`: Per-instance scores, feasibility, runtime, evaluation breakdown, score-engine and stage timing, monitor counters, and `controller` metrics.
- `choices.npz`: Compressed NumPy archive (`time_indices`, `room_indices`) storing the best observed solution.
- `anytime_trace.jsonl`: Generation-by-generation incumbent quality and diversity trace.
- `controller_trace.jsonl`: Controller runs only (`random`, `greedy`, `single`, `multi`). Records review triggers, `StatePacket` inputs, role records (`probes` for `greedy`, Gemini request and token records for `single`/`multi`), validation status, actions, and outcomes.
- `run.log`: Human-readable progress and completion status log.
- `solution.xml` and `expected.json`: Exported only when the final solution has zero hard violations (`score.hard == 0`) and passes student sectioning validation.

### Official ITC 2019 Validation

Scores in `summary.json` and `result.json` come from the internal evaluator (`validation.official_status` starts as `"not_run"`). Official competition validation is a separate step via `tools/validate_itc2019.py`, either by submitting with a local `ITC2019_TOKEN` (`submit --solution <solution.xml>`) or importing a manually saved official validator JSON response (`import --solution <solution.xml> --response <response.json>`), which writes `official_validation.json` next to `solution.xml`.

### Comparing Completed Runs

Use `compare_pipelines.py` to generate a paired comparison JSON across completed `<run_id>/` directories. `--baseline`, `--single`, and `--multi` are required, while `--random` and `--greedy` are optional:

```powershell
python compare_pipelines.py --baseline <baseline_run_dir> --random <random_run_dir> --greedy <greedy_run_dir> --single <single_run_dir> --multi <multi_run_dir> --output comparison.json
```

`compare_pipelines.py` checks matching instance lists, solver and repair settings, dependency versions, Gemini model and `llm.py` hashes, shared review parameters, zero API failures (`comparison_eligible: true`), and `fixed_execution_budget == 10` for `random`/`greedy`. For runs without `--time-limit-seconds` (such as `profiles/phase2.json`), pass `--allow-generation-only`.
