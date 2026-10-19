# Event-Triggered LLM Control for University Course Timetabling

This repository compares three ITC 2019 course-timetabling pipelines: an NSGA-III baseline, a single-agent controller, and an Inspector-Planner controller. The two controllers use the Gemini Developer API with an explicit Free Tier confirmation; the baseline does not call an LLM. There is no OpenAI or paid-provider fallback.

## Pipelines

- [NSGA-III baseline](<NSGA-III/README.md>)
- [Single-Agent Controller](<Single-Agent Controller/README.md>)
- [Multi-Agent Controller](<Multi-Agent Controller/README.md>)

The shared ITC 2019 parser, evaluator, solver, regions, and improvement operators are in `NSGA-III/itc2019/`.
All three pipelines share bounded room-clash repair and a coordinated repair pass for required distribution constraints. A controller may submit a candidate that strictly reduces hard violations even if it is not yet feasible; once its source is feasible, it must preserve feasibility and improve soft cost. Only an internally feasible final candidate is exported as `solution.xml`.
The two technical reference documents are [Current pipelines](PIPELINES.md), describing the implementation and remaining limitations, and [Improvement history](IMPROVEMENT_HISTORY.md), tracing changes and decisions. Setup and CMD instructions remain in the pipeline guides.

## Local pilot setup

The pilot subset contains two instances from each official release group, ranging from 417 to 1,083 classes and including student data. The names and grouping follow the [ITC 2019 instance releases](https://www.itc2019.org/early-instances), [middle](https://www.itc2019.org/middle-instances), and [late](https://www.itc2019.org/late-instances) instance pages.

| Release group | Instances | Classes |
| --- | --- | ---: |
| Early | `muni-fi-spr16`, `muni-fsps-spr17` | 575, 561 |
| Middle | `yach-fal17`, `nbi-spr18` | 417, 782 |
| Late | `mary-fal18`, `bet-spr18` | 951, 1,083 |

The subset is recorded in [`pilot_instances.txt`](pilot_instances.txt). The measured local preset uses mutation rate `0.35`, class fraction `0.01`, crossover `0.7`, and offspring hard repair `8` rounds / `2` paired trials. Population/partitions are `12/2` for `nbi-spr18` and `yach-fal17`, and `20/3` for the other instances. Only `yach-fal17` gets a larger room-CP allowance (15 seconds, deterministic limit 2). These are bounded engineering choices, not globally optimal parameters. The same instance-specific settings apply to all three pipelines.

[`profiles/pilot.json`](profiles/pilot.json) supplies caps measured on the current solver, with explicit `provenance.budget_status` entries for open or time-censored horizons. Calibration covered all six instances at seeds 17/42 after screening and a seed-73 check. Four instances reached internal feasibility at both confirmation seeds; `yach-fal17` and `bet-spr18` still contain hard violations. The preset is therefore a working configuration, not a guarantee of feasible solutions or convergence.

[`run_pilot.py`](run_pilot.py) applies the preset and per-instance overrides automatically. Calling a pipeline's `main.py` directly uses its generic CLI defaults unless flags are provided. Search evaluations include routine repair, monitor repair, region extraction, operator trials and rechecks; cache hits count logically, while physical calls are recorded separately. Final verification is separate. The last offspring batch shrinks to respect the evaluation cap. Wall caps include conservative headroom and an API reserve, not a prediction of baseline runtime. Reaching `hard = 0` does not stop the default run: soft optimization continues until a cap.

After installing `NSGA-III/requirements.txt` in your venv, run these commands from the repository root in CMD. The first previews the commands without optimizing; the second runs all six instances:

```cmd
python run_pilot.py --pipeline baseline --seed 17 --dry-run
python run_pilot.py --pipeline baseline --seed 17
```

Run just one instance, or use another seed:

```cmd
python run_pilot.py --pipeline baseline --instance muni-fsps-spr17 --seed 42
```

Both controller pipelines require `GEMINI_API_KEY`. The default model is `gemini-3.8-flash`; `--model` or `GEMINI_MODEL` can select another model in the bounded Free Tier allowlist. Verify the key's project says **Free Tier** in AI Studio, without enabling billing, before passing `--confirm-free-tier`. This flag is a user confirmation: the API key itself cannot prove its billing tier. An allowlisted model can still be billed if used with a paid project.

Install `Single-Agent Controller/requirements.txt` in the same venv, then start with an offline setup check:

```cmd
python tools/gemini_preflight.py
python run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --dry-run
```

Only after setting the key locally and confirming Free Tier, these commands send real Gemini requests:

```cmd
python tools/gemini_preflight.py --live --confirm-free-tier
python run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --confirm-free-tier
```

Use `--pipeline multi` for Inspector–Planner. The profile allows at most 12 reviews (single up to 12 calls, multi up to 24), a 15-second request timeout, 1,024 maximum output tokens and low thinking. Request starts are paced at least 10 seconds apart; this is not a promise that every project's quota permits that rate. Pacing and API latency count toward runtime. Quota/billing/auth/network failures stop the controller run, save it as incomplete and return exit code 3; they are not model-selected `no_op` decisions and the comparison tool rejects them. No automatic retries or provider/model fallback are used. The transport has offline tests, not a live-model performance validation. See the [single guide](Single-Agent%20Controller/GUIDE.md), [multi guide](Multi-Agent%20Controller/GUIDE.md) and [current pipelines](PIPELINES.md).

## Official validation

The internal evaluator is not a substitute for the [official ITC 2019 validator](https://www.itc2019.org/validator). Log in to the ITC 2019 site, open **Validation**, upload a generated solution XML, and select **Validate**. The solution must use the ITC solution format and its instance name must match the problem instance. The [official FAQ](https://www.itc2019.org/faq) describes the validation and upload flow. Validation is required before treating a generated schedule as officially valid.

`tools/validate_itc2019.py` re-scores/exports saved feasible calibration choices without rerunning GA, or compares a saved official JSON response. Feasible future runs also write `expected.json` beside the XML. Official responses are stored in a separate `official_validation.json` receipt bound to the exact XML hash. `OK` with a different total is reported as `penalty_mismatch`, not a pass. The tool never calls the publication/upload endpoint. See [the validation workflow](NSGA-III/GUIDELINE.md#9-đối-chiếu-với-validator-chính-thức).

## Requirements

Python 3.10 or later. Install the requirements for the pipeline being used; controllers additionally need a Gemini API key from a confirmed Free Tier project. Keep credentials in local environment variables, never in the repository or command-line arguments.
