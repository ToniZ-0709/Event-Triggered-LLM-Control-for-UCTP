# Event-Triggered LLM Control for University Course Timetabling

Three ITC 2019 pipelines share one NSGA-III solver and differ in how they control regional improvement steps.

| Pipeline | Control | Documentation |
| --- | --- | --- |
| Baseline | Shared solver and repair, without an LLM | [Baseline](NSGA-III/README.md) |
| Single-agent | One call selects the target, operator and next review interval | [Single-agent](Single-Agent%20Controller/README.md) |
| Inspector-Planner | Inspector selects the target; Planner selects the operator and interval | [Multi-agent](Multi-Agent%20Controller/README.md) |

[PIPELINES.md](PIPELINES.md) is the technical reference for scoring, repair, controller contracts, budgets and limitations. Shared implementation is in `NSGA-III/itc2019/`.

Presentation: [PDF slides](output/pdf/PIPELINES_SLIDES.pdf) and [editable LaTeX source](output/latex/pipelines/main.tex). These teaching slides describe the earlier controller contract; use PIPELINES.md for the current StatePacket v2 and action interface.

## Current implementation

Both controllers consume the same six-block StatePacket v2, with bounded read-only diagnostics, fixed-boundary context and detailed action outcomes. The original random `move`, `swap` and `destroy_repair` remain available. `guided_move` prioritizes screened one-gene alternatives; `distribution_repair` targets an actionable violated required rule. The optional `--enable-blocker-chain` flag on a controller's direct entrypoint publishes a separate expanded region before permitting coordinated chain moves.

[implementation_plan.md](implementation_plan.md) records the status of Sections 1, 2 and 4. Section 3 (XAI) remains a proposal. Offline regression tests run with `python -m pytest NSGA-III/tests -q`.

## Setup

Python 3.10 or later. For the baseline:

```cmd
python -m pip install -r NSGA-III/requirements.txt
```

Controller requirements are listed in their pipeline READMEs. Both controllers use Google's Gemini Developer API through `google-genai`. Set `GEMINI_API_KEY` locally; the model and request limits come from the selected profile. `--model` or `GEMINI_MODEL` can override the model within the adapter's allowlist.

API runs require Free Tier confirmation (`--confirm-free-tier` or `GEMINI_FREE_TIER_CONFIRMED=1`) after checking the key's project in AI Studio. This records a manual billing-tier check; the key itself does not verify the project's tier. Credentials belong in environment variables.

## Configuration

| Profile | Scope |
| --- | --- |
| [pilot.json](profiles/pilot.json) | Original six-instance preset |
| [expanded_pilot.json](profiles/expanded_pilot.json) | Nine instances, retaining the original six presets |
| [tg_no_timecap.json](profiles/tg_no_timecap.json) | Separate `tg-spr18` follow-up without a global wall-clock cap |

[`run_pilot.py`](run_pilot.py) applies shared settings and per-instance overrides. Select the method with `--pipeline baseline`, `single` or `multi`, and a profile with `--profile`. Direct `main.py` calls use CLI defaults rather than loading a profile.

Preview commands without optimizing or calling the API:

```cmd
python run_pilot.py --pipeline baseline --seed 17 --dry-run
python run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --dry-run
```

Execution commands are in the pipeline READMEs. The XML instances are in `Dataset_ITC2019/`; `pilot_instances.txt` lists the original six.

## Saved results

Each pipeline writes to its own `results_itc2019/` directory. Runs contain configuration and source hashes, scores, saved choices and generation traces; controller runs also include API decision traces. `solution.xml` is exported only after internal feasibility checks.

| Report | Recorded status |
| --- | --- |
| [Expanded comparison](solver_checks/expanded_pilot_comparison.json) | 54/54 completed cells: nine instances, two seeds and three methods |
| [No-time-cap follow-up](solver_checks/tg_no_timecap_comparison.json) | 6/6 completed cells; prior failures and interruption retained |

The expanded report has 16/18 triples with matched evaluation budgets. Scores are internal; official validation has not been completed. Experiment plots and their source manifest are in `output/expanded_pilot/`. The follow-up and original endpoints are separate datasets.

The saved campaigns predate StatePacket v2 and the new operators. Their scores are historical results, not evidence of improvement from the current code. New experiments require a fresh campaign state root and output namespace because source hashes have changed.

## Search diagnostics figures

Generate separate final-quality, convergence and resource figures for each instance/seed from saved reports. This command uses no API and leaves the original nine-panel figures untouched:

```cmd
python -m pip install -r tools/requirements-plotting.txt
python tools/plot_search_results.py --report solver_checks/tg_no_timecap_comparison.json --output output/search_diagnostics/tg_no_timecap --extra reviews diversity operators objectives
```

Outputs include editable PDF/SVG, PNG previews, source CSV, geometry checks and manifests preserving attempts, missing usage and comparison limits. Optional objective plots require saved incumbent vectors; legacy population minima are never substituted. Example figures for seeds 17 and 42 are in `output/search_diagnostics/tg_no_timecap/`.
