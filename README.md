# Event-Triggered LLM Control for University Course Timetabling

Three ITC 2019 pipelines share one NSGA-III solver and differ in how they control regional improvement steps.

| Pipeline | Control | Documentation |
| --- | --- | --- |
| Baseline | Shared solver and repair, without an LLM | [Baseline](NSGA-III/README.md) |
| Single-agent | One call selects the target, operator and next review interval | [Single-agent](Single-Agent%20Controller/README.md) |
| Inspector-Planner | Inspector selects the target; Planner selects the operator and interval | [Multi-agent](Multi-Agent%20Controller/README.md) |

[PIPELINES.md](PIPELINES.md) is the technical reference for scoring, repair, controller contracts, budgets and limitations. Shared implementation is in `NSGA-III/itc2019/`.

## Current implementation

The controllers use the six-block StatePacket v2, with bounded diagnostics, boundary context and recent outcomes. Single-agent selects the region, operator and review interval in one call. Inspector-Planner first selects a region, then passes the StatePacket and diagnosis to Planner for its action.

The work plan in [FinalizedWork.md](FinalizedWork.md) covers the planned Random and Greedy controls, XAI, experiments and result figures. Random and Greedy are not implemented. The offline XAI trace reader is a prototype and does not explain the model's internal reasoning.

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

## Outputs

Each pipeline writes to its own `results_itc2019/` directory. Runs contain configuration and source hashes, scores, saved choices and generation traces; controller runs also include API decision traces. `solution.xml` is exported only after internal feasibility checks.

See [PIPELINES.md](PIPELINES.md) for the output contract and comparison limits. A run's source hashes identify the implementation that produced it; historical scores should not be attributed to a different StatePacket version.
