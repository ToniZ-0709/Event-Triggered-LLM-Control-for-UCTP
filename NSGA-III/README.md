# NSGA-III Baseline

This folder contains the baseline solver and shared ITC 2019 implementation used by all three pipelines. The baseline does not call an LLM. The repository-level [README](../README.md) describes the pilot benchmark configuration and official validation.

## Setup and run

The XML dataset is in `../Dataset_ITC2019/`. From this folder:

```powershell
python -m pip install -r requirements.txt
python main.py --instance wbg-fal10 --generations 20 --seed 42
```

Inspect an instance without optimizing:

```powershell
python main.py --instance wbg-fal10 --inspect
```

Run the six-instance preset measured on the current solver (open and time-censored horizons remain explicitly marked in the profile):

```powershell
python ../run_pilot.py --pipeline baseline --seed 17
python ../run_pilot.py --pipeline baseline --instance muni-fsps-spr17 --seed 42 --dry-run
```

To stop each instance early as soon as a population contains an internally feasible candidate, keep a generation cap and add `--stop-on-feasible`; a wall-clock limit is optional:

```powershell
python main.py --instance muni-fsps-spr17 --generations 200 --evaluations 2400 --seed 17 --stop-on-feasible
```

The solver checks feasibility after each completed generation. It stops at the first such generation or at the generation cap, whichever comes first. If no feasible candidate is found by the cap, no `solution.xml` is written. Without a time limit, runtime can be long, and one generation may itself take substantial time.

Defaults are population 20, partitions 3, crossover 0.7, and offspring repair 8 rounds / 2 paired trials. `--inner-partitions` enables an optional second reference layer; it is not selected by the local preset. Mutation/crossover, initial repair, full evaluation caps, and shared repair budgets are exposed in `python main.py --help`. See [Current pipelines](../PIPELINES.md) for implementation details and [Improvement history](../IMPROVEMENT_HISTORY.md) for the decisions behind them.

The wrapper preset differs from generic CLI defaults: mutation is 0.35 / class fraction 0.01; `nbi-spr18` and `yach-fal17` use population 12 / partitions 2. `yach-fal17` also uses a 15-second room-CP deadline and deterministic limit 2. Use `run_pilot.py` to apply the measured settings consistently to every pipeline. Its time ceilings include safety/API headroom; they are not expected baseline runtimes.

## Method and output

Each chromosome selects a time and room option for every class. NSGA-III searches four weighted penalty objectives: time, room, soft distribution, and student conflict. Hard violations are a constraint. Construction uses required-constraint propagation and bounded forward checking. The shared monitor combines multi-class distribution CP repair, plateau search and adaptive near-feasible room repair. Room CP widens from 16 to 32 and at most 64 classes/options within one shared deadline (generic default: 8 seconds), then uses a small fallback (2 seconds / 128 full checks). The profile can override that allowance by instance. Required global block/break constraints and fixed boundaries are modeled directly. Failed room targets wait 5, then 10, then 20 generations before retrying. A shared incumbent retains good scored candidates even if survival rejects them. All three pipelines use these same mechanisms.

The bounded score cache saves repeated computation without reducing logical evaluation charges. Outputs under `results_itc2019/` include configuration, per-instance scores, logical/physical scoring counts, sectioning fallback status, stage timing, repair events and anytime traces. Two final verification calls are separate from the search budget. A `solution.xml` is written only when the final internal feasibility and sectioning checks pass.

If greedy/beam student sectioning fails, a bounded course-level capacity plan reassigns earlier students as well. Its bundle enumeration and CP budget are finite, so failure still does not establish instance infeasibility. Internal feasibility is not a substitute for the official ITC 2019 validator.
