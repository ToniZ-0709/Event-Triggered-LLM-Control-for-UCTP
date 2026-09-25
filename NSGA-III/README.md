# NSGA-III Baseline

This folder contains the baseline solver and shared ITC 2019 implementation used by all three pipelines. The baseline does not call an LLM. The repository-level [README](../README.md) describes the pilot benchmark configuration and official validation.

## Setup and run

The XML dataset is in `../Dataset_ITC2019/`. From this folder:

```powershell
python -m pip install -r requirements.txt
python main.py --instance wbg-fal10 --population 40 --generations 20 --seed 42
```

Inspect an instance without optimizing:

```powershell
python main.py --instance wbg-fal10 --inspect
```

Run a manifest with one instance name per line:

```powershell
python main.py --manifest ../pilot_instances.txt --population 40 --partitions 4 --generations 1000 --seed 17 --time-limit-seconds 90
```

The primary options are `--instance`, `--manifest`, `--population`, `--partitions`, `--generations`, `--seed`, `--time-limit-seconds`, and `--output`. Run `python main.py --help` for the full list.

## Method and output

Each chromosome selects a time and room option for every class. NSGA-III searches four weighted penalty objectives: time, room, soft distribution, and student conflict. Hard violations are tracked as a constraint. Runs are written under `results_itc2019/`, including configuration, summary, per-instance score, and anytime trace files. A `solution.xml` is written only when the internal feasibility checker accepts the schedule.

Student sectioning uses a heuristic and can fail to find a feasible assignment even when one exists. An internally feasible solution still needs validation with the official ITC 2019 validator before it is reported as valid.
