# NSGA-III Baseline

The baseline runs the shared ITC 2019 solver, including the same construction, scoring, repair, and incumbent tracking used by both controller pipelines. It makes no LLM calls.

Commands below run from the repository root:

```cmd
python -m pip install -r NSGA-III/requirements.txt
python NSGA-III/main.py --instance wbg-fal10 --inspect
python run_pilot.py --pipeline baseline --instance muni-fsps-spr17 --seed 17 --dry-run
```

To optimize using the pilot profile:

```cmd
python run_pilot.py --pipeline baseline --instance muni-fsps-spr17 --seed 17
```

Omit `--instance` to process the profile's instance list. `--stop-on-feasible` is available on the direct entrypoint; the default search continues improving soft cost after feasibility. Run `python NSGA-III/main.py --help` for CLI options.

Results go to `NSGA-III/results_itc2019/`. See the [repository README](../README.md) for profiles and archived results.
