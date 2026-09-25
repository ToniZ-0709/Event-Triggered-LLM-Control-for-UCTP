# Multi-Agent Controller

This pipeline adds an Inspector-Planner controller to the shared NSGA-III solver. At each review, the Inspector selects a target region; the Planner selects an allowed improvement operator and the next review interval. A review uses up to two sequential LLM calls. The solver validates the decisions and evaluates any proposed change. Shared solver code is in [`../NSGA-III/itc2019/`](../NSGA-III/itc2019/).

## Setup and run

From this folder, install the requirements. API-backed optimization requires an OpenAI API key in `OPENAI_API_KEY`; provide the API model ID with `--model` or `OPENAI_MODEL` when running.

```powershell
python -m pip install -r requirements.txt
python main.py --manifest ../pilot_instances.txt --population 40 --partitions 4 --generations 1000 --seed 17 --time-limit-seconds 90 --api-timeout-seconds 15 --max-api-calls 16
```

Repeat with seeds `42` and `73`. For comparison, use the same model, instance manifest, seeds, solver settings, and time limit as the single-agent pipeline. API latency is included in the time limit. `--inspect` parses the instances without making API calls. Run `python main.py --help` for controller-specific options.

Results are saved under `results_itc2019/`. The run metadata and API request records include configuration details. API use may incur charges.
