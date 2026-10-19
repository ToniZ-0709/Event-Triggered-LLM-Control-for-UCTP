# Single-Agent Controller

This pipeline adds one bounded LLM decision to the shared NSGA-III solver. At each review, the agent selects a target region, an allowed improvement operator, and the next review interval. The solver validates the response and evaluates any proposed change. Shared solver code is in [`../NSGA-III/itc2019/`](../NSGA-III/itc2019/).

## Setup and run

From this folder, install the requirements. Set `GEMINI_API_KEY` locally and verify its project is **Free Tier** in AI Studio. There is no OpenAI dependency or paid-provider fallback. The profile selects `gemini-3.8-flash`.

```cmd
python -m pip install -r requirements.txt
python ../tools/gemini_preflight.py
python ../run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --dry-run
python ../run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --confirm-free-tier
```

The wrapper uses the shared local profile and per-instance budgets. Add `--instance NAME` to run one instance, or change `--seed` for another run. Both agents should use the same model and profile. API latency is included in the time cap. `python main.py --instance NAME --inspect` parses XML without API calls. See [GUIDE.md](GUIDE.md) for operation, [Current pipelines](../PIPELINES.md) for implementation, and [Improvement history](../IMPROVEMENT_HISTORY.md) for decisions.

Results are saved under `results_itc2019/`. Only the last command sends real Gemini calls. `--confirm-free-tier` is a human confirmation, not a billing-tier lookup: an API key linked to a paid project can still incur charges. Quota/network failures produce an incomplete run (exit code 3), not a successful no-op experiment. See the guide before running all six instances.
