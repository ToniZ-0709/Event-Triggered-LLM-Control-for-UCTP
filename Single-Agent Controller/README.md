# Single-Agent Controller

A single Gemini call selects WHERE to intervene, HOW to modify the region, and WHEN to review again. The shared solver validates the decision, executes an allowed operator, and scores the candidate before environmental selection.

The input is the StatePacket (Schema v1), containing global population metrics, stagnation history, candidate conflict regions, recent outcomes, and allowed actions. Valid operators are `move`, `swap`, `destroy_repair`, and `no_op`.

Commands below run from the repository root:

```cmd
python -m pip install -r "Single-Agent Controller/requirements.txt"
python tools/gemini_preflight.py
python run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --dry-run
```

Optimization requires `GEMINI_API_KEY` and confirmation that its project is Free Tier. The selected profile supplies the model and request limits.

```cmd
python run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --confirm-free-tier
```

Results are saved to `Single-Agent Controller/results_itc2019/`. A failed API request saves an incomplete run and returns exit code 3. See the [repository README](../README.md) for profiles and archived results.
