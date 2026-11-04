# Single-Agent Controller

One Gemini call selects WHERE to intervene, HOW to modify the region and WHEN to review again. The shared solver validates the decision, executes an allowed operator and scores the candidate before selection.

The input is the shared six-block StatePacket v2. Decisions use the ActionPlan fields; local diagnostics guide operator selection and do not supply executable placements. Actions include `move`, `swap`, `destroy_repair`, `guided_move` and `distribution_repair`. The direct entrypoint optionally accepts `--enable-blocker-chain` for an explicitly published expanded target. The response does not include a rationale.

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

Results go to `Single-Agent Controller/results_itc2019/`. A failed API request saves an incomplete run and returns exit code 3. See [PIPELINES.md](../PIPELINES.md) for review rules, actions, accounting and comparison requirements, or the [repository README](../README.md) for profiles and saved reports.
