# Inspector-Planner Controller

A review uses up to two sequential Gemini calls. Inspector selects WHERE to intervene; after target validation, Planner receives the state and diagnosis to select HOW and WHEN. The shared solver executes and scores the proposed change.

Both roles use the same six-block StatePacket v2. Inspector selects IDs from `candidate_regions[].target` and references `boundary_context[].constraints.ids`; Planner chooses an operator from that region's `available_actions`. The shared actions now include `guided_move` and `distribution_repair`. The direct entrypoint optionally accepts `--enable-blocker-chain` for a separately published expanded target. Decisions retain their existing response fields; Section 3 XAI fields remain deferred.

Commands below run from the repository root:

```cmd
python -m pip install -r "Multi-Agent Controller/requirements.txt"
python tools/gemini_preflight.py
python run_pilot.py --pipeline multi --instance muni-fsps-spr17 --seed 17 --dry-run
```

Optimization requires `GEMINI_API_KEY` and confirmation that its project is Free Tier. Both roles use the model and request limits in the selected profile.

```cmd
python run_pilot.py --pipeline multi --instance muni-fsps-spr17 --seed 17 --confirm-free-tier
```

Results go to `Multi-Agent Controller/results_itc2019/`. A failed API request saves an incomplete run and returns exit code 3. See [PIPELINES.md](../PIPELINES.md) for review rules, actions, accounting and comparison requirements, or the [repository README](../README.md) for profiles and saved reports.
