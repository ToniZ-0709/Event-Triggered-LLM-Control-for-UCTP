# Inspector-Planner Controller

A review uses up to two sequential Gemini calls. Inspector selects WHERE to intervene, then after target validation, Planner receives the StatePacket and Inspector diagnosis to select HOW and WHEN. The shared solver validates and executes the proposed change. A review can end before the Planner call.

Both roles operate on the StatePacket (Schema v1). Inspector selects target IDs (`solution_id`, `region_id`, `boundary_id`) and relevant constraint IDs from candidate regions. Planner receives this diagnosis and selects an operator from `move`, `swap`, `destroy_repair`, and `no_op`, alongside execution budget and next review interval.

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

Results are saved to `Multi-Agent Controller/results_itc2019/`. A failed API request saves an incomplete run and returns exit code 3. See the [repository README](../README.md) for profiles and archived results.
