# Event-Triggered LLM Control for University Course Timetabling

An event-triggered Large Language Model (LLM) framework integrated with an NSGA-III evolutionary solver for complex university course timetabling on the ITC 2019 benchmark.

---

## 1. System Pipelines

The system evaluates three core timetabling pipelines sharing an identical underlying solver, constraint model, and per-instance evaluation budgets:

| Pipeline | Control Architecture | Technical Documentation |
| :--- | :--- | :--- |
| **Baseline** | Autonomous NSGA-III solver with built-in CP-SAT repairs (No LLM) | [NSGA-III Baseline](NSGA-III/README.md) |
| **Single-agent** | Single LLM invocation determining WHERE to intervene, HOW to repair, and WHEN to review | [Single-Agent Controller](Single-Agent%20Controller/README.md) |
| **Multi-agent** | Sequential role separation: Inspector determines WHERE $\rightarrow$ Planner determines HOW and WHEN | [Multi-Agent Controller](Multi-Agent%20Controller/README.md) |

---

## 2. Pilot Experimental Results

### Experimental Protocol
The pilot used 9 ITC 2019 instances, 2 seeds and 3 pipelines. The retained data comprises 16 matched triples from the original campaign and 2 from a separate `tg-spr18` follow-up. Budgets were set per instance:

| Instance | Classes | Rooms | Students | Evaluation Cap | Generation Cap | Wall-clock Cap (s) | Population N / Partitions H |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `muni-fi-spr16` | 457 | 40 | 2,752 | 1,200 | 200 | 840 | 20 / 3 |
| `muni-fsps-spr17` | 474 | 29 | 1,939 | 1,200 | 200 | 630 | 20 / 3 |
| `yach-fal17` | 134 | 43 | 863 | 1,200 | 200 | 1,050 | 12 / 2 |
| `nbi-spr18` | 196 | 13 | 1,061 | 1,200 | 200 | 750 | 12 / 2 |
| `mary-fal18` | 511 | 37 | 0 | 1,100 | 200 | 810 | 20 / 3 |
| `bet-spr18` | 682 | 47 | 0 | 2,300 | 230 | 2,880 | 20 / 3 |
| `pu-cs-fal07` | 174 | 13 | 2,002 | 1,200 | 200 | 1,200 | 20 / 3 |
| `lums-spr18` | 487 | 73 | 0 | 1,200 | 200 | 1,200 | 20 / 3 |
| `tg-spr18` | 676 | 24 | 0 | 1,200 | 200 | 1,200* | 20 / 3 |

*\* Note: For `tg-spr18`, a follow-up run without wall-clock time limit completed the full 1,200 evaluations for both seeds.*

### Empirical Performance Summary
The table below summarizes head-to-head outcomes across all 18 evaluated triples (9 instances $\times$ 2 seeds), with `tg-spr18` evaluated under removed wall-clock time limit to ensure full completion to 1,200 evaluations:

| Comparison | Wins / 18 Triples | Unranked Infeasible Tie | Interpretation |
| :--- | :---: | :---: | :--- |
| **Single-agent vs. Baseline** | **13 / 18 (72.2%)** | 1 | Single-agent wins in 13 instance-seed comparisons. |
| **Multi-agent vs. Baseline** | **12 / 18 (66.7%)** | 2 | Multi-agent wins in 12 instance-seed comparisons. |
| **Multi-agent vs. Single-agent** | **10 / 18 (55.6%)** | 3 | Multi-agent wins in 10 instance-seed comparisons. |

*(In the 16 original budget-matched triples under strict wall-clock time limits: Single vs. Baseline is 11/16 [68.8%], Multi vs. Baseline is 10/16 [62.5%], and Multi vs. Single is 9/16 [56.3%]).*

The follow-up did not use a global wall-clock cap. These are descriptive results from two seeds; they do not establish a general or statistically significant advantage. Equal positive hard-violation counts are unranked. The original time-capped `tg-spr18` runs are not retained. Reports and traces are in [ExperimentResult/phase1](ExperimentResult/phase1/).

### Resource and Token Overhead (Full 18 Matched Triples)

The table below summarizes cumulative resource consumption and API overhead across all 18 evaluated triples:

| Resource Metric | Baseline | Single-agent | Multi-agent | Multi vs. Single Ratio |
| :--- | ---: | ---: | ---: | ---: |
| **Completed Triples** | 18 | 18 | 18 | 1.0x |
| **Total Logical Evaluations** | 23,600 | 23,600 | 23,600 | 1.0x |
| **Feasible Solutions ($H = 0$)** | 11 / 18 | 12 / 18 | 11 / 18 | — |
| **Review Events (WHEN triggered)** | 0 | 121 | 134 | 1.11x |
| **Total API Invocations** | 0 | 121 | 268 | ~2.21x |
| **Input Tokens** | 0 | 837,663 | 1,870,804 | ~2.23x |
| **Output Tokens** | 0 | 9,460 | 27,951 | ~2.95x |
| **Total Native Tokens** | 0 | 847,123 | 1,898,915 | ~2.24x |
| **Cumulative API Latency (s)** | 0.0 | 247.2 | 1,631.2 | ~6.60x |
| **Total Execution Runtime (s)** | 10,491.6 | 11,545.6 | 13,449.6 | ~1.16x |

*(For the 16 original time-capped triples: Single-agent used 111 calls and 778,308 tokens, while Multi-agent used 250 calls and 1,772,823 tokens).*

---

## 3. Setup and Execution

### Prerequisites
- Python 3.10 or later
- Valid Google Gemini API Key (`GEMINI_API_KEY`) configured in your local environment

### Installation
Install dependencies for the baseline solver:
```cmd
python -m pip install -r NSGA-III/requirements.txt
```

For controller pipelines, install their respective dependencies:
```cmd
python -m pip install -r "Single-Agent Controller/requirements.txt"
python -m pip install -r "Multi-Agent Controller/requirements.txt"
```

Check the local SDK, key presence and configured model (no API call):
```cmd
python tools/gemini_preflight.py
```

Add `--live --confirm-free-tier` to send one test request.

### Running Experiments
Execute dry-run previews without consuming API quotas:
```cmd
python run_pilot.py --pipeline baseline --seed 17 --dry-run
python run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --dry-run
python run_pilot.py --pipeline multi --instance muni-fsps-spr17 --seed 17 --dry-run
```

To run on instances beyond the default pilot preset (e.g. `pu-cs-fal07`, `lums-spr18`, `tg-spr18`), provide `--profile profiles/expanded_pilot.json`:
```cmd
python run_pilot.py --profile profiles/expanded_pilot.json --pipeline single --instance pu-cs-fal07 --seed 17 --dry-run
```

Execute live optimization runs:
```cmd
python run_pilot.py --pipeline single --instance muni-fsps-spr17 --seed 17 --confirm-free-tier
python run_pilot.py --pipeline multi --instance muni-fsps-spr17 --seed 17 --confirm-free-tier
```

---

## 4. Output Artifacts

Each pipeline writes structured execution artifacts to its dedicated `results_itc2019/` directory:
- `run_config.json`: Hardware, solver parameters, XML instance hashes, and dependency versions.
- `result.json` / `summary.json`: Final objective scores, hard violation counts, runtime, and evaluation accounting.
- `choices.npz`: Complete chromosome placement state.
- `anytime_trace.jsonl`: Generation-level search metrics and incumbent quality.
- `controller_trace.jsonl`: Full trace of LLM prompt payloads, structured JSON responses, validation checks, and operator outcomes.
- `solution.xml`: Exported for internally feasible solutions; official validation is a separate step.

The phase-one source comparison is recorded in [source-verification.json](ExperimentResult/phase1/source-verification.json). The available `llm.py` has matching prompt and response-schema hashes in the recorded calls, but its file hash differs from the original; exact reproduction of the API adapter is therefore not established.

To check the archived files:

```cmd
python tools/audit_expanded_pilot.py
```

The audit does not perform official validation.

### Plot Phase 1 results

Install the plotting dependency:

```cmd
python -m pip install -r tools/requirements-plotting.txt
```

Generate all three graphs for every available instance and seed:

```cmd
python tools/plot_phase1.py
```

Select instances or seeds:

```cmd
python tools/plot_phase1.py --instance mary-fal18 tg-spr18 --seed 17 42
```

Generate one graph for one instance and seed:

```cmd
python tools/plot_phase1.py --instance mary-fal18 --seed 17 --graph convergence
```

Graph names are `final_quality`, `convergence` and `resources`. Pass more than one name after `--graph` to select several. PNG files are written to `ExperimentResult/phase1/graphs/<instance>/seed_<seed>/`. Use `--output` to choose a different base folder. Running the same command again replaces the selected PNG files.
