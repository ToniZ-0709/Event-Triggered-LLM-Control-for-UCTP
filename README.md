# Event-Triggered LLM Control for University Course Timetabling

This repository compares three ITC 2019 course-timetabling pipelines: an NSGA-III baseline, a single-agent controller, and an Inspector-Planner controller. The two controller pipelines use the OpenAI Responses API to select bounded solver actions; the baseline does not call an LLM.

## Pipelines

- [NSGA-III baseline](<NSGA-III/README.md>)
- [Single-Agent Controller](<Single-Agent Controller/README.md>)
- [Multi-Agent Controller](<Multi-Agent Controller/README.md>)

The shared ITC 2019 parser, evaluator, solver, regions, and improvement operators are in `NSGA-III/itc2019/`.

## Pilot benchmark setup

The pilot subset contains two instances from each official release group, ranging from 417 to 1,083 classes and including student data. The names and grouping follow the [ITC 2019 instance releases](https://www.itc2019.org/early-instances), [middle](https://www.itc2019.org/middle-instances), and [late](https://www.itc2019.org/late-instances) instance pages.

| Release group | Instances | Classes |
| --- | --- | ---: |
| Early | `muni-fi-spr16`, `muni-fsps-spr17` | 575, 561 |
| Middle | `yach-fal17`, `nbi-spr18` | 417, 782 |
| Late | `mary-fal18`, `bet-spr18` | 951, 1,083 |

The subset is recorded in [`pilot_instances.txt`](pilot_instances.txt). The configured run uses paired seeds `17`, `42`, and `73`, population `40`, partitions `4`, generation cap `1000`, and a `90` second per-instance wall-clock limit for all three pipelines. This produces 54 instance runs and takes about 81 minutes at the full time limit when run serially.

Both controller pipelines require an OpenAI API key in `OPENAI_API_KEY`. Supply the API model ID with `--model` or `OPENAI_MODEL`; the baseline does not use a model. Controller limits are a 15 second API timeout, 8 API calls for single-agent, and 16 for multi-agent, allowing up to 8 review opportunities per method.

Run each method from its own directory, repeating the command for each seed. For example, from `NSGA-III/`:

```powershell
python main.py --manifest ../pilot_instances.txt --population 40 --partitions 4 --generations 1000 --seed 17 --time-limit-seconds 90
```

For either controller, use the same solver options and add `--api-timeout-seconds 15`; also set `--max-api-calls 8` for single-agent or `16` for multi-agent. Configure `OPENAI_API_KEY` and provide the API model ID at runtime. API latency counts toward the wall-clock budget; controller runs may incur API charges. Keep each method's result directory for comparison with `compare_pipelines.py`.

## Official validation

The internal evaluator is not a substitute for the [official ITC 2019 validator](https://www.itc2019.org/validator). Log in to the ITC 2019 site, open **Validation**, upload a generated solution XML, and select **Validate**. The solution must use the ITC solution format and its instance name must match the problem instance. The [official FAQ](https://www.itc2019.org/faq) describes the validation and upload flow. Validation is required before treating a generated schedule as officially valid.

## Requirements

Python 3.10 or later. Install the requirements for the pipeline being used; controller pipelines additionally require an OpenAI API key.
