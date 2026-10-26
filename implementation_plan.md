# Improvement plan

Implementation status (26 October 2026): Sections 1, 2 and 4 are implemented in the shared pipeline. Section 3 remains a proposal and is unchanged. This document retains the design specification and historical examples; [PIPELINES.md](PIPELINES.md) is the current operational reference.

| Section | Current implementation |
| --- | --- |
| 1 | Shared six-block StatePacket v2, deterministic bounded local diagnostics, boundary/rule context, detailed history and trace metadata. No full-score calls or solver RNG during extraction. |
| 2 | Original actions retained; `guided_move` and scoped `distribution_repair` enabled. `blocker_chain` is implemented as an explicit-region opt-in via `--enable-blocker-chain`, disabled by default. Source/trials/recheck share one allowance. |
| 3 | Deferred; no explanation envelope, additional explanation API calls or XAI study added. |
| 4 | `tools/plot_search_results.py` renders per-instance/seed endpoint, convergence and resource figures with source CSV, manifests and optional diagnostic plots. Historical figures/reports remain separate. |

Verification: `python -m pytest NSGA-III/tests -q`. Plotting dependencies: `python -m pip install -r tools/requirements-plotting.txt`. The generated `output/search_diagnostics/tg_no_timecap/` examples visualize archived runs; they are not new experimental evidence for v2.

Verified on 26 October 2026: 158 offline tests passed, including both controller flows with a mock decision client, fixed-boundary/global-rule checks and inclusive evaluation accounting. A packet preview from an archived `tg-spr18` endpoint made zero full-score requests. Eighteen diagnostic PDFs (the two follow-up seeds plus feasible/mixed-feasibility examples from the earlier campaign) passed rendered alignment, minimum-font and collision checks. The figures use large screen-readable layouts and 300-dpi PNG previews alongside editable PDF/SVG; journal-column/TIFF defaults are not their export target. No new live API experiment was run. Legacy traces without incumbent vectors explicitly skip the optional objective-component figure.

## 1. StatePacket

### 1.1 Original gaps addressed

- The packet reports `search progress`, `candidate regions` and `permitted actions`.
- Constraint IDs and outside-class IDs provide little context about the actual rules and blockers.
- Raw domain counts do not identify useful alternatives or their consequences.
- Compact action history omits source/trial/candidate scores already present in detailed logs.
- Improve the packet by connecting each region to its rules, fixed neighbours, checked alternatives and relevant past results.

### 1.2 Full structure

- Objects group related fields; arrays hold regions, classes, rules and reviews.
- Keep the six top-level blocks aligned with the slides.

```text
global_search_state:
  progress: generation, elapsed_seconds, remaining_seconds
  evaluations: used, limit, remaining
  population: size, feasible_count, feasible_nondominated_count
  quality:
    population_best: hard, internal_total # best current population member by (hard, internal_total)
    incumbent: hard, internal_total # best fully scored candidate seen so far by the same ordering
    best_feasible: hard, internal_total, official_score # best hard=0 member of the current population; null if none
    objective_distribution: time, room, distribution, student
      each objective: min, median, max
    metric_source
  diversity: decision, placement, reference_direction_coverage

recent_quality:
  progress: mode, stagnant_generations, trigger_reasons[]
  history[]: generation, hard, internal_total

candidate_regions[]:
  target: solution_id, region_id, boundary_id
  source_quality: hard, internal_total, weighted_objectives
  active_classes[]: class_id
    placement: time {days, weeks, start, length}, room_id
    domains: time_options, room_options, no_room
    penalties: current_time, current_room
  penalty_summary: hard_contribution, soft_contribution, attribution, unlocalized_penalties[]
  available_actions: operators[]
    movement: movable_class_count, compatible_swap_pairs
    diagnostics[]: operator_id, class_ids[], scope, selection_policy, checks[], blocking_reasons[]
      coverage: candidates_total, checked, complete
      counts: pass, fail, unknown
      witnesses[]: class_ids[], placements[]
        evidence: level, resolved_constraint_ids[], created_constraint_ids[],
                  hard_components_delta, weighted_time_room_delta, full_score, limitations[]

boundary_context[]: region_id
  outside_classes: count, returned_count, truncated
    items[]: class_id, placement {time, room_id}, mutable, relation
  constraints: ids[], detail_count, returned_count, truncated
    details[]: constraint_id, type, required, parameters[], class_ids[]
      satisfaction: status, violation_count
      cost: raw_units, penalty, weight, weighted_cost

recent_outcomes:
  reviews[]: generation, comparable_to_current
    target: solution_id, region_id, boundary_id
    action: operator_id, execution_budget, next_review_interval
    execution: attempts, full_evaluations, scoring_rechecks, elapsed_seconds, stop_reason
    result: decision_status, validation, operator_status, selected,
            source_quality, best_trial_quality, candidate_quality,
            incumbent_before, incumbent_after, incumbent_change

action_limits: operators[]
  execution_budget: min, max
  review_interval: min, max
```

- Remove `schema_version` and `instance` from the model payload retain them in trace metadata with seed/run identity, state fingerprints, provenance and diagnostic costs.
- Preserve source, population-best and incumbent scores separately, including objective vectors when available.
- Regional contributions remain `heuristic_nonadditive`; retain unlocalized penalties.
- Keep original masks/slot units, target IDs and the constraint-ID registry. Join boundary entries by `region_id` within the same snapshot.
- Use `null` for unavailable values. A no-room class has `room_id: null` and `no_room: true`.
- Preserve progress rules: soft-only changes while infeasible do not reset hard stagnation. The current trigger `deadline` means a scheduled generation checkpoint.
- Global limits and per-region availability stay separate. `no_op` requires null targets and budget zero.
- `population_best` describes the current population. `incumbent` is a run-wide archive and may be infeasible until a hard-zero candidate is found. `best_feasible` is the best hard-zero member of the current population; report `official_score` only when available for that candidate.

### 1.3 Rules, blockers and alternatives

- For each relevant constraint, show its type, parameters, required/soft status, participants and known satisfaction.
- Include fixed outside placements that currently conflict or block a tested alternative. Evaluate using all relevant participants, even if the displayed context is truncated; group rules need the full group.
- Check the operators already available:
  - `move`: test a time or room change while holding the other value fixed.
  - `swap`: report domain compatibility separately from conflict-check results.
  - `destroy_repair`: describe its coupled scope and mutable classes from existing data; do not run a repair just to predict its result.
- Check every alternative when the pool is small; otherwise use a bounded deterministic sample. Report the sampling policy, checked/total counts and disjoint `pass`, `fail` and `unknown` results, with blocking reasons and a few examples.
- A `pass` means only that the reported local checks passed. It does not guarantee a better full score; no checked alternatives means no conclusion about whether improvement exists. Examples guide the operator but are not executable placements in the current `ActionPlan`.

### 1.4 Effects and previous actions

- **Local effects:** name the violations resolved or created and exact time/room penalty changes. Do not infer a full score from local checks; student resectioning and other coupled effects may change it.
- Use `after - before` for deltas, so a negative cost change is an improvement. Keep objective components separate to show trade-offs.
- For each review, record the requested action and budget, execution (attempts, evaluations, rechecks, elapsed time, stop reason), and result (source, best trial, candidate, selection, and incumbent before/after/change).
- Reuse existing review events and `OperatorOutcome`; add instrumentation only for missing fields. Preserve decision and validation statuses, distinguish an intentional `no_op` from a fallback, and leave unavailable historical values as `null`.
- Compare a past action only when source placements, fixed boundary and relevant action settings/budget match. The same region ID alone is not enough; otherwise set comparability to `null` or `false` with the reason.

### 1.5 Computation and token limits

- Use one read-only, deterministic extractor for both controller pipelines. Diagnostics must not change solutions, consume solver RNG, alter region ranking or run repairs.
- Keep existing region/class/history limits. Initial per-region caps: 32 outside placements, 12 rule descriptions and two placement examples; check at most 24 one-gene alternatives per class.
- Start with zero full-score evaluations. If previews are added later, cap and charge them, reserve the final operator recheck, and update the remaining budget.
- Track extraction time/work and tokens per role/review. When a cap is reached, report partial coverage. Trim optional detail first; retain validation IDs and hard-blocker evidence. If essential context still exceeds the token limit, reduce the number of regions or skip the review with a recorded reason.
- Report total, returned and truncated counts. `pass`/`fail`/`unknown` candidate counts are disjoint and sum to checked alternatives; blocking reasons may overlap.
- Never expose student identities or present local attendance checks as complete student resectioning.

### 1.6 Old and new packets: seed 17, generation 23

- Sources: [recorded trace, fourth record](<Multi-Agent Controller/results_itc2019/t/17-1/multi/run_n1_p20_g200_r3_s17_654cf84b9432e189/tg-spr18/controller_trace.jsonl>) and matching [tg-spr18.xml](Dataset_ITC2019/tg-spr18.xml).
- The excerpts show region `0:0`, two active classes, six related constraints and a review from generation 18. Unshown regions, classes, rules and reviews are omitted.
- The new packet illustrates the proposed format for the same recorded evidence; it is not output from a new implementation or experiment.

**Old: recorded StatePacket excerpt**

```json
{
  "schema_version": 1,
  "instance": "tg-spr18",
  "global": {
    "generation": 23,
    "remaining_evaluations": 182,
    "best_hard": 15,
    "best_internal_total": 74030,
    "incumbent_hard": 15,
    "incumbent_internal_total": 74030,
    "progress_mode": "hard",
    "stagnant_generations": 3
  },
  "recent_best_quality": [[16, 74030], [15, 74030], [15, 74030], [15, 74030], [15, 74030]],
  "candidate_regions": [
    {
      "solution_id": "0",
      "region_id": "0:0",
      "boundary_id": "8782921793ed",
      "class_ids": ["17", "206", "296", "315", "397"],
      "boundary_class_ids": [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "18", "19", "20", "21", "22",
        "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33"
      ],
      "boundary_class_count": 608,
      "boundary_truncated": true,
      "constraint_ids": [
        "distribution:72", "distribution:73", "room_conflict:17:603", "room_conflict:17:616", "room_conflict:17:618",
        "room_conflict:17:620"
      ],
      "hard_contribution": 11,
      "soft_contribution": 2510,
      "class_summaries": [
        {
          "class_id": "17",
          "current_time": {"days": "1000000", "weeks": "1111111110111111", "start": 162, "length": 34},
          "current_room_id": "13",
          "time_options": 30,
          "room_options": 1,
          "current_time_penalty": 0,
          "current_room_penalty": 0
        },
        {
          "class_id": "206",
          "current_time": {"days": "0000100", "weeks": "1000100000010100", "start": 150, "length": 46},
          "current_room_id": "20",
          "time_options": 1,
          "room_options": 12,
          "current_time_penalty": 0,
          "current_room_penalty": 30
        }
      ],
      "movable_class_count": 5,
      "compatible_swap_pairs": 0,
      "available_operators": ["move", "destroy_repair"]
    }
  ],
  "recent_outcomes": [
    {
      "generation": 18,
      "region_id": "0:2",
      "operator_id": "destroy_repair",
      "validation": "valid_action",
      "outcome": "no_improving_candidate",
      "attempts": 10,
      "evaluations": 10,
      "elapsed_seconds": 0.974,
      "best_hard_change": 0,
      "best_internal_total_change": 0
    }
  ],
  "allowed_actions": {"operators": ["move", "swap", "destroy_repair", "no_op"], "execution_budget": [1, 24], "review_interval": [3, 20]}
}
```

**New: the same evidence in the target format**

```json
{
  "global_search_state": {
    "progress": {"generation": 23},
    "evaluations": {"remaining": 182},
    "quality": {"population_best": {"hard": 15, "internal_total": 74030}, "incumbent": {"hard": 15, "internal_total": 74030}}
  },
  "recent_quality": {
    "progress": {"mode": "hard", "stagnant_generations": 3},
    "history": [
      {"generation": null, "hard": 16, "internal_total": 74030},
      {"generation": null, "hard": 15, "internal_total": 74030},
      {"generation": null, "hard": 15, "internal_total": 74030},
      {"generation": null, "hard": 15, "internal_total": 74030},
      {"generation": null, "hard": 15, "internal_total": 74030}
    ]
  },
  "candidate_regions": [
    {
      "target": {"solution_id": "0", "region_id": "0:0", "boundary_id": "8782921793ed"},
      "source_quality": {"hard": 15, "internal_total": 74030},
      "active_classes": [
        {
          "class_id": "17",
          "placement": {"time": {"days": "1000000", "weeks": "1111111110111111", "start": 162, "length": 34}, "room_id": "13"},
          "domains": {"time_options": 30, "room_options": 1, "no_room": false},
          "penalties": {"current_time": 0, "current_room": 0}
        },
        {
          "class_id": "206",
          "placement": {"time": {"days": "0000100", "weeks": "1000100000010100", "start": 150, "length": 46}, "room_id": "20"},
          "domains": {"time_options": 1, "room_options": 12, "no_room": false},
          "penalties": {"current_time": 0, "current_room": 30}
        }
      ],
      "penalty_summary": {"hard_contribution": 11, "soft_contribution": 2510, "attribution": "heuristic_nonadditive"},
      "available_actions": {
        "operators": ["move", "destroy_repair"],
        "movement": {"movable_class_count": 5, "compatible_swap_pairs": 0},
        "diagnostics": null
      }
    }
  ],
  "boundary_context": [
    {
      "region_id": "0:0",
      "outside_classes": {
        "count": 608,
        "returned_count": 4,
        "truncated": true,
        "items": [
          {
            "class_id": "603",
            "placement": {"time": {"days": "1000000", "weeks": "0000000000000100", "start": 150, "length": 34}, "room_id": "13"},
            "mutable": false,
            "relation": "room_conflict_with_17"
          },
          {
            "class_id": "616",
            "placement": {"time": {"days": "1000000", "weeks": "0000000000000100", "start": 186, "length": 10}, "room_id": "13"},
            "mutable": false,
            "relation": "room_conflict_with_17"
          },
          {
            "class_id": "618",
            "placement": {"time": {"days": "1000000", "weeks": "0001000000000000", "start": 150, "length": 22}, "room_id": "13"},
            "mutable": false,
            "relation": "room_conflict_with_17"
          },
          {
            "class_id": "620",
            "placement": {"time": {"days": "1000000", "weeks": "0000001000000000", "start": 150, "length": 46}, "room_id": "13"},
            "mutable": false,
            "relation": "room_conflict_with_17"
          }
        ]
      },
      "constraints": {
        "ids": [
          "distribution:72", "distribution:73", "room_conflict:17:603", "room_conflict:17:616", "room_conflict:17:618",
          "room_conflict:17:620"
        ],
        "details": [
          {
            "constraint_id": "distribution:72",
            "type": "SameDays",
            "required": true,
            "parameters": [],
            "class_ids": ["397", "600"],
            "satisfaction": null
          },
          {
            "constraint_id": "distribution:73",
            "type": "SameTime",
            "required": true,
            "parameters": [],
            "class_ids": ["397", "600"],
            "satisfaction": null
          }
        ]
      }
    }
  ],
  "recent_outcomes": {
    "reviews": [
      {
        "generation": 18,
        "target": {"solution_id": "0", "region_id": "0:2", "boundary_id": "c361913d17af"},
        "comparable_to_current": false,
        "action": {"operator_id": "destroy_repair", "execution_budget": 10, "next_review_interval": 5},
        "execution": {"attempts": 10, "full_evaluations": 10, "scoring_rechecks": 0, "elapsed_seconds": 0.974, "stop_reason": null},
        "result": {
          "decision_status": "decision_received",
          "validation": "valid_action",
          "operator_status": "no_improving_candidate",
          "selected": false,
          "source_quality": {"hard": 16, "internal_total": 74030},
          "best_trial_quality": {"hard": 17, "internal_total": 74030},
          "candidate_quality": null,
          "incumbent_before": {"hard": 16, "internal_total": 74030},
          "incumbent_after": {"hard": 16, "internal_total": 74030},
          "incumbent_change": {"hard": 0, "internal_total": 0}
        }
      }
    ]
  },
  "action_limits": {
    "operators": ["move", "swap", "destroy_repair", "no_op"],
    "execution_budget": {"min": 1, "max": 24},
    "review_interval": {"min": 3, "max": 20}
  }
}
```

- The XML supplies the unique allowed times and rule definitions; recorded conflicts explain why class 17 has only room 13. Rule satisfaction stays `null` when required outside placements are unavailable.
- Generation-18 execution and scores come from the full historical review. It is not comparable to the current region because its active classes and boundary differ. Missing generation labels and unrecorded alternative checks remain `null`—the example does not invent screening results or gains.

## 2. Actions

### 2.1 Original actions retained

- `move`: randomly changes one time or room value in the selected region.
- `swap`: exchanges compatible placements; unavailable when no compatible pair exists.
- `destroy_repair`: already attempts joint regional CP repair, then a bounded sequential fallback.
- `no_op`: makes no intervention.
- Add actions that use StatePacketv2 evidence or target a specific bottleneck. Another generic joint-repair action would largely overlap with `destroy_repair`.

### 2.2 Implemented additions

**`guided_move`: prioritize promising single-class changes**

- Scope: one time or room value, with every other placement fixed.
- Use the bounded internal shortlist behind StatePacketv2 diagnostics; recheck its source identity before reuse.
- Rank using checked hard-violation changes while infeasible, then directly known time/room costs. After feasibility, prioritize known soft-cost reductions.
- Keep unknown effects explicit. A candidate need not pass every local check to reduce total hard cost; full evaluation decides acceptance.
- Available when the region has a movable class and the bounded proposal pool is nonempty. Missing screening results do not mean that no move exists.
- Evaluate distinct proposals within the requested budget; return the best admissible candidate. The LLM chooses the operator, not a placement.

**`distribution_repair`: focus repair on one violated required rule**

- Scope: movable participants of one supported, currently violated required distribution inside the selected region.
- Code selects the rule deterministically by violation units, then smaller movable group and stable ID. Publish the chosen rule ID in the action diagnostic.
- Prioritize proposals for that rule; keep every other class fixed, including participants outside the selected region. Evaluate group constraints over all participants.
- Reuse the existing CP primitives. Adapt the current distribution helper: it presently selects a global rule and expands from its participants, so it cannot be exposed unchanged.
- Available only when the violated rule has an active participant capable of changing a relevant placement attribute. A related constraint ID alone is insufficient.
- Require improvement in the full hard score; fixing the selected rule while creating more violations elsewhere is not accepted.
- Keep `destroy_repair` for broader regional repair. Targeting soft distributions after feasibility is a separate extension.

### 2.3 Action contract and StatePacketv2

- Keep the existing ActionPlan fields: `solution_id, region_id, boundary_id, operator_id, execution_budget, next_review_interval`.
- Publish new operators in both `action_limits.operators` and each eligible region's `available_actions.operators`.
- Diagnostics explain the proposal scope, selected rule where relevant, checks, coverage and limitations. Code determines internal classes/rules; the LLM cannot invent targets or move outside the published region.
- Apply one final admission rule to the new actions: lower source hard cost while infeasible; after hard zero, preserve feasibility and lower internal total. Population survival and incumbent updates remain separate.
- Share the requested evaluation allowance across all internal stages. Charge source scoring when needed, candidate scoring and logical cache requests under the existing accounting policy; reserve the final recheck. Bound construction/search time as well.
- Record actual attempts, evaluations, rechecks, changed classes, targeted rule, stop reason and source/trial/candidate scores in the action trace; summarize relevant results in StatePacketv2.
- Update operator validation, response enums and both role prompts together during implementation. Existing shared solver repairs remain under their current budgets.

### 2.4 Connection to the recorded case

- In Section 1.6, class 17 has 30 time choices and one room: `guided_move` can prioritize time changes, but cannot give it another room.
- Class 206 has one time and 12 rooms: its single-class alternatives are room changes.
- Rules `distribution:72` and `distribution:73` link class 397 to outside class 600. Class 397 has only one time option: changing its room cannot repair their day/time relationship. These rules are not actionable in the current region, even if violated; a region with a participant able to change time is needed.
- These facts identify possible action scopes. They do not show that either new action would improve the saved timetable.

### 2.5 Optional addition: `blocker_chain`

- Implemented but disabled by default. Enable explicitly with `--enable-blocker-chain` on a controller's direct entrypoint when studying fixed outside blockers.
- Publish a separate capped region containing the focal class and selected blockers before requesting a decision; rebuild its target/boundary IDs and StatePacketv2 context.
- Search a bounded sequence of coordinated moves within that region. Intermediate states may be worse; only a final candidate satisfying the shared admission rule can enter the population.
- Start from existing chain/plateau mechanisms where suitable, after restricting their mutable classes and charging all work.
- Never silently turn a class listed as fixed boundary context into a movable class during execution.

## 3. Explainability

### 3.1 What needs to be explained

- **WHERE:** why this region and which constraints/classes matter.
- **HOW:** why the chosen operator fits the available changes.
- **WHEN:** why this evaluation budget and next review interval were chosen.
- **RESULT:** what code actually changed, accepted or rejected, and whether the incumbent improved.
- StatePacketv2 supplies evidence; the actions define possible interventions. Add a recorded justification connecting that evidence to the choice, then connect the choice to its observed result.
- A plausible justification alone does not establish faithfulness to the model's decision process. Assess factual support, human usefulness and faithfulness separately ([Jacovi and Goldberg, 2020](https://aclanthology.org/2020.acl-main.386/); [Turpin et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/ed3fea9033a80fea1376299fa7863f4a-Paper-Conference.pdf)).

### 3.2 Add a short justification to each role response

```text
role_response:
  decision: existing role-specific decision fields
  explanation:
    summary: brief stated reason
    claims[]: claim_type, value, evidence_refs[]
    expected_effect: objective, direction, scope, limitations[]
    alternative: operator_id, assessment, evidence_refs[]

review_explanation:
  review_id, role, packet_hash, prompt_hash, original_role_response
  evidence_checks[]: claim, status, reason
  execution: validated_action, changed_classes[], scoring_calls, stop_reason
  outcome: source_quality, best_trial_quality, candidate_quality,
           admission, population_selection, incumbent_change, limitations[]
```

- Inspector explains WHERE; Planner explains HOW/WHEN; single-agent explains all three.
- Request at most two summary sentences, three factual claims and one optional alternative. Cite exact packet paths and class/constraint IDs; do not request an internal reasoning transcript.
- Record the response before execution, in the same model call as the decision. An expected effect is an aim or bounded prediction, not a known outcome.
- Put explanation fields in the response envelope, keeping the ActionPlan unchanged. Update strict response schemas and both controller parsers to extract the decision separately.
- Link each role to its exact input: Planner sees StatePacketv2 plus the Inspector diagnosis. A diagnosis alone does not verify a fact.
- Store these records in the existing review trace; preserve the original justification when later checks find errors.

### 3.3 Check the evidence and explain the solver result

- Validate references against that packet. Check supported claim types against actual values: conflicts, domain freedom, operator availability, budgets and previous results.
- Label claims `supported`, `contradicted` or `unverifiable`. An existing ID proves only that the reference resolves; it does not prove the sentence is correct.
- Assess explanations separately from actions. A missing or unsupported explanation is recorded; a separately valid decision still follows the existing action policy.
- Generate the result summary from code records: changed placements, verified violation/objective changes, full evaluations, admission result, population selection and incumbent changes. Missing observations stay unknown.
- Keep local checks, full internal scores and official validation distinct. Record source/candidate objective vectors so a total-score change does not hide tradeoffs.
- Explain survival beyond selected/rejected only when the relevant NSGA-III rank/niche/selection data was recorded. A source improvement does not necessarily survive or improve the incumbent.
- Mention alternatives as unavailable, checked or untested. Claiming that an untested operator would have been worse requires a separately budgeted comparison.
- Use a deterministic report template for the outcome summary; preserve the model's earlier expectation alongside the observed result.

### 3.4 What the recorded case can already explain

- Source: the same seed-17, generation-23 trace used in Section 1.6.
- Inspector returned all 25 published constraint IDs for region `0:0`; that list does not identify its reason for preferring this region.
- Planner returned `destroy_repair`, budget 10 and next review interval 5. Neither role supplied a justification.
- The outcome can be explained from existing records:

```json
{
  "generation": 23,
  "model_justification": null,
  "action": {
    "region_id": "0:0",
    "operator_id": "destroy_repair",
    "execution_budget": 10,
    "next_review_interval": 5
  },
  "observed_result": {
    "source": {"hard": 15, "internal_total": 74030},
    "best_logged_trial": {"hard": 16, "internal_total": 74110},
    "admissible_candidate": null,
    "selected": false,
    "evaluations": 10,
    "incumbent_change": {"hard": 0, "internal_total": 0}
  }
}
```

- Code can state: the best logged trial had higher hard cost, no candidate was admitted, and the incumbent stayed unchanged.
- The reason for choosing the region/operator/budget remains unknown. Do not invent a historical LLM explanation from these results.

### 3.5 Evidence needed for the research question

- **Factual support:** report supported, contradicted and unverifiable claim rates, plus missing-explanation and valid-reference rates. Report coverage rather than excluding uncheckable claims silently.
- **Human usefulness:** compare the same cases presented as StatePacketv2 with raw logs versus the evidence-linked explanation. Measure accuracy and time for identifying the conflict, action scope and outcome; counterbalance cases.
- **Faithfulness:** later run controlled evidence-masking checks on frozen packets while retaining target IDs, action eligibility and budgets. Repeat comparisons to distinguish sensitivity from model variability. This supplies evidence of reliance, not proof of complete internal reasoning.
- **Cost and search quality:** report added tokens/latency and solver outcomes, since requesting explanations can change both cost and decisions.
- Start with evidence-linked justifications and deterministic outcome summaries. Claim improved explainability only against the measured criteria above; the existing trace alone establishes execution visibility, not why the LLM chose an action.

## 4. Result graphs

Implemented by `tools/plot_search_results.py`. Example: `python tools/plot_search_results.py --report solver_checks/tg_no_timecap_comparison.json --output output/search_diagnostics/tg_no_timecap --extra reviews diversity operators objectives`. Each instance/seed gets separate PDF/SVG/PNG, source CSV and a provenance/limitations manifest. Sections below specify interpretation rules, including support for legacy traces.

- Keep the final comparison bar chart; add a convergence figure and a runtime/resource figure for each instance.
- Use the same pipeline colours throughout. Initially show seeds separately; each figure can contain a few related panels.

### 4.1 Final quality: where each pipeline finished

- Compare final hard violations for the three pipelines; annotate internal total and evaluations used.
- When all pipelines are feasible, compare their internal totals. For equal positive hard counts, any soft comparison is descriptive, not a feasibility win.
- Keep hard and soft values separate rather than mixing them into one scalar bar.
- Record the seed, limits, completion status and whether endpoints reached comparable budgets. Label incomplete or unmatched runs explicitly.

### 4.2 Convergence: how quickly quality improved

- **Panel A:** elapsed wall-clock seconds versus incumbent hard violations.
- **Panel B:** logical full evaluations versus the same incumbent hard violations.
- Use step curves for all three pipelines. Wall time includes initialization, solver work, API handling and waiting; evaluations show progress per scoring allowance.
- Read `best_observed_hard` and `best_observed_internal_total` from the anytime trace. Population-best values can fluctuate and are not the preserved incumbent.
- Add a separate feasible-quality panel when hard zero is reached; show incumbent internal total from that point onward.
- Before feasibility, soft cost can rise when hard cost falls. Such a rise does not by itself mean search quality worsened; retain the actual hard/total pair.
- Mark controller reviews and observed accepted interventions to locate changes or plateaus. A nearby improvement alone does not establish that the LLM caused it.
- Start at the first recorded observation, retain its actual elapsed time and stop at the run's final observation. Do not erase initialization time, interpolate an unknown initial score or extend an ended run artificially.

### 4.3 Runtime and API usage: what progress cost

- **Panel A:** total recorded runtime for each pipeline.
- Optionally split runtime into API request handling and all remaining work. The remainder includes solver work, initialization, diagnostics and output handling; it is not a measured pure-solver timer.
- Current `latency_seconds` includes pacing wait. Do not add `pacing_wait_seconds` again or combine overlapping repair/stage timers into a misleading stack.
- **Panel B:** cumulative provider-reported tokens at completed-review checkpoints versus elapsed time. Baseline has zero model usage.
- Use the provider's native total; thinking/cache counts may already be included. Missing usage remains unknown, or explicitly reported as a known lower bound.
- Cumulative tokens only increase or stay flat. To see increasing/decreasing request cost, add a per-review token or latency plot.
- Report API calls and token totals before attempting monetary estimates. Dollar cost requires applicable billing information and dated pricing; token usage alone does not establish actual charges.

### 4.4 Additional plots for a specific pattern

- **Repeated expensive reviews:** tokens and latency per review; distinguish request size from review frequency.
- **Early stagnation:** placement diversity versus evaluations, aligned with the hard-cost curve.
- **Operator behaviour:** review markers coloured by operator and labelled selected/rejected/no improvement; distinguish source improvement from incumbent improvement.
- **Objective tradeoffs:** weighted objective components or feasible Pareto views when the required solution vectors are saved. Population objective minima can belong to different timetables and cannot represent one incumbent vector.
- Add these when they answer a concrete question; the three core figures remain the default.

### 4.5 Data and comparison rules

- Reuse `anytime_trace.jsonl` for quality/time/evaluations/diversity, `controller_trace.jsonl` for actions and role usage, and saved summaries/configuration for endpoints, limits and provenance.
- Match completed reviews to same-generation post-review anytime checkpoints for cumulative usage. Label checkpoint resolution; current logs do not provide exact per-request start/end timestamps.
- Later logging can add explicit review/request timestamps and incumbent objective vectors for finer timing/component plots.
- Use consistent dataset, source snapshot, solver settings and recorded hardware conditions for speed comparisons. Keep time-capped and no-time-cap campaigns separate.
- Compare over shared observed budget ranges; retain visible end markers for early termination. A target not reached within the budget is “not reached”, not zero time.
- With the current small seed set, show individual runs and sample counts. Add aggregate curves and labelled uncertainty only with an appropriate repeated-run design; do not imply significance from two seeds.
- Report whether costs cover a completed run or the whole campaign including retries. Failed attempts and unknown usage must remain visible.
