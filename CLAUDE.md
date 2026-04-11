# CLAUDE.md

## Purpose

This repository contains an MVP notebook for a workshop on agentic supply-chain workflows on the NVIDIA stack. It should emphasize:

- multi-turn agent execution
- explicit higher-level skills composed from deterministic tools
- structured tool calling for Nemotron-style models
- fallback parsing for malformed or partially structured outputs
- sequence-sensitive evaluation
- clear ownership boundaries across `runtime/`, `envs/`, `rollouts/`, `training/`, and `eval/`

This is a pedagogical artifact, not a production system. The goal is to show concrete patterns, tradeoffs, and evaluation methods in a form that is easy to present live.

## Local model access reference

When this repo needs to call the locally deployed LLM, use `documents/llm-access.md` as the source of truth for the endpoint, model id, smoke test, and cache mapping.

## Core scenario

Build one end-to-end example around **late order recovery for a constrained customer shipment**.

Example task:

> Customer order `SO-10482` for `1,200` units is at risk of missing the committed delivery date. Determine whether the order can still be fulfilled on time. If not, recommend the best mitigation action.

The agent should evaluate options such as:

- fulfill from original source DC
- transfer from alternate DC
- expedite from supplier
- partially fulfill
- recommend a substitute SKU if enabled

## Scope

### In scope

- a single rich scenario rather than many shallow ones
- explicit skill selection and tool sequencing
- small synthetic datasets that are easy to inspect in notebook cells
- deterministic tools and reproducible evaluation
- at least one malformed-output or recovery example
- one GRPO training run

### Out of scope

- production orchestration frameworks
- general-purpose agent platforms
- highly realistic ERP simulations
- benchmark suites with many scenarios
- deployment or inference optimization work

## Design rules

- Favor clarity over completeness.
- Keep the logic explicit and easy to follow.
- Separate scenario data, skills, tools, prompting, parsing, execution, fallback handling, and evaluation.
- Keep the environment explicit: state, validity, transitions, preconditions, and reward-relevant facts should live in repo code.
- Keep scenario-specific contracts repo-owned even when execution surfaces map to NAT, NeMo Gym, and `openpipe-art`.
- Keep malformed calls, repairs, and rejects explicit in canonical traces rather than silently hiding them.
- Avoid hidden magic, deep abstraction layers, unnecessary async, excessive visualization, large synthetic datasets, and production-grade packaging.
- Prefer short cells, typed Python where practical, explicit comments, and predictable helper functions.
- Keep the notebook as a consumer and teaching surface, not the source of truth for architecture-critical logic.

## RL architecture

Keep the RL-facing code split across `runtime/`, `envs/`, `rollouts/`, `training/`, and `eval/`.

Use `documents/RL_ARCHITECTURE.md` as the source of truth for layer responsibilities and best practices. Keep scenario contracts repo-owned.

## Notebook requirements

The notebook should include these sections:

1. **Introduction**
   - what the notebook demonstrates
   - why supply chain is a good domain for agentic workflows
   - why sequence correctness matters
2. **Scenario definition**
   - business problem
   - target task
   - success criteria
3. **Synthetic data model**
   - orders, shipment status, inventory, transfer lanes, supplier options, fulfillment capacity, and optional substitute SKUs
4. **Skill definitions**
   - purpose, inputs, outputs, and allowed tool-usage patterns
   - the distinction between higher-level skills and low-level tools
5. **Tool definitions**
   - name, inputs, outputs, execution logic, and call-order dependencies
6. **Structured tool-call schema**
   - canonical representation plus valid and invalid examples
7. **Agent loop**
   - think -> emit tool call -> validate -> execute -> observe -> continue
   - skill selection or skill transition when needed
   - bounded iterations and stop conditions
8. **Fallback parsing**
   - malformed JSON, mixed text plus structured call, missing fields, unknown tools, unsafe arguments, and repair vs reject policy
9. **Worked examples**
   - one successful trajectory
   - one failure or repair trajectory
10. **Evaluation**
   - skill selection quality, tool validity, tool accuracy, sequence correctness, task success, recovery quality, and efficiency
11. **Training-oriented discussion**
   - what should be supervised at the skill and trajectory level
   - how trajectories could be scored
   - how sequence-sensitive rewards could be defined
   - how this maps to `openpipe-art`
   - what earlier trainer-facing, rollout-shaping, and scale-out systems assumptions existed and why they were narrowed or removed

## MVP behavior constraints

- A typical successful run should require **5 to 10 tool calls**.
- The notebook should use **2 to 4 explicit higher-level skills**.
- The sequence must matter and be machine-checkable.

Representative dependencies include:

- order lookup before shipment analysis
- shipment analysis before mitigation
- source inventory check before alternate sourcing
- alternate inventory discovery before transfer ETA estimation
- candidate mitigation options before scoring and recommendation

## Skills and tools

Use **NeMo Agent Toolkit (NAT)** as the runtime-facing tool and skill architecture for interactive and demo execution. Keep the notebook's skill layer compact, explicit, and inspectable. See `documents/RL_ARCHITECTURE.md` for the broader layer split.

The runtime skill architecture should expose these canonical interfaces:

- `list_skills` for discovery: skill name, description, tags, and discovered files.
- `search_skills` for metadata search only: name, description, tags, declared assets, and filenames.
- `get_skill` as the detailed read path, loading either the full `SKILL.md` body or a specific sidecar file by relative path.
- `run_skill_command` for executing scripts present in the skills folder.

These NAT-facing interfaces should own skill discovery, inspection, and execution surfaces. Keep them separate from the deterministic business tools used by the supply-chain scenario and from NeMo Gym-owned training rollout execution.

Suggested skills:

- `diagnose_order_risk(order_id)`
- `assess_primary_fulfillment(order_id)`
- `evaluate_alternate_recovery_paths(order_id)`
- `synthesize_recommendation(order_id)`

Each skill should:

- have a clear purpose and boundary
- invoke one or more tools in an inspectable sequence
- return structured intermediate outputs when useful
- support evaluation of both skill selection and tool-use quality
- live in a directory-backed skill package with `SKILL.md` plus optional sidecars or scripts when helpful

Use a compact deterministic business-tool library, ideally **7 to 9** tools, such as:

- `get_order(order_id)`
- `get_shipment_status(order_id)`
- `get_inventory(sku, dc_id)`
- `find_alternate_inventory(sku, region)`
- `get_transfer_eta(from_dc, to_dc, sku, qty)`
- `get_supplier_expedite_options(sku, qty)`
- `get_fulfillment_capacity(dc_id, date)`
- `score_recovery_options(options, objective)`
- `recommend_action(context)` or direct final answer generation after scoring

All business tools should behave deterministically over small synthetic in-memory tables.

## Structured tool-call schema

The notebook must define a canonical Nemotron-style tool-call format.

```json
{
  "thought": "optional short reasoning summary or omitted entirely",
  "tool_call": {
    "name": "get_inventory",
    "arguments": {
      "sku": "SKU-100",
      "dc_id": "DC-WEST-01"
    }
  }
}
```

## Required stack

These are the active requirements for the notebook design:

- **NeMo Agent Toolkit (NAT)** for runtime-facing tool and skill execution
- **NeMo Gym** for training-time environments and rollout collection
- **`openpipe-art`** for trainer-facing datasets, rewards, and handoff

Integration should stay narrow, local, and demonstrative. See `documents/RL_ARCHITECTURE.md` for the responsibility split around this stack.

## Target deployment environment

The notebook may still mention the earlier **8 H100** target environment as historical scaling context, but it should not depend on scale-out-systems-specific deployment assumptions or expand into deployment engineering or inference optimization at this stage.


