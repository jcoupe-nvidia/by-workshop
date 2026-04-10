# CLAUDE.md

## Purpose

This repository contains an MVP notebook for a workshop on agentic supply-chain workflows on the NVIDIA stack. It should emphasize:

- multi-turn agent execution
- explicit higher-level skills composed from deterministic tools
- structured tool calling for Nemotron-style models
- fallback parsing for malformed or partially structured outputs
- sequence-sensitive evaluation
- training-oriented patterns aligned with NeMo RL / Megatron agent workflows

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

### Out of scope

- full training pipelines
- production orchestration frameworks
- general-purpose agent platforms
- highly realistic ERP simulations
- benchmark suites with many scenarios
- deployment or inference optimization work

## Design rules

- Favor clarity over completeness.
- Keep the logic explicit and easy to follow.
- Separate scenario data, skills, tools, prompting, parsing, execution, fallback handling, and evaluation.
- Avoid hidden magic, deep abstraction layers, unnecessary async, excessive visualization, large synthetic datasets, and production-grade packaging.
- Prefer short cells, typed Python where practical, explicit comments, and predictable helper functions.

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
   - how this maps to NeMo RL / Megatron workflows

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

Use a compact skill layer that composes deterministic tools into reusable, inspectable behaviors.

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

Use a compact tool library, ideally **7 to 9** tools, such as:

- `get_order(order_id)`
- `get_shipment_status(order_id)`
- `get_inventory(sku, dc_id)`
- `find_alternate_inventory(sku, region)`
- `get_transfer_eta(from_dc, to_dc, sku, qty)`
- `get_supplier_expedite_options(sku, qty)`
- `get_fulfillment_capacity(dc_id, date)`
- `score_recovery_options(options, objective)`
- `recommend_action(context)` or direct final answer generation after scoring

All tools should behave deterministically over small synthetic in-memory tables.

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

## Required NVIDIA stack

These are hard requirements for the notebook design:

- **NeMo Data Designer**
  - for authoring and iterating on synthetic scenario data, traces, repair cases, and evaluation examples
- **NeMo RL**
  - as the main training-oriented reference for exporting trajectories, rewards, and evaluator outputs
- **NVIDIA Megatron**
  - as the model-training and scaling context that the training-oriented discussion should align to alongside NeMo RL
- **NVIDIA ProRL**
  - for framing reward design around tool validity, sequence correctness, recovery quality, and task success
- **OpenCode**
  - as the explicitly named harness for the agent loop, tool registry, execution tracing, and worked examples

Integration should stay narrow, local, and demonstrative. Prefer one concrete export or handoff example rather than building a full training or platform stack inside the notebook.

## Target deployment environment

The target deployment environment is **8 H100 GPUs**.

The notebook may reference this target environment for alignment, sizing context, and downstream planning, but it should not expand into deployment engineering or inference optimization at this stage.

## Optional follow-up

- **NeMo Guardrails**
  - optional to-do after the required libraries are in place
  - most relevant if fallback parsing later expands into policy checks, unsafe tool-call rejection, argument validation, or recovery rules
