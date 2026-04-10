# Phased Development Plan

Build a workshop-ready, notebook-centric MVP for late-order recovery on the NVIDIA stack, using a real model-driven agent loop over deterministic tools and a small set of helper Python modules for clarity and reuse.

## Assumptions
- Primary deliverable: a workshop notebook plus a few helper Python modules.
- The first implementation pass should include a real model-driven loop, not only mocked outputs.
- Non-goals remain those in [CLAUDE.md](CLAUDE.md): no production orchestration, no large benchmark suite, and no deployment engineering.

## Fixed Constraints From The Spec
- Keep the scenario centered on late order recovery for `SO-10482` with machine-checkable sequence dependencies from [CLAUDE.md](CLAUDE.md).
- Target a typical successful run of `5-10` tool calls, `2-4` explicit skills, and `7-9` deterministic tools.
- Explicitly cover the required stack references: NeMo Data Designer, NeMo RL, NVIDIA Megatron, NVIDIA ProRL, and OpenCode.

## Proposed Repo Shape
- [notebooks/late_order_recovery_workshop.ipynb](notebooks/late_order_recovery_workshop.ipynb): the main pedagogical artifact and live demo flow.
- [src/scenario_data.py](src/scenario_data.py): small synthetic in-memory tables for orders, shipments, inventory, transfer lanes, supplier expedite options, capacity, and substitutes.
- [src/tools.py](src/tools.py): deterministic tool implementations and tool registry.
- [src/skills.py](src/skills.py): explicit higher-level skills with allowed tool-use patterns.
- [src/schema.py](src/schema.py): canonical Nemotron-style structured tool-call schema plus validators.
- [src/agent_loop.py](src/agent_loop.py): think/emit/validate/execute/observe loop, model adapter boundary, and trace capture.
- [src/fallbacks.py](src/fallbacks.py): repair/reject parsing logic for malformed or mixed outputs.
- [src/evaluation.py](src/evaluation.py): sequence-sensitive evaluators and simple scoring helpers.
- [README.md](README.md): concise repo entrypoint and how to run the notebook.
- Optional later: [examples/](examples/) or [artifacts/](artifacts/) for saved traces and evaluator outputs if that improves workshop usability.

## Phase 1: Design The Scenario And Skeleton ✅
- Create the notebook outline to mirror the eleven required sections from [CLAUDE.md](CLAUDE.md).
- Lock the single scenario, success criteria, sequence dependencies, and mitigation option set before writing execution code.
- Define the minimal helper-module boundaries so the notebook stays short and presentation-friendly.
- Decide the real-model integration seam early: one thin adapter in `src/agent_loop.py`, with deterministic tool execution remaining local.

## Phase 2: Build Synthetic Data And Deterministic Tools
- Implement the small in-memory scenario dataset in `src/scenario_data.py`.
- Implement `7-9` deterministic tools in `src/tools.py`, matching the spec closely: order lookup, shipment status, source inventory, alternate inventory, transfer ETA, supplier expedite, capacity, scoring, and final recommendation support.
- Make tool outputs strongly structured so traces, validation, and evaluation are easy to inspect in notebook cells.
- Encode sequence dependencies explicitly so invalid call orders can be detected rather than only implied.

## Phase 3: Add Explicit Skills And Tool-Use Policy
- Implement `2-4` explicit skills in `src/skills.py`, likely the four suggested in [CLAUDE.md](CLAUDE.md) unless simplification is needed after prototyping.
- For each skill, define purpose, inputs, outputs, and allowed tool sequences.
- Keep skill transitions visible in the notebook so users can see when the agent is diagnosing, checking primary fulfillment, exploring alternates, and synthesizing a recommendation.

## Phase 4: Implement Real Model Agent Loop
- Build the canonical tool-call schema and validation logic in `src/schema.py`.
- Implement the OpenCode-style execution loop in `src/agent_loop.py`: prompt model, parse output, validate tool call, execute tool, append observation, continue with bounded iterations.
- Keep model integration narrow: one adapter function or class that can call the chosen model endpoint while the rest of the stack remains deterministic and testable.
- Capture full trajectory state so the notebook can replay a successful run and inspect intermediate reasoning safely.

## Phase 5: Add Fallback Parsing And Recovery
- Implement malformed-output handling in `src/fallbacks.py` for malformed JSON, mixed text plus JSON, missing fields, unknown tools, and unsafe arguments.
- Define a clear repair-vs-reject policy and surface that policy in the notebook with one worked repair trajectory.
- Make fallback behavior inspectable and deterministic enough that workshop attendees can see exactly why a call was repaired or rejected.

## Phase 6: Worked Examples And Evaluation
- Add one successful trajectory and one failure-or-repair trajectory inside the notebook.
- Implement evaluators in `src/evaluation.py` for skill selection quality, tool validity, tool accuracy, sequence correctness, task success, recovery quality, and efficiency.
- Ensure at least one evaluator explicitly checks ordered dependencies such as inventory discovery before transfer ETA and candidate option generation before scoring.

## Phase 7: Training-Oriented Wrap-Up
- Add notebook sections showing how traces, repair cases, and evaluator outputs could be authored or iterated in NeMo Data Designer.
- Add one concrete export or handoff example for NeMo RL and frame reward design using NVIDIA ProRL concepts.
- Connect the trajectory and reward design discussion to NVIDIA Megatron and the target environment of `8 H100 GPUs`, but keep it conceptual rather than operational.
- Capture NeMo Guardrails as a clearly labeled optional follow-up rather than core MVP scope.

## Implementation Order Recommendation
1. Notebook skeleton and scenario specification.
2. Synthetic data and deterministic tools.
3. Skills and sequence rules.
4. Structured schema, validator, and fallback parser.
5. Real model adapter and agent loop.
6. Worked traces and evaluation.
7. Training-oriented discussion and repo polish.

## Key Risks To Manage Early
- A real model may produce inconsistent tool-call formatting, so the schema and fallback layer should be implemented before deep prompt tuning.
- If the notebook owns too much logic, the live demo will become hard to follow; push reusable mechanics into the small helper modules.
- If tools are too realistic or datasets too large, the workshop loses clarity; keep the data synthetic, small, and deterministic.
- If evaluation is added too late, it will be harder to prove sequence sensitivity; design evaluator hooks while building the tool registry and agent loop.

## First Execution Milestone
- Reach a notebook state where one real model run can diagnose `SO-10482`, call deterministic tools in a valid sequence, compare at least two mitigation options, and produce a final recommendation with a trace that can be evaluated afterward.
