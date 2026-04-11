# REFACTOR.md

## Objective

Refactor this repository so it cleanly leverages:

- **NeMo Agent Toolkit (NAT)** for agent runtime orchestration
- **NVIDIA ProRL** for scalable multi-turn rollout generation
- **NeMo RL** for policy optimization and post-training
- **NVIDIA Megatron / Megatron Bridge** for large-scale training systems, distributed execution, checkpoint interoperability, and model packaging

The primary goal is to eliminate overlap in responsibilities across runtime, environment logic, rollout collection, evaluation, training semantics, and large-scale training systems, while following best practices for **multi-turn RL training for agent tool-calling agents**.

This refactor should preserve the workshop/demo value of the repository, but reshape it into a cleaner architecture that can support **real scalable training jobs** later.

---

## Core Responsibility Split

Use this role split consistently and enforce it at the module boundary level.

### 1. NeMo Agent Toolkit (NAT)

Owns:

- agent runtime behavior
- tool definitions
- tool schemas
- workflow/subworkflow composition
- prompt/runtime policy
- observability and tracing hooks

NAT should answer:

**"How should the agent act on this turn?"**

NAT must **not** own:

- rollout scheduling
- reward design
- RL trainer logic
- distributed training topology
- checkpoint conversion pipelines
- hardware execution strategy

### 2. NVIDIA ProRL

Owns:

- multi-turn rollout generation
- episode lifecycle
- concurrency-facing rollout orchestration
- turn alignment and trace capture
- failure/retry representation for data collection
- rollout dataset production

ProRL should answer:

**"How do we generate many high-quality multi-turn episodes?"**

ProRL must **not** own:

- tool semantics
- workflow/business logic
- training algorithms
- model parallel configuration
- checkpoint packaging strategy

### 3. NeMo RL

Owns:

- reward-driven post-training logic
- RL/SFT dataset views over trajectories
- algorithm selection
- trainer-facing reward consumption
- curriculum staging
- checkpoint evaluation logic

NeMo RL should answer:

**"How do we improve the model from those episodes?"**

NeMo RL must **not** own:

- tool schemas
- runtime prompt policy
- rollout serving/concurrency logic
- cluster launch topology
- Megatron parallelism strategy

### 4. NVIDIA Megatron / Megatron Bridge

Owns:

- distributed training systems concerns
- model construction/import/export where Megatron-specific
- tensor/pipeline/context/expert parallel configuration
- scalable launch recipes
- checkpoint interoperability and conversion boundaries
- performance-oriented training system configuration
- hardware-targeted execution profiles

Megatron should answer:

**"How do we execute training efficiently and correctly at scale?"**

Megatron must **not** own:

- tool-calling runtime behavior
- episode definitions
- reward semantics
- rollout logic
- business-task environment transitions
- offline evaluation policy

### 5. The Repository’s Own Environment Layer

Owns:

- task state
- deterministic transition logic
- action validity
- terminal conditions
- reward decomposition inputs
- sequence-sensitive task semantics

The environment should answer:

**"What is the task state, what actions are valid now, and what happened after an action?"**

This layer must remain independent from runtime orchestration, rollout infrastructure, trainer logic, and distributed systems concerns.

---

## Design Principles

### 1. One layer per concern

Use the following split consistently:

- **NAT**: single-episode runtime orchestration
- **Environment layer**: task state, validity, transitions, task semantics
- **ProRL**: scalable multi-turn rollout collection
- **NeMo RL**: training semantics and post-training logic
- **Megatron**: large-scale training systems and distributed execution
- **Eval layer**: offline benchmarking and reporting

### 2. No overlap in responsibility

Avoid these anti-patterns:

- runtime code deciding training/export format details
- training code redefining tool schemas or runtime prompt policy
- rollout code implementing task/business logic
- evaluation code duplicating environment transition rules
- Megatron-facing system code deciding rewards or rollout structure
- notebook cells acting as the system of record for architecture

### 3. Treat training semantics and training systems as different things

This repo should distinguish clearly between:

- **training semantics**: what to optimize, what rewards to use, what datasets to build, what curriculum to run
- **training systems**: how to execute the training efficiently on large hardware, how to configure parallelism, how to manage checkpoint boundaries

Rule:

- **NeMo RL owns training semantics**
- **Megatron owns training systems**

### 4. Treat the environment as first-class

The multi-turn order recovery scenario should be represented as an explicit environment/state machine, not as logic scattered across:

- notebook cells
- evaluation functions
- agent loop code
- fallback handlers

### 5. Preserve deterministic tools

The current deterministic tool design is a strength for RL and should remain in place. Tool execution semantics should stay deterministic and machine-checkable.

### 6. Design for future large jobs

This repo should be refactored so it is still lightweight enough for a workshop, but structurally capable of growing into:

- distributed SFT jobs
- distributed RL/post-training jobs
- checkpoint conversion flows
- cluster launch profiles
- reproducible training recipes for large hardware

---

## Target Architecture

Refactor the repo into six major packages plus notebook/demo glue.

### A. `runtime/`
Purpose: NAT-facing agent runtime package

Owns:

- tool definitions
- tool schemas
- workflow/subworkflow composition
- prompt/runtime policy
- agent execution adapter
- turn-level trace emission
- observability hooks

This layer should implement the agent as it runs a single episode.

### B. `envs/`
Purpose: explicit RL/task environment package

Owns:

- episode state
- transition logic
- allowed action logic / tool constraints
- terminal conditions
- reward decomposition inputs
- sequence-validity rules
- task-specific validation logic

This layer should define the task formally, independent of the runtime.

### C. `rollouts/`
Purpose: ProRL-facing rollout collection package

Owns:

- episode runner integration
- rollout orchestration
- concurrency-facing adapters
- trace serialization
- failure/timeouts/retry representation
- dataset production for training

This layer should collect many episodes and preserve exact turn structure.

### D. `training/`
Purpose: NeMo RL-facing training semantics package

Owns:

- training dataset adapters
- reward aggregation for trainer consumption
- algorithm selection hooks
- curriculum staging
- experiment definitions
- RL/SFT consumption views over canonical trajectories

This layer should define **what** gets optimized, not how distributed execution is performed.

### E. `systems/`
Purpose: Megatron-facing scalable training systems package

Owns:

- model system configuration
- distributed execution profiles
- parallelism settings
- checkpoint conversion/interoperability
- launch recipes
- hardware-targeted configuration presets
- performance-oriented training system utilities

This layer should define **how** training runs at scale.

### F. `eval/`
Purpose: offline evaluation and reporting

Owns:

- offline metrics
- benchmark summaries
- trajectory analysis
- regression reports

This should be separate from `envs/` so that:
- `envs/` owns training-relevant task semantics and validation
- `eval/` owns human-facing reports and offline metrics

---

## Proposed Repository Structure

Refactor toward this structure:

```text
.
├── notebooks/
│   └── late_order_recovery_workshop.ipynb
├── documents/
│   └── llm-access.md
├── src/
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── tools.py
│   │   ├── workflows.py
│   │   ├── schemas.py
│   │   ├── prompts.py
│   │   └── tracing.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── late_order_env.py
│   │   ├── state.py
│   │   ├── transitions.py
│   │   ├── rewards.py
│   │   └── validators.py
│   ├── rollouts/
│   │   ├── __init__.py
│   │   ├── prorl_adapter.py
│   │   ├── episode_runner.py
│   │   ├── serializers.py
│   │   └── trace_types.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── nemo_rl_adapter.py
│   │   ├── datasets.py
│   │   ├── curriculum.py
│   │   ├── experiments.py
│   │   └── reward_views.py
│   ├── systems/
│   │   ├── __init__.py
│   │   ├── megatron_bridge.py
│   │   ├── checkpointing.py
│   │   ├── parallelism.py
│   │   ├── launch_configs.py
│   │   └── model_recipes.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── reports.py
│   │   └── regression.py
│   ├── scenario_data.py
│   └── main.py
├── artifacts/
└── REFACTOR.md
```

This does **not** require all logic to be rewritten immediately, but the direction should be clear and enforced.

---

## Required File-Level Refactor Mapping

Translate the existing files into the new structure as follows.

### Existing: `src/tools.py`
Refactor to: `src/runtime/tools.py`

Requirements:

- keep deterministic tool implementations
- make tool inputs/outputs explicitly schema-driven
- keep business logic deterministic
- remove any training/export concerns
- remove any evaluation-side reward logic from this module

### Existing: `src/skills.py`
Refactor to: `src/runtime/workflows.py`

Requirements:

- rename conceptually from "skills" to "workflows" or "policies"
- represent allowed tool-use patterns and workflow decomposition
- do not use this file to encode trainer-specific behavior
- do not use this file to encode rollout infrastructure behavior
- this file should describe runtime behavior only

### Existing: `src/schema.py`
Refactor to: `src/runtime/schemas.py` and `src/envs/validators.py`

Requirements:

- keep runtime action/observation schemas in `runtime/schemas.py`
- move task/environment-specific validation rules into `envs/validators.py`
- ensure there is one canonical schema definition for each action/event type
- avoid duplicate schema definitions in rollout/training/system code

### Existing: `src/agent_loop.py`
Refactor to: `src/runtime/agent.py` plus `src/rollouts/episode_runner.py`

Requirements:

- shrink the current monolithic loop
- `runtime/agent.py` should focus on single-episode agent execution
- `rollouts/episode_runner.py` should handle episode running and trace capture
- separate "agent decides" from "rollout system records and batches"

### Existing: `src/fallbacks.py`
Refactor to: `src/runtime/agent.py`, `src/envs/transitions.py`, or `src/rollouts/serializers.py` depending on concern

Requirements:

- keep runtime robustness where needed
- do **not** silently hide malformed tool calls during RL rollouts
- represent malformed call / repair / reject as explicit trace events
- ensure these events can be used for reward penalties and analysis

### Existing: `src/evaluation.py`
Refactor to: `src/envs/rewards.py` and `src/eval/metrics.py`

Requirements:

- move training-relevant reward logic inputs into `envs/rewards.py`
- move offline benchmarking/reporting logic into `eval/metrics.py`
- keep sequence-sensitive scoring, but separate:
  - step-level training rewards
  - offline evaluation summaries

### Existing: `src/training_export.py`
Refactor to:
- `src/rollouts/serializers.py`
- `src/training/datasets.py`
- `src/training/nemo_rl_adapter.py`
- `src/training/experiments.py`
- `src/systems/megatron_bridge.py`
- `src/systems/checkpointing.py`
- `src/systems/launch_configs.py`

Requirements:

- eliminate the "catch-all export file" pattern
- define a canonical trajectory format first
- build training dataset views from that canonical format
- keep NeMo RL integration isolated from runtime code
- keep ProRL rollout serialization isolated from trainer code
- keep Megatron/checkpoint/launch concerns isolated from RL semantics
- move any conceptual Megatron config sketch into the `systems/` package

---

## Canonical Data Model

Introduce a canonical episode/trajectory format that every layer uses.

### Required episode structure

Each episode should preserve:

- task input
- initial environment state metadata
- each turn in order
- model action at each turn
- tool call payload at each turn
- tool observation/result at each turn
- any validation/fallback/repair events
- step-level rewards or reward annotations
- terminal outcome
- summary metrics

### Required turn/event schema

Define explicit event types such as:

- `user_task`
- `model_thought` (optional; only if preserved intentionally)
- `tool_call`
- `tool_result`
- `tool_validation_error`
- `tool_repair_attempt`
- `tool_reject`
- `agent_message`
- `terminal_outcome`

Claude should create strongly typed Python dataclasses or Pydantic models for these.

### Critical rule

The repo must stop relying on unstructured trace text as the primary training artifact.

Structured event records should become the source of truth.

### Boundary rule

- `runtime/` emits canonical events
- `rollouts/` serializes canonical events
- `training/` consumes canonical events as datasets/reward views
- `systems/` never redefines the event schema

---

## Environment Requirements

Create an explicit task environment for the late-order recovery scenario.

### The environment must own:

- what information is known vs unknown to the agent
- which subgoals are completed
- which actions/tools are valid at a given point
- whether preconditions for a tool call are satisfied
- what the resulting observation is
- whether the episode is terminal
- what task-level reward inputs or penalties apply

### Environment state should include at least:

- order id
- source DC status
- alternate DC feasibility
- supplier expedite feasibility
- partial fulfillment feasibility
- substitute SKU viability
- tool calls already made
- current recommendation candidate
- failure flags / invalid action counters
- terminal status

### Environment transitions should be deterministic

Do not embed transition semantics in notebook cells or in ad hoc evaluation logic.

### Environment ownership rule

The environment defines what happened.
It does **not** decide:
- how the model was trained
- how rollouts are scheduled
- how distributed execution is configured

---

## Reward Design Requirements

Implement dense, sequence-aware rewards for multi-turn RL.

### Reward dimensions

At minimum, support these step-level signals:

- valid structured tool call
- correct tool choice for current state
- correct argument extraction
- dependency satisfaction before calling a tool
- no redundant tool calls
- progress toward resolving the order-risk decision
- correct final recommendation
- concise completion without unnecessary steps

### Penalties

At minimum, include penalties for:

- malformed tool call
- invalid schema
- calling a tool before its prerequisites are satisfied
- repeated/redundant calls
- looping behavior
- hallucinated unsupported conclusions
- overlong episodes
- silent fallback reliance

### Important rule

Do not reward only final success.

The training design must reward the **decision process** turn by turn.

### Ownership rule

- `envs/` provides reward-relevant task signals and transition facts
- `training/` builds trainer-facing reward views
- `systems/` must not define or alter reward semantics

---

## Runtime Requirements

The runtime layer should be rewritten to resemble a NAT-friendly structure.

### Requirements

- isolate model interaction from environment logic
- isolate tool registry from rollout logic
- make the agent loop thin and composable
- centralize prompt/runtime policy
- emit structured traces natively
- avoid parsing mixed free-form text when structured outputs are expected

### Specific guidance

- prefer one canonical structured action format
- minimize fallback parsing where possible
- keep repair logic explicit and auditable
- make it easy to swap the backend model endpoint without changing agent semantics

### Ownership rule

The runtime may call tools and emit events.
It must not:
- schedule large rollout jobs
- construct RL datasets
- choose distributed training topology
- own checkpoint conversion logic

---

## ProRL Rollout Requirements

The ProRL-facing layer should treat multi-turn rollouts as first-class.

### Requirements

- create a rollout adapter that can run many episodes
- preserve exact turn alignment
- preserve failure and repair events
- serialize episodes in a stable format
- keep rollout collection independent from model training
- support future concurrency and batching without changing the episode schema

### Important rule

Do not make rollout collection depend on notebook execution order.

### Ownership rule

The rollout layer may orchestrate episodes at scale.
It must not:
- redefine tool schemas
- redefine the task environment
- define reward semantics
- define Megatron execution settings

---

## NeMo RL Training Requirements

The NeMo RL-facing layer should consume structured trajectories, not notebook-generated ad hoc exports.

### Requirements

- create dataset adapters over canonical trajectories
- support SFT-style successful-trace extraction
- support RL-style stepwise reward extraction
- support future curriculum stages
- isolate trainer configuration from runtime logic
- keep reward views explicit and inspectable

### Training stages to support

Design the code so it can support this staged progression:

1. **SFT on successful trajectories**
2. **Short-horizon RL with dense rewards**
3. **Full multi-turn RL with sequence-aware rewards**
4. **Robustness curriculum with malformed calls and dead ends**

### Ownership rule

The training layer decides:
- what data views exist
- what reward views exist
- what experiments are run

It must not decide:
- rollout service orchestration
- runtime tool interfaces
- Megatron parallel configuration details

---

## Megatron / Scalable Training Systems Requirements

The Megatron-facing systems layer should make the repo a real scalable training starter.

### Requirements

- create a dedicated `systems/` package
- isolate Megatron/Megatron Bridge integration from RL semantics
- support checkpoint import/export boundaries
- support scalable launch profiles
- support future tensor/pipeline/context parallel settings
- support hardware-targeted configs such as local dev vs multi-GPU workstation vs cluster jobs
- define stable configuration surfaces for large-job execution

### Suggested modules

- `systems/megatron_bridge.py`
  - Megatron-specific model/provider construction
  - import/export boundaries
  - checkpoint interoperability helpers

- `systems/parallelism.py`
  - tensor parallel
  - pipeline parallel
  - context parallel
  - optional expert parallel surfaces later

- `systems/launch_configs.py`
  - local dev profile
  - single-node multi-GPU profile
  - cluster/H100-scale profile

- `systems/checkpointing.py`
  - save/load boundaries
  - conversion helpers
  - artifact naming and checkpoint organization

- `systems/model_recipes.py`
  - model-size presets
  - execution-ready training system recipes

### Important rule

Megatron exists here as the **training systems layer**, not as the owner of RL logic.

### Ownership rule

The systems layer may decide:
- how a job runs at scale
- how parallelism is configured
- how checkpoints are packaged or converted

It must not decide:
- tool-calling policies
- rollout semantics
- reward functions
- task environment logic
- offline benchmark definitions

---

## Notebook Requirements

The notebook should remain useful, but should no longer contain core system logic.

### The notebook may do:

- demo setup
- scenario walkthrough
- example episode execution
- evaluation display
- artifact visualization

### The notebook must not do:

- define canonical schemas
- define transition rules
- define reward logic inline
- act as the only way to run the system
- contain the only export path for training artifacts
- contain the only large-job config path

---

## Implementation Plan

Claude should execute the refactor in this order.

### Phase 1: Establish canonical interfaces

1. create new package structure under `src/`
2. introduce typed event/trajectory models
3. introduce environment state and transition skeleton
4. split runtime schemas from environment validation
5. establish boundary between training semantics and training systems

### Phase 2: Move runtime logic

1. move tools into `runtime/tools.py`
2. move skills/policies into `runtime/workflows.py`
3. refactor the agent loop into `runtime/agent.py`
4. make trace emission structured and canonical

### Phase 3: Formalize environment and rewards

1. create `envs/state.py`
2. create `envs/transitions.py`
3. create `envs/rewards.py`
4. move sequence-sensitive logic out of generic evaluation code

### Phase 4: Build rollout layer

1. create `rollouts/trace_types.py`
2. create `rollouts/serializers.py`
3. create `rollouts/episode_runner.py`
4. add `rollouts/prorl_adapter.py`

### Phase 5: Build training semantics layer

1. create `training/datasets.py`
2. create `training/reward_views.py`
3. create `training/nemo_rl_adapter.py`
4. create `training/experiments.py`

### Phase 6: Build scalable training systems layer

1. create `systems/megatron_bridge.py`
2. create `systems/parallelism.py`
3. create `systems/checkpointing.py`
4. create `systems/launch_configs.py`
5. create `systems/model_recipes.py`

### Phase 7: Rebuild offline evaluation

1. create `eval/metrics.py`
2. create `eval/reports.py`
3. ensure offline metrics consume canonical traces

### Phase 8: Update notebook and entrypoints

1. update notebook imports
2. add a simple `src/main.py` or CLI entrypoint
3. ensure notebook is a consumer of the library, not the source of truth

---

## Concrete Refactor Constraints

Claude should follow these constraints during implementation.

### Code quality

- prefer small typed modules
- use dataclasses or Pydantic models for structured records
- keep functions focused and side-effect-light
- document public interfaces
- avoid hidden global state

### Scalability

- do not hard-code notebook-only assumptions
- do not bury config in ad hoc cells
- make launch surfaces config-driven
- keep large-job config separate from experiment semantics
- support future distributed execution without changing core task interfaces

### Backward compatibility

- preserve the workshop scenario behavior
- preserve the existing late-order recovery flow
- preserve deterministic tool semantics
- preserve the ability to run a demo end-to-end locally

### Refactor discipline

- do not do unnecessary feature expansion
- do not introduce speculative abstractions without immediate use
- do not rewrite scenario data unless needed for environment formalization
- do not remove the notebook; demote it to a consumer

---

## Deliverables

Claude should produce the following as part of the refactor.

### Required code deliverables

- new package structure under `src/`
- canonical trajectory/event types
- explicit environment state transitions
- split runtime vs rollout vs training vs systems layers
- updated imports and entrypoints
- updated notebook wiring
- initial scalable training systems scaffolding for Megatron

### Required documentation deliverables

- updated `README.md` describing the new architecture
- module docstrings explaining responsibilities
- migration notes summarizing what moved where
- brief note clarifying NeMo RL vs Megatron responsibilities

### Optional but encouraged

- a small CLI entrypoint for running one episode
- a simple example of serialized trajectory output
- a short architecture diagram in markdown
- one example large-job config profile

---

## Acceptance Criteria

The refactor is complete when all of the following are true:

1. **Clear ownership boundaries**
   - runtime, environment, rollouts, evaluation, training semantics, and training systems each have distinct responsibilities

2. **No catch-all training export file**
   - training/export/system logic is split cleanly by concern

3. **Structured traces are canonical**
   - all major flows use typed structured event records

4. **Environment is explicit**
   - transition rules and task semantics are not hidden in notebook/evaluation code

5. **Notebook is demoted**
   - notebook consumes the library rather than defining core behavior

6. **Training semantics and training systems are separate**
   - NeMo RL-facing logic and Megatron-facing logic do not overlap

7. **Multi-turn RL readiness improves**
   - the codebase is easier to use for:
     - successful trajectory collection
     - failed trajectory collection
     - reward shaping
     - rollout batching
     - NeMo RL ingestion
     - ProRL integration

8. **Scalable training readiness improves**
   - the codebase has a real starter path for:
     - larger distributed jobs
     - checkpoint conversion boundaries
     - launch profiles
     - Megatron integration

9. **Existing scenario still works**
   - the late-order recovery demo can still run end-to-end

---

## Direct Instruction to Claude

Please refactor the repository according to this document.

Priorities:

1. maximize separation of concerns
2. make the environment explicit
3. make structured trajectories canonical
4. prepare the codebase for multi-turn RL training of tool-calling agents
5. prepare the codebase for real scalable training jobs
6. preserve the current workshop scenario and demo value

When choices are ambiguous, prefer the option that most clearly separates:

- runtime orchestration
- environment/task logic
- rollout collection
- training semantics
- training systems
- offline evaluation

Do not keep architecture-critical logic trapped in the notebook.
