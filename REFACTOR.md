# REFACTOR.md

## Objective

Refactor the repository so it cleanly separates:

- **NAT** for interactive single-episode agent runtime behavior
- **NeMo Gym** for training-time environments and rollout collection
- **repo-owned canonical contracts and traces** shared across integrations
- **`openpipe-art`** for training-oriented datasets, rewards, and post-training flows

The result should preserve the workshop/demo value of the repo while making it structurally ready for real multi-turn training and export workflows.

---

## Responsibility Split

Use these boundaries consistently.

| Layer | Owns | Does not own |
| --- | --- | --- |
| `runtime/` with NAT | tool definitions, tool schemas, prompt/runtime policy, skill discovery/loading/execution, interactive single-episode agent behavior, trace emission | training rollout scheduling, reward design, trainer logic, deployment topology |
| `envs/` aligned to NeMo Gym | task state, deterministic transitions, action validity, terminal conditions, reward-relevant task facts, verification-ready environment semantics | runtime orchestration, rollout infrastructure policy, trainer logic |
| `rollouts/` | canonical multi-turn trace capture, serialization, retry/failure representation, adapters between repo contracts and NeMo Gym rollout collection | tool semantics, task logic, training algorithms, ownership of a competing rollout framework |
| `training/` with `openpipe-art` | dataset adapters, reward views, curriculum, experiment definitions, trainer-facing post-training logic | runtime policy, tool schemas, rollout orchestration, deployment details |
| `eval/` | offline metrics, reports, regression analysis over canonical traces, NeMo Gym facts, and `openpipe-art` artifacts | environment transitions, runtime policy, trainer behavior |

Historical references to older rollout or trainer-facing stacks may remain in docs, but they are not the active implementation target.

---

## Design Principles

### 1. One layer per concern

- `runtime/`: NAT-aligned interactive single-episode orchestration
- `envs/`: task semantics, validity, transitions, and NeMo Gym-aligned verification surfaces
- `rollouts/`: canonical trace capture, serialization, and NeMo Gym adapters
- `training/`: trainer-facing datasets and reward views
- `eval/`: repo-owned offline benchmarking and reporting across runtime, environment, and training artifacts

### 2. No overlap in ownership

Avoid:

- runtime code deciding training/export formats
- training code redefining tool schemas or prompt policy
- rollout code implementing business logic
- evaluation code duplicating environment transitions
- notebook cells acting as the system of record

### 3. Treat the environment as first-class

Represent late-order recovery as an explicit environment/state machine, not logic scattered across notebook cells, evaluation helpers, agent-loop code, or fallback handlers.

### 4. Preserve deterministic tools

Keep business tools deterministic and machine-checkable. That is a core strength for evaluation and RL.

### 5. Design for future training extension

Stay lightweight enough for a workshop, but keep the structure compatible with richer SFT, RL, and export flows later.

---

## Target Architecture

Refactor the repo into five active packages plus notebook/demo glue.

### `runtime/`

NAT-facing runtime package for single-episode execution. It should own:

- tool definitions and schemas
- prompt/runtime policy
- agent execution
- turn-level trace emission
- observability hooks
- directory-backed skill discovery, inspection, and command execution

#### Runtime skill architecture

Replace the flat workflow module with `src/runtime/skills/`, where each skill lives in its own folder and may include:

- `SKILL.md`
- optional sidecar files
- optional scripts

Expose these canonical interfaces from the runtime layer:

- `list_skills`: discovery with name, description, tags, and discovered files
- `search_skills`: metadata-only search across name, description, tags, declared assets, and filenames
- `get_skill`: load the full `SKILL.md` or a sidecar file by relative path
- `run_skill_command`: execute scripts within the skill folder

### `envs/`

Explicit task environment package, designed to map cleanly onto NeMo Gym environment and verification surfaces, for:

- episode state
- deterministic transitions
- allowed-action and tool-precondition logic
- terminal conditions
- reward-relevant task facts
- sequence-validity rules
- task-specific validation

### `rollouts/`

Canonical trace and rollout-adapter package for:

- episode runner integration
- adapters into NeMo Gym rollout collection
- trace serialization
- retry/failure representation
- stable multi-turn episode capture

### `training/`

`openpipe-art`-facing package for:

- dataset adapters
- reward views
- curriculum staging
- experiment definitions
- RL/SFT views over canonical trajectories

This layer defines what gets optimized, not how distributed execution runs.

### `eval/`

Offline evaluation package for:

- metrics
- summaries
- trajectory analysis
- regression reports
- consumption of NAT traces, NeMo Gym environment facts, and `openpipe-art` artifacts

Keep `eval/` separate from `envs/`: the environment owns task semantics, while evaluation owns reporting and cross-surface analysis.

---

## Proposed Repository Structure

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
│   │   ├── skills/
│   │   │   ├── __init__.py
│   │   │   ├── api.py
│   │   │   ├── diagnose-order-risk/
│   │   │   │   ├── SKILL.md
│   │   │   │   └── scripts/
│   │   │   └── ...
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
│   │   ├── export_adapters.py
│   │   ├── episode_runner.py
│   │   ├── serializers.py
│   │   └── trace_types.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── openpipe_art_adapter.py
│   │   ├── datasets.py
│   │   ├── curriculum.py
│   │   ├── experiments.py
│   │   └── reward_views.py
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

This structure is the target direction. It does not require every module to be rewritten at once.

---

## File-Level Refactor Mapping

### `src/tools.py` -> `src/runtime/tools.py`

- keep deterministic tool implementations
- make tool inputs and outputs schema-driven
- remove training/export concerns
- remove evaluation-side reward logic

### `src/skills.py` -> `src/runtime/skills/`

- replace the flat module with a directory-backed skills package
- keep each skill in its own folder with `SKILL.md` and optional sidecars or scripts
- implement `list_skills`, `search_skills`, `get_skill`, and `run_skill_command`
- keep this package NAT-facing and runtime-owned

### `src/schema.py` -> `src/runtime/schemas.py` and `src/envs/validators.py`

- keep runtime action/observation schemas in `runtime/schemas.py`
- move environment-specific validation into `envs/validators.py`
- maintain one canonical schema per action or event type

### `src/agent_loop.py` -> `src/runtime/agent.py` and `src/rollouts/episode_runner.py`

- keep `runtime/agent.py` focused on single-episode execution
- keep `rollouts/episode_runner.py` focused on adapting, running, and recording canonical episodes without becoming the long-term owner of training rollout execution
- separate agent decisions from rollout collection

### `src/fallbacks.py` -> `src/runtime/agent.py`, `src/envs/transitions.py`, or `src/rollouts/serializers.py`

- keep runtime robustness where needed
- do not silently hide malformed tool calls during RL rollouts
- represent malformed calls, repairs, and rejects as explicit trace events

### `src/evaluation.py` -> `src/envs/rewards.py` and `src/eval/metrics.py`

- move training-relevant reward inputs into `envs/rewards.py`
- move offline benchmarking and reporting into `eval/metrics.py`
- let evaluation consume NAT traces, NeMo Gym environment facts, and `openpipe-art` artifacts without redefining them
- keep step-level rewards separate from offline evaluation summaries

### `src/training_export.py` -> `src/rollouts/serializers.py`, `src/training/datasets.py`, `src/training/openpipe_art_adapter.py`, and `src/training/experiments.py`

- eliminate the catch-all export file pattern
- define a canonical trajectory format first
- build training dataset views from that format
- isolate rollout serialization and NeMo Gym handoff from trainer code
- isolate `openpipe-art` integration from runtime code

---

## Canonical Data Model

Introduce one canonical episode/trajectory format shared across runtime, rollouts, training, and evaluation.

### Episode contents

Each episode should preserve:

- task input
- initial environment-state metadata
- turns in order
- model actions
- tool-call payloads
- tool results
- validation, fallback, repair, and reject events
- step-level rewards or reward annotations
- terminal outcome
- summary metrics

### Event types

Use explicit typed records such as:

- `user_task`
- `model_thought` if intentionally preserved
- `tool_call`
- `tool_result`
- `tool_validation_error`
- `tool_repair_attempt`
- `tool_reject`
- `agent_message`
- `terminal_outcome`

Use dataclasses or Pydantic models. Structured event records should replace unstructured trace text as the source of truth.

### Data boundaries

- `runtime/` emits canonical events
- `rollouts/` serializes canonical events and adapts them to NeMo Gym collection surfaces
- `training/` consumes them as datasets and reward views
- `eval/` analyzes them offline alongside environment facts and training artifacts

---

## Environment Requirements

Create an explicit task environment for late-order recovery.

### The environment must own

- known versus unknown information
- completed subgoals
- valid actions and tool preconditions
- resulting observations
- terminal status
- reward-relevant task facts and penalties

### Minimum state

- order id
- source DC status
- alternate DC feasibility
- supplier expedite feasibility
- partial-fulfillment feasibility
- substitute-SKU viability
- tool calls already made
- current recommendation candidate
- invalid-action or failure counters
- terminal status

Transitions must remain deterministic and must not live primarily in notebook code or ad hoc evaluation helpers.

---

## Reward Design Requirements

Implement dense, sequence-aware rewards for multi-turn RL.

### Positive signals

- valid structured tool calls
- correct tool choice for the current state
- correct argument extraction
- prerequisite satisfaction before tool use
- progress toward resolving the order-risk decision
- no redundant calls
- correct final recommendation
- concise completion

### Penalties

- malformed tool calls
- invalid schemas
- unmet prerequisites
- repeated or redundant calls
- looping behavior
- hallucinated conclusions
- overlong episodes
- silent fallback reliance

Do not reward only final success. The reward design should reflect the decision process turn by turn.

Ownership split:

- `envs/` provides reward-relevant task signals
- `training/` builds trainer-facing reward views

---

## Runtime Requirements

Rewrite the runtime toward a NAT-friendly structure.

- isolate model interaction from environment logic
- isolate the tool registry from rollout logic
- keep the agent loop thin and composable
- centralize prompt/runtime policy
- emit structured traces natively
- prefer one canonical action format
- minimize fallback parsing
- keep repair logic explicit and auditable
- make backend model swaps possible without changing agent semantics

The runtime may call tools and emit events, but it should not schedule large training rollout jobs, construct RL datasets, or own training/deployment concerns. NAT owns the interactive loop, not the scalable training rollout substrate.

---

## Rollout Requirements

Treat multi-turn rollout contracts as first-class, while using NeMo Gym as the intended owner of training-time rollout execution.

- run many episodes through a stable adapter into NeMo Gym collection
- preserve exact turn alignment
- preserve failure and repair events
- serialize episodes in a stable format
- keep repo rollout code focused on adapters and serialization rather than becoming a second framework
- support future concurrency and batching without changing the episode schema

Rollout collection must not depend on notebook execution order.

---

## `openpipe-art` Training Requirements

The training layer should consume structured trajectories, not notebook-generated ad hoc exports.

- create dataset adapters over canonical trajectories
- support successful-trace extraction for SFT
- support stepwise reward extraction for RL
- support future curriculum stages
- keep trainer configuration separate from runtime logic
- keep reward views explicit and inspectable

Design for this progression:

1. SFT on successful trajectories
2. Short-horizon RL with dense rewards
3. Full multi-turn RL with sequence-aware rewards
4. Robustness curriculum with malformed calls and dead ends

The training layer defines data views, reward views, and experiments. It should not redefine runtime interfaces, rollout semantics, task logic, or offline benchmarks.

---

## Notebook Requirements

The notebook should remain useful, but it should no longer contain core system logic.

The notebook may:

- set up the demo
- walk through the scenario
- run example episodes
- display evaluation results
- visualize artifacts

The notebook must not:

- define canonical schemas
- define transition rules
- define reward logic inline
- act as the only execution path
- act as the only export path

---

## Implementation Plan

### Phase 1: Establish canonical interfaces

1. Create the new package structure under `src/`.
2. Introduce typed event and trajectory models.
3. Introduce environment state and transition skeletons.
4. Split runtime schemas from environment validation.

### Phase 2: Move runtime logic

1. Move tools into `runtime/tools.py`.
2. Replace the flat skills module with `runtime/skills/` and implement the four canonical skill interfaces.
3. Refactor the agent loop into `runtime/agent.py`.
4. Make trace emission structured and canonical.

### Phase 3: Formalize environment and rewards

1. Create `envs/state.py`.
2. Create `envs/transitions.py`.
3. Create `envs/rewards.py`.
4. Move sequence-sensitive logic out of generic evaluation code.

### Phase 4: Build the rollout layer

1. Create `rollouts/trace_types.py`.
2. Create `rollouts/serializers.py`.
3. Create `rollouts/episode_runner.py`.
4. Add any adapters needed for canonical trace handoff into NeMo Gym and `openpipe-art`.

### Phase 5: Build training semantics

1. Create `training/datasets.py`.
2. Create `training/reward_views.py`.
3. Create `training/openpipe_art_adapter.py`.
4. Create `training/experiments.py`.

### Phase 6: Remove outdated systems assumptions

This is already done:

1. `src/systems/` was deleted because it was empty and unused.
2. Scale-out config sketches and references were removed from training/export paths, the notebook, and docs.
3. No active training/export path depends on systems-layer assumptions.

### Phase 7: Rebuild offline evaluation

1. Create `eval/metrics.py`.
2. Create `eval/reports.py`.
3. Ensure offline metrics consume canonical traces.

### Phase 8: Update notebook and entrypoints

1. Update notebook imports.
2. Add a simple `src/main.py` or CLI entrypoint.
3. Make the notebook a consumer of the library rather than the source of truth.

---

## Constraints

### Code quality

- prefer small typed modules
- use dataclasses or Pydantic for structured records
- keep functions focused and side-effect-light
- document public interfaces
- avoid hidden global state

### Scalability

- do not hard-code notebook-only assumptions
- do not bury config in ad hoc cells
- make launch surfaces config-driven
- keep environment- or deployment-specific config separate from experiment semantics
- support future training growth without changing core task interfaces

### Backward compatibility

- preserve the workshop scenario behavior
- preserve the late-order recovery flow
- preserve deterministic tool semantics
- preserve the ability to run the demo end to end locally

### Refactor discipline

- do not add unnecessary feature scope
- avoid speculative abstractions
- do not rewrite scenario data unless the environment refactor requires it
- do not remove the notebook; demote it to a consumer

---

## Deliverables

### Required code deliverables

- new package structure under `src/`
- canonical trajectory and event types
- explicit environment state transitions
- clean runtime vs environment vs rollout vs training vs evaluation boundaries
- updated imports and entrypoints
- updated notebook wiring
- an `openpipe-art`-first training/export path

### Required documentation deliverables

- updated `README.md` describing the architecture
- module docstrings explaining responsibilities
- migration notes summarizing what moved where
- a brief note distinguishing active `openpipe-art` responsibilities from historical context

### Optional but encouraged

- a small CLI entrypoint for one episode
- a serialized trajectory example
- a short markdown architecture diagram

---

## Acceptance Criteria

The refactor is complete when all of the following are true:

1. Responsibility boundaries are clear across runtime, environment, rollouts, training, and evaluation.
2. There is no catch-all training/export module.
3. Structured traces are canonical across major flows.
4. The environment is explicit and no longer hidden in notebook or evaluation code.
5. The notebook consumes the library rather than defining core behavior.
6. NAT owns the interactive runtime loop, while NeMo Gym owns training-time rollout execution over repo-defined contracts.
7. `openpipe-art`-facing logic remains separate from runtime, rollout adaptation, and evaluation ownership.
8. Evaluation remains repo-owned and consumes NAT traces, NeMo Gym environment facts, and `openpipe-art` artifacts without redefining them.
9. Multi-turn RL readiness improves for successful trajectories, failed trajectories, reward shaping, rollout batching, and `openpipe-art` ingestion.
10. Outdated stack assumptions are contained and clearly marked as historical where they remain.
11. The late-order recovery scenario still runs end to end.

---

## Direct Instruction

Refactor the repository according to this document with these priorities:

1. Maximize separation of concerns.
2. Make the environment explicit.
3. Make structured trajectories canonical.
4. Prepare the codebase for multi-turn RL training of tool-calling agents.
5. Prepare the codebase for real `openpipe-art`-oriented training and export flows.
6. Preserve the current workshop scenario and demo value.

When choices are ambiguous, prefer the option that most clearly separates NAT runtime orchestration, NeMo Gym environment and rollout execution, repo-owned contracts, `openpipe-art` training semantics, and offline evaluation. Do not keep architecture-critical logic trapped in the notebook.
