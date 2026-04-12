# RL Architecture

## Purpose

This note defines the RL-layer boundary in this repo and the design rules for changing it.

The goal is a clear, inspectable, sequence-sensitive workflow for a multi-step agent where task semantics, trajectories, and learning signals stay explicit. This repo is not a generic RL platform or a second rollout framework.

Version reference:

- `nvidia-nat==1.6.0`
- `nemo-gym==0.2.0`
- `nemo-rl==0.5.0rc0`

Use `documents/NVIDIA_SOFTWARE_MAPPING.md` as the source of truth for the versioned NVIDIA stack mapping.

Boundary rule:

- `runtime/` decides how the agent acts
- `envs/` decide what those actions mean
- `rollouts/` record what happened
- `training/` turns experience into optimization targets
- `eval/` measures quality and regressions

## Core Contracts

Use one small set of canonical contracts everywhere.

### 1. Task Contract

Defines:

- `task_name`
- task-specific prompt inputs
- environment binding
- tool or action-space constraints
- validity and terminal rules
- reward-relevant facts the environment must expose
- success criteria and evaluation hooks

`task_name` is the stable routing key for processors, environments, prompt defaults, and evaluation rules.

### 2. Datum Contract

Normalize training and rollout inputs into one canonical per-example record containing:

- initial message or interaction history
- task identity
- environment facts needed for validation or scoring
- loss mask or weighting fields
- stable example identity for tracing and reproducibility

This datum is the source of truth for routing and trainer-side preprocessing. Notebook records should not become a competing format.

### 3. Trace Contract

Canonical traces must preserve:

- ordered turns
- actions, tool calls, and arguments
- tool results
- validation failures, rejects, retries, and repairs
- stop reasons and terminal state
- environment facts needed for scoring or audit

The trace must preserve sequence and failure information faithfully enough for offline evaluation and RL dataset construction.

### 4. Async Metadata Contract

If async GRPO is enabled, trajectories must also include:

- generation weight version
- intended training weight version or equivalent routing metadata
- age or freshness metadata
- replay acceptance or rejection reason

This makes stale trajectories explicit and filterable instead of silently mixing them into training.

## Layer Responsibilities

### `runtime/`

Owns interactive execution policy.

Responsible for:

- tool definitions and schemas
- tool-call planning and orchestration
- prompt assembly and runtime policy
- skill discovery and loading
- agent-loop control and stop conditions
- runtime events and observability
- user-facing or demo-facing execution surfaces

Must not own:

- task truth
- environment transitions
- reward semantics
- trainer objectives
- offline scoring policy

It may emit rich events, but it must not redefine task contracts or mutate environment facts after a step is evaluated.

### `envs/`

Owns task truth.

Responsible for:

- state representation
- transition rules
- action validity and tool preconditions
- terminal conditions
- task-specific validation
- machine-checkable task dependencies
- reward-relevant facts exposed by the episode

It should return explicit step results: state updates, validity outcomes, terminal conditions, and reward-relevant facts.

Must not own:

- prompt policy
- curriculum
- experiment scheduling
- trainer loss definitions
- report presentation

It may emit scalar rewards for convenience, but its primary outputs are task facts and transition results. Training decides how those facts become optimization targets.

### `rollouts/`

Owns episode capture and interchange.

Responsible for:

- canonical trace and event schemas
- serialization and deserialization
- preservation of turn order
- preservation of failures, retries, rejects, and repairs
- adapters between execution surfaces and stored trajectories
- collection-facing batching or plumbing
- validation that stored traces match repo contracts

Must not:

- redefine task semantics
- reinterpret environment truth
- apply trainer-specific reward shaping
- become a competing framework with its own task meaning

If async execution exists, `rollouts/` may preserve freshness and version metadata, but it does not choose the training objective.

### `training/`

Owns trainer-facing learning views.

Responsible for:

- dataset construction from canonical data and traces
- reward views and target construction
- masking and weighting
- curriculum staging
- batching for optimization
- experiment definitions
- training handoff artifacts

Must not become:

- a second runtime
- a second environment
- a second evaluation layer
- a hidden task-definition layer

It consumes canonical traces and environment facts, then derives trainer-facing targets from them without inventing new task semantics.

### `eval/`

Owns offline measurement and regressions.

Responsible for measuring:

- task success
- sequence correctness
- action validity
- tool accuracy
- recovery quality
- efficiency
- trend regressions
- skill selection quality, where relevant

It scores canonical traces against environment facts and repo-defined success criteria.

Must not:

- re-implement transitions
- mutate training objectives
- silently change task semantics
- become the hidden owner of success definitions

Evaluation should be able to explain why an episode passed or failed from stored traces and environment outputs.

## Boundary Test

Keep these boundaries intact:

- `runtime/` decides how the agent acts
- `envs/` decide what those actions mean
- `rollouts/` record what happened
- `training/` turns experience into learning signals
- `eval/` measures quality and regressions

If two layers own the same semantic decision, the design is drifting.

Quick checks:

- deciding whether a tool call was valid is environment logic
- deciding how invalid calls affect optimization is training logic
- deciding whether a run counts as success for benchmarking is evaluation logic
- only preserving event history is rollout logic
- changing agent behavior at execution time is runtime logic

## GRPO-Aligned Design Practices

1. Keep contracts canonical.
   Define task contracts, datum schemas, trace schemas, and success criteria in one repo-owned place and reuse them across runtime, rollout, training, and evaluation.
2. Make task routing explicit.
   Route examples through a stable `task_name` contract with explicit mappings to processors, environments, prompt defaults, and evaluation rules.
3. Separate task truth from optimization.
   Environments expose transitions, validity, terminal conditions, and reward-relevant facts. Training turns those facts into masks, reward views, advantages, and trainer-facing targets.
4. Make sequence correctness first-class.
   Preserve and score action order, dependency order, invalid attempts, repairs, and recovery behavior. A correct-looking final answer reached through an invalid sequence should not automatically count as fully successful.
5. Capture full trajectories, not only outcomes.
   Store successful actions, failed calls, rejects, validation errors, retries, repairs, and stop reasons.
6. Prefer deterministic, inspectable tools and validators.
   Training signals are more useful when tool behavior and environment checks are stable and easy to audit.
7. Keep skills explicit and bounded.
   Prefer a small number of higher-level skills with clear responsibilities and visible tool-use patterns.
8. Use repo-owned adapters to external frameworks.
   External RL or serving libraries should connect through thin adapters; task semantics must stay in repo code.
9. Preserve reproducibility.
   Use stable example IDs, deterministic tools where possible, fixed synthetic data where appropriate, stable serialization, and machine-checkable traces.
10. Keep the architecture teachable.
    Favor short modules, typed records, explicit flow, and visible data transformations over abstraction-heavy designs.
11. Keep notebooks as consumers, not sources of truth.
    Notebooks may run episodes, visualize traces, and explain behavior, but canonical schemas, routing, transitions, reward facts, and scoring contracts must live in repo code.
12. Support async only with explicit freshness controls.
    If async GRPO is introduced, trajectories must carry version and freshness metadata, and replay acceptance rules must be explicit.

## Async GRPO Addendum

This section applies only if async GRPO is enabled.

Async collection assumes generated trajectories may be older than the currently optimized policy, so stale or mismatched experience must be visible and controllable.

Additional requirements:

- trajectory records include generation weight version
- replay logic enforces explicit age or freshness limits
- stale-trajectory rejection is representable in stored metadata
- training can apply off-policy corrections where required
- weight synchronization events are visible in trace or collector metadata
- inference and training resource boundaries stay explicit

Responsibility split under async mode:

- `runtime/` still owns action-time behavior
- `envs/` still own transitions and task truth
- `rollouts/` preserve trajectory metadata and version fields
- `training/` owns freshness policy, replay acceptance, and off-policy correction behavior
- `eval/` may report freshness-related regressions but must not redefine replay policy

## Anti-Patterns

Avoid:

- burying core task logic in notebook cells
- hiding malformed tool calls, repairs, or rejects instead of tracing them explicitly
- mixing runtime execution, rollout collection, reward shaping, and evaluation in one module
- duplicating task validity or sequence semantics across runtime, training, and evaluation
- letting training exports redefine the task contract
- letting rollout code become a second framework with its own task semantics
- letting framework adapters own routing, rewards, or success logic
- silently consuming stale async trajectories without explicit policy
- adding platform-specific complexity that obscures the teaching example

## One-Sentence Summary

Define task truth once, record behavior faithfully, derive learning signals explicitly, and evaluate against the same canonical contracts.