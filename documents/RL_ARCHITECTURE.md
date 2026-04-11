# RL Architecture

## Purpose

This note defines the responsibility split across the RL-facing layers in this repository and the high-value practices that should guide changes.

The goal is a clear, inspectable, sequence-sensitive workflow for a multi-turn agent, not a production RL platform.

## Layer Responsibilities

### `runtime/`

The runtime owns interactive single-episode behavior:

- tool definitions and schemas
- structured tool-call execution
- prompt and runtime policy
- skill discovery and loading
- agent loop control and stop conditions
- runtime event emission
- observability hooks
- user-facing or demo-facing execution surfaces

It decides how the agent acts. It should not own task truth, training targets, or offline scoring.

### `envs/`

The environment owns task truth:

- state representation
- transition rules
- action validity
- tool preconditions
- terminal conditions
- reward-relevant facts
- machine-checkable task dependencies
- task-specific validation

It decides what happened after an action. It should not own prompt policy, curriculum, or report presentation.

### `rollouts/`

The rollout layer owns episode capture and interchange:

- canonical trace and event schemas
- serialization and deserialization
- preservation of turn order, failures, retries, and repairs
- explicit reject and validation-error representation
- adapters between execution surfaces and stored trajectories
- batching or collection-facing trace plumbing

It records behavior faithfully. It must not redefine task semantics, runtime policy, or trainer logic, and it should not turn into a competing rollout framework.

### `training/`

The training layer owns trainer-facing learning views:

- dataset construction from canonical traces
- reward views and targets
- curriculum staging
- experiment definitions
- training handoff artifacts

It decides how experience becomes learning signals and what gets optimized. It should not become a second runtime, environment, or evaluation layer.

### `eval/`

The evaluation layer owns offline measurement and regressions:

- skill selection quality
- tool validity and tool accuracy
- sequence correctness
- task success and recovery quality
- efficiency and trend tracking

It should score canonical traces and environment facts without re-implementing transitions or mutating training objectives.

## Quick Boundary Test

Keep these boundaries intact:

- `runtime/` decides how the agent acts.
- `envs/` decides what those actions mean.
- `rollouts/` records what happened.
- `training/` turns trajectories into learning signals.
- `eval/` measures quality and regressions.

If a change makes two layers responsible for the same semantic decision, the design is probably drifting.

## RL Best Practices

Always follow these principles:

1. Keep contracts canonical.
   Define tool schemas, trace schemas, task semantics, and success criteria in one repo-owned place and reuse them everywhere.

2. Make sequence correctness first-class.
   Reward and evaluate the order of actions, not just the final answer. A correct recommendation reached through an invalid sequence should not be treated as fully successful.

3. Prefer deterministic, inspectable tools.
   Training signals are much more useful when tool behavior is stable, easy to reason about, and small enough to inspect manually.

4. Separate "what happened" from "how to score it."
   Environment transitions and factual episode state should be distinct from reward views, trainer objectives, and offline metrics. `envs/` should expose reward-relevant facts; `training/` should turn them into trainer-facing views.

5. Capture full trajectories, not only outcomes.
   Store successful steps, failed calls, validation errors, rejects, retries, repairs, and stop reasons so both evaluation and RL can learn from the full episode.

6. Keep skills explicit and bounded.
   Use a small number of higher-level skills with clear responsibilities and allowed tool-use patterns rather than hidden orchestration logic.

7. Reward behavior that transfers.
   Favor signals tied to valid tool use, dependency order, recovery quality, and task completion instead of brittle heuristics or presentation-only formatting.

8. Use narrow adapters to external frameworks.
   External libraries should plug into repo contracts through thin adapters rather than forcing task semantics to live inside framework-specific code.

9. Preserve reproducibility.
   Use fixed synthetic data, deterministic tools, stable serialization, and machine-checkable traces so results can be compared across runs.

10. Keep the demo architecture teachable.
    Favor clarity, short modules, typed records, and explicit flow over abstraction-heavy designs that hide where RL signals come from.

11. Keep the notebook as a consumer, not the source of truth.
    Demo notebooks should run episodes and explain behavior, but canonical schemas, transitions, rewards, and trace contracts should live in repo code.

## Anti-Patterns

Avoid these failure modes:

- burying core task logic in notebook cells
- silently hiding malformed tool calls or repair behavior instead of tracing them explicitly
- mixing runtime execution, rollout collection, reward shaping, and evaluation in one module
- duplicating reward or sequence semantics across runtime, training, and evaluation
- letting training exports redefine the task contract
- letting the rollout layer become a second framework with its own task semantics
- adding platform-specific complexity that obscures the pedagogical example
