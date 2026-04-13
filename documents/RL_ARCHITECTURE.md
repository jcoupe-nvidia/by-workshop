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

## NeMo Gym Server Mapping

The NeMo Gym three-server architecture maps to the repo layer split:

| NeMo Gym server | Repo layer | Responsibility |
| --- | --- | --- |
| Agent server | `runtime/` | Orchestrates the rollout lifecycle: calls the model, routes tool calls to the resources server, collects the final reward. Does not run an LLM itself. |
| Model server | external (stateless) | Stateless LLM inference endpoint. Receives a conversation, returns text, tool calls, or code. No memory or orchestration logic. |
| Resources server | `envs/` | Owns tasks, tools, per-rollout session state, and verification logic. Returns reward signals for training. |

The rollout lifecycle follows three phases:

1. **Initialize**: `seed_session()` sets up isolated per-episode state.
2. **Agent loop**: model generates output; if tool calls, route to resources server, append results, repeat until stop criteria.
3. **Grade**: `verify()` evaluates the rollout and returns a reward.

This mapping reinforces the existing boundary rule: `runtime/` orchestrates, `envs/` owns task truth, and the model server stays stateless.

## Layer Responsibilities

### `runtime/`

Owns interactive execution policy. Maps to the NeMo Gym Agent server role.

Responsible for:

- tool definitions and schemas
- tool-call planning and orchestration
- prompt assembly and runtime policy
- skill discovery and loading
- agent-loop control and stop conditions
- runtime events and observability
- user-facing or demo-facing execution surfaces
- rollout lifecycle orchestration (seed → loop → verify)

Must not own:

- task truth
- environment transitions
- reward semantics
- trainer objectives
- offline scoring policy

It may emit rich events, but it must not redefine task contracts or mutate environment facts after a step is evaluated.

### `envs/`

Owns task truth. Maps to the NeMo Gym Resources server role.

Responsible for:

- state representation
- transition rules
- action validity and tool preconditions
- terminal conditions
- task-specific validation
- machine-checkable task dependencies
- reward-relevant facts exposed by the episode
- session-isolated per-rollout state so concurrent rollouts cannot interfere
- `seed_session()` initialization with direct assignment (not `setdefault`) to avoid silent re-seed bugs
- `verify()` implementation that evaluates rollout results and returns reward signals
- tool error propagation back to the model so the agent can self-correct rather than crashing the episode

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
   Route examples through a stable `task_name` contract with explicit mappings to processors, environments, prompt defaults, and evaluation rules. NeMo Gym uses `agent_ref` in JSONL data rows to route each example to its resources server; the repo equivalent is a stable `task_name` field.
3. Separate task truth from optimization.
   Environments expose transitions, validity, terminal conditions, and reward-relevant facts. Training turns those facts into masks, reward views, advantages, and trainer-facing targets.
4. Make sequence correctness first-class.
   Preserve and score action order, dependency order, invalid attempts, repairs, and recovery behavior. A correct-looking final answer reached through an invalid sequence should not automatically count as fully successful. For multi-step environments where tool outputs inform subsequent calls, enforce sequential tool execution (`parallel_tool_calls: false`) to make information dependencies explicit and trainable.
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
13. Isolate per-rollout state via session identity.
    Every concurrent rollout must have its own session-scoped state so parallel episodes never interfere. Initialize session state with explicit assignment in `seed_session()` and key all mutable environment data by session ID.
14. Propagate tool errors to the model instead of failing the episode.
    When a tool call raises an exception, return the error as an observation so the agent can attempt recovery. This makes error-handling behavior visible in traces and trainable via reward signals.
15. Choose verification strategy deliberately.
    Prefer trajectory matching (exact sequence comparison) when the correct path is unique and order matters. Prefer state matching (compare final environment state after replaying predicted vs ground-truth calls) when multiple valid tool sequences can produce the correct outcome. Document which strategy each task uses and why.
16. Profile reward distributions before training.
    Before starting GRPO training, collect rollouts with the target model and inspect the reward distribution across prompts. Rewards that are all-zero or all-one indicate a verification or difficulty mismatch that must be fixed before training can provide signal.
17. Warm-start RL with supervised fine-tuning when practical.
    Use a small set of high-quality demonstrations to teach chat template format, tool-call syntax, and general readability via SFT, then transition to RL for exploration and self-correction. This avoids wasting RL compute on format learning.

## Verification and Reward Design

These guidelines are drawn from NeMo Gym's task verification patterns and apply to the `envs/` layer.

### What makes good verification

A `verify()` implementation should be:

- **Reliable**: the same rollout must receive the same score every time. Avoid non-deterministic scoring paths.
- **Meaningful**: scores should reflect actual task performance (correctness, efficiency, sequence quality), not incidental surface features.
- **Scalable**: verification should be fast and local. Avoid expensive external API calls per-verification during training when deterministic checks suffice.

### Verification strategies

| Strategy | When to use | Trade-offs |
| --- | --- | --- |
| Trajectory matching | Correct tool-call sequence is unique and order is the thing being trained. | Simple to implement and debug; brittle when multiple valid paths exist. |
| State matching | Multiple tool-call sequences can produce the correct final outcome. | Rewards correct outcomes regardless of path; requires defining comparable mutable state and replay logic. |
| Answer extraction | Final answer can be parsed and compared to ground truth. | Lightweight; does not evaluate process quality. |
| Composite | Task requires both correct outcome and correct process. | Combines sequence and outcome signals; reward function must weight each component explicitly. |

For this repo's supply-chain scenario, composite verification is the default: score both sequence correctness (were dependencies respected?) and outcome quality (was the recommendation valid?).

### Reward shaping guidance

- Binary rewards (0/1) work well for GRPO. Continuous rewards are supported but introduce more variance.
- Guard `verify()` against malformed model output. Wrap JSON parsing in `try/except` and default to reward 0.0 on parse failure rather than crashing the episode.
- Include structured diagnostic fields alongside the scalar reward (e.g., `accuracy`, `set_overlap`, `sequence_violations`) so evaluation and debugging can inspect why a reward was assigned.
- Track efficiency metrics (tool call count vs expected count) as a secondary signal when relevant.

## On-Policy Corrections for Multi-Step Training

When training with GRPO on multi-step tool-calling rollouts, three sources of train-generation mismatch can degrade on-policy training:

1. **Re-tokenization**: token IDs produced in one model call may re-tokenize differently when appended to the prompt for the next call, because detokenization then re-tokenization is not always identity.
2. **Re-chat-templating**: structured tool-call output parsed into OpenAI-format objects may re-render differently when the chat template reconstructs the prompt string for the next call.
3. **Non-monotonic history**: if rollout history is truncated or summarized between calls (e.g., to manage context length), the prompt seen at training time differs from the prompt seen at generation time.

### Implications for this repo

- When constructing multi-step rollouts for training, propagate raw token IDs and log probabilities from prior model calls rather than relying on re-tokenization of detokenized text.
- If the training framework supports on-policy token ID fixes (as NeMo RL does), enable them for multi-step rollouts.
- Avoid modifying rollout history between model calls unless the training pipeline explicitly accounts for non-monotonic prompts.
- These corrections are handled by the training framework integration, not by `envs/` or `runtime/`. The repo's responsibility is to preserve the information (token IDs, generation metadata) needed for the fix.

## Multi-Step Environment Design Patterns

These patterns come from NeMo Gym's environment tutorials and apply to `envs/` and `rollouts/` in this repo.

### Session state management

- Use a session identifier to key all mutable per-episode state. In NeMo Gym this is `SESSION_ID_KEY`; in this repo, use whatever session or episode identifier the environment layer provides.
- Initialize state in `seed_session()` with direct assignment. Using `setdefault` in the seed path silently ignores re-initialization, which causes subtle bugs when session IDs are reused.
- In tool methods, `setdefault` is acceptable as a safe fallback if a session was somehow not initialized.
- Clean up session state at episode end to avoid memory leaks during long training runs.

### Information dependency

- When the output of one tool call informs the arguments of the next (the core pattern in this repo's supply-chain scenario), the environment should enforce sequential execution.
- Mark datasets with `parallel_tool_calls: false` to force the model to call tools one at a time rather than batching them.

### Dynamic tool routing

- For environments with many tools, a single catch-all endpoint (`/{tool_name}`) can dispatch to registered Python functions, avoiding per-tool endpoint boilerplate.
- The catch-all route must be registered after core lifecycle endpoints (`seed_session`, `verify`) to avoid intercepting them.

### Error as observation

- Tool execution errors should be returned as observations, not HTTP errors. This allows the model to learn self-correction behavior.
- Log the error in the trace so evaluation can distinguish tool errors from successful calls.

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
- crashing the episode on tool execution errors instead of returning the error as an observation for the model to learn from
- sharing mutable state across concurrent rollouts instead of isolating state by session or episode identifier
- using `setdefault` in session initialization where it silently ignores re-seed attempts
- skipping reward distribution profiling before starting GRPO training
- relying on re-tokenized or re-templated text as ground truth for on-policy training when raw token IDs are available
- defining verification logic outside `envs/` (e.g., in `runtime/` adapters or notebook cells) where it cannot be reused by training and evaluation

## One-Sentence Summary

Define task truth once, record behavior faithfully, derive learning signals explicitly, and evaluate against the same canonical contracts.