# RL Architecture

## Purpose

This note defines the RL-layer boundary in this repo and the design rules for changing it.

The goal is a clear, inspectable, sequence-sensitive workflow for a multi-step agent where task semantics, trajectories, and learning signals stay explicit. This repo is not a generic RL platform or a second rollout framework.

Version reference: `nvidia-nat==1.6.0`, `nemo-gym==0.2.0`, `nemo-rl==0.5.0rc0`. See `documents/NVIDIA_SOFTWARE_MAPPING.md` for the full stack mapping.

## Layer Boundary

| Layer | Owns | Maps to (NeMo Gym) | Must not own |
| --- | --- | --- | --- |
| `runtime/` | Interactive execution policy: tool definitions/schemas, tool-call planning/orchestration, prompt assembly, skill discovery/loading, agent-loop control/stop conditions, runtime events, rollout lifecycle orchestration (seed → loop → verify) | Agent server | Task truth, environment transitions, reward semantics, trainer objectives, offline scoring |
| `envs/` | Task truth: state, transitions, action validity/preconditions, terminal conditions, task-specific validation, machine-checkable dependencies, reward-relevant facts, session-isolated per-rollout state, `seed_session()` init, `verify()` returning rewards, tool error propagation as observations | Resources server | Prompt policy, curriculum, experiment scheduling, trainer loss definitions |
| `rollouts/` | Episode capture and interchange: canonical trace/event schemas, serialization, turn-order preservation, failure/retry/reject/repair preservation, execution-to-storage adapters, collection batching | — | Redefining task semantics, reinterpreting environment truth, applying reward shaping, becoming a competing framework |
| `training/` | Trainer-facing learning views: dataset construction from canonical traces, reward views/targets, masking/weighting, curriculum staging, batching, experiment definitions, training handoff | — | Becoming a second runtime, environment, evaluation layer, or hidden task-definition layer |
| `eval/` | Offline measurement: task success, sequence correctness, action validity, tool accuracy, recovery quality, efficiency, trend regressions, skill selection quality | — | Re-implementing transitions, mutating training objectives, changing task semantics, hiding success definitions |

If two layers own the same semantic decision, the design is drifting. Quick test: validity checking is `envs/`; optimization impact of invalid calls is `training/`; benchmark success counting is `eval/`; event preservation is `rollouts/`; execution-time behavior change is `runtime/`.

## NeMo Gym Three-Server Lifecycle

The NeMo Gym model server is external and stateless (receives conversation, returns text/tool calls, no memory or orchestration). The rollout lifecycle:

1. **Initialize**: `seed_session()` sets up isolated per-episode state.
2. **Agent loop**: model generates → if tool calls, route to resources server → append results → repeat until stop criteria.
3. **Grade**: `verify()` evaluates the rollout and returns a reward.

## Core Contracts

### Task Contract

Defines: `task_name` (stable routing key for processors, environments, prompts, and eval rules), task-specific prompt inputs, environment binding, tool/action-space constraints, validity and terminal rules, reward-relevant facts the environment must expose, success criteria and evaluation hooks.

### Datum Contract

One canonical per-example record: initial message or interaction history, task identity, environment facts for validation/scoring, loss mask or weighting fields, stable example identity for tracing and reproducibility. This is the source of truth for routing and trainer-side preprocessing — notebook records must not become a competing format.

### Trace Contract

Canonical traces must preserve: ordered turns, actions/tool calls/arguments, tool results, validation failures/rejects/retries/repairs, stop reasons and terminal state, environment facts for scoring/audit. Must be faithful enough for offline evaluation and RL dataset construction.

### Async Metadata Contract

When async GRPO is enabled, trajectories must also include generation weight version, intended training weight version or routing metadata, age/freshness metadata, and replay acceptance/rejection reason.

## Design Practices

1. **Canonical contracts**: define task contracts, datum schemas, trace schemas, and success criteria once and reuse everywhere.
2. **Explicit task routing**: route via a stable `task_name` with explicit mappings to processors, environments, prompts, and eval rules. (NeMo Gym equivalent: `agent_ref` in JSONL data rows.)
3. **Task truth separate from optimization**: `envs/` exposes transitions, validity, terminal conditions, and facts; `training/` turns those into masks, rewards, advantages, and targets.
4. **Sequence correctness first-class**: preserve and score action order, dependency order, invalid attempts, repairs, and recovery. A correct final answer via an invalid sequence should not count as fully successful. Enforce sequential tool execution (`parallel_tool_calls: false`) when tool outputs inform subsequent calls.
5. **Full trajectories, not only outcomes**: store successes, failures, rejects, validation errors, retries, repairs, and stop reasons.
6. **Deterministic, inspectable tools and validators**: stable tool behavior and environment checks make training signals easier to audit.
7. **Explicit, bounded skills**: small number of higher-level skills with clear responsibilities and visible tool-use patterns.
8. **Thin adapters to external frameworks**: external RL/serving libraries connect through thin repo-owned adapters; task semantics stay in repo code.
9. **Reproducibility**: stable example IDs, deterministic tools, fixed synthetic data, stable serialization, machine-checkable traces.
10. **Teachable architecture**: short modules, typed records, explicit flow, visible data transformations over abstraction-heavy designs.
11. **Notebooks as consumers**: notebooks run episodes, visualize traces, and explain behavior; canonical schemas, routing, transitions, reward facts, and scoring contracts live in repo code.

## Session State and Multi-Step Patterns

- Key all mutable per-episode state by session/episode identifier so concurrent rollouts never interfere.
- Initialize state in `seed_session()` with direct assignment, not `setdefault` (which silently ignores re-initialization). `setdefault` is acceptable only in tool methods as a safe fallback.
- Clean up session state at episode end to avoid memory leaks during long training runs.
- When tool outputs inform subsequent calls (the core supply-chain pattern), enforce sequential execution and mark datasets with `parallel_tool_calls: false`.
- Return tool execution errors as observations (not HTTP errors or exceptions). Log errors in traces so evaluation can distinguish errors from successful calls.
- For environments with many tools, a single catch-all endpoint (`/{tool_name}`) can dispatch to registered functions; register it after core lifecycle endpoints.

## Verification and Reward Design

A `verify()` implementation should be **reliable** (same rollout always gets the same score), **meaningful** (reflects actual task performance), and **scalable** (fast, local, no expensive external calls per-verification).

| Strategy | When to use | Trade-offs |
| --- | --- | --- |
| Trajectory matching | Correct sequence is unique and order matters | Simple; brittle when multiple valid paths exist |
| State matching | Multiple valid sequences can produce the correct outcome | Rewards correct outcomes regardless of path; requires comparable mutable state and replay |
| Answer extraction | Final answer can be parsed and compared to ground truth | Lightweight; does not evaluate process quality |
| Composite | Task requires both correct outcome and correct process | Combines signals; must weight components explicitly |

For this repo's supply-chain scenario, **composite verification** is the default: score both sequence correctness and outcome quality.

Reward shaping:
- Binary rewards (0/1) work well for GRPO. Continuous rewards are supported but add variance.
- Guard `verify()` with `try/except` and default to reward 0.0 on parse failure.
- Include structured diagnostic fields alongside the scalar reward (e.g., `accuracy`, `set_overlap`, `sequence_violations`).
- Track efficiency metrics (tool call count vs expected) as a secondary signal.
- Profile reward distributions before GRPO training. All-zero or all-one distributions indicate verification or difficulty mismatches.

## On-Policy Corrections for Multi-Step Training

Three sources of train-generation mismatch in multi-step GRPO:

1. **Re-tokenization**: token IDs may re-tokenize differently after detokenization round-trip.
2. **Re-chat-templating**: structured tool-call output re-rendered through chat templates may differ.
3. **Non-monotonic history**: truncated or summarized rollout history changes training-time prompts.

Mitigations: propagate raw token IDs and log probabilities rather than re-tokenizing detokenized text; enable on-policy token ID fixes when the training framework supports them; avoid modifying rollout history between model calls unless explicitly accounted for. These corrections belong in the training framework integration, not in `envs/` or `runtime/`.

## Warm-Start and Async GRPO

**SFT warm-start**: use a small set of high-quality demonstrations to teach chat template format, tool-call syntax, and readability via SFT before transitioning to RL. Avoids wasting RL compute on format learning.

**Async GRPO** (when enabled): trajectories must carry generation weight version, freshness metadata, and replay acceptance/rejection reasons. `rollouts/` preserves this metadata; `training/` owns freshness policy and off-policy correction; `eval/` may report freshness regressions but must not redefine replay policy.

## Anti-Patterns

- Burying core task logic in notebook cells
- Hiding malformed tool calls instead of tracing them
- Mixing runtime, rollout, reward shaping, and evaluation in one module
- Duplicating task validity across layers or letting training exports redefine the task contract
- Letting rollout code or framework adapters own routing, rewards, or success logic
- Silently consuming stale async trajectories without explicit policy
- Crashing episodes on tool errors instead of returning error-as-observation
- Sharing mutable state across concurrent rollouts
- Using `setdefault` in `seed_session()` initialization
- Skipping reward distribution profiling before GRPO training
- Relying on re-tokenized text as on-policy ground truth when raw token IDs are available
- Defining verification logic outside `envs/`
- Adding platform-specific complexity that obscures the teaching example

## One-Sentence Summary

Define task truth once, record behavior faithfully, derive learning signals explicitly, and evaluate against the same canonical contracts.
