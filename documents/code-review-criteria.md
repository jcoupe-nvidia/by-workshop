# Code Review Criteria

Reference checklist for reviewing this repository. Ordered by priority.

## Review priorities

1. Transparent, replayable, end-to-end RL workflow design with GRPO readiness.
2. Correct responsibility boundaries across `runtime/`, `envs/`, `rollouts/`, `training/`, and `eval/`, including the NVIDIA software mapping and the NeMo Gym three-server architecture.
3. Verification and reward design quality: whether `verify()` is reliable, meaningful, and scalable, and whether the chosen verification strategy is appropriate.
4. Multi-step environment correctness: session state isolation, error-as-observation, sequential tool execution, on-policy training compatibility.
5. Demo teachability: easy to explain through concrete code paths and artifacts.
6. Observability and offline analysis surfaces.
7. Scalability risks for multi-turn rollout collection, trainer handoff, or regression analysis.

## Conduct rules

- Prioritize findings over summary.
- Report only high-severity and medium-severity issues; ignore low-severity nits and minor style feedback.
- Focus on architecture, behavioral risks, ownership violations, scalability risks, and missing interfaces.
- Review concrete code paths, not inferred intent from filenames or docs.
- Follow data flow and ownership boundaries across relevant modules before concluding a responsibility is misplaced.
- Make every finding evidence-based: cite file or code path, describe observed behavior, explain impact.
- Include a specific recommended fix for every finding; prefer the smallest change that materially addresses the issue.
- Collapse related symptoms into a single higher-signal issue when they share a root cause.
- Do not report findings that overlap with items in `future-work-list.md`.
- Distinguish confirmed issues from assumptions; move uncertainty into `Open Questions`.
- Check whether tests or observability are sufficient to catch an issue when that affects severity.
- If no high-severity or medium-severity issues are found, say so explicitly.

## End-to-end traces to validate

Trace at least:
- One successful multi-turn episode path.
- One malformed, rejected, failed, or repaired path (when it exists).
- One path from runtime action → canonical trace → trainer-facing view → offline evaluation surface.

## Responsibility separation checks

Check whether the code cleanly separates:
- Runtime orchestration
- Environment and task semantics
- Rollout generation
- Training semantics and training systems
- Offline evaluation

## NVIDIA software mapping checks

- `NAT` owns runtime-facing agent execution, tool registration, and skill surfaces.
- `NeMo Gym` owns environment-backed training-time execution and rollout collection surfaces.
- `NeMo RL` owns trainer-facing datasets, reward views, and training handoff.
- `eval/` remains repo-owned rather than absorbed into training or runtime code.
- Repo-owned task contracts stay canonical instead of being redefined inside framework-specific adapters.

## NeMo Gym three-server architecture checks

- Agent server responsibilities map to `runtime/` (orchestrates rollout lifecycle, calls model, routes tool calls, collects reward; does not run an LLM itself).
- Model server is external and stateless.
- Resources server responsibilities map to `envs/` (owns tasks, tools, per-rollout session state, verification logic; returns reward signals).
- Rollout lifecycle follows Initialize → Agent loop → Grade phases.

## GRPO and multi-turn RL compatibility checks

- Stable canonical trajectory structure.
- Deterministic replay from saved artifacts where claimed.
- Explicit episode identity and lineage across runtime, rollout, training, and evaluation.
- Clear reward and advantage-consumption boundaries.
- Reward views traceable to canonical environment facts.
- Batch-friendly or parallel-collection-friendly interfaces.
- Explicit failure, repair, reject, and terminal events.
- Trainer-facing data views independent from runtime code.

## Verification and reward design checks

- `verify()` is reliable, meaningful, and scalable.
- Verification strategy is deliberately chosen: trajectory matching, state matching, or composite.
- For the supply-chain scenario, composite verification is the default.
- Binary rewards unless continuous is justified.
- `verify()` guards against malformed output with try/except, defaults to 0.0.
- Structured diagnostic fields accompany the scalar reward.
- Efficiency metrics tracked as secondary signal.
- Reward distribution profiled before GRPO training.

## Multi-step environment checks

- Session state isolation: mutable state keyed by session identifier.
- `seed_session()` uses direct assignment, not `setdefault`.
- Session state cleanup at episode end.
- Sequential tool execution enforced when outputs inform subsequent calls.
- Tool errors returned as observations, not HTTP errors.
- Tool errors logged in traces.

## On-policy training compatibility checks

- Raw token IDs and log probabilities preserved rather than re-tokenizing detokenized text.
- Rollout history not modified between model calls unless explicitly accounted for.
- On-policy token ID fixes handled by training framework, not `envs/` or `runtime/`.

## Contract durability checks

- Stable schema ownership for tool calls, events, traces, reward-relevant facts, trainer-facing examples.
- Serialization preserves ordering, failures, retries, stop reasons.
- Async metadata contract fields when async GRPO is enabled.

## Reward attribution checks

- Where reward-relevant facts are created.
- Where facts are transformed into reward or advantage inputs.
- Whether offline evaluation reuses canonical semantics.
- Verification logic lives in `envs/` (reusable by training and evaluation).

## NAT usage checks

See `documents/NAT.md` for best practices. Key points:
- NAT stays as the runtime layer, does not absorb task-truth, reward, or evaluation.
- Skills are explicit, small, bounded.
- Related tools use NAT function groups.
- Workflow configuration is declarative where appropriate.
- Middleware does not obscure tool behavior.
- Malformed calls, rejects, retries preserved in traces.
- Observability wired into runtime.
- Evaluation uses datasets and custom evaluators.
- Deterministic tools have isolated unit tests.

## Anti-patterns to check

- Burying core task logic in notebook cells.
- Hiding malformed tool calls instead of tracing them.
- Mixing runtime, rollout, reward, and evaluation in one module.
- Duplicating task validity across runtime, training, and evaluation.
- Training exports redefining the task contract.
- Rollout code becoming a second framework.
- Framework adapters owning routing, rewards, or success logic.
- Crashing episodes on tool errors instead of returning error-as-observation.
- Shared mutable state across concurrent rollouts.
- `setdefault` in session initialization.
- Skipping reward distribution profiling.
- Defining verification logic outside `envs/`.
- Platform-specific complexity obscuring the teaching example.

## Teachability checks

- Can a reviewer point to exact files and data structures answering "what happened", "why", "what was scored", "which layer owned that decision"?
- Are there human-readable artifacts for understanding episodes without reverse-engineering?
- Do framework adapters stay thin enough that repo-owned contracts remain the teaching surface?

## Observability checks

- Structured event tracing.
- Clear episode and turn boundaries.
- Failure and fallback visibility.
- Serialization surfaces for later analysis.
- Metrics or hooks for regression analysis.

## SFT warm-start readiness

Check whether a small set of high-quality demonstrations exists for chat template format, tool-call syntax, and general readability.
