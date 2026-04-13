---
description: Fix issues from code review and run tests
---

Read `@PLAN.md`, `@CLAUDE.md`, `@documents/RL_ARCHITECTURE.md`, `@documents/NVIDIA_SOFTWARE_MAPPING.md`, and `@documents/NAT.md`.

Fix the issues identified in `@code-review-issues.md`.

Prioritize fixes using the same review order:

1. Best-practice RL architecture, with a strong focus on GRPO-readiness: verification and reward design, on-policy training compatibility, and multi-step environment correctness.
2. Verification and reward design quality: ensure `verify()` is reliable, meaningful, and scalable; uses the appropriate verification strategy (trajectory matching, state matching, or composite); guards against malformed output; includes structured diagnostic fields; and profiles reward distributions before GRPO training.
3. Multi-step environment correctness: session state isolation by session identifier with direct assignment in `seed_session()` (not `setdefault`); session cleanup at episode end; sequential tool execution enforced via `parallel_tool_calls: false`; tool errors returned as observations rather than crashing the episode; tool errors logged in traces.
4. Ensure the software defined in `documents/NVIDIA_SOFTWARE_MAPPING.md` is used for the layer and task described, including correct alignment with the NeMo Gym three-server architecture (Agent server -> `runtime/`, Model server -> external/stateless, Resources server -> `envs/`).
5. NAT usage follows the best practices defined in `documents/NAT.md`, including runtime-only ownership, explicit skills, function groups, declarative config, intentional middleware, trace preservation, and observability.
6. On-policy training compatibility: preserve raw token IDs and log probabilities from prior model calls rather than relying on re-tokenization; avoid modifying rollout history between model calls; keep these corrections in the training framework integration, not in `envs/` or `runtime/`.
7. Explainability of the code and clarity of responsibility boundaries across the layers defined in `documents/RL_ARCHITECTURE.md`, including verification logic living in `envs/` (not scattered across `runtime/` adapters or notebook cells).
8. Strength and completeness of the observability layer.

Before making changes:
- review `@code-review-issues.md` and group the findings into concrete fix categories
- confirm which findings are actionable code changes versus open questions or assumptions
- preserve the ownership boundaries described in `@documents/RL_ARCHITECTURE.md`
- follow the NAT best practices described in `@documents/NAT.md` when fixing runtime-layer issues
- preserve the current phase status and sequencing in `@PLAN.md`
- avoid speculative refactors that are not needed to resolve the review findings

During execution:
- implement fixes for the actionable findings in `@code-review-issues.md`
- prefer changes that improve architectural clarity, separation of concerns, and observability without expanding scope unnecessarily
- keep runtime, environment, rollout, training, systems, and evaluation responsibilities clearly separated
- when fixing environment-layer issues, ensure:
  - `seed_session()` initializes state with direct assignment, never `setdefault`
  - `setdefault` is only used in tool methods as a safe fallback for uninitialized sessions
  - all mutable per-episode state is keyed by session identifier
  - session state is cleaned up at episode end
  - tool execution errors are returned as observations, not raised as exceptions
  - `verify()` wraps JSON parsing in try/except and defaults to reward 0.0 on parse failure
  - `verify()` returns structured diagnostic fields alongside the scalar reward
- when fixing verification and reward issues, ensure:
  - the verification strategy is documented and deliberately chosen (trajectory matching vs state matching vs composite)
  - for the supply-chain scenario, composite verification is the default
  - binary rewards (0/1) are used unless continuous rewards are explicitly justified
  - reward distribution profiling is supported before GRPO training begins
- when fixing rollout or training issues, ensure:
  - sequential tool execution is enforced where tool outputs inform subsequent calls
  - raw token IDs and generation metadata are preserved for on-policy training compatibility
  - rollout history is not silently modified between model calls
  - async metadata fields (weight version, freshness, replay acceptance) are present if async GRPO is in scope
- when fixing runtime-layer issues, ensure:
  - runtime orchestrates the rollout lifecycle (seed -> loop -> verify) but does not own task truth or verification logic
  - the model server is treated as external and stateless
  - malformed calls, rejects, retries, and repairs are preserved in traces
- eliminate anti-patterns listed in `documents/RL_ARCHITECTURE.md`:
  - crashing the episode on tool execution errors instead of returning the error as an observation
  - sharing mutable state across concurrent rollouts
  - using `setdefault` in session initialization
  - skipping reward distribution profiling before GRPO training
  - relying on re-tokenized or re-templated text for on-policy training
  - defining verification logic outside `envs/`
- update or add tests only when they materially improve confidence in the fixes
- do not modify or commit `@code-review-issues.md`; treat it as review input only

Validation:
- run all relevant tests for the changed code
- if the repository has a single project-wide test command, run that as well
- report any failing tests, skipped tests, or missing test coverage clearly
- if there are lint or type-check commands that are standard for the repo, run them when relevant to the changes
- verify that `verify()` implementations are deterministic (same rollout produces the same score)
- verify that session state isolation is correct by checking that no mutable state is shared across sessions

Return the result in this format:

## Fixed
- list the issues from `@code-review-issues.md` that were addressed
- include concrete file references and a short explanation of each fix

## Validation
- list the commands or test suites that were run
- summarize pass/fail status and any important output

## Remaining
- list findings from `@code-review-issues.md` that were not fixed
- explain blockers, open questions, or reasons for deferring them
