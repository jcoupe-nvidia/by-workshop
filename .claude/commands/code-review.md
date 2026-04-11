---
description: Review code with RL architecture priorities
---

Read `@PLAN.md`, `@CLAUDE.md`, `@documents/RL_ARCHITECTURE.md`, and `@documents/NVIDIA_SOFTWARE_MAPPING.md`.

Review the current state of the code on the `main` branch.

Do not limit the review to the working tree diff or branch-local changes.

Write the review findings to `@code-review-issues.md`.

Do not commit `code-review-issues.md`; treat it as review output only.

Review priorities, in order from most important to least important:

1. Best-practice RL architecture, with a strong focus on GRPO-readiness.
2. Ensure the software defined in `documents/NVIDIA_SOFTWARE_MAPPING.md` is used for the layer and task described.
3. Explainability of the code and clarity of responsibility boundaries across the layers defined in `documents/RL_ARCHITECTURE.md`.
4. Strength and completeness of the observability layer.

During the review:
- prioritize findings over summary
- report only high-severity and medium-severity issues; ignore low-severity nits and minor style feedback
- focus on architecture, behavioral risks, ownership violations, scalability risks, and missing interfaces rather than style nits
- treat the checked-out `main` branch code as the review target unless the user explicitly says otherwise
- evaluate whether the code cleanly separates:
  - runtime orchestration
  - environment and task semantics
  - rollout generation
  - training semantics
  - training systems
  - offline evaluation
- check whether the NVIDIA software mapping is reflected in the implementation, including:
  - `NAT` owning runtime-facing agent execution, tool registration, and skill surfaces
  - `NeMo Gym` owning environment-backed training-time execution and rollout collection surfaces
  - `openpipe-art` owning trainer-facing datasets, reward views, and training handoff
  - `eval/` remaining repo-owned rather than being absorbed into training or runtime code
  - repo-owned task contracts staying canonical instead of being redefined inside framework-specific adapters
- check whether the design is compatible with multi-turn RL collection and GRPO-style post-training, including:
  - stable canonical trajectory structure
  - clear reward and advantage-consumption boundaries
  - scalable rollout collection surfaces
  - explicit failure, repair, reject, and terminal events
  - trainer-facing data views that are independent from runtime code
- check whether module names, APIs, and data flow make the roles of NAT, the repo's rollout layer, and `openpipe-art` easy to explain to a workshop or engineering audience while keeping older trainer-facing, rollout-shaping, and scale-out systems references clearly historical
- check whether observability is strong enough for debugging and training analysis, including:
  - structured event tracing
  - clear episode and turn boundaries
  - failure and fallback visibility
  - serialization surfaces for later analysis
  - metrics or hooks that support regression analysis
- call out places where responsibilities overlap, where abstractions are ambiguous, where the wrong NVIDIA software appears to own a layer, or where future scaling would likely become harder
- identify missing tests only when they materially affect the review priorities above

Return the review in this format:

## Findings
- list only high-severity and medium-severity issues, ordered by severity
- for each issue, include the affected file, concrete evidence from the code or diff, and why the issue matters

## Open Questions
- list assumptions, ambiguities, or missing context that affect confidence

## Summary
- briefly summarize overall alignment with the refactor goals
- mention residual risks or notable testing and observability gaps

After producing the review, ensure the same content is written to `@code-review-issues.md`.
