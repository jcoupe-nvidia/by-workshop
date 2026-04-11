---
description: Review code with RL architecture priorities
---

Read `@REFACTOR.md`, `@PLAN.md`, and `@CLAUDE.md`.

Review the current branch changes or working tree diff.

Write the review findings to `@code-review-issues.md`.

Do not commit `code-review-issues.md`; treat it as review output only.

Review priorities, in order from most important to least important:

1. Best-practice RL architecture, with a strong focus on GRPO-readiness.
2. Explainability of the code and clarity of responsibility boundaries across NVIDIA software:
   - NeMo Agent Toolkit
   - `openpipe-art`
   - historical trainer-facing, rollout-shaping, and scale-out systems references where they still appear
3. Strength and completeness of the observability layer.

During the review:
- prioritize findings over summary
- focus on architecture, behavioral risks, ownership violations, scalability risks, and missing interfaces rather than style nits
- evaluate whether the code cleanly separates:
  - runtime orchestration
  - environment and task semantics
  - rollout generation
  - training semantics
  - training systems
  - offline evaluation
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
- call out places where responsibilities overlap, where abstractions are ambiguous, or where future scaling would likely become harder
- identify missing tests only when they materially affect the review priorities above

Return the review in this format:

## Findings
- list issues first, ordered by severity
- include concrete file references and explain why each issue matters

## Open Questions
- list assumptions, ambiguities, or missing context that affect confidence

## Summary
- briefly summarize overall alignment with the refactor goals
- mention residual risks or notable testing and observability gaps

After producing the review, ensure the same content is written to `@code-review-issues.md`.
