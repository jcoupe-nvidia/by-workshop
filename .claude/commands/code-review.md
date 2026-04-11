---
description: Review code for scalable, transparent RL workflows
---

Read `@PLAN.md`, `@CLAUDE.md`, `@documents/RL_ARCHITECTURE.md`, and `@documents/NVIDIA_SOFTWARE_MAPPING.md`.

Review the current state of the code on the `main` branch.

Do not limit the review to the working tree diff or branch-local changes.

Write the review findings and recommended fixes to `@code-review-issues.md`.

If `@code-review-issues.md` does not exist, create it using the template below. If it already exists, replace its contents with the completed review using the same section structure.

Do not commit `code-review-issues.md`; treat it as review output only.

Primary objective:

Determine whether the repository supports a scalable, transparent, replayable, and workshop-teachable RL workflow for a multi-turn agent, not just clean module boundaries.

Review priorities, in order from most important to least important:

1. Transparent, replayable, end-to-end RL workflow design, with a strong focus on GRPO readiness.
2. Correct responsibility boundaries across `runtime/`, `envs/`, `rollouts/`, `training/`, and `eval/`, including the NVIDIA software mapping.
3. Demo teachability: whether the system is easy to explain to a workshop audience through concrete code paths and artifacts.
4. Strength and completeness of the observability and offline analysis surfaces.
5. Scalability risks that would make multi-turn rollout collection, trainer handoff, or regression analysis harder as the example grows.

During the review:
- prioritize findings over summary
- report only high-severity and medium-severity issues; ignore low-severity nits and minor style feedback
- focus on architecture, behavioral risks, ownership violations, scalability risks, and missing interfaces rather than style nits
- treat the checked-out `main` branch code as the review target unless the user explicitly says otherwise
- review the concrete code paths that implement the behavior instead of inferring intent from filenames, comments, or docs alone
- follow data flow and ownership boundaries across relevant modules before concluding that a responsibility is misplaced or missing
- make every finding evidence-based: cite the affected file or code path, describe the observed behavior, and explain the user or system impact
- include a specific recommended fix for every finding; prefer the smallest change or design adjustment that would materially address the issue
- avoid duplicate or overlapping findings; collapse related symptoms into a single higher-signal issue when they share the same root cause
- distinguish confirmed issues from assumptions or speculative concerns; move uncertainty into `Open Questions`
- prioritize correctness, reliability, safety, and regression risk over hypothetical future improvements
- check whether tests, assertions, or observability are sufficient to catch the issue or prevent regressions when that materially affects severity
- if no high-severity or medium-severity issues are found, say so explicitly instead of forcing findings
- validate end-to-end behavior, not just static structure, by tracing at least:
  - one successful multi-turn episode path
  - one malformed, rejected, failed, or repaired path when such a path exists in the codebase
  - one path from runtime action -> canonical trace -> trainer-facing view -> offline evaluation surface
- treat missing replayability, weak reward attribution, or inability to explain an episode end to end as medium-severity issues when they materially reduce debugging quality, training reliability, or workshop teachability
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
  - deterministic replay from saved artifacts or serialized episode state where the design claims this is supported
  - explicit episode identity and lineage across runtime, rollout, training, and evaluation artifacts
  - clear reward and advantage-consumption boundaries
  - reward views that can be traced back to canonical environment facts rather than opaque trainer-only logic
  - scalable rollout collection surfaces
  - batch-friendly or parallel-collection-friendly interfaces without pushing task semantics into framework adapters
  - explicit failure, repair, reject, and terminal events
  - trainer-facing data views that are independent from runtime code
- check whether canonical contracts are transparent and durable enough for analysis, including:
  - stable schema ownership for tool calls, events, traces, reward-relevant facts, and trainer-facing examples
  - serialization that preserves ordering, failures, retries, and stop reasons
  - enough structure to compare runs across regressions and training iterations
- check whether the code makes reward attribution and sequence-sensitive scoring easy to audit, including:
  - where reward-relevant facts are created
  - where those facts are transformed into reward or advantage inputs
  - whether offline evaluation reuses canonical semantics instead of silently drifting
- check whether module names, APIs, and data flow make the roles of NAT, the repo's rollout layer, and `openpipe-art` easy to explain to a workshop or engineering audience while keeping older trainer-facing, rollout-shaping, and scale-out systems references clearly historical
- check whether the code and artifacts are teachable in an interactive demo, including:
  - whether a reviewer can point to the exact files and data structures that answer "what happened", "why did it happen", "what was scored", and "which layer owned that decision"
  - whether there are human-readable artifacts or summaries that make episode behavior understandable without reverse-engineering framework internals
  - whether adapters to `NAT`, `NeMo Gym`, and `openpipe-art` stay thin enough that the repo-owned contracts remain the main teaching surface
- check whether observability is strong enough for debugging and training analysis, including:
  - structured event tracing
  - clear episode and turn boundaries
  - failure and fallback visibility
  - serialization surfaces for later analysis
  - metrics, summaries, or hooks that support regression analysis
  - ability to inspect both raw canonical traces and derived trainer/eval views without ambiguity
- call out places where responsibilities overlap, where abstractions are ambiguous, where the wrong NVIDIA software appears to own a layer, or where future scaling would likely become harder
- identify missing tests only when they materially affect the review priorities above

Return the review in this format:

```md
## Findings
- list only high-severity and medium-severity issues, ordered by severity
- if there are no such issues, state `No high-severity or medium-severity findings.`
- for each issue, include:
  - severity
  - affected file or code path
  - concrete evidence from the code or diff
  - why the issue matters
  - a recommended fix that is specific and actionable

## Open Questions
- list assumptions, ambiguities, or missing context that affect confidence

## Evidence Reviewed
- list the concrete code paths, traces, or artifacts you inspected
- say whether conclusions came from static code inspection, executable paths, serialized artifacts, or a combination

## Summary
- briefly summarize overall alignment with the scalable, transparent RL-demo goals
- mention residual risks or notable replayability, testing, teachability, and observability gaps
```

After producing the review, ensure the same content is written to `@code-review-issues.md`.
