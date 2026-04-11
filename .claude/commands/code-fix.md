---
description: Fix issues from code review and run tests
---

Read `@PLAN.md`, `@CLAUDE.md`, `@documents/RL_ARCHITECTURE.md`, and `@documents/NVIDIA_SOFTWARE_MAPPING.md`.

Fix the issues identified in `@code-review-issues.md`.

Prioritize fixes using the same review order:

1. Best-practice RL architecture, with a strong focus on GRPO-readiness.
2. Ensure the software defined in `documents/NVIDIA_SOFTWARE_MAPPING.md` is used for the layer and task described.
3. Explainability of the code and clarity of responsibility boundaries across the layers defined in `documents/RL_ARCHITECTURE.md`.
4. Strength and completeness of the observability layer.

Before making changes:
- review `@code-review-issues.md` and group the findings into concrete fix categories
- confirm which findings are actionable code changes versus open questions or assumptions
- preserve the ownership boundaries described in `@documents/RL_ARCHITECTURE.md`
- preserve the current phase status and sequencing in `@PLAN.md`
- avoid speculative refactors that are not needed to resolve the review findings

During execution:
- implement fixes for the actionable findings in `@code-review-issues.md`
- prefer changes that improve architectural clarity, separation of concerns, and observability without expanding scope unnecessarily
- keep runtime, environment, rollout, training, systems, and evaluation responsibilities clearly separated
- update or add tests only when they materially improve confidence in the fixes
- do not modify or commit `@code-review-issues.md`; treat it as review input only

Validation:
- run all relevant tests for the changed code
- if the repository has a single project-wide test command, run that as well
- report any failing tests, skipped tests, or missing test coverage clearly
- if there are lint or type-check commands that are standard for the repo, run them when relevant to the changes

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
