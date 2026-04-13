---
description: Fix issues from code review and run tests
---

Fix the issues identified in `@code-review-issues.md`.

Key reference documents (read as needed, not all at once):
- `CLAUDE.md` — repo purpose, design rules, and constraints
- `PLAN.md` — migration status (preserve current phase status)
- `documents/RL_ARCHITECTURE.md` — layer boundaries, design practices, and anti-patterns
- `documents/NVIDIA_SOFTWARE_MAPPING.md` — NVIDIA stack mapping
- `documents/NAT.md` — NAT best practices for runtime-layer fixes

Fix priorities, in order:

1. RL architecture and GRPO-readiness: verification/reward design, on-policy training compatibility, multi-step environment correctness.
2. Verification and reward quality: `verify()` reliability, strategy choice, malformed-output guards, diagnostic fields, reward distribution profiling.
3. Multi-step environment correctness: session isolation, sequential tool execution, error-as-observation.
4. NVIDIA software mapping alignment including NeMo Gym three-server architecture.
5. NAT usage best practices.
6. On-policy training compatibility.
7. Responsibility boundary clarity and explainability.
8. Observability completeness.

Before making changes:
- Read `code-review-issues.md` and enumerate every distinct finding.
- For each finding, decide whether it is actionable or an open question.
- Preserve ownership boundaries from `documents/RL_ARCHITECTURE.md` — especially the layer boundary table and anti-patterns list.
- Avoid speculative refactors not needed to resolve the findings.

To avoid filling the context window, delegate fixes to focused subagents. Before spawning any subagent, analyze the findings for dependencies:

1. Parse `code-review-issues.md` to build a list of individual findings (each heading or numbered item that describes a distinct problem).
2. Skip findings that are open questions, duplicates, or overlap with items already tracked in `future-work-list.md`.
3. Build a dependency graph across the remaining actionable findings. Two findings are dependent if any of the following hold:
   - They touch the same file.
   - One finding's fix changes a function signature, type, data structure, or contract that the other finding relies on.
   - One finding is about producing data (e.g., adding a diagnostic field, changing a return type) and another is about consuming that data (e.g., using the field downstream, adjusting reward logic).
   - They share a logical precondition — for example, fixing session isolation must land before fixing verify() guards that assume isolated sessions.
   - Fixing them independently would create merge conflicts or contradictory code.
4. Group dependent findings into clusters. Each cluster becomes one subagent. Independent findings that belong to no cluster each get their own subagent.
5. For each subagent (whether single-issue or multi-issue cluster), the prompt must include:
   - The full text of every finding assigned to it (severity, affected files, evidence, and recommended fix).
   - For clusters: an explicit note on why the findings are grouped and what ordering constraints exist between them.
   - Instructions to read the reference docs relevant to the affected files' layer before making changes:
     - `src/envs/` → `documents/RL_ARCHITECTURE.md` §§ Session State, Verification and Reward Design.
     - `src/runtime/` → `documents/NAT.md`.
     - `src/rollouts/` or `src/envs/nemo_gym_adapter.py` → `documents/RL_ARCHITECTURE.md` § NeMo Gym Three-Server Lifecycle.
     - `src/training/` → `documents/RL_ARCHITECTURE.md` §§ Design Practices, On-Policy Corrections.
     - `src/eval/`, `src/scenario_data.py`, `src/shared/` → `documents/RL_ARCHITECTURE.md`.
   - A directive to fix only the files named in its assigned findings and return a short summary of what changed and why.
6. Launch all subagents in parallel — dependency ordering is handled within each cluster, not between subagents.

After all subagents complete, run the full test suite to validate. Do not modify or commit `code-review-issues.md`.

Return the result in this format:

```md
## Fixed
- list the issues from code-review-issues.md that were addressed
- include concrete file references and a short explanation of each fix

## Validation
- list the commands or test suites that were run
- summarize pass/fail status and any important output

## Remaining
- list findings from code-review-issues.md that were not fixed
- explain blockers, open questions, or reasons for deferring them
```
