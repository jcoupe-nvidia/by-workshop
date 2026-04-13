---
description: Review code for scalable, transparent RL workflows
---

You are performing a code review of this repository. The review targets the checked-out `main` branch — all source code, not just the working tree diff.

Read `future-work-list.md` first. Items tracked there have been triaged and deferred. Do not report any finding whose root cause overlaps with a deferred item. When an issue is related but distinct, note the relationship and explain what is new.

Write the review findings to `code-review-issues.md`, replacing its contents if it already exists. Do not commit the file.

Primary objective: Determine whether the repository supports a scalable, transparent, replayable, and workshop-teachable RL workflow for a multi-turn agent — not just clean module boundaries.

The detailed review criteria and checklists are in `documents/code-review-criteria.md`. Read that file for the full review specification.

Key reference documents (read as needed, not all at once):
- `CLAUDE.md` — repo purpose, design rules, and constraints
- `PLAN.md` — migration status
- `documents/RL_ARCHITECTURE.md` — layer responsibilities and best practices
- `documents/NVIDIA_SOFTWARE_MAPPING.md` — NVIDIA stack mapping
- `documents/NAT.md` — NAT best practices

To avoid filling the context window, launch parallel subagents to review each module area. Each subagent should read `documents/code-review-criteria.md` plus the reference docs relevant to its scope, then review only its assigned source files. Suggested split:

1. **envs/** — Read `documents/RL_ARCHITECTURE.md`, then review `src/envs/`. Focus: session isolation, verification/reward design, error-as-observation, composite verification, seed_session correctness.

2. **runtime/** — Read `documents/NAT.md` and `documents/RL_ARCHITECTURE.md`, then review `src/runtime/`. Focus: agent loop, tool registration, skill surfaces, NAT patterns, malformed call handling, workflow config.

3. **rollouts/** — Read `documents/RL_ARCHITECTURE.md` and `documents/NVIDIA_SOFTWARE_MAPPING.md`, then review `src/rollouts/` and `src/envs/nemo_gym_adapter.py`. Focus: NeMo Gym three-server architecture, rollout collection, serialization, session management, episode capture.

4. **training/** — Read `documents/RL_ARCHITECTURE.md` and `documents/NVIDIA_SOFTWARE_MAPPING.md`, then review `src/training/`. Focus: GRPO readiness, NeMo RL adapter, reward distribution profiling, train-eval consistency, on-policy compatibility, DatumSpec contracts.

5. **eval/ and cross-cutting** — Read `documents/RL_ARCHITECTURE.md`, then review `src/eval/`, `src/scenario_data.py`, and `src/shared/`. Focus: evaluation metrics, reward attribution traceability, contract durability, teachability, observability.

After collecting subagent findings, synthesize into a single review. Deduplicate findings that share root causes. Verify that cross-module issues (e.g., train-eval mismatches) are captured.

Return the review in this format:

```md
## Findings
- list only high-severity and medium-severity issues, ordered by severity
- if there are no such issues, state `No high-severity or medium-severity findings.`
- for each issue, include:
  - severity
  - affected file or code path
  - concrete evidence from the code
  - why the issue matters
  - a recommended fix that is specific and actionable

## Open Questions
- list assumptions, ambiguities, or missing context that affect confidence

## Evidence Reviewed
- list the concrete code paths, traces, or artifacts inspected
- say whether conclusions came from static code inspection, executable paths, serialized artifacts, or a combination

## Summary
- briefly summarize overall alignment with the scalable, transparent RL-demo goals
- mention residual risks or notable replayability, testing, teachability, and observability gaps
```
