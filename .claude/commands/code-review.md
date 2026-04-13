---
description: Review code for scalable, transparent RL workflows
---

Read `@PLAN.md`, `@CLAUDE.md`, `@documents/RL_ARCHITECTURE.md`, `@documents/NVIDIA_SOFTWARE_MAPPING.md`, and `@documents/NAT.md`.

Review the current state of the code on the `main` branch.

Do not limit the review to the working tree diff or branch-local changes.

Write the review findings and recommended fixes to `@code-review-issues.md`.

If `@code-review-issues.md` does not exist, create it using the template below. If it already exists, replace its contents with the completed review using the same section structure.

Do not commit `code-review-issues.md`; treat it as review output only.

Primary objective:

Determine whether the repository supports a scalable, transparent, replayable, and workshop-teachable RL workflow for a multi-turn agent, not just clean module boundaries.

Review priorities, in order from most important to least important:

1. Transparent, replayable, end-to-end RL workflow design, with a strong focus on GRPO readiness.
2. Correct responsibility boundaries across `runtime/`, `envs/`, `rollouts/`, `training/`, and `eval/`, including the NVIDIA software mapping and the NeMo Gym three-server architecture.
3. Verification and reward design quality: whether `verify()` is reliable, meaningful, and scalable, and whether the chosen verification strategy (trajectory matching, state matching, or composite) is appropriate for each task.
4. Multi-step environment correctness: session state isolation, error-as-observation, sequential tool execution, and on-policy training compatibility.
5. Demo teachability: whether the system is easy to explain to a workshop audience through concrete code paths and artifacts.
6. Strength and completeness of the observability and offline analysis surfaces.
7. Scalability risks that would make multi-turn rollout collection, trainer handoff, or regression analysis harder as the example grows.

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
  - `NeMo RL` owning trainer-facing datasets, reward views, and training handoff
  - `eval/` remaining repo-owned rather than being absorbed into training or runtime code
  - repo-owned task contracts staying canonical instead of being redefined inside framework-specific adapters
- check whether the NeMo Gym three-server architecture is correctly reflected, including:
  - Agent server responsibilities mapping to `runtime/` (orchestrates rollout lifecycle, calls model, routes tool calls, collects reward; does not run an LLM itself)
  - Model server treated as external and stateless (receives conversation, returns text or tool calls, no memory or orchestration)
  - Resources server responsibilities mapping to `envs/` (owns tasks, tools, per-rollout session state, and verification logic; returns reward signals)
  - the rollout lifecycle following Initialize (`seed_session`) -> Agent loop -> Grade (`verify`) phases
  - no confusion between which server owns orchestration vs task truth vs inference
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
- check whether verification and reward design follows the practices in `documents/RL_ARCHITECTURE.md`, including:
  - `verify()` is reliable (same rollout always gets the same score), meaningful (reflects actual task performance), and scalable (fast, local, no expensive external calls per-verification)
  - the verification strategy is deliberately chosen and documented: trajectory matching when correct sequence is unique and order matters; state matching when multiple valid paths exist; composite when both process and outcome matter
  - for the supply-chain scenario specifically, composite verification is the default (score both sequence correctness and outcome quality)
  - binary rewards (0/1) are used unless continuous rewards are explicitly justified
  - `verify()` guards against malformed model output with try/except and defaults to reward 0.0 on parse failure rather than crashing
  - structured diagnostic fields (e.g., accuracy, set_overlap, sequence_violations) accompany the scalar reward for debugging and evaluation
  - efficiency metrics (tool call count vs expected count) are tracked as a secondary signal
  - reward distribution is profiled before GRPO training to detect all-zero or all-one distributions that indicate verification or difficulty mismatches
- check whether multi-step environment patterns are correctly implemented, including:
  - session state isolation: all mutable per-episode state keyed by session identifier so concurrent rollouts cannot interfere
  - `seed_session()` uses direct assignment, not `setdefault`, to avoid silent re-seed bugs; `setdefault` is acceptable only in tool methods as a safe fallback
  - session state cleanup at episode end to avoid memory leaks during long training runs
  - sequential tool execution enforced when tool outputs inform subsequent calls (the core supply-chain pattern); datasets marked with `parallel_tool_calls: false`
  - tool execution errors returned as observations (not HTTP errors or exceptions that crash the episode) so the model can learn self-correction
  - tool errors logged in traces so evaluation can distinguish errors from successful calls
- check whether the codebase is compatible with on-policy multi-step training, including:
  - raw token IDs and log probabilities from prior model calls are preserved and propagated rather than relying on re-tokenization of detokenized text
  - rollout history is not modified between model calls unless the training pipeline explicitly accounts for non-monotonic prompts
  - on-policy token ID fixes are enabled or enableable for multi-step rollouts when using NeMo RL
  - these corrections are handled by the training framework integration, not by `envs/` or `runtime/`
- check whether canonical contracts are transparent and durable enough for analysis, including:
  - stable schema ownership for tool calls, events, traces, reward-relevant facts, and trainer-facing examples
  - serialization that preserves ordering, failures, retries, and stop reasons
  - enough structure to compare runs across regressions and training iterations
  - async metadata contract fields (generation weight version, training weight version, age/freshness, replay acceptance/rejection) when async GRPO is enabled
- check whether the code makes reward attribution and sequence-sensitive scoring easy to audit, including:
  - where reward-relevant facts are created
  - where those facts are transformed into reward or advantage inputs
  - whether offline evaluation reuses canonical semantics instead of silently drifting
  - whether verification logic lives in `envs/` where it can be reused by training and evaluation, not scattered across `runtime/` adapters or notebook cells
- check whether module names, APIs, and data flow make the roles of NAT, the repo's rollout layer, and `NeMo RL` easy to explain to a workshop or engineering audience while keeping older trainer-facing, rollout-shaping, and scale-out systems references clearly historical
- check whether the code and artifacts are teachable in an interactive demo, including:
  - whether a reviewer can point to the exact files and data structures that answer "what happened", "why did it happen", "what was scored", and "which layer owned that decision"
  - whether there are human-readable artifacts or summaries that make episode behavior understandable without reverse-engineering framework internals
  - whether adapters to `NAT`, `NeMo Gym`, and `NeMo RL` stay thin enough that the repo-owned contracts remain the main teaching surface
- check whether observability is strong enough for debugging and training analysis, including:
  - structured event tracing
  - clear episode and turn boundaries
  - failure and fallback visibility
  - serialization surfaces for later analysis
  - metrics, summaries, or hooks that support regression analysis
  - ability to inspect both raw canonical traces and derived trainer/eval views without ambiguity
- check whether NAT usage follows the best practices defined in `documents/NAT.md`, including:
  - NAT stays as the runtime layer and does not absorb task-truth, reward, or evaluation responsibilities
  - skills are explicit, small, and bounded with clear business-phase boundaries
  - related tools that share config or state use NAT function groups instead of duplicated registrations
  - workflow configuration is declarative (YAML-based) where appropriate instead of hardcoded in Python
  - middleware is added intentionally and does not obscure tool behavior or debugging
  - malformed calls, rejects, retries, and repairs are preserved in traces rather than silently fixed inside NAT wrappers
  - observability is wired into the runtime for function execution details, latency, and token usage
  - evaluation uses datasets and custom evaluators for repeatable checks, not only hand-run demos
  - deterministic tools have isolated unit tests via `nat.test.ToolTestRunner` or equivalent
  - model and provider configuration uses validated, provider-agnostic config surfaces
- check for the anti-patterns listed in `documents/RL_ARCHITECTURE.md`, including:
  - burying core task logic in notebook cells
  - hiding malformed tool calls, repairs, or rejects instead of tracing them explicitly
  - mixing runtime execution, rollout collection, reward shaping, and evaluation in one module
  - duplicating task validity or sequence semantics across runtime, training, and evaluation
  - letting training exports redefine the task contract
  - letting rollout code become a second framework with its own task semantics
  - letting framework adapters own routing, rewards, or success logic
  - silently consuming stale async trajectories without explicit policy
  - crashing the episode on tool execution errors instead of returning the error as an observation
  - sharing mutable state across concurrent rollouts instead of isolating state by session or episode identifier
  - using `setdefault` in session initialization where it silently ignores re-seed attempts
  - skipping reward distribution profiling before starting GRPO training
  - relying on re-tokenized or re-templated text as ground truth for on-policy training when raw token IDs are available
  - defining verification logic outside `envs/` (e.g., in `runtime/` adapters or notebook cells) where it cannot be reused by training and evaluation
  - adding platform-specific complexity that obscures the teaching example
- check whether SFT warm-start readiness exists when practical: a small set of high-quality demonstrations for chat template format, tool-call syntax, and general readability that could precede RL, avoiding wasted RL compute on format learning
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
