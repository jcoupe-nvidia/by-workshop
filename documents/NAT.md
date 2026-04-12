# NeMo Agent Toolkit Best Practices for `by-workshop`

These practices are tailored to this repo’s current shape: a NAT-backed `runtime/`, repo-owned task semantics, explicit skills, deterministic tools, and sequence-sensitive evaluation. :contentReference[oaicite:0]{index=0}

## 1. Keep NAT as the runtime layer, not the task-truth layer
- Keep `runtime/` responsible for prompts, agent loop control, skill discovery/loading, structured tool execution, and observability.
- Keep `envs/` responsible for state, transitions, tool preconditions, terminal conditions, and reward-relevant facts.
- Do not let NAT configuration, adapters, or workflow code redefine task semantics that already belong to the repo. :contentReference[oaicite:1]{index=1}

## 2. Keep skills explicit, small, and bounded
- Preserve the current pattern of a small number of high-level skills with clear responsibilities.
- Each skill should represent a distinct business phase, not a grab-bag of unrelated tool calls.
- Avoid hiding orchestration logic inside prompts when it can live in explicit skill boundaries or deterministic validation. :contentReference[oaicite:2]{index=2}

## 3. Group related NAT functions when they share config or state
- If multiple tools share the same client, cache, credentials, or connection details, model them as NAT function groups instead of separate unrelated registrations.
- Use function groups to reduce duplicated configuration, avoid wasteful per-tool client construction, and keep related logic together. :contentReference[oaicite:3]{index=3}

## 4. Make workflow configuration declarative where it helps
- NAT workflows are defined through YAML configuration, so use config files for selectable models, middleware, evaluators, and runtime variants rather than hardcoding these choices deep in Python.
- Prefer environment-variable interpolation for deploy-time values such as endpoints, API keys, and model names. :contentReference[oaicite:4]{index=4}

## 5. Add middleware intentionally, not globally
- Use timeout middleware for LLM and tool calls so failures are explicit and bounded.
- Use cache middleware selectively, especially for evaluation runs, where NAT supports cache modes like `eval`.
- Do not add middleware that obscures tool behavior or makes debugging harder for this workshop repo. :contentReference[oaicite:5]{index=5}

## 6. Preserve malformed calls, rejects, retries, and repairs in traces
- Keep the repo’s current behavior of explicitly representing malformed calls, repair paths, reject paths, and sequence failures.
- Do not silently “fix” bad tool calls inside NAT wrappers or middleware without recording that event in rollout traces. :contentReference[oaicite:6]{index=6}

## 7. Treat observability as a first-class development surface
- Wire NAT observability into the runtime so traces show function execution details, latency, token usage, and request timelines.
- If this repo later composes child workflows or remote workflows, propagate parent/child trace linkage so the full execution appears as one connected tree. :contentReference[oaicite:7]{index=7}

## 8. Evaluate against datasets, not only hand-run demos
- Keep notebook demos, but use NAT evaluation flows for repeatable checks.
- Back evaluations with datasets and custom evaluators/loaders where needed so regression testing stays machine-checkable.
- This fits the repo’s existing emphasis on sequence correctness, tool validity, and recovery quality. :contentReference[oaicite:8]{index=8}

## 9. Unit test tools in isolation
- Test deterministic tools via direct function calls against the `TOOL_REGISTRY` without spinning up the full workflow stack (see `tests/test_nat_tool_runner.py`).
- This is especially important here because the repo’s business tools are deterministic and are the foundation for higher-level skill quality. :contentReference[oaicite:9]{index=9}

## 10. Profile runtime behavior before optimizing prompts or models
- NAT’s profiler collects per-invocation usage stats, latency, token usage, and workflow bottleneck data for offline analysis.
- Use profiling results to decide where to optimize model selection, concurrency, or call structure instead of guessing from a few interactive runs. :contentReference[oaicite:10]{index=10}

## 11. Prefer provider-agnostic, validated config surfaces
- When introducing model- or provider-specific options, use NAT’s gated-field pattern so unsupported options fail clearly rather than leaking across providers.
- This keeps config evolution safer as the repo experiments with different local or remote model backends. :contentReference[oaicite:11]{index=11}

## 12. Keep the notebook as a consumer, never the source of truth
- Continue treating notebooks as a teaching and inspection layer only.
- Canonical contracts, schemas, transitions, rewards, and evaluation logic should stay in repo code and docs. :contentReference[oaicite:12]{index=12}

---

## Repo-specific checklist

Before merging a NAT-related change, check:

- Does this change keep `runtime/` separate from `envs/`, `rollouts/`, `training/`, and `eval` responsibilities? :contentReference[oaicite:13]{index=13}
- Does it preserve the current explicit skill structure instead of hiding logic in prompts? :contentReference[oaicite:14]{index=14}
- Are tool schemas, sequence rules, and trace schemas still repo-owned and canonical? :contentReference[oaicite:15]{index=15}
- Are retries, rejects, malformed calls, and repairs still visible in traces? :contentReference[oaicite:16]{index=16}
- Should shared tools be converted into a NAT function group? :contentReference[oaicite:17]{index=17}
- Should the new behavior be controlled in YAML config instead of hardcoded Python? :contentReference[oaicite:18]{index=18}
- Does the change need timeout, cache, observability, evaluation, or profiling hooks? :contentReference[oaicite:19]{index=19}
- Did we add or update isolated tool tests? :contentReference[oaicite:20]{index=20}
