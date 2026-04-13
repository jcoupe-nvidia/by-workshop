## Findings

### HIGH-1: GRPO environment replays entire message log from scratch on every step — O(N²) and stateless

**Severity:** High

**Affected file:** `src/training/run_grpo_training.py`, lines 94–173 (`_score_tool_call_sequence`) and lines 208–270 (`LateOrderRecoveryEnv.step`)

**Evidence:** The NeMo RL `LateOrderRecoveryEnv.step()` method calls `_score_tool_call_sequence(message_log)` on every generation step. `_score_tool_call_sequence` creates a **new** `RepoEnv()`, resets it, and replays **all** assistant messages from the beginning of the episode each time. On step N, it replays all N previous messages. This produces O(N²) tool executions per episode and destroys any persistent environment state between NeMo RL step calls.

```python
# line 106-107 — fresh env every call
env = RepoEnv()
env.reset("SO-10482")
# line 112 — replays entire message_log
for msg in message_log:
```

**Why it matters:** This is the only path that performs real GRPO training with NeMo RL. The quadratic overhead limits scalability as episode length grows. More critically, each step's reward is the **cumulative** total since episode start, not the **marginal** reward for the latest action. NeMo RL expects per-step marginal rewards for advantage computation in GRPO. Feeding cumulative sums as if they were per-step rewards inflates early advantages and distorts the policy gradient.

**Recommended fix:** Persist environment state across step calls by storing it in `self` on the Ray actor. On each step, advance only the **new** assistant message(s) through `env.step()`, compute the **marginal** reward (current step total minus previous total), and return that as the step reward.

---

### HIGH-2: `shared/tool_schemas.py` creates a transitive `runtime/` dependency for `training/`

**Severity:** High

**Affected files:** `src/shared/tool_schemas.py` (line 27), `src/training/nemo_rl_adapter.py` (line 38), `src/training/run_grpo_training.py` (line 81)

**Evidence:** `shared/tool_schemas.py` imports `TOOL_REGISTRY` and `TOOL_DEPENDENCIES` from `src.runtime.tools`:

```27:27:src/shared/tool_schemas.py
from src.runtime.tools import TOOL_REGISTRY, TOOL_DEPENDENCIES
```

`training/nemo_rl_adapter.py` and `training/run_grpo_training.py` both import from `shared/tool_schemas.py`. This means the training layer transitively depends on runtime at module load time. RL_ARCHITECTURE.md requires that training "must not become a second runtime" and that frameworks connect through "thin adapters." The `shared/` module was explicitly created so that "either consumer" does not need to "import the other's framework-specific code," yet it imports directly from runtime.

Additionally, `run_grpo_training.py` has its own direct import at line 147:

```147:147:src/training/run_grpo_training.py
        from src.runtime.tools import TOOL_REGISTRY
```

**Why it matters:** A training-layer change could break on runtime-layer refactoring, and the boundary becomes harder to teach as a clean split. During distributed GRPO training, loading `runtime/` modules (including the `requests` library, prompt builders, and agent loop code) into Ray workers is unnecessary and could cause import errors in environments where only the env/training packages are installed.

**Recommended fix:** Move `TOOL_REGISTRY`, `TOOL_DEPENDENCIES`, and tool function implementations out of `runtime/tools.py` into `shared/` or a new `src/tools/` module. Make `runtime/tools.py` a thin re-export for backward compatibility. This eliminates the circular dependency path. Alternatively, make `shared/tool_schemas.py` self-contained by registering tool metadata (names, params, descriptions, dependencies) directly, without importing the function objects.

---

### HIGH-3: `execution.py` records `succeeded=True` before re-validation confirms success

**Severity:** High

**Affected file:** `src/runtime/execution.py`, lines 82–92

**Evidence:** When `try_repair` returns `REPAIRED`, the code records a successful repair attempt **before** re-validating:

```82:92:src/runtime/execution.py
        if fb.action == FallbackAction.REPAIRED and fb.repaired:
            if recorder is not None:
                recorder.record_repair_attempt(
                    original_output=raw_output,
                    repaired_output=fb.repaired,
                    repairs_applied=fb.repairs_applied,
                    succeeded=True,           # ← recorded before re-validation
                )
            result = validate_tool_call(fb.repaired, tool_registry)  # ← may fail
            fallback_result = fb
            was_repaired = True
```

If `validate_tool_call(fb.repaired, tool_registry)` returns a `ValidationError`, the trace will contain a `TOOL_REPAIR_ATTEMPT` event with `succeeded=True`, followed by a `TOOL_VALIDATION_ERROR` for the same turn. The `was_repaired=True` flag is also set unconditionally. This inconsistency propagates into env reward computation (via `enrich_episode`), training DatumSpec construction, and offline evaluation.

**Why it matters:** The canonical trace is the source of truth for reward attribution, offline evaluation, and workshop teaching. A repair marked as succeeded that actually failed creates a misleading trace that rewards the wrong behavior during GRPO training and produces incorrect recovery-quality evaluation scores.

**Recommended fix:** Move the `record_repair_attempt` call to after re-validation, and set `succeeded` based on whether the re-validated result is a `ParsedToolCall` or `ParsedFinalAnswer`:

```python
result = validate_tool_call(fb.repaired, tool_registry)
re_valid = isinstance(result, (ParsedToolCall, ParsedFinalAnswer))
if recorder is not None:
    recorder.record_repair_attempt(
        original_output=raw_output,
        repaired_output=fb.repaired,
        repairs_applied=fb.repairs_applied,
        succeeded=re_valid,
    )
if re_valid:
    fallback_result = fb
    was_repaired = True
```

---

### MEDIUM-1: `run_grpo_training.py` uses a stripped-down parser that diverges from runtime semantics

**Severity:** Medium

**Affected file:** `src/training/run_grpo_training.py`, lines 176–190 (`_try_parse_tool_call`)

**Evidence:** The training environment uses `_try_parse_tool_call`, a private JSON parser that only attempts `json.loads` on the full string or individual lines. It does not use the shared `validate_and_repair` pipeline, the repair heuristics in `runtime/fallbacks.py`, or the structural checks in `runtime/schemas.py`. This means a model output that would be successfully repaired during interactive runtime (e.g., trailing commas, extra text wrapping JSON, fuzzy tool names) will be treated as `None` (unparseable) during training scoring.

**Why it matters:** Training-time reward signals will penalize outputs that the interactive runtime would accept. This creates a distribution mismatch between training rewards and evaluation metrics, which is a direct obstacle for GRPO training alignment. The repo's design rule states that "identical validation semantics" should hold between interactive and training-time paths — `nemo_gym_rollouts.py` correctly uses `validate_and_repair` for this purpose, but `run_grpo_training.py` does not.

**Recommended fix:** Replace `_try_parse_tool_call` with a call to `validate_and_repair` from `src.runtime.execution`, or at minimum use `validate_tool_call` from `src.runtime.schemas` to maintain parsing parity. If the runtime import is unacceptable in the training path (per HIGH-2), extract the parsing logic into a shared module.

---

### MEDIUM-2: `_live_train` always raises `RuntimeError` — notebook cannot demonstrate real GRPO training

**Severity:** Medium

**Affected file:** `src/training/grpo_notebook.py`, lines 391–408

**Evidence:** The `_live_train` function unconditionally raises `RuntimeError` after importing `grpo_train`:

```401:408:src/training/grpo_notebook.py
    from nemo_rl.algorithms.grpo import grpo_train  # noqa: F401

    raise RuntimeError(
        "Live NeMo RL training requires a fully configured policy model, "
        "tokenizer, and distributed GPU resources. Use the CLI entrypoint "
        "(python -m src.training.run_grpo_training) with the appropriate "
        "Hydra config for full training."
    )
```

`run_grpo_notebook(dry_run=False)` always falls through to `_dry_run_train`, meaning the notebook's GRPO section never performs real training. The notebook's GRPO cell uses `dry_run=True` by default.

**Why it matters:** CLAUDE.md lists "one GRPO training run" as an in-scope requirement. The notebook path only produces dry-run mock metrics. The real training path exists in `run_grpo_training.py` (a CLI entrypoint), but it has architectural issues (HIGH-1, MEDIUM-1) and cannot be exercised from the notebook. For a workshop demo, the audience sees mock numbers labeled as training output, which reduces credibility and teachability.

**Recommended fix:** Either (a) implement `_live_train` to perform a single-step GRPO update using the datum group when GPU resources are available, or (b) document clearly in the notebook output that the training step is simulated and that `run_grpo_training.py` is the real entrypoint, removing the `dry_run=False` path entirely to avoid confusion.

---

### MEDIUM-3: `enrich_episode()` mutates the input Episode in place without idempotency guard

**Severity:** Medium

**Affected file:** `src/rollouts/episode_runner.py`, lines 127–242

**Evidence:** `enrich_episode()` directly mutates the `Episode` object: it sets `episode.env_state_init`, sets `event.reward` on individual events, and overwrites `episode.metrics.total_reward`. There is no check for whether the episode has already been enriched:

```159:162:src/rollouts/episode_runner.py
    if not episode.env_state_init:
        episode.env_state_init = env.get_initial_state_snapshot()
```

The `env_state_init` guard only prevents overwriting the init snapshot. Reward fields are always overwritten. Calling `enrich_episode()` twice on the same episode object will double-count env state effects (the env replays all events again) and overwrite reward values with potentially different amounts (if intermediate state drifted).

**Why it matters:** `enrich_episode` is called from at least four places: `run_enriched_episode`, `run_enriched_episode_nat`, test fixtures in `conftest.py`, and notebook cells. If a test or notebook accidentally enriches an already-enriched episode, the rewards will be silently corrupted. For a teaching example, this is a debugging trap.

**Recommended fix:** Add an idempotency check — e.g., set `episode.metadata["enriched"] = True` after enrichment and raise or return early if it is already set. Alternatively, make `enrich_episode` work on a deep copy of the episode.

---

### MEDIUM-4: No formal schema for the Nemotron-style outer tool-call envelope

**Severity:** Medium

**Affected files:** `src/shared/tool_schemas.py`, `src/runtime/schemas.py`

**Evidence:** The canonical Nemotron-style envelope `{"thought": ..., "tool_call": {"name": ..., "arguments": {...}}}` and the final-answer envelope `{"thought": ..., "final_answer": {...}}` are defined only in:
- The system prompt text in `build_default_system_prompt()` (lines 137–158 of `tool_schemas.py`)
- Ad-hoc dictionary key checks in `validate_tool_call()` in `schemas.py`

Neither envelope has a Pydantic model or JSON Schema definition. The per-tool **argument** schemas are properly modeled via Pydantic (`GetOrderInput`, etc.), but the wrapping structure that every model response must conform to is validated only via string-level heuristics.

**Why it matters:** Changes to the envelope format (e.g., adding a new key, changing `tool_call` to `function_call` for provider compatibility) require coordinated edits across prompt text, parsing code, and fallback heuristics. A formal schema would make the contract machine-checkable and durable, which is especially valuable for a workshop teaching structured tool calling.

**Recommended fix:** Add Pydantic models (e.g., `NemotronToolCallEnvelope`, `NemotronFinalAnswerEnvelope`) to `shared/tool_schemas.py` and use them in `validate_tool_call()` for primary validation, falling back to heuristic parsing only in the fallback/repair layer.

---

### MEDIUM-5: `run_grpo_training.py` hardcodes `"SO-10482"` and lacks multi-scenario support

**Severity:** Medium

**Affected file:** `src/training/run_grpo_training.py`, lines 54–76 and line 107

**Evidence:** `_score_tool_call_sequence` always calls `env.reset("SO-10482")` regardless of which `SCENARIO_PROMPTS` prompt was selected. The dataset yields prompts for SO-10483, SO-10484, and SO-10485, but the environment always evaluates against SO-10482's data. Orders SO-10483 through SO-10485 do not exist in `scenario_data.py`, so `make_initial_state` will fail or return empty data when the env tries to look them up.

**Why it matters:** Multi-prompt training data combined with single-scenario scoring means 75% of training examples will receive incorrect rewards (the env evaluates tool calls against the wrong order's data). This silently corrupts the GRPO training signal. For a workshop, this also confuses the audience about what the training loop is actually optimizing.

**Recommended fix:** Either (a) extract the order ID from the prompt text at scoring time and pass it to `env.reset()`, or (b) reduce `SCENARIO_PROMPTS` to the single SO-10482 prompt that has backing data. The former is more realistic; the latter is simpler and sufficient for an MVP.

---

### MEDIUM-6: `eval_task_success` checks structural completeness but not factual correctness

**Severity:** Medium

**Affected file:** `src/eval/metrics.py`, `eval_task_success` function

**Evidence:** The evaluator checks:
- `episode.is_complete` is True
- `terminal.final_answer` exists
- `final_answer` has keys `action` and `rationale`

It does not verify whether the recommended `action` is a correct or reasonable mitigation for the scenario (e.g., whether it matches available inventory, lead times, or feasibility from the scenario data). A plausible-sounding but factually wrong recommendation receives a perfect task-success score.

**Why it matters:** CLAUDE.md and RL_ARCHITECTURE.md emphasize that "a correct-looking final answer reached through an invalid sequence should not automatically count as fully successful." The current evaluator enforces sequence constraints but not answer correctness. For workshop teaching, this gap undermines the lesson that evaluation should be end-to-end.

**Recommended fix:** Add a factual-correctness check: verify that the recommended action references a feasible option (e.g., one that appeared in `score_recovery_options` output) and that the `expected_delivery` date is consistent with the scenario data. This can be a partial-credit component within the existing `task_success` dimension.

---

## Open Questions

1. **Is the O(N²) replay in `run_grpo_training.py` intentional for simplicity?** The sliding-puzzle NeMo RL example it was modeled after may use a similar pattern for short episodes. If episodes are bounded to ~10 steps, the overhead is manageable, but the marginal-vs-cumulative reward issue remains regardless.

2. **Should `shared/tool_schemas.py` own tool metadata independently of `runtime/tools.py`?** Moving tool implementations to `shared/` would simplify the dependency graph but might conflict with the intent that `runtime/` owns tool registration and `shared/` only owns schemas.

3. **Is the `_live_train` RuntimeError a deliberate scope decision or a placeholder?** If the intent is that the notebook only demonstrates dry-run training, the code and docs should say so unambiguously.

4. **The `nemo_gym_rollouts.py` module also imports from `runtime/` (tools, schemas, fallbacks, execution, tracing). Per the NVIDIA mapping, rollouts are NeMo Gym-owned.** Does the architectural intent allow rollouts to depend on runtime for validation semantics, or should the shared execution pipeline be extracted into a neutral module?

5. **The docstring in `eval/metrics.py` (line 21) references `src.runtime.tools.TOOL_DEPENDENCIES` but the actual import (line 35) correctly uses `src.envs.state.TOOL_DEPENDENCIES`.** This is a documentation-only inconsistency — no code impact, but could mislead a reviewer.

## Evidence Reviewed

**Static code inspection (all files read in full):**
- `src/runtime/`: `__init__.py`, `agent.py`, `execution.py`, `fallbacks.py`, `nat_llm.py`, `nat_tools.py`, `prompts.py`, `schemas.py`, `tools.py`, `tracing.py`, `workflows.py`, `workflow_config.yaml`, `skills/__init__.py`, `skills/api.py`, all four `SKILL.md` files
- `src/envs/`: `__init__.py`, `late_order_env.py`, `nemo_gym_adapter.py`, `rewards.py`, `state.py`, `transitions.py`, `validators.py`
- `src/rollouts/`: `__init__.py`, `episode_runner.py`, `export_adapters.py`, `nemo_gym_rollouts.py`, `scripted_traces.py`, `serializers.py`, `trace_types.py`
- `src/training/`: `__init__.py`, `curriculum.py`, `datasets.py`, `experiments.py`, `grpo_config.yaml`, `grpo_notebook.py`, `nemo_rl_adapter.py`, `reward_plots.py`, `reward_views.py`, `run_grpo_training.py`
- `src/eval/`: `__init__.py`, `metrics.py`, `reports.py`
- `src/shared/`: `__init__.py`, `tool_schemas.py`
- `src/main.py`, `src/scenario_data.py`, `src/__init__.py`
- `tests/`: all 11 test files including `conftest.py`
- `notebooks/late_order_recovery_workshop.ipynb` (structural cell inventory)
- `artifacts/review-smoke/`: `atif_trajectories.jsonl`, `grpo_trajectory_group.jsonl`
- All documents: `CLAUDE.md`, `PLAN.md`, `documents/RL_ARCHITECTURE.md`, `documents/NVIDIA_SOFTWARE_MAPPING.md`, `documents/NAT.md`, `documents/llm-access.md`, `README.md`

**Executable paths traced:**
1. **Successful multi-turn episode:** `run_agent_episode` → `EpisodeRecorder` events → `enrich_episode` (env replay) → `episode_to_datum_spec` → `build_grpo_datum_group` → `save_datum_group_jsonl` — confirmed via `build_successful_episode` in scripted traces and `test_grpo_notebook.py`.
2. **Malformed/repair/reject path:** `build_repair_episode` → `enrich_episode` with `TOOL_VALIDATION_ERROR` / `TOOL_REPAIR_ATTEMPT` / `TOOL_REJECT` events → reward annotation → DatumSpec with repair metadata — confirmed via `test_training_integration.py` and `test_eval.py`.
3. **Runtime → canonical trace → trainer view → eval:** `Episode` (trace_types) → `serializers` round-trip → `episode_to_training_trajectory` → `episode_to_datum_spec` → `evaluate_trajectory` — confirmed via `test_rollouts.py` and `test_training_integration.py`.
4. **NeMo Gym resource server path:** `collect_server_backed_rollouts` → `LateOrderResourceServer.seed_session` → `verify` → `process_agent_actions` → `build_episode` → `enrich` → `episode_to_nemo_gym_row` — confirmed via `test_training_integration.py`.
5. **NeMo RL GRPO path (static):** `run_grpo_training.py` → `LateOrderRecoveryEnv.step` → `_score_tool_call_sequence` → `RepoEnv.step` — reviewed statically; not executable without GPU cluster.

**Test suite:** All 187 tests pass (pytest, 1.27s).

## Summary

The repository demonstrates strong architectural discipline across the five-layer split (`runtime/`, `envs/`, `rollouts/`, `training/`, `eval/`). Canonical trace types, explicit event vocabularies, deterministic tools, and the enrichment pipeline are well-designed and would teach clearly in a workshop. The NAT skill architecture, ATIF export, and NeMo Gym resource-server integration follow the documented best practices. Test coverage is broad, including repair/reject paths, cross-layer contract assertions, and isolated tool tests.

The primary risks center on GRPO training readiness:

- **The full NeMo RL training path** (`run_grpo_training.py`) has three compounding issues: O(N²) stateless replay producing cumulative rather than marginal rewards (HIGH-1), a private parser that diverges from runtime validation semantics (MEDIUM-1), and a hardcoded order ID that misaligns multi-prompt training data with single-scenario scoring (MEDIUM-5). Together, these mean that a real GRPO training run would produce distorted policy gradients.

- **The notebook GRPO path** always falls back to dry-run mock metrics (MEDIUM-2), so the audience never sees actual training behavior.

- **A premature `succeeded=True` in the repair pipeline** (HIGH-3) can corrupt canonical traces that feed both training rewards and offline evaluation, undermining the repo's core "faithful trace" contract.

- **The `shared/` → `runtime/` import chain** (HIGH-2) is a structural boundary violation that makes it harder to deploy training code independently and harder to teach the layer split cleanly.

Observability, serialization, offline evaluation, and the scripted-episode teaching flow are solid. The remaining gaps — no formal envelope schema (MEDIUM-4), mutable enrichment without idempotency (MEDIUM-3), and structural-only task-success evaluation (MEDIUM-6) — are real but lower-impact for the workshop's current scope.
