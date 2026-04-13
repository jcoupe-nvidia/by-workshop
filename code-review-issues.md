# Code Review: Late Order Recovery Workshop Repository

Review date: 2026-04-13
Scope: Full source tree on `main` branch. Primary objective: whether the repo supports a scalable, transparent, replayable, and workshop-teachable RL workflow for a multi-turn agent.

---

## Findings

### HIGH-1: False-alarm optimal sequence is unreachable under the dependency graph

**Severity:** HIGH
**Files:** `src/envs/state.py`, `src/envs/rewards.py`, `src/eval/metrics.py`

**Evidence:** `TOOL_DEPENDENCIES` in `state.py` unconditionally requires `find_alternate_inventory` before `score_recovery_options`:

```python
"score_recovery_options": {"find_alternate_inventory"},
```

But the false-alarm optimal sequence in `rewards.py` skips `find_alternate_inventory`:

```python
_FALSE_ALARM_SEQUENCE = [
    "get_order",
    "get_shipment_status",
    "get_inventory",
    "get_fulfillment_capacity",
    "score_recovery_options",
    "recommend_action",
]
```

`check_preconditions` in `transitions.py` enforces these dependencies, so `score_recovery_options` will be rejected unless `find_alternate_inventory` was called. The documented optimal path for SO-10485 / SO-10490 is not executable.

**Impact:** Composite verification and efficiency scoring are systematically wrong for false-alarm scenarios. Offline `eval_efficiency` uses `get_optimal_tool_sequence` as the baseline count, while `eval_sequence_correctness` uses `TOOL_DEPENDENCIES` — the same repo-owned ground truth implies incompatible targets. Training-time efficiency rewards in `compute_step_reward` are tied to the same inconsistent baseline. Agents must either violate the documented optimum or cannot progress legally.

**Fix:** Either relax `score_recovery_options` prerequisites to be scenario-aware (e.g. optional when alternates are unnecessary) or insert `find_alternate_inventory` into `_FALSE_ALARM_SEQUENCE`. Then verify that every "optimal" sequence is reachable under `check_preconditions`. Update `eval_efficiency` and env efficiency rewards to share one canonical definition of "optimal" under the same precondition graph.

---

### HIGH-2: Three incompatible definitions of `task_success` across layers

**Severity:** HIGH
**Files:** `src/eval/metrics.py`, `src/envs/nemo_gym_adapter.py`, `src/envs/rewards.py`, `src/training/nemo_rl_adapter.py`

**Evidence:** Four code paths define `task_success` differently:

1. **Offline eval** (`eval_task_success`): Structural checks (has terminal, required fields present) plus fuzzy grounding against `ranked_options` text. Does not call `get_expected_action`.
2. **NeMo Gym export** (`episode_to_nemo_gym_row`): Any non-empty `action` except `"escalate"` counts as success — no comparison to ground truth.
3. **Environment terminal reward** (`compute_terminal_reward`): Compares recommended action to `get_expected_action(order_id)` with fuzzy matching.
4. **DatumSpec export** (`nemo_rl_adapter.py`): Sets `task_success = 1 if episode.is_complete` — meaning "has a terminal outcome", not business success.

**Impact:** Train–eval and eval–profiler comparisons on "task success" are ambiguous. A rollout can maximize offline `eval_task_success`, emit high `task_success` in JSONL, and still get penalized by `compute_terminal_reward` (or the reverse). Teachability ("what was scored, which layer owned that decision") breaks across layers. The NeMo Gym `RewardProfiler` will show inflated success rates for wrong recommendations.

**Fix:** Define a single repo-owned predicate (e.g. `task_success_facts(episode | state)` in `envs/`) derived from the same helpers as `compute_terminal_reward`. Have `eval_task_success`, `episode_to_nemo_gym_row`, `nemo_rl_adapter`, and `evaluate_nemo_gym_result` all call into it — or use distinct, non-colliding names when different signals are intentional.

---

### HIGH-3: Terminal episode cleanup prevents post-collection export

**Severity:** HIGH
**Files:** `src/envs/nemo_gym_adapter.py`, `src/rollouts/nemo_gym_rollouts.py`

**Evidence:** On terminal episodes, `verify()` removes the session from `_sessions`:

```python
if session.env.is_terminal:
    session_id = getattr(body, "session_id", None)
    if session_id and session_id in _sessions:
        _sessions.pop(session_id, None)
```

`get_session_episode` only reads live sessions, and `export_session_episodes` depends on it:

```python
for sid in session_ids:
    ep = LateOrderResourceServer.get_session_episode(sid)
    if ep is not None:
        episodes.append(ep)
```

**Impact:** After a rollout finishes, the canonical `Episode` for that `session_id` can no longer be fetched. Any workflow that saves session IDs during collection and expects to export or inspect traces later gets empty results. This breaks durable observability, regression analysis, and "replay from saved artifacts" for the training-time path.

**Fix:** Persist the built `Episode` (or its JSONL line) before removing the session — e.g. append to a `completed_episodes` registry keyed by `session_id`, or write to disk in `verify()` when `is_terminal`. Alternatively, defer `_sessions.pop` until an explicit export acknowledgment or TTL after export.

---

### HIGH-4: Notebook `_live_train` ignores collected datum specs

**Severity:** HIGH
**File:** `src/training/grpo_notebook.py`

**Evidence:** `_live_train(datum_specs, stage_config)` accepts the GRPO datum group from NeMo Gym rollouts but never passes those specs into `setup()` or `grpo_train()`. Training uses freshly constructed `LateOrderDataset` / dataloaders. The `datum_specs` parameter appears only in the return value as `len(datum_specs)`:

```python
train_dataset = LateOrderDataset(tokenizer=tokenizer, length=ds_length)
# ...
grpo_train(policy, policy_generation, dataloader, ...)
return {
    "num_datum_specs": len(datum_specs),
    "training_backend": "nemo_rl",
}
```

**Impact:** With `dry_run=False`, the notebook path still runs generic on-the-fly GRPO over `LateOrderDataset` (random `SCENARIO_PROMPTS`), not the collected episodes, shaped rewards, or `build_grpo_datum_group` output. The documented "collect → build GRPO group → train" story is broken; the notebook "live" training is misleading relative to the artifacts produced in the same run.

**Fix:** Either feed the built `datum_specs` into the NeMo RL pipeline, or remove/fence `dry_run=False` / `_live_train` until a real ingestion path exists, and document that full training is only `python -m src.training.run_grpo_training` with `LateOrderDataset`.

---

### MEDIUM-1: `verify()` lacks defensive try/except, can crash rollout collection

**Severity:** MEDIUM
**File:** `src/envs/nemo_gym_adapter.py`

**Evidence:** `verify` calls `process_agent_actions` and builds the HTTP response with no surrounding `try/except`. The code review criteria and `RL_ARCHITECTURE.md` expect `verify()` to tolerate malformed grading inputs and default to `reward=0.0` on failure.

**Impact:** Unexpected exceptions in extraction or grading surface as transport/framework errors instead of a graded `reward=0.0`, breaking rollout collection stability.

**Fix:** Wrap the grading path in `try/except`, log the exception with session/correlation IDs, and return `BaseVerifyResponse(..., reward=0.0)`.

---

### MEDIUM-2: Reward distribution profiling callback is defined but never invoked

**Severity:** MEDIUM
**File:** `src/training/run_grpo_training.py`

**Evidence:** `_RewardProfilingCallback` is defined and instantiated, and the runner prints "Reward distribution profiling deferred to after first training step." But `maybe_profile` is never called — there is no registration with `grpo_train` or any hook.

**Impact:** `RL_ARCHITECTURE.md` calls for profiling reward distributions before GRPO. The main runner prints intent but does not run the check, so degenerate reward distributions go undetected.

**Fix:** Wire `maybe_profile` into an actual NeMo RL callback/post-process hook, or call `profile_reward_distribution` from a documented extension point after the first batch.

---

### MEDIUM-3: Dual reward semantics between online training and offline export

**Severity:** MEDIUM
**Files:** `src/training/run_grpo_training.py`, `src/training/nemo_rl_adapter.py`, `src/training/reward_views.py`

**Evidence:** Online GRPO uses marginal rewards from `LateOrderTrainingEnv.step()` (raw per-step `step_reward.total`). The adapter/export path applies curriculum-shaped scalar rewards via `build_episode_reward_view` (step vs trajectory blend, stage component weights, `combined_reward`).

**Impact:** The same repo exposes two trainer-facing objectives with different scalar semantics. Train–eval consistency and "clear reward boundaries" from `RL_ARCHITECTURE.md` are weakened.

**Fix:** Document the split explicitly (online = env marginal totals; offline JSONL = shaped `combined_reward`), or unify by applying the same `StageConfig` / `build_episode_reward_view` mapping inside the training env.

---

### MEDIUM-4: Business tool errors advance dependency state as if successful

**Severity:** MEDIUM
**File:** `src/envs/transitions.py`

**Evidence:** If `tool_result` contains `"error"`, `_update_discovery_facts` is skipped but the tool is still appended to `tools_called`:

```python
state.tools_called.append(tool_name)
# ...
if "error" in tool_result:
    return  # skips _update_discovery_facts only
```

Downstream tools become legally callable (dependencies satisfied) even when the upstream call returned a business error.

**Impact:** Verification rewards can credit "valid" steps and dependency satisfaction when the agent did not actually obtain usable observations. Process-quality and composite signals are diluted.

**Fix:** Treat tool-level business errors as first-class transition outcomes: avoid appending to `tools_called` for failed business outcomes, or set `StepResult.valid=False` with a dedicated `error_type`, requiring explicit repair/retry before dependencies unlock.

---

### MEDIUM-5: Dual agent execution paths risk behavioral drift

**Severity:** MEDIUM
**Files:** `src/envs/nemo_gym_adapter.py`, `src/rollouts/nemo_gym_rollouts.py`, `src/runtime/agent.py`

**Evidence:** The NeMo Gym path executes tools via `rollouts.nemo_gym_rollouts.process_agent_actions`. Interactive demos use `runtime/agent.py` with `EpisodeRecorder`. Both share `validate_and_repair`, but tool dispatch, metrics, and recorder behavior are independently implemented.

**Impact:** Two parallel orchestration paths must stay aligned manually. Any future change to tool dispatch, metrics, or recorder behavior in one path can diverge from the other — in tension with avoiding a "second framework" and with teachability ("one place" for agent execution).

**Fix:** Extract a single internal "execute validated tool call + record" function used by both paths, or add golden-trace tests that assert parity between them.

---

### MEDIUM-6: Multi-function-call responses silently drop extra calls without trace events

**Severity:** MEDIUM
**File:** `src/envs/nemo_gym_adapter.py`

**Evidence:** If the model returns multiple function calls in one response, only the first is processed; the rest are discarded with no recorder event:

```python
if len(function_calls) > 1:
    actions = [function_calls[0]]
```

**Impact:** The canonical trace no longer reflects the model's actual output. Dropped calls are invisible to evaluation and training, undermining sequence-sensitive fidelity.

**Fix:** Record a `TOOL_VALIDATION_ERROR` or metadata event listing omitted call names/IDs, even if only the first is executed.

---

### MEDIUM-7: Default system prompt prescribes a universal tool chain conflicting with scenario-specific optimal paths

**Severity:** MEDIUM
**File:** `src/shared/tool_schemas.py`

**Evidence:** `build_default_system_prompt` prescribes one universal sequence including alternate sourcing for all scenarios:

```
4. Follow the correct tool call sequence...
   - Then find_alternate_inventory to search other DCs
   - Then get_transfer_eta for transfer options...
```

This conflicts with `get_optimal_tool_sequence` which defines shorter false-alarm paths.

**Impact:** The training/export prompt teaches a single path, while env rewards and eval encode scenario-dependent length and tools. Models will over-call tools on false-alarm orders.

**Fix:** Parameterize the prompt by scenario/task metadata, or replace the fixed sequence list with a general rule ("follow `TOOL_DEPENDENCIES` and stop when primary fulfillment is sufficient").

---

### MEDIUM-8: Simulation clock duplicated across three modules

**Severity:** MEDIUM
**Files:** `src/runtime/workflows.py`, `src/runtime/prompts.py`, `src/runtime/tools.py`

**Evidence:** The scenario "as of" date is hardcoded as `datetime(2026, 4, 10)` in `workflows.py`, `"Today's date is 2026-04-10"` in `prompts.py`, and the same `today` in `tools.py`.

**Impact:** Task-truth timing is duplicated in the runtime layer instead of a single repo-owned contract. Drift between prompt, diagnosis math, and recommendation ETA breaks machine-checkable consistency.

**Fix:** Centralize "simulation clock" in `scenario_data` or `envs` and import one value everywhere.

---

### MEDIUM-9: `ExperimentPlan` / curriculum infrastructure disconnected from training runners

**Severity:** MEDIUM
**Files:** `src/training/experiments.py`, `src/training/run_grpo_training.py`, `src/training/grpo_notebook.py`

**Evidence:** `build_default_experiment_plan()` builds a four-stage plan with `data_dir`, `checkpoint_from`, and stage-specific hyperparameters. Nothing in `run_grpo_training.py` or `grpo_notebook.py` imports or consumes `ExperimentPlan` or `ExperimentConfig`.

**Impact:** Curriculum and experiment definitions read as executable training design in code but are disconnected from actual GRPO entrypoints — risk of teaching/reviewer confusion and "phantom" staging.

**Fix:** Add a thin wiring layer or trim docs/strings to say the plan is illustrative until wired.

---

### MEDIUM-10: Workflow YAML `model` block unused; silent parse failure on malformed YAML

**Severity:** MEDIUM
**Files:** `src/runtime/workflows.py`, `src/runtime/workflow_config.yaml`, `src/runtime/agent.py`

**Evidence:** `workflow_config.yaml` documents model settings with env interpolation, but nothing in `workflows.py` consumes the `model` block. Interactive paths use hardcoded endpoints in `agent.py` and `nat_llm.py`. `_try_load_workflow_yaml()` returns `None` on any exception with a bare `except Exception`, silently falling back to filesystem-derived registries.

**Impact:** Declarative config is only partly authoritative. Silent YAML failure can change workflow ordering/transitions vs. intent with no signal.

**Fix:** Consume the `model` block or remove it; replace bare `except` with logged errors; add a test that the YAML parses and matches expected keys.

---

### MEDIUM-11: `run_skill_command` uses `python3` instead of venv interpreter

**Severity:** MEDIUM
**File:** `src/runtime/skills/api.py`

**Evidence:** Subprocess invoked as `["python3", str(script_path)]` instead of `sys.executable` or the venv path.

**Impact:** Repo rules require `/opt/nemo_rl_venv/bin/python`. System `python3` may be a different version or missing dependencies, breaking the reproducibility contract.

**Fix:** Use `sys.executable` or an explicit config path to the venv interpreter.

---

### MEDIUM-12: Advantage plot labels misstate the computation

**Severity:** MEDIUM
**Files:** `src/training/reward_plots.py`, `src/training/nemo_rl_adapter.py`

**Evidence:** `build_grpo_datum_group` stores z-scored, clamped values in `group_advantage`. Plots label advantages as "reward − group mean", but the actual computation is `(shaped_reward - group_mean) / group_std`, clamped to [-5, 5].

**Impact:** Teaching surfaces misstate what the numbers are. Tests assert z-scores sum to ~zero, which matches z-scoring, not "minus mean" only.

**Fix:** Update titles/labels to "z-scored group advantage (clamped)" or change the computation to match the documented baseline.

---

### MEDIUM-13: `_SESSION_MAX_SIZE` declared but never enforced

**Severity:** MEDIUM
**File:** `src/envs/nemo_gym_adapter.py`

**Evidence:** A concurrent-session cap is declared (`_SESSION_MAX_SIZE = 1000`) but never checked in `seed_session`. Only TTL eviction runs.

**Impact:** Memory can grow without the documented bound during long-running collection jobs.

**Fix:** Enforce `_SESSION_MAX_SIZE` in `seed_session` or remove the unused constant.

---

## Open Questions

1. ~~**False-alarm design intent**~~ **Resolved:** Option A — relax `score_recovery_options` prerequisites so false-alarm agents can shortcut directly to scoring without `find_alternate_inventory`. The dependency graph in `TOOL_DEPENDENCIES` should make `find_alternate_inventory` an optional prerequisite for `score_recovery_options`, and `_FALSE_ALARM_SEQUENCE` stays as-is. HIGH-1 fix direction is confirmed.

2. ~~**Dual reward semantics intent**~~ **Resolved:** Not intentional. Online and offline reward semantics should be unified. MEDIUM-3 is confirmed as an actionable fix.

3. ~~**NeMo RL consumption of `group_advantage`**~~ **Resolved:** NeMo RL should compute its own baseline internally from grouped rollouts, not use precomputed z-scores. The `group_advantage` field in exported JSONL should be treated as informational/diagnostic only, not consumed by the trainer. If the field is currently wired into NeMo RL's advantage computation, it should be disconnected to avoid double-normalization.

4. ~~**ATIF field availability**~~ **Resolved:** ATIF validation/repair data should be mapped to canonical `EventType` entries. The `_atif_to_episode` conversion should enrich reconstructed episodes with these fields.

5. ~~**`verify()` exception risk**~~ **Resolved:** Wrap in try/except and log the exception while assigning `reward=0.0`. MEDIUM-1 is confirmed as an actionable fix.

## Evidence Reviewed

**Static code inspection** of all Python source files in:
- `src/envs/` — `state.py`, `rewards.py`, `transitions.py`, `tools.py`, `nemo_gym_adapter.py`, `__init__.py`, `environment.py`
- `src/runtime/` — `agent.py`, `execution.py`, `fallbacks.py`, `nat_llm.py`, `nat_tools.py`, `prompts.py`, `tools.py`, `tracing.py`, `workflows.py`, `workflow_config.yaml`, `atif_adapter.py`, `skills/api.py`
- `src/rollouts/` — `nemo_gym_rollouts.py`, `episode_runner.py`, `export_adapters.py`, `trace_types.py`
- `src/training/` — `run_grpo_training.py`, `grpo_notebook.py`, `nemo_rl_adapter.py`, `reward_views.py`, `reward_plots.py`, `curriculum.py`, `experiments.py`, `grpo_config.yaml`
- `src/eval/` — `metrics.py`, `reports.py`
- `src/shared/` — `tool_schemas.py`
- `src/scenario_data.py`
- Tests in `tests/` (for coverage assessment only)

Cross-module traces validated:
- **Successful episode path:** `seed_session` → `verify` loop → `process_agent_actions` → `env.step` → `compute_step_reward` → terminal → `Episode` build → `export_adapters` → `DatumSpec`.
- **Malformed/repair path:** Model output → `validate_and_repair` → `TOOL_REPAIR_ATTEMPT` / `TOOL_REJECT` events → `EpisodeRecorder` → `Episode.events`.
- **Training-facing view:** `Episode` → `training_record_to_datum_spec` → `build_episode_reward_view` → `build_grpo_datum_group` → JSONL. Separately: `LateOrderTrainingEnv.step` → marginal rewards → `grpo_train` (live path).

All conclusions from static code inspection; no runtime execution was performed.

## Summary

The repository demonstrates strong architectural separation between runtime, environment, rollout, training, and evaluation layers. The core agent loop, structured tool calling, malformed-output handling, and deterministic business tools are well-implemented and teachable. The NeMo Gym three-server mapping and NAT skill surfaces are correctly structured.

**Residual risks:**

- **Reward/evaluation coherence** is the primary gap. The false-alarm dependency inconsistency (HIGH-1) and triple `task_success` definition (HIGH-2) mean that composite verification, training efficiency signals, and offline metrics can simultaneously give contradictory signals for the same episode. This is the most important fix before using the repo as a teaching surface for "transparent, replayable RL."
- **Training pipeline completeness** has two gaps: the notebook's `_live_train` does not consume its own collected data (HIGH-4), and reward profiling is declared but inoperative (MEDIUM-2).
- **Observability durability** is weakened by terminal session cleanup deleting episodes before export (HIGH-3) and silent dropping of multi-function-call responses (MEDIUM-6).
- **Replayability** is generally strong but is undermined by the business-error dependency advancement issue (MEDIUM-4) and the dual execution paths (MEDIUM-5) that could diverge.
- **Teachability** is good at the individual-module level but suffers from scattered definitions of success criteria, duplicated simulation clock values, and disconnected curriculum infrastructure that appears executable but is not wired.
