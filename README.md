# by-workshop

Notebook-centric workshop repo for agentic supply-chain workflows on the NVIDIA stack.

The repository is being refactored from a custom notebook-led demo into a library that uses the actual NVIDIA stack for runtime orchestration, rollout collection, post-training, and scalable training systems. The target end state is one late-order-recovery scenario implemented with deterministic tools, explicit skills, structured tool calling, fallback handling, sequence-sensitive evaluation, and real NVIDIA-library-backed integration paths.

## What This Repo Demonstrates

- Multi-turn agent execution against a local OpenAI-compatible model endpoint
- Explicit higher-level skills composed from deterministic tools
- Nemotron-style structured tool calling with validation and dependency checks
- Repair/reject fallback handling for malformed tool outputs
- Sequence-sensitive evaluation and training-oriented export
- Ongoing migration to real `NeMo Agent Toolkit`, repo-owned canonical rollouts, and `openpipe-art`, with older trainer-facing, rollout-shaping, and scale-out systems references preserved only as historical context where useful

## Active Environment Stack

The intended working environment for this repo now explicitly includes the active libraries used by the current notebook and refactor path:

- `nvidia-nat` for NeMo Agent Toolkit runtime orchestration
- `nemo-gym` for environment-oriented rollouts and reward inspection
- `openpipe-art` for training-oriented exports and post-training alignment

The current `environment.yaml` installs:

- package installs for `nvidia-nat`, `nemo-gym`, and `openpipe-art`

Earlier planning docs may still mention older trainer-facing, rollout-shaping, or scale-out systems framing because they originally explained the roles of training semantics, rollout infrastructure, and large-scale systems. Those references are now historical unless explicitly called out as active again.

## Core Scenario

The end-to-end example centers on customer order `SO-10482` for `1,200` units of `SKU-4090`, which is at risk of missing its committed delivery date.

The agent investigates whether the original source DC can still fulfill the order on time. If not, it evaluates mitigation paths such as:

- transfer from an alternate DC
- supplier expedite
- partial fulfillment
- substitute SKU recommendation when enabled

## Repository Layout

- `notebooks/late_order_recovery_workshop.ipynb`: main workshop notebook and live demo flow
- `documents/llm-access.md`: local model endpoint, model id, and smoke test
- `src/scenario_data.py`: synthetic in-memory orders, shipments, inventory, transfer lanes, supplier options, capacity, and substitutes
- `src/runtime/`: NAT-facing runtime package
- `src/envs/`: explicit task environment state, transitions, validators, and rewards
- `src/rollouts/`: canonical trace types plus repo-owned rollout and serialization logic
- `src/training/`: `openpipe-art`-facing training semantics
- `src/systems/`: legacy or historical systems scaffolding, if still present
- `src/eval/`: offline evaluation and reporting
- `src/main.py`: entrypoint for local checks and episode execution
- legacy top-level modules such as `src/tools.py` and `src/agent_loop.py` currently remain as compatibility shims during migration

## Execution Flow

The notebook keeps the workflow explicit and machine-checkable:

1. Diagnose order risk.
2. Assess primary fulfillment from the source DC.
3. Evaluate alternate recovery paths.
4. Score options and synthesize a recommendation.

The typical successful path uses `5-10` tool calls across a compact deterministic tool library.

## Quickstart

Run these commands from the repository root:

```bash
conda env create -f environment.yaml
conda activate by-workshop
python -m src.main --check-imports
jupyter lab
```

Then open `notebooks/late_order_recovery_workshop.ipynb`.

## Verify NVIDIA Libraries

After creating the environment, confirm the required stack is present:

```bash
nat --version
python - <<'PY'
from importlib.metadata import version
import nemo_gym

print("nemo_gym:", nemo_gym.__name__)
print("openpipe-art version:", version("openpipe-art"))
PY
```

Notes:

- `NeMo Agent Toolkit` publishes the `nat` CLI through the `nvidia-nat` package.
- `openpipe-art` is now the primary training-oriented package in `environment.yaml`.
- `nemo-gym` remains useful for explicit environment and reward-signal inspection even though the training-oriented path now centers `openpipe-art`.
- Upstream NVIDIA docs generally prefer `uv`-managed virtual environments for these libraries; this repo keeps a `conda` environment only as a workshop convenience wrapper.

## Local Model

The live agent loop expects a locally deployed OpenAI-compatible chat endpoint:

- endpoint: `http://0.0.0.0:8000/v1/chat/completions`
- model: `nvidia/nemotron-3-nano`

See `documents/llm-access.md` for the smoke test request and cache mapping.

## Minimal Python Example

You can run the live agent loop outside the notebook as well:

```python
from src.agent_loop import run_agent, print_trace_summary
from src.evaluation import evaluate_trajectory, print_evaluation

trace = run_agent("SO-10482", verbose=False)
print_trace_summary(trace)

evaluation = evaluate_trajectory(trace)
print_evaluation(evaluation)
```

## Outputs

When you run the later notebook sections, the repo can produce:

- worked successful and repair trajectories
- seven-dimension trajectory evaluations
- canonical episodes and trajectory artifacts intended for `openpipe-art`-oriented consumption
- rollout artifacts intended for the repo's own canonical trace collection layer
- historical notes about the earlier trainer-facing, rollout-shaping, and scale-out systems framing where they help explain the current architecture

The notebook writes generated training artifacts under `artifacts/` when those export cells are executed.

## Migration Status

The repo is still mid-migration from local stand-ins and older stack assumptions to the current NAT + canonical rollouts + `openpipe-art` path.

- `Phase 1` remains largely valid, but its canonical contracts still need to be revalidated against the real rollout and training/export integrations, plus any legacy systems notes that remain in the tree.
- `Phase 2` remains valid structurally, but its public wording and adapters should stop implying a handoff to an earlier rollout or trainer stack.
- `Phase 3` must be revisited so the environment and rewards are described as feeding canonical traces and `openpipe-art`, not an earlier `NAT -> rollout -> trainer` path.
- `Phase 4` must be revisited so the rollout layer is treated as the active repo-owned trace layer rather than a temporary stand-in for an external rollout stack.
- `Phase 5` must be revisited most heavily because the active training/export path should now be `openpipe-art`-first.

## Scope

This repository is a pedagogical artifact for a workshop, not a production system. It favors clarity, inspectability, deterministic behavior, and sequence correctness over realism or orchestration depth.
