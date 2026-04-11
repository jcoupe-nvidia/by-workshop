# by-workshop

Notebook-centric workshop repo for agentic supply-chain workflows on the NVIDIA stack.

The repository is being refactored from a custom notebook-led demo into a library that uses the actual NVIDIA stack for runtime orchestration, rollout collection, post-training, and scalable training systems. The target end state is one late-order-recovery scenario implemented with deterministic tools, explicit skills, structured tool calling, fallback handling, sequence-sensitive evaluation, and real NVIDIA-library-backed integration paths.

## What This Repo Demonstrates

- Multi-turn agent execution against a local OpenAI-compatible model endpoint
- Explicit higher-level skills composed from deterministic tools
- Nemotron-style structured tool calling with validation and dependency checks
- Repair/reject fallback handling for malformed tool outputs
- Sequence-sensitive evaluation and training-oriented export
- Ongoing migration to real `NeMo Agent Toolkit`, `NeMo RL`, `ProRL` rollout infrastructure, and `Megatron Bridge`

## Required NVIDIA Stack

The intended working environment for this repo now explicitly includes the actual NVIDIA libraries, not only local code that imitates their roles:

- `nvidia-nat` for NeMo Agent Toolkit runtime orchestration
- `nemo-gym` plus `prorl-agent-server` for rollout and agent-server infrastructure around ProRL-style collection
- `nemo-rl` from source for trainer-facing RL and post-training flows
- `megatron-bridge` for Megatron-backed training systems and checkpoint/interoperability surfaces

The current `environment.yaml` installs:

- package installs for `nvidia-nat`, `nemo-gym`, and `megatron-bridge`
- source-repo installs for `nemo-rl` and `prorl-agent-server`

This is stricter than the earlier repo state, which only depended on `requests` and relied on local scaffolding.

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
- `src/rollouts/`: canonical trace types plus rollout and ProRL-adapter work
- `src/training/`: NeMo RL-facing training semantics
- `src/systems/`: Megatron-Bridge-facing training systems
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
import nemo_rl
import nemo_gym
import prorl_agent_server
from megatron.bridge import AutoBridge

print("nemo_rl:", nemo_rl.__name__)
print("nemo_gym:", nemo_gym.__name__)
print("prorl_agent_server:", prorl_agent_server.__name__)
print("megatron_bridge:", AutoBridge.__name__)
PY
```

Notes:

- `NeMo Agent Toolkit` publishes the `nat` CLI through the `nvidia-nat` package.
- `NeMo RL` and `ProRL-Agent-Server` are currently installed from source repositories in `environment.yaml` because their upstream setup is source-first.
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
- canonical episodes and trajectory artifacts intended for NeMo RL consumption
- rollout artifacts intended for ProRL-style collection
- Megatron-Bridge-facing systems scaffolding for an `8x H100` target environment

The notebook writes generated training artifacts under `artifacts/` when those export cells are executed.

## Migration Status

The repo is still mid-migration from local stand-ins to real NVIDIA-library-backed paths.

- `Phase 1` remains largely valid, but its canonical contracts still need to be revalidated against the real rollout, trainer, and systems integrations.
- `Phase 2` must be reopened: the runtime package exists, but the executable runtime path still needs to be replaced with actual `NeMo Agent Toolkit` orchestration.
- `Phase 3` must be revisited: the explicit environment exists, but it is not yet the authoritative environment flowing through the real `NAT -> ProRL -> NeMo RL` execution path.
- `Phase 4` must also be reopened: the rollout package exists, but it still needs to be replaced or wrapped around actual `ProRL` rollout infrastructure.
- `Phases 5-8` remain the remaining planned work, with `Phases 5-6` carrying the core `NeMo RL` and `Megatron Bridge` integrations.

## Scope

This repository is a pedagogical artifact for a workshop, not a production system. It favors clarity, inspectability, deterministic behavior, and sequence correctness over realism or orchestration depth.
