# by-workshop

Notebook-centric MVP for a workshop on agentic supply-chain workflows on the NVIDIA stack.

The repository demonstrates a real model-driven agent loop over deterministic tools for a single late-order recovery scenario, with explicit skills, structured tool calls, fallback parsing, sequence-sensitive evaluation, and training-oriented export.

## What This Repo Demonstrates

- Multi-turn agent execution against a local OpenAI-compatible model endpoint
- Explicit higher-level skills composed from deterministic tools
- Nemotron-style structured tool calling with validation and dependency checks
- Repair/reject fallback handling for malformed tool outputs
- Sequence-sensitive evaluation and training-oriented export aligned to NeMo RL, NVIDIA ProRL, and NVIDIA Megatron

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
- `src/tools.py`: deterministic tool implementations and registry
- `src/skills.py`: explicit higher-level skills and allowed tool-use patterns
- `src/schema.py`: canonical structured tool-call schema and validators
- `src/agent_loop.py`: OpenCode-inspired think/emit/validate/execute/observe loop
- `src/fallbacks.py`: repair/reject handling for malformed or mixed outputs
- `src/evaluation.py`: sequence-sensitive evaluators
- `src/training_export.py`: NeMo RL export, ProRL-style rewards, and Megatron config sketch

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
jupyter lab
```

Then open `notebooks/late_order_recovery_workshop.ipynb`.

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
- NeMo RL-style JSONL exports
- a conceptual Megatron training config for an `8x H100` target environment

The notebook writes generated training artifacts under `artifacts/` when those export cells are executed.

## Scope

This repository is a pedagogical artifact for a workshop, not a production system. It favors clarity, inspectability, deterministic behavior, and sequence correctness over realism or orchestration depth.
