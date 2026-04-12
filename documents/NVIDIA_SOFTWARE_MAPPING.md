# NVIDIA Software Mapping

## Purpose

This document maps the repo layers to the NVIDIA software used by the workshop and clarifies the responsibility of each layer.

Use `documents/RL_ARCHITECTURE.md` for the library-agnostic layer definitions. Use this document for the NVIDIA-specific mapping.

## Reference Package Versions

The workshop materials are aligned to the package versions observed in the reference `nemo-rl` environment:

| Package | Version | Notes |
| --- | --- | --- |
| `nvidia-nat` | `1.6.0` | Installed package for `NVIDIA NeMo Agent Toolkit` |
| `nemo-gym` | `0.2.0` | Environment and rollout integration layer |
| `nemo-rl` | `0.5.0rc0` | Training integration layer used in this repo |

## Layer Mapping

| Layer | NVIDIA software | Responsibility |
| --- | --- | --- |
| shared contracts | None; repo-owned | Canonical task contracts, datum schemas, trace and event schemas, sequence rules, deterministic business-tool semantics, and offline evaluation logic shared across all integrations. |
| `runtime/` | `NeMo Agent Toolkit (NAT)` | Interactive agent runtime, tool registration, structured tool calls, skill discovery, skill loading, prompt/runtime policy, and single-episode execution surfaces. |
| `envs/` | `NeMo Gym` | Environment-backed task execution, task-state transitions, action and tool-precondition verification, session-scoped external state, and task verification / reward-producing environment logic over repo-defined task contracts. |
| `rollouts/` | `NeMo Gym` | Multi-turn rollout orchestration and collection, environment-backed tool execution, and rollout lifecycle management, with the repo preserving canonical trace formats, explicit failure and repair events, and adapters. |
| `training/` | `NeMo RL` | Trainer-facing dataset construction from canonical data and traces, GRPO-aligned task routing, reward views and targets, masking and weighting, curriculum staging, and experiment handoff defining what gets optimized. |
| `eval/` | None; repo-owned | Offline metrics, regression summaries, and sequence-sensitive scoring over canonical traces and artifacts from the rest of the stack. |

## Responsibility Split

### Repo-owned responsibilities

The repo remains the source of truth for:

- deterministic business tools
- task semantics and success criteria
- canonical task, datum, trace, and event schemas
- sequence rules and machine-checkable dependencies
- explicit task routing contracts
- offline evaluation logic
- notebook-independent architecture and contracts

### NVIDIA software responsibilities

The NVIDIA software should be used as follows:

- `NAT` powers the interactive runtime surface.
- `NeMo Gym` powers environment-backed execution, verification, and rollout collection.
- `NeMo RL` powers trainer-facing learning views and optimization handoff.

These integrations should stay narrow and adapter-based. The software should consume repo-defined contracts rather than redefine the task.

## Software-Specific Notes

### `NeMo Agent Toolkit (NAT)`

`NAT` is the runtime integration layer for the workshop.

Use it for:

- workflow and agent execution
- tool registration and invocation
- runtime orchestration
- observability hooks
- interactive demo or user-facing execution surfaces

Do not use it as the source of truth for task semantics, trace contracts, or reward definitions. NAT is framework-agnostic and is designed to connect agents, tools, workflows, and observability rather than define the task itself. :contentReference[oaicite:1]{index=1}

### `NeMo Gym`

`NeMo Gym` is the environment and rollout integration layer.

Use it for:

- agent-server orchestration of multi-step and multi-turn rollouts
- routing model outputs and tool calls during training-time execution
- resources-server task execution and per-rollout session state
- task verification and reward-returning environment logic
- rollout collection over environment-backed tasks

This maps cleanly to the repo split where `envs/` own task truth and `rollouts/` own faithful episode capture. In NeMo Gym, the Agent server orchestrates the rollout lifecycle, while the Resources server owns tasks, tools, state, and verification. :contentReference[oaicite:2]{index=2}

### `NeMo RL`

`NeMo RL` is the training integration layer.

Use it for:

- trainer-facing dataset construction
- GRPO-aligned data processing
- canonical per-datum trainer inputs
- task-aware routing into processors and environments
- masking, weighting, and trainer-facing reward views
- curriculum staging and experiment configuration

NeMo RL expects a canonical per-example structure such as `DatumSpec`, including fields like `message_log`, `extra_env_info`, `loss_multiplier`, and `task_name`. It also uses explicit task routing from `task_name` into processor and environment mappings. That makes it the right fit for `training/`, not for runtime or task-definition ownership. :contentReference[oaicite:3]{index=3}

## Quick Rules

- Do not move task semantics from the repo into framework-specific code.
- Do not let the notebook become the only place where task logic or schemas exist.
- Do not let training code redefine runtime or environment responsibilities.
- Do not let evaluation duplicate environment transitions or task verification logic.
- Do not hide malformed calls, rejects, retries, or repairs inside framework adapters.
- Prefer thin adapters from repo contracts into `NAT`, `NeMo Gym`, and `NeMo RL`.
- Keep `task_name` and other routing fields repo-owned and explicit.
- Keep canonical datum and trace schemas repo-owned even when exporting to trainer-specific formats.

## One-Line Mapping

- `NAT` runs the agent.
- `NeMo Gym` executes and verifies the task during rollouts.
- `NeMo RL` turns canonical experience into trainer-facing learning signals.
- The repo remains the source of truth for the task.