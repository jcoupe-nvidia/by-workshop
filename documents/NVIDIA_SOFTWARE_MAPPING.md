# NVIDIA Software Mapping

## Purpose

This document maps the repo layers to the NVIDIA software used by the workshop and clarifies the responsibility of each layer.

Use `documents/RL_ARCHITECTURE.md` for the library-agnostic layer definitions. Use this document for the NVIDIA-specific mapping.

## Layer Mapping

| Layer | NVIDIA software | Responsibility |
| --- | --- | --- |
| shared contracts | None; repo-owned | Canonical task semantics, deterministic business tools, trace and event schemas, sequence rules, and offline evaluation logic shared across all integrations. |
| `runtime/` | `NeMo Agent Toolkit (NAT)` | Interactive agent runtime, tool registration, structured tool calls, skill discovery, skill loading, and single-episode execution. |
| `envs/` | `NeMo Gym` | Environment validation surfaces, deterministic task-state transitions, action and tool-precondition verification, and training-time environment execution over repo-defined task contracts. |
| `rollouts/` | `NeMo Gym` | Multi-turn episode rollout collection and environment-backed execution, with the repo preserving canonical trace formats, explicit failure and repair events, and adapters. |
| `training/` | `openpipe-art` | Trainer-facing datasets, reward views, curriculum staging, experiment handoff, and post-training alignment surfaces that define what gets optimized. |
| `eval/` | None; repo-owned | Offline metrics, regression summaries, and sequence-sensitive scoring over canonical traces and artifacts from the rest of the stack. |

## Responsibility Split

### Repo-owned responsibilities

The repo remains the source of truth for:

- deterministic business tools
- task semantics and success criteria
- canonical trace and event schemas
- sequence rules and machine-checkable dependencies
- offline evaluation logic
- notebook-independent architecture and contracts

### NVIDIA software responsibilities

The NVIDIA software should be used as follows:

- `NAT` powers the interactive runtime surface.
- `NeMo Gym` powers training-time environment execution and rollout collection.
- `openpipe-art` powers trainer-facing learning views and handoff.

These integrations should stay narrow and adapter-based. The software should consume repo-defined contracts rather than redefine the task.

## Quick Rules

- Do not move task semantics from the repo into framework-specific code.
- Do not let the notebook become the only place where task logic or schemas exist.
- Do not let training code redefine runtime or environment responsibilities.
- Do not let evaluation duplicate environment transitions or reward shaping.
- Do not hide malformed calls, rejects, or repairs inside framework adapters.
- Prefer thin adapters from repo contracts into `NAT`, `NeMo Gym`, and `openpipe-art`.
