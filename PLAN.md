# Migration Plan

## Status

The current Phase 11 work is to migrate the training backend and trainer handoff from `openpipe-art` to `nemo-rl` while preserving the repo-owned responsibility split across `runtime/`, `envs/`, `rollouts/`, `training/`, and `eval/`.

Use `documents/RL_ARCHITECTURE.md` as the source of truth for the current responsibility split across `runtime/`, `envs/`, `rollouts/`, `training/`, and `eval/`.

## Completed History

1. Established canonical repo contracts and shared trace types.
2. Moved interactive agent behavior and skill surfaces into `src/runtime`.
3. Made task state, transitions, and reward-relevant facts explicit in `src/envs`.
4. Built canonical episode capture and serialization in `src/rollouts`.
5. Split training-facing datasets, reward views, and experiments into `src/training`.
6. Removed outdated systems assumptions and quarantined historical scale-out references.
7. Rebuilt offline evaluation on top of canonical traces and environment-owned semantics in `src/eval`.
8. Demoted the notebook to a consumer of library code instead of an architecture source of truth.
9. Aligned the public surfaces and documentation with the final layer split and target stack.
10. Replaced the notebook's export-only training discussion with a real GRPO training run: rollout collection, trajectory group assembly with group-relative advantages, training step execution via openpipe-art, ATIF trace export for NAT inspection, and reward distribution visualization.

## Phase 11 Plan

11.1. Confirm the installed `nemo-rl` API surface and identify the concrete equivalents for grouped GRPO inputs, training entry points, checkpoint outputs, and artifact serialization.
11.2. Replace `src/training/openpipe_art_adapter.py` with a trainer-neutral or `nemo-rl`-specific adapter that keeps `TrainingRecord`, reward views, curriculum staging, and GRPO grouping repo-owned.
11.3. Update `src/training/grpo_notebook.py` so the live GRPO path no longer imports `art` or checks `openpipe-art` versions, while preserving dry-run behavior and ATIF export for NAT inspection.
11.4. Update `src/training/__init__.py` and `src/main.py` to remove public `art` naming from module exports, CLI flags, and user-facing help text.
11.5. Retarget `tests/test_grpo_notebook.py` and `tests/test_training_integration.py` to assert repo-owned training invariants instead of `art`-specific types and metadata.
11.6. Update `README.md`, `CLAUDE.md`, `documents/NVIDIA_SOFTWARE_MAPPING.md`, and `notebooks/late_order_recovery_workshop.ipynb` so the documented stack is `NAT` + `NeMo Gym` + `nemo-rl`, with `openpipe-art` references completely removed.

## Target Outcome

The repository should continue to reflect the architecture described in `CLAUDE.md` and `documents/RL_ARCHITECTURE.md` while preserving the late-order-recovery scenario, deterministic tools, and end-to-end workshop flow. The notebook should still demonstrate a complete GRPO training handoff including reward visualization and NAT-compatible trace artifacts, but with `nemo-rl` as the training backend instead of `openpipe-art`.
