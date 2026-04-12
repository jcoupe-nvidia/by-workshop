# Migration Plan

## Status

Phase 11 is complete. The training backend and trainer handoff have been migrated from `openpipe-art` to `nemo-rl`. The repo-owned responsibility split across `runtime/`, `envs/`, `rollouts/`, `training/`, and `eval/` is preserved.

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
11. Migrated the training backend from `openpipe-art` to `nemo-rl`:
    - Replaced `src/training/openpipe_art_adapter.py` with `src/training/nemo_rl_adapter.py`, converting Episodes and TrainingRecords into NeMo RL `DatumSpec` dicts instead of `art.Trajectory` / `art.TrajectoryGroup` objects.
    - Updated `src/training/grpo_notebook.py` to use DatumSpec groups instead of art trajectory groups, with dry-run as the primary demo path since full NeMo RL training requires distributed GPU resources.
    - Updated `src/training/__init__.py` and `src/main.py` to remove all `art`-specific naming from module exports, CLI flags (`--art-export` → `--nemo-rl-export`), and user-facing help text.
    - Retargeted `tests/test_grpo_notebook.py` and `tests/test_training_integration.py` to assert repo-owned training invariants against DatumSpec dicts instead of art-specific types.
    - Updated `README.md`, `documents/NVIDIA_SOFTWARE_MAPPING.md`, `src/shared/tool_schemas.py`, and `notebooks/late_order_recovery_workshop.ipynb` so the documented stack is `NAT` + `NeMo Gym` + `NeMo RL`, with `openpipe-art` references completely removed from code and docs.

## Target Outcome

The repository reflects the architecture described in `CLAUDE.md` and `documents/RL_ARCHITECTURE.md` while preserving the late-order-recovery scenario, deterministic tools, and end-to-end workshop flow. The notebook demonstrates a complete GRPO training handoff including reward visualization and NAT-compatible trace artifacts, with `nemo-rl` as the training backend.
