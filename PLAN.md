# Migration History

## Status

The migration is complete.

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

## Outcome

The repository now reflects the architecture described in `CLAUDE.md` and `documents/RL_ARCHITECTURE.md` while preserving the late-order-recovery scenario, deterministic tools, and end-to-end workshop flow.
