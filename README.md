# by-workshop

Notebook-centric workshop repo for agentic supply-chain workflows on the NVIDIA stack.

## What This Repo Demonstrates

- Multi-turn agent execution against a local OpenAI-compatible model endpoint
- Explicit higher-level skills composed from deterministic tools
- Nemotron-style structured tool calling with validation and dependency checks
- Repair/reject fallback handling for malformed tool outputs
- Sequence-sensitive evaluation and training-oriented export
- NAT-backed runtime, repo-owned canonical rollouts, and `openpipe-art`-facing training

## Core Scenario

The end-to-end example centers on customer order `SO-10482` for `1,200` units of `SKU-4090`, which is at risk of missing its committed delivery date.

The agent investigates whether the original source DC can still fulfill the order on time. If not, it evaluates mitigation paths such as:

- transfer from an alternate DC
- supplier expedite
- partial fulfillment
- substitute SKU recommendation when enabled

## Package Architecture

```
src/
├── runtime/         # NAT-facing single-episode runtime
│   ├── agent.py         model adapter + agent loop
│   ├── schemas.py       tool-call validation
│   ├── tools.py         deterministic business tools (7-9)
│   ├── workflows.py     4 higher-level workflow decompositions
│   ├── prompts.py       prompt construction
│   ├── fallbacks.py     repair / reject logic
│   ├── tracing.py       structured event emission
│   ├── nat_tools.py     NAT Function wrappers
│   ├── nat_llm.py       NIM config
│   ├── atif_adapter.py  ATIF trajectory conversion
│   └── skills/          directory-backed skill packages
│       ├── api.py           list_skills, search_skills, get_skill, run_skill_command
│       └── <skill-name>/    SKILL.md + optional sidecars
│
├── envs/            # Explicit task environment
│   ├── state.py         LateOrderEnvState, Subgoal enum
│   ├── transitions.py   state machine transitions
│   ├── rewards.py       dense, sequence-aware reward signals
│   ├── validators.py    dependency checking
│   ├── late_order_env.py  LateOrderRecoveryEnv
│   └── nemo_gym_adapter.py  nemo-gym-compatible export
│
├── rollouts/        # Canonical traces and serialization
│   ├── trace_types.py     Episode, Event, EventType (source of truth)
│   ├── episode_runner.py  run + enrich episodes with env rewards
│   ├── serializers.py     Episode <-> JSONL
│   ├── export_adapters.py Episode -> training trajectory + ATIF
│   └── scripted_traces.py pre-built episodes for workshop demos
│
├── training/        # openpipe-art-facing training semantics
│   ├── curriculum.py          4-stage training progression
│   ├── reward_views.py        stage-aware reward shaping
│   ├── datasets.py            training record construction
│   ├── openpipe_art_adapter.py  openpipe-art record building
│   └── experiments.py         experiment config
│
├── eval/            # Offline evaluation and reporting
│   ├── metrics.py       7 evaluators, TrajectoryEvaluation
│   └── reports.py       display and report helpers
│
├── scenario_data.py   # Synthetic in-memory data tables
├── main.py            # CLI entrypoint
└── <legacy shims>     # Backward-compat re-exports (see migration notes)
```

### Ownership boundaries

| Package | Owns | Does NOT own |
|---|---|---|
| `runtime/` | Tool definitions, schemas, prompts, fallbacks, tracing, agent loop, skill discovery | Rollout orchestration, reward computation, training datasets |
| `envs/` | Task state, transitions, rewards, validators | Runtime behavior, training semantics |
| `rollouts/` | Trace types, episode running, serialization, export adapters | Tool schemas, reward formulas, dataset views |
| `training/` | Curriculum, reward views, datasets, openpipe-art adapters | Runtime interfaces, rollout orchestration |
| `eval/` | Offline metrics, reports | Reward definitions (consumed from `envs/`) |

## Active Environment Stack

- `nvidia-nat` for NeMo Agent Toolkit runtime orchestration
- `nemo-gym` for environment-oriented rollouts and reward inspection
- `openpipe-art` for training-oriented exports and post-training alignment

## Execution Flow

The notebook keeps the workflow explicit and machine-checkable:

1. Diagnose order risk.
2. Assess primary fulfillment from the source DC.
3. Evaluate alternate recovery paths.
4. Score options and synthesize a recommendation.

The typical successful path uses `5-10` tool calls across a compact deterministic tool library.

## Quickstart

```bash
conda env create -f environment.yaml
conda activate by-workshop
python -m src.main --check-imports
jupyter lab
```

Then open `notebooks/late_order_recovery_workshop.ipynb`.

### Verify NVIDIA libraries

```bash
nat --version
python - <<'PY'
from importlib.metadata import version
import nemo_gym

print("nemo_gym:", nemo_gym.__name__)
print("openpipe-art version:", version("openpipe-art"))
PY
```

## Local Model

The live agent loop expects a locally deployed OpenAI-compatible chat endpoint:

- endpoint: `http://0.0.0.0:8000/v1/chat/completions`
- model: `nvidia/nemotron-3-nano`

See `documents/llm-access.md` for the smoke test request and cache mapping.

## Minimal Python Example

Run the live agent loop outside the notebook:

```python
from src.runtime.agent import run_agent, print_trace_summary
from src.eval.metrics import evaluate_trajectory
from src.eval.reports import print_evaluation

trace = run_agent("SO-10482", verbose=False)
print_trace_summary(trace)

evaluation = evaluate_trajectory(trace)
print_evaluation(evaluation)
```

Or via the CLI:

```bash
python -m src.main --episode SO-10482
python -m src.main --rollout SO-10482
python -m src.main --rollout SO-10482 --save-jsonl artifacts/episode.jsonl
```

## Outputs

When you run the later notebook sections, the repo can produce:

- worked successful and repair trajectories
- seven-dimension trajectory evaluations
- canonical episodes and trajectory artifacts intended for `openpipe-art`-oriented consumption
- rollout artifacts intended for the repo's own canonical trace collection layer

The notebook writes generated training artifacts under `artifacts/` when those export cells are executed.

## Module Migration Reference

The following modules were split during refactoring. The original top-level files remain as backward-compat shims that re-export from their canonical homes.

| Original module | Canonical location(s) |
|---|---|
| `src/tools.py` | `src/runtime/tools.py` |
| `src/skills.py` | `src/runtime/workflows.py` + `src/runtime/skills/` |
| `src/schema.py` | `src/runtime/schemas.py` + `src/envs/validators.py` |
| `src/agent_loop.py` | `src/runtime/agent.py` + `src/runtime/prompts.py` + `src/runtime/tracing.py` |
| `src/fallbacks.py` | `src/runtime/fallbacks.py` |
| `src/evaluation.py` | `src/eval/metrics.py` + `src/eval/reports.py` |
| `src/training_export.py` | `src/rollouts/export_adapters.py` + `src/training/reward_views.py` + `src/training/datasets.py` |

The notebook imports from the canonical modules directly. The shims exist only for external consumers that may still reference the old paths.

## Migration Status

All migration phases are complete.

- **Phase 1**: Canonical contracts — trace types, event vocabulary, package skeleton
- **Phase 2**: NAT-friendly runtime — tools, schemas, prompts, fallbacks, agent loop, skills
- **Phase 3**: Explicit environment — state machine, transitions, dense rewards
- **Phase 4**: Rollout layer — episode runner, serializers, export adapters
- **Phase 5**: Training semantics — curriculum, reward views, datasets, openpipe-art adapter
- **Phase 6**: Removed `src/systems/` and scale-out config sketches
- **Phase 7**: Rebuilt offline evaluation on canonical Episode traces
- **Phase 8**: Demoted notebook to library consumer, finished public surfaces

## Historical Context

Earlier planning docs may reference trainer-facing, rollout-shaping, or scale-out systems framing. Those terms describe the historical evolution of the design:

- **Trainer-facing** referred to early training-export assumptions before `openpipe-art` became the primary training path
- **Rollout-shaping** referred to rollout infrastructure concepts before the repo adopted its own canonical trace types
- **Scale-out systems** referred to multi-node deployment assumptions (8 H100 environment) before the scope was narrowed to local single-episode demonstration

These references are preserved where they explain design decisions but no active code path depends on them.

## Scope

This repository is a pedagogical artifact for a workshop, not a production system. It favors clarity, inspectability, deterministic behavior, and sequence correctness over realism or orchestration depth.
