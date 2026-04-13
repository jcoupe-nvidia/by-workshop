# by-workshop

MVP workshop repo for agentic supply-chain workflows on the NVIDIA stack.

This repository is a teaching example, not a production system. It is designed to show a single rich, inspectable example of multi-step agent execution, deterministic tool use, fallback handling, and sequence-sensitive evaluation.

## What This Repo Demonstrates

- Multi-step agent execution against a local OpenAI-compatible model endpoint
- Explicit higher-level skills composed from deterministic business tools
- Nemotron-style structured tool calling with validation, repair, and reject paths
- Machine-checkable dependency ordering where sequence correctness matters
- Repo-owned canonical traces, rewards, and offline evaluation
- NVIDIA integrations through `NeMo Agent Toolkit (NAT)`, `NeMo Gym`, and `NeMo RL`

## Core Scenario

The workshop centers on late order recovery for customer order `SO-10482` for `1,200` units.

The target task is:

> Determine whether the order can still be fulfilled on time. If not, recommend the best mitigation action.

The agent investigates options such as:

- fulfill from the original source DC
- transfer from an alternate DC
- expedite from a supplier
- partially fulfill
- recommend a substitute SKU when available

A typical successful run should require `5-10` tool calls and use `2-4` explicit higher-level skills.

## Architecture

The notebook is a consumer and teaching surface. The source of truth lives in repo code and documents.

| Layer | Primary responsibility | NVIDIA mapping |
| --- | --- | --- |
| shared contracts | Repo-owned task semantics, deterministic tools, canonical trace/event schemas, sequence rules, and offline evaluation logic | repo-owned |
| `runtime/` | Interactive agent runtime, structured tool calls, prompts, fallbacks, skill discovery, and single-episode execution | `NeMo Agent Toolkit (NAT)` |
| `envs/` | Task truth, state transitions, tool preconditions, terminal conditions, and reward-relevant facts | `NeMo Gym` |
| `rollouts/` | Canonical episode capture, serialization, explicit retries/rejects/repairs, and adapters for collection/export | `NeMo Gym` adapters over repo-owned traces |
| `training/` | Trainer-facing datasets, reward views, curriculum staging, and training handoff artifacts | `NeMo RL` |
| `eval/` | Offline measurement of skill selection, tool quality, sequence correctness, success, recovery, and efficiency | repo-owned |

Keep these boundaries intact:

- `runtime/` decides how the agent acts.
- `envs/` decides what those actions mean.
- `rollouts/` records what happened.
- `training/` turns trajectories into learning signals.
- `eval/` measures quality and regressions.

## NVIDIA Stack Versions

The workshop docs and examples are aligned to the versions observed in the reference `nemo-rl` environment:

- `nvidia-nat==1.6.0`
- `nemo-gym==0.2.0`
- `nemo-rl==0.5.0rc0`

See `documents/NVIDIA_SOFTWARE_MAPPING.md` for the versioned stack reference and layer mapping.

## Repository Layout

```text
src/
  runtime/
    agent.py
    fallbacks.py
    nat_llm.py
    nat_tools.py
    prompts.py
    schemas.py
    tools.py
    tracing.py
    workflows.py
    skills/
      api.py
      diagnose-order-risk/SKILL.md
      assess-primary-fulfillment/SKILL.md
      evaluate-alternate-recovery-paths/SKILL.md
      synthesize-recommendation/SKILL.md
  envs/
    late_order_env.py
    nemo_gym_adapter.py
    rewards.py
    state.py
    transitions.py
    validators.py
  rollouts/
    episode_runner.py
    export_adapters.py
    nemo_gym_rollouts.py
    scripted_traces.py
    serializers.py
    trace_types.py
  training/
    curriculum.py
    datasets.py
    experiments.py
    nemo_rl_adapter.py
    reward_views.py
  eval/
    metrics.py
    reports.py
  main.py
  scenario_data.py
notebooks/
  late_order_recovery_workshop.ipynb
documents/
  RL_ARCHITECTURE.md
  NVIDIA_SOFTWARE_MAPPING.md
  llm-access.md
tests/
```

## Skills And Tools

The runtime exposes four directory-backed skills:

- `diagnose_order_risk(order_id)`
- `assess_primary_fulfillment(order_id)`
- `evaluate_alternate_recovery_paths(order_id)`
- `synthesize_recommendation(order_id)`

The deterministic business tool library currently contains nine tools:

- `get_order(order_id)`
- `get_shipment_status(order_id)`
- `get_inventory(sku, dc_id)`
- `find_alternate_inventory(sku, region)`
- `get_transfer_eta(from_dc, to_dc, sku, qty)`
- `get_supplier_expedite_options(sku, qty)`
- `get_fulfillment_capacity(dc_id, date)`
- `score_recovery_options(options, objective)`
- `recommend_action(context)`

Representative dependency ordering includes:

- order lookup before shipment analysis
- shipment analysis before mitigation
- source inventory check before alternate sourcing
- alternate inventory discovery before transfer ETA estimation
- candidate mitigation options before scoring and recommendation

## Structured Tool Calls

The canonical tool-call shape is Nemotron-style structured JSON:

```json
{
  "thought": "optional short reasoning summary or omitted entirely",
  "tool_call": {
    "name": "get_inventory",
    "arguments": {
      "sku": "SKU-100",
      "dc_id": "DC-WEST-01"
    }
  }
}
```

Malformed outputs are handled explicitly through repair or reject logic rather than being silently hidden.

## Prerequisites

This repo requires two containers running on a GPU machine: one serving the model via NIM, and one providing the NeMo RL development environment.

### 1. Start the NIM model server

On the remote machine, pull and run the Nemotron NIM container:

```bash
docker run -it --rm \
  --name=nemotron-3-nano-30b-a3b \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=16GB \
  -e NGC_API_KEY="$NGC_CLI_API_KEY" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e NIM_MODEL_PROFILE="vllm-bf16-tp1-pp1-6303b8767cbc4d12ace1473dfaee17f021c6fd7784f7da8f14a85f285ed546f6" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u "$(id -u)" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/nemotron-3-nano:1
```

### 2. Launch the NeMo RL development container

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  -v "$PWD":/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo-rl:v0.5.0
```

### 3. Attach VS Code or Cursor to the container

Open VS Code or Cursor and attach to the running NeMo RL container using the **Dev Containers** extension (Remote - Containers). Once attached, open the `/workspace` folder.

### 4. Copy the model cache into the development container

Once inside the container, get its hostname (which is also its container ID):

```bash
hostname
```

Then from the **host machine**, create the cache directory and copy the model weights:

```bash
docker exec <container-id> mkdir -p /opt/nim/.cache/
docker cp /home/nvidia/.cache/nim/nemotron-3-nano-30b-a3b/. <container-id>:/opt/nim/.cache/
```

Replace `<container-id>` with the value returned by `hostname` inside the container.

### 5. Activate the virtual environment and install dependencies

Inside the NeMo RL container:

```bash
source /opt/nemo_rl_venv/bin/activate
echo "$VIRTUAL_ENV"
python -V
which python
```

Install additional Python packages:

```bash
python -m pip install \
  "openai<3" \
  "jupyterlab<5" \
  "ipykernel<8" \
  "nvidia-nat" \
  "nemo-gym==0.2.0" \
  requests
```

### 6. Validate imports

```bash
python -m src.main --check-imports
```

## Quickstart

Start the notebook experience (inside the container with the venv activated):

```bash
jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

Then open `notebooks/late_order_recovery_workshop.ipynb`.

## Running From The CLI

Run a single direct episode:

```bash
python -m src.main --episode
```

Run the same episode through NAT-backed tool dispatch:

```bash
python -m src.main --episode --nat
```

Run an enriched rollout with environment rewards:

```bash
python -m src.main --rollout
```

Export rollout artifacts:

```bash
python -m src.main --rollout --save-jsonl artifacts/episode.jsonl
python -m src.main --rollout --nemo-gym-export artifacts/nemo_gym_rows.jsonl
python -m src.main --rollout --nemo-rl-export artifacts/nemo_rl_datums.jsonl
```

## Local Model

The live agent loop expects a local OpenAI-compatible endpoint:

- base URL: `http://0.0.0.0:8000/v1`
- chat completions URL: `http://0.0.0.0:8000/v1/chat/completions`
- model: `nvidia/nemotron-3-nano`

Use `documents/llm-access.md` for the smoke test request and cache mapping.

## Evaluation Focus

Offline evaluation is sequence-sensitive and centers on:

- skill selection quality
- tool validity
- tool accuracy
- sequence correctness
- task success
- recovery quality
- efficiency

The workshop intentionally treats sequence correctness as first-class: a plausible final recommendation reached through an invalid tool sequence should not be considered fully successful.

## Reference Docs

- `PLAN.md`: migration status and current repo outcome
- `CLAUDE.md`: repo purpose, scope, scenario, required stack, and notebook requirements
- `documents/RL_ARCHITECTURE.md`: layer boundaries and RL design rules
- `documents/NVIDIA_SOFTWARE_MAPPING.md`: mapping from repo layers to `NAT`, `NeMo Gym`, and `NeMo RL`
