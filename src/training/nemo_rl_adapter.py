"""
NeMo RL training adapter.

Converts enriched Episodes and TrainingRecords into ``DatumSpec`` dicts
compatible with NeMo RL's GRPO training pipeline. This is the canonical
bridge between repo-owned episode traces and NeMo RL's trainer inputs.

Owns:
    - Episode/TrainingRecord -> DatumSpec conversion
    - Batch -> grouped DatumSpec list assembly
    - Stage-aware reward shaping applied to DatumSpec extra_env_info
    - SFT-specific message formatting for supervised fine-tuning stages
    - OpenAI-style tool definition export for DatumSpec message_log

Does NOT own:
    - Episode types or enrichment (see rollouts/)
    - Reward computation from environment transitions (see envs.rewards)
    - Reward component weighting (see training.reward_views)
    - Curriculum stage definitions (see training.curriculum)
    - Dataset filtering (see training.datasets)
    - NeMo RL training loops or distributed execution (NeMo RL owns those)
"""
from __future__ import annotations

import json
from typing import Any

from src.rollouts.trace_types import (
    ASYNC_META_GEN_WEIGHT_VERSION,
    ASYNC_META_TRAIN_WEIGHT_VERSION,
    ASYNC_META_TRAJECTORY_AGE_MS,
    ASYNC_META_REPLAY_STATUS,
    Episode,
    EventType,
    ToolCallPayload,
    ToolResultPayload,
    TerminalOutcomePayload,
    ValidationErrorPayload,
    RepairAttemptPayload,
    RejectPayload,
)
from src.shared.tool_schemas import build_openai_tool_definitions, build_default_system_prompt
from src.training.curriculum import StageConfig, TrainingStage
from src.training.reward_views import (
    EpisodeRewardView,
    build_episode_reward_view,
)
from src.training.datasets import TrainingRecord


# ---------------------------------------------------------------------------
# Task success helper
# ---------------------------------------------------------------------------

def _compute_task_success(episode: Episode) -> int:
    """Compute task success using the canonical predicate from envs.rewards."""
    from src.envs.rewards import task_success_facts
    final_answer = None
    if episode.terminal and hasattr(episode.terminal, 'final_answer'):
        final_answer = episode.terminal.final_answer
    success_info = task_success_facts(
        final_answer=final_answer,
        order_id=episode.task_id,
        is_complete=episode.is_complete,
    )
    return 1 if success_info["success"] else 0


# ---------------------------------------------------------------------------
# Episode -> DatumSpec conversion
# ---------------------------------------------------------------------------

def episode_to_datum_spec(
    episode: Episode,
    reward: float | None = None,
    idx: int = 0,
) -> dict[str, Any]:
    """Convert a canonical Episode to a NeMo RL DatumSpec dict.

    Builds the OpenAI-style message_log from the episode's events,
    preserving tool calls and tool results in the format that NeMo RL
    expects for training.

    Args:
        episode: Canonical Episode from the rollout layer.
        reward: Override reward. If None, uses episode.metrics.total_reward.
        idx: Index for the DatumSpec (used by NeMo RL for batching).

    Returns:
        A dict conforming to nemo_rl.data.interfaces.DatumSpec.
    """
    messages: list[dict[str, Any]] = []

    messages.append({"role": "system", "content": _extract_system_content(episode)})
    messages.append({"role": "user", "content": episode.task_prompt})

    tool_call_counter = 0
    pending_tool_calls: list[dict[str, Any]] = []
    pending_call_ids: list[str] = []

    for event in episode.events:
        if event.event_type == EventType.TOOL_CALL:
            payload = event.payload
            if isinstance(payload, ToolCallPayload):
                tool_call_counter += 1
                call_id = f"call_{tool_call_counter}"
                pending_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": payload.tool_name,
                        "arguments": json.dumps(payload.arguments),
                    },
                })
                pending_call_ids.append(call_id)

        elif event.event_type == EventType.TOOL_RESULT:
            payload = event.payload
            if isinstance(payload, ToolResultPayload) and pending_tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": list(pending_tool_calls),
                })
                for cid in pending_call_ids:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": cid,
                        "content": json.dumps(payload.result),
                    })
                pending_tool_calls = []
                pending_call_ids = []

        elif event.event_type == EventType.TOOL_VALIDATION_ERROR:
            payload = event.payload
            if isinstance(payload, ValidationErrorPayload):
                messages.append({
                    "role": "assistant",
                    "content": payload.raw_model_output or "(malformed output)",
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"error_{event.step_index}",
                    "content": json.dumps({
                        "error": payload.error_type,
                        "message": payload.message,
                    }),
                })

        elif event.event_type == EventType.TOOL_REPAIR_ATTEMPT:
            payload = event.payload
            if isinstance(payload, RepairAttemptPayload):
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"repair_{event.step_index}",
                    "content": json.dumps({
                        "repair_succeeded": payload.succeeded,
                        "repairs_applied": payload.repairs_applied,
                    }),
                })

        elif event.event_type == EventType.TOOL_REJECT:
            payload = event.payload
            if isinstance(payload, RejectPayload):
                messages.append({
                    "role": "assistant",
                    "content": payload.raw_model_output or "(rejected output)",
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"reject_{event.step_index}",
                    "content": json.dumps({
                        "rejected": True,
                        "reason": payload.reason,
                        "repairs_attempted": payload.repairs_attempted,
                    }),
                })

        elif event.event_type == EventType.TERMINAL_OUTCOME:
            payload = event.payload
            if isinstance(payload, TerminalOutcomePayload):
                if payload.final_answer is not None:
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps(payload.final_answer),
                    })

    effective_reward = reward if reward is not None else episode.metrics.total_reward

    event_rewards = [
        e.reward for e in episode.events
        if e.reward is not None
    ]

    extra_env_info: dict[str, Any] = {
        "reward": effective_reward,
        "episode_id": episode.episode_id,
        "task_id": episode.task_id,
        "model_id": episode.model_id,
        "per_step_rewards": event_rewards,
        "metrics": {
            "valid_tool_calls": episode.metrics.valid_tool_calls,
            "invalid_tool_calls": episode.metrics.invalid_tool_calls,
            "repair_attempts": episode.metrics.repair_attempts,
            "rejects": episode.metrics.rejects,
            "episode_length": episode.metrics.total_steps,
            "task_success": _compute_task_success(episode),
        },
        "tools": _build_tool_definitions(),
        "parallel_tool_calls": False,
    }

    # Async GRPO metadata (RL_ARCHITECTURE.md § Async Metadata Contract).
    # Propagate from episode.metadata if present; default to synchronous
    # placeholders so the contract is always visible in serialized output.
    extra_env_info[ASYNC_META_GEN_WEIGHT_VERSION] = episode.metadata.get(
        ASYNC_META_GEN_WEIGHT_VERSION, 0,
    )
    extra_env_info[ASYNC_META_TRAIN_WEIGHT_VERSION] = episode.metadata.get(
        ASYNC_META_TRAIN_WEIGHT_VERSION, 0,
    )
    extra_env_info[ASYNC_META_TRAJECTORY_AGE_MS] = episode.metadata.get(
        ASYNC_META_TRAJECTORY_AGE_MS, 0,
    )
    extra_env_info[ASYNC_META_REPLAY_STATUS] = episode.metadata.get(
        ASYNC_META_REPLAY_STATUS, "accepted",
    )

    return {
        "message_log": messages,
        "length": len(messages),
        "extra_env_info": extra_env_info,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": "late_order_recovery",
    }


def _extract_system_content(episode: Episode) -> str:
    """Extract the system prompt from episode events, or return a default.

    Uses the repo-owned default prompt from shared.tool_schemas rather than
    importing the runtime prompt builder, keeping the training layer
    independent of NAT runtime machinery.
    """
    return build_default_system_prompt()


def _build_tool_definitions() -> list[dict[str, Any]]:
    """Build OpenAI-style tool definitions for DatumSpec."""
    return build_openai_tool_definitions()


# ---------------------------------------------------------------------------
# Async GRPO freshness helper
# ---------------------------------------------------------------------------

def is_fresh(episode: Episode, max_age_ms: int = 0) -> bool:
    """Check whether an episode's trajectory is fresh enough for training.

    Returns True for synchronous runs (where trajectory_age_ms is 0 or
    absent). When async GRPO is enabled, enforces a staleness limit:
    trajectories older than ``max_age_ms`` are considered stale and
    should be filtered or down-weighted before training.

    Args:
        episode: Canonical Episode with async metadata in episode.metadata.
        max_age_ms: Maximum allowed age in milliseconds. 0 means no limit
                    (always fresh), which is the correct default for
                    synchronous collection.

    Returns:
        True if the episode is fresh enough for training.
    """
    if max_age_ms <= 0:
        return True
    age = episode.metadata.get(ASYNC_META_TRAJECTORY_AGE_MS, 0)
    return age <= max_age_ms


# ---------------------------------------------------------------------------
# TrainingRecord -> shaped DatumSpec
# ---------------------------------------------------------------------------

def training_record_to_datum_spec(
    record: TrainingRecord,
    stage_config: StageConfig,
    idx: int = 0,
) -> dict[str, Any]:
    """Convert a TrainingRecord into a DatumSpec with stage-shaped reward.

    Applies stage-specific reward shaping from the reward view, then builds
    a DatumSpec with the shaped reward assigned.

    Args:
        record: TrainingRecord from the dataset layer.
        stage_config: Curriculum stage config for reward shaping.
        idx: Index for the DatumSpec.

    Returns:
        A DatumSpec dict with stage-shaped reward.
    """
    reward_view = build_episode_reward_view(record.reward_summary, stage_config)
    shaped_reward = reward_view.combined_reward

    datum = episode_to_datum_spec(
        episode=record.episode,
        reward=shaped_reward,
        idx=idx,
    )
    datum["extra_env_info"]["stage"] = stage_config.stage.value
    datum["extra_env_info"]["stage_description"] = stage_config.description

    return datum


# ---------------------------------------------------------------------------
# Batch -> grouped DatumSpec list
# ---------------------------------------------------------------------------

def training_batch_to_datum_specs(
    records: list[TrainingRecord],
    stage_config: StageConfig,
) -> list[dict[str, Any]]:
    """Convert a batch of TrainingRecords into a list of DatumSpec dicts.

    Each record is converted to a DatumSpec with stage-shaped rewards.
    The list can be used for NeMo RL dataset construction.

    Args:
        records: List of TrainingRecords from a stage dataset.
        stage_config: Curriculum stage config for reward shaping.

    Returns:
        A list of DatumSpec dicts ready for NeMo RL.
    """
    return [
        training_record_to_datum_spec(record, stage_config, idx=i)
        for i, record in enumerate(records)
    ]


def enriched_episodes_to_datum_specs(
    episodes: list[Episode],
    rewards: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Convert a list of Episodes directly to DatumSpec dicts.

    Bypasses the TrainingRecord/stage layer for simple batch conversion.

    Args:
        episodes: List of canonical Episodes.
        rewards: Optional per-episode reward overrides.

    Returns:
        A list of DatumSpec dicts.
    """
    specs = []
    for i, episode in enumerate(episodes):
        reward = rewards[i] if rewards is not None else None
        specs.append(episode_to_datum_spec(episode, reward=reward, idx=i))
    return specs


# ---------------------------------------------------------------------------
# GRPO-ready grouped DatumSpec export
# ---------------------------------------------------------------------------

def build_grpo_datum_group(
    records: list[TrainingRecord],
    stage_config: StageConfig,
    group_size: int = 4,
) -> list[dict[str, Any]]:
    """Build a GRPO-ready group of DatumSpecs with group-relative advantages.

    Groups records for the same task_id, computes each record's
    stage-shaped reward, and annotates every DatumSpec with group-relative
    advantage (reward - group_mean) so that NeMo RL can use GRPO
    without recomputing baselines.

    Args:
        records: TrainingRecords (may span multiple task_ids).
        stage_config: Curriculum stage config for reward shaping.
        group_size: Expected group size (for metadata; grouping uses
                    all available records per task).

    Returns:
        A list of DatumSpec dicts with per-datum group_advantage in
        extra_env_info and per-step rewards.
    """
    from collections import defaultdict

    by_task: dict[str, list[TrainingRecord]] = defaultdict(list)
    for record in records:
        by_task[record.episode.task_id].append(record)

    datum_specs: list[dict[str, Any]] = []
    global_idx = 0

    for task_id, task_records in by_task.items():
        group_rewards: list[float] = []
        group_data: list[tuple[TrainingRecord, float, list[float]]] = []

        for record in task_records:
            reward_view = build_episode_reward_view(
                record.reward_summary, stage_config,
            )
            shaped_reward = reward_view.combined_reward
            per_step = get_per_step_rewards(reward_view)
            group_rewards.append(shaped_reward)
            group_data.append((record, shaped_reward, per_step))

        group_mean = sum(group_rewards) / len(group_rewards) if group_rewards else 0.0
        group_var = (
            sum((r - group_mean) ** 2 for r in group_rewards) / len(group_rewards)
            if group_rewards else 0.0
        )
        group_std = max(group_var ** 0.5, 1e-8)

        for record, shaped_reward, per_step in group_data:
            z_score = (shaped_reward - group_mean) / group_std
            advantage = round(max(-5.0, min(5.0, z_score)), 4)

            datum = episode_to_datum_spec(
                episode=record.episode,
                reward=shaped_reward,
                idx=global_idx,
            )
            datum["extra_env_info"]["stage"] = stage_config.stage.value
            datum["extra_env_info"]["group_task_id"] = task_id
            datum["extra_env_info"]["group_size"] = len(task_records)
            datum["extra_env_info"]["group_mean_reward"] = round(group_mean, 4)
            datum["extra_env_info"]["group_advantage"] = advantage
            datum["extra_env_info"]["per_step_rewards"] = per_step

            datum_specs.append(datum)
            global_idx += 1

    return datum_specs


def get_per_step_rewards(view: EpisodeRewardView) -> list[float]:
    """Extract per-step shaped rewards from an EpisodeRewardView.

    Re-exported here for convenience; canonical definition is in
    training.reward_views.
    """
    from src.training.reward_views import get_per_step_rewards as _get
    return _get(view)


# ---------------------------------------------------------------------------
# Group metadata helpers
# ---------------------------------------------------------------------------

def get_group_metadata(datum_specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract group-level metadata from a list of DatumSpec dicts."""
    if not datum_specs:
        return {}
    first_info = datum_specs[0].get("extra_env_info", {})
    return {
        "stage": first_info.get("stage", ""),
        "method": "grpo",
        "group_size": first_info.get("group_size", len(datum_specs)),
    }


def get_group_metrics(datum_specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute group-level metrics from a list of DatumSpec dicts."""
    if not datum_specs:
        return {}
    rewards = [
        d.get("extra_env_info", {}).get("reward", 0.0)
        for d in datum_specs
    ]
    task_ids = {
        d.get("extra_env_info", {}).get("group_task_id", d.get("task_name", ""))
        for d in datum_specs
    }
    return {
        "num_datum_specs": len(datum_specs),
        "num_tasks": len(task_ids),
        "avg_reward": (
            sum(rewards) / len(rewards) if rewards else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# SFT-specific formatting
# ---------------------------------------------------------------------------

def build_sft_datum_spec(record: TrainingRecord, idx: int = 0) -> dict[str, Any]:
    """Build a DatumSpec formatted for supervised fine-tuning.

    SFT datum specs use the same message format but reward is not
    used for loss computation. The datum captures the ideal conversation
    for cross-entropy training.

    Args:
        record: A TrainingRecord from the SFT stage dataset.
        idx: Index for the DatumSpec.

    Returns:
        A DatumSpec dict with SFT-appropriate messages.
    """
    datum = episode_to_datum_spec(record.episode, reward=1.0, idx=idx)
    datum["extra_env_info"]["training_mode"] = "sft"
    return datum


def build_sft_datum_group(records: list[TrainingRecord]) -> list[dict[str, Any]]:
    """Build a list of DatumSpec dicts for SFT training.

    Args:
        records: TrainingRecords from the SFT stage.

    Returns:
        A list of DatumSpec dicts with SFT-appropriate messages.
    """
    return [build_sft_datum_spec(r, idx=i) for i, r in enumerate(records)]


# ---------------------------------------------------------------------------
# JSONL serialization (for offline handoff to NeMo RL training pipelines)
# ---------------------------------------------------------------------------

def save_datum_specs_jsonl(
    datum_specs: list[dict[str, Any]],
    path: str,
) -> None:
    """Write DatumSpec dicts to a JSONL file.

    Each line is the JSON serialization of one DatumSpec.
    """
    with open(path, "w") as f:
        for d in datum_specs:
            f.write(json.dumps(d) + "\n")


def save_datum_group_jsonl(
    datum_specs: list[dict[str, Any]],
    path: str,
) -> None:
    """Write a group of DatumSpec dicts to a JSONL file.

    Writes the full group as a single JSON line (group metadata + all
    datum specs), suitable for NeMo RL's batch ingestion.
    """
    with open(path, "w") as f:
        metadata = get_group_metadata(datum_specs)
        metrics = get_group_metrics(datum_specs)
        payload = {
            "datum_specs": datum_specs,
            "metadata": metadata,
            "metrics": metrics,
        }
        f.write(json.dumps(payload) + "\n")
