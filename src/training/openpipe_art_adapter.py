"""
openpipe-art training adapter.

Converts enriched Episodes and training records into real ``art.Trajectory``
and ``art.TrajectoryGroup`` objects for openpipe-art consumption. This is the
canonical bridge between repo-owned episode traces and openpipe-art's training
infrastructure.

Owns:
    - Episode/TrainingRecord -> art.Trajectory conversion
    - Batch -> art.TrajectoryGroup assembly
    - Stage-aware reward shaping applied to art.Trajectory.reward
    - SFT-specific message formatting for supervised fine-tuning stages
    - OpenAI-style tool definition export for art.Trajectory.tools

Does NOT own:
    - Episode types or enrichment (see rollouts/)
    - Reward computation from environment transitions (see envs.rewards)
    - Reward component weighting (see training.reward_views)
    - Curriculum stage definitions (see training.curriculum)
    - Dataset filtering (see training.datasets)
    - art training loops or backends (openpipe-art owns those)
"""
from __future__ import annotations

import json
from typing import Any

import art

from src.rollouts.trace_types import (
    Episode,
    EventType,
    ToolCallPayload,
    ToolResultPayload,
    TerminalOutcomePayload,
    ValidationErrorPayload,
    RepairAttemptPayload,
    RejectPayload,
)
from src.runtime.nat_tools import build_openai_tool_definitions
from src.training.curriculum import StageConfig, TrainingStage
from src.training.reward_views import (
    EpisodeRewardView,
    build_episode_reward_view,
)
from src.training.datasets import TrainingRecord


# ---------------------------------------------------------------------------
# Episode -> art.Trajectory conversion
# ---------------------------------------------------------------------------

def episode_to_art_trajectory(
    episode: Episode,
    reward: float | None = None,
) -> art.Trajectory:
    """Convert a canonical Episode to an art.Trajectory.

    Builds the OpenAI-style messages_and_choices list from the episode's
    events, preserving tool calls and tool results in the format that
    art.Trajectory expects for training.

    Args:
        episode: Canonical Episode from the rollout layer.
        reward: Override reward. If None, uses episode.metrics.total_reward.

    Returns:
        An art.Trajectory with messages, tools, reward, and metadata.
    """
    messages: list[dict[str, Any]] = []
    tools = _build_art_tools()

    # System prompt from the first event or a default
    messages.append({"role": "system", "content": _extract_system_content(episode)})
    messages.append({"role": "user", "content": episode.task_prompt})

    # Walk events and build the conversation
    pending_tool_calls: list[dict[str, Any]] = []
    tool_call_counter = 0

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

        elif event.event_type == EventType.TOOL_RESULT:
            payload = event.payload
            if isinstance(payload, ToolResultPayload) and pending_tool_calls:
                # Emit the assistant message with tool_calls
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": list(pending_tool_calls),
                })
                # Emit tool result messages for each pending call
                for tc in pending_tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(payload.result),
                    })
                pending_tool_calls = []

        elif event.event_type == EventType.TOOL_VALIDATION_ERROR:
            # Emit validation errors as system feedback so the learner
            # sees the full canonical parse -> validate -> error path.
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
            # Emit repair attempts so the learner can observe the
            # fallback repair chain rather than a sanitized transcript.
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
            # Emit rejects as explicit system feedback.
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

    return art.Trajectory(
        messages_and_choices=messages,
        tools=tools,
        reward=effective_reward,
        metrics={
            "valid_tool_calls": episode.metrics.valid_tool_calls,
            "invalid_tool_calls": episode.metrics.invalid_tool_calls,
            "repair_attempts": episode.metrics.repair_attempts,
            "rejects": episode.metrics.rejects,
            "episode_length": episode.metrics.total_steps,
            "task_success": 1 if episode.is_complete else 0,
        },
        metadata={
            "task_id": episode.task_id,
            "model_id": episode.model_id,
        },
    )


def _extract_system_content(episode: Episode) -> str:
    """Extract the system prompt from episode events, or return a default."""
    for event in episode.events:
        if event.event_type == EventType.USER_TASK:
            # The system prompt is typically set before the user task;
            # for now use a sensible default since the runtime builds it.
            break
    from src.runtime.prompts import build_system_prompt
    return build_system_prompt()


def _build_art_tools() -> list[dict[str, Any]]:
    """Build OpenAI-style tool definitions for art.Trajectory.tools."""
    return build_openai_tool_definitions()


# ---------------------------------------------------------------------------
# TrainingRecord -> shaped art.Trajectory
# ---------------------------------------------------------------------------

def training_record_to_art_trajectory(
    record: TrainingRecord,
    stage_config: StageConfig,
) -> art.Trajectory:
    """Convert a TrainingRecord into an art.Trajectory with stage-shaped reward.

    Applies stage-specific reward shaping from the reward view, then builds
    an art.Trajectory with the shaped reward assigned.

    Args:
        record: TrainingRecord from the dataset layer.
        stage_config: Curriculum stage config for reward shaping.

    Returns:
        An art.Trajectory with stage-shaped reward.
    """
    reward_view = build_episode_reward_view(record.reward_summary, stage_config)
    shaped_reward = reward_view.combined_reward

    trajectory = episode_to_art_trajectory(
        episode=record.episode,
        reward=shaped_reward,
    )
    trajectory.metadata["stage"] = stage_config.stage.value
    trajectory.metadata["stage_description"] = stage_config.description

    return trajectory


# ---------------------------------------------------------------------------
# Batch -> art.TrajectoryGroup
# ---------------------------------------------------------------------------

def training_batch_to_art_group(
    records: list[TrainingRecord],
    stage_config: StageConfig,
) -> art.TrajectoryGroup:
    """Convert a batch of TrainingRecords into an art.TrajectoryGroup.

    Each record is converted to an art.Trajectory with stage-shaped
    rewards. The group can be passed directly to art's training backends.

    Args:
        records: List of TrainingRecords from a stage dataset.
        stage_config: Curriculum stage config for reward shaping.

    Returns:
        An art.TrajectoryGroup ready for training.
    """
    trajectories = [
        training_record_to_art_trajectory(record, stage_config)
        for record in records
    ]
    return art.TrajectoryGroup(
        trajectories=trajectories,
        metadata={"stage": stage_config.stage.value},
        metrics={
            "num_trajectories": len(trajectories),
            "avg_reward": (
                sum(t.reward for t in trajectories) / len(trajectories)
                if trajectories else 0.0
            ),
        },
    )


def enriched_episodes_to_art_group(
    episodes: list[Episode],
    rewards: list[float] | None = None,
) -> art.TrajectoryGroup:
    """Convert a list of Episodes directly to an art.TrajectoryGroup.

    Bypasses the TrainingRecord/stage layer for simple batch conversion.

    Args:
        episodes: List of canonical Episodes.
        rewards: Optional per-episode reward overrides.

    Returns:
        An art.TrajectoryGroup.
    """
    trajectories = []
    for i, episode in enumerate(episodes):
        reward = rewards[i] if rewards is not None else None
        trajectories.append(episode_to_art_trajectory(episode, reward=reward))
    return art.TrajectoryGroup(trajectories=trajectories)


# ---------------------------------------------------------------------------
# SFT-specific formatting
# ---------------------------------------------------------------------------

def build_sft_art_trajectory(record: TrainingRecord) -> art.Trajectory:
    """Build an art.Trajectory formatted for supervised fine-tuning.

    SFT trajectories use the same message format but reward is not
    used for loss computation. The trajectory captures the ideal
    conversation for cross-entropy training.

    Args:
        record: A TrainingRecord from the SFT stage dataset.

    Returns:
        An art.Trajectory with SFT-appropriate messages.
    """
    trajectory = episode_to_art_trajectory(record.episode, reward=1.0)
    trajectory.metadata["training_mode"] = "sft"
    return trajectory


def build_sft_art_group(records: list[TrainingRecord]) -> art.TrajectoryGroup:
    """Build an art.TrajectoryGroup for SFT training.

    Args:
        records: TrainingRecords from the SFT stage.

    Returns:
        An art.TrajectoryGroup with SFT trajectories.
    """
    trajectories = [build_sft_art_trajectory(r) for r in records]
    return art.TrajectoryGroup(
        trajectories=trajectories,
        metadata={"training_mode": "sft"},
    )


# ---------------------------------------------------------------------------
# JSONL serialization (for offline handoff to art training pipelines)
# ---------------------------------------------------------------------------

def save_art_trajectories_jsonl(
    trajectories: list[art.Trajectory],
    path: str,
) -> None:
    """Write art.Trajectory objects to a JSONL file.

    Each line is the Pydantic model_dump_json() of one trajectory.
    """
    with open(path, "w") as f:
        for t in trajectories:
            f.write(t.model_dump_json() + "\n")


def save_art_group_jsonl(
    group: art.TrajectoryGroup,
    path: str,
) -> None:
    """Write an art.TrajectoryGroup to a JSONL file.

    Writes the full group as a single JSON line (group metadata + all
    trajectories), suitable for art's batch ingestion.
    """
    with open(path, "w") as f:
        f.write(group.model_dump_json() + "\n")
