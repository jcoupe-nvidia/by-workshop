"""
Training/export adapters: maps canonical Episodes to training-oriented and ATIF trajectories.

Owns:
    - Converting enriched Episodes into training trajectory records
    - Converting enriched Episodes into ATIF Trajectories (via atif_adapter)
    - Per-step observation/action/reward triple extraction from Episode events
    - Stable JSONL serialization of training trajectories

Does NOT own:
    - Episode type definitions (see rollouts.trace_types)
    - Reward computation (see envs.rewards)
    - Environment state or transitions (see envs/)
    - Training dataset views or curriculum staging (see training/)

Two output formats:
    1. TrainingTrajectory -- lightweight observation/action/reward triples for
       downstream training consumption
    2. ATIF Trajectory    -- full NAT trajectory format for finetuning infrastructure

Usage::

    from src.rollouts.export_adapters import (
        episode_to_training_trajectory,
        episode_to_atif_trajectory,
    )

    training_traj = episode_to_training_trajectory(enriched_result.episode)
    atif_traj = episode_to_atif_trajectory(enriched_result.episode)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from src.rollouts.trace_types import (
    Episode,
    EventType,
    ToolCallPayload,
    ToolResultPayload,
    TerminalOutcomePayload,
    ValidationErrorPayload,
)
from src.envs.rewards import EpisodeRewardSummary


@dataclass
class TrainingTrajectoryStep:
    """One step in a training-oriented trajectory record."""

    step_index: int
    observation: str
    action: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingTrajectory:
    """Full trajectory formatted for downstream training consumption."""

    task_id: str
    model_id: str
    steps: list[TrainingTrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    episode_length: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def episode_to_training_trajectory(
    episode: Episode,
    reward_summary: EpisodeRewardSummary | None = None,
) -> TrainingTrajectory:
    """Convert a canonical Episode into a training trajectory record."""

    steps: list[TrainingTrajectoryStep] = []
    step_idx = 0

    last_observation = episode.task_prompt
    pending_action: _PendingAction | None = None
    pending_errors: list[dict[str, Any]] = []

    for event in episode.events:
        if event.event_type == EventType.TOOL_CALL:
            payload = event.payload
            if isinstance(payload, ToolCallPayload):
                pending_action = _PendingAction(
                    tool_name=payload.tool_name,
                    arguments=payload.arguments,
                    thought=payload.thought,
                    reward=event.reward or 0.0,
                    errors_before=list(pending_errors),
                )
                pending_errors = []

        elif event.event_type == EventType.TOOL_RESULT:
            payload = event.payload
            if isinstance(payload, ToolResultPayload) and pending_action is not None:
                action_str = json.dumps({
                    "tool_call": {
                        "name": pending_action.tool_name,
                        "arguments": pending_action.arguments,
                    },
                    "thought": pending_action.thought,
                })

                info: dict[str, Any] = {
                    "tool_name": pending_action.tool_name,
                    "valid": True,
                }
                if pending_action.errors_before:
                    info["errors_before"] = pending_action.errors_before

                steps.append(TrainingTrajectoryStep(
                    step_index=step_idx,
                    observation=last_observation,
                    action=action_str,
                    reward=pending_action.reward,
                    done=False,
                    info=info,
                ))
                step_idx += 1
                last_observation = json.dumps(payload.result)
                pending_action = None

        elif event.event_type == EventType.TOOL_VALIDATION_ERROR:
            payload = event.payload
            if isinstance(payload, ValidationErrorPayload):
                pending_errors.append({
                    "error_type": payload.error_type,
                    "message": payload.message,
                    "reward": event.reward,
                })

        elif event.event_type == EventType.TERMINAL_OUTCOME:
            payload = event.payload
            if isinstance(payload, TerminalOutcomePayload):
                if payload.final_answer is not None:
                    action_str = json.dumps({"final_answer": payload.final_answer})
                    steps.append(TrainingTrajectoryStep(
                        step_index=step_idx,
                        observation=last_observation,
                        action=action_str,
                        reward=event.reward or 0.0,
                        done=True,
                        info={
                            "tool_name": "<final_answer>",
                            "valid": True,
                            "terminal_reason": payload.reason,
                        },
                    ))
                    step_idx += 1
                elif steps:
                    steps[-1].done = True
                    steps[-1].info["terminal_reason"] = payload.reason

    total_reward = sum(s.reward for s in steps)

    meta: dict[str, Any] = {
        "wall_time_seconds": episode.metrics.wall_time_seconds,
        "model_calls": episode.metrics.model_calls,
        "valid_tool_calls": episode.metrics.valid_tool_calls,
        "invalid_tool_calls": episode.metrics.invalid_tool_calls,
        "repair_attempts": episode.metrics.repair_attempts,
        "rejects": episode.metrics.rejects,
    }
    if reward_summary is not None:
        meta["avg_step_reward"] = reward_summary.avg_step_reward
        meta["penalty_counts"] = reward_summary.penalty_counts

    return TrainingTrajectory(
        task_id=episode.task_id,
        model_id=episode.model_id,
        steps=steps,
        total_reward=round(total_reward, 4),
        episode_length=len(steps),
        metadata=meta,
    )


def training_trajectory_to_jsonl(trajectory: TrainingTrajectory) -> str:
    """Serialize a training trajectory to a single JSONL line."""

    record = {
        "task_id": trajectory.task_id,
        "model_id": trajectory.model_id,
        "total_reward": trajectory.total_reward,
        "episode_length": trajectory.episode_length,
        "steps": [asdict(s) for s in trajectory.steps],
        "metadata": trajectory.metadata,
    }
    return json.dumps(record)


def save_training_trajectories_jsonl(
    trajectories: list[TrainingTrajectory],
    path: str,
) -> None:
    """Write training trajectories to a JSONL file."""

    with open(path, "w") as f:
        for traj in trajectories:
            f.write(training_trajectory_to_jsonl(traj) + "\n")


def episode_to_atif_trajectory(
    episode: Episode,
    agent_name: str = "by-workshop-agent",
    agent_version: str = "0.1.0",
) -> Any:
    """Convert an Episode to a NAT ATIF Trajectory."""

    from src.runtime.atif_adapter import episode_to_atif

    return episode_to_atif(
        episode,
        agent_name=agent_name,
        agent_version=agent_version,
    )


def save_atif_trajectories_jsonl(
    trajectories: list[Any],
    path: str,
) -> None:
    """Write ATIF trajectories to a JSONL file."""

    with open(path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_json_dict()) + "\n")


class _PendingAction:
    """Transient state for building training trajectory steps from events."""

    __slots__ = ("tool_name", "arguments", "thought", "reward", "errors_before")

    def __init__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        thought: str | None,
        reward: float,
        errors_before: list[dict[str, Any]],
    ) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.thought = thought
        self.reward = reward
        self.errors_before = errors_before
