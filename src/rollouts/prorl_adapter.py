"""
ProRL adapter: maps canonical Episodes to NeMo RL-compatible trajectory format.

Owns:
    - Converting enriched Episodes into NeMo RL trajectory records
    - Per-step observation/action/reward triple extraction from Episode events
    - Stable JSONL serialization of NeMo RL trajectories
    - ProRL-style reward decomposition view over Episode rewards

Does NOT own:
    - Episode type definitions (see rollouts.trace_types)
    - Reward computation (see envs.rewards)
    - Environment state or transitions (see envs/)
    - Megatron training configuration (see systems/)
    - Training dataset views or curriculum staging (see training/)

This module replaces the AgentTrace-consuming export path in training_export.py
with a new path that consumes the canonical Episode type. It produces records
suitable for NeMo RL ingestion without encoding trainer-specific semantics.

Usage::

    from src.rollouts.prorl_adapter import episode_to_nemo_trajectory

    trajectory = episode_to_nemo_trajectory(enriched_result.episode)
    jsonl_line = nemo_trajectory_to_jsonl(trajectory)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from src.rollouts.trace_types import (
    Episode,
    Event,
    EventType,
    ToolCallPayload,
    ToolResultPayload,
    TerminalOutcomePayload,
    ValidationErrorPayload,
)
from src.envs.rewards import EpisodeRewardSummary


# -- NeMo RL trajectory types ------------------------------------------------

@dataclass
class NeMoRLStep:
    """One step in a NeMo RL-compatible trajectory record.

    Each step represents one agent turn with:
        - observation: what the agent saw before acting
        - action: the tool call or final answer (as JSON string)
        - reward: per-step reward signal from the environment
        - done: whether this step ends the episode
        - info: metadata for debugging and ProRL reward decomposition
    """
    step_index: int
    observation: str
    action: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class NeMoRLTrajectory:
    """Full trajectory formatted for NeMo RL ingestion.

    This is the stable output format for downstream training consumption.
    It preserves the NeMo RL conventions: one record per episode with
    nested step arrays containing observation/action/reward triples.
    """
    task_id: str
    model_id: str
    steps: list[NeMoRLStep] = field(default_factory=list)
    total_reward: float = 0.0
    episode_length: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# -- Conversion from Episode to NeMo RL trajectory ----------------------------

def episode_to_nemo_trajectory(
    episode: Episode,
    reward_summary: EpisodeRewardSummary | None = None,
) -> NeMoRLTrajectory:
    """Convert a canonical Episode into a NeMo RL trajectory record.

    The conversion walks episode events to extract observation/action/reward
    triples. Each valid tool call becomes one NeMo RL step. Validation errors
    and repairs are captured in the step info but don't become separate steps
    — this matches the NeMo RL expectation of clean action sequences.

    The final answer (if present) becomes the terminal step.

    Args:
        episode: An Episode (ideally already enriched with rewards via
                 episode_runner.enrich_episode).
        reward_summary: Optional reward summary for metadata. If None,
                        the adapter uses Episode.metrics.total_reward.

    Returns:
        NeMoRLTrajectory ready for serialization.
    """
    steps: list[NeMoRLStep] = []
    step_idx = 0

    # Walk events to build observation/action pairs
    # Observations come from preceding tool results
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
                # Emit a NeMo RL step for this tool call
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

                steps.append(NeMoRLStep(
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
                    # Final answer becomes the terminal step
                    action_str = json.dumps({"final_answer": payload.final_answer})
                    steps.append(NeMoRLStep(
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
                else:
                    # Non-answer terminal (max_iterations, error)
                    # Mark the last step as done if it exists
                    if steps:
                        steps[-1].done = True
                        steps[-1].info["terminal_reason"] = payload.reason

    total_reward = sum(s.reward for s in steps)

    # Build metadata
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

    return NeMoRLTrajectory(
        task_id=episode.task_id,
        model_id=episode.model_id,
        steps=steps,
        total_reward=round(total_reward, 4),
        episode_length=len(steps),
        metadata=meta,
    )


# -- Serialization helpers ----------------------------------------------------

def nemo_trajectory_to_jsonl(trajectory: NeMoRLTrajectory) -> str:
    """Serialize a NeMo RL trajectory to a single JSONL line."""
    record = {
        "task_id": trajectory.task_id,
        "model_id": trajectory.model_id,
        "total_reward": trajectory.total_reward,
        "episode_length": trajectory.episode_length,
        "steps": [asdict(s) for s in trajectory.steps],
        "metadata": trajectory.metadata,
    }
    return json.dumps(record)


def save_nemo_trajectories_jsonl(
    trajectories: list[NeMoRLTrajectory],
    path: str,
) -> None:
    """Write NeMo RL trajectories to a JSONL file.

    Args:
        trajectories: List of trajectories to write.
        path: Output file path.
    """
    with open(path, "w") as f:
        for traj in trajectories:
            f.write(nemo_trajectory_to_jsonl(traj) + "\n")


# -- Internal helpers ---------------------------------------------------------

class _PendingAction:
    """Transient state for building NeMo RL steps from events."""
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
