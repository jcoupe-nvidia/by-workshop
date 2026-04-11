"""
nemo-gym adapter: formats environment outputs for nemo-gym rollout collection.

nemo-gym collects rollouts via HTTP agent servers and profiles rewards by
aggregating numeric fields from result rows. This module converts enriched
episodes into the JSONL row format that nemo-gym's RewardProfiler can
process, and provides the input row format for rollout collection.

Owns:
    - Episode -> nemo-gym result row conversion
    - Task -> nemo-gym input row formatting
    - Reward field extraction for profiling

Does NOT own:
    - Environment state or transitions (see envs/)
    - Episode types (see rollouts.trace_types)
    - HTTP agent server implementation
    - Rollout orchestration (that's nemo-gym's job)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.envs.rewards import EpisodeRewardSummary, RewardSignal
from src.rollouts.trace_types import Episode, EpisodeMetrics


# ---------------------------------------------------------------------------
# nemo-gym input row format
# ---------------------------------------------------------------------------

def build_rollout_input_row(
    order_id: str,
    task_prompt: str,
    agent_name: str = "late-order-recovery-agent",
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
) -> dict[str, Any]:
    """Build a nemo-gym input row for a rollout collection run.

    This is the format expected by nemo-gym's RolloutCollectionHelper.
    Each row describes one task for the agent to solve.

    Args:
        order_id: The order ID to investigate.
        task_prompt: The full task prompt for the agent.
        agent_name: Agent server name for routing.
        temperature: Sampling temperature.
        max_output_tokens: Max tokens per model response.
    """
    return {
        "agent_ref": {"name": agent_name},
        "responses_create_params": {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        },
        "order_id": order_id,
        "task_prompt": task_prompt,
    }


# ---------------------------------------------------------------------------
# nemo-gym result row format (from enriched episodes)
# ---------------------------------------------------------------------------

@dataclass
class NemoGymResultRow:
    """A single result row in nemo-gym's expected format.

    Numeric fields are automatically aggregated by nemo-gym's RewardProfiler
    (mean, max, min, median, std, histogram). Non-numeric fields are preserved
    in the "response" dict for inspection.
    """
    # Identity
    order_id: str
    agent_ref: dict[str, str]

    # Numeric reward fields (profiled by nemo-gym)
    total_reward: float
    avg_step_reward: float
    valid_tool_calls: int
    invalid_tool_calls: int
    repair_attempts: int
    rejects: int
    episode_length: int
    task_success: int  # 1 if final_answer with correct action, 0 otherwise

    # Reward component averages (for detailed profiling)
    avg_valid_call: float
    avg_correct_tool: float
    avg_correct_arguments: float
    avg_dependency_satisfied: float
    avg_non_redundant: float
    avg_progress: float
    avg_efficiency: float
    terminal_quality: float

    # Penalty counts
    total_penalties: int

    # Non-numeric metadata (in "response" dict for nemo-gym)
    response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict suitable for nemo-gym JSONL output."""
        d: dict[str, Any] = {
            "agent_ref": self.agent_ref,
            "order_id": self.order_id,
            "total_reward": self.total_reward,
            "avg_step_reward": self.avg_step_reward,
            "valid_tool_calls": self.valid_tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
            "repair_attempts": self.repair_attempts,
            "rejects": self.rejects,
            "episode_length": self.episode_length,
            "task_success": self.task_success,
            "avg_valid_call": self.avg_valid_call,
            "avg_correct_tool": self.avg_correct_tool,
            "avg_correct_arguments": self.avg_correct_arguments,
            "avg_dependency_satisfied": self.avg_dependency_satisfied,
            "avg_non_redundant": self.avg_non_redundant,
            "avg_progress": self.avg_progress,
            "avg_efficiency": self.avg_efficiency,
            "terminal_quality": self.terminal_quality,
            "total_penalties": self.total_penalties,
            "response": self.response,
        }
        return d


def episode_to_nemo_gym_row(
    episode: Episode,
    reward_summary: EpisodeRewardSummary,
    agent_name: str = "late-order-recovery-agent",
) -> NemoGymResultRow:
    """Convert an enriched Episode + reward summary to a nemo-gym result row.

    The numeric fields in the result row are what nemo-gym's RewardProfiler
    will aggregate across rollouts.
    """
    metrics = episode.metrics

    # Compute average reward components across steps
    step_rewards = reward_summary.step_rewards
    n_steps = max(len(step_rewards), 1)

    avg_valid_call = sum(r.valid_call for r in step_rewards) / n_steps
    avg_correct_tool = sum(r.correct_tool for r in step_rewards) / n_steps
    avg_correct_arguments = sum(r.correct_arguments for r in step_rewards) / n_steps
    avg_dependency_satisfied = sum(r.dependency_satisfied for r in step_rewards) / n_steps
    avg_non_redundant = sum(r.non_redundant for r in step_rewards) / n_steps
    avg_progress = sum(r.progress for r in step_rewards) / n_steps
    avg_efficiency = sum(r.efficiency for r in step_rewards) / n_steps

    terminal_quality = (
        reward_summary.terminal_reward.terminal_quality
        if reward_summary.terminal_reward else 0.0
    )

    total_penalties = sum(reward_summary.penalty_counts.values())

    # Determine task success
    task_success = 0
    if episode.terminal and episode.terminal.final_answer:
        action = episode.terminal.final_answer.get("action", "")
        if action and action != "escalate":
            task_success = 1

    # Build response metadata
    response: dict[str, Any] = {
        "usage": {
            "total_steps": metrics.total_steps,
            "model_calls": metrics.model_calls,
            "wall_time_seconds": metrics.wall_time_seconds,
        },
        "terminal_reason": episode.terminal.reason if episode.terminal else "unknown",
        "final_answer": episode.terminal.final_answer if episode.terminal else None,
        "penalty_counts": reward_summary.penalty_counts,
    }

    return NemoGymResultRow(
        order_id=episode.task_id,
        agent_ref={"name": agent_name},
        total_reward=round(reward_summary.total_reward, 4),
        avg_step_reward=round(reward_summary.avg_step_reward, 4),
        valid_tool_calls=metrics.valid_tool_calls,
        invalid_tool_calls=metrics.invalid_tool_calls,
        repair_attempts=metrics.repair_attempts,
        rejects=metrics.rejects,
        episode_length=metrics.total_steps,
        task_success=task_success,
        avg_valid_call=round(avg_valid_call, 4),
        avg_correct_tool=round(avg_correct_tool, 4),
        avg_correct_arguments=round(avg_correct_arguments, 4),
        avg_dependency_satisfied=round(avg_dependency_satisfied, 4),
        avg_non_redundant=round(avg_non_redundant, 4),
        avg_progress=round(avg_progress, 4),
        avg_efficiency=round(avg_efficiency, 4),
        terminal_quality=round(terminal_quality, 4),
        total_penalties=total_penalties,
        response=response,
    )


# ---------------------------------------------------------------------------
# JSONL serialization
# ---------------------------------------------------------------------------

def save_nemo_gym_rows_jsonl(
    rows: list[NemoGymResultRow],
    path: str,
) -> None:
    """Write nemo-gym result rows to a JSONL file."""
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict()) + "\n")


def save_nemo_gym_inputs_jsonl(
    input_rows: list[dict[str, Any]],
    path: str,
) -> None:
    """Write nemo-gym input rows to a JSONL file."""
    with open(path, "w") as f:
        for row in input_rows:
            f.write(json.dumps(row) + "\n")
