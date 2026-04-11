"""
Backward-compatibility shim for src.training_export.

The canonical definitions now live in:
    - src.rollouts.export_adapters  (training trajectory types, export, serialization)
    - src.training.reward_views     (stage-aware reward views and shaping)
    - src.training.datasets         (training record types and dataset assembly)
    - src.envs.rewards              (reward computation)

This module re-exports everything so existing imports continue to work.
It also provides adapter functions that accept the backward-compatible
AgentTrace type, converting to canonical Episode before calling canonical code.
"""
from __future__ import annotations

from src.rollouts.export_adapters import (  # noqa: F401
    TrainingTrajectoryStep,
    TrainingTrajectory,
    episode_to_training_trajectory,
    training_trajectory_to_jsonl as trajectory_to_jsonl,
    save_training_trajectories_jsonl as save_trajectories_jsonl,
)

from src.training.reward_views import (  # noqa: F401
    STAGE_REWARD_WEIGHTS,
    StepRewardView,
    EpisodeRewardView,
    build_episode_reward_view,
    get_per_step_rewards,
)

# -- AgentTrace adapters (kept for notebook backward compat) -----------------
#
# The functions below accept the legacy AgentTrace type and convert to
# canonical Episode / enriched-episode paths before calling canonical code.
# New code should use the canonical modules directly.

from typing import Any

from src.runtime.agent import AgentTrace, ToolCallRecord
from src.evaluation import (
    evaluate_trajectory,
    TrajectoryEvaluation,
    DimensionScore,
    DIMENSION_WEIGHTS,
    EVAL_DIMENSIONS,
    OPTIMAL_TOOL_SEQUENCE,
    _agent_trace_to_episode,
)
from src.rollouts.episode_runner import enrich_episode
from src.runtime.tools import TOOL_DEPENDENCIES


STEP_REWARD_COMPONENTS = {
    "tool_validity":        0.25,
    "sequence_correctness": 0.35,
    "tool_accuracy":        0.20,
    "recovery_bonus":       0.20,
}


def compute_step_rewards(
    trace: AgentTrace,
    expected_arguments: dict[str, dict[str, Any]] | None = None,
) -> list[float]:
    """Compute per-step reward signals for a legacy AgentTrace.

    Kept for backward compatibility.  New code should use the canonical
    path: build an Episode, enrich it via episode_runner, then read
    rewards from the EpisodeRewardSummary.
    """
    if expected_arguments is None:
        from src.envs.rewards import EXPECTED_ARGUMENTS
        expected_arguments = EXPECTED_ARGUMENTS

    rewards: list[float] = []
    called_so_far: set[str] = set()

    for step in trace.steps:
        components: dict[str, float] = {}
        components["tool_validity"] = 1.0 if step.valid else -0.5

        if step.valid:
            deps = TOOL_DEPENDENCIES.get(step.tool_name, set())
            missing_deps = deps - called_so_far
            components["sequence_correctness"] = 1.0 if not missing_deps else -1.0
            called_so_far.add(step.tool_name)
        else:
            components["sequence_correctness"] = -0.5

        if step.valid and step.tool_name in expected_arguments:
            expected = expected_arguments[step.tool_name]
            matches = sum(
                1 for k, v in expected.items()
                if str(step.arguments.get(k)) == str(v)
            )
            components["tool_accuracy"] = matches / len(expected) if expected else 1.0
        else:
            components["tool_accuracy"] = 0.5

        if step.fallback_action == "repaired" and step.valid:
            components["recovery_bonus"] = 0.5
        elif step.fallback_action == "repaired" and not step.valid:
            components["recovery_bonus"] = -0.25
        else:
            components["recovery_bonus"] = 0.0

        reward = sum(
            components[k] * STEP_REWARD_COMPONENTS[k]
            for k in STEP_REWARD_COMPONENTS
        )
        rewards.append(round(reward, 4))

    return rewards


def compute_trajectory_reward(
    evaluation: TrajectoryEvaluation,
    step_rewards: list[float],
    step_weight: float = 0.4,
    trajectory_weight: float = 0.6,
) -> float:
    """Combine step-level and trajectory-level rewards into a single signal.

    Kept for backward compatibility.  New code should use
    build_episode_reward_view() from src.training.reward_views.
    """
    avg_step_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
    trajectory_score = evaluation.overall
    combined = (step_weight * avg_step_reward) + (trajectory_weight * trajectory_score)
    return round(combined, 4)


def print_reward_breakdown(
    trace: AgentTrace,
    step_rewards: list[float],
    evaluation: TrajectoryEvaluation,
) -> None:
    """Pretty-print the per-step and trajectory reward breakdown.

    Kept for backward compatibility.
    """
    print("Per-step rewards (training decomposition)")
    print("-" * 75)
    print(f"{'Step':>4}  {'Tool':<30}  {'Valid':>5}  {'Reward':>7}  Notes")
    print("-" * 75)

    for i, (step, reward) in enumerate(zip(trace.steps, step_rewards)):
        notes = []
        if step.fallback_action == "repaired":
            notes.append("repaired")
        if not step.valid:
            notes.append("invalid")
        note_str = ", ".join(notes) if notes else ""
        print(
            f"{i:>4}  {step.tool_name:<30}  "
            f"{'Y' if step.valid else 'N':>5}  "
            f"{reward:>+7.3f}  {note_str}"
        )

    print("-" * 75)
    total = compute_trajectory_reward(evaluation, step_rewards)
    avg_step = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
    print(f"Avg step reward: {avg_step:+.3f}")
    print(f"Trajectory score: {evaluation.overall:.3f}")
    print(f"Combined reward:  {total:+.4f}")


def export_training_trajectory(
    trace: AgentTrace,
    evaluation: TrajectoryEvaluation,
    step_rewards: list[float],
    model_id: str = "nvidia/nemotron-3-nano",
) -> TrainingTrajectory:
    """Convert an AgentTrace + evaluation into a training trajectory record.

    Kept for backward compatibility.  New code should use
    episode_to_training_trajectory() from src.rollouts.export_adapters
    with a canonical Episode.
    """
    import re

    episode = _agent_trace_to_episode(trace)
    episode.model_id = model_id

    # Try to extract order ID for environment enrichment
    order_match = re.search(r"(SO-\d+)", trace.task)
    order_id = order_match.group(1) if order_match else None

    if order_id:
        try:
            enriched = enrich_episode(episode, order_id=order_id)
            traj = episode_to_training_trajectory(
                enriched.episode,
                reward_summary=enriched.reward_summary,
            )
        except (ValueError, KeyError):
            # Fall back to export without environment rewards
            traj = episode_to_training_trajectory(episode)
    else:
        traj = episode_to_training_trajectory(episode)

    # Augment metadata with legacy evaluation fields
    traj.metadata["evaluation_overall"] = evaluation.overall
    traj.metadata["evaluation_passed"] = evaluation.passed

    return traj
