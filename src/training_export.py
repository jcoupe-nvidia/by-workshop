"""
Training-oriented export utilities for trajectory export and reward design.

Converts agent traces and evaluation results into formats suitable for
downstream training workflows:

    - Training trajectory export:  JSONL records with per-step observations,
                                   actions, and rewards for post-training flows.
    - Reward computation:          Per-step and trajectory-level reward signals
                                   derived from the seven evaluation dimensions.
    - Legacy systems config:       A historical config sketch that references
                                   the exported data without driving active code paths.

Integration is narrow and demonstrative -- the goal is to show *what* gets
exported and *how* reward signals are shaped, not to build a full pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

from src.agent_loop import AgentTrace, ToolCallRecord, trace_to_trajectory
from src.evaluation import (
    evaluate_trajectory,
    TrajectoryEvaluation,
    DimensionScore,
    DIMENSION_WEIGHTS,
    EVAL_DIMENSIONS,
    OPTIMAL_TOOL_SEQUENCE,
)
from src.tools import TOOL_DEPENDENCIES


# ---------------------------------------------------------------------------
# 1. Training trajectory export
# ---------------------------------------------------------------------------

@dataclass
class TrainingTrajectoryStep:
    """One step in a training-compatible trajectory record."""
    step_index: int
    observation: str       # what the agent saw before acting
    action: str            # the tool call or final answer (as JSON string)
    reward: float          # per-step reward signal
    done: bool             # whether this step ends the episode
    info: dict[str, Any]   # metadata: tool validity, fallback, etc.


@dataclass
class TrainingTrajectory:
    """Full trajectory formatted for downstream training consumption."""
    task_id: str
    model_id: str
    steps: list[TrainingTrajectoryStep]
    total_reward: float
    episode_length: int
    metadata: dict[str, Any]


def export_training_trajectory(
    trace: AgentTrace,
    evaluation: TrajectoryEvaluation,
    step_rewards: list[float],
    model_id: str = "nvidia/nemotron-3-nano",
) -> TrainingTrajectory:
    """Convert an AgentTrace + evaluation into a training trajectory record.

    Args:
        trace: The agent trace from a completed run.
        evaluation: The seven-dimension evaluation of the trace.
        step_rewards: Per-step reward values (from compute_step_rewards).
        model_id: The model identifier for provenance.

    Returns:
        TrainingTrajectory ready for serialization.
    """
    training_steps: list[TrainingTrajectoryStep] = []

    for i, step in enumerate(trace.steps):
        # Observation: the tool result from the previous step (or task prompt)
        if i == 0:
            observation = trace.task
        else:
            prev = trace.steps[i - 1]
            observation = json.dumps(prev.result)

        # Action: the tool call as a structured JSON string
        action = json.dumps({
            "tool_call": {
                "name": step.tool_name,
                "arguments": step.arguments,
            },
            "thought": step.thought,
        })

        reward = step_rewards[i] if i < len(step_rewards) else 0.0
        is_last = (i == len(trace.steps) - 1) and trace.completed

        training_steps.append(TrainingTrajectoryStep(
            step_index=i,
            observation=observation,
            action=action,
            reward=reward,
            done=is_last,
            info={
                "tool_name": step.tool_name,
                "valid": step.valid,
                "fallback_action": step.fallback_action,
                "iteration": step.iteration,
            },
        ))

    # Add final answer step if present
    if trace.final_answer:
        final_obs = json.dumps(trace.steps[-1].result) if trace.steps else trace.task
        final_action = json.dumps({"final_answer": trace.final_answer})
        # Final step gets the task-success component of the reward
        final_reward = next(
            (s.score for s in evaluation.scores if s.dimension == "task_success"),
            0.0,
        )
        training_steps.append(TrainingTrajectoryStep(
            step_index=len(trace.steps),
            observation=final_obs,
            action=final_action,
            reward=final_reward,
            done=True,
            info={"tool_name": "<final_answer>", "valid": True},
        ))

    total_reward = sum(s.reward for s in training_steps)

    return TrainingTrajectory(
        task_id=trace.task,
        model_id=model_id,
        steps=training_steps,
        total_reward=round(total_reward, 4),
        episode_length=len(training_steps),
        metadata={
            "evaluation_overall": evaluation.overall,
            "evaluation_passed": evaluation.passed,
            "wall_time_seconds": trace.wall_time_seconds,
            "model_calls": trace.model_calls,
            "fallback_repairs": trace.fallback_repairs,
            "fallback_rejects": trace.fallback_rejects,
        },
    )


def trajectory_to_jsonl(trajectory: TrainingTrajectory) -> str:
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


# ---------------------------------------------------------------------------
# 2. Step and trajectory reward computation
# ---------------------------------------------------------------------------

# Per-step reward components and their weights.
# These map directly to the evaluation dimensions but are applied at the
# step level rather than the trajectory level.
STEP_REWARD_COMPONENTS = {
    "tool_validity":        0.25,   # was this specific call well-formed?
    "sequence_correctness": 0.35,   # were dependencies satisfied at this step?
    "tool_accuracy":        0.20,   # did arguments match expected values?
    "recovery_bonus":       0.20,   # bonus for successful fallback repair
}


def compute_step_rewards(
    trace: AgentTrace,
    expected_arguments: dict[str, dict[str, Any]] | None = None,
) -> list[float]:
    """Compute per-step reward signals using the repo's reward decomposition.

    Each step receives a reward based on:
        - Tool validity:  +1.0 if valid, -0.5 if invalid
        - Sequence correctness: +1.0 if all dependencies met, -1.0 if violated
        - Tool accuracy:  +1.0 if arguments match expected, +0.5 if partially correct
        - Recovery bonus:  +0.5 if the step was successfully repaired via fallback

    These component scores are combined using STEP_REWARD_COMPONENTS weights.

    Args:
        trace: The agent trace to score.
        expected_arguments: Optional dict of tool_name -> expected args for
                           accuracy checking. If None, accuracy gets +0.5 default.

    Returns:
        List of float rewards, one per step in trace.steps.
    """
    if expected_arguments is None:
        from src.evaluation import EXPECTED_ARGUMENTS
        expected_arguments = EXPECTED_ARGUMENTS

    rewards: list[float] = []
    called_so_far: set[str] = set()

    for step in trace.steps:
        components: dict[str, float] = {}

        # Tool validity
        components["tool_validity"] = 1.0 if step.valid else -0.5

        # Sequence correctness
        if step.valid:
            deps = TOOL_DEPENDENCIES.get(step.tool_name, set())
            missing_deps = deps - called_so_far
            components["sequence_correctness"] = 1.0 if not missing_deps else -1.0
            called_so_far.add(step.tool_name)
        else:
            components["sequence_correctness"] = -0.5

        # Tool accuracy
        if step.valid and step.tool_name in expected_arguments:
            expected = expected_arguments[step.tool_name]
            matches = sum(
                1 for k, v in expected.items()
                if str(step.arguments.get(k)) == str(v)
            )
            components["tool_accuracy"] = matches / len(expected) if expected else 1.0
        else:
            components["tool_accuracy"] = 0.5  # neutral default

        # Recovery bonus
        if step.fallback_action == "repaired" and step.valid:
            components["recovery_bonus"] = 0.5
        elif step.fallback_action == "repaired" and not step.valid:
            components["recovery_bonus"] = -0.25
        else:
            components["recovery_bonus"] = 0.0

        # Weighted sum
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

    The total reward is framed as a blend of:
        - Dense step-level rewards (encourage correct behavior at each step)
        - Sparse trajectory-level reward (overall task success)

    The trajectory-level component uses the weighted evaluation score
    from the seven-dimension evaluator.

    Args:
        evaluation: The trajectory evaluation result.
        step_rewards: Per-step rewards from compute_step_rewards.
        step_weight: Weight for the step-level component (0-1).
        trajectory_weight: Weight for the trajectory-level component (0-1).

    Returns:
        Combined reward signal (float).
    """
    avg_step_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
    trajectory_score = evaluation.overall

    combined = (step_weight * avg_step_reward) + (trajectory_weight * trajectory_score)
    return round(combined, 4)


# ---------------------------------------------------------------------------
# 3. Reward signal summary (for display / inspection)
# ---------------------------------------------------------------------------

def print_reward_breakdown(
    trace: AgentTrace,
    step_rewards: list[float],
    evaluation: TrajectoryEvaluation,
) -> None:
    """Pretty-print the per-step and trajectory reward breakdown."""
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


# ---------------------------------------------------------------------------
# 4. Reference scale-out config sketch
# ---------------------------------------------------------------------------

def reference_scaleout_config_sketch(
    num_trajectories: int = 100,
    model_name: str = "nvidia/nemotron-3-nano",
) -> dict[str, Any]:
    """Generate a reference scale-out config sketch for illustration only.

    This is a conceptual reference, not a runnable config. It shows how the
    exported trajectory data and reward signals could be wired into a larger
    training system without making that system an active dependency.

    Args:
        num_trajectories: Expected number of training trajectories.
        model_name: Base model for fine-tuning.

    Returns:
        Dict representing the config sketch.
    """
    return {
        "# NOTE": "Reference scale-out systems sketch -- not an active implementation path",
        "model": {
            "name": model_name,
            "type": "causal_lm",
            "precision": "bf16",
        },
        "training": {
            "method": "ppo",
            "num_episodes": num_trajectories,
            "max_steps_per_episode": 15,
            "batch_size": 4,
            "learning_rate": 1e-6,
            "kl_penalty_coeff": 0.02,
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
        },
        "reward": {
            "type": "composite",
            "step_reward_weight": 0.4,
            "trajectory_reward_weight": 0.6,
            "components": {
                "tool_validity": {"weight": 0.25, "type": "binary"},
                "sequence_correctness": {"weight": 0.35, "type": "dependency_check"},
                "tool_accuracy": {"weight": 0.20, "type": "argument_match"},
                "recovery_bonus": {"weight": 0.20, "type": "fallback_success"},
            },
            "trajectory_components": {
                dim: {"weight": w}
                for dim, w in DIMENSION_WEIGHTS.items()
            },
        },
        "data": {
            "format": "jsonl",
            "trajectory_file": "trajectories/training_export.jsonl",
            "fields": ["observation", "action", "reward", "done"],
        },
        "infrastructure": {
            "gpus": 8,
            "gpu_type": "H100",
            "tensor_parallel": 2,
            "pipeline_parallel": 1,
            "data_parallel": 4,
            "note": (
                "8x H100 target environment. Tensor parallelism splits the model "
                "across 2 GPUs; data parallelism processes 4 trajectory batches "
                "concurrently. For this small model, the setup is conservative -- "
                "the same cluster could handle larger models with adjusted parallelism."
            ),
        },
        "opencode_inspired_architecture": {
            "role": (
                "OpenCode-inspired architecture implemented locally for "
                "visibility, trace capture, and evaluation"
            ),
            "trace_format": "AgentTrace with ToolCallRecord steps",
            "evaluation_format": "TrajectoryEvaluation with 7 DimensionScores",
        },
    }


# ---------------------------------------------------------------------------
# 5. Export to file helpers
# ---------------------------------------------------------------------------

def save_trajectories_jsonl(
    trajectories: list[TrainingTrajectory],
    path: str,
) -> None:
    """Write a list of training trajectories to a JSONL file.

    Args:
        trajectories: List of trajectories to export.
        path: Output file path.
    """
    with open(path, "w") as f:
        for traj in trajectories:
            f.write(trajectory_to_jsonl(traj) + "\n")


def save_training_config(config: dict[str, Any], path: str) -> None:
    """Write a reference scale-out config sketch to a JSON file.

    Args:
        config: Config dict from reference_scaleout_config_sketch.
        path: Output file path.
    """
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
