"""
NeMo RL-specific training consumption adapter.

Takes NeMo RL trajectories from rollouts.prorl_adapter and adapts them for
NeMo RL trainer ingestion: applies stage-specific reward views, prepares
trainer-ready records, and handles the training-side contract.

Owns:
    - Applying training reward views to NeMo RL trajectories
    - Building trainer-ready records with stage-shaped rewards
    - Batch preparation for NeMo RL trainer ingestion
    - SFT-specific formatting for supervised fine-tuning stages

Does NOT own:
    - NeMo RL trajectory type definitions or Episode conversion
      (see rollouts.prorl_adapter)
    - Reward computation from environment transitions (see envs.rewards)
    - Reward component weighting (see training.reward_views)
    - Curriculum stage definitions (see training.curriculum)
    - Dataset filtering (see training.datasets)
    - Distributed execution (see systems/)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from src.rollouts.prorl_adapter import (
    NeMoRLTrajectory,
    NeMoRLStep,
    episode_to_nemo_trajectory,
)
from src.rollouts.trace_types import Episode
from src.envs.rewards import EpisodeRewardSummary
from src.training.curriculum import StageConfig, TrainingStage
from src.training.reward_views import (
    EpisodeRewardView,
    build_episode_reward_view,
    get_per_step_rewards,
)
from src.training.datasets import TrainingRecord


# -- Trainer-ready record types -----------------------------------------------

@dataclass
class NeMoRLTrainingRecord:
    """A NeMo RL trajectory with training-specific reward shaping applied.

    This is the final format consumed by NeMo RL trainer:
        - trajectory: the observation/action/reward sequence
        - stage: which curriculum stage this record belongs to
        - reward_view: the shaped reward breakdown for inspection
    """
    trajectory: NeMoRLTrajectory
    stage: TrainingStage
    reward_view: EpisodeRewardView | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record = {
            "task_id": self.trajectory.task_id,
            "model_id": self.trajectory.model_id,
            "total_reward": self.trajectory.total_reward,
            "episode_length": self.trajectory.episode_length,
            "stage": self.stage.value,
            "steps": [asdict(s) for s in self.trajectory.steps],
            "metadata": {**self.trajectory.metadata, **self.metadata},
        }
        if self.reward_view is not None:
            record["reward_view"] = self.reward_view.to_dict()
        return record


# -- Trajectory shaping -------------------------------------------------------

def shape_trajectory_rewards(
    trajectory: NeMoRLTrajectory,
    reward_view: EpisodeRewardView,
) -> NeMoRLTrajectory:
    """Apply stage-shaped rewards to a NeMo RL trajectory in place.

    Replaces the per-step rewards on the trajectory with the shaped
    rewards from the training reward view. This ensures the trainer
    sees stage-appropriate reward signals.

    Args:
        trajectory: The NeMo RL trajectory to update.
        reward_view: The shaped reward view from training.reward_views.

    Returns:
        The same trajectory with updated rewards.
    """
    shaped_rewards = get_per_step_rewards(reward_view)

    for i, step in enumerate(trajectory.steps):
        if i < len(shaped_rewards):
            step.reward = shaped_rewards[i]

    trajectory.total_reward = round(sum(s.reward for s in trajectory.steps), 4)
    return trajectory


# -- Record building ----------------------------------------------------------

def build_nemo_training_record(
    training_record: TrainingRecord,
    stage_config: StageConfig,
) -> NeMoRLTrainingRecord:
    """Convert a TrainingRecord into a NeMo RL training-ready record.

    Pipeline:
        1. Convert Episode to NeMo RL trajectory (via prorl_adapter)
        2. Build stage-specific reward view (via reward_views)
        3. Apply shaped rewards to the trajectory
        4. Package as NeMoRLTrainingRecord

    Args:
        training_record: A filtered training record from datasets.py.
        stage_config: The curriculum stage config for reward shaping.

    Returns:
        NeMoRLTrainingRecord ready for NeMo RL trainer ingestion.
    """
    episode = training_record.episode
    reward_summary = training_record.reward_summary

    # Step 1: Convert to NeMo RL trajectory
    trajectory = episode_to_nemo_trajectory(episode, reward_summary)

    # Step 2: Build reward view
    reward_view = build_episode_reward_view(reward_summary, stage_config)

    # Step 3: Apply shaped rewards
    shape_trajectory_rewards(trajectory, reward_view)

    return NeMoRLTrainingRecord(
        trajectory=trajectory,
        stage=stage_config.stage,
        reward_view=reward_view,
        metadata={
            "original_task_id": episode.task_id,
            "stage_description": stage_config.description,
        },
    )


def build_nemo_training_batch(
    training_records: list[TrainingRecord],
    stage_config: StageConfig,
) -> list[NeMoRLTrainingRecord]:
    """Convert a batch of TrainingRecords into NeMo RL training records.

    Args:
        training_records: Filtered training records for one stage.
        stage_config: The curriculum stage config.

    Returns:
        List of NeMoRLTrainingRecords.
    """
    return [
        build_nemo_training_record(record, stage_config)
        for record in training_records
    ]


# -- Serialization ------------------------------------------------------------

def training_record_to_jsonl(record: NeMoRLTrainingRecord) -> str:
    """Serialize a NeMo RL training record to a single JSONL line."""
    return json.dumps(record.to_dict())


def save_training_records_jsonl(
    records: list[NeMoRLTrainingRecord],
    path: str,
) -> None:
    """Write NeMo RL training records to a JSONL file.

    Args:
        records: Training records to write.
        path: Output file path.
    """
    with open(path, "w") as f:
        for record in records:
            f.write(training_record_to_jsonl(record) + "\n")


# -- SFT formatting helpers ---------------------------------------------------

@dataclass
class SFTTrainingRecord:
    """Supervised fine-tuning record for NeMo RL SFT stage.

    Contains prompt and completion pairs for cross-entropy training.
    The completion is the serialized tool-call sequence that the model
    should learn to produce.
    """
    prompt: str
    completion: str
    task_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "completion": self.completion,
            "metadata": self.metadata,
        }


def build_sft_record(training_record: TrainingRecord) -> SFTTrainingRecord:
    """Build an SFT training record from a TrainingRecord.

    The prompt is the task description. The completion is the serialized
    sequence of tool calls and final answer that represents the ideal
    agent behavior.

    Args:
        training_record: A TrainingRecord from the SFT stage dataset.

    Returns:
        SFTTrainingRecord with prompt/completion pair.
    """
    from src.training.datasets import extract_sft_record

    sft = extract_sft_record(training_record)
    episode = training_record.episode

    # Build the completion: each tool call on its own line, then final answer
    completion_parts: list[str] = []
    for tc in sft.tool_call_sequence:
        completion_parts.append(json.dumps(tc))
    if sft.final_answer is not None:
        completion_parts.append(json.dumps({"final_answer": sft.final_answer}))

    completion = "\n".join(completion_parts)

    return SFTTrainingRecord(
        prompt=episode.task_prompt,
        completion=completion,
        task_id=episode.task_id,
        metadata={"num_tool_calls": len(sft.tool_call_sequence)},
    )


def save_sft_records_jsonl(
    records: list[SFTTrainingRecord],
    path: str,
) -> None:
    """Write SFT training records to a JSONL file."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record.to_dict()) + "\n")
