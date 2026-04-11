"""
openpipe-art training adapter.

Takes training trajectories from ``rollouts.export_adapters`` and prepares
them for openpipe-art consumption: applies stage-specific reward views,
builds training-ready records, and handles serialization for the
openpipe-art ingestion contract.

Owns:
    - Applying training reward views to training trajectories
    - Building training-ready records with stage-shaped rewards
    - Batch preparation for openpipe-art training runs
    - SFT-specific formatting for supervised fine-tuning stages

Does NOT own:
    - Training trajectory type definitions or Episode conversion
      (see rollouts.export_adapters)
    - Reward computation from environment transitions (see envs.rewards)
    - Reward component weighting (see training.reward_views)
    - Curriculum stage definitions (see training.curriculum)
    - Dataset filtering (see training.datasets)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

from src.rollouts.export_adapters import (
    TrainingTrajectory,
    episode_to_training_trajectory,
)
from src.training.curriculum import StageConfig, TrainingStage
from src.training.reward_views import (
    EpisodeRewardView,
    build_episode_reward_view,
    get_per_step_rewards,
)
from src.training.datasets import TrainingRecord


@dataclass
class OpenPipeArtTrainingRecord:
    """A training trajectory with stage-specific reward shaping applied for openpipe-art."""

    trajectory: TrainingTrajectory
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


def shape_trajectory_rewards(
    trajectory: TrainingTrajectory,
    reward_view: EpisodeRewardView,
) -> TrainingTrajectory:
    """Apply stage-shaped rewards to a training trajectory in place."""

    shaped_rewards = get_per_step_rewards(reward_view)

    for i, step in enumerate(trajectory.steps):
        if i < len(shaped_rewards):
            step.reward = shaped_rewards[i]

    trajectory.total_reward = round(sum(s.reward for s in trajectory.steps), 4)
    return trajectory


def build_openpipe_art_training_record(
    training_record: TrainingRecord,
    stage_config: StageConfig,
) -> OpenPipeArtTrainingRecord:
    """Convert a TrainingRecord into an openpipe-art training record."""

    episode = training_record.episode
    reward_summary = training_record.reward_summary

    trajectory = episode_to_training_trajectory(episode, reward_summary)
    reward_view = build_episode_reward_view(reward_summary, stage_config)
    shape_trajectory_rewards(trajectory, reward_view)

    return OpenPipeArtTrainingRecord(
        trajectory=trajectory,
        stage=stage_config.stage,
        reward_view=reward_view,
        metadata={
            "original_task_id": episode.task_id,
            "stage_description": stage_config.description,
        },
    )


def build_openpipe_art_training_batch(
    training_records: list[TrainingRecord],
    stage_config: StageConfig,
) -> list[OpenPipeArtTrainingRecord]:
    """Convert a batch of TrainingRecords into openpipe-art training records."""

    return [
        build_openpipe_art_training_record(record, stage_config)
        for record in training_records
    ]


def training_record_to_jsonl(record: OpenPipeArtTrainingRecord) -> str:
    """Serialize a training record to a single JSONL line."""

    return json.dumps(record.to_dict())


def save_training_records_jsonl(
    records: list[OpenPipeArtTrainingRecord],
    path: str,
) -> None:
    """Write training records to a JSONL file."""

    with open(path, "w") as f:
        for record in records:
            f.write(training_record_to_jsonl(record) + "\n")


@dataclass
class SFTTrainingRecord:
    """Supervised fine-tuning record for the SFT stage."""

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
    """Build an SFT training record from a TrainingRecord."""

    from src.training.datasets import extract_sft_record

    sft = extract_sft_record(training_record)
    episode = training_record.episode

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
