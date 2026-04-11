"""
Training-oriented dataset views over enriched Episodes.

Takes enriched Episodes (with rewards attached by episode_runner) and produces
filtered, stage-appropriate training datasets. Works with curriculum StageConfig
to select episodes matching each stage's filter criteria.

Owns:
    - Episode filtering by curriculum stage criteria
    - Training record construction from enriched episodes
    - Stage-aware dataset assembly and partitioning
    - SFT vs RL record format differences

Does NOT own:
    - Episode type definitions (see rollouts.trace_types)
    - Reward computation (see envs.rewards)
    - Reward shaping (see training.reward_views)
    - Curriculum stage definitions (see training.curriculum)
    - openpipe-art record building (see training.openpipe_art_adapter)
    - Episode serialization (see rollouts.serializers)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.rollouts.trace_types import (
    Episode,
    EpisodeMetrics,
    EventType,
    ToolCallPayload,
)
from src.envs.rewards import EpisodeRewardSummary
from src.rollouts.episode_runner import EnrichedEpisodeResult
from src.training.curriculum import StageConfig, TrainingStage


# -- Training record types ----------------------------------------------------

@dataclass
class TrainingRecord:
    """One episode formatted for training consumption.

    Contains the episode, its reward summary, and stage-specific metadata.
    Downstream adapters (e.g. openpipe_art_adapter, SFT extraction) consume
    this to produce their format-specific records.
    """
    episode: Episode
    reward_summary: EpisodeRewardSummary
    stage: TrainingStage
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingDataset:
    """A stage-specific collection of training records.

    Assembled by filtering enriched episodes through a stage's criteria.
    """
    stage: TrainingStage
    stage_config: StageConfig
    records: list[TrainingRecord] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def total_reward(self) -> float:
        if not self.records:
            return 0.0
        return round(
            sum(r.reward_summary.total_reward for r in self.records), 4,
        )

    @property
    def avg_reward(self) -> float:
        if not self.records:
            return 0.0
        return round(self.total_reward / len(self.records), 4)

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "num_records": self.size,
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "stage_description": self.stage_config.description,
        }


# -- Episode filtering --------------------------------------------------------

def matches_stage_filter(
    episode: Episode,
    reward_summary: EpisodeRewardSummary,
    stage_config: StageConfig,
) -> bool:
    """Check whether an enriched episode matches a stage's filter criteria.

    The filter criteria in StageConfig.episode_filter use keys that map to
    Episode and EpisodeMetrics fields:

        is_complete          -> episode.is_complete
        total_reward_min     -> reward_summary.total_reward >= threshold
        invalid_tool_calls_max -> episode.metrics.invalid_tool_calls <= max
        rejects_max          -> episode.metrics.rejects <= max
        min_steps            -> episode.metrics.valid_tool_calls >= min
        max_steps            -> episode.metrics.valid_tool_calls <= max
        include_failures     -> True (no additional filtering)
        include_repairs      -> True (no additional filtering)

    Args:
        episode: The episode to check.
        reward_summary: The episode's reward summary.
        stage_config: The stage whose filter criteria to apply.

    Returns:
        True if the episode passes all filter criteria.
    """
    filters = stage_config.episode_filter
    metrics = episode.metrics

    # is_complete
    if "is_complete" in filters:
        if filters["is_complete"] and not episode.is_complete:
            return False

    # total_reward_min
    if "total_reward_min" in filters:
        if reward_summary.total_reward < filters["total_reward_min"]:
            return False

    # invalid_tool_calls_max
    if "invalid_tool_calls_max" in filters:
        if metrics.invalid_tool_calls > filters["invalid_tool_calls_max"]:
            return False

    # rejects_max
    if "rejects_max" in filters:
        if metrics.rejects > filters["rejects_max"]:
            return False

    # min_steps (valid tool calls)
    if "min_steps" in filters:
        if metrics.valid_tool_calls < filters["min_steps"]:
            return False

    # max_steps (valid tool calls)
    if "max_steps" in filters:
        if metrics.valid_tool_calls > filters["max_steps"]:
            return False

    # include_failures and include_repairs are permissive flags — no filtering
    return True


# -- Dataset assembly ---------------------------------------------------------

def build_training_dataset(
    enriched_results: list[EnrichedEpisodeResult],
    stage_config: StageConfig,
) -> TrainingDataset:
    """Build a training dataset for one curriculum stage.

    Filters enriched episodes through the stage's criteria and wraps
    matching episodes into TrainingRecords.

    Args:
        enriched_results: List of enriched episode results from rollout.
        stage_config: The curriculum stage to build a dataset for.

    Returns:
        TrainingDataset with filtered, stage-appropriate records.
    """
    records: list[TrainingRecord] = []

    for result in enriched_results:
        if matches_stage_filter(result.episode, result.reward_summary, stage_config):
            # Truncate episode if needed for stage's max_episode_length
            episode = _maybe_truncate(result.episode, stage_config.max_episode_length)

            records.append(TrainingRecord(
                episode=episode,
                reward_summary=result.reward_summary,
                stage=stage_config.stage,
                metadata={
                    "original_task_id": result.episode.task_id,
                    "env_final_state": result.env_final_state,
                },
            ))

    return TrainingDataset(
        stage=stage_config.stage,
        stage_config=stage_config,
        records=records,
    )


def build_all_stage_datasets(
    enriched_results: list[EnrichedEpisodeResult],
    stage_configs: list[StageConfig],
) -> dict[TrainingStage, TrainingDataset]:
    """Build training datasets for all curriculum stages.

    Each episode may appear in multiple stage datasets if it matches
    multiple stage filters (e.g., a 3-step successful episode matches
    both SFT_SUCCESSFUL and SHORT_HORIZON_RL).

    Args:
        enriched_results: All enriched episode results.
        stage_configs: List of curriculum stage configs.

    Returns:
        Dict mapping each stage to its TrainingDataset.
    """
    return {
        config.stage: build_training_dataset(enriched_results, config)
        for config in stage_configs
    }


# -- SFT record extraction ---------------------------------------------------

@dataclass
class SFTRecord:
    """One episode formatted for supervised fine-tuning.

    SFT records contain the prompt and the ideal response sequence
    (tool calls in order) for cross-entropy training. No reward signals.
    """
    task_prompt: str
    tool_call_sequence: list[dict[str, Any]]
    final_answer: dict[str, Any] | None
    task_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "tool_call_sequence": self.tool_call_sequence,
            "final_answer": self.final_answer,
        }


def extract_sft_record(record: TrainingRecord) -> SFTRecord:
    """Extract an SFT-formatted record from a TrainingRecord.

    Produces a prompt + ideal response sequence suitable for
    supervised fine-tuning with cross-entropy loss.
    """
    episode = record.episode
    tool_calls: list[dict[str, Any]] = []

    for event in episode.events:
        if event.event_type == EventType.TOOL_CALL:
            payload = event.payload
            if isinstance(payload, ToolCallPayload):
                tool_calls.append({
                    "tool_call": {
                        "name": payload.tool_name,
                        "arguments": payload.arguments,
                    },
                    "thought": payload.thought,
                })

    return SFTRecord(
        task_prompt=episode.task_prompt,
        tool_call_sequence=tool_calls,
        final_answer=episode.final_answer,
        task_id=episode.task_id,
    )


def extract_sft_dataset(dataset: TrainingDataset) -> list[SFTRecord]:
    """Extract SFT records from all episodes in a training dataset."""
    return [extract_sft_record(r) for r in dataset.records]


# -- Internal helpers ---------------------------------------------------------

def _maybe_truncate(episode: Episode, max_tool_calls: int) -> Episode:
    """Return the episode unchanged if within bounds, or a shallow copy
    with events truncated to max_tool_calls valid tool calls.

    This does not deep-copy — it reuses the same Event objects.
    """
    tool_call_count = episode.metrics.valid_tool_calls
    if tool_call_count <= max_tool_calls:
        return episode

    # Find the event index of the Nth tool call
    seen = 0
    cutoff_index = len(episode.events)
    for i, event in enumerate(episode.events):
        if event.event_type == EventType.TOOL_CALL:
            seen += 1
            if seen > max_tool_calls:
                cutoff_index = i
                break

    # Shallow copy with truncated events
    truncated = Episode(
        task_id=episode.task_id,
        task_prompt=episode.task_prompt,
        model_id=episode.model_id,
        env_state_init=episode.env_state_init,
        events=episode.events[:cutoff_index],
        terminal=None,  # truncated episodes lose their terminal
        metrics=EpisodeMetrics(
            total_steps=cutoff_index,
            valid_tool_calls=max_tool_calls,
            invalid_tool_calls=episode.metrics.invalid_tool_calls,
            repair_attempts=episode.metrics.repair_attempts,
            repair_successes=episode.metrics.repair_successes,
            rejects=episode.metrics.rejects,
            model_calls=episode.metrics.model_calls,
            wall_time_seconds=episode.metrics.wall_time_seconds,
            total_reward=episode.metrics.total_reward,
        ),
        metadata={**episode.metadata, "truncated_at": max_tool_calls},
    )
    return truncated
