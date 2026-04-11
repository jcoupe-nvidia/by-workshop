"""Tests for src.rollouts — scripted traces, enrichment, serialization, export."""
from __future__ import annotations

import json
import tempfile
import os

import pytest

from src.rollouts.trace_types import (
    Episode,
    Event,
    EventType,
    EpisodeMetrics,
    ToolCallPayload,
    ToolResultPayload,
    TerminalOutcomePayload,
)
from src.rollouts.scripted_traces import build_successful_episode, build_repair_episode
from src.rollouts.episode_runner import enrich_episode, EnrichedEpisodeResult
from src.rollouts.serializers import (
    episode_to_dict,
    dict_to_episode,
    episode_to_jsonl,
    jsonl_to_episode,
    save_episodes_jsonl,
    load_episodes_jsonl,
)
from src.rollouts.export_adapters import (
    episode_to_training_trajectory,
    training_trajectory_to_jsonl,
    save_training_trajectories_jsonl,
    TrainingTrajectory,
    TrainingTrajectoryStep,
)


class TestScriptedTraces:
    def test_successful_episode_structure(self):
        ep = build_successful_episode()
        assert ep.task_id == "SO-10482"
        assert ep.model_id == "scripted"
        assert ep.metrics.valid_tool_calls == 9
        assert ep.metrics.invalid_tool_calls == 0
        assert ep.is_complete is True
        assert ep.final_answer is not None

    def test_successful_episode_tool_sequence(self):
        ep = build_successful_episode()
        names = ep.tool_names_called
        assert names[0] == "get_order"
        assert names[-1] == "recommend_action"
        assert len(names) == 9

    def test_repair_episode_structure(self):
        ep = build_repair_episode()
        assert ep.task_id == "SO-10482"
        assert ep.metrics.valid_tool_calls == 9
        assert ep.metrics.invalid_tool_calls == 1
        assert ep.metrics.repair_attempts == 2
        assert ep.metrics.rejects == 1
        assert ep.is_complete is True

    def test_repair_episode_has_error_events(self):
        ep = build_repair_episode()
        error_events = [e for e in ep.events
                        if e.event_type == EventType.TOOL_VALIDATION_ERROR]
        reject_events = [e for e in ep.events
                         if e.event_type == EventType.TOOL_REJECT]
        repair_events = [e for e in ep.events
                         if e.event_type == EventType.TOOL_REPAIR_ATTEMPT]
        assert len(error_events) == 1
        assert len(reject_events) == 1
        assert len(repair_events) == 2


class TestEnrichEpisode:
    def test_enriched_has_rewards(self, enriched_success):
        assert enriched_success.reward_summary.total_reward > 0
        assert len(enriched_success.reward_summary.step_rewards) == 9

    def test_events_have_reward_annotations(self, enriched_success):
        tool_call_events = [e for e in enriched_success.episode.events
                            if e.event_type == EventType.TOOL_CALL]
        rewarded = [e for e in tool_call_events if e.reward is not None]
        assert len(rewarded) == 9

    def test_repair_enriched(self, enriched_repair):
        assert enriched_repair.reward_summary.total_reward > 0

    def test_env_final_state(self, enriched_success):
        assert "order_id" in enriched_success.env_final_state
        assert enriched_success.env_final_state["order_id"] == "SO-10482"

    def test_enriched_result_type(self, enriched_success):
        assert isinstance(enriched_success, EnrichedEpisodeResult)
        assert isinstance(enriched_success.episode, Episode)


class TestSerializers:
    def test_episode_roundtrip_dict(self, successful_episode):
        d = episode_to_dict(successful_episode)
        restored = dict_to_episode(d)
        assert restored.task_id == successful_episode.task_id
        assert len(restored.events) == len(successful_episode.events)
        assert restored.metrics.valid_tool_calls == successful_episode.metrics.valid_tool_calls

    def test_episode_roundtrip_jsonl(self, successful_episode):
        line = episode_to_jsonl(successful_episode)
        restored = jsonl_to_episode(line)
        assert restored.task_id == successful_episode.task_id
        assert restored.tool_names_called == successful_episode.tool_names_called

    def test_save_load_jsonl(self, successful_episode, repair_episode):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            save_episodes_jsonl([successful_episode, repair_episode], path)
            loaded = load_episodes_jsonl(path)
            assert len(loaded) == 2
            assert loaded[0].task_id == "SO-10482"
            assert loaded[1].metrics.repair_attempts == 2
        finally:
            os.unlink(path)


class TestExportAdapters:
    def test_training_trajectory_structure(self, enriched_success):
        traj = episode_to_training_trajectory(
            enriched_success.episode,
            reward_summary=enriched_success.reward_summary,
        )
        assert isinstance(traj, TrainingTrajectory)
        assert traj.task_id == "SO-10482"
        assert traj.episode_length > 0
        assert len(traj.steps) == traj.episode_length
        assert all(isinstance(s, TrainingTrajectoryStep) for s in traj.steps)

    def test_last_step_is_done(self, enriched_success):
        traj = episode_to_training_trajectory(
            enriched_success.episode,
            reward_summary=enriched_success.reward_summary,
        )
        assert traj.steps[-1].done is True

    def test_jsonl_serialization(self, enriched_success):
        traj = episode_to_training_trajectory(
            enriched_success.episode,
            reward_summary=enriched_success.reward_summary,
        )
        line = training_trajectory_to_jsonl(traj)
        parsed = json.loads(line)
        assert parsed["task_id"] == "SO-10482"
        assert len(parsed["steps"]) == traj.episode_length

    def test_save_training_trajectories(self, enriched_success, enriched_repair):
        traj_s = episode_to_training_trajectory(
            enriched_success.episode,
            reward_summary=enriched_success.reward_summary,
        )
        traj_r = episode_to_training_trajectory(
            enriched_repair.episode,
            reward_summary=enriched_repair.reward_summary,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            save_training_trajectories_jsonl([traj_s, traj_r], path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
        finally:
            os.unlink(path)
