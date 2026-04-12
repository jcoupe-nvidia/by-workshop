"""Tests for the GRPO notebook orchestration helpers.

Covers:
    - Rollout collection: correct episode counts and enrichment
    - GRPO group assembly: datum group structure, advantages, stage shaping
    - Artifact export: ATIF + datum group JSONL files exist and parse
    - Plot data extraction: correct keys and shape
    - Dry-run training: mock metrics structure
    - End-to-end run_grpo_notebook: full pipeline smoke test
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from src.training.grpo_notebook import (
    collect_enriched_rollouts,
    build_grpo_group_from_rollouts,
    export_artifacts,
    extract_reward_plot_data,
    run_grpo_notebook,
    GRPORunResult,
    _dry_run_train,
)
from src.training.nemo_rl_adapter import get_group_metadata
from src.training.curriculum import TrainingStage, get_stage_config
from src.training.datasets import build_training_dataset, _truncate_reward_summary
from src.rollouts.episode_runner import EnrichedEpisodeResult
from src.rollouts.trace_types import EventType


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

class TestCollectEnrichedRollouts:
    """Verify rollout collection produces correct episode counts and structure."""

    def test_collects_requested_count(self):
        results = collect_enriched_rollouts(num_rollouts=4)
        assert len(results) == 4

    def test_all_results_are_enriched(self):
        results = collect_enriched_rollouts(num_rollouts=2)
        for r in results:
            assert isinstance(r, EnrichedEpisodeResult)
            assert r.episode.task_id == "SO-10482"
            assert r.reward_summary is not None
            assert r.reward_summary.total_reward != 0.0

    def test_includes_repair_episodes_when_enabled(self):
        results = collect_enriched_rollouts(num_rollouts=4, include_repairs=True)
        repair_count = sum(
            1 for r in results
            if r.episode.metrics.repair_attempts > 0
        )
        assert repair_count >= 1, "Expected at least one repair episode"

    def test_no_repairs_when_disabled(self):
        results = collect_enriched_rollouts(num_rollouts=4, include_repairs=False)
        for r in results:
            assert r.episode.metrics.repair_attempts == 0

    def test_single_rollout(self):
        results = collect_enriched_rollouts(num_rollouts=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# GRPO group assembly
# ---------------------------------------------------------------------------

class TestBuildGrpoGroupFromRollouts:
    """Verify GRPO datum group structure and advantage computation."""

    @pytest.fixture
    def rollouts(self):
        return collect_enriched_rollouts(num_rollouts=4, include_repairs=True)

    def test_group_has_correct_datum_count(self, rollouts):
        datum_specs, _, _ = build_grpo_group_from_rollouts(rollouts)
        assert len(datum_specs) == len(rollouts)

    def test_group_has_grpo_metadata(self, rollouts):
        datum_specs, _, _ = build_grpo_group_from_rollouts(rollouts)
        metadata = get_group_metadata(datum_specs)
        assert metadata["method"] == "grpo"
        assert metadata["stage"] == "full_multiturn_rl"

    def test_datums_have_advantage_metadata(self, rollouts):
        datum_specs, _, _ = build_grpo_group_from_rollouts(rollouts)
        for d in datum_specs:
            info = d["extra_env_info"]
            assert "group_advantage" in info
            assert "group_mean_reward" in info
            assert "group_task_id" in info

    def test_advantages_sum_to_zero(self, rollouts):
        datum_specs, _, _ = build_grpo_group_from_rollouts(rollouts)
        advantages = [
            d["extra_env_info"]["group_advantage"]
            for d in datum_specs
        ]
        assert abs(sum(advantages)) < 1e-3, (
            f"Advantages should sum to ~0, got {sum(advantages)}"
        )

    def test_reward_views_match_datum_count(self, rollouts):
        _, _, views = build_grpo_group_from_rollouts(rollouts)
        assert len(views) == len(rollouts)

    def test_stage_config_returned(self, rollouts):
        _, stage_config, _ = build_grpo_group_from_rollouts(rollouts)
        assert stage_config.stage == TrainingStage.FULL_MULTITURN_RL

    def test_custom_stage(self, rollouts):
        datum_specs, stage_config, _ = build_grpo_group_from_rollouts(
            rollouts, stage=TrainingStage.SHORT_HORIZON_RL,
        )
        assert stage_config.stage == TrainingStage.SHORT_HORIZON_RL
        assert get_group_metadata(datum_specs)["stage"] == "short_horizon_rl"

    def test_datums_have_positive_rewards(self, rollouts):
        datum_specs, _, _ = build_grpo_group_from_rollouts(rollouts)
        for d in datum_specs:
            reward = d["extra_env_info"]["reward"]
            assert reward > 0, f"Expected positive reward, got {reward}"

    def test_datums_have_valid_datum_spec_keys(self, rollouts):
        datum_specs, _, _ = build_grpo_group_from_rollouts(rollouts)
        for d in datum_specs:
            assert "message_log" in d
            assert "length" in d
            assert "extra_env_info" in d
            assert "loss_multiplier" in d
            assert "idx" in d
            assert "task_name" in d


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

class TestExportArtifacts:
    """Verify ATIF and datum group artifacts are written correctly."""

    @pytest.fixture
    def grpo_data(self):
        rollouts = collect_enriched_rollouts(num_rollouts=2)
        datum_specs, _, _ = build_grpo_group_from_rollouts(rollouts)
        return rollouts, datum_specs

    def test_atif_file_created(self, grpo_data):
        rollouts, datum_specs = grpo_data
        with tempfile.TemporaryDirectory() as tmpdir:
            atif_path, _ = export_artifacts(rollouts, datum_specs, tmpdir)
            assert os.path.exists(atif_path)
            with open(atif_path) as f:
                lines = f.readlines()
            assert len(lines) == len(rollouts)

    def test_atif_lines_are_valid_json(self, grpo_data):
        rollouts, datum_specs = grpo_data
        with tempfile.TemporaryDirectory() as tmpdir:
            atif_path, _ = export_artifacts(rollouts, datum_specs, tmpdir)
            with open(atif_path) as f:
                for line in f:
                    parsed = json.loads(line)
                    assert "agent" in parsed
                    assert "steps" in parsed

    def test_group_jsonl_created(self, grpo_data):
        rollouts, datum_specs = grpo_data
        with tempfile.TemporaryDirectory() as tmpdir:
            _, group_path = export_artifacts(rollouts, datum_specs, tmpdir)
            assert os.path.exists(group_path)
            with open(group_path) as f:
                data = json.loads(f.readline())
            assert "datum_specs" in data

    def test_artifact_dir_created(self, grpo_data):
        rollouts, datum_specs = grpo_data
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "nested", "dir")
            export_artifacts(rollouts, datum_specs, subdir)
            assert os.path.isdir(subdir)


# ---------------------------------------------------------------------------
# Plot data extraction
# ---------------------------------------------------------------------------

class TestExtractRewardPlotData:
    """Verify plot data structure from a GRPORunResult."""

    @pytest.fixture
    def result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            return run_grpo_notebook(
                num_rollouts=4,
                dry_run=True,
                artifact_dir=tmpdir,
            )

    def test_required_keys(self, result):
        data = extract_reward_plot_data(result)
        expected_keys = {
            "total_rewards", "advantages", "per_step_rewards",
            "shaped_step_data", "episode_labels", "stage",
            "step_weight", "trajectory_weight",
        }
        assert expected_keys <= set(data.keys())

    def test_list_lengths_match(self, result):
        data = extract_reward_plot_data(result)
        n = len(result.enriched_results)
        assert len(data["total_rewards"]) == n
        assert len(data["advantages"]) == n
        assert len(data["episode_labels"]) == n
        assert len(data["shaped_step_data"]) == n

    def test_per_step_rewards_are_lists(self, result):
        data = extract_reward_plot_data(result)
        for psr in data["per_step_rewards"]:
            assert isinstance(psr, list)
            assert all(isinstance(v, (int, float)) for v in psr)

    def test_shaped_step_data_has_required_keys(self, result):
        data = extract_reward_plot_data(result)
        for d in data["shaped_step_data"]:
            assert "step_rewards" in d
            assert "terminal_reward" in d
            assert "combined" in d
            assert "trajectory_reward" in d

    def test_labels_contain_repair_markers(self, result):
        data = extract_reward_plot_data(result)
        repair_labels = [l for l in data["episode_labels"] if "[R=" in l]
        assert len(repair_labels) >= 1


# ---------------------------------------------------------------------------
# Dry-run training
# ---------------------------------------------------------------------------

class TestDryRunTrain:
    """Verify dry-run training produces valid mock metrics."""

    def test_returns_expected_keys(self):
        rollouts = collect_enriched_rollouts(num_rollouts=4)
        datum_specs, stage_config, _ = build_grpo_group_from_rollouts(rollouts)
        metrics = _dry_run_train(datum_specs, stage_config)

        assert "step" in metrics
        assert "loss" in metrics
        assert "mean_reward" in metrics
        assert "reward_std" in metrics
        assert "mean_advantage" in metrics
        assert "num_datum_specs" in metrics

    def test_num_datums_matches(self):
        rollouts = collect_enriched_rollouts(num_rollouts=4)
        datum_specs, stage_config, _ = build_grpo_group_from_rollouts(rollouts)
        metrics = _dry_run_train(datum_specs, stage_config)
        assert metrics["num_datum_specs"] == 4


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

class TestRunGrpoNotebook:
    """End-to-end smoke test for run_grpo_notebook."""

    def test_dry_run_completes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_grpo_notebook(
                num_rollouts=4,
                dry_run=True,
                artifact_dir=tmpdir,
            )
            assert isinstance(result, GRPORunResult)
            assert result.dry_run is True
            assert len(result.enriched_results) == 4
            assert len(result.datum_specs) == 4
            assert result.wall_time_seconds > 0
            assert os.path.exists(result.atif_path)
            assert os.path.exists(result.group_jsonl_path)

    def test_train_metrics_populated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_grpo_notebook(
                num_rollouts=2,
                dry_run=True,
                artifact_dir=tmpdir,
            )
            assert result.train_metrics["step"] == 1
            assert result.train_metrics["num_datum_specs"] == 2

    def test_print_summary_runs(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_grpo_notebook(
                num_rollouts=2,
                dry_run=True,
                artifact_dir=tmpdir,
            )
            result.print_summary()
            captured = capsys.readouterr()
            assert "GRPO Run Summary" in captured.out
            assert "dry-run" in captured.out


# ---------------------------------------------------------------------------
# Fix validation: truncated reward summary
# ---------------------------------------------------------------------------

class TestTruncatedRewardSummaryConsistency:
    """When build_training_dataset truncates an episode, the TrainingRecord's
    reward_summary must match the truncated events, not the full episode."""

    def test_truncated_record_has_fewer_step_rewards(self):
        """A truncated TrainingRecord should carry only the reward signals
        for the retained tool calls."""
        rollouts = collect_enriched_rollouts(num_rollouts=1)
        stage_config = get_stage_config(TrainingStage.SHORT_HORIZON_RL)

        dataset = build_training_dataset(rollouts, stage_config)

        for record in dataset.records:
            n_valid = record.episode.metrics.valid_tool_calls
            n_step_rewards = len(record.reward_summary.step_rewards)
            assert n_step_rewards == n_valid, (
                f"TrainingRecord has {n_step_rewards} step rewards but "
                f"{n_valid} valid tool calls after truncation"
            )

    def test_truncated_record_has_no_terminal_reward(self):
        """Truncated episodes lose their terminal, so the reward summary
        should have terminal_reward=None."""
        rollouts = collect_enriched_rollouts(num_rollouts=1)
        stage_config = get_stage_config(TrainingStage.SHORT_HORIZON_RL)
        dataset = build_training_dataset(rollouts, stage_config)

        for record in dataset.records:
            if record.episode.terminal is None:
                assert record.reward_summary.terminal_reward is None

    def test_truncated_total_reward_matches_step_sum(self):
        """The truncated reward_summary.total_reward should match the sum
        of its step_rewards totals."""
        rollouts = collect_enriched_rollouts(num_rollouts=1)
        stage_config = get_stage_config(TrainingStage.SHORT_HORIZON_RL)
        dataset = build_training_dataset(rollouts, stage_config)

        for record in dataset.records:
            expected = sum(sr.total for sr in record.reward_summary.step_rewards)
            if record.reward_summary.terminal_reward:
                expected += record.reward_summary.terminal_reward.total
            assert abs(record.reward_summary.total_reward - round(expected, 4)) < 1e-4


# ---------------------------------------------------------------------------
# Fix validation: per-event rewards in NeMo Gym rollouts
# ---------------------------------------------------------------------------

class TestNemoGymRolloutEventRewards:
    """Environment-backed rollouts should have per-event reward annotations,
    matching the same contract as enrich_episode()."""

    def test_tool_call_events_have_rewards(self):
        """Every TOOL_CALL event in an environment-backed rollout should
        have a non-None reward."""
        rollouts = collect_enriched_rollouts(num_rollouts=1)
        episode = rollouts[0].episode

        tool_call_events = [
            e for e in episode.events
            if e.event_type == EventType.TOOL_CALL
        ]
        assert len(tool_call_events) > 0
        for e in tool_call_events:
            assert e.reward is not None, (
                f"TOOL_CALL event at step {e.step_index} has reward=None"
            )

    def test_terminal_event_has_reward(self):
        """The TERMINAL_OUTCOME event should have a non-None reward."""
        rollouts = collect_enriched_rollouts(num_rollouts=1)
        episode = rollouts[0].episode

        terminal_events = [
            e for e in episode.events
            if e.event_type == EventType.TERMINAL_OUTCOME
        ]
        assert len(terminal_events) == 1
        assert terminal_events[0].reward is not None

    def test_event_rewards_sum_to_total(self):
        """The sum of all event rewards should approximately equal
        the episode total reward."""
        rollouts = collect_enriched_rollouts(num_rollouts=1)
        episode = rollouts[0].episode

        event_reward_sum = sum(
            e.reward for e in episode.events if e.reward is not None
        )
        assert abs(event_reward_sum - episode.metrics.total_reward) < 0.1
