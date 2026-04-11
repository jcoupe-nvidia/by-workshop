"""Tests for the GRPO notebook orchestration helpers.

Covers:
    - Rollout collection: correct episode counts and enrichment
    - GRPO group assembly: trajectory group structure, advantages, stage shaping
    - Artifact export: ATIF + trajectory group JSONL files exist and parse
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
from src.training.curriculum import TrainingStage, get_stage_config
from src.rollouts.episode_runner import EnrichedEpisodeResult


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
    """Verify GRPO trajectory group structure and advantage computation."""

    @pytest.fixture
    def rollouts(self):
        return collect_enriched_rollouts(num_rollouts=4, include_repairs=True)

    def test_group_has_correct_trajectory_count(self, rollouts):
        group, _, _ = build_grpo_group_from_rollouts(rollouts)
        assert len(group.trajectories) == len(rollouts)

    def test_group_has_grpo_metadata(self, rollouts):
        group, _, _ = build_grpo_group_from_rollouts(rollouts)
        assert group.metadata["method"] == "grpo"
        assert group.metadata["stage"] == "full_multiturn_rl"

    def test_trajectories_have_advantage_metadata(self, rollouts):
        group, _, _ = build_grpo_group_from_rollouts(rollouts)
        for t in group.trajectories:
            assert "group_advantage" in t.metadata
            assert "group_mean_reward" in t.metadata
            assert "group_task_id" in t.metadata

    def test_advantages_sum_to_zero(self, rollouts):
        group, _, _ = build_grpo_group_from_rollouts(rollouts)
        advantages = [
            t.metadata["group_advantage"]
            for t in group.trajectories
        ]
        assert abs(sum(advantages)) < 1e-3, (
            f"Advantages should sum to ~0, got {sum(advantages)}"
        )

    def test_reward_views_match_trajectory_count(self, rollouts):
        _, _, views = build_grpo_group_from_rollouts(rollouts)
        assert len(views) == len(rollouts)

    def test_stage_config_returned(self, rollouts):
        _, stage_config, _ = build_grpo_group_from_rollouts(rollouts)
        assert stage_config.stage == TrainingStage.FULL_MULTITURN_RL

    def test_custom_stage(self, rollouts):
        group, stage_config, _ = build_grpo_group_from_rollouts(
            rollouts, stage=TrainingStage.SHORT_HORIZON_RL,
        )
        assert stage_config.stage == TrainingStage.SHORT_HORIZON_RL
        assert group.metadata["stage"] == "short_horizon_rl"

    def test_trajectories_have_positive_rewards(self, rollouts):
        group, _, _ = build_grpo_group_from_rollouts(rollouts)
        for t in group.trajectories:
            assert t.reward > 0, f"Expected positive reward, got {t.reward}"


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

class TestExportArtifacts:
    """Verify ATIF and trajectory group artifacts are written correctly."""

    @pytest.fixture
    def grpo_data(self):
        rollouts = collect_enriched_rollouts(num_rollouts=2)
        group, _, _ = build_grpo_group_from_rollouts(rollouts)
        return rollouts, group

    def test_atif_file_created(self, grpo_data):
        rollouts, group = grpo_data
        with tempfile.TemporaryDirectory() as tmpdir:
            atif_path, _ = export_artifacts(rollouts, group, tmpdir)
            assert os.path.exists(atif_path)
            with open(atif_path) as f:
                lines = f.readlines()
            assert len(lines) == len(rollouts)

    def test_atif_lines_are_valid_json(self, grpo_data):
        rollouts, group = grpo_data
        with tempfile.TemporaryDirectory() as tmpdir:
            atif_path, _ = export_artifacts(rollouts, group, tmpdir)
            with open(atif_path) as f:
                for line in f:
                    parsed = json.loads(line)
                    assert "agent" in parsed
                    assert "steps" in parsed

    def test_group_jsonl_created(self, grpo_data):
        rollouts, group = grpo_data
        with tempfile.TemporaryDirectory() as tmpdir:
            _, group_path = export_artifacts(rollouts, group, tmpdir)
            assert os.path.exists(group_path)
            with open(group_path) as f:
                data = json.loads(f.readline())
            assert "trajectories" in data

    def test_artifact_dir_created(self, grpo_data):
        rollouts, group = grpo_data
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "nested", "dir")
            export_artifacts(rollouts, group, subdir)
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
        group, stage_config, _ = build_grpo_group_from_rollouts(rollouts)
        metrics = _dry_run_train(group, stage_config)

        assert "step" in metrics
        assert "loss" in metrics
        assert "mean_reward" in metrics
        assert "reward_std" in metrics
        assert "mean_advantage" in metrics
        assert "num_trajectories" in metrics

    def test_num_trajectories_matches(self):
        rollouts = collect_enriched_rollouts(num_rollouts=4)
        group, stage_config, _ = build_grpo_group_from_rollouts(rollouts)
        metrics = _dry_run_train(group, stage_config)
        assert metrics["num_trajectories"] == 4


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
            assert len(result.trajectory_group.trajectories) == 4
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
            assert result.train_metrics["num_trajectories"] == 2

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
