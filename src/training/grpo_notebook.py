"""
Notebook-facing GRPO orchestration helper.

Wraps the non-notebook logic for a bounded GRPO training run:
    - NeMo Gym environment-backed rollout collection
    - Dataset assembly and stage filtering
    - GRPO trajectory group building with group-relative advantages
    - art backend training lifecycle (or dry-run mock)
    - ATIF trace export for NAT inspection
    - Compact result object for notebook consumption

This module keeps the notebook readable and preserves the repo's
"notebook as consumer" rule.

Owns:
    - Rollout collection via NeMo Gym execution path
    - Training orchestration lifecycle
    - Artifact directory layout and path management
    - Compact result types for notebook display

Does NOT own:
    - Execution pipeline (see rollouts.nemo_gym_rollouts)
    - Environment state or transitions (see envs/)
    - GRPO grouping (see training.openpipe_art_adapter)
    - Reward shaping (see training.reward_views)
    - ATIF conversion (see runtime.atif_adapter)
    - Backend training logic (openpipe-art owns that)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import art

from src.rollouts.nemo_gym_rollouts import collect_nemo_gym_rollouts
from src.rollouts.episode_runner import EnrichedEpisodeResult
from src.rollouts.export_adapters import (
    episode_to_atif_trajectory,
    save_atif_trajectories_jsonl,
)
from src.training.curriculum import TrainingStage, get_stage_config, StageConfig
from src.training.datasets import TrainingRecord, build_training_dataset
from src.training.reward_views import (
    build_episode_reward_view,
    get_per_step_rewards,
    EpisodeRewardView,
)
from src.training.openpipe_art_adapter import (
    build_grpo_trajectory_group,
    save_art_group_jsonl,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class GRPORunResult:
    """Compact result from a notebook GRPO training run.

    Attributes:
        enriched_results: All enriched episodes collected during rollouts.
        trajectory_group: The art.TrajectoryGroup built for GRPO.
        reward_views: Per-episode reward views (stage-shaped).
        stage_config: The curriculum stage used.
        train_metrics: Metrics from the training step (or dry-run mock).
        checkpoint_path: Path to the saved checkpoint (None if dry-run).
        artifact_dir: Root directory for all produced artifacts.
        atif_path: Path to the ATIF JSONL export.
        group_jsonl_path: Path to the art.TrajectoryGroup JSONL export.
        dry_run: Whether this was a dry-run (no actual training).
        wall_time_seconds: Total wall time for the run.
    """
    enriched_results: list[EnrichedEpisodeResult]
    trajectory_group: art.TrajectoryGroup
    reward_views: list[EpisodeRewardView]
    stage_config: StageConfig
    train_metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    artifact_dir: str = ""
    atif_path: str = ""
    group_jsonl_path: str = ""
    dry_run: bool = True
    wall_time_seconds: float = 0.0

    def print_summary(self) -> None:
        """Print a compact summary suitable for notebook output."""
        n = len(self.enriched_results)
        n_traj = len(self.trajectory_group.trajectories)
        mode = "dry-run" if self.dry_run else "live"

        print(f"GRPO Run Summary ({mode})")
        print("=" * 60)
        print(f"  Episodes collected:     {n}")
        print(f"  Trajectories in group:  {n_traj}")
        print(f"  Stage:                  {self.stage_config.stage.value}")
        print(f"  Step/traj blend:        {self.stage_config.step_reward_weight}"
              f" / {self.stage_config.trajectory_reward_weight}")
        print()

        # Reward summary
        rewards = [t.reward for t in self.trajectory_group.trajectories]
        if rewards:
            advantages = [
                t.metadata.get("group_advantage", 0.0)
                for t in self.trajectory_group.trajectories
            ]
            print(f"  Reward range:           [{min(rewards):+.4f}, {max(rewards):+.4f}]")
            print(f"  Reward mean:            {sum(rewards)/len(rewards):+.4f}")
            print(f"  Advantage range:        [{min(advantages):+.4f}, {max(advantages):+.4f}]")
        print()

        # Training metrics
        if self.train_metrics:
            print("  Training metrics:")
            for k, v in self.train_metrics.items():
                print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")
            print()

        # Artifacts
        print("  Artifacts:")
        print(f"    ATIF export:          {self.atif_path}")
        print(f"    TrajectoryGroup:      {self.group_jsonl_path}")
        if self.checkpoint_path:
            print(f"    Checkpoint:           {self.checkpoint_path}")
        print(f"  Wall time:              {self.wall_time_seconds:.1f}s")


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_enriched_rollouts(
    num_rollouts: int = 4,
    include_repairs: bool = True,
) -> list[EnrichedEpisodeResult]:
    """Collect enriched episodes via NeMo Gym environment-backed execution.

    Runs scripted action sequences through the NeMo Gym execution
    pipeline (validate → repair → reject → execute → record) with the
    environment computing rewards in real-time. This exercises the
    documented NVIDIA ownership split: NeMo Gym owns training-time
    execution and rollout collection rather than training on canned
    trajectories.

    Args:
        num_rollouts: Total number of episodes to collect.
        include_repairs: Whether to include repair episodes in the mix.

    Returns:
        List of EnrichedEpisodeResult from environment-backed execution.
    """
    return collect_nemo_gym_rollouts(
        num_rollouts=num_rollouts,
        include_repairs=include_repairs,
    )


# ---------------------------------------------------------------------------
# GRPO group assembly
# ---------------------------------------------------------------------------

def build_grpo_group_from_rollouts(
    enriched_results: list[EnrichedEpisodeResult],
    stage: TrainingStage = TrainingStage.FULL_MULTITURN_RL,
) -> tuple[art.TrajectoryGroup, StageConfig, list[EpisodeRewardView]]:
    """Build a GRPO trajectory group from enriched rollout results.

    Converts enriched episodes into TrainingRecords, applies stage-aware
    reward shaping, and builds a GRPO trajectory group with group-relative
    advantages.

    Args:
        enriched_results: Enriched episodes from rollout collection.
        stage: Which curriculum stage to use for reward shaping.

    Returns:
        Tuple of (TrajectoryGroup, StageConfig, list of EpisodeRewardViews).
    """
    stage_config = get_stage_config(stage)

    # Build TrainingRecords from enriched results
    records: list[TrainingRecord] = []
    reward_views: list[EpisodeRewardView] = []

    for result in enriched_results:
        view = build_episode_reward_view(result.reward_summary, stage_config)
        reward_views.append(view)

        record = TrainingRecord(
            episode=result.episode,
            reward_summary=result.reward_summary,
            stage=stage,
            metadata={"source": "nemo_gym_rollout"},
        )
        records.append(record)

    # Build GRPO trajectory group with group-relative advantages
    group = build_grpo_trajectory_group(
        records,
        stage_config,
        group_size=len(records),
    )

    return group, stage_config, reward_views


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

def export_artifacts(
    enriched_results: list[EnrichedEpisodeResult],
    trajectory_group: art.TrajectoryGroup,
    artifact_dir: str = "artifacts/grpo_run",
) -> tuple[str, str]:
    """Export ATIF traces and trajectory group to the artifact directory.

    Args:
        enriched_results: Enriched episodes for ATIF export.
        trajectory_group: The GRPO trajectory group to export.
        artifact_dir: Root directory for artifacts.

    Returns:
        Tuple of (atif_path, group_jsonl_path).
    """
    os.makedirs(artifact_dir, exist_ok=True)

    # Export ATIF traces for NAT inspection
    atif_trajectories = [
        episode_to_atif_trajectory(r.episode)
        for r in enriched_results
    ]
    atif_path = os.path.join(artifact_dir, "atif_trajectories.jsonl")
    save_atif_trajectories_jsonl(atif_trajectories, atif_path)

    # Export art.TrajectoryGroup for training consumption
    group_path = os.path.join(artifact_dir, "grpo_trajectory_group.jsonl")
    save_art_group_jsonl(trajectory_group, group_path)

    return atif_path, group_path


# ---------------------------------------------------------------------------
# Training step (real or dry-run)
# ---------------------------------------------------------------------------

def _dry_run_train(
    trajectory_group: art.TrajectoryGroup,
    stage_config: StageConfig,
) -> dict[str, float]:
    """Simulate a training step without a real backend.

    Computes summary statistics from the trajectory group that would
    normally come from the training backend. Used when no GPU backend
    is available.

    Args:
        trajectory_group: The GRPO trajectory group.
        stage_config: Stage config for hyperparameter reference.

    Returns:
        Mock training metrics dict.
    """
    rewards = [t.reward for t in trajectory_group.trajectories]
    advantages = [
        t.metadata.get("group_advantage", 0.0)
        for t in trajectory_group.trajectories
    ]

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    reward_std = (
        (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
        if rewards else 0.0
    )

    return {
        "step": 1,
        "loss": round(0.15 + 0.1 * (1.0 - mean_reward), 6),
        "mean_reward": round(mean_reward, 4),
        "reward_std": round(reward_std, 4),
        "mean_advantage": round(sum(advantages) / len(advantages) if advantages else 0.0, 4),
        "num_trajectories": len(trajectory_group.trajectories),
        "learning_rate": stage_config.reward_config.get(
            "learning_rate", 5e-7,
        ),
        "kl_penalty_coef": 0.02,
    }


# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------

def run_grpo_notebook(
    num_rollouts: int = 4,
    include_repairs: bool = True,
    stage: TrainingStage = TrainingStage.FULL_MULTITURN_RL,
    artifact_dir: str = "artifacts/grpo_run",
    dry_run: bool = True,
) -> GRPORunResult:
    """Run a complete GRPO training cycle for the notebook.

    This is the main entry point for notebook cells. It orchestrates:
        1. Collect enriched rollouts via NeMo Gym environment-backed execution
        2. Build GRPO trajectory group with group-relative advantages
        3. Export ATIF traces and trajectory group artifacts
        4. Execute training step (or dry-run mock)
        5. Return compact result for notebook display

    Args:
        num_rollouts: Number of episodes to collect (default 4).
        include_repairs: Include repair episodes for diversity.
        stage: Curriculum stage for reward shaping.
        artifact_dir: Directory for output artifacts.
        dry_run: If True, simulate training without a real backend.

    Returns:
        GRPORunResult with all artifacts, metrics, and episode data.
    """
    start = time.monotonic()

    # Step 1: Collect rollouts
    enriched_results = collect_enriched_rollouts(
        num_rollouts=num_rollouts,
        include_repairs=include_repairs,
    )

    # Step 2: Build GRPO group
    trajectory_group, stage_config, reward_views = build_grpo_group_from_rollouts(
        enriched_results, stage=stage,
    )

    # Step 3: Export artifacts
    atif_path, group_path = export_artifacts(
        enriched_results, trajectory_group, artifact_dir=artifact_dir,
    )

    # Step 4: Training step
    train_metrics: dict[str, float] = {}
    checkpoint_path: str | None = None

    if dry_run:
        train_metrics = _dry_run_train(trajectory_group, stage_config)
    else:
        # Real training via art backend
        import asyncio

        model = art.TrainableModel(
            name="by-workshop-grpo",
            project="by-workshop",
            base_model="nvidia/nemotron-3-nano",
            base_path=os.path.join(artifact_dir, ".art"),
        )

        try:
            from art.local import LocalBackend
            backend = LocalBackend(
                path=os.path.join(artifact_dir, ".art"),
            )
        except ImportError:
            backend = art.ServerlessBackend()

        model.register(backend)

        result = asyncio.run(
            backend.train(
                model,
                [trajectory_group],
                learning_rate=5e-7,
                kl_penalty_coef=0.02,
            )
        )

        train_metrics = dict(result.metrics)
        train_metrics["step"] = result.step
        checkpoint_path = result.checkpoint_path

    wall_time = time.monotonic() - start

    return GRPORunResult(
        enriched_results=enriched_results,
        trajectory_group=trajectory_group,
        reward_views=reward_views,
        stage_config=stage_config,
        train_metrics=train_metrics,
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
        atif_path=atif_path,
        group_jsonl_path=group_path,
        dry_run=dry_run,
        wall_time_seconds=wall_time,
    )


# ---------------------------------------------------------------------------
# Reward/advantage data extraction for plotting
# ---------------------------------------------------------------------------

def extract_reward_plot_data(
    result: GRPORunResult,
) -> dict[str, Any]:
    """Extract structured data for reward visualization from a GRPO run result.

    Returns a dict with keys suitable for direct consumption by
    reward_plots.plot_grpo_rewards().

    Args:
        result: A completed GRPORunResult.

    Returns:
        Dict with total_rewards, shaped_rewards, advantages,
        per_step_rewards, and episode_labels.
    """
    trajectories = result.trajectory_group.trajectories

    total_rewards = [t.reward for t in trajectories]
    advantages = [
        t.metadata.get("group_advantage", 0.0)
        for t in trajectories
    ]
    per_step_rewards = [
        json.loads(t.metadata.get("per_step_rewards", "[]"))
        for t in trajectories
    ]

    # Build labels from episode metadata
    labels = []
    for i, t in enumerate(trajectories):
        task_id = t.metadata.get("task_id", "?")
        repairs = t.metrics.get("repair_attempts", 0)
        label = f"ep{i} ({task_id})"
        if repairs > 0:
            label += f" [R={int(repairs)}]"
        labels.append(label)

    # Per-step reward views from the EpisodeRewardView objects
    shaped_step_data = []
    for view in result.reward_views:
        shaped_step_data.append({
            "step_rewards": [sv.shaped_reward for sv in view.step_views],
            "terminal_reward": (
                view.terminal_view.shaped_reward
                if view.terminal_view else None
            ),
            "combined": view.combined_reward,
            "trajectory_reward": view.trajectory_reward,
        })

    return {
        "total_rewards": total_rewards,
        "advantages": advantages,
        "per_step_rewards": per_step_rewards,
        "shaped_step_data": shaped_step_data,
        "episode_labels": labels,
        "stage": result.stage_config.stage.value,
        "step_weight": result.stage_config.step_reward_weight,
        "trajectory_weight": result.stage_config.trajectory_reward_weight,
    }
