"""
Notebook-facing GRPO orchestration helper.

Wraps the non-notebook logic for a bounded GRPO training run:
    - NeMo Gym environment-backed rollout collection
    - Dataset assembly and stage filtering
    - GRPO datum group building with group-relative advantages
    - NeMo RL training lifecycle (or dry-run mock)
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
    - GRPO grouping (see training.nemo_rl_adapter)
    - Reward shaping (see training.reward_views)
    - ATIF conversion (see runtime.atif_adapter)
    - Backend training logic (NeMo RL owns that)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from src.rollouts.nemo_gym_rollouts import (
    collect_server_backed_rollouts,
)
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
from src.training.nemo_rl_adapter import (
    build_grpo_datum_group,
    save_datum_group_jsonl,
    get_group_metadata,
    get_group_metrics,
)


# ---------------------------------------------------------------------------
# Curriculum experiment plan
# ---------------------------------------------------------------------------
# See training.experiments for the illustrative 4-stage curriculum plan
# (ExperimentPlan / build_default_experiment_plan). That plan documents
# the intended progression but is not yet wired into this notebook helper.

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class GRPORunResult:
    """Compact result from a notebook GRPO training run.

    Attributes:
        enriched_results: All enriched episodes collected during rollouts.
        datum_specs: The DatumSpec dicts built for GRPO.
        reward_views: Per-episode reward views (stage-shaped).
        stage_config: The curriculum stage used.
        train_metrics: Metrics from the training step (or dry-run mock).
        checkpoint_path: Path to the saved checkpoint (None if dry-run).
        artifact_dir: Root directory for all produced artifacts.
        atif_path: Path to the ATIF JSONL export.
        group_jsonl_path: Path to the DatumSpec group JSONL export.
        dry_run: Whether this was a dry-run (no actual training).
        wall_time_seconds: Total wall time for the run.
    """
    enriched_results: list[EnrichedEpisodeResult]
    datum_specs: list[dict[str, Any]]
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
        n_datums = len(self.datum_specs)
        mode = "dry-run" if self.dry_run else "live"

        print(f"GRPO Run Summary ({mode})")
        print("=" * 60)
        print(f"  Episodes collected:     {n}")
        print(f"  DatumSpecs in group:    {n_datums}")
        print(f"  Stage:                  {self.stage_config.stage.value}")
        print(f"  Step/traj blend:        {self.stage_config.step_reward_weight}"
              f" / {self.stage_config.trajectory_reward_weight}")
        print()

        rewards = [
            d.get("extra_env_info", {}).get("reward", 0.0)
            for d in self.datum_specs
        ]
        if rewards:
            advantages = [
                d.get("extra_env_info", {}).get("group_advantage", 0.0)
                for d in self.datum_specs
            ]
            print(f"  Reward range:           [{min(rewards):+.4f}, {max(rewards):+.4f}]")
            print(f"  Reward mean:            {sum(rewards)/len(rewards):+.4f}")
            print(f"  Advantage range:        [{min(advantages):+.4f}, {max(advantages):+.4f}]")
        print()

        if self.train_metrics:
            print("  Training metrics:")
            for k, v in self.train_metrics.items():
                print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")
            print()

        print("  Artifacts:")
        print(f"    ATIF export:          {self.atif_path}")
        print(f"    DatumSpec group:      {self.group_jsonl_path}")
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
    """Collect enriched episodes via the NeMo Gym resource server protocol.

    Exercises the documented NeMo Gym ownership split by calling
    seed_session() + verify() on LateOrderResourceServer for each
    episode. This is the same protocol that NeMo Gym's
    RolloutCollectionHelper uses during distributed training.

    NeMo Gym is a hard dependency — failures propagate immediately.

    Args:
        num_rollouts: Total number of episodes to collect.
        include_repairs: Whether to include repair/reject episodes in the mix.

    Returns:
        List of EnrichedEpisodeResult from NeMo Gym execution.
    """
    return collect_server_backed_rollouts(
        num_rollouts=num_rollouts,
        include_repairs=include_repairs,
    )


# ---------------------------------------------------------------------------
# GRPO group assembly
# ---------------------------------------------------------------------------

def build_grpo_group_from_rollouts(
    enriched_results: list[EnrichedEpisodeResult],
    stage: TrainingStage = TrainingStage.FULL_MULTISTEP_RL,
) -> tuple[list[dict[str, Any]], StageConfig, list[EpisodeRewardView]]:
    """Build a GRPO datum group from enriched rollout results.

    Converts enriched episodes into TrainingRecords, applies stage-aware
    reward shaping, and builds a GRPO datum group with group-relative
    advantages.

    Args:
        enriched_results: Enriched episodes from rollout collection.
        stage: Which curriculum stage to use for reward shaping.

    Returns:
        Tuple of (list of DatumSpec dicts, StageConfig, list of EpisodeRewardViews).
    """
    stage_config = get_stage_config(stage)

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

    datum_specs = build_grpo_datum_group(
        records,
        stage_config,
        group_size=len(records),
    )

    return datum_specs, stage_config, reward_views


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------

def export_artifacts(
    enriched_results: list[EnrichedEpisodeResult],
    datum_specs: list[dict[str, Any]],
    artifact_dir: str = "artifacts/grpo_run",
) -> tuple[str, str]:
    """Export ATIF traces and datum group to the artifact directory.

    Args:
        enriched_results: Enriched episodes for ATIF export.
        datum_specs: The GRPO datum group to export.
        artifact_dir: Root directory for artifacts.

    Returns:
        Tuple of (atif_path, group_jsonl_path).
    """
    os.makedirs(artifact_dir, exist_ok=True)

    atif_trajectories = [
        episode_to_atif_trajectory(r.episode)
        for r in enriched_results
    ]
    atif_path = os.path.join(artifact_dir, "atif_trajectories.jsonl")
    save_atif_trajectories_jsonl(atif_trajectories, atif_path)

    group_path = os.path.join(artifact_dir, "grpo_datum_group.jsonl")
    save_datum_group_jsonl(datum_specs, group_path)

    return atif_path, group_path


# ---------------------------------------------------------------------------
# Training step (real or dry-run)
# ---------------------------------------------------------------------------

def _dry_run_train(
    datum_specs: list[dict[str, Any]],
    stage_config: StageConfig,
) -> dict[str, float]:
    """Simulate a training step without a real backend.

    Computes summary statistics from the datum group that would
    normally come from the training backend. Used when no GPU backend
    is available.

    Args:
        datum_specs: The GRPO datum group.
        stage_config: Stage config for hyperparameter reference.

    Returns:
        Mock training metrics dict.
    """
    rewards = [
        d.get("extra_env_info", {}).get("reward", 0.0)
        for d in datum_specs
    ]
    advantages = [
        d.get("extra_env_info", {}).get("group_advantage", 0.0)
        for d in datum_specs
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
        "num_datum_specs": len(datum_specs),
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
    stage: TrainingStage = TrainingStage.FULL_MULTISTEP_RL,
    artifact_dir: str = "artifacts/grpo_run",
    dry_run: bool = True,
) -> GRPORunResult:
    """Run a complete GRPO training cycle for the notebook.

    This is the main entry point for notebook cells. It orchestrates:
        1. Collect enriched rollouts via NeMo Gym environment-backed execution
        2. Build GRPO datum group with group-relative advantages
        3. Export ATIF traces and datum group artifacts
        4. Execute training step (real via NeMo RL or dry-run)
        5. Return compact result for notebook display

    When dry_run=True (the default), simulates a training step using
    the collected rollouts. Set dry_run=False to attempt a real NeMo RL
    training step; if GPU resources are unavailable, falls back to
    dry-run mode automatically. For full training, use
    ``python -m src.training.run_grpo_training``.

    Args:
        num_rollouts: Number of episodes to collect (default 4).
        include_repairs: Include repair episodes for diversity.
        stage: Curriculum stage for reward shaping.
        artifact_dir: Directory for output artifacts.
        dry_run: If True, always simulate training. If False, attempt live
                 training with automatic dry-run fallback.

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
    datum_specs, stage_config, reward_views = build_grpo_group_from_rollouts(
        enriched_results, stage=stage,
    )

    # Step 3: Export artifacts
    atif_path, group_path = export_artifacts(
        enriched_results, datum_specs, artifact_dir=artifact_dir,
    )

    # Step 4: Training step
    train_metrics: dict[str, float] = {}
    checkpoint_path: str | None = None
    effective_dry_run = dry_run

    if not dry_run:
        import warnings
        warnings.warn(
            "dry_run=False runs fresh GRPO training via NeMo RL (generating new rollouts). "
            "The collected datum_specs are exported as artifacts but not consumed by "
            "the live training path. For offline training on collected episodes, use: "
            "python -m src.training.run_grpo_training",
            stacklevel=2,
        )
        try:
            live_metrics = _live_train(stage_config)
            train_metrics.update(live_metrics)
        except Exception as exc:
            effective_dry_run = True
            train_metrics["_fallback_reason_code"] = "nemo_rl_unavailable"
            train_metrics.update(_dry_run_train(datum_specs, stage_config))
            import warnings
            warnings.warn(
                f"NeMo RL training unavailable ({type(exc).__name__}: {exc}), "
                f"falling back to dry-run. For full training, use: "
                f"python -m src.training.run_grpo_training",
                stacklevel=2,
            )
    else:
        train_metrics.update(_dry_run_train(datum_specs, stage_config))

    wall_time = time.monotonic() - start

    return GRPORunResult(
        enriched_results=enriched_results,
        datum_specs=datum_specs,
        reward_views=reward_views,
        stage_config=stage_config,
        train_metrics=train_metrics,
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
        atif_path=atif_path,
        group_jsonl_path=group_path,
        dry_run=effective_dry_run,
        wall_time_seconds=wall_time,
    )


def _live_train(
    stage_config: StageConfig,
) -> dict[str, float]:
    """Run a real NeMo RL GRPO training step (fresh rollouts, NOT replay).

    WARNING: The ``datum_specs`` collected earlier in run_grpo_notebook() are
    exported as JSONL artifacts but are NOT consumed by this live training
    path. NeMo RL's grpo_train() generates fresh rollouts internally via
    LateOrderDataset + LateOrderTrainingEnv (an IterableDataset + multi-step
    environment). The collected datum_specs use a different reward shaping
    pipeline (see reward_views.py) and cannot be injected into NeMo RL's
    training loop without framework-level changes.

    For offline training on previously collected episodes, use:
        python -m src.training.run_grpo_training

    Delegates to the same infrastructure as ``python -m src.training.run_grpo_training``
    but skips CLI argument parsing.  Requires GPU resources (7 GPUs per
    ``grpo_config.yaml``) and the model weights at the configured path.
    When GPUs are unavailable, ``run_grpo_notebook(dry_run=False)``
    catches the error and falls back to dry-run automatically.
    """
    import pprint
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
    from nemo_rl.algorithms.utils import get_tokenizer, set_seed
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.utils.config import load_config
    from nemo_rl.utils.logger import get_next_experiment_dir
    from nemo_rl.distributed.virtual_cluster import init_ray

    from src.training.run_grpo_training import (
        LateOrderDataset,
        LateOrderTrainingEnv,
    )

    OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "grpo_config.yaml",
    )
    config = load_config(config_path)
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])

    init_ray()
    set_seed(config["grpo"]["seed"])

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer,
    )

    ds_length = (
        config["grpo"]["num_prompts_per_step"]
        * config["grpo"]["num_generations_per_prompt"]
        * config["grpo"]["max_num_steps"]
    )
    train_dataset = LateOrderDataset(tokenizer=tokenizer, length=ds_length)
    val_dataset = LateOrderDataset(
        tokenizer=tokenizer, length=config["grpo"]["max_val_samples"],
    )

    (
        policy, policy_generation, cluster,
        dataloader, val_dataloader,
        loss_fn, logger, checkpointer,
        grpo_state, master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    import ray
    env = LateOrderTrainingEnv.options(num_gpus=0).remote(
        cfg=dict(config["env"]["late_order_recovery"]),
    )
    task_to_env = {"late_order_recovery": env}

    grpo_train(
        policy, policy_generation,
        dataloader, val_dataloader, tokenizer,
        loss_fn, task_to_env, task_to_env,
        logger, checkpointer, grpo_state, master_config,
    )

    return {
        "step": config["grpo"]["max_num_steps"],
        "training_backend": "nemo_rl",
    }


# ---------------------------------------------------------------------------
# Reward/advantage data extraction for plotting
# ---------------------------------------------------------------------------

def profile_reward_distribution(
    datum_specs: list[dict[str, Any]],
    label: str = "pre-training",
) -> dict[str, float]:
    """Profile the reward distribution of a datum group and print diagnostics.

    Checks for degenerate distributions (all-zero, all-one, zero variance)
    that indicate verification or difficulty mismatches before GRPO training
    wastes compute (RL_ARCHITECTURE.md § Verification and Reward Design).

    Args:
        datum_specs: The GRPO datum group to profile.
        label: Label for the diagnostic output.

    Returns:
        Dict with mean, std, min, max, and a pass/fail flag.
    """
    rewards = [
        d.get("extra_env_info", {}).get("reward", 0.0)
        for d in datum_specs
    ]
    n = len(rewards)
    if n == 0:
        print(f"[{label}] WARNING: No datum specs to profile.")
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0, "pass": False}

    mean_r = sum(rewards) / n
    var_r = sum((r - mean_r) ** 2 for r in rewards) / n
    std_r = var_r ** 0.5
    min_r = min(rewards)
    max_r = max(rewards)

    all_zero = all(abs(r) < 1e-6 for r in rewards)
    all_one = all(abs(r - 1.0) < 1e-6 for r in rewards)
    degenerate = all_zero or all_one or std_r < 1e-6

    print(f"[{label}] Reward distribution (n={n}):")
    print(f"  mean={mean_r:+.4f}  std={std_r:.4f}  min={min_r:+.4f}  max={max_r:+.4f}")
    if degenerate:
        print(f"  WARNING: Degenerate reward distribution detected.")
        if all_zero:
            print(f"  All rewards are zero — model may not produce valid tool calls.")
        elif all_one:
            print(f"  All rewards are 1.0 — verification may be too lenient.")
        else:
            print(f"  Zero variance — GRPO advantages will be zero, no training signal.")
    else:
        print(f"  OK: Reward variance is sufficient for GRPO training.")

    return {
        "mean": round(mean_r, 4),
        "std": round(std_r, 4),
        "min": round(min_r, 4),
        "max": round(max_r, 4),
        "n": n,
        "pass": not degenerate,
    }


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
    datum_specs = result.datum_specs

    total_rewards = [
        d.get("extra_env_info", {}).get("reward", 0.0)
        for d in datum_specs
    ]
    advantages = [
        d.get("extra_env_info", {}).get("group_advantage", 0.0)
        for d in datum_specs
    ]
    per_step_rewards = [
        d.get("extra_env_info", {}).get("per_step_rewards", [])
        for d in datum_specs
    ]

    labels = []
    for i, d in enumerate(datum_specs):
        info = d.get("extra_env_info", {})
        task_id = info.get("task_id", "?")
        repairs = info.get("metrics", {}).get("repair_attempts", 0)
        label = f"ep{i} ({task_id})"
        if repairs > 0:
            label += f" [R={int(repairs)}]"
        labels.append(label)

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
