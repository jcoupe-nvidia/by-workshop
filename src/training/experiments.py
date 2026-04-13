"""
Experiment configuration: ties curriculum stages to concrete training runs.

Defines ExperimentConfig records that pair a curriculum stage with the
data paths, hyperparameters, reward views, and checkpoint references
needed to execute a training run. Provides a default experiment plan
for the late-order-recovery scenario.

Owns:
    - Experiment definition (stage + hyperparameters + data paths)
    - Default experiment plan for the workshop scenario
    - Experiment summaries and inspection helpers

Does NOT own:
    - Curriculum stage definitions (see training.curriculum)
    - Dataset construction or filtering (see training.datasets)
    - Reward computation or shaping (see envs.rewards, training.reward_views)
    - NeMo RL DatumSpec building (see training.nemo_rl_adapter)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.training.curriculum import (
    TrainingStage,
    StageConfig,
    get_stage_config,
    get_curriculum,
)


# -- Experiment configuration -------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for one training experiment (one curriculum stage run).

    Attributes:
        name: Human-readable experiment name.
        stage: Which curriculum stage this experiment executes.
        stage_config: The full StageConfig for this stage.
        model_name: Base model for training.
        data_dir: Directory containing training data for this stage.
        output_dir: Directory for checkpoints and outputs.
        hyperparameters: Training hyperparameters (learning rate, batch size, etc).
        reward_config: Stage-specific reward configuration overrides.
        checkpoint_from: Path to a checkpoint to initialize from (previous stage).
        notes: Human-readable notes about this experiment.
    """
    name: str
    stage: TrainingStage
    stage_config: StageConfig
    model_name: str = "nvidia/nemotron-3-nano"
    data_dir: str = ""
    output_dir: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    reward_config: dict[str, float] = field(default_factory=dict)
    checkpoint_from: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage.value,
            "model_name": self.model_name,
            "data_dir": self.data_dir,
            "output_dir": self.output_dir,
            "hyperparameters": dict(self.hyperparameters),
            "reward_config": dict(self.reward_config),
            "checkpoint_from": self.checkpoint_from,
            "entry_condition": self.stage_config.entry_condition,
            "max_episode_length": self.stage_config.max_episode_length,
            "step_reward_weight": self.stage_config.step_reward_weight,
            "trajectory_reward_weight": self.stage_config.trajectory_reward_weight,
            "notes": self.notes,
        }


@dataclass
class ExperimentPlan:
    """An ordered sequence of experiments forming a complete training plan.

    Represents the full curriculum progression from SFT through robustness.

    NOTE: This plan is illustrative — it documents the intended 4-stage
    curriculum progression but is not yet wired into run_grpo_training.py
    or grpo_notebook.py. See those modules for the active training paths.
    """
    experiments: list[ExperimentConfig] = field(default_factory=list)
    plan_name: str = ""
    description: str = ""

    @property
    def stages(self) -> list[TrainingStage]:
        return [e.stage for e in self.experiments]

    def get_experiment(self, stage: TrainingStage) -> ExperimentConfig | None:
        for exp in self.experiments:
            if exp.stage == stage:
                return exp
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "description": self.description,
            "num_experiments": len(self.experiments),
            "stages": [e.stage.value for e in self.experiments],
            "experiments": [e.to_dict() for e in self.experiments],
        }


# -- Default experiment plan ---------------------------------------------------

# Hyperparameters aligned with the current workshop training flow for each stage
_SFT_HYPERPARAMS: dict[str, Any] = {
    "method": "sft",
    "learning_rate": 5e-5,
    "batch_size": 8,
    "max_epochs": 3,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
}

_SHORT_HORIZON_RL_HYPERPARAMS: dict[str, Any] = {
    "method": "grpo",
    "learning_rate": 1e-6,
    "batch_size": 4,
    "max_steps_per_episode": 3,
    "kl_penalty_coeff": 0.05,
    "group_size": 4,
    "baseline": "group_mean",
}

_FULL_MULTISTEP_RL_HYPERPARAMS: dict[str, Any] = {
    "method": "grpo",
    "learning_rate": 5e-7,
    "batch_size": 4,
    "max_steps_per_episode": 12,
    "kl_penalty_coeff": 0.02,
    "group_size": 4,
    "baseline": "group_mean",
}

_ROBUSTNESS_HYPERPARAMS: dict[str, Any] = {
    "method": "grpo",
    "learning_rate": 3e-7,
    "batch_size": 4,
    "max_steps_per_episode": 15,
    "kl_penalty_coeff": 0.03,
    "group_size": 4,
    "baseline": "group_mean",
}


def build_default_experiment_plan(
    model_name: str = "nvidia/nemotron-3-nano",
    base_data_dir: str = "data/training",
    base_output_dir: str = "outputs/training",
) -> ExperimentPlan:
    """Build the default 4-stage experiment plan for late-order recovery.

    NOTE: This plan is illustrative — it documents the intended curriculum
    progression and hyperparameter choices but is not yet consumed by the
    active training runners (run_grpo_training.py, grpo_notebook.py).
    See those modules for the currently active training paths.

    This creates one ExperimentConfig per curriculum stage with:
        - Stage-appropriate hyperparameters
        - Sequential checkpoint chaining (each stage initializes from previous)
        - Separate data and output directories per stage

    Args:
        model_name: Base model for all stages.
        base_data_dir: Root directory for training data.
        base_output_dir: Root directory for outputs.

    Returns:
        ExperimentPlan with four experiments.
    """
    curriculum = get_curriculum()

    sft_config = get_stage_config(TrainingStage.SFT_SUCCESSFUL)
    short_config = get_stage_config(TrainingStage.SHORT_HORIZON_RL)
    full_config = get_stage_config(TrainingStage.FULL_MULTISTEP_RL)
    robust_config = get_stage_config(TrainingStage.ROBUSTNESS)

    sft_output = f"{base_output_dir}/01_sft"
    short_output = f"{base_output_dir}/02_short_horizon_rl"
    full_output = f"{base_output_dir}/03_full_multistep_rl"
    robust_output = f"{base_output_dir}/04_robustness"

    experiments = [
        ExperimentConfig(
            name="01_sft_successful_trajectories",
            stage=TrainingStage.SFT_SUCCESSFUL,
            stage_config=sft_config,
            model_name=model_name,
            data_dir=f"{base_data_dir}/sft",
            output_dir=sft_output,
            hyperparameters=_SFT_HYPERPARAMS,
            reward_config={},
            checkpoint_from=None,
            notes=(
                "SFT on expert-quality trajectories. Teaches structured "
                "tool-call format, argument extraction, and baseline "
                "decision sequence for late-order recovery."
            ),
        ),
        ExperimentConfig(
            name="02_short_horizon_rl",
            stage=TrainingStage.SHORT_HORIZON_RL,
            stage_config=short_config,
            model_name=model_name,
            data_dir=f"{base_data_dir}/short_horizon",
            output_dir=short_output,
            hyperparameters=_SHORT_HORIZON_RL_HYPERPARAMS,
            reward_config=short_config.reward_config,
            checkpoint_from=f"{sft_output}/checkpoint_best",
            notes=(
                "Short-horizon GRPO on 1-3 tool-call episodes. Dense "
                "per-step rewards for tool selection and argument accuracy. "
                "Higher KL penalty to stay close to SFT policy. Group-relative "
                "advantage computed over grouped trajectories for the same task."
            ),
        ),
        ExperimentConfig(
            name="03_full_multistep_rl",
            stage=TrainingStage.FULL_MULTISTEP_RL,
            stage_config=full_config,
            model_name=model_name,
            data_dir=f"{base_data_dir}/full_multistep",
            output_dir=full_output,
            hyperparameters=_FULL_MULTISTEP_RL_HYPERPARAMS,
            reward_config=full_config.reward_config,
            checkpoint_from=f"{short_output}/checkpoint_best",
            notes=(
                "Full multi-step GRPO on complete episodes (5-10 tool calls). "
                "Sequence-aware rewards covering the full decision process: "
                "diagnosis, assessment, alternate recovery, recommendation. "
                "Group-relative advantage over grouped trajectories."
            ),
        ),
        ExperimentConfig(
            name="04_robustness_curriculum",
            stage=TrainingStage.ROBUSTNESS,
            stage_config=robust_config,
            model_name=model_name,
            data_dir=f"{base_data_dir}/robustness",
            output_dir=robust_output,
            hyperparameters=_ROBUSTNESS_HYPERPARAMS,
            reward_config=robust_config.reward_config,
            checkpoint_from=f"{full_output}/checkpoint_best",
            notes=(
                "Robustness curriculum with malformed calls, dead ends, "
                "repair/reject cycles, and looping. Explicit penalties for "
                "silent fallback reliance and hallucinated conclusions. "
                "Lower learning rate and clip range for stability."
            ),
        ),
    ]

    return ExperimentPlan(
        experiments=experiments,
        plan_name="late_order_recovery_training",
        description=(
            "4-stage curriculum for training a tool-calling agent on the "
            "late-order recovery scenario (SO-10482). Progresses from SFT "
            "through short-horizon GRPO, full multi-step GRPO, and robustness "
            "training. RL stages use GRPO with group-relative advantage over "
            "grouped trajectories. Exports are consumed by NeMo RL."
        ),
    )


# -- Inspection helpers -------------------------------------------------------

def print_experiment_plan(plan: ExperimentPlan) -> None:
    """Pretty-print an experiment plan."""
    print(f"Experiment Plan: {plan.plan_name}")
    print(f"  {plan.description}")
    print()

    for i, exp in enumerate(plan.experiments):
        stage_tag = exp.stage.value.upper()
        print(f"  Stage {i + 1}: {exp.name} [{stage_tag}]")
        print(f"    Model:      {exp.model_name}")
        print(f"    Method:     {exp.hyperparameters.get('method', '?')}")
        print(f"    LR:         {exp.hyperparameters.get('learning_rate', '?')}")
        print(f"    Batch size: {exp.hyperparameters.get('batch_size', '?')}")
        print(f"    Max steps:  {exp.stage_config.max_episode_length}")
        print(f"    Blend:      step={exp.stage_config.step_reward_weight} "
              f"traj={exp.stage_config.trajectory_reward_weight}")
        if exp.checkpoint_from:
            print(f"    Init from:  {exp.checkpoint_from}")
        print(f"    Entry:      {exp.stage_config.entry_condition}")
        print(f"    Notes:      {exp.notes}")
        print()
