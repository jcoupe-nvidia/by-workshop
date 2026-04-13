"""
Training-oriented reward views over environment reward signals.

Bridges envs.rewards to what the training pipeline needs: stage-aware reward
blending, component selection, and normalization for different training
objectives (SFT cross-entropy, RL policy gradient, etc.).

Reward semantics split (documented per MEDIUM-3 code review):
    - Online (LateOrderTrainingEnv.step): Returns raw marginal per-step
      rewards from envs.rewards.compute_step_reward(). These are the
      environment's native dense rewards used directly by NeMo RL's GRPO.
    - Offline (build_episode_reward_view): Applies stage-specific component
      weights and step/trajectory blending from the curriculum config.
      Used for JSONL export, datum group building, and offline analysis.

    These are intentionally different: the online path gives the trainer
    raw signal; the offline path shapes rewards for curriculum progression.
    Both derive from the same underlying RewardSignal components.

Owns:
    - Stage-aware reward blending (step vs trajectory weights per curriculum stage)
    - Reward component selection and masking per stage
    - Combined reward computation from step-level and trajectory-level signals
    - Reward normalization for SFT vs RL training objectives

Does NOT own:
    - Reward computation from environment transitions (see envs.rewards)
    - Curriculum stage definitions (see training.curriculum)
    - Training trajectory format (see rollouts.export_adapters)
    - Dataset construction or episode filtering (see training.datasets)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.envs.rewards import (
    RewardSignal,
    EpisodeRewardSummary,
    REWARD_WEIGHTS,
)
from src.training.curriculum import StageConfig, TrainingStage


# -- Default reward component weights per training objective -------------------

# SFT uses cross-entropy loss, not reward signals. These are placeholders
# so the same data pipeline works; actual SFT ignores them.
SFT_REWARD_WEIGHTS: dict[str, float] = {
    k: 0.0 for k in REWARD_WEIGHTS
}

# Short-horizon RL emphasizes per-step correctness
SHORT_HORIZON_REWARD_WEIGHTS: dict[str, float] = {
    "valid_call":           0.25,
    "correct_tool":         0.25,
    "correct_arguments":    0.20,
    "dependency_satisfied": 0.15,
    "non_redundant":        0.05,
    "progress":             0.05,
    "efficiency":           0.05,
    "terminal_quality":     0.00,
}

# Full multi-step RL balances step and trajectory signals
FULL_MULTISTEP_REWARD_WEIGHTS: dict[str, float] = {
    "valid_call":           0.15,
    "correct_tool":         0.15,
    "correct_arguments":    0.10,
    "dependency_satisfied": 0.15,
    "non_redundant":        0.10,
    "progress":             0.15,
    "efficiency":           0.10,
    "terminal_quality":     0.10,
}

# Robustness stage increases penalty sensitivity
ROBUSTNESS_REWARD_WEIGHTS: dict[str, float] = {
    "valid_call":           0.20,
    "correct_tool":         0.10,
    "correct_arguments":    0.10,
    "dependency_satisfied": 0.15,
    "non_redundant":        0.15,
    "progress":             0.10,
    "efficiency":           0.05,
    "terminal_quality":     0.15,
}

# Map stages to their default component weights
STAGE_REWARD_WEIGHTS: dict[TrainingStage, dict[str, float]] = {
    TrainingStage.SFT_SUCCESSFUL: SFT_REWARD_WEIGHTS,
    TrainingStage.SHORT_HORIZON_RL: SHORT_HORIZON_REWARD_WEIGHTS,
    TrainingStage.FULL_MULTISTEP_RL: FULL_MULTISTEP_REWARD_WEIGHTS,
    TrainingStage.ROBUSTNESS: ROBUSTNESS_REWARD_WEIGHTS,
}


# -- Reward view dataclass ----------------------------------------------------

@dataclass
class StepRewardView:
    """Training-oriented view of one step's reward, shaped by stage config.

    Unlike the raw RewardSignal from envs.rewards (which has fixed component
    weights), this view applies stage-specific weights and blending.
    """
    step_index: int
    raw_reward: float         # original env reward (RewardSignal.total)
    shaped_reward: float      # reward after stage-specific weighting
    components: dict[str, float] = field(default_factory=dict)
    penalties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "raw_reward": self.raw_reward,
            "shaped_reward": self.shaped_reward,
            "components": dict(self.components),
            "penalties": list(self.penalties),
        }


@dataclass
class EpisodeRewardView:
    """Training-oriented view of an episode's rewards, shaped by stage config.

    Combines step-level and trajectory-level signals using the stage's
    blend weights (step_reward_weight, trajectory_reward_weight).
    """
    step_views: list[StepRewardView]
    terminal_view: StepRewardView | None
    trajectory_reward: float       # trajectory-level signal (avg of shaped step rewards)
    combined_reward: float         # final blended reward for training
    step_reward_weight: float      # from stage config
    trajectory_reward_weight: float

    @property
    def num_steps(self) -> int:
        return len(self.step_views)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_steps": self.num_steps,
            "trajectory_reward": self.trajectory_reward,
            "combined_reward": self.combined_reward,
            "step_reward_weight": self.step_reward_weight,
            "trajectory_reward_weight": self.trajectory_reward_weight,
            "steps": [s.to_dict() for s in self.step_views],
            "terminal": self.terminal_view.to_dict() if self.terminal_view else None,
        }


# -- Core reward shaping functions --------------------------------------------

def shape_step_reward(
    signal: RewardSignal,
    step_index: int,
    weights: dict[str, float],
) -> StepRewardView:
    """Apply stage-specific component weights to a raw RewardSignal.

    Args:
        signal: Raw reward signal from the environment.
        step_index: Position in the episode.
        weights: Stage-specific component weights to apply.

    Returns:
        StepRewardView with shaped reward.
    """
    components = {
        "valid_call": signal.valid_call,
        "correct_tool": signal.correct_tool,
        "correct_arguments": signal.correct_arguments,
        "dependency_satisfied": signal.dependency_satisfied,
        "non_redundant": signal.non_redundant,
        "progress": signal.progress,
        "efficiency": signal.efficiency,
        "terminal_quality": signal.terminal_quality,
    }

    shaped = round(
        sum(components[k] * weights.get(k, 0.0) for k in components),
        4,
    )

    return StepRewardView(
        step_index=step_index,
        raw_reward=signal.total,
        shaped_reward=shaped,
        components=components,
        penalties=list(signal.penalties),
    )


def build_episode_reward_view(
    reward_summary: EpisodeRewardSummary,
    stage_config: StageConfig,
    component_weights: dict[str, float] | None = None,
) -> EpisodeRewardView:
    """Build a training-oriented reward view for an episode.

    Takes the raw environment reward summary and applies stage-specific
    shaping: component weights, step/trajectory blending, and normalization.

    Args:
        reward_summary: Raw reward summary from episode_runner enrichment.
        stage_config: The curriculum stage config controlling blend weights.
        component_weights: Override component weights. If None, uses the
                           default weights for the stage.

    Returns:
        EpisodeRewardView ready for training consumption.
    """
    if component_weights is None:
        # Prefer the stage_config.reward_config when it has component-level
        # weights, so curriculum definitions directly control reward shaping.
        # Fall back to STAGE_REWARD_WEIGHTS (module-level defaults) and then
        # to the environment's base REWARD_WEIGHTS.
        if stage_config.reward_config:
            component_weights = stage_config.reward_config
        else:
            component_weights = STAGE_REWARD_WEIGHTS.get(
                stage_config.stage, REWARD_WEIGHTS,
            )

    # Shape each step reward
    step_views: list[StepRewardView] = []
    for i, signal in enumerate(reward_summary.step_rewards):
        step_views.append(shape_step_reward(signal, i, component_weights))

    # Shape terminal reward if present
    terminal_view: StepRewardView | None = None
    if reward_summary.terminal_reward is not None:
        terminal_view = shape_step_reward(
            reward_summary.terminal_reward,
            len(step_views),
            component_weights,
        )

    # Step-level signal: average of per-step shaped rewards (excludes terminal)
    step_shaped = [sv.shaped_reward for sv in step_views]
    avg_step = round(
        sum(step_shaped) / len(step_shaped) if step_shaped else 0.0, 4,
    )

    # Trajectory-level signal: average of all shaped rewards (steps + terminal)
    all_shaped = list(step_shaped)
    if terminal_view is not None:
        all_shaped.append(terminal_view.shaped_reward)

    trajectory_reward = round(
        sum(all_shaped) / len(all_shaped) if all_shaped else 0.0, 4,
    )

    # Blend step-level and trajectory-level using stage config
    sw = stage_config.step_reward_weight
    tw = stage_config.trajectory_reward_weight

    combined = round(sw * avg_step + tw * trajectory_reward, 4)

    return EpisodeRewardView(
        step_views=step_views,
        terminal_view=terminal_view,
        trajectory_reward=trajectory_reward,
        combined_reward=combined,
        step_reward_weight=sw,
        trajectory_reward_weight=tw,
    )


def get_per_step_rewards(view: EpisodeRewardView) -> list[float]:
    """Extract the shaped per-step reward list from an EpisodeRewardView.

    Returns one float per step (plus terminal if present), suitable for
    direct use as the reward signal in training trajectory records.
    """
    rewards = [sv.shaped_reward for sv in view.step_views]
    if view.terminal_view is not None:
        rewards.append(view.terminal_view.shaped_reward)
    return rewards
