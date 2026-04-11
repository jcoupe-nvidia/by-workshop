"""
Training curriculum staging for multi-turn RL on tool-calling agents.

Defines the staged training progression that later phases will implement:

    Stage 1 — SFT on successful trajectories
              Supervised fine-tuning using expert-quality episode traces.

    Stage 2 — Short-horizon RL with dense rewards
              RL over 1-3 tool-call episodes with step-level reward signals
              for tool validity, argument accuracy, and dependency satisfaction.

    Stage 3 — Full multi-turn RL with sequence-aware rewards
              RL over complete episodes (5-10 tool calls) with rewards that
              cover the full decision process: diagnosis, assessment,
              alternate recovery, and recommendation synthesis.

    Stage 4 — Robustness curriculum with malformed calls and dead ends
              Training on adversarial and edge-case episodes: malformed tool
              calls, repair/reject cycles, dead-end recovery paths, and
              looping behavior.

Owns:
    - Stage definitions and ordering
    - Per-stage data selection criteria
    - Per-stage reward configuration
    - Stage transition conditions

Does NOT own:
    - Reward computation (see envs.rewards, training.reward_views)
    - Dataset construction (see training.datasets)
    - openpipe-art adapter specifics (see training.openpipe_art_adapter)
    - Historical systems execution details (see systems/)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrainingStage(Enum):
    """The four curriculum stages, in order."""
    SFT_SUCCESSFUL = "sft_successful"
    SHORT_HORIZON_RL = "short_horizon_rl"
    FULL_MULTITURN_RL = "full_multiturn_rl"
    ROBUSTNESS = "robustness"


@dataclass
class StageConfig:
    """Configuration contract for one curriculum stage.

    Attributes:
        stage: Which curriculum stage this configures.
        description: Human-readable purpose of the stage.
        episode_filter: Criteria for selecting episodes into this stage's
                        training set. Keys are field names on Episode or
                        EpisodeMetrics; values are predicates or thresholds.
        max_episode_length: Maximum tool calls per episode for this stage.
        reward_config: Stage-specific reward weighting overrides.
                       Keys match reward component names; values are weights.
        step_reward_weight: Blend weight for dense step-level rewards (0-1).
        trajectory_reward_weight: Blend weight for sparse trajectory reward (0-1).
        entry_condition: Description of when this stage should begin
                         (e.g. "after SFT converges on successful traces").
    """
    stage: TrainingStage
    description: str
    episode_filter: dict[str, Any] = field(default_factory=dict)
    max_episode_length: int = 15
    reward_config: dict[str, float] = field(default_factory=dict)
    step_reward_weight: float = 0.4
    trajectory_reward_weight: float = 0.6
    entry_condition: str = ""


# ---------------------------------------------------------------------------
# Default curriculum: the 4-stage progression from REFACTOR.md
# ---------------------------------------------------------------------------

DEFAULT_CURRICULUM: list[StageConfig] = [
    StageConfig(
        stage=TrainingStage.SFT_SUCCESSFUL,
        description=(
            "Supervised fine-tuning on successful trajectories. "
            "Teaches the model the correct tool-call format, argument "
            "structure, and a baseline decision sequence."
        ),
        episode_filter={
            "is_complete": True,
            "total_reward_min": 0.7,
            "invalid_tool_calls_max": 0,
            "rejects_max": 0,
        },
        max_episode_length=10,
        reward_config={},  # SFT uses cross-entropy, not RL rewards
        step_reward_weight=0.0,
        trajectory_reward_weight=0.0,
        entry_condition="Initial stage. Requires a set of expert-quality episodes.",
    ),
    StageConfig(
        stage=TrainingStage.SHORT_HORIZON_RL,
        description=(
            "Short-horizon RL with dense per-step rewards. "
            "Episodes are truncated to 1-3 tool calls so the model "
            "learns individual tool selection and argument extraction "
            "before tackling full multi-turn sequences."
        ),
        episode_filter={
            "is_complete": True,
            "min_steps": 1,
            "max_steps": 3,
        },
        max_episode_length=3,
        reward_config={
            "tool_validity": 0.30,
            "sequence_correctness": 0.30,
            "tool_accuracy": 0.25,
            "recovery_bonus": 0.15,
        },
        step_reward_weight=0.7,
        trajectory_reward_weight=0.3,
        entry_condition="After SFT checkpoint shows stable tool-call formatting.",
    ),
    StageConfig(
        stage=TrainingStage.FULL_MULTITURN_RL,
        description=(
            "Full multi-turn RL with sequence-aware rewards. "
            "Episodes span the complete decision process (5-10 tool calls): "
            "diagnosis, primary assessment, alternate recovery, and "
            "recommendation synthesis."
        ),
        episode_filter={
            "is_complete": True,
            "min_steps": 4,
        },
        max_episode_length=12,
        reward_config={
            "tool_validity": 0.20,
            "sequence_correctness": 0.35,
            "tool_accuracy": 0.20,
            "recovery_bonus": 0.10,
            "task_success": 0.15,
        },
        step_reward_weight=0.4,
        trajectory_reward_weight=0.6,
        entry_condition=(
            "After short-horizon RL shows reliable single-tool accuracy "
            "and dependency satisfaction."
        ),
    ),
    StageConfig(
        stage=TrainingStage.ROBUSTNESS,
        description=(
            "Robustness curriculum with malformed calls and dead ends. "
            "Training includes episodes with repair/reject cycles, "
            "dependency violations, looping, and recovery from bad states. "
            "Penalties for silent fallback reliance and hallucinated conclusions."
        ),
        episode_filter={
            # Intentionally includes failed and partially-successful episodes
            "include_failures": True,
            "include_repairs": True,
        },
        max_episode_length=15,
        reward_config={
            "tool_validity": 0.20,
            "sequence_correctness": 0.25,
            "tool_accuracy": 0.15,
            "recovery_bonus": 0.20,
            "penalty_malformed": -0.10,
            "penalty_loop": -0.10,
        },
        step_reward_weight=0.5,
        trajectory_reward_weight=0.5,
        entry_condition=(
            "After full multi-turn RL shows stable task completion. "
            "This stage hardens the policy against adversarial inputs."
        ),
    ),
]


def get_curriculum() -> list[StageConfig]:
    """Return the default 4-stage curriculum progression."""
    return list(DEFAULT_CURRICULUM)


def get_stage_config(stage: TrainingStage) -> StageConfig:
    """Look up the config for a specific training stage."""
    for config in DEFAULT_CURRICULUM:
        if config.stage == stage:
            return config
    raise ValueError(f"No config defined for stage: {stage}")
