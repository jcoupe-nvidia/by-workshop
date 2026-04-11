"""
Training semantics package — openpipe-art is the primary downstream consumer.

Owns:
    - Training-oriented datasets, reward views, and staged training progression
    - Episode -> art.Trajectory / art.TrajectoryGroup conversion
    - Experiment configuration and curriculum staging
    - SFT and RL record formatting

Does NOT own:
    - Episode type definitions or serialization (see rollouts/)
    - Reward computation from environment transitions (see envs.rewards)
    - Offline evaluation metrics (see eval/)

Modules:
    - curriculum:           4-stage training curriculum (SFT -> short RL -> full RL -> robustness)
    - reward_views:         Stage-aware reward shaping over environment reward signals
    - datasets:             Episode filtering and training dataset assembly per stage
    - openpipe_art_adapter: Episode/TrainingRecord -> art.Trajectory/TrajectoryGroup conversion
    - experiments:          Experiment configs tying stages to training runs
"""
from src.training.curriculum import (
    TrainingStage,
    StageConfig,
    get_curriculum,
    get_stage_config,
)
from src.training.reward_views import (
    EpisodeRewardView,
    StepRewardView,
    build_episode_reward_view,
    get_per_step_rewards,
)
from src.training.datasets import (
    TrainingRecord,
    TrainingDataset,
    build_training_dataset,
    build_all_stage_datasets,
    extract_sft_dataset,
)
from src.training.openpipe_art_adapter import (
    episode_to_art_trajectory,
    training_record_to_art_trajectory,
    training_batch_to_art_group,
    enriched_episodes_to_art_group,
    build_sft_art_trajectory,
    build_sft_art_group,
    save_art_trajectories_jsonl,
    save_art_group_jsonl,
)
from src.training.experiments import (
    ExperimentConfig,
    ExperimentPlan,
    build_default_experiment_plan,
    print_experiment_plan,
)

__all__ = [
    # curriculum
    "TrainingStage",
    "StageConfig",
    "get_curriculum",
    "get_stage_config",
    # reward_views
    "EpisodeRewardView",
    "StepRewardView",
    "build_episode_reward_view",
    "get_per_step_rewards",
    # datasets
    "TrainingRecord",
    "TrainingDataset",
    "build_training_dataset",
    "build_all_stage_datasets",
    "extract_sft_dataset",
    # openpipe_art_adapter (art.Trajectory / art.TrajectoryGroup)
    "episode_to_art_trajectory",
    "training_record_to_art_trajectory",
    "training_batch_to_art_group",
    "enriched_episodes_to_art_group",
    "build_sft_art_trajectory",
    "build_sft_art_group",
    "save_art_trajectories_jsonl",
    "save_art_group_jsonl",
    # experiments
    "ExperimentConfig",
    "ExperimentPlan",
    "build_default_experiment_plan",
    "print_experiment_plan",
]
