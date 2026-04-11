"""
Training semantics package — openpipe-art is the primary downstream consumer.

Owns:
    - Training-oriented datasets, reward views, and staged training progression
    - openpipe-art record building and batch preparation
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
    - openpipe_art_adapter: openpipe-art record building and serialization
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
    OpenPipeArtTrainingRecord,
    build_openpipe_art_training_record,
    build_openpipe_art_training_batch,
    save_training_records_jsonl,
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
    # openpipe_art_adapter
    "OpenPipeArtTrainingRecord",
    "build_openpipe_art_training_record",
    "build_openpipe_art_training_batch",
    "save_training_records_jsonl",
    # experiments
    "ExperimentConfig",
    "ExperimentPlan",
    "build_default_experiment_plan",
    "print_experiment_plan",
]
