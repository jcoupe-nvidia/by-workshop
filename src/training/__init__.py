"""
NeMo RL-facing training semantics package.

Owns:
    - Trainer-facing datasets, reward views, and staged training progression
    - NeMo RL-specific consumption and batch preparation
    - Experiment configuration and curriculum staging
    - SFT and RL record formatting

Does NOT own:
    - Episode type definitions or serialization (see rollouts/)
    - Reward computation from environment transitions (see envs.rewards)
    - Megatron parallelism, checkpointing, or launch configs (see systems/)
    - Offline evaluation metrics (see eval/)

Modules:
    - curriculum:       4-stage training curriculum (SFT -> short RL -> full RL -> robustness)
    - reward_views:     Stage-aware reward shaping over environment reward signals
    - datasets:         Episode filtering and training dataset assembly per stage
    - nemo_rl_adapter:  NeMo RL trainer-ready record building and serialization
    - experiments:      Experiment configs tying stages to training runs
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
from src.training.nemo_rl_adapter import (
    NeMoRLTrainingRecord,
    build_nemo_training_record,
    build_nemo_training_batch,
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
    # nemo_rl_adapter
    "NeMoRLTrainingRecord",
    "build_nemo_training_record",
    "build_nemo_training_batch",
    "save_training_records_jsonl",
    # experiments
    "ExperimentConfig",
    "ExperimentPlan",
    "build_default_experiment_plan",
    "print_experiment_plan",
]
