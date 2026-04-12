"""
Training semantics package — NeMo RL is the primary downstream consumer.

Owns:
    - Training-oriented datasets, reward views, and staged training progression
    - Episode -> DatumSpec conversion for NeMo RL
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
    - nemo_rl_adapter:      Episode/TrainingRecord -> DatumSpec conversion for NeMo RL
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
from src.training.nemo_rl_adapter import (
    episode_to_datum_spec,
    training_record_to_datum_spec,
    training_batch_to_datum_specs,
    enriched_episodes_to_datum_specs,
    build_sft_datum_spec,
    build_sft_datum_group,
    save_datum_specs_jsonl,
    save_datum_group_jsonl,
)
from src.training.experiments import (
    ExperimentConfig,
    ExperimentPlan,
    build_default_experiment_plan,
    print_experiment_plan,
)
from src.training.grpo_notebook import (
    GRPORunResult,
    run_grpo_notebook,
    collect_enriched_rollouts,
    build_grpo_group_from_rollouts,
    export_artifacts,
    extract_reward_plot_data,
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
    # nemo_rl_adapter (DatumSpec)
    "episode_to_datum_spec",
    "training_record_to_datum_spec",
    "training_batch_to_datum_specs",
    "enriched_episodes_to_datum_specs",
    "build_sft_datum_spec",
    "build_sft_datum_group",
    "save_datum_specs_jsonl",
    "save_datum_group_jsonl",
    # experiments
    "ExperimentConfig",
    "ExperimentPlan",
    "build_default_experiment_plan",
    "print_experiment_plan",
    # grpo_notebook
    "GRPORunResult",
    "run_grpo_notebook",
    "collect_enriched_rollouts",
    "build_grpo_group_from_rollouts",
    "export_artifacts",
    "extract_reward_plot_data",
]
