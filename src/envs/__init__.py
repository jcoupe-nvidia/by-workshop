"""
Explicit RL/task environment package: state, transitions, rewards, validators.

Public surface:
    - LateOrderRecoveryEnv: main environment class (reset, step, terminate)
    - LateOrderEnvState: full environment state dataclass
    - StepResult: transition outcome with metadata for reward computation
    - RewardSignal: decomposed step-level reward
    - EpisodeRewardSummary: aggregated episode rewards
    - Subgoal: enum of diagnostic subgoals
    - check_dependencies: backward-compatible dependency check function
"""
from src.envs.state import (  # noqa: F401
    LateOrderEnvState,
    Subgoal,
    SUBGOAL_ORDER,
    TOOL_COMPLETES_SUBGOAL,
    TOOL_TO_SUBGOAL,
    make_initial_state,
)
from src.envs.transitions import (  # noqa: F401
    StepResult,
    apply_tool_call,
    apply_terminal,
    record_invalid_action,
    check_preconditions,
    should_force_terminate,
    MAX_EPISODE_STEPS,
)
from src.envs.rewards import (  # noqa: F401
    RewardSignal,
    EpisodeRewardSummary,
    compute_step_reward,
    compute_terminal_reward,
    summarize_episode_rewards,
    REWARD_WEIGHTS,
    EXPECTED_ARGUMENTS,
    OPTIMAL_TOOL_SEQUENCE,
)
from src.envs.late_order_env import LateOrderRecoveryEnv  # noqa: F401
from src.envs.validators import check_dependencies  # noqa: F401
