"""
Dense, sequence-aware reward signals for the late-order recovery environment.

Owns:
    - Step-level reward computation from environment transition facts
    - Penalty definitions for invalid, redundant, and malformed actions
    - Episode-level reward aggregation
    - Reward signal decomposition for training-oriented inspection and shaping

Does NOT own:
    - Environment state or transitions (see envs.state, envs.transitions)
    - Training-oriented reward views or dataset adapters (see training/)
    - Tool implementations or execution (see runtime.tools)
    - Offline evaluation metrics (see eval/)

Reward design principles:
    - Reward the decision process turn by turn, not just final success
    - Dense signals at every step: validity, correctness, progress, efficiency
    - Explicit penalties for malformed calls, dependency violations, redundancy,
      looping, hallucinated conclusions, overlong episodes, silent fallback reliance
    - Reward components are inspectable and decomposable for debugging

Reward shape rationale:
    RL_ARCHITECTURE.md recommends binary (0/1) rewards for GRPO simplicity.
    This module uses continuous decomposed rewards instead, because the
    supply-chain scenario requires per-step training signal: binary task-level
    rewards would not credit partial progress (e.g., correct tool sequence
    with wrong arguments) or penalize specific failure modes (e.g., dependency
    violations vs. redundant calls) during multi-step training. The increased
    GRPO variance from continuous rewards is acceptable given the small
    action space and deterministic tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.envs.state import LateOrderEnvState, Subgoal, SUBGOAL_ORDER
from src.envs.transitions import StepResult


# -- Reward component weights -------------------------------------------------

REWARD_WEIGHTS: dict[str, float] = {
    "valid_call":           0.15,  # was this a structurally valid tool call?
    "correct_tool":         0.15,  # is this tool appropriate for current state?
    "correct_arguments":    0.10,  # did arguments match expected values?
    "dependency_satisfied": 0.15,  # were prerequisites met before this call?
    "non_redundant":        0.10,  # is this a new call, not a repeat?
    "progress":             0.15,  # does this advance toward the goal?
    "efficiency":           0.10,  # are we staying close to optimal step count?
    "terminal_quality":     0.10,  # quality of the final answer (only on terminal step)
}

# Penalty magnitudes (negative values, applied as components)
PENALTY_MALFORMED_CALL = -1.0
PENALTY_INVALID_SCHEMA = -0.8
PENALTY_DEPENDENCY_VIOLATION = -1.0
PENALTY_REDUNDANT_CALL = -0.5
PENALTY_LOOPING = -1.5  # repeated pattern of the same tool
PENALTY_HALLUCINATED_CONCLUSION = -1.0
PENALTY_OVERLONG_EPISODE = -0.3  # per step over optimal
PENALTY_SILENT_FALLBACK = -0.3  # fallback repair succeeded but is penalized


# -- Per-scenario expected arguments (computed from scenario data) -------------

def get_expected_arguments(order_id: str) -> dict[str, dict[str, Any]]:
    """Compute the expected tool arguments for a given order.

    Derives expectations from scenario_data so they stay in sync
    automatically when scenarios are added or modified.
    """
    from src.scenario_data import ORDERS, INVENTORY

    if order_id not in ORDERS:
        return {}

    order = ORDERS[order_id]
    sku = order["sku"]
    source_dc = order["source_dc"]
    committed_date = order["committed_date"]
    qty = order["qty"]

    available = INVENTORY.get((sku, source_dc), {}).get("available", 0)
    shortfall = max(0, qty - available)
    expedite_qty = shortfall if shortfall > 0 else qty

    return {
        "get_order": {"order_id": order_id},
        "get_shipment_status": {"order_id": order_id},
        "get_inventory": {"sku": sku, "dc_id": source_dc},
        "get_fulfillment_capacity": {"dc_id": source_dc, "date": committed_date},
        "find_alternate_inventory": {"sku": sku, "region": "ALL"},
        "get_supplier_expedite_options": {"sku": sku, "qty": expedite_qty},
    }


# Backward-compatible alias: default to SO-10482 for existing call sites
EXPECTED_ARGUMENTS: dict[str, dict[str, Any]] = get_expected_arguments("SO-10482")


# -- Per-scenario optimal tool sequences and expected actions ------------------

# Canonical full sequence used as default and for backward compatibility.
_FULL_TOOL_SEQUENCE = [
    "get_order",
    "get_shipment_status",
    "get_inventory",
    "get_fulfillment_capacity",
    "find_alternate_inventory",
    "get_transfer_eta",
    "get_supplier_expedite_options",
    "score_recovery_options",
    "recommend_action",
]

# Per-scenario expected optimal action derived from scenario_data.py comments.
EXPECTED_ACTION: dict[str, str] = {
    "SO-10482": "transfer",
    "SO-10483": "supplier_expedite",
    "SO-10484": "partial_fulfillment",
    "SO-10485": "fulfill_from_source",
    "SO-10486": "substitute",
    "SO-10487": "escalate",
    "SO-10488": "transfer",
    "SO-10489": "supplier_expedite",
    "SO-10490": "fulfill_from_source",
    "SO-10491": "transfer",
}

# "False alarm" scenarios where the original DC can fulfill — the agent
# should stop after diagnosing that no recovery is needed.
_FALSE_ALARM_SEQUENCE = [
    "get_order",
    "get_shipment_status",
    "get_inventory",
    "get_fulfillment_capacity",
    "score_recovery_options",
    "recommend_action",
]

# Scenarios requiring the full alternate-sourcing investigation.
_FULL_INVESTIGATION_IDS = {
    "SO-10482", "SO-10483", "SO-10484",
    "SO-10486", "SO-10487", "SO-10488", "SO-10489", "SO-10491",
}
_FALSE_ALARM_IDS = {"SO-10485", "SO-10490"}


def get_optimal_tool_sequence(order_id: str) -> list[str]:
    """Return the minimal dependency-satisfying tool sequence for a scenario.

    False-alarm scenarios (SO-10485, SO-10490) omit alternate-sourcing tools
    because the agent should recognize that the order can ship from the
    original DC without needing transfers or supplier expedites.
    """
    if order_id in _FALSE_ALARM_IDS:
        return list(_FALSE_ALARM_SEQUENCE)
    return list(_FULL_TOOL_SEQUENCE)


def get_optimal_step_count(order_id: str) -> int:
    """Return the number of steps in the optimal trajectory for a scenario."""
    return len(get_optimal_tool_sequence(order_id))


def get_expected_action(order_id: str) -> str:
    """Return the expected optimal recovery action for a scenario."""
    return EXPECTED_ACTION.get(order_id, "unknown")


# Backward-compatible module-level aliases (default to full sequence).
OPTIMAL_TOOL_SEQUENCE = list(_FULL_TOOL_SEQUENCE)
OPTIMAL_STEP_COUNT = len(OPTIMAL_TOOL_SEQUENCE)

# Tools that are considered "correct" at each subgoal phase
CORRECT_TOOLS_BY_SUBGOAL: dict[Subgoal, set[str]] = {
    Subgoal.ORDER_DIAGNOSED: {"get_order", "get_shipment_status"},
    Subgoal.PRIMARY_ASSESSED: {"get_inventory", "get_fulfillment_capacity"},
    Subgoal.ALTERNATES_EVALUATED: {
        "find_alternate_inventory", "get_transfer_eta", "get_supplier_expedite_options",
    },
    Subgoal.RECOMMENDATION_SYNTHESIZED: {"score_recovery_options", "recommend_action"},
}

# Required fields in a valid final answer
REQUIRED_ANSWER_FIELDS = {"action", "rationale"}
OPTIONAL_ANSWER_FIELDS = {"confidence", "expected_delivery", "meets_committed_date"}


# -- Reward signal dataclass --------------------------------------------------

@dataclass
class RewardSignal:
    """Decomposed reward for a single environment step.

    Each component is a float in [-1.5, 1.0] range. The total reward
    is the weighted sum using REWARD_WEIGHTS. Components are exposed
    individually so training adapters can inspect and shape them.
    """
    valid_call: float = 0.0
    correct_tool: float = 0.0
    correct_arguments: float = 0.0
    dependency_satisfied: float = 0.0
    non_redundant: float = 0.0
    progress: float = 0.0
    efficiency: float = 0.0
    terminal_quality: float = 0.0
    penalties: list[str] = field(default_factory=list)  # human-readable penalty labels

    @property
    def total(self) -> float:
        """Weighted sum of all components."""
        components = {
            "valid_call": self.valid_call,
            "correct_tool": self.correct_tool,
            "correct_arguments": self.correct_arguments,
            "dependency_satisfied": self.dependency_satisfied,
            "non_redundant": self.non_redundant,
            "progress": self.progress,
            "efficiency": self.efficiency,
            "terminal_quality": self.terminal_quality,
        }
        return round(
            sum(components[k] * REWARD_WEIGHTS[k] for k in REWARD_WEIGHTS),
            4,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_call": self.valid_call,
            "correct_tool": self.correct_tool,
            "correct_arguments": self.correct_arguments,
            "dependency_satisfied": self.dependency_satisfied,
            "non_redundant": self.non_redundant,
            "progress": self.progress,
            "efficiency": self.efficiency,
            "terminal_quality": self.terminal_quality,
            "penalties": list(self.penalties),
            "total": self.total,
        }


# -- Step-level reward computation --------------------------------------------

def compute_step_reward(
    step: StepResult,
    state: LateOrderEnvState,
    tool_arguments: dict[str, Any] | None = None,
    was_repaired: bool = False,
) -> RewardSignal:
    """Compute the dense reward signal for one environment step.

    Args:
        step: The StepResult from the transition function.
        state: The environment state after the transition.
        tool_arguments: The arguments the agent provided (for accuracy checking).
        was_repaired: Whether this step required fallback repair.

    Returns:
        RewardSignal with decomposed component scores.
    """
    signal = RewardSignal()

    # -- Valid call component --
    if step.valid:
        signal.valid_call = 1.0
    else:
        signal.valid_call = -0.5
        if step.error_type == "malformed":
            signal.penalties.append("malformed_call")
            signal.valid_call = PENALTY_MALFORMED_CALL
        elif step.error_type == "unknown_tool":
            signal.penalties.append("invalid_schema")
            signal.valid_call = PENALTY_INVALID_SCHEMA

    # -- Dependency satisfaction --
    if step.valid:
        signal.dependency_satisfied = 1.0
    elif step.error_type == "dependency_violation":
        signal.dependency_satisfied = PENALTY_DEPENDENCY_VIOLATION
        signal.penalties.append("dependency_violation")
    else:
        signal.dependency_satisfied = 0.0

    if not step.valid:
        # Invalid steps get no credit for other components
        return signal

    # -- Correct tool for current state --
    next_subgoal = None
    for sg in SUBGOAL_ORDER:
        if sg not in state.completed_subgoals:
            next_subgoal = sg
            break

    if next_subgoal is not None:
        correct_tools = CORRECT_TOOLS_BY_SUBGOAL.get(next_subgoal, set())
        # Also allow tools from already-completed subgoals' later stages
        # (e.g., get_transfer_eta after find_alternate_inventory within ALTERNATES_EVALUATED)
        if step.tool_name in correct_tools:
            signal.correct_tool = 1.0
        else:
            # Check if it's a tool for a future subgoal (premature but not wrong)
            for future_sg in SUBGOAL_ORDER:
                if future_sg in state.completed_subgoals:
                    continue
                if step.tool_name in CORRECT_TOOLS_BY_SUBGOAL.get(future_sg, set()):
                    signal.correct_tool = 0.3  # premature but valid
                    break
            else:
                signal.correct_tool = 0.0
    else:
        # All subgoals complete — any valid tool is fine
        signal.correct_tool = 0.5

    # -- Correct arguments (per-scenario) --
    scenario_expected = get_expected_arguments(state.order_id)
    if tool_arguments is not None and step.tool_name in scenario_expected:
        expected = scenario_expected[step.tool_name]
        matches = sum(
            1 for k, v in expected.items()
            if str(tool_arguments.get(k)) == str(v)
        )
        signal.correct_arguments = matches / len(expected) if expected else 1.0
    else:
        signal.correct_arguments = 0.5  # neutral default for tools without expectations

    # -- Non-redundancy --
    if step.is_redundant:
        signal.non_redundant = PENALTY_REDUNDANT_CALL
        signal.penalties.append("redundant_call")
        # Check for looping (same tool called 3+ times)
        count = state.tool_call_counts.get(step.tool_name, 0)
        if count >= 3:
            signal.non_redundant = PENALTY_LOOPING
            signal.penalties.append("looping")
    else:
        signal.non_redundant = 1.0

    # -- Progress toward goal --
    if step.subgoal_completed is not None:
        signal.progress = 1.0
    elif step.is_progress:
        signal.progress = 0.5  # contributing to a subgoal but not completing it
    else:
        signal.progress = 0.0

    # -- Efficiency (per-scenario optimal step count) --
    scenario_optimal = get_optimal_step_count(state.order_id)
    valid_calls = state.valid_tool_call_count
    if valid_calls <= scenario_optimal:
        signal.efficiency = 1.0
    else:
        overshoot = valid_calls - scenario_optimal
        signal.efficiency = max(-1.0, PENALTY_OVERLONG_EPISODE * overshoot)
        if overshoot > 0:
            signal.penalties.append("overlong_episode")

    # -- Silent fallback penalty --
    if was_repaired:
        signal.penalties.append("silent_fallback")
        # Reduce the valid_call credit: the call worked but needed help
        signal.valid_call = max(0.0, signal.valid_call + PENALTY_SILENT_FALLBACK)

    return signal


# -- Terminal reward computation ----------------------------------------------

def compute_terminal_reward(
    state: LateOrderEnvState,
) -> RewardSignal:
    """Compute the reward signal for the terminal step.

    Evaluates the quality of the final answer and overall episode.
    """
    signal = RewardSignal()
    signal.valid_call = 0.0  # not a tool call
    signal.dependency_satisfied = 0.0
    signal.non_redundant = 0.0
    signal.correct_tool = 0.0
    signal.correct_arguments = 0.0
    signal.progress = 0.0
    signal.efficiency = 0.0

    if state.terminal_reason != "final_answer" or state.final_answer is None:
        # Episode ended without a proper answer
        if state.terminal_reason == "max_iterations":
            signal.terminal_quality = -0.5
            signal.penalties.append("max_iterations_reached")
        elif state.terminal_reason == "error":
            signal.terminal_quality = -1.0
            signal.penalties.append("error_termination")
        else:
            signal.terminal_quality = -0.5
        return signal

    answer = state.final_answer

    # Check required fields
    present = set(answer.keys()) & REQUIRED_ANSWER_FIELDS
    missing = REQUIRED_ANSWER_FIELDS - present
    if missing:
        signal.terminal_quality = 0.3
        signal.penalties.append("missing_answer_fields")
        return signal

    # All required fields present
    base_score = 0.7

    # Bonus for optional fields
    optional_present = set(answer.keys()) & OPTIONAL_ANSWER_FIELDS
    base_score += 0.1 * len(optional_present) / max(1, len(OPTIONAL_ANSWER_FIELDS))

    # Check if the recommendation is grounded in scored options
    if state.recovery_options_scored:
        base_score += 0.1
    else:
        signal.penalties.append("hallucinated_conclusion")
        base_score -= 0.3

    # Check subgoal coverage
    subgoal_ratio = len(state.completed_subgoals) / len(SUBGOAL_ORDER)
    if subgoal_ratio < 1.0:
        penalty = (1.0 - subgoal_ratio) * 0.3
        base_score -= penalty
        signal.penalties.append("incomplete_subgoals")

    # Outcome correctness: does the recommended action match the expected
    # optimal action for this scenario? (closes MEDIUM-3 / composite
    # verification requirement from RL_ARCHITECTURE.md lines 303-304)
    expected = get_expected_action(state.order_id)
    recommended = str(answer.get("action", "")).lower().strip()
    if expected != "unknown" and recommended:
        if expected in recommended or recommended in expected:
            base_score += 0.2
        else:
            base_score -= 0.1
            signal.penalties.append("wrong_action")

    signal.terminal_quality = round(min(1.0, max(-1.0, base_score)), 3)
    return signal


# -- Episode-level reward aggregation -----------------------------------------

@dataclass
class EpisodeRewardSummary:
    """Aggregated reward signals for a complete episode."""
    step_rewards: list[RewardSignal]
    terminal_reward: RewardSignal | None
    total_reward: float
    avg_step_reward: float
    penalty_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_reward": self.total_reward,
            "avg_step_reward": self.avg_step_reward,
            "num_steps": len(self.step_rewards),
            "penalty_counts": dict(self.penalty_counts),
            "step_rewards": [r.to_dict() for r in self.step_rewards],
            "terminal_reward": self.terminal_reward.to_dict() if self.terminal_reward else None,
        }


def summarize_episode_rewards(
    step_rewards: list[RewardSignal],
    terminal_reward: RewardSignal | None = None,
) -> EpisodeRewardSummary:
    """Aggregate step-level rewards into an episode summary."""
    all_rewards = list(step_rewards)
    if terminal_reward is not None:
        all_rewards.append(terminal_reward)

    totals = [r.total for r in all_rewards]
    total = round(sum(totals), 4)
    avg = round(total / len(totals), 4) if totals else 0.0

    # Count penalties across all steps
    penalty_counts: dict[str, int] = {}
    for r in all_rewards:
        for p in r.penalties:
            penalty_counts[p] = penalty_counts.get(p, 0) + 1

    return EpisodeRewardSummary(
        step_rewards=step_rewards,
        terminal_reward=terminal_reward,
        total_reward=total,
        avg_step_reward=avg,
        penalty_counts=penalty_counts,
    )
