"""
Deterministic state transitions for the late-order recovery environment.

Owns:
    - Validating whether a tool call is allowed given current state
    - Advancing the environment state after a valid tool execution
    - Updating discovery facts from tool results
    - Completing subgoals when their constituent tools finish
    - Recording failure events (malformed calls, dependency violations, etc.)
    - Determining terminal conditions

Does NOT own:
    - Tool implementations or execution (see runtime.tools)
    - Reward computation (see envs.rewards)
    - Prompt policy or agent orchestration (see runtime/)
    - Rollout or training concerns

Transition contract:
    Given (state, action) -> (new_state, step_info)
    where action is a tool call name + arguments + result,
    and step_info carries metadata for downstream reward computation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from src.envs.state import (
    LateOrderEnvState,
    Subgoal,
    SUBGOAL_ORDER,
    TOOL_COMPLETES_SUBGOAL,
    TOOL_TO_SUBGOAL,
)
from src.envs.state import TOOL_DEPENDENCIES


# -- Transition result --------------------------------------------------------

@dataclass
class StepResult:
    """The outcome of one environment step.

    Carries everything downstream consumers need (rewards, tracing, eval)
    without requiring them to diff two state snapshots.
    """
    valid: bool
    state: LateOrderEnvState
    tool_name: str = ""
    error_type: str | None = None  # "dependency_violation", "unknown_tool", "redundant_call", "malformed", "terminal"
    error_message: str | None = None
    subgoal_completed: Subgoal | None = None
    is_redundant: bool = False
    dependencies_satisfied: bool = True
    is_progress: bool = False  # did this step advance toward the goal?
    is_terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# Maximum episode length before forced termination
MAX_EPISODE_STEPS = 20


# -- Precondition checking ---------------------------------------------------

def check_preconditions(
    state: LateOrderEnvState,
    tool_name: str,
) -> tuple[bool, str | None, str | None]:
    """Check whether calling tool_name is valid given current state.

    Returns:
        (is_valid, error_type, error_message)
    """
    # Already terminal?
    if state.is_terminal:
        return False, "terminal", "Episode is already terminal."

    # Known tool?
    if tool_name not in TOOL_DEPENDENCIES:
        return False, "unknown_tool", f"Unknown tool: '{tool_name}'."

    # Dependency check
    required = TOOL_DEPENDENCIES.get(tool_name, set())
    called = state.tools_called_set
    missing = required - called
    if missing:
        return (
            False,
            "dependency_violation",
            f"Cannot call '{tool_name}': missing prerequisites {missing}. "
            f"Called so far: {sorted(called)}.",
        )

    return True, None, None


# -- State advancement from tool results ------------------------------------

def _update_discovery_facts(
    state: LateOrderEnvState,
    tool_name: str,
    tool_result: dict[str, Any],
) -> None:
    """Update state discovery facts based on a tool's result.

    This extracts reward-relevant facts from deterministic tool outputs
    so the environment knows what the agent has learned.
    """
    if "error" in tool_result:
        return

    if tool_name == "get_order":
        state.order_found = True

    elif tool_name == "get_shipment_status":
        state.shipment_status = tool_result.get("status")

    elif tool_name == "get_inventory":
        available = tool_result.get("available")
        if available is not None:
            state.source_dc_available = available
            state.source_dc_shortfall = max(0, state.qty - available)

    elif tool_name == "get_fulfillment_capacity":
        remaining = tool_result.get("remaining", 0)
        state.source_dc_capacity_ok = remaining >= state.qty

    elif tool_name == "find_alternate_inventory":
        matching = tool_result.get("matching_dcs", [])
        state.alternate_dcs_found = [
            dc["dc_id"] for dc in matching
            if dc.get("dc_id") != state.source_dc
        ]
        state.alternate_total_available = tool_result.get("total_available", 0)
        subs = tool_result.get("substitutes", [])
        state.substitute_skus_found = len(subs) if subs else 0

    elif tool_name == "get_transfer_eta":
        from_dc = tool_result.get("from_dc", "")
        if from_dc and from_dc not in state.transfer_etas_checked:
            state.transfer_etas_checked.append(from_dc)

    elif tool_name == "get_supplier_expedite_options":
        options = tool_result.get("options", [])
        state.supplier_options_found = len(options)

    elif tool_name == "score_recovery_options":
        state.recovery_options_scored = True
        best = tool_result.get("best_option")
        if best:
            state.recommendation_candidate = best

    elif tool_name == "recommend_action":
        state.recommendation_candidate = tool_result


def _check_subgoal_completion(
    state: LateOrderEnvState,
    tool_name: str,
) -> Subgoal | None:
    """Check if calling this tool completes a subgoal.

    A subgoal is complete when its final tool has been called AND all
    prior tools in the subgoal's workflow have also been called.
    """
    subgoal = TOOL_COMPLETES_SUBGOAL.get(tool_name)
    if subgoal is None:
        return None

    # Already completed?
    if subgoal in state.completed_subgoals:
        return None

    state.completed_subgoals.add(subgoal)
    return subgoal


# -- Main transition function ------------------------------------------------

def apply_tool_call(
    state: LateOrderEnvState,
    tool_name: str,
    tool_arguments: dict[str, Any],
    tool_result: dict[str, Any],
) -> StepResult:
    """Apply a valid tool call to the environment state.

    This is the main transition function. It assumes the tool has already
    been executed by the runtime and its result is available. The
    environment updates its internal state and returns a StepResult with
    metadata for reward computation.

    Args:
        state: Current environment state (will be mutated).
        tool_name: Name of the tool that was called.
        tool_arguments: Arguments the agent provided.
        tool_result: Deterministic result from the tool.

    Returns:
        StepResult with transition metadata.
    """
    state.total_steps += 1

    # Check preconditions
    ok, err_type, err_msg = check_preconditions(state, tool_name)
    if not ok:
        if err_type == "dependency_violation":
            state.dependency_violation_count += 1
            state.invalid_action_count += 1
        elif err_type == "unknown_tool":
            state.invalid_action_count += 1
        return StepResult(
            valid=False,
            state=state,
            tool_name=tool_name,
            error_type=err_type,
            error_message=err_msg,
            dependencies_satisfied=(err_type != "dependency_violation"),
            is_terminal=state.is_terminal,
        )

    # Check for redundancy
    is_redundant = tool_name in state.tool_call_counts and state.tool_call_counts[tool_name] > 0
    if is_redundant:
        state.redundant_call_count += 1

    # Update discovery facts (skips internally when tool_result has "error")
    _update_discovery_facts(state, tool_name, tool_result)

    # Business error: the call was structurally valid (correct name, met
    # dependencies) but the tool returned an error.  Don't record it in
    # tools_called so downstream dependencies remain unsatisfied.
    if "error" in tool_result:
        return StepResult(
            valid=True,
            state=state,
            tool_name=tool_name,
            is_redundant=is_redundant,
            dependencies_satisfied=True,
            is_progress=False,
            is_terminal=state.is_terminal,
            metadata={
                "business_error": True,
                "subgoals_done": len(state.completed_subgoals),
                "total_subgoals": len(SUBGOAL_ORDER),
            },
        )

    # Record the successful call
    state.tools_called.append(tool_name)
    state.tool_call_counts[tool_name] = state.tool_call_counts.get(tool_name, 0) + 1

    # Check subgoal completion
    subgoal_completed = _check_subgoal_completion(state, tool_name)

    # Determine if this step represents forward progress
    is_progress = (
        subgoal_completed is not None
        or (not is_redundant and tool_name in TOOL_TO_SUBGOAL)
    )

    return StepResult(
        valid=True,
        state=state,
        tool_name=tool_name,
        subgoal_completed=subgoal_completed,
        is_redundant=is_redundant,
        dependencies_satisfied=True,
        is_progress=is_progress,
        is_terminal=state.is_terminal,
        metadata={
            "call_count": state.tool_call_counts[tool_name],
            "subgoals_done": len(state.completed_subgoals),
            "total_subgoals": len(SUBGOAL_ORDER),
        },
    )


def record_invalid_action(
    state: LateOrderEnvState,
    error_type: str,
    error_message: str,
) -> StepResult:
    """Record a malformed or otherwise invalid action (not a dependency violation).

    Use this for malformed JSON, schema errors, or other parse failures
    that aren't tool-call-specific.
    """
    state.total_steps += 1
    state.invalid_action_count += 1
    if error_type == "malformed":
        state.malformed_call_count += 1

    return StepResult(
        valid=False,
        state=state,
        error_type=error_type,
        error_message=error_message,
        is_terminal=state.is_terminal,
    )


def record_repair_attempt(
    state: LateOrderEnvState,
    succeeded: bool,
) -> None:
    """Record that a fallback repair was attempted."""
    state.repair_attempt_count += 1
    if succeeded:
        state.repair_success_count += 1


def record_reject(state: LateOrderEnvState) -> None:
    """Record that a fallback repair was rejected."""
    state.reject_count += 1


def apply_terminal(
    state: LateOrderEnvState,
    reason: str,
    final_answer: dict[str, Any] | None = None,
) -> StepResult:
    """Mark the episode as terminal.

    Args:
        state: Current environment state.
        reason: Why the episode ended ("final_answer", "max_iterations", "error").
        final_answer: The agent's final recommendation, if any.
    """
    state.is_terminal = True
    state.terminal_reason = reason
    state.final_answer = final_answer

    return StepResult(
        valid=True,
        state=state,
        is_terminal=True,
        metadata={"terminal_reason": reason},
    )


def should_force_terminate(state: LateOrderEnvState) -> bool:
    """Check if the episode should be forcibly terminated.

    Returns True if the episode has exceeded the step limit.
    """
    return state.total_steps >= MAX_EPISODE_STEPS
