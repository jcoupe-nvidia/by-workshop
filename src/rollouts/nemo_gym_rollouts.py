"""
NeMo Gym rollout collection adapter.

Bridges the rollout layer with NeMo Gym's training-time infrastructure:

    - Training-time execution pipeline (validate → repair → reject → execute → record)
    - Converts enriched Episodes to NemoGymResultRows for reward profiling
    - Builds RolloutCollectionConfig for NeMo Gym rollout collection runs
    - Runs reward profiling on collected rollouts via RewardProfiler
    - Durable trajectory export from NeMo Gym sessions

Owns:
    - Training-time agent-action execution pipeline
    - Enriched episode -> NeMo Gym result row conversion (via envs.nemo_gym_adapter)
    - RolloutCollectionConfig construction for the late-order-recovery task
    - RewardProfiler integration for post-collection metrics
    - Session episode persistence through canonical serializers

Does NOT own:
    - Environment state or reward computation (see envs/)
    - NeMo Gym resource server protocol (see envs.nemo_gym_adapter)
    - Episode types or enrichment (see rollouts.trace_types, rollouts.episode_runner)
    - Training dataset construction (see training/)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_gym.rollout_collection import (
    RolloutCollectionConfig,
    RolloutCollectionHelper,
)
from nemo_gym.reward_profile import RewardProfiler

from src.envs.nemo_gym_adapter import (
    NemoGymResultRow,
    build_rollout_input_row,
    episode_to_nemo_gym_row,
    save_nemo_gym_inputs_jsonl,
    save_nemo_gym_rows_jsonl,
)
from src.rollouts.episode_runner import EnrichedEpisodeResult
from src.rollouts.trace_types import Episode, EventType, ToolCallPayload


# ---------------------------------------------------------------------------
# Training-time execution pipeline
# ---------------------------------------------------------------------------
# This section owns the canonical validate → repair → reject → execute →
# record loop for NeMo Gym training-time rollout collection. The NeMo Gym
# resource server in envs/ delegates here so that envs/ stays focused on
# environment validation surfaces while rollouts/ owns execution plumbing.
#
# NOTE: This pipeline re-uses validate_tool_call, try_repair, and
# FallbackAction from src.runtime.* but orchestrates them with real-time
# environment interaction (env.step, env.record_invalid, etc.) which the
# runtime's interactive agent loop does not do. The interactive runtime
# attaches rewards post-hoc via episode_runner.enrich_episode(). If the
# core validate → repair → reject logic changes in runtime, this pipeline
# must be updated to match. A shared runtime-owned helper is a future
# option but deferred to avoid premature abstraction in a pedagogical repo.


@dataclass
class AgentAction:
    """A single agent action to process through the execution pipeline.

    Attributes:
        action_type: "function_call" for tool invocations, "message" for terminal.
        tool_name: Tool name (for function_call actions).
        arguments: Tool arguments dict (for function_call actions).
        content: Message content (for message actions).
    """
    action_type: str  # "function_call" or "message"
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    content: str = ""


def process_agent_action(
    action: AgentAction,
    env: Any,  # LateOrderRecoveryEnv
    recorder: Any,  # EpisodeRecorder
) -> float:
    """Process one agent action through the canonical execution pipeline.

    Runs the same validate → repair → reject → execute → record loop that
    the interactive runtime uses, ensuring training-time traces have the
    same semantics and artifact format as interactive traces.

    Args:
        action: The agent action to process.
        env: The LateOrderRecoveryEnv managing task state.
        recorder: The EpisodeRecorder capturing canonical events.

    Returns:
        The accumulated reward for this action.
    """
    from src.runtime.tools import TOOL_REGISTRY
    from src.runtime.schemas import validate_tool_call
    from src.runtime.fallbacks import try_repair, FallbackAction

    if action.action_type == "function_call":
        return _process_function_call(
            action, env, recorder, TOOL_REGISTRY, validate_tool_call,
            try_repair, FallbackAction,
        )
    elif action.action_type == "message":
        return _process_message(action, env, recorder)
    return 0.0


def _process_function_call(
    action: AgentAction,
    env: Any,
    recorder: Any,
    tool_registry: dict,
    validate_fn: Any,
    repair_fn: Any,
    fallback_action_cls: Any,
) -> float:
    """Execute the validate → repair → reject → execute → record loop."""
    reward = 0.0
    tool_name = action.tool_name
    arguments = action.arguments

    raw_output = json.dumps({
        "tool_call": {
            "name": tool_name,
            "arguments": arguments,
        }
    })

    # Run the canonical validate → repair → reject loop
    result = validate_fn(raw_output, tool_registry)
    was_repaired = False

    if hasattr(result, "error_type"):
        # Validation failed — attempt repair
        fb = repair_fn(raw_output, tool_registry)
        if fb.action == fallback_action_cls.REPAIRED and fb.repaired:
            recorder.record_repair_attempt(
                original_output=raw_output,
                repaired_output=fb.repaired,
                repairs_applied=fb.repairs_applied,
                succeeded=True,
            )
            env.record_fallback_repair(succeeded=True)
            result = validate_fn(fb.repaired, tool_registry)
            was_repaired = True
        elif fb.action == fallback_action_cls.REJECTED:
            recorder.record_repair_attempt(
                original_output=raw_output,
                repaired_output=None,
                repairs_applied=fb.repairs_applied,
                succeeded=False,
            )
            recorder.record_reject(
                reason=fb.rejection_reason or "Unrecoverable output",
                raw_model_output=raw_output,
                repairs_attempted=fb.repairs_applied,
            )
            env.record_fallback_repair(succeeded=False)
            env.record_fallback_reject()
            step = env.record_invalid(
                "rejected",
                fb.rejection_reason or "Unrecoverable output",
            )
            step_idx = len(env._step_rewards) - 1
            step_reward = env.get_step_reward(step_idx)
            if step_reward:
                reward += step_reward.total
            return reward

    # If still invalid after repair, record and return
    if hasattr(result, "error_type"):
        recorder.record_validation_error(
            error_type=result.error_type,
            message=result.message,
            raw_model_output=raw_output,
        )
        step = env.record_invalid(result.error_type, result.message)
        step_idx = len(env._step_rewards) - 1
        step_reward = env.get_step_reward(step_idx)
        if step_reward:
            reward += step_reward.total
        return reward

    # Valid tool call — record, execute, and step
    valid_name = result.tool_name
    valid_args = result.arguments
    recorder.record_tool_call(
        tool_name=valid_name,
        arguments=valid_args,
        thought=getattr(result, "thought", None),
        raw_model_output=raw_output,
    )
    recorder.increment_model_calls()

    if valid_name in tool_registry:
        fn, _params, _desc = tool_registry[valid_name]
        try:
            tool_result = fn(**valid_args)
        except Exception:
            tool_result = {"error": "execution_failed"}
        recorder.record_tool_result(valid_name, tool_result)
        env.step(valid_name, valid_args, tool_result, was_repaired=was_repaired)
        step_idx = len(env._step_rewards) - 1
        step_reward = env.get_step_reward(step_idx)
        if step_reward:
            reward += step_reward.total

    return reward


def _process_message(
    action: AgentAction,
    env: Any,
    recorder: Any,
) -> float:
    """Process a terminal message action."""
    reward = 0.0
    content = action.content

    final_answer = None
    try:
        final_answer = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        final_answer = {"action": "unknown", "raw": str(content)}

    recorder.record_terminal("final_answer", final_answer=final_answer)
    env.terminate("final_answer", final_answer)
    terminal_reward = env.get_terminal_reward()
    if terminal_reward:
        reward += terminal_reward.total

    return reward


def process_agent_actions(
    actions: list[AgentAction],
    env: Any,
    recorder: Any,
) -> float:
    """Process a sequence of agent actions, accumulating reward.

    Args:
        actions: Ordered list of AgentActions to process.
        env: The LateOrderRecoveryEnv managing task state.
        recorder: The EpisodeRecorder capturing canonical events.

    Returns:
        Total accumulated reward across all actions.
    """
    total = 0.0
    for action in actions:
        total += process_agent_action(action, env, recorder)
    return total


# ---------------------------------------------------------------------------
# Per-event reward attachment
# ---------------------------------------------------------------------------


def _attach_event_rewards(episode: Episode, env: Any) -> None:
    """Populate Event.reward on episode events from the environment's step rewards.

    After environment-backed execution, the env holds dense step rewards
    and a terminal reward. This function walks the episode events and
    attaches the corresponding reward to each TOOL_CALL, VALIDATION_ERROR,
    and TERMINAL_OUTCOME event, matching the same annotation contract that
    enrich_episode() provides for the interactive runtime path.

    Mutates episode.events in place.
    """
    step_idx = 0
    step_rewards = env._step_rewards
    terminal_reward = env._terminal_reward

    for event in episode.events:
        if event.event_type == EventType.TOOL_CALL:
            payload = event.payload
            if isinstance(payload, ToolCallPayload) and step_idx < len(step_rewards):
                event.reward = step_rewards[step_idx].total
                step_idx += 1

        elif event.event_type == EventType.TOOL_VALIDATION_ERROR:
            if step_idx < len(step_rewards):
                event.reward = step_rewards[step_idx].total
                step_idx += 1

        elif event.event_type == EventType.TERMINAL_OUTCOME:
            if terminal_reward is not None:
                event.reward = terminal_reward.total


# ---------------------------------------------------------------------------
# Environment-backed rollout collection
# ---------------------------------------------------------------------------
# These functions build scripted action sequences and run them through
# the execution pipeline above, producing episodes that went through
# the same NeMo Gym environment-backed execution path as real training
# rollouts. This closes the gap identified in the code review: the GRPO
# path now exercises the documented NVIDIA ownership split rather than
# training on canned trajectories.


def _build_successful_actions() -> list[AgentAction]:
    """Build the scripted action sequence for a successful SO-10482 trajectory.

    Returns AgentAction objects suitable for process_agent_actions().
    Uses the same tool sequence as scripted_traces.build_successful_episode()
    but as AgentAction structs rather than pre-built Event objects.
    """
    from src.runtime.tools import TOOL_REGISTRY

    # Execute tools to get real results for downstream actions
    def _call(name: str, args: dict) -> dict:
        fn, _, _ = TOOL_REGISTRY[name]
        return fn(**args)

    actions: list[AgentAction] = []

    # Skill 1: diagnose_order_risk
    actions.append(AgentAction("function_call", "get_order", {"order_id": "SO-10482"}))
    actions.append(AgentAction("function_call", "get_shipment_status", {"order_id": "SO-10482"}))

    # Skill 2: assess_primary_fulfillment
    actions.append(AgentAction("function_call", "get_inventory", {"sku": "SKU-4090", "dc_id": "DC-WEST-01"}))
    actions.append(AgentAction("function_call", "get_fulfillment_capacity", {"dc_id": "DC-WEST-01", "date": "2026-04-18"}))

    # Skill 3: evaluate_alternate_recovery_paths
    actions.append(AgentAction("function_call", "find_alternate_inventory", {"sku": "SKU-4090", "region": "ALL"}))
    east_transfer = _call("get_transfer_eta", {"from_dc": "DC-EAST-02", "to_dc": "DC-WEST-01", "sku": "SKU-4090", "qty": 900})
    actions.append(AgentAction("function_call", "get_transfer_eta", {"from_dc": "DC-EAST-02", "to_dc": "DC-WEST-01", "sku": "SKU-4090", "qty": 900}))
    actions.append(AgentAction("function_call", "get_supplier_expedite_options", {"sku": "SKU-4090", "qty": 900}))

    # Skill 4: synthesize_recommendation (includes substitute path)
    options = [
        {"source": "DC-EAST-02", "description": "dc_transfer from DC-EAST-02",
         "path_type": "dc_transfer", "lead_days": east_transfer["lead_days"],
         "cost_per_unit": east_transfer["cost_per_unit"],
         "total_cost": east_transfer["total_cost"],
         "feasible": east_transfer["feasible"], "covers_full_qty": True},
        {"source": "supplier:GlobalChip Express",
         "description": "supplier_expedite from GlobalChip Express",
         "path_type": "supplier_expedite", "lead_days": 7,
         "cost_per_unit": 8.00, "total_cost": 7200.0,
         "feasible": True, "covers_full_qty": True},
        {"source": "supplier:FastSemi Direct",
         "description": "supplier_expedite from FastSemi Direct",
         "path_type": "supplier_expedite", "lead_days": 5,
         "cost_per_unit": 12.00, "total_cost": 10800.0,
         "feasible": True, "covers_full_qty": True},
        {"source": "substitute:SKU-4090-B@DC-WEST-01",
         "description": "substitute SKU-4090-B at DC-WEST-01",
         "path_type": "substitute", "lead_days": 0,
         "cost_per_unit": 0.0, "total_cost": 0.0,
         "feasible": False, "covers_full_qty": False},
    ]
    scored = _call("score_recovery_options", {"options": options, "objective": "minimize_delay"})
    actions.append(AgentAction("function_call", "score_recovery_options", {"options": options, "objective": "minimize_delay"}))

    rec = _call("recommend_action", {"context": {
        "best_option": scored["best_option"],
        "order": {"order_id": "SO-10482", "sku": "SKU-4090", "qty": 1200, "committed_date": "2026-04-18"},
        "objective": "minimize_delay",
    }})
    actions.append(AgentAction("function_call", "recommend_action", {"context": {
        "best_option": scored["best_option"],
        "order": {"order_id": "SO-10482", "sku": "SKU-4090", "qty": 1200, "committed_date": "2026-04-18"},
        "objective": "minimize_delay",
    }}))

    # Terminal
    import json
    final_answer = {
        "action": rec["action"],
        "rationale": rec["rationale"],
        "expected_delivery": rec["expected_delivery"],
        "meets_committed_date": rec["meets_committed_date"],
        "confidence": rec["confidence"],
    }
    actions.append(AgentAction("message", content=json.dumps(final_answer)))

    return actions


def _build_repair_actions() -> list[AgentAction]:
    """Build a scripted action sequence with repairs and a rejection.

    Mirrors scripted_traces.build_repair_episode(): includes a rejected
    plain-text output and two repaired tool calls. Unlike the scripted
    traces approach, the rejection and repair handling happens in the
    execution pipeline (process_agent_action) rather than being
    pre-baked into Event objects.

    Note: Since the execution pipeline handles repairs at the
    validate→repair level, we inject deliberate errors by using typo
    tool names that the fuzzy matcher can correct.
    """
    from src.runtime.tools import TOOL_REGISTRY
    import json

    def _call(name: str, args: dict) -> dict:
        fn, _, _ = TOOL_REGISTRY[name]
        return fn(**args)

    actions: list[AgentAction] = []

    # Step 1: get_order (clean)
    actions.append(AgentAction("function_call", "get_order", {"order_id": "SO-10482"}))

    # Step 2: get_shipment_status with typo -> fuzzy repair
    # "get_shipmnt_status" is edit distance 1 from "get_shipment_status"
    actions.append(AgentAction("function_call", "get_shipmnt_status", {"order_id": "SO-10482"}))

    # Steps 3-5: clean calls
    actions.append(AgentAction("function_call", "get_inventory", {"sku": "SKU-4090", "dc_id": "DC-WEST-01"}))
    actions.append(AgentAction("function_call", "get_fulfillment_capacity", {"dc_id": "DC-WEST-01", "date": "2026-04-18"}))

    # Step 6: find_alternate_inventory with typo -> fuzzy repair
    actions.append(AgentAction("function_call", "find_alternat_inventory", {"sku": "SKU-4090", "region": "ALL"}))

    # Steps 7-8: clean calls
    east_transfer = _call("get_transfer_eta", {"from_dc": "DC-EAST-02", "to_dc": "DC-WEST-01", "sku": "SKU-4090", "qty": 900})
    actions.append(AgentAction("function_call", "get_transfer_eta", {"from_dc": "DC-EAST-02", "to_dc": "DC-WEST-01", "sku": "SKU-4090", "qty": 900}))
    actions.append(AgentAction("function_call", "get_supplier_expedite_options", {"sku": "SKU-4090", "qty": 900}))

    # Steps 9-10: score and recommend (includes substitute path)
    options = [
        {"source": "DC-EAST-02", "description": "dc_transfer from DC-EAST-02",
         "path_type": "dc_transfer", "lead_days": east_transfer["lead_days"],
         "cost_per_unit": east_transfer["cost_per_unit"],
         "total_cost": east_transfer["total_cost"],
         "feasible": east_transfer["feasible"], "covers_full_qty": True},
        {"source": "supplier:GlobalChip Express",
         "description": "supplier_expedite from GlobalChip Express",
         "path_type": "supplier_expedite", "lead_days": 7,
         "cost_per_unit": 8.00, "total_cost": 7200.0,
         "feasible": True, "covers_full_qty": True},
        {"source": "substitute:SKU-4090-B@DC-WEST-01",
         "description": "substitute SKU-4090-B at DC-WEST-01",
         "path_type": "substitute", "lead_days": 0,
         "cost_per_unit": 0.0, "total_cost": 0.0,
         "feasible": False, "covers_full_qty": False},
    ]
    scored = _call("score_recovery_options", {"options": options, "objective": "minimize_delay"})
    actions.append(AgentAction("function_call", "score_recovery_options", {"options": options, "objective": "minimize_delay"}))

    rec = _call("recommend_action", {"context": {
        "best_option": scored["best_option"],
        "order": {"order_id": "SO-10482", "sku": "SKU-4090", "qty": 1200, "committed_date": "2026-04-18"},
        "objective": "minimize_delay",
    }})
    actions.append(AgentAction("function_call", "recommend_action", {"context": {
        "best_option": scored["best_option"],
        "order": {"order_id": "SO-10482", "sku": "SKU-4090", "qty": 1200, "committed_date": "2026-04-18"},
        "objective": "minimize_delay",
    }}))

    # Terminal
    final_answer = {
        "action": rec["action"],
        "rationale": rec["rationale"],
        "expected_delivery": rec["expected_delivery"],
        "meets_committed_date": rec["meets_committed_date"],
        "confidence": rec["confidence"],
    }
    actions.append(AgentAction("message", content=json.dumps(final_answer)))

    return actions


def collect_environment_backed_rollout(
    actions: list[AgentAction],
    order_id: str = "SO-10482",
    task_prompt: str = "Investigate order SO-10482",
) -> EnrichedEpisodeResult:
    """Collect one episode by running actions through environment-backed execution.

    Each action goes through the canonical validate → repair → reject →
    execute → record pipeline with the environment computing rewards in
    real-time. This is the same execution path that the NeMo Gym
    resource server's verify() uses during training-time rollout
    collection.

    Args:
        actions: Ordered list of AgentActions to execute.
        order_id: The order ID for this episode.
        task_prompt: The task prompt for the episode.

    Returns:
        EnrichedEpisodeResult with reward-annotated episode.
    """
    from src.envs.late_order_env import LateOrderRecoveryEnv
    from src.runtime.tracing import EpisodeRecorder

    env = LateOrderRecoveryEnv()
    env.reset(order_id)

    recorder = EpisodeRecorder(
        task_id=order_id,
        task_prompt=task_prompt,
        model_id="nemo-gym-rollout",
    )
    recorder.record_user_task(task_prompt)

    # Run all actions through the execution pipeline
    process_agent_actions(actions, env, recorder)

    # Build the canonical episode from the recorder
    episode = recorder.build_episode()
    episode.env_state_init = env.get_initial_state_snapshot()

    # Get reward summary directly from the environment (real-time, not post-hoc)
    reward_summary = env.get_episode_reward_summary()
    episode.metrics.total_reward = round(reward_summary.total_reward, 4)

    # Attach per-event rewards from the environment so that downstream
    # exporters (episode_to_training_trajectory, episode_to_art_trajectory)
    # see the same dense reward annotations as the enrich_episode() path.
    _attach_event_rewards(episode, env)

    return EnrichedEpisodeResult(
        episode=episode,
        reward_summary=reward_summary,
        env_final_state=env.get_state_snapshot(),
    )


def _build_reject_actions() -> list[AgentAction]:
    """Build a scripted action sequence with an unrecoverable reject and a dependency violation.

    This produces an episode with:
    - A completely unknown tool name that cannot be fuzzy-matched (rejected)
    - A dependency violation (calling get_transfer_eta before find_alternate_inventory)
    - Normal recovery completing the task after the failures

    Unlike _build_repair_actions() which only generates typo-repairable calls,
    this builder produces concrete rejects and dependency violations that
    exercise the robustness-stage training signals.
    """
    from src.runtime.tools import TOOL_REGISTRY
    import json

    def _call(name: str, args: dict) -> dict:
        fn, _, _ = TOOL_REGISTRY[name]
        return fn(**args)

    actions: list[AgentAction] = []

    # Step 1: get_order (clean)
    actions.append(AgentAction("function_call", "get_order", {"order_id": "SO-10482"}))

    # Step 2: completely unknown tool -> rejected (no fuzzy match possible)
    actions.append(AgentAction(
        "function_call", "analyze_demand_forecast", {"order_id": "SO-10482"}
    ))

    # Step 3: get_shipment_status (clean, continuing after reject)
    actions.append(AgentAction("function_call", "get_shipment_status", {"order_id": "SO-10482"}))

    # Step 4-5: clean calls
    actions.append(AgentAction("function_call", "get_inventory", {"sku": "SKU-4090", "dc_id": "DC-WEST-01"}))
    actions.append(AgentAction("function_call", "get_fulfillment_capacity", {"dc_id": "DC-WEST-01", "date": "2026-04-18"}))

    # Step 6: find_alternate_inventory (clean)
    actions.append(AgentAction("function_call", "find_alternate_inventory", {"sku": "SKU-4090", "region": "ALL"}))

    # Step 7-8: clean calls
    east_transfer = _call("get_transfer_eta", {"from_dc": "DC-EAST-02", "to_dc": "DC-WEST-01", "sku": "SKU-4090", "qty": 900})
    actions.append(AgentAction("function_call", "get_transfer_eta", {"from_dc": "DC-EAST-02", "to_dc": "DC-WEST-01", "sku": "SKU-4090", "qty": 900}))
    actions.append(AgentAction("function_call", "get_supplier_expedite_options", {"sku": "SKU-4090", "qty": 900}))

    # Steps 9-10: score and recommend (includes substitute)
    options = [
        {"source": "DC-EAST-02", "description": "dc_transfer from DC-EAST-02",
         "path_type": "dc_transfer", "lead_days": east_transfer["lead_days"],
         "cost_per_unit": east_transfer["cost_per_unit"],
         "total_cost": east_transfer["total_cost"],
         "feasible": east_transfer["feasible"], "covers_full_qty": True},
        {"source": "supplier:GlobalChip Express",
         "description": "supplier_expedite from GlobalChip Express",
         "path_type": "supplier_expedite", "lead_days": 7,
         "cost_per_unit": 8.00, "total_cost": 7200.0,
         "feasible": True, "covers_full_qty": True},
        {"source": "substitute:SKU-4090-B@DC-WEST-01",
         "description": "substitute SKU-4090-B at DC-WEST-01",
         "path_type": "substitute", "lead_days": 0,
         "cost_per_unit": 0.0, "total_cost": 0.0,
         "feasible": False, "covers_full_qty": False},
    ]
    scored = _call("score_recovery_options", {"options": options, "objective": "minimize_delay"})
    actions.append(AgentAction("function_call", "score_recovery_options", {"options": options, "objective": "minimize_delay"}))

    rec = _call("recommend_action", {"context": {
        "best_option": scored["best_option"],
        "order": {"order_id": "SO-10482", "sku": "SKU-4090", "qty": 1200, "committed_date": "2026-04-18"},
        "objective": "minimize_delay",
    }})
    actions.append(AgentAction("function_call", "recommend_action", {"context": {
        "best_option": scored["best_option"],
        "order": {"order_id": "SO-10482", "sku": "SKU-4090", "qty": 1200, "committed_date": "2026-04-18"},
        "objective": "minimize_delay",
    }}))

    # Terminal
    final_answer = {
        "action": rec["action"],
        "rationale": rec["rationale"],
        "expected_delivery": rec["expected_delivery"],
        "meets_committed_date": rec["meets_committed_date"],
        "confidence": rec["confidence"],
    }
    actions.append(AgentAction("message", content=json.dumps(final_answer)))

    return actions


def collect_nemo_gym_rollouts(
    num_rollouts: int = 4,
    include_repairs: bool = True,
) -> list[EnrichedEpisodeResult]:
    """Collect enriched episodes via NeMo Gym environment-backed execution.

    Builds scripted action sequences and runs them through the
    environment-backed execution pipeline (the same path used by the
    NeMo Gym resource server's verify()). This ensures the GRPO
    training path exercises the documented NVIDIA ownership split:

        - Environment validates actions and computes rewards in real-time
        - EpisodeRecorder captures canonical events (including repairs/rejects)
        - Episodes carry env_state_init and durable reward metadata

    Args:
        num_rollouts: Total number of episodes to collect.
        include_repairs: Whether to include repair episodes for diversity.

    Returns:
        List of EnrichedEpisodeResults from environment-backed execution.
    """
    results: list[EnrichedEpisodeResult] = []

    for i in range(num_rollouts):
        if include_repairs and i % 3 == 1:
            actions = _build_repair_actions()
            failure_mode = "repair"
        elif include_repairs and i % 3 == 2:
            actions = _build_reject_actions()
            failure_mode = "reject"
        else:
            actions = _build_successful_actions()
            failure_mode = "clean"

        result = collect_environment_backed_rollout(actions)
        result.episode.metadata["failure_mode"] = failure_mode
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# NeMo Gym resource-server-backed rollout collection (primary path)
# ---------------------------------------------------------------------------
# This section exercises the documented NeMo Gym ownership split: the
# resource server's seed_session() + verify() protocol is the first-class
# training-time interface. The scripted harness above is the fallback.


def collect_via_resource_server(
    actions: list[AgentAction],
    order_id: str = "SO-10482",
    task_prompt: str = "Investigate order SO-10482",
) -> EnrichedEpisodeResult:
    """Collect one episode through the NeMo Gym resource server protocol.

    Exercises the full seed_session() → verify() loop by constructing
    NeMo Gym request objects and calling the resource server directly.
    This is the same protocol that NeMo Gym's RolloutCollectionHelper
    uses over HTTP during distributed training, but run in-process for
    the workshop demo.

    Args:
        actions: Ordered list of AgentActions to send as verify() requests.
        order_id: The order ID for this episode.
        task_prompt: The task prompt for the episode.

    Returns:
        EnrichedEpisodeResult from the resource server session.
    """
    import asyncio
    from src.envs.nemo_gym_adapter import (
        LateOrderResourceServer,
        _sessions,
    )
    from nemo_gym.base_resources_server import (
        BaseSeedSessionRequest,
    )

    # Use __new__ to skip Pydantic validation — the server's seed_session()
    # and verify() methods don't need config/server_client fields for
    # in-process use. The same pattern is used in tests.
    server = LateOrderResourceServer.__new__(LateOrderResourceServer)

    # Step 1: seed_session — initialize a fresh environment
    # Capture the session_id from the _sessions dict since
    # BaseSeedSessionResponse doesn't carry it as a field.
    pre_keys = set(_sessions.keys())
    seed_req = BaseSeedSessionRequest()
    asyncio.run(server.seed_session(seed_req))
    new_keys = set(_sessions.keys()) - pre_keys
    session_id = new_keys.pop()

    # Step 2: verify — send each action as a NeMo Gym response
    for action in actions:
        verify_req = _build_verify_request_from_action(action, session_id)
        asyncio.run(server.verify(verify_req))

    # Step 3: Retrieve the episode and reward from the session
    session = _sessions[session_id]
    episode = session.recorder.build_episode()
    episode.env_state_init = session.env.get_initial_state_snapshot()

    reward_summary = session.env.get_episode_reward_summary()
    episode.metrics.total_reward = round(reward_summary.total_reward, 4)

    # Attach per-event rewards from the environment
    _attach_event_rewards(episode, session.env)

    result = EnrichedEpisodeResult(
        episode=episode,
        reward_summary=reward_summary,
        env_final_state=session.env.get_state_snapshot(),
    )

    # Clean up session
    _sessions.pop(session_id, None)

    return result


def _build_verify_request_from_action(action: AgentAction, session_id: str):
    """Build a BaseVerifyRequest from an AgentAction for resource server verify().

    Constructs the NeMo Gym request format that the resource server expects,
    using the same NeMo Gym types as the actual training-time path.
    """
    import json as _json
    import uuid as _uuid
    from nemo_gym.openai_utils import (
        NeMoGymResponse,
        NeMoGymResponseCreateParamsNonStreaming,
        NeMoGymResponseFunctionToolCall,
        NeMoGymResponseOutputMessage,
        NeMoGymResponseOutputText,
    )
    from nemo_gym.base_resources_server import BaseVerifyRequest

    output_items = []

    if action.action_type == "function_call":
        output_items.append(NeMoGymResponseFunctionToolCall(
            arguments=_json.dumps(action.arguments),
            call_id=f"call_{_uuid.uuid4().hex[:8]}",
            name=action.tool_name,
        ))
    elif action.action_type == "message":
        output_items.append(NeMoGymResponseOutputMessage(
            id=f"msg_{_uuid.uuid4().hex[:8]}",
            content=[NeMoGymResponseOutputText(
                text=action.content, annotations=[],
            )],
        ))

    response = NeMoGymResponse(
        id=f"resp_{_uuid.uuid4().hex[:8]}",
        created_at=1000.0,
        model="nemo-gym-rollout",
        object="response",
        output=output_items,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    params = NeMoGymResponseCreateParamsNonStreaming(input="rollout")
    req = BaseVerifyRequest(
        responses_create_params=params,
        response=response,
    )
    # Attach session_id for multi-session lookup
    object.__setattr__(req, "session_id", session_id)
    return req


def collect_server_backed_rollouts(
    num_rollouts: int = 4,
    include_repairs: bool = True,
) -> list[EnrichedEpisodeResult]:
    """Collect enriched episodes via the NeMo Gym resource server protocol.

    This is the primary rollout collection path. It exercises the full
    NeMo Gym ownership split by calling seed_session() + verify() on the
    LateOrderResourceServer rather than bypassing the protocol.

    Builds scripted action sequences and routes them through the resource
    server, producing the same enriched episodes as the scripted fallback
    but through the documented training-time integration surface.

    Args:
        num_rollouts: Total number of episodes to collect.
        include_repairs: Whether to include repair/reject episodes.

    Returns:
        List of EnrichedEpisodeResults from resource-server-backed execution.
    """
    results: list[EnrichedEpisodeResult] = []

    for i in range(num_rollouts):
        if include_repairs and i % 3 == 1:
            actions = _build_repair_actions()
            failure_mode = "repair"
        elif include_repairs and i % 3 == 2:
            actions = _build_reject_actions()
            failure_mode = "reject"
        else:
            actions = _build_successful_actions()
            failure_mode = "clean"

        result = collect_via_resource_server(actions)
        result.episode.metadata["failure_mode"] = failure_mode
        result.episode.metadata["collection_path"] = "nemo_gym_resource_server"
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# NeMo Gym result row conversion
# ---------------------------------------------------------------------------


def enriched_to_nemo_gym_row(
    result: EnrichedEpisodeResult,
    agent_name: str = "late-order-recovery-agent",
) -> NemoGymResultRow:
    """Convert an enriched episode result to a NeMo Gym result row.

    This is the bridge between the rollout layer (enriched episodes)
    and NeMo Gym's reward profiling infrastructure.
    """
    return episode_to_nemo_gym_row(
        episode=result.episode,
        reward_summary=result.reward_summary,
        agent_name=agent_name,
    )


def enriched_batch_to_nemo_gym_rows(
    results: list[EnrichedEpisodeResult],
    agent_name: str = "late-order-recovery-agent",
) -> list[NemoGymResultRow]:
    """Convert a batch of enriched results to NeMo Gym result rows."""
    return [enriched_to_nemo_gym_row(r, agent_name) for r in results]


def build_collection_config(
    input_jsonl_path: str,
    output_jsonl_path: str,
    agent_name: str = "late-order-recovery-agent",
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
    num_samples_in_parallel: int | None = None,
    num_repeats: int | None = None,
    limit: int | None = None,
) -> RolloutCollectionConfig:
    """Build a NeMo Gym RolloutCollectionConfig for the late-order task.

    This config is consumed by RolloutCollectionHelper.run_from_config()
    to orchestrate rollout collection against the resource server.

    Args:
        input_jsonl_path: Path to JSONL file with task input rows.
        output_jsonl_path: Path for collected rollout output.
        agent_name: Agent server name for routing.
        temperature: Sampling temperature.
        max_output_tokens: Max tokens per model response.
        num_samples_in_parallel: Concurrent rollout limit.
        num_repeats: Repeat each task N times for variance.
        limit: Max tasks to process.
    """
    return RolloutCollectionConfig(
        input_jsonl_fpath=input_jsonl_path,
        output_jsonl_fpath=output_jsonl_path,
        agent_name=agent_name,
        responses_create_params={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        },
        num_samples_in_parallel=num_samples_in_parallel,
        num_repeats=num_repeats,
        limit=limit,
    )


def prepare_rollout_inputs(
    order_ids: list[str],
    task_prompts: list[str],
    output_path: str,
    agent_name: str = "late-order-recovery-agent",
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
) -> str:
    """Prepare a NeMo Gym input JSONL file for rollout collection.

    Writes one row per order to the output path. Returns the path.

    Args:
        order_ids: List of order IDs to investigate.
        task_prompts: Corresponding task prompts.
        output_path: Path to write the input JSONL.
        agent_name: Agent server name.
        temperature: Sampling temperature.
        max_output_tokens: Max tokens per response.
    """
    rows = [
        build_rollout_input_row(
            order_id=oid,
            task_prompt=prompt,
            agent_name=agent_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        for oid, prompt in zip(order_ids, task_prompts)
    ]
    save_nemo_gym_inputs_jsonl(rows, output_path)
    return output_path


def profile_rollout_rewards(
    input_rows: list[dict[str, Any]],
    result_rows: list[NemoGymResultRow],
    output_dir: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run NeMo Gym RewardProfiler on collected rollout results.

    Computes group-level and agent-level reward statistics across
    the collected rollout batch.

    Args:
        input_rows: The original task input rows (list of dicts).
        result_rows: Collected result rows (NemoGymResultRow objects).
        output_dir: If provided, writes profiling results to disk.

    Returns:
        Tuple of (group_level_metrics, agent_level_metrics).
    """
    profiler = RewardProfiler()

    result_dicts = [row.to_dict() for row in result_rows]
    group_metrics, agent_metrics = profiler.profile_from_data(
        rows=input_rows,
        results=result_dicts,
    )

    if output_dir is not None:
        base_path = Path(output_dir)
        profiler.write_to_disk(group_metrics, agent_metrics, base_path)

    return group_metrics, agent_metrics


def save_enriched_as_nemo_gym(
    results: list[EnrichedEpisodeResult],
    output_path: str,
    agent_name: str = "late-order-recovery-agent",
) -> str:
    """Convert enriched episodes to NeMo Gym result rows and write to JSONL.

    This is the main export path from the rollout layer to NeMo Gym's
    reward profiling and training infrastructure.

    Args:
        results: Enriched episode results from run_enriched_episode().
        output_path: Path to write the output JSONL.
        agent_name: Agent name for result row metadata.

    Returns:
        The output path.
    """
    rows = enriched_batch_to_nemo_gym_rows(results, agent_name)
    save_nemo_gym_rows_jsonl(rows, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Durable trajectory export from NeMo Gym sessions
# ---------------------------------------------------------------------------


def export_session_episodes(
    session_ids: list[str],
    output_path: str,
) -> str:
    """Export NeMo Gym session episodes to canonical JSONL for offline analysis.

    Retrieves canonical Episodes from active NeMo Gym sessions and
    writes them through the canonical serializer. This closes the
    observability gap: training-time fallback events, rejects, and
    turn boundaries are now persisted in a durable artifact that
    offline eval and regression tooling can consume after collection.

    Args:
        session_ids: List of NeMo Gym session IDs to export.
        output_path: Path to write the canonical JSONL.

    Returns:
        The output path.
    """
    from src.envs.nemo_gym_adapter import LateOrderResourceServer
    from src.rollouts.serializers import save_episodes_jsonl

    episodes = []
    for sid in session_ids:
        ep = LateOrderResourceServer.get_session_episode(sid)
        if ep is not None:
            episodes.append(ep)

    save_episodes_jsonl(episodes, output_path)
    return output_path
