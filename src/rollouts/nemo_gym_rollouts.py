"""
NeMo Gym rollout collection adapter.

Bridges the rollout layer with NeMo Gym's training-time infrastructure:

    - Training-time execution pipeline (validate → repair → reject → execute → record)
    - Converts enriched Episodes to NemoGymResultRows
    - Builds RolloutCollectionConfig for NeMo Gym rollout collection runs
    - Durable trajectory export from NeMo Gym sessions

Owns:
    - Training-time agent-action execution pipeline
    - Enriched episode -> NeMo Gym result row conversion (via envs.nemo_gym_adapter)
    - RolloutCollectionConfig construction for the late-order-recovery task
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
from typing import Any

from nemo_gym.rollout_collection import (
    RolloutCollectionConfig,
    RolloutCollectionHelper,
)
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
# Async helper — Jupyter-compatible coroutine runner
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine in both Jupyter and non-Jupyter contexts.

    Uses asyncio.run() for standalone scripts. Falls back to
    get_event_loop().run_until_complete() when an event loop is already
    running (e.g., inside Jupyter notebooks).
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Training-time execution pipeline
# ---------------------------------------------------------------------------
# This section owns the canonical validate → repair → reject → execute →
# record loop for NeMo Gym training-time rollout collection. The NeMo Gym
# resource server in envs/ delegates here so that envs/ stays focused on
# environment validation surfaces while rollouts/ owns execution plumbing.
#
# NOTE: This pipeline delegates the core validate → repair → reject logic to
# the shared helper in src.runtime.execution.validate_and_repair(), ensuring
# identical validation semantics between interactive and training-time paths.
# The training-time pipeline adds environment interaction (env.step,
# env.record_invalid, etc.) around the shared validation core.


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
    """Execute the validate → repair → reject → execute → record loop.

    Uses the shared validate_and_repair() pipeline from runtime.execution
    to ensure identical validation semantics between interactive and
    training-time execution paths.
    """
    from src.runtime.execution import validate_and_repair

    reward = 0.0
    raw_output = json.dumps({
        "tool_call": {
            "name": action.tool_name,
            "arguments": action.arguments,
        }
    })

    vr = validate_and_repair(raw_output, tool_registry, recorder=recorder)

    # Handle rejection
    if (vr.fallback_result is not None
            and vr.fallback_result.action == fallback_action_cls.REJECTED):
        env.record_fallback_repair(succeeded=False)
        env.record_fallback_reject()
        recorder.record_validation_error(
            error_type="rejected",
            message=vr.fallback_result.rejection_reason or "Unrecoverable output",
            raw_model_output=raw_output,
        )
        env.record_invalid(
            "rejected",
            vr.fallback_result.rejection_reason or "Unrecoverable output",
        )
        step_idx = env.get_step_count() - 1
        step_reward = env.get_step_reward(step_idx)
        if step_reward:
            reward += step_reward.total
        return reward

    if vr.was_repaired:
        env.record_fallback_repair(succeeded=True)

    # Still invalid after repair attempt
    if vr.validation_error is not None:
        recorder.record_validation_error(
            error_type=vr.validation_error.error_type,
            message=vr.validation_error.message,
            raw_model_output=raw_output,
        )
        env.record_invalid(vr.validation_error.error_type, vr.validation_error.message)
        step_idx = env.get_step_count() - 1
        step_reward = env.get_step_reward(step_idx)
        if step_reward:
            reward += step_reward.total
        return reward

    # Valid tool call — record, execute, and step
    result = vr.parsed
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
        except Exception as exc:
            tool_result = {"error": f"{type(exc).__name__}: {exc}"}
        recorder.record_tool_result(valid_name, tool_result)
        env.step(valid_name, valid_args, tool_result, was_repaired=vr.was_repaired)
        step_idx = env.get_step_count() - 1
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

    This is the **training-time** execution path, called by the NeMo Gym
    resource server's ``verify()`` during rollout collection.  The
    interactive demo path lives in ``runtime/agent.py`` and uses its own
    ``EpisodeRecorder``.  Both paths share the core validation logic via
    ``src.runtime.execution.validate_and_repair()``, which ensures
    identical validate → repair → reject semantics regardless of caller.
    Divergence between the two paths is therefore limited to how
    environment interaction (``env.step``, ``env.record_invalid``, etc.)
    is wired around the shared validation core.

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
    step_rewards = env.get_all_step_rewards()
    terminal_reward = env.get_terminal_reward()

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
    Uses the canonical tool sequence from canonical_sequences so that
    scenario data is single-sourced with scripted_traces.
    """
    from src.rollouts.canonical_sequences import build_successful_steps, build_final_answer

    steps, final_answer = build_successful_steps()

    actions: list[AgentAction] = []
    for step in steps:
        actions.append(AgentAction("function_call", step.name, step.arguments))

    actions.append(AgentAction("message", content=json.dumps(final_answer)))

    return actions


def _build_repair_actions() -> list[AgentAction]:
    """Build a scripted action sequence with repairs and a rejection.

    Mirrors scripted_traces.build_repair_episode(): includes two repaired
    tool calls with typo names. Unlike the scripted traces approach,
    the repair handling happens in the execution pipeline
    (process_agent_action) rather than being pre-baked into Event objects.

    Uses canonical step definitions and recovery options from
    canonical_sequences to stay in sync with scripted_traces.
    """
    from src.rollouts.canonical_sequences import (
        get_base_step_defs, build_recovery_options, build_recommend_args,
        build_final_answer, _call_tool,
    )

    base = get_base_step_defs()
    actions: list[AgentAction] = []

    # Step 1: get_order (clean)
    actions.append(AgentAction("function_call", base[0].name, base[0].arguments))

    # Step 2: get_shipment_status with typo -> fuzzy repair
    actions.append(AgentAction("function_call", "get_shipmnt_status", base[1].arguments))

    # Steps 3-4: clean calls
    actions.append(AgentAction("function_call", base[2].name, base[2].arguments))
    actions.append(AgentAction("function_call", base[3].name, base[3].arguments))

    # Step 5: find_alternate_inventory with typo -> fuzzy repair
    actions.append(AgentAction("function_call", "find_alternat_inventory", base[4].arguments))

    # Steps 6-7: clean calls
    actions.append(AgentAction("function_call", base[5].name, base[5].arguments))
    actions.append(AgentAction("function_call", base[6].name, base[6].arguments))

    # Steps 8-9: score and recommend (no third supplier in repair episode)
    east_transfer = _call_tool(base[5].name, base[5].arguments)
    options = build_recovery_options(east_transfer, include_third_supplier=False)

    scored = _call_tool("score_recovery_options", {"options": options, "objective": "minimize_delay"})
    actions.append(AgentAction("function_call", "score_recovery_options",
                               {"options": options, "objective": "minimize_delay"}))

    rec_args = build_recommend_args(scored)
    rec = _call_tool("recommend_action", rec_args)
    actions.append(AgentAction("function_call", "recommend_action", rec_args))

    # Terminal
    final_answer = build_final_answer(rec)
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
    """Build a scripted action sequence with an unrecoverable reject.

    This produces an episode with:
    - A completely unknown tool name that cannot be fuzzy-matched (rejected)
    - Normal recovery completing the task after the failure

    Uses canonical step definitions and recovery options from
    canonical_sequences to stay in sync with scripted_traces.
    """
    from src.rollouts.canonical_sequences import (
        get_base_step_defs, build_recovery_options, build_recommend_args,
        build_final_answer, _call_tool,
    )

    base = get_base_step_defs()
    actions: list[AgentAction] = []

    # Step 1: get_order (clean)
    actions.append(AgentAction("function_call", base[0].name, base[0].arguments))

    # Step 2: completely unknown tool -> rejected (no fuzzy match possible)
    actions.append(AgentAction(
        "function_call", "analyze_demand_forecast", {"order_id": "SO-10482"},
    ))

    # Step 3: get_shipment_status (clean, continuing after reject)
    actions.append(AgentAction("function_call", base[1].name, base[1].arguments))

    # Steps 4-7: clean calls
    for i in range(2, 7):
        actions.append(AgentAction("function_call", base[i].name, base[i].arguments))

    # Steps 8-9: score and recommend (no third supplier in reject episode)
    east_transfer = _call_tool(base[5].name, base[5].arguments)
    options = build_recovery_options(east_transfer, include_third_supplier=False)

    scored = _call_tool("score_recovery_options", {"options": options, "objective": "minimize_delay"})
    actions.append(AgentAction("function_call", "score_recovery_options",
                               {"options": options, "objective": "minimize_delay"}))

    rec_args = build_recommend_args(scored)
    rec = _call_tool("recommend_action", rec_args)
    actions.append(AgentAction("function_call", "recommend_action", rec_args))

    # Terminal
    final_answer = build_final_answer(rec)
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
    _run_async(server.seed_session(seed_req))
    new_keys = set(_sessions.keys()) - pre_keys
    session_id = new_keys.pop()

    # Capture a reference to the session before the verify loop, because
    # verify() cleans up terminated sessions from _sessions (HIGH-2 fix).
    session = _sessions[session_id]

    # Step 2: verify — send each action as a NeMo Gym response
    for action in actions:
        verify_req = _build_verify_request_from_action(action, session_id)
        _run_async(server.verify(verify_req))

    # Step 3: Retrieve the episode and reward from the session
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

    # Session cleanup is handled by verify() on terminal (HIGH-2 fix).
    # Clean up any non-terminal session as a safety net.
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
