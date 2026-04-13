"""
Pre-built scripted episode traces for workshop demonstration.

These functions produce canonical Episode objects with deterministic tool
sequences over the synthetic scenario data.  They are *raw* episodes (no
rewards attached).  Pass the result through ``enrich_episode()`` from
``rollouts.episode_runner`` to get per-step rewards and a full reward summary.

Usage::

    from src.rollouts.scripted_traces import build_successful_episode, build_repair_episode
    from src.rollouts.episode_runner import enrich_episode

    episode = build_successful_episode()
    enriched = enrich_episode(episode)
"""
from __future__ import annotations

from src.rollouts.canonical_sequences import (
    build_successful_steps,
    get_base_step_defs,
    build_recovery_options,
    build_recommend_args,
    build_final_answer,
    _call_tool,
)
from src.rollouts.trace_types import (
    Episode,
    EpisodeMetrics,
    Event,
    EventType,
    ToolCallPayload,
    ToolResultPayload,
    ValidationErrorPayload,
    RepairAttemptPayload,
    RejectPayload,
    TerminalOutcomePayload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_call_event(
    step: int,
    name: str,
    arguments: dict,
    thought: str,
) -> Event:
    return Event(
        event_type=EventType.TOOL_CALL,
        step_index=step,
        payload=ToolCallPayload(
            tool_name=name,
            arguments=arguments,
            thought=thought,
        ),
    )


def _tool_result_event(step: int, name: str, result: dict) -> Event:
    return Event(
        event_type=EventType.TOOL_RESULT,
        step_index=step,
        payload=ToolResultPayload(tool_name=name, result=result),
    )


# ---------------------------------------------------------------------------
# Successful episode
# ---------------------------------------------------------------------------

def build_successful_episode() -> Episode:
    """Build a scripted successful trajectory for SO-10482.

    Calls tools in the correct dependency order using real tool functions
    over synthetic data.  No model is needed.

    Returns:
        A raw Episode (no rewards) with 9 tool calls ending in a
        final-answer recommendation.
    """
    steps, final_answer = build_successful_steps()

    events: list[Event] = []
    step = 0

    for ts in steps:
        result = _call_tool(ts.name, ts.arguments)
        events.append(_tool_call_event(step, ts.name, ts.arguments, ts.thought))
        step += 1
        events.append(_tool_result_event(step, ts.name, result))
        step += 1

    terminal = TerminalOutcomePayload(
        reason="final_answer",
        final_answer=final_answer,
    )
    events.append(Event(
        event_type=EventType.TERMINAL_OUTCOME,
        step_index=step,
        payload=terminal,
    ))

    return Episode(
        task_id="SO-10482",
        task_prompt="Investigate order SO-10482",
        model_id="scripted",
        events=events,
        terminal=terminal,
        metrics=EpisodeMetrics(
            total_steps=len(events),
            valid_tool_calls=9,
            invalid_tool_calls=0,
            repair_attempts=0,
            repair_successes=0,
            rejects=0,
            model_calls=9,
            wall_time_seconds=8.4,
        ),
    )


# ---------------------------------------------------------------------------
# Repair episode (with fallback repairs and a rejection)
# ---------------------------------------------------------------------------

def build_repair_episode() -> Episode:
    """Build a trajectory with fallback repairs and a rejection.

    Demonstrates:
    - Step 2: model emits plain text (rejected), agent retries
    - Step 3: model emits JSON with trailing comma (repaired)
    - Step 6: model emits typo in tool name (repaired via fuzzy match)

    Returns:
        A raw Episode (no rewards) with 9 valid tool calls, 1 rejection,
        and 2 successful repairs.
    """
    base = get_base_step_defs()
    events: list[Event] = []
    step = 0

    def call(
        name: str,
        args: dict,
        thought: str,
        repairs: list[str] | None = None,
    ) -> dict:
        nonlocal step
        result = _call_tool(name, args)
        if repairs:
            events.append(Event(
                event_type=EventType.TOOL_REPAIR_ATTEMPT,
                step_index=step,
                payload=RepairAttemptPayload(
                    original_output="<malformed>",
                    repaired_output="<repaired>",
                    repairs_applied=repairs,
                    succeeded=True,
                ),
            ))
            step += 1
        events.append(_tool_call_event(step, name, args, thought))
        step += 1
        events.append(_tool_result_event(step, name, result))
        step += 1
        return result

    # Step 1: get_order (clean)
    call(base[0].name, base[0].arguments,
         "Start by looking up the order details.")

    # Step 2: model emits plain text instead of JSON -> REJECTED
    events.append(Event(
        event_type=EventType.TOOL_VALIDATION_ERROR,
        step_index=step,
        payload=ValidationErrorPayload(
            error_type="no_json",
            message="No JSON object found in model output.",
            raw_model_output="I think we should check the shipment status next.",
        ),
    ))
    step += 1
    events.append(Event(
        event_type=EventType.TOOL_REJECT,
        step_index=step,
        payload=RejectPayload(
            reason="No JSON object found in model output.",
            raw_model_output="I think we should check the shipment status next.",
        ),
    ))
    step += 1

    # Step 3: get_shipment_status (repaired -- trailing comma)
    call(base[1].name, base[1].arguments,
         "Check shipping status after retry.",
         repairs=["fixed_trailing_commas"])

    # Step 4: get_inventory (clean)
    call(base[2].name, base[2].arguments,
         "Check source DC inventory.")

    # Step 5: get_fulfillment_capacity (clean)
    call(base[3].name, base[3].arguments,
         "Check capacity at source DC.")

    # Step 6: find_alternate_inventory (repaired -- tool name typo)
    call(base[4].name, base[4].arguments,
         "Search alternates. Model originally wrote 'find_alternat_inventory'.",
         repairs=["corrected_tool_name:find_alternat_inventory->find_alternate_inventory"])

    # Step 7: get_transfer_eta (clean)
    east_transfer = call(base[5].name, base[5].arguments,
                         "Get transfer ETA from DC-EAST-02.")

    # Step 8: get_supplier_expedite_options (clean)
    call(base[6].name, base[6].arguments,
         "Check supplier expedite for the shortfall.")

    # Step 9: score_recovery_options (clean, includes substitute)
    options = build_recovery_options(east_transfer, include_third_supplier=False)
    scored = call(
        "score_recovery_options",
        {"options": options, "objective": "minimize_delay"},
        "Score recovery options including substitute.",
    )

    # Step 10: recommend_action (clean)
    rec_args = build_recommend_args(scored)
    rec = call("recommend_action", rec_args,
               "Produce final recommendation.")

    # Terminal outcome
    final_answer = build_final_answer(rec)
    terminal = TerminalOutcomePayload(
        reason="final_answer",
        final_answer=final_answer,
    )
    events.append(Event(
        event_type=EventType.TERMINAL_OUTCOME,
        step_index=step,
        payload=terminal,
    ))

    return Episode(
        task_id="SO-10482",
        task_prompt="Investigate order SO-10482",
        model_id="scripted",
        events=events,
        terminal=terminal,
        metrics=EpisodeMetrics(
            total_steps=len(events),
            valid_tool_calls=9,
            invalid_tool_calls=1,
            repair_attempts=2,
            repair_successes=2,
            rejects=1,
            model_calls=12,
            wall_time_seconds=14.2,
        ),
    )
