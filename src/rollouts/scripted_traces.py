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
from src.runtime.tools import TOOL_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_tool(name: str, arguments: dict) -> dict:
    """Execute a tool from the registry and return its result."""
    fn, _, _ = TOOL_REGISTRY[name]
    return fn(**arguments)


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
    events: list[Event] = []
    step = 0

    def call(name: str, args: dict, thought: str) -> dict:
        nonlocal step
        result = _call_tool(name, args)
        events.append(_tool_call_event(step, name, args, thought))
        step += 1
        events.append(_tool_result_event(step, name, result))
        step += 1
        return result

    # Skill 1: diagnose_order_risk
    call("get_order", {"order_id": "SO-10482"},
         "Start by looking up the order details.")
    call("get_shipment_status", {"order_id": "SO-10482"},
         "Check current shipment status to assess risk.")

    # Skill 2: assess_primary_fulfillment
    call("get_inventory", {"sku": "SKU-4090", "dc_id": "DC-WEST-01"},
         "Check inventory at primary DC.")
    call("get_fulfillment_capacity", {"dc_id": "DC-WEST-01", "date": "2026-04-18"},
         "Check fulfillment capacity at source DC on committed date.")

    # Skill 3: evaluate_alternate_recovery_paths
    call("find_alternate_inventory", {"sku": "SKU-4090", "region": "ALL"},
         "Primary DC has only 300 of 1200 needed. Search all DCs.")
    east_transfer = call(
        "get_transfer_eta",
        {"from_dc": "DC-EAST-02", "to_dc": "DC-WEST-01",
         "sku": "SKU-4090", "qty": 900},
        "DC-EAST-02 has 1000 available. Get transfer ETA for 900-unit shortfall.",
    )
    supplier_opts = call(
        "get_supplier_expedite_options",
        {"sku": "SKU-4090", "qty": 900},
        "Check supplier expedite options for the shortfall.",
    )

    # Skill 4: synthesize_recommendation
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
    ]
    scored = call(
        "score_recovery_options",
        {"options": options, "objective": "minimize_delay"},
        "Score all recovery options to find the best path.",
    )
    rec = call(
        "recommend_action",
        {"context": {
            "best_option": scored["best_option"],
            "order": {"order_id": "SO-10482", "sku": "SKU-4090",
                      "qty": 1200, "committed_date": "2026-04-18"},
            "objective": "minimize_delay",
        }},
        "Produce the final recommendation based on scored options.",
    )

    # Terminal outcome
    final_answer = {
        "action": rec["action"],
        "rationale": rec["rationale"],
        "expected_delivery": rec["expected_delivery"],
        "meets_committed_date": rec["meets_committed_date"],
        "confidence": rec["confidence"],
    }
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
        # If this was a repaired call, record the repair event first
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
    call("get_order", {"order_id": "SO-10482"},
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
    call("get_shipment_status", {"order_id": "SO-10482"},
         "Check shipping status after retry.",
         repairs=["fixed_trailing_commas"])

    # Step 4: get_inventory (clean)
    call("get_inventory", {"sku": "SKU-4090", "dc_id": "DC-WEST-01"},
         "Check source DC inventory.")

    # Step 5: get_fulfillment_capacity (clean)
    call("get_fulfillment_capacity", {"dc_id": "DC-WEST-01", "date": "2026-04-18"},
         "Check capacity at source DC.")

    # Step 6: find_alternate_inventory (repaired -- tool name typo)
    call("find_alternate_inventory", {"sku": "SKU-4090", "region": "ALL"},
         "Search alternates. Model originally wrote 'find_alternat_inventory'.",
         repairs=["corrected_tool_name:find_alternat_inventory->find_alternate_inventory"])

    # Step 7: get_transfer_eta (clean)
    east_transfer = call(
        "get_transfer_eta",
        {"from_dc": "DC-EAST-02", "to_dc": "DC-WEST-01",
         "sku": "SKU-4090", "qty": 900},
        "Get transfer ETA from DC-EAST-02.",
    )

    # Step 8: get_supplier_expedite_options (clean)
    call("get_supplier_expedite_options",
         {"sku": "SKU-4090", "qty": 900},
         "Check supplier expedite for the shortfall.")

    # Step 9: score_recovery_options (clean)
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
    ]
    scored = call(
        "score_recovery_options",
        {"options": options, "objective": "minimize_delay"},
        "Score recovery options.",
    )

    # Step 10: recommend_action (clean)
    rec = call(
        "recommend_action",
        {"context": {
            "best_option": scored["best_option"],
            "order": {"order_id": "SO-10482", "sku": "SKU-4090",
                      "qty": 1200, "committed_date": "2026-04-18"},
            "objective": "minimize_delay",
        }},
        "Produce final recommendation.",
    )

    # Terminal outcome
    final_answer = {
        "action": rec["action"],
        "rationale": rec["rationale"],
        "expected_delivery": rec["expected_delivery"],
        "meets_committed_date": rec["meets_committed_date"],
        "confidence": rec["confidence"],
    }
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
