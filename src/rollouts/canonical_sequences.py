"""Canonical tool sequences for scripted episodes and rollout builders.

Single source of truth for:
- Tool call ordering, arguments, and thoughts for SO-10482
- Recovery options construction from tool results
- Final answer construction from recommend_action results

Consumed by:
- scripted_traces.py (builds Episode Events)
- nemo_gym_rollouts.py (builds AgentActions)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.runtime.tools import TOOL_REGISTRY


@dataclass
class ToolStep:
    """One step in a canonical tool sequence."""
    name: str
    arguments: dict[str, Any]
    thought: str


def _call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool from the registry."""
    fn, _, _ = TOOL_REGISTRY[name]
    return fn(**args)


def get_base_step_defs() -> list[ToolStep]:
    """Return the 7 tool steps before scoring (static arguments).

    These steps have arguments that don't depend on earlier tool results,
    so they can be reused directly by repair/reject builders that need
    to modify tool names or insert additional error steps.
    """
    return [
        ToolStep(
            "get_order", {"order_id": "SO-10482"},
            "Start by looking up the order details.",
        ),
        ToolStep(
            "get_shipment_status", {"order_id": "SO-10482"},
            "Check current shipment status to assess risk.",
        ),
        ToolStep(
            "get_inventory", {"sku": "SKU-4090", "dc_id": "DC-WEST-01"},
            "Check inventory at primary DC.",
        ),
        ToolStep(
            "get_fulfillment_capacity",
            {"dc_id": "DC-WEST-01", "date": "2026-04-18"},
            "Check fulfillment capacity at source DC on committed date.",
        ),
        ToolStep(
            "find_alternate_inventory", {"sku": "SKU-4090", "region": "ALL"},
            "Primary DC has only 300 of 1200 needed. Search all DCs.",
        ),
        ToolStep(
            "get_transfer_eta",
            {"from_dc": "DC-EAST-02", "to_dc": "DC-WEST-01",
             "sku": "SKU-4090", "qty": 900},
            "DC-EAST-02 has 1000 available. Get transfer ETA for 900-unit shortfall.",
        ),
        ToolStep(
            "get_supplier_expedite_options", {"sku": "SKU-4090", "qty": 900},
            "Check supplier expedite options for the shortfall.",
        ),
    ]


def build_successful_steps() -> tuple[list[ToolStep], dict[str, Any]]:
    """Build the canonical successful tool sequence for SO-10482.

    Returns (steps, final_answer) where steps is the ordered list of
    tool calls and final_answer is built from the recommend_action result.
    """
    steps = list(get_base_step_defs())

    east_transfer = _call_tool("get_transfer_eta", steps[5].arguments)

    options = build_recovery_options(east_transfer, include_third_supplier=True)
    scored_args = {"options": options, "objective": "minimize_delay"}
    steps.append(ToolStep(
        "score_recovery_options", scored_args,
        "Score all recovery options including substitute SKU.",
    ))

    scored = _call_tool("score_recovery_options", scored_args)
    rec_args = build_recommend_args(scored)
    steps.append(ToolStep(
        "recommend_action", rec_args,
        "Produce the final recommendation based on scored options.",
    ))

    rec = _call_tool("recommend_action", rec_args)
    final_answer = build_final_answer(rec)

    return steps, final_answer


def build_recovery_options(
    east_transfer: dict[str, Any],
    include_third_supplier: bool = False,
) -> list[dict[str, Any]]:
    """Build the canonical recovery options list for SO-10482.

    Args:
        east_transfer: Result from get_transfer_eta for DC-EAST-02.
        include_third_supplier: Whether to include the FastSemi Direct option
            (the successful episode includes it; the repair episode doesn't).
    """
    options: list[dict[str, Any]] = [
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
    if include_third_supplier:
        options.append(
            {"source": "supplier:FastSemi Direct",
             "description": "supplier_expedite from FastSemi Direct",
             "path_type": "supplier_expedite", "lead_days": 5,
             "cost_per_unit": 12.00, "total_cost": 10800.0,
             "feasible": True, "covers_full_qty": True},
        )
    options.append(
        {"source": "substitute:SKU-4090-B@DC-WEST-01",
         "description": "substitute SKU-4090-B at DC-WEST-01",
         "path_type": "substitute", "lead_days": 0,
         "cost_per_unit": 0.0, "total_cost": 0.0,
         "feasible": False, "covers_full_qty": False},
    )
    return options


def build_recommend_args(scored: dict[str, Any]) -> dict[str, Any]:
    """Build recommend_action arguments from scored recovery options."""
    return {"context": {
        "best_option": scored["best_option"],
        "order": {
            "order_id": "SO-10482", "sku": "SKU-4090",
            "qty": 1200, "committed_date": "2026-04-18",
        },
        "objective": "minimize_delay",
    }}


def build_final_answer(rec: dict[str, Any]) -> dict[str, Any]:
    """Build the final answer dict from a recommend_action result."""
    return {
        "action": rec["action"],
        "rationale": rec["rationale"],
        "expected_delivery": rec["expected_delivery"],
        "meets_committed_date": rec["meets_committed_date"],
        "confidence": rec["confidence"],
    }
