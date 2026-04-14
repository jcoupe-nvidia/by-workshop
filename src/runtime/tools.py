"""
Deterministic tool implementations and tool registry.

Each tool is a pure function over the synthetic data in scenario_data.py.
Tools are registered in TOOL_REGISTRY so the agent loop can dispatch by name.

Owns:
    - Tool function implementations
    - Tool registry (name -> function, params, description)
    - Sequence dependency graph
    - Tool registration decorator

Does NOT own:
    - Prompt policy or agent orchestration (see runtime.prompts, runtime.agent)
    - Task-specific validation beyond dependencies (see envs.validators)
    - Reward semantics or training concerns

Tools (9 total):
    get_order(order_id)                         -- look up order details
    get_shipment_status(order_id)               -- current shipment state
    get_inventory(sku, dc_id)                   -- on-hand at a specific DC
    find_alternate_inventory(sku, region)        -- search other DCs in region
    get_transfer_eta(from_dc, to_dc, sku, qty)  -- transfer time estimate
    get_supplier_expedite_options(sku, qty)      -- supplier rush options
    get_fulfillment_capacity(dc_id, date)        -- DC capacity on a date
    score_recovery_options(options, objective)   -- rank mitigation candidates
    recommend_action(context)                   -- produce final recommendation

Call-order dependencies (machine-checkable):
    get_order            -> prerequisite for all other tools
    get_shipment_status  -> requires get_order first
    get_inventory        -> requires get_order (to know sku, source dc)
    find_alternate_inventory -> requires get_inventory (source checked first)
    get_transfer_eta     -> requires find_alternate_inventory
    get_supplier_expedite_options -> requires get_order
    get_fulfillment_capacity     -> requires get_order
    score_recovery_options       -> requires find_alternate_inventory
    recommend_action             -> requires score_recovery_options
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable

SIMULATION_DATE = datetime(2026, 4, 10)
SIMULATION_DATE_STR = "2026-04-10"

from src.scenario_data import (
    ORDERS,
    SHIPMENTS,
    INVENTORY,
    TRANSFER_LANES,
    SUPPLIER_OPTIONS,
    FULFILLMENT_CAPACITY,
    SUBSTITUTE_SKUS,
    REGION_DCS,
)

# Type alias for tool functions
ToolFn = Callable[..., dict[str, Any]]

# Central registry: tool_name -> (function, parameter_spec, description)
TOOL_REGISTRY: dict[str, tuple[ToolFn, dict[str, str], str]] = {}

# Re-export TOOL_DEPENDENCIES from envs.state where it is canonically defined.
# The dependency graph is task truth owned by envs/; runtime re-exports for
# convenience so existing call sites don't need to change their import path.
from src.envs.state import TOOL_DEPENDENCIES  # noqa: E402


def register_tool(
    name: str, params: dict[str, str], description: str
) -> Callable[[ToolFn], ToolFn]:
    """Decorator to register a tool function."""
    def decorator(fn: ToolFn) -> ToolFn:
        TOOL_REGISTRY[name] = (fn, params, description)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

@register_tool(
    name="get_order",
    params={"order_id": "str"},
    description="Look up order details by order ID.",
)
def get_order(order_id: str) -> dict[str, Any]:
    if order_id not in ORDERS:
        return {"error": f"Order {order_id} not found."}
    return {**ORDERS[order_id]}


@register_tool(
    name="get_shipment_status",
    params={"order_id": "str"},
    description="Get the current shipment status for an order.",
)
def get_shipment_status(order_id: str) -> dict[str, Any]:
    if order_id not in SHIPMENTS:
        return {"error": f"No shipment record for order {order_id}."}
    return {**SHIPMENTS[order_id]}


@register_tool(
    name="get_inventory",
    params={"sku": "str", "dc_id": "str"},
    description="Check on-hand, reserved, and available inventory for a SKU at a specific DC.",
)
def get_inventory(sku: str, dc_id: str) -> dict[str, Any]:
    key = (sku, dc_id)
    if key not in INVENTORY:
        return {"error": f"No inventory record for {sku} at {dc_id}.", "sku": sku, "dc_id": dc_id}
    return {**INVENTORY[key]}


@register_tool(
    name="find_alternate_inventory",
    params={"sku": "str", "region": "str"},
    description="Search for available inventory of a SKU across DCs in a region. Use region='ALL' to search everywhere.",
)
def find_alternate_inventory(sku: str, region: str) -> dict[str, Any]:
    dc_list = REGION_DCS.get(region, [])
    if not dc_list:
        return {"error": f"Unknown region '{region}'.", "known_regions": list(REGION_DCS.keys())}

    results: list[dict[str, Any]] = []
    for dc_id in dc_list:
        key = (sku, dc_id)
        if key in INVENTORY and INVENTORY[key]["available"] > 0:
            results.append({**INVENTORY[key]})

    # Also include substitute SKUs if any
    substitutes: list[dict[str, Any]] = []
    if sku in SUBSTITUTE_SKUS:
        for sub in SUBSTITUTE_SKUS[sku]:
            sub_key = (sub["substitute_sku"], sub["available_dc"])
            if sub_key in INVENTORY and INVENTORY[sub_key]["available"] > 0:
                substitutes.append({
                    **INVENTORY[sub_key],
                    "is_substitute": True,
                    "compatibility": sub["compatibility"],
                    "notes": sub["notes"],
                })

    return {
        "sku": sku,
        "region": region,
        "matching_dcs": results,
        "substitutes": substitutes,
        "total_available": sum(r["available"] for r in results),
    }


@register_tool(
    name="get_transfer_eta",
    params={"from_dc": "str", "to_dc": "str", "sku": "str", "qty": "int"},
    description="Estimate transfer time and cost to move units between DCs.",
)
def get_transfer_eta(from_dc: str, to_dc: str, sku: str, qty: int) -> dict[str, Any]:
    lane_key = (from_dc, to_dc)
    if lane_key not in TRANSFER_LANES:
        return {"error": f"No transfer lane from {from_dc} to {to_dc}."}

    lane = TRANSFER_LANES[lane_key]
    inv_key = (sku, from_dc)
    available = INVENTORY.get(inv_key, {}).get("available", 0)

    if qty > lane["max_qty"]:
        return {
            "error": f"Requested qty {qty} exceeds lane max {lane['max_qty']}.",
            "from_dc": from_dc,
            "to_dc": to_dc,
        }

    feasible = available >= qty
    transfer_qty = min(qty, available)

    return {
        "from_dc": from_dc,
        "to_dc": to_dc,
        "sku": sku,
        "requested_qty": qty,
        "available_at_source": available,
        "transfer_qty": transfer_qty,
        "lead_days": lane["lead_days"],
        "cost_per_unit": lane["cost_per_unit"],
        "total_cost": round(transfer_qty * lane["cost_per_unit"], 2),
        "feasible": feasible,
    }


@register_tool(
    name="get_supplier_expedite_options",
    params={"sku": "str", "qty": "int"},
    description="Get available supplier expedite (rush) options for a SKU and quantity.",
)
def get_supplier_expedite_options(sku: str, qty: int) -> dict[str, Any]:
    if sku not in SUPPLIER_OPTIONS:
        return {"error": f"No supplier options for {sku}.", "sku": sku}

    options: list[dict[str, Any]] = []
    for opt in SUPPLIER_OPTIONS[sku]:
        feasible = qty >= opt["moq"] and qty <= opt["max_qty"]
        options.append({
            "supplier": opt["supplier"],
            "lead_days": opt["lead_days"],
            "moq": opt["moq"],
            "max_qty": opt["max_qty"],
            "cost_per_unit": opt["cost_per_unit"],
            "total_cost": round(qty * opt["cost_per_unit"], 2),
            "feasible": feasible,
            "feasibility_note": (
                "OK" if feasible
                else f"Qty {qty} outside range [{opt['moq']}, {opt['max_qty']}]"
            ),
        })

    return {"sku": sku, "requested_qty": qty, "options": options}


@register_tool(
    name="get_fulfillment_capacity",
    params={"dc_id": "str", "date": "str"},
    description="Check available fulfillment capacity at a DC on a specific date.",
)
def get_fulfillment_capacity(dc_id: str, date: str) -> dict[str, Any]:
    key = (dc_id, date)
    if key not in FULFILLMENT_CAPACITY:
        return {"error": f"No capacity data for {dc_id} on {date}.", "dc_id": dc_id, "date": date}
    return {**FULFILLMENT_CAPACITY[key]}


@register_tool(
    name="score_recovery_options",
    params={"options": "list[dict]", "objective": "str"},
    description="Score and rank a list of recovery options against an objective (e.g. 'minimize_delay', 'minimize_cost').",
)
def score_recovery_options(
    options: list[dict[str, Any]], objective: str
) -> dict[str, Any]:
    if not options:
        return {"error": "No options provided to score."}

    scored: list[dict[str, Any]] = []
    for opt in options:
        # Score components (0-1, higher is better)
        lead = opt.get("lead_days", 99)
        cost = opt.get("cost_per_unit", 99.0)
        feasible = opt.get("feasible", False)
        covers_qty = opt.get("covers_full_qty", False)

        # Normalize: lower lead_days -> higher time score
        time_score = max(0.0, 1.0 - lead / 14.0)
        # Normalize: lower cost -> higher cost score
        cost_score = max(0.0, 1.0 - cost / 20.0)
        feasibility_score = 1.0 if feasible else 0.0
        coverage_score = 1.0 if covers_qty else 0.5

        if objective == "minimize_delay":
            overall = 0.4 * time_score + 0.2 * cost_score + 0.2 * feasibility_score + 0.2 * coverage_score
        elif objective == "minimize_cost":
            overall = 0.2 * time_score + 0.4 * cost_score + 0.2 * feasibility_score + 0.2 * coverage_score
        else:
            # balanced
            overall = 0.25 * time_score + 0.25 * cost_score + 0.25 * feasibility_score + 0.25 * coverage_score

        scored.append({
            **opt,
            "scores": {
                "time": round(time_score, 3),
                "cost": round(cost_score, 3),
                "feasibility": round(feasibility_score, 3),
                "coverage": round(coverage_score, 3),
                "overall": round(overall, 3),
            },
        })

    scored.sort(key=lambda x: x["scores"]["overall"], reverse=True)
    return {
        "objective": objective,
        "ranked_options": scored,
        "best_option": scored[0] if scored else None,
    }


@register_tool(
    name="recommend_action",
    params={"context": "dict"},
    description=(
        "Produce a final recommendation based on scored recovery options and order context. "
        "The context dict must include 'best_option' (the top-ranked option dict from "
        "score_recovery_options, with at least 'description', 'lead_days', 'total_cost', "
        "and 'scores'), 'order' (the original order dict with 'committed_date'), and "
        "optionally 'objective' (the optimization objective string)."
    ),
)
def recommend_action(context: dict[str, Any]) -> dict[str, Any]:
    _BEST_OPTION_ALIASES = [
        "best_option", "chosen_option", "selected_option", "option", "top_option",
    ]
    best = None
    for key in _BEST_OPTION_ALIASES:
        if key in context and isinstance(context[key], dict):
            best = context[key]
            break

    _ORDER_ALIASES = ["order", "order_details", "order_info"]
    order = {}
    for key in _ORDER_ALIASES:
        if key in context and isinstance(context[key], dict):
            order = context[key]
            break

    if not order:
        committed_date = context.get("committed_date", "unknown")
    else:
        committed_date = order.get("committed_date", "unknown")

    if not best:
        return {
            "action": "escalate",
            "rationale": "No viable recovery options found.",
            "expected_delivery": None,
            "meets_committed_date": False,
            "confidence": 0.0,
        }

    lead_days = best.get("lead_days", 0)
    expected = SIMULATION_DATE + timedelta(days=lead_days)
    expected_str = expected.strftime("%Y-%m-%d")

    try:
        committed = datetime.strptime(committed_date, "%Y-%m-%d")
        meets_date = expected <= committed
    except ValueError:
        meets_date = False

    overall_score = best.get("scores", {}).get("overall", 0.0)
    confidence = round(min(1.0, overall_score + (0.2 if meets_date else 0.0)), 2)

    return {
        "action": best.get("description", best.get("source", "unknown")),
        "rationale": (
            f"Selected based on {context.get('objective', 'balanced')} objective. "
            f"Lead time: {lead_days} days, cost: ${best.get('total_cost', 0):.2f}. "
            f"{'Meets' if meets_date else 'Misses'} committed date of {committed_date}."
        ),
        "expected_delivery": expected_str,
        "meets_committed_date": meets_date,
        "confidence": confidence,
        "selected_option": best,
    }
