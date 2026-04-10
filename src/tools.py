"""
Deterministic tool implementations and tool registry.

Each tool is a pure function over the synthetic data in scenario_data.py.
Tools are registered in TOOL_REGISTRY so the agent loop can dispatch by name.

Tools (7-9 total):
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
    score_recovery_options       -> requires at least one mitigation path
    recommend_action             -> requires score_recovery_options

Phase 2 will implement these functions and populate the registry.
"""
from __future__ import annotations

from typing import Any, Callable

# Type alias for tool functions
ToolFn = Callable[..., dict[str, Any]]

# Central registry: tool_name -> (function, parameter_spec, description)
TOOL_REGISTRY: dict[str, tuple[ToolFn, dict[str, str], str]] = {}

# Sequence dependency graph: tool_name -> set of tools that must precede it
TOOL_DEPENDENCIES: dict[str, set[str]] = {
    "get_order": set(),
    "get_shipment_status": {"get_order"},
    "get_inventory": {"get_order"},
    "find_alternate_inventory": {"get_inventory"},
    "get_transfer_eta": {"find_alternate_inventory"},
    "get_supplier_expedite_options": {"get_order"},
    "get_fulfillment_capacity": {"get_order"},
    "score_recovery_options": set(),  # requires at least one mitigation path (checked at runtime)
    "recommend_action": {"score_recovery_options"},
}


def register_tool(
    name: str, params: dict[str, str], description: str
) -> Callable[[ToolFn], ToolFn]:
    """Decorator to register a tool function."""
    def decorator(fn: ToolFn) -> ToolFn:
        TOOL_REGISTRY[name] = (fn, params, description)
        return fn
    return decorator
