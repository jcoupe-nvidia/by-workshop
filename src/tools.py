"""
Backward-compatibility shim for src.tools.

The canonical definitions now live in:
    - src.runtime.tools  (tool implementations, registry, dependencies)

This module re-exports everything so existing imports continue to work.
"""
from src.runtime.tools import (  # noqa: F401
    ToolFn,
    TOOL_REGISTRY,
    TOOL_DEPENDENCIES,
    register_tool,
    get_order,
    get_shipment_status,
    get_inventory,
    find_alternate_inventory,
    get_transfer_eta,
    get_supplier_expedite_options,
    get_fulfillment_capacity,
    score_recovery_options,
    recommend_action,
)
