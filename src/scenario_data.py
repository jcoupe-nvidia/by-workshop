"""
Synthetic in-memory scenario data for the late-order recovery workshop.

Provides small, deterministic datasets that are easy to inspect in notebook
cells. All data is defined as plain Python dicts/lists -- no database, no
external files.

Tables:
    ORDERS              -- customer orders (keyed by order_id)
    SHIPMENTS           -- shipment status per order
    INVENTORY           -- on-hand inventory by (sku, dc_id)
    TRANSFER_LANES      -- transfer routes between DCs with lead times
    SUPPLIER_OPTIONS    -- supplier expedite options by sku
    FULFILLMENT_CAPACITY-- available fulfillment capacity by (dc_id, date)
    SUBSTITUTE_SKUS     -- optional substitute SKU mappings

The core scenario is order SO-10482 for 1,200 units of SKU-4090 shipping
from DC-WEST-01 with a committed delivery date of 2026-04-18.

Phase 2 will populate these tables with concrete values.
"""
from __future__ import annotations

from typing import Any

# -- Orders ----------------------------------------------------------------
# Keys: order_id -> {customer, sku, qty, source_dc, committed_date, ...}
ORDERS: dict[str, dict[str, Any]] = {}

# -- Shipments -------------------------------------------------------------
# Keys: order_id -> {status, shipped_qty, carrier, eta, ...}
SHIPMENTS: dict[str, dict[str, Any]] = {}

# -- Inventory -------------------------------------------------------------
# Keys: (sku, dc_id) -> {on_hand, reserved, available, ...}
INVENTORY: dict[tuple[str, str], dict[str, Any]] = {}

# -- Transfer lanes --------------------------------------------------------
# Keys: (from_dc, to_dc) -> {lead_days, cost_per_unit, ...}
TRANSFER_LANES: dict[tuple[str, str], dict[str, Any]] = {}

# -- Supplier expedite options ---------------------------------------------
# Keys: sku -> [{supplier, lead_days, moq, cost_per_unit, ...}, ...]
SUPPLIER_OPTIONS: dict[str, list[dict[str, Any]]] = {}

# -- Fulfillment capacity --------------------------------------------------
# Keys: (dc_id, date_str) -> {max_units, allocated, remaining, ...}
FULFILLMENT_CAPACITY: dict[tuple[str, str], dict[str, Any]] = {}

# -- Substitute SKUs -------------------------------------------------------
# Keys: sku -> [{substitute_sku, compatibility, ...}, ...]
SUBSTITUTE_SKUS: dict[str, list[dict[str, Any]]] = {}
