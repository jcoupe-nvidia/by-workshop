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

Data is designed so that:
- DC-WEST-01 (primary) has only 300 available units -> shortfall of 900
- DC-EAST-02 has 1,000 available units -> enough to cover via transfer
- DC-CENTRAL-03 has 200 available units -> not enough alone
- A supplier expedite option exists but is expensive and slow
- A substitute SKU (SKU-4090-B) is available at DC-WEST-01
- Fulfillment capacity at DC-WEST-01 is adequate for the committed date
"""
from __future__ import annotations

from typing import Any

# -- Orders ----------------------------------------------------------------
ORDERS: dict[str, dict[str, Any]] = {
    "SO-10482": {
        "order_id": "SO-10482",
        "customer": "Acme Manufacturing",
        "sku": "SKU-4090",
        "qty": 1200,
        "source_dc": "DC-WEST-01",
        "committed_date": "2026-04-18",
        "priority": "high",
        "region": "WEST",
    },
}

# -- Shipments -------------------------------------------------------------
SHIPMENTS: dict[str, dict[str, Any]] = {
    "SO-10482": {
        "order_id": "SO-10482",
        "status": "pending",
        "shipped_qty": 0,
        "carrier": None,
        "eta": None,
        "last_update": "2026-04-10",
        "notes": "Awaiting stock allocation. Not yet picked.",
    },
}

# -- Inventory -------------------------------------------------------------
# Keys: (sku, dc_id) -> {on_hand, reserved, available}
INVENTORY: dict[tuple[str, str], dict[str, Any]] = {
    ("SKU-4090", "DC-WEST-01"): {
        "sku": "SKU-4090",
        "dc_id": "DC-WEST-01",
        "on_hand": 500,
        "reserved": 200,
        "available": 300,  # 500 - 200
    },
    ("SKU-4090", "DC-EAST-02"): {
        "sku": "SKU-4090",
        "dc_id": "DC-EAST-02",
        "on_hand": 1100,
        "reserved": 100,
        "available": 1000,  # 1100 - 100
    },
    ("SKU-4090", "DC-CENTRAL-03"): {
        "sku": "SKU-4090",
        "dc_id": "DC-CENTRAL-03",
        "on_hand": 250,
        "reserved": 50,
        "available": 200,  # 250 - 50
    },
    # Substitute SKU stock at primary DC
    ("SKU-4090-B", "DC-WEST-01"): {
        "sku": "SKU-4090-B",
        "dc_id": "DC-WEST-01",
        "on_hand": 800,
        "reserved": 0,
        "available": 800,
    },
}

# -- Transfer lanes --------------------------------------------------------
# Keys: (from_dc, to_dc) -> {lead_days, cost_per_unit, max_qty}
TRANSFER_LANES: dict[tuple[str, str], dict[str, Any]] = {
    ("DC-EAST-02", "DC-WEST-01"): {
        "from_dc": "DC-EAST-02",
        "to_dc": "DC-WEST-01",
        "lead_days": 4,
        "cost_per_unit": 2.50,
        "max_qty": 2000,
    },
    ("DC-CENTRAL-03", "DC-WEST-01"): {
        "from_dc": "DC-CENTRAL-03",
        "to_dc": "DC-WEST-01",
        "lead_days": 3,
        "cost_per_unit": 1.80,
        "max_qty": 500,
    },
}

# -- Supplier expedite options ---------------------------------------------
# Keys: sku -> [{supplier, lead_days, moq, cost_per_unit, max_qty}, ...]
SUPPLIER_OPTIONS: dict[str, list[dict[str, Any]]] = {
    "SKU-4090": [
        {
            "supplier": "GlobalChip Express",
            "lead_days": 7,
            "moq": 500,
            "cost_per_unit": 8.00,
            "max_qty": 5000,
        },
        {
            "supplier": "FastSemi Direct",
            "lead_days": 5,
            "moq": 200,
            "cost_per_unit": 12.00,
            "max_qty": 1500,
        },
    ],
}

# -- Fulfillment capacity --------------------------------------------------
# Keys: (dc_id, date_str) -> {max_units, allocated, remaining}
FULFILLMENT_CAPACITY: dict[tuple[str, str], dict[str, Any]] = {
    ("DC-WEST-01", "2026-04-18"): {
        "dc_id": "DC-WEST-01",
        "date": "2026-04-18",
        "max_units": 3000,
        "allocated": 1800,
        "remaining": 1200,  # 3000 - 1800
    },
    ("DC-WEST-01", "2026-04-22"): {
        "dc_id": "DC-WEST-01",
        "date": "2026-04-22",
        "max_units": 3000,
        "allocated": 500,
        "remaining": 2500,
    },
    ("DC-EAST-02", "2026-04-18"): {
        "dc_id": "DC-EAST-02",
        "date": "2026-04-18",
        "max_units": 2000,
        "allocated": 1200,
        "remaining": 800,
    },
}

# -- Substitute SKUs -------------------------------------------------------
# Keys: sku -> [{substitute_sku, compatibility, available_dc, notes}, ...]
SUBSTITUTE_SKUS: dict[str, list[dict[str, Any]]] = {
    "SKU-4090": [
        {
            "substitute_sku": "SKU-4090-B",
            "compatibility": "full",
            "available_dc": "DC-WEST-01",
            "notes": "Same specs, different packaging. Customer pre-approved.",
        },
    ],
}

# -- Region-to-DC mapping (used by find_alternate_inventory) ---------------
REGION_DCS: dict[str, list[str]] = {
    "WEST": ["DC-WEST-01"],
    "EAST": ["DC-EAST-02"],
    "CENTRAL": ["DC-CENTRAL-03"],
    "ALL": ["DC-WEST-01", "DC-EAST-02", "DC-CENTRAL-03"],
}
