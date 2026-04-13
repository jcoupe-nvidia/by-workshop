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

Ten scenarios share the same late-order recovery structure but vary in SKU,
quantity, source DC, urgency, and inventory distribution.  This forces the
agent to reason from the data rather than memorize one set of arguments.

Scenario design summary (optimal recovery action):
    SO-10482  SKU-4090  1,200 units  DC-WEST-01     -> transfer from DC-EAST-02
    SO-10483  SKU-100     500 units  DC-EAST-02     -> supplier expedite
    SO-10484  SKU-200   2,000 units  DC-CENTRAL-03  -> partial fulfillment
    SO-10485  SKU-300     200 units  DC-WEST-01     -> original DC (false alarm)
    SO-10486  SKU-400     600 units  DC-EAST-02     -> substitute SKU
    SO-10487  SKU-500   3,000 units  DC-CENTRAL-03  -> escalate (no viable option)
    SO-10488  SKU-600     800 units  DC-EAST-02     -> transfer from DC-CENTRAL-03
    SO-10489  SKU-700     350 units  DC-WEST-01     -> supplier expedite
    SO-10490  SKU-800     100 units  DC-CENTRAL-03  -> original DC (false alarm)
    SO-10491  SKU-900   1,500 units  DC-WEST-01     -> transfer from DC-EAST-02
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
    "SO-10483": {
        "order_id": "SO-10483",
        "customer": "NorthStar Logistics",
        "sku": "SKU-100",
        "qty": 500,
        "source_dc": "DC-EAST-02",
        "committed_date": "2026-04-21",
        "priority": "high",
        "region": "EAST",
    },
    "SO-10484": {
        "order_id": "SO-10484",
        "customer": "PacificRim Industries",
        "sku": "SKU-200",
        "qty": 2000,
        "source_dc": "DC-CENTRAL-03",
        "committed_date": "2026-04-15",
        "priority": "medium",
        "region": "CENTRAL",
    },
    "SO-10485": {
        "order_id": "SO-10485",
        "customer": "Greenfield Corp",
        "sku": "SKU-300",
        "qty": 200,
        "source_dc": "DC-WEST-01",
        "committed_date": "2026-04-22",
        "priority": "low",
        "region": "WEST",
    },
    "SO-10486": {
        "order_id": "SO-10486",
        "customer": "TechBridge Solutions",
        "sku": "SKU-400",
        "qty": 600,
        "source_dc": "DC-EAST-02",
        "committed_date": "2026-04-19",
        "priority": "high",
        "region": "EAST",
    },
    "SO-10487": {
        "order_id": "SO-10487",
        "customer": "Pinnacle Assembly",
        "sku": "SKU-500",
        "qty": 3000,
        "source_dc": "DC-CENTRAL-03",
        "committed_date": "2026-04-14",
        "priority": "critical",
        "region": "CENTRAL",
    },
    "SO-10488": {
        "order_id": "SO-10488",
        "customer": "Riverdale Electronics",
        "sku": "SKU-600",
        "qty": 800,
        "source_dc": "DC-EAST-02",
        "committed_date": "2026-04-22",
        "priority": "medium",
        "region": "EAST",
    },
    "SO-10489": {
        "order_id": "SO-10489",
        "customer": "Summit Precision",
        "sku": "SKU-700",
        "qty": 350,
        "source_dc": "DC-WEST-01",
        "committed_date": "2026-04-19",
        "priority": "high",
        "region": "WEST",
    },
    "SO-10490": {
        "order_id": "SO-10490",
        "customer": "LakeSide Fabrication",
        "sku": "SKU-800",
        "qty": 100,
        "source_dc": "DC-CENTRAL-03",
        "committed_date": "2026-04-25",
        "priority": "low",
        "region": "CENTRAL",
    },
    "SO-10491": {
        "order_id": "SO-10491",
        "customer": "Cascade Components",
        "sku": "SKU-900",
        "qty": 1500,
        "source_dc": "DC-WEST-01",
        "committed_date": "2026-04-24",
        "priority": "medium",
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
    "SO-10483": {
        "order_id": "SO-10483",
        "status": "pending",
        "shipped_qty": 0,
        "carrier": None,
        "eta": None,
        "last_update": "2026-04-10",
        "notes": "Awaiting stock allocation at DC-EAST-02.",
    },
    "SO-10484": {
        "order_id": "SO-10484",
        "status": "pending",
        "shipped_qty": 0,
        "carrier": None,
        "eta": None,
        "last_update": "2026-04-10",
        "notes": "Large order awaiting allocation. Tight deadline.",
    },
    "SO-10485": {
        "order_id": "SO-10485",
        "status": "in_progress",
        "shipped_qty": 0,
        "carrier": "FastFreight",
        "eta": "2026-04-20",
        "last_update": "2026-04-10",
        "notes": "Pick in progress. Stock confirmed at DC-WEST-01.",
    },
    "SO-10486": {
        "order_id": "SO-10486",
        "status": "pending",
        "shipped_qty": 0,
        "carrier": None,
        "eta": None,
        "last_update": "2026-04-10",
        "notes": "Awaiting allocation. Primary SKU nearly depleted.",
    },
    "SO-10487": {
        "order_id": "SO-10487",
        "status": "pending",
        "shipped_qty": 0,
        "carrier": None,
        "eta": None,
        "last_update": "2026-04-10",
        "notes": "Critical priority. Very tight 4-day deadline.",
    },
    "SO-10488": {
        "order_id": "SO-10488",
        "status": "pending",
        "shipped_qty": 0,
        "carrier": None,
        "eta": None,
        "last_update": "2026-04-10",
        "notes": "Awaiting allocation at DC-EAST-02.",
    },
    "SO-10489": {
        "order_id": "SO-10489",
        "status": "pending",
        "shipped_qty": 0,
        "carrier": None,
        "eta": None,
        "last_update": "2026-04-10",
        "notes": "Awaiting stock at DC-WEST-01. Inventory low.",
    },
    "SO-10490": {
        "order_id": "SO-10490",
        "status": "pending",
        "shipped_qty": 0,
        "carrier": None,
        "eta": None,
        "last_update": "2026-04-10",
        "notes": "Small order, ample lead time.",
    },
    "SO-10491": {
        "order_id": "SO-10491",
        "status": "pending",
        "shipped_qty": 0,
        "carrier": None,
        "eta": None,
        "last_update": "2026-04-10",
        "notes": "Awaiting allocation at DC-WEST-01.",
    },
}

# -- Inventory -------------------------------------------------------------
# Keys: (sku, dc_id) -> {on_hand, reserved, available}
#
# Each scenario's SKU has inventory distributed across all three DCs.
# The levels are set so that each scenario's optimal recovery action
# emerges naturally from the data.

INVENTORY: dict[tuple[str, str], dict[str, Any]] = {
    # --- SKU-4090 (SO-10482): primary DC short, DC-EAST-02 can cover ---
    ("SKU-4090", "DC-WEST-01"): {
        "sku": "SKU-4090", "dc_id": "DC-WEST-01",
        "on_hand": 500, "reserved": 200, "available": 300,
    },
    ("SKU-4090", "DC-EAST-02"): {
        "sku": "SKU-4090", "dc_id": "DC-EAST-02",
        "on_hand": 1100, "reserved": 100, "available": 1000,
    },
    ("SKU-4090", "DC-CENTRAL-03"): {
        "sku": "SKU-4090", "dc_id": "DC-CENTRAL-03",
        "on_hand": 250, "reserved": 50, "available": 200,
    },
    ("SKU-4090-B", "DC-WEST-01"): {
        "sku": "SKU-4090-B", "dc_id": "DC-WEST-01",
        "on_hand": 800, "reserved": 0, "available": 800,
    },

    # --- SKU-100 (SO-10483): alternates can't cover, supplier expedite wins ---
    ("SKU-100", "DC-EAST-02"): {
        "sku": "SKU-100", "dc_id": "DC-EAST-02",
        "on_hand": 150, "reserved": 50, "available": 100,
    },
    ("SKU-100", "DC-WEST-01"): {
        "sku": "SKU-100", "dc_id": "DC-WEST-01",
        "on_hand": 70, "reserved": 20, "available": 50,
    },
    ("SKU-100", "DC-CENTRAL-03"): {
        "sku": "SKU-100", "dc_id": "DC-CENTRAL-03",
        "on_hand": 100, "reserved": 20, "available": 80,
    },

    # --- SKU-200 (SO-10484): nowhere near enough anywhere, partial fill ---
    ("SKU-200", "DC-CENTRAL-03"): {
        "sku": "SKU-200", "dc_id": "DC-CENTRAL-03",
        "on_hand": 1000, "reserved": 200, "available": 800,
    },
    ("SKU-200", "DC-WEST-01"): {
        "sku": "SKU-200", "dc_id": "DC-WEST-01",
        "on_hand": 400, "reserved": 100, "available": 300,
    },
    ("SKU-200", "DC-EAST-02"): {
        "sku": "SKU-200", "dc_id": "DC-EAST-02",
        "on_hand": 250, "reserved": 50, "available": 200,
    },

    # --- SKU-300 (SO-10485): primary DC has plenty (false alarm) ---
    ("SKU-300", "DC-WEST-01"): {
        "sku": "SKU-300", "dc_id": "DC-WEST-01",
        "on_hand": 400, "reserved": 50, "available": 350,
    },
    ("SKU-300", "DC-EAST-02"): {
        "sku": "SKU-300", "dc_id": "DC-EAST-02",
        "on_hand": 100, "reserved": 10, "available": 90,
    },
    ("SKU-300", "DC-CENTRAL-03"): {
        "sku": "SKU-300", "dc_id": "DC-CENTRAL-03",
        "on_hand": 150, "reserved": 30, "available": 120,
    },

    # --- SKU-400 (SO-10486): primary depleted, substitute SKU saves the day ---
    ("SKU-400", "DC-EAST-02"): {
        "sku": "SKU-400", "dc_id": "DC-EAST-02",
        "on_hand": 80, "reserved": 30, "available": 50,
    },
    ("SKU-400", "DC-WEST-01"): {
        "sku": "SKU-400", "dc_id": "DC-WEST-01",
        "on_hand": 120, "reserved": 20, "available": 100,
    },
    ("SKU-400", "DC-CENTRAL-03"): {
        "sku": "SKU-400", "dc_id": "DC-CENTRAL-03",
        "on_hand": 50, "reserved": 20, "available": 30,
    },
    ("SKU-400-S", "DC-EAST-02"): {
        "sku": "SKU-400-S", "dc_id": "DC-EAST-02",
        "on_hand": 750, "reserved": 50, "available": 700,
    },

    # --- SKU-500 (SO-10487): massive shortage everywhere, escalate ---
    ("SKU-500", "DC-CENTRAL-03"): {
        "sku": "SKU-500", "dc_id": "DC-CENTRAL-03",
        "on_hand": 280, "reserved": 80, "available": 200,
    },
    ("SKU-500", "DC-WEST-01"): {
        "sku": "SKU-500", "dc_id": "DC-WEST-01",
        "on_hand": 130, "reserved": 30, "available": 100,
    },
    ("SKU-500", "DC-EAST-02"): {
        "sku": "SKU-500", "dc_id": "DC-EAST-02",
        "on_hand": 200, "reserved": 50, "available": 150,
    },

    # --- SKU-600 (SO-10488): DC-CENTRAL-03 has ample stock for transfer ---
    ("SKU-600", "DC-EAST-02"): {
        "sku": "SKU-600", "dc_id": "DC-EAST-02",
        "on_hand": 200, "reserved": 50, "available": 150,
    },
    ("SKU-600", "DC-CENTRAL-03"): {
        "sku": "SKU-600", "dc_id": "DC-CENTRAL-03",
        "on_hand": 1000, "reserved": 100, "available": 900,
    },
    ("SKU-600", "DC-WEST-01"): {
        "sku": "SKU-600", "dc_id": "DC-WEST-01",
        "on_hand": 250, "reserved": 50, "available": 200,
    },

    # --- SKU-700 (SO-10489): low stock everywhere, supplier expedite ---
    ("SKU-700", "DC-WEST-01"): {
        "sku": "SKU-700", "dc_id": "DC-WEST-01",
        "on_hand": 80, "reserved": 30, "available": 50,
    },
    ("SKU-700", "DC-EAST-02"): {
        "sku": "SKU-700", "dc_id": "DC-EAST-02",
        "on_hand": 100, "reserved": 20, "available": 80,
    },
    ("SKU-700", "DC-CENTRAL-03"): {
        "sku": "SKU-700", "dc_id": "DC-CENTRAL-03",
        "on_hand": 80, "reserved": 20, "available": 60,
    },

    # --- SKU-800 (SO-10490): primary DC has plenty (false alarm) ---
    ("SKU-800", "DC-CENTRAL-03"): {
        "sku": "SKU-800", "dc_id": "DC-CENTRAL-03",
        "on_hand": 550, "reserved": 50, "available": 500,
    },
    ("SKU-800", "DC-WEST-01"): {
        "sku": "SKU-800", "dc_id": "DC-WEST-01",
        "on_hand": 200, "reserved": 30, "available": 170,
    },
    ("SKU-800", "DC-EAST-02"): {
        "sku": "SKU-800", "dc_id": "DC-EAST-02",
        "on_hand": 300, "reserved": 40, "available": 260,
    },

    # --- SKU-900 (SO-10491): DC-EAST-02 can cover shortfall via transfer ---
    ("SKU-900", "DC-WEST-01"): {
        "sku": "SKU-900", "dc_id": "DC-WEST-01",
        "on_hand": 750, "reserved": 150, "available": 600,
    },
    ("SKU-900", "DC-EAST-02"): {
        "sku": "SKU-900", "dc_id": "DC-EAST-02",
        "on_hand": 1400, "reserved": 200, "available": 1200,
    },
    ("SKU-900", "DC-CENTRAL-03"): {
        "sku": "SKU-900", "dc_id": "DC-CENTRAL-03",
        "on_hand": 500, "reserved": 100, "available": 400,
    },
}

# -- Transfer lanes --------------------------------------------------------
# Keys: (from_dc, to_dc) -> {lead_days, cost_per_unit, max_qty}
# All six directional lanes between the three DCs.

TRANSFER_LANES: dict[tuple[str, str], dict[str, Any]] = {
    ("DC-EAST-02", "DC-WEST-01"): {
        "from_dc": "DC-EAST-02", "to_dc": "DC-WEST-01",
        "lead_days": 4, "cost_per_unit": 2.50, "max_qty": 2000,
    },
    ("DC-CENTRAL-03", "DC-WEST-01"): {
        "from_dc": "DC-CENTRAL-03", "to_dc": "DC-WEST-01",
        "lead_days": 3, "cost_per_unit": 1.80, "max_qty": 500,
    },
    ("DC-WEST-01", "DC-EAST-02"): {
        "from_dc": "DC-WEST-01", "to_dc": "DC-EAST-02",
        "lead_days": 5, "cost_per_unit": 3.00, "max_qty": 1500,
    },
    ("DC-CENTRAL-03", "DC-EAST-02"): {
        "from_dc": "DC-CENTRAL-03", "to_dc": "DC-EAST-02",
        "lead_days": 3, "cost_per_unit": 2.00, "max_qty": 1000,
    },
    ("DC-WEST-01", "DC-CENTRAL-03"): {
        "from_dc": "DC-WEST-01", "to_dc": "DC-CENTRAL-03",
        "lead_days": 4, "cost_per_unit": 2.20, "max_qty": 1200,
    },
    ("DC-EAST-02", "DC-CENTRAL-03"): {
        "from_dc": "DC-EAST-02", "to_dc": "DC-CENTRAL-03",
        "lead_days": 2, "cost_per_unit": 1.50, "max_qty": 800,
    },
}

# -- Supplier expedite options ---------------------------------------------
# Keys: sku -> [{supplier, lead_days, moq, cost_per_unit, max_qty}, ...]

SUPPLIER_OPTIONS: dict[str, list[dict[str, Any]]] = {
    "SKU-4090": [
        {
            "supplier": "GlobalChip Express",
            "lead_days": 7, "moq": 500,
            "cost_per_unit": 8.00, "max_qty": 5000,
        },
        {
            "supplier": "FastSemi Direct",
            "lead_days": 5, "moq": 200,
            "cost_per_unit": 12.00, "max_qty": 1500,
        },
    ],
    "SKU-100": [
        {
            "supplier": "RapidParts Co",
            "lead_days": 3, "moq": 100,
            "cost_per_unit": 5.50, "max_qty": 1000,
        },
    ],
    "SKU-200": [
        {
            "supplier": "Industrial Supply Co",
            "lead_days": 10, "moq": 500,
            "cost_per_unit": 4.00, "max_qty": 5000,
        },
    ],
    "SKU-300": [
        {
            "supplier": "StandardParts Ltd",
            "lead_days": 6, "moq": 50,
            "cost_per_unit": 3.00, "max_qty": 500,
        },
    ],
    "SKU-400": [
        {
            "supplier": "TechParts Direct",
            "lead_days": 8, "moq": 200,
            "cost_per_unit": 15.00, "max_qty": 2000,
        },
    ],
    "SKU-500": [
        {
            "supplier": "BulkSupply Corp",
            "lead_days": 6, "moq": 1000,
            "cost_per_unit": 6.00, "max_qty": 10000,
        },
    ],
    "SKU-600": [
        {
            "supplier": "ComponentsPlus",
            "lead_days": 7, "moq": 100,
            "cost_per_unit": 10.00, "max_qty": 1500,
        },
    ],
    "SKU-700": [
        {
            "supplier": "Express Components Ltd",
            "lead_days": 4, "moq": 50,
            "cost_per_unit": 7.00, "max_qty": 500,
        },
    ],
    "SKU-800": [
        {
            "supplier": "StandardParts Ltd",
            "lead_days": 5, "moq": 25,
            "cost_per_unit": 4.50, "max_qty": 300,
        },
    ],
    "SKU-900": [
        {
            "supplier": "BulkComponents Inc",
            "lead_days": 9, "moq": 200,
            "cost_per_unit": 8.50, "max_qty": 2000,
        },
    ],
}

# -- Fulfillment capacity --------------------------------------------------
# Keys: (dc_id, date_str) -> {max_units, allocated, remaining}

FULFILLMENT_CAPACITY: dict[tuple[str, str], dict[str, Any]] = {
    # SO-10482
    ("DC-WEST-01", "2026-04-18"): {
        "dc_id": "DC-WEST-01", "date": "2026-04-18",
        "max_units": 3000, "allocated": 1800, "remaining": 1200,
    },
    ("DC-WEST-01", "2026-04-22"): {
        "dc_id": "DC-WEST-01", "date": "2026-04-22",
        "max_units": 3000, "allocated": 500, "remaining": 2500,
    },
    ("DC-EAST-02", "2026-04-18"): {
        "dc_id": "DC-EAST-02", "date": "2026-04-18",
        "max_units": 2000, "allocated": 1200, "remaining": 800,
    },
    # SO-10483
    ("DC-EAST-02", "2026-04-21"): {
        "dc_id": "DC-EAST-02", "date": "2026-04-21",
        "max_units": 2000, "allocated": 800, "remaining": 1200,
    },
    # SO-10484
    ("DC-CENTRAL-03", "2026-04-15"): {
        "dc_id": "DC-CENTRAL-03", "date": "2026-04-15",
        "max_units": 2500, "allocated": 1500, "remaining": 1000,
    },
    # SO-10486
    ("DC-EAST-02", "2026-04-19"): {
        "dc_id": "DC-EAST-02", "date": "2026-04-19",
        "max_units": 2000, "allocated": 900, "remaining": 1100,
    },
    # SO-10487
    ("DC-CENTRAL-03", "2026-04-14"): {
        "dc_id": "DC-CENTRAL-03", "date": "2026-04-14",
        "max_units": 2500, "allocated": 2200, "remaining": 300,
    },
    # SO-10488
    ("DC-EAST-02", "2026-04-22"): {
        "dc_id": "DC-EAST-02", "date": "2026-04-22",
        "max_units": 2000, "allocated": 600, "remaining": 1400,
    },
    # SO-10489
    ("DC-WEST-01", "2026-04-19"): {
        "dc_id": "DC-WEST-01", "date": "2026-04-19",
        "max_units": 3000, "allocated": 1500, "remaining": 1500,
    },
    # SO-10490
    ("DC-CENTRAL-03", "2026-04-25"): {
        "dc_id": "DC-CENTRAL-03", "date": "2026-04-25",
        "max_units": 2500, "allocated": 400, "remaining": 2100,
    },
    # SO-10491
    ("DC-WEST-01", "2026-04-24"): {
        "dc_id": "DC-WEST-01", "date": "2026-04-24",
        "max_units": 3000, "allocated": 1000, "remaining": 2000,
    },
}

# -- Substitute SKUs -------------------------------------------------------
# Keys: sku -> [{substitute_sku, compatibility, available_dc, notes}, ...]
# Only some SKUs have substitutes — this is intentional.

SUBSTITUTE_SKUS: dict[str, list[dict[str, Any]]] = {
    "SKU-4090": [
        {
            "substitute_sku": "SKU-4090-B",
            "compatibility": "full",
            "available_dc": "DC-WEST-01",
            "notes": "Same specs, different packaging. Customer pre-approved.",
        },
    ],
    "SKU-400": [
        {
            "substitute_sku": "SKU-400-S",
            "compatibility": "full",
            "available_dc": "DC-EAST-02",
            "notes": "Identical part from alternate vendor. Customer pre-approved.",
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


# -- Scenario metadata (for training and eval) -----------------------------

ALL_ORDER_IDS: list[str] = sorted(ORDERS.keys())
