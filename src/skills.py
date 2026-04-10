"""
Explicit higher-level skills composed from deterministic tools.

Each skill has a clear purpose, boundary, and allowed tool-use pattern.
Skills are the unit the agent selects; tools are the unit skills invoke.

Skills (2-4 total):
    diagnose_order_risk(order_id)
        Purpose:  Understand the current state of the order and shipment.
        Tools:    get_order, get_shipment_status
        Output:   OrderRiskDiagnosis (is_at_risk, reason, days_remaining, ...)

    assess_primary_fulfillment(order_id)
        Purpose:  Check whether the original source DC can still fulfill.
        Tools:    get_inventory, get_fulfillment_capacity
        Output:   PrimaryAssessment (can_fulfill, shortfall, capacity_ok, ...)

    evaluate_alternate_recovery_paths(order_id)
        Purpose:  Explore alternate DCs, supplier expedite, and substitutes.
        Tools:    find_alternate_inventory, get_transfer_eta,
                  get_supplier_expedite_options
        Output:   list[RecoveryPath] (each with source, eta, cost, feasibility)

    synthesize_recommendation(order_id)
        Purpose:  Score all candidate options and produce a final recommendation.
        Tools:    score_recovery_options, recommend_action
        Output:   Recommendation (action, rationale, expected_date, confidence)

Skill transitions follow a natural diagnostic flow:
    diagnose -> assess_primary -> evaluate_alternates -> synthesize

Phase 3 will implement these skills with full tool orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# -- Skill output types (structured intermediates) -------------------------

@dataclass
class OrderRiskDiagnosis:
    """Output of diagnose_order_risk."""
    order_id: str
    is_at_risk: bool
    reason: str
    days_remaining: int
    sku: str
    qty: int
    source_dc: str
    committed_date: str

@dataclass
class PrimaryAssessment:
    """Output of assess_primary_fulfillment."""
    order_id: str
    can_fulfill: bool
    available_qty: int
    shortfall: int
    capacity_ok: bool
    source_dc: str

@dataclass
class RecoveryPath:
    """One candidate recovery option."""
    source: str          # e.g. "DC-EAST-02", "supplier-fastship", "SKU-4090-ALT"
    path_type: str       # "dc_transfer", "supplier_expedite", "substitute"
    available_qty: int
    eta_days: int
    cost_per_unit: float
    feasible: bool

@dataclass
class Recommendation:
    """Output of synthesize_recommendation."""
    order_id: str
    action: str          # e.g. "transfer_from_DC-EAST-02"
    rationale: str
    expected_delivery: str
    meets_committed_date: bool
    confidence: float
    scored_options: list[dict[str, Any]]

# -- Skill registry --------------------------------------------------------

SKILL_NAMES = [
    "diagnose_order_risk",
    "assess_primary_fulfillment",
    "evaluate_alternate_recovery_paths",
    "synthesize_recommendation",
]

# Allowed tool sequences per skill
SKILL_TOOL_PATTERNS: dict[str, list[str]] = {
    "diagnose_order_risk": ["get_order", "get_shipment_status"],
    "assess_primary_fulfillment": ["get_inventory", "get_fulfillment_capacity"],
    "evaluate_alternate_recovery_paths": [
        "find_alternate_inventory",
        "get_transfer_eta",
        "get_supplier_expedite_options",
    ],
    "synthesize_recommendation": ["score_recovery_options", "recommend_action"],
}
