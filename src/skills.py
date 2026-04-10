"""
Explicit higher-level skills composed from deterministic tools.

Each skill has a clear purpose, boundary, and allowed tool-use pattern.
Skills are the unit the agent selects; tools are the unit skills invoke.

Skills (4 total):
    diagnose_order_risk(ctx)
        Purpose:  Understand the current state of the order and shipment.
        Tools:    get_order, get_shipment_status
        Output:   OrderRiskDiagnosis (is_at_risk, reason, days_remaining, ...)

    assess_primary_fulfillment(ctx)
        Purpose:  Check whether the original source DC can still fulfill.
        Tools:    get_inventory, get_fulfillment_capacity
        Output:   PrimaryAssessment (can_fulfill, shortfall, capacity_ok, ...)

    evaluate_alternate_recovery_paths(ctx)
        Purpose:  Explore alternate DCs, supplier expedite, and substitutes.
        Tools:    find_alternate_inventory, get_transfer_eta,
                  get_supplier_expedite_options
        Output:   list[RecoveryPath] (each with source, eta, cost, feasibility)

    synthesize_recommendation(ctx)
        Purpose:  Score all candidate options and produce a final recommendation.
        Tools:    score_recovery_options, recommend_action
        Output:   Recommendation (action, rationale, expected_date, confidence)

Skill transitions follow a natural diagnostic flow:
    diagnose -> assess_primary -> evaluate_alternates -> synthesize
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.tools import TOOL_REGISTRY

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


# -- Skill execution tracing -----------------------------------------------

@dataclass
class SkillToolCall:
    """Record of a single tool call made during skill execution."""
    skill_name: str
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class SkillContext:
    """Accumulated state across skill executions.

    Passed through the skill chain so each skill can read prior results
    and all tool calls are recorded for evaluation.
    """
    order_id: str
    diagnosis: OrderRiskDiagnosis | None = None
    primary: PrimaryAssessment | None = None
    recovery_paths: list[RecoveryPath] = field(default_factory=list)
    recommendation: Recommendation | None = None
    tool_calls: list[SkillToolCall] = field(default_factory=list)
    skills_executed: list[str] = field(default_factory=list)


def _call_tool(ctx: SkillContext, skill_name: str, tool_name: str, **kwargs: Any) -> dict[str, Any]:
    """Dispatch a tool call via TOOL_REGISTRY and record it in the context."""
    fn, _params, _desc = TOOL_REGISTRY[tool_name]
    result = fn(**kwargs)
    ctx.tool_calls.append(SkillToolCall(
        skill_name=skill_name,
        tool_name=tool_name,
        arguments=kwargs,
        result=result,
    ))
    return result


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

# Valid skill transitions: skill -> set of skills that may follow it
SKILL_TRANSITIONS: dict[str, set[str]] = {
    "diagnose_order_risk": {"assess_primary_fulfillment"},
    "assess_primary_fulfillment": {"evaluate_alternate_recovery_paths"},
    "evaluate_alternate_recovery_paths": {"synthesize_recommendation"},
    "synthesize_recommendation": set(),  # terminal
}

# Canonical ordering for the diagnostic flow
SKILL_ORDER = [
    "diagnose_order_risk",
    "assess_primary_fulfillment",
    "evaluate_alternate_recovery_paths",
    "synthesize_recommendation",
]


def validate_skill_transition(current: str | None, next_skill: str) -> tuple[bool, str]:
    """Check whether transitioning to next_skill is valid.

    Returns (is_valid, reason).
    """
    if current is None:
        if next_skill == SKILL_ORDER[0]:
            return True, "Starting with the first skill."
        return False, f"Must start with '{SKILL_ORDER[0]}', not '{next_skill}'."

    allowed = SKILL_TRANSITIONS.get(current, set())
    if next_skill in allowed:
        return True, f"Valid transition: {current} -> {next_skill}."
    return False, f"Invalid transition: {current} -> {next_skill}. Allowed: {allowed}."


# -- Skill implementations ------------------------------------------------

def diagnose_order_risk(ctx: SkillContext) -> OrderRiskDiagnosis:
    """Understand the current state of the order and shipment.

    Calls: get_order, get_shipment_status
    Preconditions: none (this is the entry skill)
    """
    skill = "diagnose_order_risk"

    # Step 1: Look up the order
    order = _call_tool(ctx, skill, "get_order", order_id=ctx.order_id)
    if "error" in order:
        raise ValueError(f"Order lookup failed: {order['error']}")

    # Step 2: Check shipment status
    shipment = _call_tool(ctx, skill, "get_shipment_status", order_id=ctx.order_id)
    if "error" in shipment:
        raise ValueError(f"Shipment lookup failed: {shipment['error']}")

    # Determine risk: is the order at risk of missing the committed date?
    committed = datetime.strptime(order["committed_date"], "%Y-%m-%d")
    today = datetime(2026, 4, 10)
    days_remaining = (committed - today).days

    is_at_risk = (
        shipment["status"] in ("pending", "delayed", "partial")
        and days_remaining <= 10
    )

    if shipment["status"] == "pending" and shipment["shipped_qty"] == 0:
        reason = f"Order not yet shipped. {days_remaining} days until committed date."
    elif shipment["status"] == "delayed":
        reason = f"Shipment delayed. {days_remaining} days until committed date."
    elif shipment["status"] == "partial":
        reason = f"Only {shipment['shipped_qty']}/{order['qty']} shipped. {days_remaining} days remaining."
    else:
        reason = f"Shipment status: {shipment['status']}. {days_remaining} days remaining."

    diagnosis = OrderRiskDiagnosis(
        order_id=ctx.order_id,
        is_at_risk=is_at_risk,
        reason=reason,
        days_remaining=days_remaining,
        sku=order["sku"],
        qty=order["qty"],
        source_dc=order["source_dc"],
        committed_date=order["committed_date"],
    )
    ctx.diagnosis = diagnosis
    ctx.skills_executed.append(skill)
    return diagnosis


def assess_primary_fulfillment(ctx: SkillContext) -> PrimaryAssessment:
    """Check whether the original source DC can still fulfill the order.

    Calls: get_inventory, get_fulfillment_capacity
    Preconditions: diagnose_order_risk must have run (need sku, source_dc, committed_date)
    """
    skill = "assess_primary_fulfillment"

    if ctx.diagnosis is None:
        raise ValueError("Cannot assess primary fulfillment without a diagnosis. Run diagnose_order_risk first.")

    diag = ctx.diagnosis

    # Step 1: Check inventory at source DC
    inv = _call_tool(ctx, skill, "get_inventory", sku=diag.sku, dc_id=diag.source_dc)
    available = inv.get("available", 0)
    shortfall = max(0, diag.qty - available)

    # Step 2: Check fulfillment capacity at source DC on committed date
    cap = _call_tool(ctx, skill, "get_fulfillment_capacity",
                     dc_id=diag.source_dc, date=diag.committed_date)
    capacity_ok = cap.get("remaining", 0) >= diag.qty if "error" not in cap else False

    can_fulfill = shortfall == 0 and capacity_ok

    assessment = PrimaryAssessment(
        order_id=ctx.order_id,
        can_fulfill=can_fulfill,
        available_qty=available,
        shortfall=shortfall,
        capacity_ok=capacity_ok,
        source_dc=diag.source_dc,
    )
    ctx.primary = assessment
    ctx.skills_executed.append(skill)
    return assessment


def evaluate_alternate_recovery_paths(ctx: SkillContext) -> list[RecoveryPath]:
    """Explore alternate DCs, supplier expedite, and substitutes.

    Calls: find_alternate_inventory, get_transfer_eta, get_supplier_expedite_options
    Preconditions: assess_primary_fulfillment must have run (need shortfall info)
    """
    skill = "evaluate_alternate_recovery_paths"

    if ctx.diagnosis is None or ctx.primary is None:
        raise ValueError("Cannot evaluate alternates without diagnosis and primary assessment.")

    diag = ctx.diagnosis
    primary = ctx.primary
    paths: list[RecoveryPath] = []

    # Step 1: Find alternate inventory across all regions
    alt_inv = _call_tool(ctx, skill, "find_alternate_inventory",
                         sku=diag.sku, region="ALL")

    # Step 2: For each alternate DC with stock, get transfer ETA
    for dc_record in alt_inv.get("matching_dcs", []):
        dc_id = dc_record["dc_id"]
        # Skip the primary source DC -- we already checked it
        if dc_id == diag.source_dc:
            continue

        transfer = _call_tool(ctx, skill, "get_transfer_eta",
                              from_dc=dc_id, to_dc=diag.source_dc,
                              sku=diag.sku, qty=primary.shortfall)

        if "error" not in transfer:
            paths.append(RecoveryPath(
                source=dc_id,
                path_type="dc_transfer",
                available_qty=transfer["available_at_source"],
                eta_days=transfer["lead_days"],
                cost_per_unit=transfer["cost_per_unit"],
                feasible=transfer["feasible"],
            ))

    # Step 3: Get supplier expedite options
    supplier = _call_tool(ctx, skill, "get_supplier_expedite_options",
                          sku=diag.sku, qty=primary.shortfall)

    for opt in supplier.get("options", []):
        paths.append(RecoveryPath(
            source=f"supplier:{opt['supplier']}",
            path_type="supplier_expedite",
            available_qty=opt["max_qty"],
            eta_days=opt["lead_days"],
            cost_per_unit=opt["cost_per_unit"],
            feasible=opt["feasible"],
        ))

    ctx.recovery_paths = paths
    ctx.skills_executed.append(skill)
    return paths


def synthesize_recommendation(ctx: SkillContext) -> Recommendation:
    """Score all candidate options and produce a final recommendation.

    Calls: score_recovery_options, recommend_action
    Preconditions: evaluate_alternate_recovery_paths must have run
    """
    skill = "synthesize_recommendation"

    if not ctx.recovery_paths:
        raise ValueError("Cannot synthesize recommendation without recovery paths.")
    if ctx.diagnosis is None:
        raise ValueError("Cannot synthesize recommendation without diagnosis.")

    diag = ctx.diagnosis

    # Build scorable options list from recovery paths
    options = []
    for path in ctx.recovery_paths:
        options.append({
            "source": path.source,
            "path_type": path.path_type,
            "description": f"{path.path_type} from {path.source}",
            "available_qty": path.available_qty,
            "lead_days": path.eta_days,
            "cost_per_unit": path.cost_per_unit,
            "total_cost": round(path.cost_per_unit * diag.qty, 2),
            "feasible": path.feasible,
            "covers_full_qty": path.available_qty >= (ctx.primary.shortfall if ctx.primary else diag.qty),
        })

    # Step 1: Score and rank the options
    scored = _call_tool(ctx, skill, "score_recovery_options",
                        options=options, objective="minimize_delay")

    # Step 2: Produce final recommendation
    best = scored.get("best_option")
    order_data = {
        "order_id": ctx.order_id,
        "sku": diag.sku,
        "qty": diag.qty,
        "committed_date": diag.committed_date,
    }
    rec_result = _call_tool(ctx, skill, "recommend_action",
                            context={
                                "best_option": best,
                                "order": order_data,
                                "objective": "minimize_delay",
                            })

    recommendation = Recommendation(
        order_id=ctx.order_id,
        action=rec_result.get("action", "unknown"),
        rationale=rec_result.get("rationale", ""),
        expected_delivery=rec_result.get("expected_delivery", "unknown"),
        meets_committed_date=rec_result.get("meets_committed_date", False),
        confidence=rec_result.get("confidence", 0.0),
        scored_options=scored.get("ranked_options", []),
    )
    ctx.recommendation = recommendation
    ctx.skills_executed.append(skill)
    return recommendation


# -- Skill function registry -----------------------------------------------

SKILL_REGISTRY: dict[str, Any] = {
    "diagnose_order_risk": diagnose_order_risk,
    "assess_primary_fulfillment": assess_primary_fulfillment,
    "evaluate_alternate_recovery_paths": evaluate_alternate_recovery_paths,
    "synthesize_recommendation": synthesize_recommendation,
}


# -- Full diagnostic flow --------------------------------------------------

def run_diagnostic_flow(order_id: str) -> SkillContext:
    """Execute the complete diagnostic flow for an order.

    Runs all four skills in sequence, accumulating state in a SkillContext.
    Returns the context with all intermediate results and tool call records.
    """
    ctx = SkillContext(order_id=order_id)

    for skill_name in SKILL_ORDER:
        # Validate the transition
        previous = ctx.skills_executed[-1] if ctx.skills_executed else None
        valid, reason = validate_skill_transition(previous, skill_name)
        if not valid:
            raise ValueError(f"Skill transition error: {reason}")

        # Execute the skill
        SKILL_REGISTRY[skill_name](ctx)

    return ctx
