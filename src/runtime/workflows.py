"""
Runtime workflow decomposition for the late-order recovery scenario.

Each workflow has a clear purpose, boundary, and allowed tool-use pattern.
Workflows are the unit the agent selects; tools are the unit workflows invoke.

Owns:
    - Workflow output types (structured intermediates)
    - Workflow execution context and tool dispatch
    - Workflow registry, transitions, and ordering
    - Allowed tool-use patterns per workflow

Does NOT own:
    - Tool implementations (see runtime.tools)
    - Prompt or orchestration policy (see runtime.prompts, runtime.agent)
    - Reward semantics or training concerns
    - Rollout or environment state

Workflows (4 total):
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

Workflow transitions follow a natural diagnostic flow:
    diagnose -> assess_primary -> evaluate_alternates -> synthesize
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.runtime.tools import TOOL_REGISTRY

# -- Workflow output types (structured intermediates) -------------------------

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


# -- Workflow execution tracing -----------------------------------------------

@dataclass
class WorkflowToolCall:
    """Record of a single tool call made during workflow execution."""
    workflow_name: str
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]

    @property
    def skill_name(self) -> str:
        """Backward-compatible alias for workflow_name."""
        return self.workflow_name


@dataclass
class WorkflowContext:
    """Accumulated state across workflow executions.

    Passed through the workflow chain so each workflow can read prior results
    and all tool calls are recorded for evaluation.
    """
    order_id: str
    diagnosis: OrderRiskDiagnosis | None = None
    primary: PrimaryAssessment | None = None
    recovery_paths: list[RecoveryPath] = field(default_factory=list)
    recommendation: Recommendation | None = None
    tool_calls: list[WorkflowToolCall] = field(default_factory=list)
    workflows_executed: list[str] = field(default_factory=list)

    @property
    def skills_executed(self) -> list[str]:
        """Backward-compatible alias for workflows_executed."""
        return self.workflows_executed


def _call_tool(ctx: WorkflowContext, workflow_name: str, tool_name: str, **kwargs: Any) -> dict[str, Any]:
    """Dispatch a tool call via TOOL_REGISTRY and record it in the context."""
    fn, _params, _desc = TOOL_REGISTRY[tool_name]
    result = fn(**kwargs)
    ctx.tool_calls.append(WorkflowToolCall(
        workflow_name=workflow_name,
        tool_name=tool_name,
        arguments=kwargs,
        result=result,
    ))
    return result


# -- Workflow registry (derived from directory-backed skills + optional YAML) -

from src.runtime.skills.api import (
    build_skill_order,
    build_skill_tool_patterns,
    build_skill_transitions,
    list_skills,
)


def _try_load_workflow_yaml() -> dict | None:
    """Attempt to load declarative workflow config from workflow_config.yaml.

    Returns parsed YAML dict or None if the file is missing. This
    demonstrates NAT best practice #4: declarative YAML configuration
    for workflows, models, middleware, and runtime variants.
    """
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), "workflow_config.yaml")
    if not os.path.exists(yaml_path):
        return None
    try:
        import yaml
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _build_registries_from_yaml(cfg: dict) -> tuple[list[str], dict, dict]:
    """Derive workflow registries from YAML config."""
    order = cfg.get("ordering", [])
    tool_patterns = {}
    transitions = {}
    for wf in cfg.get("workflows", []):
        tool_patterns[wf["name"]] = set(wf.get("tools", []))
    for wf_name, targets in cfg.get("transitions", {}).items():
        transitions[wf_name] = set(targets) if targets else set()
    return order, tool_patterns, transitions


_yaml_cfg = _try_load_workflow_yaml()
if _yaml_cfg is not None:
    WORKFLOW_ORDER, WORKFLOW_TOOL_PATTERNS, WORKFLOW_TRANSITIONS = (
        _build_registries_from_yaml(_yaml_cfg)
    )
else:
    WORKFLOW_ORDER = build_skill_order()
    WORKFLOW_TOOL_PATTERNS = build_skill_tool_patterns()
    WORKFLOW_TRANSITIONS = build_skill_transitions()

WORKFLOW_NAMES = WORKFLOW_ORDER

# Backward-compatible aliases for code that still uses "skill" naming
SKILL_NAMES = WORKFLOW_NAMES
SKILL_TOOL_PATTERNS = WORKFLOW_TOOL_PATTERNS
SKILL_TRANSITIONS = WORKFLOW_TRANSITIONS
SKILL_ORDER = WORKFLOW_ORDER


def validate_workflow_transition(current: str | None, next_workflow: str) -> tuple[bool, str]:
    """Check whether transitioning to next_workflow is valid.

    Returns (is_valid, reason).
    """
    if current is None:
        if next_workflow == WORKFLOW_ORDER[0]:
            return True, "Starting with the first workflow."
        return False, f"Must start with '{WORKFLOW_ORDER[0]}', not '{next_workflow}'."

    allowed = WORKFLOW_TRANSITIONS.get(current, set())
    if next_workflow in allowed:
        return True, f"Valid transition: {current} -> {next_workflow}."
    return False, f"Invalid transition: {current} -> {next_workflow}. Allowed: {allowed}."


# Backward-compatible alias
validate_skill_transition = validate_workflow_transition


# -- Workflow implementations ------------------------------------------------

def diagnose_order_risk(ctx: WorkflowContext) -> OrderRiskDiagnosis:
    """Understand the current state of the order and shipment.

    Calls: get_order, get_shipment_status
    Preconditions: none (this is the entry workflow)
    """
    workflow = "diagnose_order_risk"

    # Step 1: Look up the order
    order = _call_tool(ctx, workflow, "get_order", order_id=ctx.order_id)
    if "error" in order:
        raise ValueError(f"Order lookup failed: {order['error']}")

    # Step 2: Check shipment status
    shipment = _call_tool(ctx, workflow, "get_shipment_status", order_id=ctx.order_id)
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
    ctx.workflows_executed.append(workflow)
    return diagnosis


def assess_primary_fulfillment(ctx: WorkflowContext) -> PrimaryAssessment:
    """Check whether the original source DC can still fulfill the order.

    Calls: get_inventory, get_fulfillment_capacity
    Preconditions: diagnose_order_risk must have run (need sku, source_dc, committed_date)
    """
    workflow = "assess_primary_fulfillment"

    if ctx.diagnosis is None:
        raise ValueError("Cannot assess primary fulfillment without a diagnosis. Run diagnose_order_risk first.")

    diag = ctx.diagnosis

    # Step 1: Check inventory at source DC
    inv = _call_tool(ctx, workflow, "get_inventory", sku=diag.sku, dc_id=diag.source_dc)
    available = inv.get("available", 0)
    shortfall = max(0, diag.qty - available)

    # Step 2: Check fulfillment capacity at source DC on committed date
    cap = _call_tool(ctx, workflow, "get_fulfillment_capacity",
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
    ctx.workflows_executed.append(workflow)
    return assessment


def evaluate_alternate_recovery_paths(ctx: WorkflowContext) -> list[RecoveryPath]:
    """Explore alternate DCs, supplier expedite, and substitutes.

    Calls: find_alternate_inventory, get_transfer_eta, get_supplier_expedite_options
    Preconditions: assess_primary_fulfillment must have run (need shortfall info)
    """
    workflow = "evaluate_alternate_recovery_paths"

    if ctx.diagnosis is None or ctx.primary is None:
        raise ValueError("Cannot evaluate alternates without diagnosis and primary assessment.")

    diag = ctx.diagnosis
    primary = ctx.primary
    paths: list[RecoveryPath] = []

    # Step 1: Find alternate inventory across all regions
    alt_inv = _call_tool(ctx, workflow, "find_alternate_inventory",
                         sku=diag.sku, region="ALL")

    # Step 2: For each alternate DC with stock, get transfer ETA
    for dc_record in alt_inv.get("matching_dcs", []):
        dc_id = dc_record["dc_id"]
        # Skip the primary source DC -- we already checked it
        if dc_id == diag.source_dc:
            continue

        transfer = _call_tool(ctx, workflow, "get_transfer_eta",
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
    supplier = _call_tool(ctx, workflow, "get_supplier_expedite_options",
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

    # Step 4: Consider substitute SKUs returned by find_alternate_inventory
    for sub in alt_inv.get("substitutes", []):
        sub_sku = sub.get("sku", sub.get("substitute_sku", "unknown"))
        sub_dc = sub.get("dc_id", "unknown")
        sub_available = sub.get("available", 0)
        compatibility = sub.get("compatibility", "unknown")
        feasible = (
            sub_available >= primary.shortfall
            and compatibility == "full"
        )
        paths.append(RecoveryPath(
            source=f"substitute:{sub_sku}@{sub_dc}",
            path_type="substitute",
            available_qty=sub_available,
            eta_days=0,  # same DC, no transfer needed
            cost_per_unit=0.0,  # same cost basis as original
            feasible=feasible,
        ))

    ctx.recovery_paths = paths
    ctx.workflows_executed.append(workflow)
    return paths


def synthesize_recommendation(ctx: WorkflowContext) -> Recommendation:
    """Score all candidate options and produce a final recommendation.

    Calls: score_recovery_options, recommend_action
    Preconditions: evaluate_alternate_recovery_paths must have run
    """
    workflow = "synthesize_recommendation"

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
    scored = _call_tool(ctx, workflow, "score_recovery_options",
                        options=options, objective="minimize_delay")

    # Step 2: Produce final recommendation
    best = scored.get("best_option")
    order_data = {
        "order_id": ctx.order_id,
        "sku": diag.sku,
        "qty": diag.qty,
        "committed_date": diag.committed_date,
    }
    rec_result = _call_tool(ctx, workflow, "recommend_action",
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
    ctx.workflows_executed.append(workflow)
    return recommendation


# -- Workflow function registry -----------------------------------------------

WORKFLOW_REGISTRY: dict[str, Any] = {
    "diagnose_order_risk": diagnose_order_risk,
    "assess_primary_fulfillment": assess_primary_fulfillment,
    "evaluate_alternate_recovery_paths": evaluate_alternate_recovery_paths,
    "synthesize_recommendation": synthesize_recommendation,
}

# Backward-compatible alias
SKILL_REGISTRY = WORKFLOW_REGISTRY


# -- Full diagnostic flow --------------------------------------------------

def run_diagnostic_flow(order_id: str) -> WorkflowContext:
    """Execute the complete diagnostic flow for an order.

    Runs all four workflows in sequence, accumulating state in a WorkflowContext.
    Returns the context with all intermediate results and tool call records.
    """
    ctx = WorkflowContext(order_id=order_id)

    for workflow_name in WORKFLOW_ORDER:
        # Validate the transition
        previous = ctx.workflows_executed[-1] if ctx.workflows_executed else None
        valid, reason = validate_workflow_transition(previous, workflow_name)
        if not valid:
            raise ValueError(f"Workflow transition error: {reason}")

        # Execute the workflow
        WORKFLOW_REGISTRY[workflow_name](ctx)

    return ctx
