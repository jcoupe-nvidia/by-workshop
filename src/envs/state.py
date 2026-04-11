"""
Explicit environment state for the late-order recovery task.

Owns:
    - What information the agent has discovered so far
    - Which subgoals are completed
    - Which tools have been called (and how many times)
    - Current recommendation candidate
    - Failure flags and invalid-action counters
    - Terminal status

Does NOT own:
    - Tool implementations or schemas (see runtime.tools, runtime.schemas)
    - Reward computation (see envs.rewards)
    - Transition logic (see envs.transitions)
    - Rollout or training concerns

The state is designed to be fully deterministic and serializable so it
can be snapshotted at any point during an episode for replay, evaluation,
or reward annotation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Subgoal(Enum):
    """Discrete subgoals in the late-order recovery task.

    These map 1:1 to the four canonical workflows/skills and represent
    the high-level diagnostic progression.
    """
    ORDER_DIAGNOSED = "order_diagnosed"
    PRIMARY_ASSESSED = "primary_assessed"
    ALTERNATES_EVALUATED = "alternates_evaluated"
    RECOMMENDATION_SYNTHESIZED = "recommendation_synthesized"


# Subgoal ordering — used to check forward progress
SUBGOAL_ORDER: list[Subgoal] = [
    Subgoal.ORDER_DIAGNOSED,
    Subgoal.PRIMARY_ASSESSED,
    Subgoal.ALTERNATES_EVALUATED,
    Subgoal.RECOMMENDATION_SYNTHESIZED,
]

# Which tools complete which subgoals (the *last* tool in each workflow)
TOOL_COMPLETES_SUBGOAL: dict[str, Subgoal] = {
    "get_shipment_status": Subgoal.ORDER_DIAGNOSED,
    "get_fulfillment_capacity": Subgoal.PRIMARY_ASSESSED,
    "get_supplier_expedite_options": Subgoal.ALTERNATES_EVALUATED,
    "recommend_action": Subgoal.RECOMMENDATION_SYNTHESIZED,
}

# Which tools contribute to which subgoals (all tools in a workflow)
TOOL_TO_SUBGOAL: dict[str, Subgoal] = {
    "get_order": Subgoal.ORDER_DIAGNOSED,
    "get_shipment_status": Subgoal.ORDER_DIAGNOSED,
    "get_inventory": Subgoal.PRIMARY_ASSESSED,
    "get_fulfillment_capacity": Subgoal.PRIMARY_ASSESSED,
    "find_alternate_inventory": Subgoal.ALTERNATES_EVALUATED,
    "get_transfer_eta": Subgoal.ALTERNATES_EVALUATED,
    "get_supplier_expedite_options": Subgoal.ALTERNATES_EVALUATED,
    "score_recovery_options": Subgoal.RECOMMENDATION_SYNTHESIZED,
    "recommend_action": Subgoal.RECOMMENDATION_SYNTHESIZED,
}

# Sequence dependency graph: tool_name -> set of tools that must precede it.
# This is the canonical, machine-checkable dependency contract owned by
# envs/ as part of task truth.  Runtime and eval import from here.
TOOL_DEPENDENCIES: dict[str, set[str]] = {
    "get_order": set(),
    "get_shipment_status": {"get_order"},
    "get_inventory": {"get_order"},
    "find_alternate_inventory": {"get_inventory"},
    "get_transfer_eta": {"find_alternate_inventory"},
    "get_supplier_expedite_options": {"get_order"},
    "get_fulfillment_capacity": {"get_order"},
    "score_recovery_options": {"find_alternate_inventory"},
    "recommend_action": {"score_recovery_options"},
}


@dataclass
class LateOrderEnvState:
    """Complete environment state for one late-order recovery episode.

    Fields are grouped by concern:
        - Task identity: order_id, sku, qty, source_dc, committed_date
        - Discovery facts: what the agent has learned from tool results
        - Subgoal tracking: which diagnostic phases are done
        - Tool call history: names called, call counts, total steps
        - Recommendation state: current candidate and scoring
        - Failure tracking: invalid actions, malformed calls, repairs
        - Terminal status: whether and why the episode ended
    """

    # -- Task identity (set at reset, immutable during episode) ----------------
    order_id: str = ""
    sku: str = ""
    qty: int = 0
    source_dc: str = ""
    committed_date: str = ""
    region: str = ""

    # -- Discovery facts (populated as tools return results) -------------------
    order_found: bool = False
    shipment_status: str | None = None
    source_dc_available: int | None = None
    source_dc_shortfall: int | None = None
    source_dc_capacity_ok: bool | None = None
    alternate_dcs_found: list[str] = field(default_factory=list)
    alternate_total_available: int | None = None
    supplier_options_found: int | None = None
    substitute_skus_found: int | None = None
    transfer_etas_checked: list[str] = field(default_factory=list)

    # -- Subgoal tracking ------------------------------------------------------
    completed_subgoals: set[Subgoal] = field(default_factory=set)

    # -- Tool call history -----------------------------------------------------
    tools_called: list[str] = field(default_factory=list)
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    total_steps: int = 0  # includes invalid steps

    # -- Recommendation state --------------------------------------------------
    recovery_options_scored: bool = False
    recommendation_candidate: dict[str, Any] | None = None

    # -- Failure tracking ------------------------------------------------------
    invalid_action_count: int = 0
    malformed_call_count: int = 0
    dependency_violation_count: int = 0
    repair_attempt_count: int = 0
    repair_success_count: int = 0
    reject_count: int = 0
    redundant_call_count: int = 0

    # -- Terminal status -------------------------------------------------------
    is_terminal: bool = False
    terminal_reason: str | None = None  # "final_answer", "max_iterations", "error"
    final_answer: dict[str, Any] | None = None

    # -- Convenience helpers ---------------------------------------------------

    @property
    def tools_called_set(self) -> set[str]:
        """Unique tools that have been successfully called."""
        return set(self.tools_called)

    @property
    def next_expected_subgoal(self) -> Subgoal | None:
        """The next subgoal in order that hasn't been completed yet."""
        for sg in SUBGOAL_ORDER:
            if sg not in self.completed_subgoals:
                return sg
        return None

    @property
    def all_subgoals_complete(self) -> bool:
        return len(self.completed_subgoals) == len(SUBGOAL_ORDER)

    @property
    def valid_tool_call_count(self) -> int:
        return len(self.tools_called)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a plain dict for snapshots and logging."""
        return {
            "order_id": self.order_id,
            "sku": self.sku,
            "qty": self.qty,
            "source_dc": self.source_dc,
            "committed_date": self.committed_date,
            "region": self.region,
            "order_found": self.order_found,
            "shipment_status": self.shipment_status,
            "source_dc_available": self.source_dc_available,
            "source_dc_shortfall": self.source_dc_shortfall,
            "source_dc_capacity_ok": self.source_dc_capacity_ok,
            "alternate_dcs_found": list(self.alternate_dcs_found),
            "alternate_total_available": self.alternate_total_available,
            "supplier_options_found": self.supplier_options_found,
            "substitute_skus_found": self.substitute_skus_found,
            "transfer_etas_checked": list(self.transfer_etas_checked),
            "completed_subgoals": [sg.value for sg in self.completed_subgoals],
            "tools_called": list(self.tools_called),
            "tool_call_counts": dict(self.tool_call_counts),
            "total_steps": self.total_steps,
            "recovery_options_scored": self.recovery_options_scored,
            "recommendation_candidate": self.recommendation_candidate,
            "invalid_action_count": self.invalid_action_count,
            "malformed_call_count": self.malformed_call_count,
            "dependency_violation_count": self.dependency_violation_count,
            "repair_attempt_count": self.repair_attempt_count,
            "repair_success_count": self.repair_success_count,
            "reject_count": self.reject_count,
            "redundant_call_count": self.redundant_call_count,
            "is_terminal": self.is_terminal,
            "terminal_reason": self.terminal_reason,
            "final_answer": self.final_answer,
        }


def make_initial_state(order_id: str) -> LateOrderEnvState:
    """Create the initial environment state for a given order.

    Loads task identity from scenario_data but does NOT pre-populate
    any discovery facts — those are revealed through tool calls.
    """
    from src.scenario_data import ORDERS

    if order_id not in ORDERS:
        raise ValueError(f"Unknown order: {order_id}")

    order = ORDERS[order_id]
    return LateOrderEnvState(
        order_id=order_id,
        sku=order["sku"],
        qty=order["qty"],
        source_dc=order["source_dc"],
        committed_date=order["committed_date"],
        region=order.get("region", ""),
    )
