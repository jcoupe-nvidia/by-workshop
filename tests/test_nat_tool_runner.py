"""
Isolated deterministic tool tests.

Tests a representative subset of the repo's business tools using direct
function calls without spinning up the full workflow stack. This follows
NAT best practice #9 (unit-test tools in isolation).

Note: nat.test.ToolTestRunner is referenced in the NAT docs but is not
shipped in nvidia-nat 1.6.0. These tests achieve the same goal — isolated,
deterministic tool validation — via direct calls to the TOOL_REGISTRY
functions.
"""
from __future__ import annotations

from src.runtime.tools import TOOL_REGISTRY


class TestIsolatedToolCalls:
    """Run representative tools in isolation against synthetic data."""

    def test_get_order(self):
        fn, _, _ = TOOL_REGISTRY["get_order"]
        result = fn(order_id="SO-10482")
        assert result["order_id"] == "SO-10482"
        assert result["sku"] == "SKU-4090"

    def test_get_inventory(self):
        fn, _, _ = TOOL_REGISTRY["get_inventory"]
        result = fn(sku="SKU-4090", dc_id="DC-WEST-01")
        assert "available" in result
        assert isinstance(result["available"], int)

    def test_score_recovery_options(self):
        fn, _, _ = TOOL_REGISTRY["score_recovery_options"]
        options = [
            {"source": "DC-EAST-02", "lead_days": 3, "cost_per_unit": 5.0,
             "feasible": True, "covers_full_qty": True},
            {"source": "supplier:X", "lead_days": 7, "cost_per_unit": 8.0,
             "feasible": True, "covers_full_qty": True},
        ]
        result = fn(options=options, objective="minimize_delay")
        assert "ranked_options" in result
        assert result["best_option"]["source"] == "DC-EAST-02"
