"""Tests for src.runtime.tools — deterministic business tools."""
from __future__ import annotations

import pytest

from src.runtime.tools import (
    TOOL_REGISTRY,
    TOOL_DEPENDENCIES,
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


class TestToolRegistry:
    def test_registry_has_nine_tools(self):
        assert len(TOOL_REGISTRY) == 9

    def test_registry_entry_structure(self):
        for name, (fn, params, desc) in TOOL_REGISTRY.items():
            assert callable(fn)
            assert isinstance(params, dict)
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_all_tools_in_dependency_graph(self):
        for name in TOOL_REGISTRY:
            assert name in TOOL_DEPENDENCIES


class TestGetOrder:
    def test_known_order(self):
        result = get_order(order_id="SO-10482")
        assert result["order_id"] == "SO-10482"
        assert result["sku"] == "SKU-4090"
        assert result["qty"] == 1200

    def test_unknown_order(self):
        result = get_order(order_id="SO-99999")
        assert "error" in result


class TestGetShipmentStatus:
    def test_known_order(self):
        result = get_shipment_status(order_id="SO-10482")
        assert "status" in result
        assert "shipped_qty" in result

    def test_unknown_order(self):
        result = get_shipment_status(order_id="SO-99999")
        assert "error" in result


class TestGetInventory:
    def test_known_sku_dc(self):
        result = get_inventory(sku="SKU-4090", dc_id="DC-WEST-01")
        assert "available" in result
        assert isinstance(result["available"], int)

    def test_unknown_sku(self):
        result = get_inventory(sku="SKU-FAKE", dc_id="DC-WEST-01")
        assert "error" in result


class TestFindAlternateInventory:
    def test_all_region(self):
        result = find_alternate_inventory(sku="SKU-4090", region="ALL")
        assert "total_available" in result
        assert "matching_dcs" in result
        assert isinstance(result["matching_dcs"], list)

    def test_unknown_sku(self):
        result = find_alternate_inventory(sku="SKU-FAKE", region="ALL")
        assert result["total_available"] == 0


class TestGetTransferEta:
    def test_valid_transfer(self):
        result = get_transfer_eta(
            from_dc="DC-EAST-02", to_dc="DC-WEST-01",
            sku="SKU-4090", qty=900,
        )
        assert "lead_days" in result
        assert "feasible" in result
        assert isinstance(result["lead_days"], int)

    def test_unknown_lane(self):
        result = get_transfer_eta(
            from_dc="DC-FAKE", to_dc="DC-WEST-01",
            sku="SKU-4090", qty=100,
        )
        assert "error" in result


class TestGetSupplierExpediteOptions:
    def test_known_sku(self):
        result = get_supplier_expedite_options(sku="SKU-4090", qty=900)
        assert "options" in result
        assert len(result["options"]) > 0

    def test_unknown_sku(self):
        result = get_supplier_expedite_options(sku="SKU-FAKE", qty=100)
        assert "error" in result


class TestGetFulfillmentCapacity:
    def test_known_dc_date(self):
        result = get_fulfillment_capacity(dc_id="DC-WEST-01", date="2026-04-18")
        assert "remaining" in result
        assert "max_units" in result

    def test_unknown_dc(self):
        result = get_fulfillment_capacity(dc_id="DC-FAKE", date="2026-04-18")
        assert "error" in result


class TestScoreRecoveryOptions:
    def test_minimize_delay(self):
        options = [
            {"source": "A", "lead_days": 3, "cost_per_unit": 10, "feasible": True},
            {"source": "B", "lead_days": 5, "cost_per_unit": 5, "feasible": True},
        ]
        result = score_recovery_options(options=options, objective="minimize_delay")
        assert "best_option" in result
        assert "ranked_options" in result
        assert result["best_option"]["source"] == "A"  # lower lead_days

    def test_minimize_cost(self):
        options = [
            {"source": "A", "lead_days": 3, "cost_per_unit": 10, "feasible": True},
            {"source": "B", "lead_days": 5, "cost_per_unit": 5, "feasible": True},
        ]
        result = score_recovery_options(options=options, objective="minimize_cost")
        assert result["best_option"]["source"] == "B"  # lower cost


class TestRecommendAction:
    def test_produces_recommendation(self):
        result = recommend_action(context={
            "best_option": {"source": "DC-EAST-02", "lead_days": 3},
            "order": {"order_id": "SO-10482"},
            "objective": "minimize_delay",
        })
        assert "action" in result
        assert "rationale" in result
        assert "confidence" in result


class TestToolDeterminism:
    """Tools must be deterministic over synthetic data."""

    def test_same_input_same_output(self):
        r1 = get_order(order_id="SO-10482")
        r2 = get_order(order_id="SO-10482")
        assert r1 == r2

    def test_inventory_deterministic(self):
        r1 = get_inventory(sku="SKU-4090", dc_id="DC-WEST-01")
        r2 = get_inventory(sku="SKU-4090", dc_id="DC-WEST-01")
        assert r1 == r2
