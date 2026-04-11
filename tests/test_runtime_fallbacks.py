"""Tests for src.runtime.fallbacks — repair and reject logic."""
from __future__ import annotations

import pytest

from src.runtime.fallbacks import try_repair, parse_with_fallback, FallbackAction, FallbackResult
from src.runtime.tools import TOOL_REGISTRY


class TestTryRepair:
    def test_trailing_commas(self):
        raw = '{"tool_call": {"name": "get_order", "arguments": {"order_id": "SO-10482",}}}'
        result = try_repair(raw, TOOL_REGISTRY)
        assert result.action == FallbackAction.REPAIRED
        assert "fixed_trailing_commas" in result.repairs_applied

    def test_single_quotes(self):
        raw = "{'tool_call': {'name': 'get_order', 'arguments': {'order_id': 'SO-10482'}}}"
        result = try_repair(raw, TOOL_REGISTRY)
        assert result.action == FallbackAction.REPAIRED

    def test_markdown_fence(self):
        raw = '```json\n{"tool_call": {"name": "get_order", "arguments": {"order_id": "SO-10482"}}}\n```'
        result = try_repair(raw, TOOL_REGISTRY)
        assert result.action == FallbackAction.REPAIRED

    def test_truncated_json(self):
        raw = '{"tool_call": {"name": "get_order", "arguments": {"order_id": "SO-10482"}'
        result = try_repair(raw, TOOL_REGISTRY)
        assert result.action == FallbackAction.REPAIRED

    def test_fuzzy_tool_name(self):
        raw = '{"tool_call": {"name": "get_ordr", "arguments": {"order_id": "SO-10482"}}}'
        result = try_repair(raw, TOOL_REGISTRY)
        assert result.action == FallbackAction.REPAIRED

    def test_flat_tool_call(self):
        raw = '{"name": "get_order", "arguments": {"order_id": "SO-10482"}}'
        result = try_repair(raw, TOOL_REGISTRY)
        assert result.action == FallbackAction.REPAIRED

    def test_no_json_rejected(self):
        raw = "I will now check the order status."
        result = try_repair(raw, TOOL_REGISTRY)
        assert result.action == FallbackAction.REJECTED
        assert result.rejection_reason is not None

    def test_unknown_tool_rejected(self):
        raw = '{"tool_call": {"name": "query_database", "arguments": {"sql": "SELECT *"}}}'
        result = try_repair(raw, TOOL_REGISTRY)
        assert result.action == FallbackAction.REJECTED

    def test_missing_args_rejected(self):
        raw = '{"tool_call": {"name": "get_inventory", "arguments": {"sku": "SKU-4090"}}}'
        result = try_repair(raw, TOOL_REGISTRY)
        assert result.action == FallbackAction.REJECTED


class TestParseWithFallback:
    def test_valid_input_no_fallback(self):
        raw = '{"tool_call": {"name": "get_order", "arguments": {"order_id": "SO-10482"}}}'
        parsed, fb = parse_with_fallback(raw, TOOL_REGISTRY)
        assert parsed is not None
        assert fb.action == FallbackAction.NO_ACTION

    def test_repairable_input(self):
        raw = '{"tool_call": {"name": "get_order", "arguments": {"order_id": "SO-10482",}}}'
        parsed, fb = parse_with_fallback(raw, TOOL_REGISTRY)
        assert parsed is not None
        assert fb.action == FallbackAction.REPAIRED
