"""Tests for src.runtime.schemas — tool-call validation."""
from __future__ import annotations

import pytest

from src.runtime.schemas import (
    validate_tool_call,
    extract_json,
    ParsedToolCall,
    ParsedFinalAnswer,
    ValidationError,
)
from src.runtime.tools import TOOL_REGISTRY


class TestExtractJson:
    def test_pure_json(self):
        raw = '{"tool_call": {"name": "get_order", "arguments": {"order_id": "SO-10482"}}}'
        assert extract_json(raw) is not None

    def test_json_in_text(self):
        raw = 'Let me check.\n{"tool_call": {"name": "get_order", "arguments": {"order_id": "SO-10482"}}}\nDone.'
        assert extract_json(raw) is not None

    def test_no_json(self):
        assert extract_json("I think we should check the order.") is None


class TestValidateToolCall:
    def test_valid_minimal(self):
        raw = '{"tool_call": {"name": "get_order", "arguments": {"order_id": "SO-10482"}}}'
        result = validate_tool_call(raw, TOOL_REGISTRY)
        assert isinstance(result, ParsedToolCall)
        assert result.tool_name == "get_order"
        assert result.arguments == {"order_id": "SO-10482"}

    def test_valid_with_thought(self):
        raw = '{"thought": "checking", "tool_call": {"name": "get_order", "arguments": {"order_id": "SO-10482"}}}'
        result = validate_tool_call(raw, TOOL_REGISTRY)
        assert isinstance(result, ParsedToolCall)
        assert result.thought == "checking"

    def test_valid_final_answer(self):
        raw = '{"thought": "done", "final_answer": {"action": "transfer"}}'
        result = validate_tool_call(raw, TOOL_REGISTRY)
        assert isinstance(result, ParsedFinalAnswer)
        assert result.answer["action"] == "transfer"

    def test_valid_final_answer_with_rationale(self):
        raw = '{"thought": "done", "final_answer": {"action": "transfer", "rationale": "best option"}}'
        result = validate_tool_call(raw, TOOL_REGISTRY)
        assert isinstance(result, ParsedFinalAnswer)
        assert result.answer["action"] == "transfer"
        assert result.answer["rationale"] == "best option"

    def test_no_json_returns_error(self):
        result = validate_tool_call("just some text", TOOL_REGISTRY)
        assert isinstance(result, ValidationError)
        assert result.error_type == "no_json"

    def test_unknown_tool_returns_error(self):
        raw = '{"tool_call": {"name": "query_database", "arguments": {"sql": "SELECT *"}}}'
        result = validate_tool_call(raw, TOOL_REGISTRY)
        assert isinstance(result, ValidationError)
        assert result.error_type == "unknown_tool"

    def test_missing_arguments_returns_error(self):
        raw = '{"tool_call": {"name": "get_inventory", "arguments": {"sku": "SKU-4090"}}}'
        result = validate_tool_call(raw, TOOL_REGISTRY)
        assert isinstance(result, ValidationError)
        assert result.error_type == "missing_arguments"

    def test_extra_arguments_returns_error(self):
        raw = '{"tool_call": {"name": "get_order", "arguments": {"order_id": "SO-10482", "extra": true}}}'
        result = validate_tool_call(raw, TOOL_REGISTRY)
        assert isinstance(result, ValidationError)
        assert result.error_type == "extra_arguments"

    def test_missing_tool_call_key(self):
        raw = '{"name": "get_order", "arguments": {"order_id": "SO-10482"}}'
        result = validate_tool_call(raw, TOOL_REGISTRY)
        assert isinstance(result, ValidationError)

    def test_json_embedded_in_text(self):
        raw = 'Here is my call:\n{"tool_call": {"name": "get_order", "arguments": {"order_id": "SO-10482"}}}\nEnd.'
        result = validate_tool_call(raw, TOOL_REGISTRY)
        assert isinstance(result, ParsedToolCall)
