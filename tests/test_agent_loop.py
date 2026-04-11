"""Offline smoke tests for the agent loop wiring.

Tests the full agent loop path (prompt construction, message formatting,
response parsing, tool dispatch) using mocked model responses. No live
model endpoint is needed.

This covers the same code path exercised by notebook cells 19-23.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from src.runtime.agent import (
    run_agent,
    run_agent_episode,
    call_model,
    print_trace_summary,
    AgentTrace,
    MODEL_ENDPOINT,
    MODEL_NAME,
)
from src.runtime.prompts import build_system_prompt, build_task_message
from src.rollouts.trace_types import Episode, EventType


# -- Helpers for mocking model responses -----------------------------------------

def _make_tool_call_response(tool_name: str, arguments: dict, thought: str = "") -> str:
    """Build a model response string containing a structured tool call."""
    return json.dumps({
        "thought": thought,
        "tool_call": {"name": tool_name, "arguments": arguments},
    })


def _make_final_answer_response(answer: dict) -> str:
    return json.dumps({
        "thought": "Done",
        "final_answer": answer,
    })


# A sequence of model responses that completes the SO-10482 scenario
SCRIPTED_MODEL_RESPONSES = [
    _make_tool_call_response("get_order", {"order_id": "SO-10482"}, "Look up order"),
    _make_tool_call_response("get_shipment_status", {"order_id": "SO-10482"}, "Check shipment"),
    _make_tool_call_response("get_inventory", {"sku": "SKU-4090", "dc_id": "DC-WEST-01"}, "Check DC"),
    _make_tool_call_response("get_fulfillment_capacity", {"dc_id": "DC-WEST-01", "date": "2026-04-18"}, "Check capacity"),
    _make_tool_call_response("find_alternate_inventory", {"sku": "SKU-4090", "region": "ALL"}, "Search alternates"),
    _make_tool_call_response("get_transfer_eta", {
        "from_dc": "DC-EAST-02", "to_dc": "DC-WEST-01", "sku": "SKU-4090", "qty": 900,
    }, "Get ETA"),
    _make_tool_call_response("get_supplier_expedite_options", {"sku": "SKU-4090", "qty": 900}, "Check supplier"),
    _make_tool_call_response("score_recovery_options", {
        "options": [
            {"source": "DC-EAST-02", "lead_days": 3, "cost_per_unit": 4.50, "feasible": True},
        ],
        "objective": "minimize_delay",
    }, "Score options"),
    _make_tool_call_response("recommend_action", {
        "context": {
            "best_option": {"source": "DC-EAST-02", "lead_days": 3},
            "order": {"order_id": "SO-10482"},
            "objective": "minimize_delay",
        },
    }, "Recommend"),
    _make_final_answer_response({
        "action": "dc_transfer from DC-EAST-02",
        "rationale": "Fastest option",
        "expected_delivery": "2026-04-16",
        "meets_committed_date": True,
        "confidence": 0.9,
    }),
]


def _mock_call_model_factory():
    """Create a side_effect function that returns scripted responses in order."""
    responses = list(SCRIPTED_MODEL_RESPONSES)
    call_count = [0]

    def mock_call(messages, max_tokens=1024, temperature=0.1):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(responses):
            return responses[idx]
        return _make_final_answer_response({"action": "fallback", "rationale": "max calls"})

    return mock_call


# -- Tests -----------------------------------------------------------------------

class TestPromptConstruction:
    def test_system_prompt_nonempty(self):
        prompt = build_system_prompt()
        assert len(prompt) > 100
        assert "tool_call" in prompt  # mentions structured tool call format

    def test_task_message(self):
        msg = build_task_message("SO-10482")
        assert "SO-10482" in msg


class TestAgentLoopWithMockedModel:
    @patch("src.runtime.agent.call_model")
    def test_run_agent_completes(self, mock_call):
        mock_call.side_effect = _mock_call_model_factory()
        trace = run_agent("SO-10482", verbose=False)
        assert isinstance(trace, AgentTrace)
        assert trace.completed is True
        assert trace.stop_reason == "final_answer"
        assert trace.final_answer is not None
        assert len(trace.steps) >= 5

    @patch("src.runtime.agent.call_model")
    def test_run_agent_tool_sequence(self, mock_call):
        mock_call.side_effect = _mock_call_model_factory()
        trace = run_agent("SO-10482", verbose=False)
        names = trace.tool_names_called
        assert names[0] == "get_order"
        assert "recommend_action" in names

    @patch("src.runtime.agent.call_model")
    def test_run_agent_episode_returns_episode(self, mock_call):
        mock_call.side_effect = _mock_call_model_factory()
        episode = run_agent_episode("SO-10482", verbose=False)
        assert isinstance(episode, Episode)
        assert episode.task_id == "SO-10482"
        assert episode.is_complete is True
        assert episode.metrics.valid_tool_calls >= 5

    @patch("src.runtime.agent.call_model")
    def test_run_agent_episode_has_events(self, mock_call):
        mock_call.side_effect = _mock_call_model_factory()
        episode = run_agent_episode("SO-10482", verbose=False)
        tool_calls = [e for e in episode.events if e.event_type == EventType.TOOL_CALL]
        tool_results = [e for e in episode.events if e.event_type == EventType.TOOL_RESULT]
        terminal = [e for e in episode.events if e.event_type == EventType.TERMINAL_OUTCOME]
        assert len(tool_calls) >= 5
        assert len(tool_results) >= 5
        assert len(terminal) == 1


class TestAgentLoopFallbackHandling:
    @patch("src.runtime.agent.call_model")
    def test_handles_malformed_output(self, mock_call):
        """Model emits garbage first, then valid calls."""
        responses = [
            "I think we should check the order first.",  # no JSON -> reject
            _make_tool_call_response("get_order", {"order_id": "SO-10482"}),
            _make_final_answer_response({"action": "stop", "rationale": "test"}),
        ]
        mock_call.side_effect = responses
        trace = run_agent("SO-10482", verbose=False)
        assert trace.fallback_rejects >= 1 or trace.errors  # at least one rejection

    @patch("src.runtime.agent.call_model")
    def test_handles_repairable_output(self, mock_call):
        """Model emits trailing commas, then valid calls."""
        responses = [
            '{"tool_call": {"name": "get_order", "arguments": {"order_id": "SO-10482",}}}',
            _make_final_answer_response({"action": "stop", "rationale": "test"}),
        ]
        mock_call.side_effect = responses
        trace = run_agent("SO-10482", verbose=False)
        assert trace.fallback_repairs >= 1


class TestPrintTraceSummary:
    @patch("src.runtime.agent.call_model")
    def test_print_does_not_crash(self, mock_call, capsys):
        mock_call.side_effect = _mock_call_model_factory()
        trace = run_agent("SO-10482", verbose=False)
        print_trace_summary(trace)
        captured = capsys.readouterr()
        assert "SO-10482" in captured.out
