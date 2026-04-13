"""
Prompt and runtime policy for the supply-chain recovery agent.

Owns:
    - System prompt construction (tool descriptions, rules, format spec)
    - Task message construction
    - Prompt constants and formatting

Does NOT own:
    - Model adapter or HTTP calls (see runtime.agent)
    - Tool implementations (see runtime.tools)
    - Fallback parsing (see runtime.fallbacks)
    - Trace emission (see runtime.tracing)
"""
from __future__ import annotations

from src.runtime.tools import SIMULATION_DATE_STR, TOOL_REGISTRY, TOOL_DEPENDENCIES


def build_system_prompt() -> str:
    """Build the system prompt that tells the model about available tools.

    The prompt describes:
    - The task format
    - Available tools with their parameters and descriptions
    - The canonical JSON format for tool calls
    - When to emit a final answer
    """
    tool_descriptions = []
    for name, (_fn, params, desc) in sorted(TOOL_REGISTRY.items()):
        param_str = ", ".join(f"{k}: {v}" for k, v in params.items())
        deps = TOOL_DEPENDENCIES.get(name, set())
        dep_str = ", ".join(sorted(deps)) if deps else "none"
        tool_descriptions.append(
            f"  - {name}({param_str}): {desc} [requires: {dep_str}]"
        )

    tools_block = "\n".join(tool_descriptions)

    return f"""You are a supply-chain recovery agent. Your job is to diagnose order risks and recommend mitigation actions.

You have access to the following tools:
{tools_block}

IMPORTANT RULES:
1. You must respond with ONLY a JSON object on each turn -- no extra text before or after.
2. To call a tool, respond with this exact format:
{{"thought": "your brief reasoning", "tool_call": {{"name": "tool_name", "arguments": {{...}}}}}}
3. When you have gathered enough information and are ready to give a final recommendation, respond with:
{{"thought": "your reasoning", "final_answer": {{"action": "recommended action", "rationale": "why this is best", "expected_delivery": "YYYY-MM-DD", "meets_committed_date": true/false, "confidence": 0.0-1.0}}}}
4. Follow the correct tool call sequence. You must respect dependencies:
   - Start with get_order to look up the order
   - Then get_shipment_status to check shipping state
   - Then get_inventory to check stock at the source DC
   - Then get_fulfillment_capacity to check DC capacity
   - Then find_alternate_inventory to search other DCs
   - Then get_transfer_eta for transfer options and get_supplier_expedite_options for supplier rush options
   - Then score_recovery_options to rank the options
   - Then recommend_action to produce a final recommendation
5. The "thought" field is optional but encouraged for reasoning transparency.
6. Always include all required arguments for each tool.

Today's date is {SIMULATION_DATE_STR}."""


def build_task_message(order_id: str) -> str:
    """Build the user message that presents the task."""
    return (
        f"Customer order {order_id} is at risk of missing its committed delivery date. "
        f"Determine whether the order can still be fulfilled on time from its primary source. "
        f"If not, recommend the best mitigation action. "
        f"Start by looking up the order details."
    )
