"""
Repo-owned tool schemas and OpenAI-style tool definitions.

Provides Pydantic input models for each business tool and a function to
build OpenAI function-calling-style tool definitions. These are consumed
by both the NAT runtime layer (for FunctionGroup registration) and the
training layer (for DatumSpec tool definitions) without requiring
either consumer to import the other's framework-specific code.

Owns:
    - Pydantic input models per tool
    - Input model registry (tool_name -> model class)
    - Tool metadata (descriptions, parameter specs) — self-contained
    - OpenAI-style tool definition builder
    - Default system prompt text for training export

Does NOT own:
    - Tool implementations (see runtime.tools)
    - NAT Function wrappers (see runtime.nat_tools)
    - Training DatumSpec construction (see training.nemo_rl_adapter)
    - Tool dependency graph (see envs.state)
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.envs.state import TOOL_DEPENDENCIES


# ---------------------------------------------------------------------------
# Pydantic input models (one per tool)
# ---------------------------------------------------------------------------

class GetOrderInput(BaseModel):
    order_id: str = Field(description="The sales order ID to look up.")

class GetShipmentStatusInput(BaseModel):
    order_id: str = Field(description="The order ID to check shipment status for.")

class GetInventoryInput(BaseModel):
    sku: str = Field(description="The SKU to check inventory for.")
    dc_id: str = Field(description="The distribution center ID.")

class FindAlternateInventoryInput(BaseModel):
    sku: str = Field(description="The SKU to search for.")
    region: str = Field(description="Region to search. Use 'ALL' for all regions.")

class GetTransferEtaInput(BaseModel):
    from_dc: str = Field(description="Source distribution center ID.")
    to_dc: str = Field(description="Destination distribution center ID.")
    sku: str = Field(description="The SKU to transfer.")
    qty: int = Field(description="Quantity to transfer.")

class GetSupplierExpediteOptionsInput(BaseModel):
    sku: str = Field(description="The SKU to get expedite options for.")
    qty: int = Field(description="Quantity needed.")

class GetFulfillmentCapacityInput(BaseModel):
    dc_id: str = Field(description="Distribution center ID.")
    date: str = Field(description="Date to check capacity for (YYYY-MM-DD).")

class ScoreRecoveryOptionsInput(BaseModel):
    options: list[dict[str, Any]] = Field(description="List of recovery option dicts to score.")
    objective: str = Field(description="Scoring objective: 'minimize_delay', 'minimize_cost', or 'balanced'.")

class RecommendActionInput(BaseModel):
    context: dict[str, Any] = Field(description="Dict with 'best_option', 'order', and 'objective' keys.")


# ---------------------------------------------------------------------------
# Input model registry (tool_name -> input model class)
# ---------------------------------------------------------------------------

TOOL_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "get_order": GetOrderInput,
    "get_shipment_status": GetShipmentStatusInput,
    "get_inventory": GetInventoryInput,
    "find_alternate_inventory": FindAlternateInventoryInput,
    "get_transfer_eta": GetTransferEtaInput,
    "get_supplier_expedite_options": GetSupplierExpediteOptionsInput,
    "get_fulfillment_capacity": GetFulfillmentCapacityInput,
    "score_recovery_options": ScoreRecoveryOptionsInput,
    "recommend_action": RecommendActionInput,
}


# ---------------------------------------------------------------------------
# Canonical Nemotron-style envelope schemas
# ---------------------------------------------------------------------------

class ToolCallInner(BaseModel):
    """Inner structure of a tool call: the tool name and its arguments."""
    name: str = Field(description="Name of the tool to invoke.")
    arguments: dict[str, Any] = Field(description="Tool arguments as key-value pairs.")


class NemotronToolCallEnvelope(BaseModel):
    """Canonical envelope for a Nemotron-style tool call.

    Every model response that invokes a tool must conform to this schema.
    The ``thought`` field is optional but encouraged for reasoning transparency.
    """
    thought: str | None = Field(default=None, description="Optional short reasoning summary.")
    tool_call: ToolCallInner


class FinalAnswerPayload(BaseModel):
    """Payload for a final recommendation answer."""
    action: str = Field(description="Recommended mitigation action.")
    rationale: str | None = Field(default=None, description="Why this action is best.")
    expected_delivery: str | None = Field(default=None, description="Expected delivery date (YYYY-MM-DD).")
    meets_committed_date: bool | None = Field(default=None, description="Whether the committed date is met.")
    confidence: float | None = Field(default=None, description="Confidence score 0.0–1.0.")


class NemotronFinalAnswerEnvelope(BaseModel):
    """Canonical envelope for a Nemotron-style final answer.

    Terminates the agent loop with a structured recommendation.
    """
    thought: str | None = Field(default=None, description="Optional reasoning summary.")
    final_answer: FinalAnswerPayload


# ---------------------------------------------------------------------------
# Self-contained tool metadata (descriptions + parameter specs)
# ---------------------------------------------------------------------------

TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_order": "Look up order details by order ID.",
    "get_shipment_status": "Get the current shipment status for an order.",
    "get_inventory": "Check on-hand, reserved, and available inventory for a SKU at a specific DC.",
    "find_alternate_inventory": "Search for available inventory of a SKU across DCs in a region. Use region='ALL' to search everywhere.",
    "get_transfer_eta": "Estimate transfer time and cost to move units between DCs.",
    "get_supplier_expedite_options": "Get available supplier expedite (rush) options for a SKU and quantity.",
    "get_fulfillment_capacity": "Check available fulfillment capacity at a DC on a specific date.",
    "score_recovery_options": "Score and rank a list of recovery options against an objective (e.g. 'minimize_delay', 'minimize_cost').",
    "recommend_action": "Produce a final recommendation based on scored recovery options and order context.",
}

TOOL_PARAMS: dict[str, dict[str, str]] = {
    "get_order": {"order_id": "str"},
    "get_shipment_status": {"order_id": "str"},
    "get_inventory": {"sku": "str", "dc_id": "str"},
    "find_alternate_inventory": {"sku": "str", "region": "str"},
    "get_transfer_eta": {"from_dc": "str", "to_dc": "str", "sku": "str", "qty": "int"},
    "get_supplier_expedite_options": {"sku": "str", "qty": "int"},
    "get_fulfillment_capacity": {"dc_id": "str", "date": "str"},
    "score_recovery_options": {"options": "list[dict]", "objective": "str"},
    "recommend_action": {"context": "dict"},
}


# ---------------------------------------------------------------------------
# OpenAI-style tool definitions
# ---------------------------------------------------------------------------

def build_openai_tool_definitions() -> list[dict[str, Any]]:
    """Build OpenAI function-calling-style tool definitions from Pydantic schemas.

    These are used in ATIF Trajectory.agent.tool_definitions, in
    NeMo RL DatumSpec tool definitions, and in the system prompt for structured tool calling.
    """
    definitions = []
    for tool_name, input_model in TOOL_INPUT_MODELS.items():
        description = TOOL_DESCRIPTIONS[tool_name]
        schema = input_model.model_json_schema()
        schema.pop("title", None)

        definitions.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": schema,
            },
        })
    return definitions


# ---------------------------------------------------------------------------
# Default system prompt text
# ---------------------------------------------------------------------------

def build_default_system_prompt() -> str:
    """Build the default system prompt for training export.

    This is the repo-owned canonical prompt text. The runtime layer's
    build_system_prompt() in runtime.prompts may add runtime-specific
    decorations, but this version is sufficient for training trajectory
    export where the prompt is used as context, not as live runtime policy.
    """
    tool_descriptions = []
    for name in sorted(TOOL_DESCRIPTIONS):
        params = TOOL_PARAMS[name]
        desc = TOOL_DESCRIPTIONS[name]
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
6. Always include all required arguments for each tool."""
