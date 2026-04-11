"""
Repo-owned tool schemas and OpenAI-style tool definitions.

Provides Pydantic input models for each business tool and a function to
build OpenAI function-calling-style tool definitions. These are consumed
by both the NAT runtime layer (for FunctionGroup registration) and the
training layer (for art.Trajectory tool definitions) without requiring
either consumer to import the other's framework-specific code.

Owns:
    - Pydantic input models per tool
    - Input model registry (tool_name -> model class)
    - OpenAI-style tool definition builder
    - Default system prompt text for training export

Does NOT own:
    - Tool implementations (see runtime.tools)
    - NAT Function wrappers (see runtime.nat_tools)
    - Training trajectory construction (see training.openpipe_art_adapter)
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.runtime.tools import TOOL_REGISTRY, TOOL_DEPENDENCIES


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
# OpenAI-style tool definitions
# ---------------------------------------------------------------------------

def build_openai_tool_definitions() -> list[dict[str, Any]]:
    """Build OpenAI function-calling-style tool definitions from Pydantic schemas.

    These are used in ATIF Trajectory.agent.tool_definitions, in
    art.Trajectory.tools, and in the system prompt for structured tool calling.
    """
    definitions = []
    for tool_name, input_model in TOOL_INPUT_MODELS.items():
        _fn, _params, description = TOOL_REGISTRY[tool_name]
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
6. Always include all required arguments for each tool."""
