"""
NAT-compatible tool definitions for the late-order recovery scenario.

Wraps the 9 deterministic tools from runtime.tools as NAT Function objects
with Pydantic input/output schemas. This enables NAT's tool discovery,
schema validation, and invocation infrastructure to work with our tools.

The underlying tool implementations remain unchanged — this module adds
the NAT Function interface on top of them.

Owns:
    - Pydantic input/output models per tool
    - NAT Function wrappers via LambdaFunction
    - NAT-compatible tool registry and OpenAI-style tool definitions

Does NOT own:
    - Tool implementations (see runtime.tools)
    - Agent orchestration or prompt policy
    - Reward semantics or training concerns
"""
from typing import Any

from pydantic import BaseModel, Field

from nat.builder.function import LambdaFunction
from nat.builder.function_info import FunctionInfo
from nat.data_models.function import EmptyFunctionConfig

from src.runtime.tools import TOOL_REGISTRY


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
# Generic output model (tools return dicts)
# ---------------------------------------------------------------------------

class ToolOutput(BaseModel):
    result: dict[str, Any] = Field(description="Tool execution result.")


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
# Build NAT Function wrappers
# ---------------------------------------------------------------------------

def _build_nat_function(tool_name: str) -> LambdaFunction:
    """Wrap a registered tool as a NAT LambdaFunction."""
    fn, _params, description = TOOL_REGISTRY[tool_name]
    input_model = TOOL_INPUT_MODELS[tool_name]

    async def _invoke(value: BaseModel) -> ToolOutput:
        kwargs = value.model_dump()
        result = fn(**kwargs)
        return ToolOutput(result=result)

    info = FunctionInfo(
        single_fn=_invoke,
        stream_fn=None,
        input_schema=input_model,
        single_output_schema=ToolOutput,
        stream_output_schema=type(None),
        description=description,
    )
    config = EmptyFunctionConfig(name=tool_name)
    return LambdaFunction.from_info(
        config=config,
        info=info,
        instance_name=tool_name,
    )


def build_nat_tool_registry() -> dict[str, LambdaFunction]:
    """Build a dict of tool_name -> NAT LambdaFunction for all registered tools."""
    return {name: _build_nat_function(name) for name in TOOL_REGISTRY}


# ---------------------------------------------------------------------------
# OpenAI-style tool definitions for ATIF agent.tool_definitions
# ---------------------------------------------------------------------------

def build_openai_tool_definitions() -> list[dict[str, Any]]:
    """Build OpenAI function-calling-style tool definitions from Pydantic schemas.

    These are used in ATIF Trajectory.agent.tool_definitions and in
    the system prompt for structured tool calling.
    """
    definitions = []
    for tool_name, input_model in TOOL_INPUT_MODELS.items():
        _fn, _params, description = TOOL_REGISTRY[tool_name]
        schema = input_model.model_json_schema()
        # Remove pydantic metadata keys that aren't part of OpenAI spec
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
