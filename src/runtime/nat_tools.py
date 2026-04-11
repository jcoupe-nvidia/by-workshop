"""
NAT-compatible tool definitions for the late-order recovery scenario.

Wraps the 9 deterministic tools from runtime.tools as NAT Function objects
with Pydantic input/output schemas. This enables NAT's tool discovery,
schema validation, and invocation infrastructure to work with our tools.

The underlying tool implementations remain unchanged — this module adds
the NAT Function interface on top of them.

Owns:
    - NAT Function wrappers via LambdaFunction
    - NAT-compatible tool registry and FunctionGroup dispatch

Does NOT own:
    - Tool implementations (see runtime.tools)
    - Pydantic input models or OpenAI-style tool definitions (see shared.tool_schemas)
    - Agent orchestration or prompt policy
    - Reward semantics or training concerns
"""
from typing import Any

from pydantic import BaseModel, Field

from nat.builder.function import LambdaFunction
from nat.builder.function_info import FunctionInfo
from nat.data_models.function import EmptyFunctionConfig

from src.runtime.tools import TOOL_REGISTRY
from src.shared.tool_schemas import (
    TOOL_INPUT_MODELS,
    build_openai_tool_definitions,
)


# ---------------------------------------------------------------------------
# Generic output model (tools return dicts)
# ---------------------------------------------------------------------------

class ToolOutput(BaseModel):
    result: dict[str, Any] = Field(description="Tool execution result.")


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
# NAT FunctionGroup (unified tool dispatch surface)
# ---------------------------------------------------------------------------

from nat.builder.function import FunctionGroup, FunctionGroupBaseConfig


def build_nat_function_group(
    instance_name: str = "supply_chain_tools",
) -> FunctionGroup:
    """Build a NAT FunctionGroup containing all 9 scenario tools.

    The FunctionGroup is the NAT-canonical way to register, discover,
    and invoke tools. Each tool is added with its Pydantic input schema
    so NAT can validate arguments before dispatch.

    The returned group can be used for:
        - Tool discovery via get_accessible_functions()
        - Invocation via group.get_all_functions()[name].ainvoke(input)
        - Schema export for prompt construction
    """
    config = FunctionGroupBaseConfig()
    group = FunctionGroup(config=config, instance_name=instance_name)

    for tool_name in TOOL_REGISTRY:
        fn, _params, description = TOOL_REGISTRY[tool_name]
        input_model = TOOL_INPUT_MODELS[tool_name]

        # NAT add_function unpacks Pydantic fields as kwargs to the function,
        # so the registered callable must accept **kwargs matching the schema.
        async def _invoke(_fn=fn, **kwargs) -> ToolOutput:
            result = _fn(**kwargs)
            return ToolOutput(result=result)

        group.add_function(
            name=tool_name,
            fn=_invoke,
            input_schema=input_model,
            description=description,
        )

    return group


async def invoke_tool_via_group(
    group: FunctionGroup,
    tool_name: str,
    arguments: dict,
) -> dict:
    """Invoke a single tool through a NAT FunctionGroup.

    Looks up the function by name, constructs the Pydantic input,
    and calls ainvoke(). Returns the result dict.

    Raises KeyError if tool_name is not in the group.
    """
    functions = await group.get_all_functions()
    # FunctionGroup prefixes names: {instance_name}__{tool_name}
    qualified = f"{group.instance_name}{FunctionGroup.SEPARATOR}{tool_name}"
    func = functions.get(qualified)
    if func is None:
        # Try without prefix for flexibility
        func = functions.get(tool_name)
    if func is None:
        raise KeyError(f"Tool '{tool_name}' not found in FunctionGroup")

    input_model = TOOL_INPUT_MODELS[tool_name]
    typed_input = input_model(**arguments)
    output = await func.ainvoke(typed_input)
    return output.result


# ---------------------------------------------------------------------------
# OpenAI-style tool definitions for ATIF agent.tool_definitions
# ---------------------------------------------------------------------------
# Re-exported from src.shared.tool_schemas for backward compatibility.
# build_openai_tool_definitions is imported at the top of this module.
