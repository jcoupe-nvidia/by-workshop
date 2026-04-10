"""
OpenCode-inspired agent execution loop and model adapter.

This module intentionally implements a small local version of the
architecture so the think -> emit tool call -> validate -> execute ->
observe cycle stays visible and easy to understand in the workshop.

Loop steps:
    1. Think   -- model receives context and emits a response
    2. Parse   -- extract structured tool call (or detect final answer)
    3. Validate -- check tool call against schema and dependency rules
    4. Execute -- dispatch to deterministic tool
    5. Observe -- append tool result to conversation history
    6. Continue -- loop back to step 1 (bounded by max iterations)

The model adapter is the single integration seam for the local LLM.
Everything else in this module is deterministic and testable.

Model adapter contract:
    - POST to http://0.0.0.0:8000/v1/chat/completions
    - model: "nvidia/nemotron-3-nano"
    - OpenAI-style messages list
    - Returns assistant message content as a string
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from src.schema import (
    ParsedToolCall,
    ParsedFinalAnswer,
    ValidationError,
    validate_tool_call,
    check_dependencies,
)
from src.tools import TOOL_REGISTRY, TOOL_DEPENDENCIES
from src.fallbacks import (
    FallbackAction,
    FallbackResult,
    try_repair,
    parse_with_fallback,
)

# -- Configuration ---------------------------------------------------------

MODEL_ENDPOINT = "http://0.0.0.0:8000/v1/chat/completions"
MODEL_NAME = "nvidia/nemotron-3-nano"
MAX_ITERATIONS = 15  # hard stop to prevent runaway loops
DEFAULT_MAX_TOKENS = 1024
REQUEST_TIMEOUT = 60  # seconds

# -- Trace data structures -------------------------------------------------

@dataclass
class ToolCallRecord:
    """One step in the agent's trajectory."""
    iteration: int
    thought: str | None
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    valid: bool
    validation_error: str | None = None
    fallback_action: str | None = None       # "repaired", "rejected", or None
    repairs_applied: list[str] | None = None  # list of repair labels if repaired

@dataclass
class AgentTrace:
    """Full trajectory of a single agent run."""
    task: str
    steps: list[ToolCallRecord] = field(default_factory=list)
    final_answer: dict[str, Any] | None = None
    completed: bool = False
    stop_reason: str | None = None  # "final_answer", "max_iterations", "error"
    messages: list[dict[str, str]] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    model_calls: int = 0
    errors: list[str] = field(default_factory=list)
    fallback_repairs: int = 0
    fallback_rejects: int = 0

    @property
    def tool_names_called(self) -> list[str]:
        """List of tool names called in order."""
        return [s.tool_name for s in self.steps if s.valid]

    @property
    def total_tool_calls(self) -> int:
        return len([s for s in self.steps if s.valid])


# -- Model adapter ---------------------------------------------------------

def call_model(
    messages: list[dict[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.1,
) -> str:
    """Send messages to the local Nemotron endpoint and return the response.

    This is the single integration point for the real model.
    Uses the OpenAI-compatible chat completions API.

    Raises:
        requests.RequestException: If the HTTP request fails.
        ValueError: If the response is missing expected fields.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(
        MODEL_ENDPOINT,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"No choices in model response: {data}")

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError(f"Empty content in model response: {choices[0]}")

    return content


# -- System prompt builder -------------------------------------------------

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

Today's date is 2026-04-10."""


def build_task_message(order_id: str) -> str:
    """Build the user message that presents the task."""
    return (
        f"Customer order {order_id} is at risk of missing its committed delivery date. "
        f"Determine whether the order can still be fulfilled on time from its primary source. "
        f"If not, recommend the best mitigation action. "
        f"Start by looking up the order details."
    )


# -- Agent execution loop --------------------------------------------------

def run_agent(
    order_id: str,
    max_iterations: int = MAX_ITERATIONS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.1,
    verbose: bool = True,
) -> AgentTrace:
    """Execute the full agent loop for an order.

    Steps per iteration:
        1. Call the model with conversation history
        2. Parse the response for a tool call or final answer
        3. Validate the tool call (schema + dependencies)
        4. Execute the tool
        5. Append the observation to conversation history
        6. Continue or stop

    Args:
        order_id: The order to investigate (e.g. "SO-10482").
        max_iterations: Safety bound on loop iterations.
        max_tokens: Max tokens per model response.
        temperature: Sampling temperature.
        verbose: Print each step as it happens.

    Returns:
        AgentTrace with the full trajectory.
    """
    trace = AgentTrace(task=f"Investigate order {order_id}")
    start_time = time.time()

    # Build initial messages
    system_msg = {"role": "system", "content": build_system_prompt()}
    user_msg = {"role": "user", "content": build_task_message(order_id)}
    messages: list[dict[str, str]] = [system_msg, user_msg]

    if verbose:
        print(f"=== Agent loop started for {order_id} ===")
        print(f"    Max iterations: {max_iterations}")
        print()

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"--- Iteration {iteration} ---")

        # Step 1: Call the model
        try:
            raw_response = call_model(
                messages, max_tokens=max_tokens, temperature=temperature,
            )
            trace.model_calls += 1
        except Exception as e:
            error_msg = f"Model call failed: {e}"
            trace.errors.append(error_msg)
            trace.stop_reason = "error"
            if verbose:
                print(f"  ERROR: {error_msg}")
            break

        if verbose:
            # Show a truncated version of the raw response
            display = raw_response[:200] + ("..." if len(raw_response) > 200 else "")
            print(f"  Model: {display}")

        # Step 2: Parse and validate the response
        result = validate_tool_call(raw_response, TOOL_REGISTRY)
        fallback_result: FallbackResult | None = None

        # Step 2b: If validation fails, attempt fallback repair
        if isinstance(result, ValidationError):
            fb = try_repair(raw_response, TOOL_REGISTRY)

            if fb.action == FallbackAction.REPAIRED and fb.repaired:
                # Re-validate the repaired output
                result = validate_tool_call(fb.repaired, TOOL_REGISTRY)
                fallback_result = fb
                trace.fallback_repairs += 1
                if verbose:
                    print(f"  -> Fallback REPAIRED: {fb.repairs_applied}")
            elif fb.action == FallbackAction.REJECTED:
                fallback_result = fb
                trace.fallback_rejects += 1
                if verbose:
                    print(f"  -> Fallback REJECTED: {fb.rejection_reason}")
            # If NO_ACTION, the original validation error stands

        # Step 3a: Final answer -- stop the loop
        if isinstance(result, ParsedFinalAnswer):
            trace.final_answer = result.answer
            trace.completed = True
            trace.stop_reason = "final_answer"
            messages.append({"role": "assistant", "content": raw_response})
            if verbose:
                print(f"  -> Final answer: {json.dumps(result.answer, indent=2)}")
            break

        # Step 3b: Validation error (after fallback attempt) -- tell the model and continue
        if isinstance(result, ValidationError):
            error_msg = f"Validation error ({result.error_type}): {result.message}"
            trace.errors.append(error_msg)

            fb_action = fallback_result.action.value if fallback_result else None
            fb_repairs = fallback_result.repairs_applied if fallback_result else None

            # Record the failed step
            trace.steps.append(ToolCallRecord(
                iteration=iteration,
                thought=None,
                tool_name="<invalid>",
                arguments={},
                result={"error": result.message},
                valid=False,
                validation_error=result.message,
                fallback_action=fb_action,
                repairs_applied=fb_repairs,
            ))

            # Feed the error back to the model so it can correct itself
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
                "role": "user",
                "content": (
                    f"Error: {result.message} "
                    f"Please respond with a valid JSON tool call or final answer."
                ),
            })

            if verbose:
                print(f"  -> Validation error: {result.error_type}: {result.message}")
            continue

        # Step 3c: Valid tool call -- check dependencies
        assert isinstance(result, ParsedToolCall)

        dep_ok, dep_reason = check_dependencies(
            result.tool_name,
            trace.tool_names_called,
            TOOL_DEPENDENCIES,
        )

        if not dep_ok:
            trace.errors.append(dep_reason)
            trace.steps.append(ToolCallRecord(
                iteration=iteration,
                thought=result.thought,
                tool_name=result.tool_name,
                arguments=result.arguments,
                result={"error": dep_reason},
                valid=False,
                validation_error=dep_reason,
            ))

            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
                "role": "user",
                "content": (
                    f"Error: {dep_reason} "
                    f"Please call the prerequisite tools first."
                ),
            })

            if verbose:
                print(f"  -> Dependency error: {dep_reason}")
            continue

        # Step 4: Execute the tool
        if verbose:
            print(f"  -> Calling: {result.tool_name}({result.arguments})")

        try:
            fn, _params, _desc = TOOL_REGISTRY[result.tool_name]
            tool_result = fn(**result.arguments)
        except Exception as e:
            tool_result = {"error": f"Tool execution failed: {e}"}
            trace.errors.append(str(e))

        # Record the successful step (include fallback info if repaired)
        fb_action = fallback_result.action.value if fallback_result else None
        fb_repairs = fallback_result.repairs_applied if fallback_result else None
        trace.steps.append(ToolCallRecord(
            iteration=iteration,
            thought=result.thought,
            tool_name=result.tool_name,
            arguments=result.arguments,
            result=tool_result,
            valid=True,
            fallback_action=fb_action,
            repairs_applied=fb_repairs,
        ))

        if verbose:
            result_str = json.dumps(tool_result, indent=2)
            if len(result_str) > 300:
                result_str = result_str[:300] + "..."
            print(f"  -> Result: {result_str}")

        # Step 5: Append to conversation history
        messages.append({"role": "assistant", "content": raw_response})
        messages.append({
            "role": "user",
            "content": f"Tool result for {result.tool_name}:\n{json.dumps(tool_result)}",
        })

    else:
        # Reached max iterations without a final answer
        trace.stop_reason = "max_iterations"
        if verbose:
            print(f"\n  Stopped: reached max iterations ({max_iterations}).")

    trace.wall_time_seconds = round(time.time() - start_time, 2)
    trace.messages = messages

    if verbose:
        print(f"\n=== Agent loop finished ===")
        print(f"    Stop reason:     {trace.stop_reason}")
        print(f"    Tool calls:      {trace.total_tool_calls}")
        print(f"    Model calls:     {trace.model_calls}")
        print(f"    Errors:          {len(trace.errors)}")
        print(f"    Fallback repairs:{trace.fallback_repairs}")
        print(f"    Fallback rejects:{trace.fallback_rejects}")
        print(f"    Wall time:       {trace.wall_time_seconds}s")
        if trace.final_answer:
            print(f"    Final answer:    {trace.final_answer.get('action', 'N/A')}")

    return trace


# -- Trace inspection helpers ----------------------------------------------

def print_trace_summary(trace: AgentTrace) -> None:
    """Print a compact summary of an agent trace."""
    print(f"Task: {trace.task}")
    print(f"Status: {'completed' if trace.completed else 'incomplete'} ({trace.stop_reason})")
    print(f"Tool calls: {trace.total_tool_calls} valid, {len(trace.errors)} errors")
    print(f"Fallbacks: {trace.fallback_repairs} repairs, {trace.fallback_rejects} rejects")
    print(f"Wall time: {trace.wall_time_seconds}s ({trace.model_calls} model calls)")
    print()

    print("Tool call sequence:")
    for i, step in enumerate(trace.steps, 1):
        status = "OK" if step.valid else "FAIL"
        thought = f' "{step.thought}"' if step.thought else ""
        fb_tag = f" [repaired: {step.repairs_applied}]" if step.repairs_applied else ""
        print(f"  {i:2d}. [{status}] {step.tool_name}({step.arguments}){thought}{fb_tag}")
        if not step.valid:
            print(f"      Error: {step.validation_error}")

    if trace.final_answer:
        print(f"\nFinal answer:")
        for k, v in trace.final_answer.items():
            print(f"  {k}: {v}")


def trace_to_trajectory(trace: AgentTrace) -> list[dict[str, Any]]:
    """Export a trace as a list of trajectory steps for NeMo RL export.

    Each step includes the tool call, result, and metadata needed for
    trajectory-based training.
    """
    trajectory = []
    for step in trace.steps:
        trajectory.append({
            "iteration": step.iteration,
            "tool_name": step.tool_name,
            "arguments": step.arguments,
            "result": step.result,
            "valid": step.valid,
            "thought": step.thought,
            "validation_error": step.validation_error,
            "fallback_action": step.fallback_action,
            "repairs_applied": step.repairs_applied,
        })
    if trace.final_answer:
        trajectory.append({
            "iteration": len(trace.steps) + 1,
            "tool_name": "<final_answer>",
            "arguments": {},
            "result": trace.final_answer,
            "valid": True,
            "thought": None,
            "validation_error": None,
            "fallback_action": None,
            "repairs_applied": None,
        })
    return trajectory
