"""
Single-episode agent execution loop and model adapter.

Owns:
    - Model adapter (HTTP calls to local Nemotron endpoint)
    - Single-episode agent loop (think -> parse -> validate -> execute -> observe)
    - Structured event emission via EpisodeRecorder
    - Backward-compatible AgentTrace and helper types

Does NOT own:
    - Tool implementations (see runtime.tools)
    - Prompt policy (see runtime.prompts)
    - Fallback parsing (see runtime.fallbacks)
    - Schema validation (see runtime.schemas)
    - Dependency checking (see envs.validators)
    - Rollout batching or serialization (see rollouts/)
    - Reward computation (see envs.rewards)

This module intentionally implements a small local version of the
OpenCode-inspired architecture so the think -> emit tool call -> validate ->
execute -> observe cycle stays visible and easy to understand in the workshop.

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

from src.runtime.schemas import (
    ParsedToolCall,
    ParsedFinalAnswer,
    ValidationError,
    validate_tool_call,
)
from src.runtime.tools import TOOL_REGISTRY, TOOL_DEPENDENCIES
from src.runtime.fallbacks import (
    FallbackAction,
    FallbackResult,
    try_repair,
    parse_with_fallback,
)
from src.runtime.prompts import build_system_prompt, build_task_message
from src.runtime.tracing import EpisodeRecorder
from src.envs.validators import check_dependencies
from src.rollouts.trace_types import Episode

# -- Configuration ---------------------------------------------------------

MODEL_ENDPOINT = "http://0.0.0.0:8000/v1/chat/completions"
MODEL_NAME = "nvidia/nemotron-3-nano"
MAX_ITERATIONS = 15  # hard stop to prevent runaway loops
DEFAULT_MAX_TOKENS = 1024
REQUEST_TIMEOUT = 60  # seconds

# -- Backward-compatible trace data structures -----------------------------

@dataclass
class ToolCallRecord:
    """One step in the agent's trajectory (backward-compatible)."""
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
    """Full trajectory of a single agent run (backward-compatible)."""
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

    This method returns the backward-compatible AgentTrace.
    For the canonical Episode format, use run_agent_episode().

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


# -- Structured episode runner ---------------------------------------------

def run_agent_episode(
    order_id: str,
    max_iterations: int = MAX_ITERATIONS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.1,
    verbose: bool = True,
) -> Episode:
    """Execute the full agent loop and return a canonical Episode.

    This is the structured-event version of run_agent(). It produces
    a fully populated Episode using the canonical trace_types, with
    every runtime action recorded as a typed event.

    Args:
        order_id: The order to investigate (e.g. "SO-10482").
        max_iterations: Safety bound on loop iterations.
        max_tokens: Max tokens per model response.
        temperature: Sampling temperature.
        verbose: Print each step as it happens.

    Returns:
        Episode with canonical structured events.
    """
    task_prompt = build_task_message(order_id)
    recorder = EpisodeRecorder(
        task_id=order_id,
        task_prompt=task_prompt,
        model_id=MODEL_NAME,
    )

    # Build initial messages
    system_msg = {"role": "system", "content": build_system_prompt()}
    user_msg = {"role": "user", "content": task_prompt}
    messages: list[dict[str, str]] = [system_msg, user_msg]

    recorder.record_user_task(task_prompt)

    # Track tool names called for dependency checking
    tools_called: list[str] = []

    if verbose:
        print(f"=== Agent episode started for {order_id} ===")
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
            recorder.increment_model_calls()
        except Exception as e:
            error_msg = f"Model call failed: {e}"
            recorder.record_terminal("error", error_message=error_msg)
            if verbose:
                print(f"  ERROR: {error_msg}")
            break

        if verbose:
            display = raw_response[:200] + ("..." if len(raw_response) > 200 else "")
            print(f"  Model: {display}")

        # Step 2: Parse and validate the response
        result = validate_tool_call(raw_response, TOOL_REGISTRY)
        fallback_result: FallbackResult | None = None

        # Step 2b: If validation fails, attempt fallback repair
        if isinstance(result, ValidationError):
            fb = try_repair(raw_response, TOOL_REGISTRY)

            if fb.action == FallbackAction.REPAIRED and fb.repaired:
                recorder.record_repair_attempt(
                    original_output=raw_response,
                    repaired_output=fb.repaired,
                    repairs_applied=fb.repairs_applied,
                    succeeded=True,
                )
                result = validate_tool_call(fb.repaired, TOOL_REGISTRY)
                fallback_result = fb
                if verbose:
                    print(f"  -> Fallback REPAIRED: {fb.repairs_applied}")
            elif fb.action == FallbackAction.REJECTED:
                recorder.record_repair_attempt(
                    original_output=raw_response,
                    repaired_output=None,
                    repairs_applied=fb.repairs_applied,
                    succeeded=False,
                )
                recorder.record_reject(
                    reason=fb.rejection_reason or "Unknown rejection",
                    raw_model_output=raw_response,
                    repairs_attempted=fb.repairs_applied,
                )
                fallback_result = fb
                if verbose:
                    print(f"  -> Fallback REJECTED: {fb.rejection_reason}")

        # Step 3a: Final answer -- stop the loop
        if isinstance(result, ParsedFinalAnswer):
            if result.thought:
                recorder.record_model_thought(result.thought)
            recorder.record_terminal("final_answer", final_answer=result.answer)
            messages.append({"role": "assistant", "content": raw_response})
            if verbose:
                print(f"  -> Final answer: {json.dumps(result.answer, indent=2)}")
            break

        # Step 3b: Validation error (after fallback attempt)
        if isinstance(result, ValidationError):
            recorder.record_validation_error(
                error_type=result.error_type,
                message=result.message,
                raw_model_output=raw_response,
            )

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
            tools_called,
            TOOL_DEPENDENCIES,
        )

        if not dep_ok:
            recorder.record_validation_error(
                error_type="dependency_violation",
                message=dep_reason,
                raw_model_output=raw_response,
            )

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

        if result.thought:
            recorder.record_model_thought(result.thought)

        recorder.record_tool_call(
            tool_name=result.tool_name,
            arguments=result.arguments,
            thought=result.thought,
            raw_model_output=raw_response,
        )

        try:
            fn, _params, _desc = TOOL_REGISTRY[result.tool_name]
            tool_result = fn(**result.arguments)
        except Exception as e:
            tool_result = {"error": f"Tool execution failed: {e}"}

        recorder.record_tool_result(result.tool_name, tool_result)
        tools_called.append(result.tool_name)

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
        recorder.record_terminal("max_iterations")
        if verbose:
            print(f"\n  Stopped: reached max iterations ({max_iterations}).")

    episode = recorder.build_episode()

    if verbose:
        print(f"\n=== Agent episode finished ===")
        print(f"    Stop reason:     {episode.terminal.reason if episode.terminal else 'unknown'}")
        print(f"    Tool calls:      {episode.metrics.valid_tool_calls}")
        print(f"    Model calls:     {episode.metrics.model_calls}")
        print(f"    Errors:          {episode.metrics.invalid_tool_calls}")
        print(f"    Repair attempts: {episode.metrics.repair_attempts}")
        print(f"    Rejects:         {episode.metrics.rejects}")
        print(f"    Wall time:       {episode.metrics.wall_time_seconds}s")
        if episode.final_answer:
            print(f"    Final answer:    {episode.final_answer.get('action', 'N/A')}")

    return episode


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
