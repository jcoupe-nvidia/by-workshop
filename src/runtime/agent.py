"""
Single-episode agent execution loop and model adapter.

Owns:
    - Model adapter (HTTP calls to local Nemotron endpoint)
    - Single-episode agent loop (think -> parse -> validate -> execute -> observe)
    - NAT-backed agent entrypoint (run_agent_episode_nat) using FunctionGroup dispatch
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

The module provides two execution paths:
    - run_agent_episode()     -- direct TOOL_REGISTRY dispatch + raw HTTP model calls
    - run_agent_episode_nat() -- NAT FunctionGroup dispatch + NIMModelConfig model calls

Both paths share the same parse -> validate -> fallback -> execute -> observe
cycle via _episode_loop_core() and produce identical canonical Episode traces.

A backward-compatible run_agent() delegates to run_agent_episode() and
converts the Episode to an AgentTrace via _episode_to_trace().

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
from typing import Any, Callable

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
from src.runtime.execution import validate_and_repair
from src.runtime.prompts import build_system_prompt, build_task_message
from src.runtime.tracing import EpisodeRecorder
from src.envs.validators import check_dependencies
from src.rollouts.trace_types import (
    Episode,
    EventType,
    ToolCallPayload,
    ToolResultPayload,
    ValidationErrorPayload,
    RepairAttemptPayload,
    TerminalOutcomePayload,
)

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


def _count_tool_calls_in_output(raw: str) -> int:
    """Count how many ``"tool_call"`` occurrences appear in raw model output.

    Used to detect when the model batched multiple tool calls in one
    response so only the first is processed (matching NeMo Gym adapter
    sequential enforcement).
    """
    return raw.count('"tool_call"')


# -- Core episode loop (shared by all entrypoints) -------------------------

def _episode_loop_core(
    *,
    messages: list[dict[str, str]],
    recorder: EpisodeRecorder,
    max_iterations: int,
    call_model_fn: Callable[[list[dict[str, str]]], str],
    execute_tool_fn: Callable[[str, dict[str, Any]], dict[str, Any]],
    verbose: bool,
    label: str,
) -> Episode:
    """Shared think -> parse -> validate -> fallback -> execute -> observe loop.

    All three public entrypoints (run_agent, run_agent_episode,
    run_agent_episode_nat) delegate to this function after constructing
    the appropriate model-call and tool-dispatch callables.

    Args:
        messages: Mutable conversation history (system + user already present).
        recorder: EpisodeRecorder that accumulates structured events.
        max_iterations: Safety bound on loop iterations.
        call_model_fn: ``(messages) -> raw_response_str``.
        execute_tool_fn: ``(tool_name, arguments) -> result_dict``.
        verbose: Print each step as it happens.
        label: Human-readable prefix for verbose output (e.g. "Agent episode").
    """
    tools_called: list[str] = []

    if verbose:
        print(f"=== {label} started for {recorder.task_id} ===")
        print(f"    Max iterations: {max_iterations}")
        print()

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"--- Iteration {iteration} ---")

        # Step 1: Call the model
        try:
            raw_response = call_model_fn(messages)
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

        # Detect parallel tool calls
        n_tool_calls = _count_tool_calls_in_output(raw_response)
        if n_tool_calls > 1:
            recorder.record_metadata("parallel_tool_calls_dropped", {
                "iteration": iteration,
                "detected": n_tool_calls,
                "processed": 1,
            })
            if verbose:
                print(f"  -> Parallel tool calls detected ({n_tool_calls}); processing only the first")

        # Step 2: Validate -> repair -> reject via shared pipeline
        vr = validate_and_repair(raw_response, TOOL_REGISTRY, recorder=recorder)

        if vr.was_repaired and verbose:
            print(f"  -> Fallback REPAIRED: {vr.fallback_result.repairs_applied}")
        if vr.fallback_result and vr.fallback_result.action == FallbackAction.REJECTED and verbose:
            print(f"  -> Fallback REJECTED: {vr.fallback_result.rejection_reason}")

        result = vr.parsed

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
        if vr.validation_error is not None:
            recorder.record_validation_error(
                error_type=vr.validation_error.error_type,
                message=vr.validation_error.message,
                raw_model_output=raw_response,
            )
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
                "role": "user",
                "content": (
                    f"Error: {vr.validation_error.message} "
                    f"Please respond with a valid JSON tool call or final answer."
                ),
            })
            if verbose:
                print(f"  -> Validation error: {vr.validation_error.error_type}: {vr.validation_error.message}")
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
            tool_result = execute_tool_fn(result.tool_name, result.arguments)
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
        recorder.record_terminal("max_iterations")
        if verbose:
            print(f"\n  Stopped: reached max iterations ({max_iterations}).")

    episode = recorder.build_episode()

    if verbose:
        print(f"\n=== {label} finished ===")
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


# -- Episode -> AgentTrace converter (backward compat) ---------------------

def _episode_to_trace(episode: Episode, order_id: str, start_time: float) -> AgentTrace:
    """Convert a canonical Episode into a backward-compatible AgentTrace.

    Walks the episode's event list and reconstructs ToolCallRecords,
    error counts, and fallback stats that the legacy AgentTrace exposes.
    """
    trace = AgentTrace(task=f"Investigate order {order_id}")
    trace.wall_time_seconds = round(time.time() - start_time, 2)
    trace.model_calls = episode.metrics.model_calls

    iteration = 0
    for event in episode.events:
        if event.event_type == EventType.TOOL_CALL:
            payload = event.payload
            if isinstance(payload, ToolCallPayload):
                iteration += 1
                trace.steps.append(ToolCallRecord(
                    iteration=iteration,
                    thought=payload.thought,
                    tool_name=payload.tool_name,
                    arguments=payload.arguments,
                    result={},
                    valid=True,
                ))
        elif event.event_type == EventType.TOOL_RESULT:
            payload = event.payload
            if isinstance(payload, ToolResultPayload) and trace.steps:
                trace.steps[-1].result = payload.result
        elif event.event_type == EventType.TOOL_VALIDATION_ERROR:
            payload = event.payload
            if isinstance(payload, ValidationErrorPayload):
                iteration += 1
                trace.steps.append(ToolCallRecord(
                    iteration=iteration,
                    thought=None,
                    tool_name="<invalid>",
                    arguments={},
                    result={"error": payload.message},
                    valid=False,
                    validation_error=payload.message,
                ))
                trace.errors.append(payload.message)
        elif event.event_type == EventType.TOOL_REPAIR_ATTEMPT:
            payload = event.payload
            if isinstance(payload, RepairAttemptPayload):
                if payload.succeeded:
                    trace.fallback_repairs += 1
        elif event.event_type == EventType.TOOL_REJECT:
            trace.fallback_rejects += 1
        elif event.event_type == EventType.TERMINAL_OUTCOME:
            payload = event.payload
            if isinstance(payload, TerminalOutcomePayload):
                if payload.reason == "final_answer" and payload.final_answer:
                    trace.final_answer = payload.final_answer
                    trace.completed = True
                    trace.stop_reason = "final_answer"
                else:
                    trace.stop_reason = payload.reason

    if trace.stop_reason is None:
        trace.stop_reason = "max_iterations"

    trace.messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": episode.task_prompt},
    ]

    return trace


# -- Public entrypoints ----------------------------------------------------

def run_agent(
    order_id: str,
    max_iterations: int = MAX_ITERATIONS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.1,
    verbose: bool = True,
) -> AgentTrace:
    """Execute the full agent loop for an order (backward-compatible).

    Delegates to run_agent_episode() and converts the resulting Episode
    to an AgentTrace for callers that rely on the legacy format.

    Args:
        order_id: The order to investigate (e.g. "SO-10482").
        max_iterations: Safety bound on loop iterations.
        max_tokens: Max tokens per model response.
        temperature: Sampling temperature.
        verbose: Print each step as it happens.

    Returns:
        AgentTrace with the full trajectory.
    """
    start_time = time.time()
    episode = run_agent_episode(
        order_id, max_iterations=max_iterations, max_tokens=max_tokens,
        temperature=temperature, verbose=verbose,
    )
    return _episode_to_trace(episode, order_id, start_time)


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
        task_id=order_id, task_prompt=task_prompt, model_id=MODEL_NAME,
    )

    system_msg = {"role": "system", "content": build_system_prompt()}
    user_msg = {"role": "user", "content": task_prompt}
    messages: list[dict[str, str]] = [system_msg, user_msg]
    recorder.record_user_task(task_prompt)

    def _call_model(msgs: list[dict[str, str]]) -> str:
        return call_model(msgs, max_tokens=max_tokens, temperature=temperature)

    def _execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn, _, _ = TOOL_REGISTRY[name]
        return fn(**args)

    return _episode_loop_core(
        messages=messages,
        recorder=recorder,
        max_iterations=max_iterations,
        call_model_fn=_call_model,
        execute_tool_fn=_execute_tool,
        verbose=verbose,
        label="Agent episode",
    )


def run_agent_episode_nat(
    order_id: str,
    max_iterations: int = MAX_ITERATIONS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.1,
    verbose: bool = True,
) -> Episode:
    """Execute the agent loop using NAT FunctionGroup dispatch and NIM config.

    This is the NAT-aligned version of run_agent_episode(). Instead of
    dispatching tools through the repo's TOOL_REGISTRY dict and calling
    the model via raw HTTP, it:

        1. Registers tools in a NAT FunctionGroup via build_nat_function_group()
        2. Calls the model via call_model_nim() using NIMModelConfig
        3. Dispatches tool execution through invoke_tool_via_group()

    The parse -> validate -> fallback -> observe cycle is identical to
    run_agent_episode() via the shared _episode_loop_core(), and the
    output is the same canonical Episode.

    Args:
        order_id: The order to investigate (e.g. "SO-10482").
        max_iterations: Safety bound on loop iterations.
        max_tokens: Max tokens per model response.
        temperature: Sampling temperature.
        verbose: Print each step as it happens.

    Returns:
        Episode with canonical structured events.
    """
    import asyncio
    from src.runtime.nat_tools import build_nat_function_group, invoke_tool_via_group
    from src.runtime.nat_llm import build_nim_config, call_model_nim

    function_group = build_nat_function_group()
    nim_config = build_nim_config(max_tokens=max_tokens, temperature=temperature)

    task_prompt = build_task_message(order_id)
    recorder = EpisodeRecorder(
        task_id=order_id, task_prompt=task_prompt, model_id=nim_config.model_name,
    )

    system_msg = {"role": "system", "content": build_system_prompt()}
    user_msg = {"role": "user", "content": task_prompt}
    messages: list[dict[str, str]] = [system_msg, user_msg]
    recorder.record_user_task(task_prompt)

    if verbose:
        print(f"    Model: {nim_config.model_name} via NIMModelConfig")

    def _call_model(msgs: list[dict[str, str]]) -> str:
        return call_model_nim(
            msgs, config=nim_config, max_tokens=max_tokens, temperature=temperature,
        )

    def _execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name in TOOL_REGISTRY:
            fn, _, _ = TOOL_REGISTRY[name]
            return fn(**args)
        # NAT fallback for tools not in the local registry
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run,
                        invoke_tool_via_group(function_group, name, args),
                    ).result()
            else:
                return loop.run_until_complete(
                    invoke_tool_via_group(function_group, name, args),
                )
        except Exception as e:
            return {"error": f"Tool execution failed: {e}"}

    return _episode_loop_core(
        messages=messages,
        recorder=recorder,
        max_iterations=max_iterations,
        call_model_fn=_call_model,
        execute_tool_fn=_execute_tool,
        verbose=verbose,
        label="NAT agent episode",
    )


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
    """Export a trace as a list of trajectory steps for training export.

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
