---
name: synthesize_recommendation
description: Score all candidate recovery options and produce a final actionable recommendation for the order.
tags:
  - recommendation
  - scoring
  - terminal
tools:
  - score_recovery_options
  - recommend_action
inputs:
  order_id: The sales order ID (all prior skills must have completed).
outputs:
  action: The recommended recovery action.
  rationale: Explanation of why this action was chosen.
  expected_delivery: Estimated delivery date under the recommended action.
  meets_committed_date: Whether the recommendation meets the original committed date.
  confidence: Confidence score for the recommendation (0.0 to 1.0).
  scored_options: All options with their scores for transparency.
preconditions:
  - diagnose_order_risk
  - assess_primary_fulfillment
  - evaluate_alternate_recovery_paths
next_skills: []
---

# Synthesize Recommendation

This is the **terminal skill** in the late-order recovery workflow. It takes all gathered recovery paths, scores them against the business objective, and produces a final actionable recommendation.

## Purpose

Transform raw recovery options into a scored, ranked list and select the best action. The output is the agent's final answer to the user's question.

## Tool Sequence

1. **score_recovery_options(options, objective)** — Score and rank all candidate recovery paths against the objective (e.g., `minimize_delay`, `minimize_cost`).
2. **recommend_action(context)** — Produce the final recommendation with the best option, order context, and objective.

## Scoring Dimensions

Each option is scored on:
- **Time** — Lower lead time scores higher.
- **Cost** — Lower cost per unit scores higher.
- **Feasibility** — Whether the option is actually executable.
- **Coverage** — Whether the option covers the full shortfall quantity.

## Preconditions

- All three prior skills must have completed.
- At least one recovery path must exist (from `evaluate_alternate_recovery_paths`).
