---
name: assess_primary_fulfillment
description: Check whether the original source distribution center can still fulfill the order on time given current inventory and capacity.
tags:
  - assessment
  - primary-dc
  - inventory
  - capacity
tools:
  - get_inventory
  - get_fulfillment_capacity
inputs:
  order_id: The sales order ID (diagnosis must already exist in context).
outputs:
  can_fulfill: Whether the source DC can fulfill the full order.
  available_qty: Units currently available at the source DC.
  shortfall: Units short of the ordered quantity (0 if sufficient).
  capacity_ok: Whether the DC has fulfillment capacity on the committed date.
  source_dc: The source DC that was assessed.
preconditions:
  - diagnose_order_risk
next_skills:
  - evaluate_alternate_recovery_paths
---

# Assess Primary Fulfillment

This skill checks whether the originally assigned source DC can still fulfill the order, given current inventory levels and fulfillment capacity constraints.

## Purpose

Before exploring alternate recovery paths, determine whether the simplest option — fulfilling from the source DC — is still viable.

## Tool Sequence

1. **get_inventory(sku, dc_id)** — Check on-hand, reserved, and available inventory at the source DC.
2. **get_fulfillment_capacity(dc_id, date)** — Check whether the DC has processing capacity on the committed delivery date.

## Decision Logic

The source DC **can fulfill** when:
- Available inventory >= ordered quantity (no shortfall)
- AND the DC has remaining fulfillment capacity on the committed date

## Preconditions

- `diagnose_order_risk` must have completed (provides SKU, source DC, and committed date).
