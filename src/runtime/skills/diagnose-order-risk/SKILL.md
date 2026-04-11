---
name: diagnose_order_risk
description: Understand the current state of an order and its shipment to determine whether it is at risk of missing the committed delivery date.
tags:
  - diagnosis
  - order-risk
  - entry-point
tools:
  - get_order
  - get_shipment_status
inputs:
  order_id: The sales order ID to diagnose.
outputs:
  is_at_risk: Whether the order is at risk of missing its committed date.
  reason: Human-readable explanation of the risk assessment.
  days_remaining: Days until the committed delivery date.
  sku: The SKU on the order.
  qty: The ordered quantity.
  source_dc: The assigned source distribution center.
  committed_date: The committed delivery date.
preconditions: none
next_skills:
  - assess_primary_fulfillment
---

# Diagnose Order Risk

This is the **entry skill** for the late-order recovery workflow. It establishes the baseline facts about an order by looking up the order details and checking the current shipment status.

## Purpose

Determine whether a customer order is at risk of missing its committed delivery date. This skill gathers the foundational facts that all subsequent skills depend on.

## Tool Sequence

1. **get_order(order_id)** — Retrieve order details: SKU, quantity, source DC, committed date.
2. **get_shipment_status(order_id)** — Check current shipment state: status, shipped quantity, estimated arrival.

## Decision Logic

The order is considered **at risk** when:
- Shipment status is `pending`, `delayed`, or `partial`
- AND there are 10 or fewer days until the committed delivery date

## Outputs

Returns an `OrderRiskDiagnosis` with all fields needed by downstream skills: order identity, risk flag, reasoning, and timeline.
