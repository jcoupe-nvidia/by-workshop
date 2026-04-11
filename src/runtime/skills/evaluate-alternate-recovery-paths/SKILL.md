---
name: evaluate_alternate_recovery_paths
description: Explore alternate recovery options including DC transfers, supplier expedite, and substitute SKUs when the primary source DC cannot fulfill the order.
tags:
  - recovery
  - alternate-dc
  - supplier-expedite
  - substitutes
tools:
  - find_alternate_inventory
  - get_transfer_eta
  - get_supplier_expedite_options
inputs:
  order_id: The sales order ID (diagnosis and primary assessment must exist in context).
outputs:
  recovery_paths: List of candidate recovery options, each with source, type, available quantity, ETA, cost, and feasibility.
preconditions:
  - diagnose_order_risk
  - assess_primary_fulfillment
next_skills:
  - synthesize_recommendation
---

# Evaluate Alternate Recovery Paths

This skill explores all viable alternate fulfillment paths when the primary source DC cannot fulfill the order on time.

## Purpose

Gather concrete, comparable recovery options so the final recommendation can be made from real data rather than assumptions.

## Tool Sequence

1. **find_alternate_inventory(sku, region='ALL')** — Search all DCs for available inventory of the SKU (and any registered substitutes).
2. **get_transfer_eta(from_dc, to_dc, sku, qty)** — For each alternate DC with stock, estimate transfer time and cost to the destination.
3. **get_supplier_expedite_options(sku, qty)** — Check whether suppliers can rush-ship the needed quantity.

## Recovery Path Types

- **dc_transfer** — Transfer units from an alternate DC that has stock.
- **supplier_expedite** — Rush order from a supplier with expedite capability.
- **substitute** — Use a compatible substitute SKU from an alternate DC (discovered via find_alternate_inventory).

## Preconditions

- `diagnose_order_risk` must have completed (provides SKU, quantity, region).
- `assess_primary_fulfillment` must have completed (provides shortfall quantity).
