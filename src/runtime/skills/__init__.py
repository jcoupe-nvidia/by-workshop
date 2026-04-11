"""
Directory-backed skills package for the late-order recovery scenario.

Each skill lives in its own subdirectory with a SKILL.md definition file
and optional sidecar assets. The api module provides canonical discovery
and execution interfaces: list_skills, search_skills, get_skill, and
run_skill_command.

Skill directories:
    diagnose-order-risk/             -- order and shipment diagnosis
    assess-primary-fulfillment/      -- source DC inventory and capacity check
    evaluate-alternate-recovery-paths/ -- alternate DCs, supplier expedite, substitutes
    synthesize-recommendation/       -- score options and produce final recommendation
"""
