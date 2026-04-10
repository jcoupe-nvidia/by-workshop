---
description: Start a development phase from PLAN.md
---

Read `@PLAN.md` and `@CLAUDE.md`.

Start the phase identified by: `$ARGUMENTS`

Before making changes:
- create a dedicated git branch for the phase using a predictable name such as `phase/<slug-from-arguments>`
- verify whether a branch for this phase already exists; if it does, switch to it instead of creating a duplicate
- base each new phase branch from the latest `main`
- restate the goal of the selected phase in 2-4 sentences
- identify the files you expect to create or edit
- break the phase into concrete implementation steps
- call out dependencies, assumptions, and risks that could affect the phase
- ask any critical clarification questions before coding if something is ambiguous

During execution:
- keep all changes for the phase scoped to that branch
- treat the branch as the source branch for a GitHub pull request that will be opened when the phase ends

Then execute the phase with changes scoped only to that phase, and report progress against the phase deliverables as you go.
