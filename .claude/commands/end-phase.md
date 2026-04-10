---
description: Close a development phase from PLAN.md
---

Read `@PLAN.md` and `@CLAUDE.md`.

Close the phase identified by: `$ARGUMENTS`

Before finishing:
- confirm the work is on the dedicated phase branch created for this phase, not on `main`
- review the completed work against the phase goals in `PLAN.md`
- summarize what was completed, what changed, and which files were updated
- identify any gaps, deviations, blockers, or follow-up items that remain
- run the relevant validation or tests and report the results clearly
- state whether the phase is complete, partially complete, or blocked
- recommend the next phase to start

If the phase is not fully complete, explain exactly what remains.

If the phase is complete:
- create a commit with a message that reflects the phase outcome
- push the phase branch to GitHub
- open a pull request from the phase branch into `main`
- make the pull request summary include the phase goal, key changes, validation results, and any remaining follow-up items
- return the pull request URL in the final report
