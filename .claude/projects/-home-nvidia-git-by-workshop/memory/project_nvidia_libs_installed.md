---
name: NVIDIA libs installed
description: nvidia-nat 1.6.0, nemo-gym 0.2.0, openpipe-art 0.5.17 are installed in the dev environment
type: project
---

Installed versions as of 2026-04-11:
- nvidia-nat 1.6.0 (includes nvidia-nat-atif 1.6.0, nvidia-nat-core 1.6.0)
- nemo-gym 0.2.0
- openpipe-art 0.5.17

**Why:** These are the actual libraries the Phase 9 integration must target.

**How to apply:** Use real imports from these packages. Verify module paths and function signatures against the installed code before writing adapters.
