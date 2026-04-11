---
name: Use real library imports
description: When integrating NAT, NeMo Gym, or openpipe-art, use their actual APIs — not interface-matching stubs or guessed signatures
type: feedback
---

Imports from NAT, NeMo Gym, and openpipe-art must call the real installed libraries, not just match the interface shape.

**Why:** User explicitly corrected this — the libraries are installed and the actual imports need to be used, not approximations.

**How to apply:** Before writing integration code, inspect the installed package's actual module structure and verify the real class/function signatures. Don't guess API shapes.
