# Future Work List

Items deferred from the code review that are out of scope for the current repo but should be tracked for future iterations.

## Deferred Items

### 1. NeMo RL on-policy token ID preservation for multi-step training

`run_grpo_training.py` uses `LateOrderTrainingEnv.step()` which processes detokenized assistant messages. The raw token IDs from vLLM generation are not explicitly preserved and re-used across steps. NeMo RL may handle this internally via its `force_on_policy_ratio` config, but it is unclear whether multi-step re-tokenization artifacts affect training quality for this scenario.

This is deferred because on-policy token ID management is owned by the NeMo RL framework, not by this repo. The repo's responsibility is to preserve the information needed for the fix (which it does via the `LateOrderTrainingEnv` marginal reward design), but the actual correction is a framework-level concern. Revisit if training quality regressions are observed that correlate with multi-step trajectory length.

### 2. NeMo Gym `BaseSeedSessionResponse` session_id propagation fragility

The `seed_session` method returns `BaseSeedSessionResponse(session_id=session_id)`, but whether `BaseVerifyRequest` actually carries a `session_id` field depends on the NeMo Gym version. The code uses `object.__setattr__` to attach it (line 776 of `nemo_gym_rollouts.py`), suggesting the field is not natively supported. This may break with NeMo Gym updates.

Be aware of this when upgrading `nemo-gym` beyond `0.2.0`. If a future NeMo Gym version adds native session_id support to verify requests, the `object.__setattr__` workaround should be replaced with the native API. If session_id propagation breaks after an upgrade, this is the likely root cause.

### 3. SFT warm-start demonstrations for RL

The curriculum infrastructure supports an SFT stage (`TrainingStage.SFT_SUCCESSFUL` in `src/training/curriculum.py`, `build_sft_datum_spec` in `src/training/nemo_rl_adapter.py`), but no curated demonstrations exist yet. Without SFT warm-start, the GRPO training run must learn both format (correct JSON envelope, argument naming conventions, chat template structure) and strategy (tool sequencing, recovery reasoning) simultaneously, wasting RL compute on format learning.

Add 3–5 hand-curated SFT demonstrations covering the distinct scenario types: DC transfer, supplier expedite, partial fulfillment, false alarm, and escalation. Each demonstration should include exemplary reasoning in the `thought` field and a correct tool sequence. Store them in `src/rollouts/sft_demonstrations.py` and wire them into the SFT stage of the curriculum. This is also a workshop teaching opportunity ("here is what good demonstrations look like and why they help RL").

See `RL_ARCHITECTURE.md` practice 17 and MEDIUM-4 in the code review for background.
