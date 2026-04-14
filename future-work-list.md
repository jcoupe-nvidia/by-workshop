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

See `RL_ARCHITECTURE.md` § Warm-Start and Async GRPO and MEDIUM-4 in the code review for background.

### 4. Colocated GRPO weight refit fails with ModelOpt FP8 Nemotron-H checkpoint

**Status:** Blocking — the GRPO training run cannot complete.

**Summary:** The `run_grpo_training.py` colocated GRPO pipeline fails during the weight refit phase (DTensor policy → vLLM) after the first generation step. Two distinct errors occur:

1. **Shape mismatch on Mamba `in_proj` weights.** The Nemotron-3-Nano checkpoint (`/models/nemotron3nano`) is an FP8-quantized ModelOpt checkpoint (`hf_quant_config.json` with `quant_algo: FP8`). During vLLM's initial model load, `ModelOptFp8LinearMethod.process_weights_after_loading` transposes certain linear weights (e.g. Mamba `mixer.in_proj.weight` from `[10304, 2688]` to `[2688, 10304]`) for efficient FP8 GEMM. After the DTensor policy completes a training step and sends updated BF16 weights back via IPC ZMQ, `default_weight_loader` asserts `param.size() == loaded_weight.size()` — which fails because the DTensor policy retains the original HF layout while vLLM's internal parameters are transposed.

2. **Missing `weight_loader` attribute.** Even after patching `default_weight_loader` to auto-transpose reversed 2D shapes, a second error occurs: vLLM's `NemotronHMixer.load_weights` tries to access `param.weight_loader` on FusedMoE expert parameters whose `weight_loader` attribute was removed when `ModelOptFp8LinearMethod.process_weights_after_loading` replaced `layer.weight` with a new `Parameter(weight.t())` during initial load.

**Root cause:** NeMo RL's FP8 refit support patches `Fp8LinearMethod.process_weights_after_loading` and `Fp8MoEMethod.process_weights_after_loading` (block-quantized FP8 path) but does **not** patch `ModelOptFp8LinearMethod.process_weights_after_loading` or `ModelOptFp8MoEMethod.process_weights_after_loading`. The ModelOpt quantization path creates new `Parameter` objects (losing `weight_loader`) and transposes weights — both of which break the refit cycle.

**Environment details:**
- NeMo RL 0.5.0rc0, vLLM 0.11.2, Ray 2.49.2
- 8× NVIDIA H100 NVL (93 GiB each)
- Model: `nvidia/nemotron-3-nano` (FP8 ModelOpt, `torch_dtype: bfloat16`)
- Config: colocated vLLM + DTensor policy, `enable_sleep_mode: true`, `cpu_offload: true`

**Files involved:**
- `/opt/nemo-rl/nemo_rl/models/generation/vllm/vllm_backend.py` — `update_weights_via_ipc_zmq` and the COMPLETE handler that calls `process_weights_after_loading`
- `/opt/nemo-rl/nemo_rl/models/generation/fp8.py` — patches `Fp8LinearMethod` but not `ModelOptFp8LinearMethod`
- `vllm/model_executor/layers/quantization/modelopt.py:310-317` — `process_weights_after_loading` transposes weight and creates new Parameter
- `vllm/model_executor/model_loader/weight_utils.py:889` — `default_weight_loader` shape assertion
- `vllm/model_executor/models/nemotron_h.py:683` — accesses `param.weight_loader` on expert weights

**Attempted fixes (reverted):**
- Auto-transpose in `default_weight_loader`: fixes the shape assertion but exposes the `weight_loader` attribute error downstream.
- Skip `process_weights_after_loading` in the COMPLETE handler: the `weight_loader` error occurs inside `load_weights` itself (in the NemotronHMixer expert loading path), not in the post-processing.
- Adding `_fix_transposed_weights` helper in `vllm_backend.py`: parameter name mismatch between DTensor (HF names) and vLLM (mapped names via `hf_to_vllm_mapper`) caused silent no-ops.

**Recommended fix path:**
1. Upstream to NeMo RL: add ModelOpt-aware patches in `fp8.py` alongside the existing `Fp8LinearMethod` patches — specifically patching `ModelOptFp8LinearMethod.process_weights_after_loading` to avoid replacing Parameters (matching the pattern used for `Fp8LinearMethod`).
2. Alternatively, provide a BF16 (non-quantized) checkpoint for training. The FP8 checkpoint is optimized for inference; GRPO training dequantizes to BF16 anyway, so a native BF16 checkpoint would avoid the ModelOpt code path entirely.
3. As a short-term workaround, use non-colocated mode (separate vLLM inference server + DTensor training) to avoid the weight refit path. This requires dedicating GPUs to each role.

**Additional operational notes from the attempt:**
- GPU 0 was initially occupied by a NIM inference container; `CUDA_VISIBLE_DEVICES` was used to exclude it. The `grpo_config.yaml` default of `gpus_per_node: 7` assumes GPU 0 is reserved.
- Abruptly killed Ray workers leak CUDA IPC memory that persists until container restart. Multiple retries consumed GPUs 4 and 7 this way. In future attempts, use `ray stop --force` and verify `nvidia-smi` shows 0 MiB before relaunching.
- Setup takes ~200s (vLLM engine init: ~120s, DTensor policy load: ~15s, CUDA graph capture: ~50s, overhead: ~15s). The CUDA graph capture is sequential across workers in colocated mode.
