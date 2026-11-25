# Lumnus B70 engine, l34 (reference, not a release)

This branch documents **l34**: the vLLM engine we run for **Qwen3.8-Flash-Next on four Intel Arc Pro B70 cards** (Xe2, Battlemage, 32 GB each). It is the line we built and served on through the vLLM 0.34 cycle. We publish it as a reference: the code, the exact stack it ran on, and what we measured. It is not a supported release, it is not upstream-clean, and there are no binaries. If you run vLLM on Xe2 cards, the commits and the numbers may save you some work.

Our released kit (b70.7, vLLM 0.32) stays on `main` of this repo. l34 is its successor line.

## 1. What it is

1. An engine branch: vLLM `l34` = [zhaohb's Qwen4Exp-on-XPU branch (vllm #61607)](https://github.com/vllm-project/vllm/pull/61607) + vLLM main through the v0.34.0 release + the open upstream PRs listed in §6 + our own changes, each written as a PR-shaped commit.
2. A kernels branch: vllm-xpu-kernels `l34`, with our Xe2 kernels (W4A8 grouped GEMM, INT8 KV decode and prefill, MoE routing and gather ops).
3. The model: Qwen3.8-Flash-Next, a hybrid model (linear-attention GDN layers plus full attention, mixture of experts, per-layer n-gram embeddings, hyper-connections), served with a DFlash2 drafter, a 262,144-token context and 16 concurrent sequences on 4 cards, tensor and expert parallel.

## 2. The stack

Four parts, plus the driver stack. The vLLM and kernels branches are required; the patched torch libraries and the Triton build are what we ran.

| part | where | version |
|---|---|---|
| vLLM engine | [`Lumnus/vllm` branch `l34`](https://github.com/Lumnus/vllm/tree/l34), tag `l34-final` | commit `e1f970f791ccc96f1904509748c42658c813c83a`; run from the source tree (`PYTHONPATH`) |
| XPU kernels | [`Lumnus/vllm-xpu-kernels` branch `l34`](https://github.com/Lumnus/vllm-xpu-kernels/tree/l34), tag `l34-final` | commit `e245d929a5`, wheel `vllm_xpu_kernels-0.1.18.dev0+l34.k17k.e245d929a` (build it from the branch) |
| PyTorch | official `torch 2.15.0+xpu` wheel | optional: `libc10_xpu.so` and `libtorch_xpu.so` rebuilt with the two patches in [`l34/patches/pytorch/`](l34/patches/pytorch/) (§3 item 11) |
| Triton | `triton_xpu 3.9.1` built from intel-xpu-backend-for-triton main `9b4b1c768` plus ten open upstream PRs (#8769, #8736, #8761, #8391, #8740, #8122, #8765, #8711, #8660, #8289) | `3.9.1+git13963861.l34`; the stock 3.9 wheel also loads |
| oneAPI runtime | pip `intel-sycl-rt 2026.2.0`, `oneccl 2022.1.2`, `mkl 2026.1.0`, `umf 1.2.0` | Python 3.12 |
| GPU driver stack | compute-runtime 26.35.39758.10, IGC 2.41.5, gmmlib 22.10.0; Linux 6.18.63 `xe`, GuC 70.72.1 | |

Every version, wheel hash and checkpoint hash is in [`l34/MANIFEST.md`](l34/MANIFEST.md).

## 3. Main features

1. **Native INT8 KV cache (per token, per head) on FlashAttention XPU.** KV is stored as int8 with per-token, per-head scales. Decode runs a native INT8 paged-decode kernel. Prefill runs INT8 inside the SYCL chunk-prefill kernel (`--attention-config '{"int8_prefill_dot":"sycl_qk_pv"}'`: Q·K and P·V as s8×s8→s32, no dequantized copy of the cache), 18 % faster than prefill over a BF16 copy at 8K to 128K and with no scratch buffer to reserve.
2. **INT8 and W4A8 execution.** The `x2` checkpoint keeps the experts at int4 weights with int8 activations (W4A8), and attention, linear attention, the shared expert and the hyper-connection projections at calibrated W8A8; the head is INT8. On the cards: an Xe2 W4A8 grouped GEMM (s8×s4 DPAS), a fused W4A8 decode MoE, a native per-token INT8 activation quant, oneDNN W8A8 above 64 rows and Triton below.
3. **DFlash2 speculative decoding with an 8-bit drafter** (W8A8, 7 draft tokens). The MTP path also runs (k=3), with the drafter's linears quantized; DFlash accepts more.
4. **Extensible KV cache.** The device KV pool grows and shrinks while the engine serves, on growable device buffers; a 128 GiB host KV tier sits behind it.
5. **Expert-parallel MoE with EPLB.** An incremental rebalancing policy that moves a bounded number of experts per step, the expert placement saved and restored across restarts, load weighted by prefill and decode tokens, device and host expert tiers with staging, and an expert store.
6. **Vision tower in host RAM, promoted on demand.** The image encoder lives in host memory and moves to the cards after repeated use, then back after an idle period, so a text-only workload keeps that memory for KV.
7. **Runtime control and serving profiles.** `GET /runtime_control` reports every dynamic subsystem per rank; `POST /runtime_control/{op}` changes one setting between two steps (KV size, expert tiers, EPLB, drafter tokens, scheduler, tower residency); `/runtime_control/schema` lists every parameter with its unit, range and whether it applies live. A profile is a JSON set of those settings applied in one ordered pass. The engine never retunes itself; the caller decides.
8. **QSA sparse attention variant** of the same model and tree, with the INT8 KV cache.
9. **Han mask.** Qwen3.8-Flash-Next leaks Chinese characters into answers in other languages. The engine masks Han tokens at sampling time; in `auto` mode the mask turns off only when the user's own latest message contains Chinese or asks for it. A per-request flag overrides it.
10. **Fused hyper-connection kernels**: the up projection fused, the down projection folded into INT8, batch-invariant split-K.
11. **Card-0 memory fixes.** Out of the box every process that initializes XPU builds a 96-queue stream pool on card 0, so card 0 has less free memory than the others and the KV pool, sized by the poorest card, shrinks. Here the API server and engine core never initialize XPU, each worker selects its card before torch initializes, the kernels' KV-offload copy creates no stream pools, and two torch-xpu patches (a capture-status guard and lazily created stream queues) remove the rest. The card-to-card gap at startup drops from 0.72 to 0.06 GiB.
12. **Smaller pieces:** top-k-first sampling and verification on XPU, the n-gram embedding table in INT8 in pinned host memory with an in-graph gather, and an XPU sizing of the KV pool without CUDA's growth headroom.

## 4. Measured

One machine: 4× Arc Pro B70 on one host, tensor parallel 4 with expert parallel, 16 sequences, 262,144-token context. Every row was taken with no other client on the engine (0 running and 0 waiting requests at each row). Decode per step = 1000 × (1 + accepted tokens per draft) / tok/s; two runs per row. The b70.7 column is the released kit on the same cards; "recorded" means taken in an earlier window, not side by side.

### 4.1 The l34 serving engine (x2 checkpoint, INT8 execution, INT8 KV) against b70.7 D8-7

| row | l34 | b70.7 D8-7 | read |
|---|---|---|---|
| decode, 1 stream, prose (t 0.7): ms per step · tok/s | 19.21–19.27 · 122–123 | 19.57 · 115–120 (recorded) | level per step |
| decode, 1 stream, code (t 0): ms per step · tok/s | 19.28–19.30 · 253–279 | 19.04 · 292–302 (recorded) | level per step; fewer tokens accepted per draft on code (3.9–4.4 against 4.7) |
| prefill 32K, uncached (tok/s) | **7,668–7,675** | 6,753–6,790 (recorded) | +13 % |
| prefill 131K, uncached (tok/s) | **5,860–5,865** | 5,003–5,006 (recorded) | +17 % |
| KV on the cards (tokens, 262,144 context) | **992,043**; 1,003,974 at a later boot | 678,259 | +46 % |
| quality: MMLU 300 / TruthfulQA-MC1 200 | 0.920 / 0.885 | — | no paired difference against Intel's W4A16 AutoRound build on the same engine (0.920 / 0.880) |
| decode, 16 streams | not measured on this checkpoint | 766–769 prose, 1,903–1,941 code | see §4.2 for the BF16 variant |

The checkpoints differ: b70.7 D8-7 runs Intel's W4A16 AutoRound weights with online INT8; l34 runs the calibrated x2 W4A8 weights.

### 4.2 BF16 execution, INT8 KV, Intel W4A16 weights: l34 against b70.7 D16e-7, same window

Measured at the l34 cut (2026-11-11), both engines in one window, same harness.

| row | l34 | b70.7 D16e-7 | read |
|---|---|---|---|
| decode, 1 stream: ms per step, prose / code | 21.15 / 21.26 | 21.45 / 21.6–21.8 | −1 to −2 % |
| decode, 16 streams: tok/s, prose / code | 727–733 / 1,734–1,749 | 684 / 1,671 | +6 % / +4 % |
| prefill 32K / 128K (tok/s) | 6,556–6,571 / 4,864–4,869 | 6,186–6,341 / 4,712–4,722 | +4 % / +3 % |
| 16 × 2K prompts: prefill / decode (tok/s) | 6,704 / 844 | 6,190–6,342 / 770–773 | +6 % / +9 % |
| KV (tokens) | 637,611 | 606,145 | +5 % |
| quality MMLU / TQA | no paired difference | reference | |
| drafter acceptance, prose · code | 1.313 · 4.418 | 1.30–1.32 · 4.40 | level |

### 4.3 INT8 prefill attention kernel (one card, ms for a 4,096-token chunk at the end of an L-token sequence)

| L | over a BF16 copy | `sycl_qk_pv` | change |
|---|---|---|---|
| 8K | 2.427 | 1.972 | −18.7 % |
| 32K | 9.730 | 8.004 | −17.7 % |
| 128K | 39.483 | 32.362 | −18.0 % |

## 5. Known limitations

1. A reference, not a release: tested on one machine with one model family, and not upstream-clean. Many changes are open upstream PRs or PR-shaped commits not yet posted (§6); the Han mask is our own non-upstream change.
2. It needs our kernels branch; the stock vllm-xpu-kernels wheel does not carry the W4A8, INT8-KV and routing ops.
3. The x2 W4A8 checkpoint is not published yet. The same branch serves Intel's public `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` with BF16 execution (§4.2).
4. The 2-card x2 configuration has booted and passed its gates once; it has not been run at scale.
5. The torch patches are drop-in libraries for the official 2.15.0 wheel. The rebuilt `libtorch_xpu.so` does not export ten `std::filesystem` symbols that the official `libtorch_cpu.so` takes from it; a recent libstdc++ (6.0.33) supplies them, an older one does not.
6. A first start has no saved expert placement and decodes at less than half speed until EPLB has rebalanced under load once (a few minutes; the preset explains how). Later starts restore the saved placement.
7. Cross-boot output at more than one concurrent sequence is equal only up to near-ties; single-stream output is deterministic within a boot.

## 6. How to run

Build the kernels wheel from its branch, install it into an XPU venv with `torch 2.15.0+xpu`, check out the vLLM branch, and fetch the weights. Then, on 4 cards (placeholders in `<>`; the full line with every switch is [`l34/presets/qfn4-l34d8di8-8.sh`](l34/presets/qfn4-l34d8di8-8.sh)):

```bash
PYTHONPATH=<VLLM_TREE> \
VLLM_XPU_MOE_ENTRY_QUANT=1 VLLM_XPU_MOE_SHARED_IN_GATHER=1 VLLM_XPU_HC_UP_FUSED=3 VLLM_XPU_HC_DOWN_FOLD=1 \
VLLM_XPU_W4A8_FUSED_MOE=1 VLLM_XPU_ATTN_META_ONCE=1 VLLM_XPU_MOE_TOPK_ROUTE=1 VLLM_XPU_MOE_ROUTE_FUSED=1 \
B70_CJK_MASK=1 B70_CJK_MASK_DEFAULT=auto ZE_AFFINITY_MASK=0,1,2,3 ZE_FLAT_DEVICE_HIERARCHY=FLAT \
<VENV>/bin/python -m vllm.entrypoints.cli.main serve <MODEL_DIR>/qfn-w4a8-x2 \
  --tensor-parallel-size 4 --enable-expert-parallel --max-model-len 262144 --max-num-seqs 16 \
  --kv-cache-dtype int8_per_token_head --attention-backend FLASH_ATTN \
  --attention-config '{"int8_prefill_dot":"sycl_qk_pv"}' \
  --speculative-config '{"method":"dflash","model":"<DRAFTER_DIR>","num_speculative_tokens":7,"draft_tensor_parallel_size":4,"attention_backend":"TRITON_ATTN"}' \
  --engram-config '{"cpu_offload":true,"int8_table_path":"<PLE_INT8_TABLE>"}' \
  --quantization-config '{"targets":{"re:.*lm_head$":"int8_per_channel","re:.*\\.mlp\\.experts$":{"activation":"int8_per_token"}}}' \
  --kv-offloading-size 128 --kv-offloading-backend native ...
```

The drafter is the public `incoai/Qwen3.8-Flash-Next-DFlash2` (revision `a22a174`) quantized to W8A8 (int8 per output channel, dynamic per-token activations, data-free); the INT8 n-gram table is built with `tools/build_int8_ple.py` on `main`. The 4-card line pins about 180 GiB of host RAM: 49 GiB for the n-gram table and 128 GiB for the host KV tier.

## 7. Upstream PRs it carries

State as of 2026-11-18; none of them is merged unless it says so.

1. **The base:** vllm #61607 (zhaohb, Qwen4Exp on XPU), open; our ten fixes to it, zhaohb/vllm #1–#10, open.
2. **Third-party PRs carried:** vllm #62113 (16-byte-aligned per-token-head KV), #64190 and #64244 (Triton W8A8 INT8 GEMM), #63347 (padded full-attention block growth), #65280 (extensible KV cache on XPU), #62060 (staging UVA-offloaded experts), #61815 and #58131 (merged upstream after our base); vllm-xpu-kernels #196 (oneDNN W8A8), #487 (fail closed on missing FA2 shapes), #644 (INT4 GEMV), #692 (VMM ops for growable buffers).
3. **Ours, posted and open:** vllm #64925 and #64926 (QSA on XPU), #63564 (prompt-logprobs chunk bound), #66365, #66367; drafts #66369, #66371–#66374 (UVA expert placement and staging); vllm-xpu-kernels #688 (native per-token INT8 activation quant), #662, #666; oneCCL #229.
4. **Ours, written as PR branches and not posted:** 23 branches, among them the FlashAttention INT8 per-token-head KV route and its paged-decode kernel (credit to wiesion's kernel in intel/llm-scaler #744), QSA with INT8 KV, W4A8 MoE, online INT8 linears and head, the EPLB series (int4 support, state file, load weighting, slot tiers, incremental policy), and the native INT8 activation-quant wiring.
5. **Mapped but not yet written:** the rest of the line, as 13 kernels PRs and 15 vLLM PRs cut from upstream main.

## 8. Credits and license

Built on [vLLM](https://github.com/vllm-project/vllm) and [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels), on zhaohb's Qwen4Exp XPU branch, on [wu1ff's B70 chain](https://github.com/wu1ff/B70-LLM-Controller), and on the third-party PRs named in §7, whose authorship the branches keep. Apache-2.0, as on `main` (LICENSE, NOTICE). Model weights are not included and keep their own licenses.
