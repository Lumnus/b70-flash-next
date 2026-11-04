0.32.0-b70.6 of b70-flash-next: vLLM 0.32 serving for Intel Arc Pro B70, Qwen3.8-Flash-Next on 4 and 2 cards, with INT8 execution and an INT8 KV cache.

## What's new

1. **INT8 execution** (D8, S8): dense linears, the lm_head, the routed experts (W4A8) and the drafter run int8.
2. **INT8 KV cache** (`int8_per_token_head`) on FlashAttention and on the sparse-attention kernels.
3. **PLE gather in the graph:** the INT8 PLE table is gathered on the device, inside the captured graph.
4. **Kernel cuts and a fast sampler:** fewer kernels per MoE layer, in-place all-reduce, top-k-first sampling and verification.
5. **Whole-wheel kernels:** vllm-xpu-kernels 0.1.16.2 plus 15 commits, built and installed as one wheel (branch `b70/v0.1.16-b70.6`).
6. **Six presets:** D8-6, D16e-6, S8-6 and S16e-6 on 4 cards, D8-6 and S8-6 on 2 cards; D is dense and S sparse attention, 8 is INT8 and 16e is BF16 execution with 8-bit KV.
7. **A new engine series,** `v0.32/engine/patches/vllm-v6/` (51 commits), fork tag `v0.32.0-b70.6`.

| preset | cards | decode, 1 stream: prose / code | prefill 32K / 128K (tok/s) |
|---|---|---|---|
| D8-6 | 4 | 124 / 303 | 6,758 / 4,972 |
| D16e-6 | 4 | 101 / 255 | 6,344 / 4,716 |
| S8-6 | 4 | 113 / 288 | 5,944 / 5,586 |
| S16e-6 | 4 | 100 / 236 | 5,280 / 5,061 |
| D8-6 | 2 | 72 / 107 | 2,865 / 2,412 |
| S8-6 | 2 | 68 / 92 | 2,820 / 2,893 |

The final PLE and fidelity validation ran on the TP4 dense engines (D8-6, D16e-6). The TP2 and sparse variants carry the same levers; they had a short sanity pass.

The 1-card Qwen3.8-27B and Qwen3.6-35B-A3B presets are booted on this release, numbers on their pages. Their 2-card presets are not re-measured on these kernels.

Engine [`v0.32.0-b70.6`](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.6), kernels [`b70/v0.1.16-b70.6`](https://github.com/Lumnus/vllm-xpu-kernels/tree/b70/v0.1.16-b70.6). [Changelog](CHANGELOG.md), [provenance](PROVENANCE.md), [switches](docs/switches.md).
