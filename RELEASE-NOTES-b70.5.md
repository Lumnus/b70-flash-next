**0.32.0-b70.5**: one engine, three models. Qwen3.8-Flash-Next, Qwen3.8-27B and Qwen3.6-35B-A3B on Intel Arc Pro B70,
vLLM v0.32.0, one preset per model and card count.

**Fix, please apply on b70.4: `B70_OFFLOAD_GDN_BACKSTEP=0`.** b70.4's presets shipped it at 1. With the CPU KV tier and
prefix caching, the backstep makes a prefix-cache hit restore a GDN state built through a different prefill chunk split than
the cold path, and the output diverges from the first token; in an agent session the model re-does tool calls it has already
finished. Every b70.5 preset ships 0, and the engine forces it off under `mamba_cache_mode=align`.

**Models** (each row on its model page, with the build and date it was measured on):

| model | preset | context | decode, 1 stream (tok/s) | prefill 32K (tok/s) | MMLU / TruthfulQA, thinking on |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next (Intel AutoRound) | 4 cards | 262,144 | 156.4 | — | — |
| | 2 cards | 262,144 | 57.1 | 2,768 | 92.0 / 89.0 |
| Qwen3.8-27B (INT4 AutoRound, W4A8) | 2 cards | 262,144 | ~94 prose | 3,504 | 91.0 / 83.5 |
| | 1 card | 147,456 | 56–62 prose, 142–149 code | 1,947 | — |
| Qwen3.6-35B-A3B (Intel INT4 mixed AutoRound) | 2 cards | 262,144 | 152–168 prose, 376 code | 9,997 | 88.7 / 88.5 |
| | 1 card | 196,608 | 118–120 prose, 310–314 code | 7,263 | — |

On one 4-card host the three run side by side (Flash-Next on 2 cards, the 35B and the 27B on one card each), with 41–43 GiB
of host RAM left of 251.5.

**What is new**

1. **Layout:** `v0.32/engine/` (the series, the checks, `serve.sh MODEL PRESET`) and `v0.32/models/<model>/` (presets,
   `meta.md`, chat templates, weight file lists). All moves are `git mv`.
2. **Engine:** fork tag `v0.32.0-b70.5` = b70.4 + 8 commits (`v0.32/engine/patches/vllm-models/`): FP8 lm_head for
   Qwen3.5/3.8, the Triton W8A8 INT8 GEMM tiles (vllm-project/vllm#64190, #64244), the asynchronous hot/cold swap, a DFlash2
   shortlist draft head, native int8 activation quantisation with an oneDNN W8A8 GEMM, and the backstep guard. All behind
   switches.
3. **Kernels:** vllm-xpu-kernels 0.1.16.2 + five patches: #660, #662, the native int8 ops (two patches) and
   vllm-project/vllm-xpu-kernels#605.
4. **Flash-Next 2-card** turns on the asynchronous hot/cold swap: faster after a change of traffic (38.2 against 33.0 tok/s at
   one stream after an agent-session replay) and no decode stall per swap tick.
5. **Qwen3.8-27B on one card** with fp8 KV and an INT8 (W8A8) drafter made by `quantize_dflash2_w8a8.py`.

**Not measured on this release:** the Flash-Next 4-card preset on the b70.5 kernels (only the backstep key changed), the 27B
2-card preset on the b70.5 kernels, quality on the 1-card presets, any multi-day run.

Fork: [Lumnus/vllm `v0.32.0-b70.5`](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.5). Full list: [CHANGELOG](CHANGELOG.md).
Provenance and hashes: [PROVENANCE.md](PROVENANCE.md) §9.
