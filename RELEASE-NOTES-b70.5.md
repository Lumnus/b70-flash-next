**0.32.0-b70.5**: sparse attention for Qwen3.8-Flash-Next on Intel Arc Pro B70, and one engine for three models.

**Sparse attention (QSA) on B70.** Qwen3.8-Flash-Next runs its own sparse attention end to end in vLLM on Arc Pro B70: the
QSA indexer picks a 2,048-token selection for each full-attention layer, and Triton kernels on XPU attend over it. It is
the default of both Flash-Next presets, on 4 cards and on 2. Against dense attention on the same engine: prefill 1.31× at
260K tokens (4,480 against 3,420 tok/s), an agent-session replay in 204.7 s against 230–235 s, long-context tasks at 128K
80/78 % against 72/72 %. Dense attention on FlashAttention stays a parameter; both are verified.

**One engine, three models.** The same engine and Python environment serve Qwen3.8-Flash-Next, Qwen3.8-27B and
Qwen3.6-35B-A3B, one preset per model and card count. On one 4-card host the three run side by side (Flash-Next on 2 cards,
the 35B and the 27B on one card each), with 41–43 GiB of host RAM left of 251.5.

**Fix, please apply on b70.4: `B70_OFFLOAD_GDN_BACKSTEP=0`.** b70.4's presets shipped it at 1. With the CPU KV tier and
prefix caching, the backstep makes a prefix-cache hit restore a GDN state built through a different prefill chunk split than
the cold path, and the output diverges from the first token; in an agent session the model re-does tool calls it has already
finished. Every b70.5 preset ships 0, and the engine forces it off under `mamba_cache_mode=align`.

**Models** (each row on its model page, with the build and date it was measured on; quality thinking on unless marked):

| model | preset | attention · KV | decode, 1 stream (tok/s) | peak decode, total (tok/s, streams) | best prefill (tok/s) | MMLU / TruthfulQA |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next (Intel AutoRound) | 4 cards | sparse · bf16 | 156.4 | 580.9 (8) | 5,382 at 32K | 92.0 / 89.0 on 2 cards |
| | 4 cards | dense · bf16 | 169.3 | 605.5 (8) | 6,305 at 8K | 86.7 / 89.0, thinking off |
| | 2 cards | sparse · bf16 | 57.1 | 119.7 (8) | 2,768 at 32K | 92.0 / 89.0 |
| Qwen3.8-27B (INT4 AutoRound, W4A8) | 2 cards | FlashAttention · bf16 | ~94 prose | 370 (8) | 3,504 at 32K | 91.0 / 83.5 |
| | 1 card | FlashAttention · fp8 | 56–62 prose, 142–149 code | 413–419 code (4) | 1,947 at 32K | — |
| Qwen3.6-35B-A3B (Intel INT4 mixed AutoRound) | 2 cards | FlashAttention · bf16 | 152–168 prose, 376 code | 808 prose (16) | 9,997 at 32K | 88.7 / 88.5 |
| | 1 card | FlashAttention · bf16 | 118–120 prose, 310–314 code | 756–798 code (4) | 8,386 at 8K | — |

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
5. **Flash-Next chat template** in the kit (`v0.32/models/qwen38-flash-next/chat_template.jinja`, wired into both presets):
   a constant serving line, the thinking guidance and the effort mapping, with "off wins" (`enable_thinking=false` or a
   none/off effort turns thinking off).
6. **Qwen3.8-27B on one card** with fp8 KV and an INT8 (W8A8) drafter made by `quantize_dflash2_w8a8.py`.
7. **Lessons learned** (`docs/lessons-learned.md`): what works on this stack, what does not, and how to tune and test it.

Fork: [Lumnus/vllm `v0.32.0-b70.5`](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.5). Full list: [CHANGELOG](CHANGELOG.md).
Provenance and hashes: [PROVENANCE.md](PROVENANCE.md) §9.
