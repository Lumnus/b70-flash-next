# Qwen3.8-27B on the 0.32 engine

Qwen3.8-27B is a dense hybrid model (GDN linear attention plus 16 full-attention layers), served here from an INT4
AutoRound checkpoint run as W4A8 (oneDNN int4 × int8 on XMX for GEMMs of 512 rows and more), with DFlash2 speculative
decoding (7 draft tokens) and a shortlist draft head. Two presets: one card (147,456-token context) and two cards (the full
262,144).

| | |
|---|---|
| weights | `devan-carlin/Qwen3.8-27B-int4-AutoRound` @ `5efb20f2396e8223c2832183c29db23998d7a147` (Apache-2.0; auto-round → INC, int4, group 128, symmetric); file list with object ids: [`weights.oid`](weights.oid) |
| drafter | `incoai/Qwen3.8-27B-DFlash2` @ `015e795645c74b1a0eeef3b570031fb62e769bc5` (Apache-2.0; [`drafter.oid`](drafter.oid)). The 1-card preset runs it converted to W8A8 int8 with [`quantize_dflash2_w8a8.py`](quantize_dflash2_w8a8.py) (`model.safetensors` sha256 `ff81cc3abb11647963aba4ff8a5ebfa23f806da2d1cf4c2a2464edcffdffd68c`, 1.91 instead of 3.58 GiB); the 2-card preset runs it as published (bf16), the configuration its numbers were measured on |
| draft vocab | [`draft-vocab-67263.txt`](draft-vocab-67263.txt) (sha256 `02ccd9cdd46d078210c8c09a2a4888c63eee3908eada324a1db4ef9e672aa12e`): steveseguin's token shortlist from [b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab) @ `b4f5f8c6` ([The Unlicense](draft-vocab.LICENSE)) plus every added token of the tokenizer, minus the drafter's mask token; built by [`build_draft_vocab.py`](build_draft_vocab.py) |
| presets | [`preset-1card`](preset-1card.args) / [`.env`](preset-1card.env) · [`preset-2card`](preset-2card.args) / [`.env`](preset-2card.env) |
| chat template | [`chat_template-1card.jinja`](chat_template-1card.jinja) / [`chat_template-2card.jinja`](chat_template-2card.jinja): the checkpoint's template plus one constant serving line first in the system block (model, card count, context), a thinking guidance line and an effort mapping (`none`/`off` turns thinking off; `minimal` → low; `high`/`max`/`ultra` → xhigh). `chat_template_kwargs: {"serving_context": ""}` drops the serving line. The two files differ only in that line |
| attention | target on the XPU FlashAttention backend (vLLM's XPU default), the drafter on `TRITON_ATTN`; both presets, both verified (gates below) |
| KV cache dtype | 1 card: `--kv-cache-dtype fp8` (e4m3, scale 1.0). 2 cards: no flag, so bf16 (`auto`) |
| engine switches | `B70_HEAD_FP8=1` (FP8 lm_head, shared by the drafter), `B70_DFLASH_DRAFT_VOCAB`, `VLLM_XPU_INC_WNA16_BACKEND=w4a8`; 1-card adds `--kv-cache-dtype fp8` and `B70_XPU_INT8_ONEDNN=1` (the W8A8 drafter on oneDNN int8) |
| run | `v0.32/engine/serve.sh qwen38-27b preset-1card` (or `preset-2card`) with `MODEL_PATH` at the weights, the drafter at `/models/qwen3.8-27b/DFlash2-w8a8` (1 card) or `/models/qwen3.8-27b/DFlash2` (2 cards) and the draft vocab at `/models/qwen3.8-27b/draft-vocab-67263.txt` |

## Presets

| | 1 card | 2 cards |
|---|---|---|
| measured | 2026-10-31, the b70.5 engine and kernels, card 2 of our host | 2026-10-29, the b70.5 engine without the backstep guard and kernels without #605, cards 1 and 2, `B70_OFFLOAD_GDN_BACKSTEP=1` (it acts only on CPU-tier reuse) |
| tensor parallel, slots, context | TP 1, 8, 147,456 (the start-up fit reports 149,760) | TP 2 (draft TP 2), 8, 262,144 |
| KV cache | fp8, 150,054 GPU tokens | bf16, 341,668 GPU tokens (booted with a 32 GiB tier; 335,345 with the shipped 64) |
| drafter | W8A8 int8 on oneDNN | bf16 |
| GPU memory | 0.97: the ceiling (0.98 is refused at start-up); 29.75 GiB in use after a burst and a 139K prefill, nothing spilled to host memory | 0.89 |
| KV tier (host RAM) | 32 GiB | 64 GiB |
| decode, 1 stream, prose / code (tok/s) | 56–62 / 142–149 | ~94 (91.2 and 96.5, prose) |
| decode, 4 concurrent, prose / code (total tok/s) | 175–177 / 413–419 | 258 (259.9 and 256.9, prose) |
| decode, 8 concurrent (total tok/s) | not run | 370 (376.7 and 363.0) |
| prefill, uncached (tok/s) | 1,947 at 32K; 926 at 139K | 3,504 at 32K; 2,434 at 128K |
| host RAM the engine takes | ~36 GiB | ~55–78 GiB |
| boot to ready | ~183 s | 106–254 s |
| gates | greedy probes; burst 0/124; needle at 64K and 140K × 3 depths 6/6; a 10-turn agent loop with real tools, 45 completions, no repeated finished tool calls. On the same preset with the bf16 drafter: burst 0/186, burst with prefix-cache hits 0/267, structured output with speculative decoding 36 + 18 plain, token compare at concurrency 1 and 8 pass, needle 32K/64K/90K 9/9, two 10-turn agent loops with 91 completions and prompts to 52K tokens, 0 length and 0 repetition stops | burst 0/90; structured output 54/54; teacher-forced against a bf16-head reference: KL mean 0.022, top-1 95.4 % |
| quality, thinking on (medium) | not measured on 1 card (fp8 KV) | MMLU 300: 91.0 · TruthfulQA MC1 200: 83.5 · agent capability gate: 40/50 · needle to 240K 9/9 · 64K document 5/5 |

How to read it:

1. **One card is not as fast as two.** Every token reads every weight byte from one card: decode is about two thirds of
   two cards and prefill a bit over half. Code-shaped output decodes faster than prose because the drafter is accepted more
   often there.
2. **Why fp8 KV on one card.** At bf16 KV one card holds 30,784 tokens at 0.92 (about 136 KiB per token: 16
   full-attention layers, the drafter's 5 sliding-window layers, page padding, GDN states). fp8 roughly halves the attention
   part: 74,880 at 0.935, 100,630 at 0.97, and 149,760 with the W8A8 drafter.
3. **The W8A8 drafter** frees about 1.7 GiB of weights for KV, about +49K fp8 tokens at the same GPU memory. Against the bf16
   drafter on the same card and day: decode 58.9 / 145.8 against 55.6 / 134.4 tok/s at one stream (prose / code), 175.9 /
   416.0 against 162.5 / 391.2 at 4 concurrent, prefill 32K level (1,947 / 1,946); accepted tokens per draft 0.686 / 0.207
   against 0.713 / 0.214 (−3–4 %). A CPU paired test of the conversion (286 steps against a bf16 target) found the accepted
   length unchanged within noise (3.524 against 3.573, sign test p = 1).
4. **Weights choice.** Three INT4/INT8 builds were measured on two cards with the same engine switches; quality did not
   separate them (McNemar, all n.s.), so speed decided:

   | build, 2 cards | MMLU / TruthfulQA | decode 1 / 4 / 8 (tok/s) | prefill 32K / 128K |
   |---|---|---|---|
   | **devan-carlin AutoRound (shipped)** | 91.0 / 83.5 | ~94 / 258 / 370 | 3,504 / 2,434 |
   | `RedHatAI/Qwen3.8-27B-INT4` @ `91bd022` | 89.3 / 84.5 | ~90 / 250 / 362 | 2,311 / 1,453 |
   | `RukaRat/Qwen3.8-27B-INT8-W8A8-imatrix-MTP` @ `23f7b028` (196,608 context: one 262K request does not fit) | 89.0 / 81.5 | ~69 / 208 / 325 | 4,061 / 2,684 |

5. **The b70.5 build and the 2-card numbers.** The dense 27B does not call the grouped GEMM that #605 changes, and the
   backstep acts only on CPU-tier reuse, so neither b70.5 change touches the path the 2-card numbers measure.
