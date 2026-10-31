# Qwen3.6-35B-A3B on the 0.32 engine

Qwen3.6-35B-A3B is a hybrid MoE (30 GDN and 10 full-attention layers, 256 experts, top-8, about 3B active), served here
from Intel's mixed INT4 AutoRound checkpoint: the linears of attention, GDN and the shared expert run as W4A8 through
oneDNN, the routed experts as int4 W4A16 through the XPU grouped GEMM. DFlash speculative decoding with 7 draft tokens. Two
presets: one card (196,608-token context) and two cards (the full 262,144, 16 slots).

| | |
|---|---|
| weights | `Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound` @ `65f69c73f17488236c85c85211f6ba28d7106157`; file list with object ids: [`weights.oid`](weights.oid) |
| drafter | `z-lab/Qwen3.6-35B-A3B-DFlash` @ `f181eece646affea2c38b2765f1aaa01a9734ccd` (Apache-2.0; [`drafter.oid`](drafter.oid)), drafter attention on the XPU FlashAttention backend |
| presets | [`preset-1card`](preset-1card.args) / [`.env`](preset-1card.env) · [`preset-2card`](preset-2card.args) / [`.env`](preset-2card.env) |
| chat template | [`chat_template-1card.jinja`](chat_template-1card.jinja) / [`chat_template-2card.jinja`](chat_template-2card.jinja): the checkpoint's template plus one constant serving line first in the system block (model, card count, context) and "off wins" (`reasoning_effort` none/off/false/0/disabled turns thinking off). `chat_template_kwargs: {"serving_context": ""}` drops the serving line. Per-effort thinking budgets are server-side (`B70_THINKING_BUDGET`) |
| sampling defaults | the model card's thinking-mode preset for precise coding: temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence 0. The card's general preset (temperature 1.0, presence 1.5) scored 34 against 41 of 50 on our agent capability gate |
| kernels | requires vllm-xpu-kernels #605 (kernels patch 0005): the routed-expert grouped GEMM of this model has shapes whose scale pitch is under 64 bytes |
| engine switches | `B70_HEAD_FP8=1` (FP8 lm_head, shared by the drafter), `VLLM_XPU_INC_WNA16_BACKEND=w4a8`, the repetition stop widened to paragraph loops (`max=512,min=32,count=3`) |
| run | `v0.32/engine/serve.sh qwen36-35b-a3b preset-1card` (or `preset-2card`) with `MODEL_PATH` at the weights and the drafter at `/models/qwen3.6-35b-a3b/DFlash` |

## Presets

| | 1 card | 2 cards |
|---|---|---|
| measured | 2026-10-30/31, the b70.5 engine and kernels, card 1 of our host | 2026-10-29 on cards 1 and 2: speed on the b70.5 kernels and the engine before the backstep guard; gates again on the b70.5 engine |
| tensor parallel, slots, context | TP 1, 8, 196,608 (the start-up fit reports 207,360) | TP 2 (draft TP 2), 16, 262,144 |
| KV cache | bf16, 204,553 GPU tokens | bf16, 1,060,309 GPU tokens (four full 262K requests) |
| VRAM | weights 19.62 GiB, peak activation 1.64, KV 6.22; 2.34 GiB free of 31.89 under load (GPU memory 0.92) | 9.91 GiB of weights per card; ~28.4 GiB per card in use (GPU memory 0.89) |
| KV tier (host RAM) | 32 GiB | 64 GiB |
| decode, 1 stream, prose / code (tok/s) | 118–120 / 310–314 | 152–168 / 376 (math with thinking 400) |
| decode, concurrent (total tok/s) | 4: 308–316 prose, 756–798 code | 4 / 8 / 16: 418 / 583 / 808 (prose) |
| prefill, uncached (tok/s) | 8,386 at 8K (time to first token 0.97 s); 7,263 at 32K (4.5 s) | 9,997 at 32K (3.3 s); 6,355 at 128K (20.6 s) |
| host RAM the engine takes | ~42 GiB | ~72 GiB |
| boot to ready | 128–151 s | 272 s cold, 151 s with warm caches |
| gates | smoke (4-way 263.6 tok/s); burst 0/186; burst with prefix-cache hits 0/267; structured output with speculative decoding 36 + 18 plain; token compare at concurrency 1: 7 of 7 identical; at 8: near-ties and one 1.375-nat split (teacher-forced 1 against 8: KL mean 0.0062, top-1 98.4 %); warm against cold prefix cache, 1,600 generated tokens: 0 of 160 differ; two 10-turn agent loops with real tools: 96 completions, prompts to 63.8K tokens, 0 length, 0 repetition, no repeated finished tool calls | burst 0/186; burst with prefix-cache hits 0/267; structured output 54/54; warm against cold prefix cache 0/160; agent loops: 190 completions to 67.6K tokens with no repeated finished tool calls; on the b70.5 engine again: burst 0/186, 0/267, structured output 36 + 18, warm against cold 0/160, 59 completions to 63K |
| quality, thinking on (medium) | not re-run on 1 card (same weights, kernels, KV type and sampling) | MMLU 300: 88.7 · TruthfulQA MC1 200: 88.5 · agent capability gate 35/50 · needle to 128K 6/6 · 64K document 5/5 |

How to read it:

1. **Drafter acceptance depends on the text.** On free prose the drafter is accepted for about 2 of 7 tokens, on code and
   math for 5 to 6; the code column is what an agent's tool calls look like.
2. **One card against two**: one-stream prose 0.77×, prefill at 32K 0.73×; one card holds 196,608 tokens of context with 8
   slots.
3. **Against the official FP8 build** (`Qwen/Qwen3.6-35B-A3B-FP8` @ `95a723d`, two cards, 8 slots): quality not different
   (MMLU 89.3 / TruthfulQA 87.5, McNemar n.s.), slower (120 tok/s at one stream), 581,101 KV tokens. The FP8 weights (35 GB)
   do not fit one card.
4. **Against the other two models on the same quality items** (two cards each): MMLU lower than Qwen3.8-Flash-Next 2-card
   (88.7 against 92.7, McNemar p 0.006); not different from Qwen3.8-27B (p 0.15). TruthfulQA: not different from either.

## Known issues (Qwen3.6-35B-A3B)

1. **The agent capability gate dropped from 41 to 35 of 50** after the sampling change to the coding preset (Fisher p ≈ 0.23,
   not significant); MMLU and TruthfulQA did not move.
2. **One stop of the 1-card engine hung more than 30 s** before it was killed; the boot after it was clean and passed the
   gates. Not reproduced, cause unknown.
3. **About 10 % of quality items hit the 2,400-token thinking cap** at medium effort (30 of 300); the same instrument was used
   for every row.
