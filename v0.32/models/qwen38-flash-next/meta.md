# Qwen3.8-Flash-Next

`Qwen4Exp`: a hybrid GDN/attention MoE with a QSA indexer and a PLE n-gram table. Served here with DFlash2 speculative decoding
(7 draft tokens) and an 8-bit KV cache.

## Presets

| preset | cards · slots | attention | execution | drafter | PLE table | KV tier | files |
|---|---|---|---|---|---|---|---|
| D8-6 | 4 · 12 | dense FlashAttention | INT8 | W8A8 | host RAM | 128 GiB | [`.args`](preset-d8-6-4card.args) · [`.env`](preset-d8-6-4card.env) |
| D8-6, 16 slots | 4 · 16 | dense FlashAttention | INT8 | W8A8 | host RAM | 128 GiB | [`.args`](preset-d8-6s16-4card.args) · [`.env`](preset-d8-6s16-4card.env) |
| D16e-6 | 4 · 16 | dense FlashAttention | BF16 | BF16 | host RAM | 128 GiB | [`.args`](preset-d16e-6-4card.args) · [`.env`](preset-d16e-6-4card.env) |
| S8-6 | 4 · 12 | sparse QSA | INT8 | W8A8 | host RAM | 128 GiB | [`.args`](preset-s8-6-4card.args) · [`.env`](preset-s8-6-4card.env) |
| S16e-6 | 4 · 16 | sparse QSA | BF16 | BF16 | host RAM | 128 GiB | [`.args`](preset-s16e-6-4card.args) · [`.env`](preset-s16e-6-4card.env) |
| D8-6 | 2 · 4 | dense FlashAttention | INT8 | W8A8 | NVMe | 32 GiB | [`.args`](preset-d8-6-2card.args) · [`.env`](preset-d8-6-2card.env) |
| S8-6 | 2 · 4 | sparse QSA | INT8 | W8A8 | NVMe | 32 GiB | [`.args`](preset-s8-6-2card.args) · [`.env`](preset-s8-6-2card.env) |

The 2-card presets run hot/cold expert offload: 145 of 256 experts per layer per rank live in host RAM.

## Composing other modes

Each preset is a list of switches on one tree. D, S, 8, 16 and the card count are independent:

1. **Dense to sparse:** `SERVE_CONFIG=serve-config-qsa.json`, drop `--attention-backend FLASH_ATTN` and `B70_PTH_FA_*`, add the `B70_QSA_*` block.
2. **INT8 to BF16 execution:** `B70_DENSE_INT8=0`, `B70_HEAD_INT8=0`, `B70_XPU_INT8_ONEDNN=0`, `B70_MOE_W4A8=0`, drop `B70_DFLASH_KV_INT8` and take the BF16 drafter.
3. **BF16 KV:** `--kv-cache-dtype auto`, drop `B70_PTH_FA_*`.
4. **16 slots:** `--max-num-seqs 16`, capture sizes every multiple of 8 through 128, then 256 / 512 / 1024.
5. **4 cards to 2:** the hot/cold block, `B70_PLE_INT8_NVME*` in place of `B70_PLE_INT8`, TP 2, slots and capture list.

[Switches](../../../docs/switches.md) lists every key.

## Numbers

Measured on our host (AMD EPYC 7402, 256 GB, 4× B70). Decode is total tok/s, thinking off, `ignore_eos`, 512 tokens, best of two runs: prose at temperature 0.7, code at 0. Needle: thinking off, temperature 0. Quality: thinking on (medium), temperature 0.
Prefill is uncached, mean of two.

| | D8-6, 4 cards | D16e-6, 4 cards | S8-6, 4 cards | S16e-6, 4 cards | D8-6, 2 cards | S8-6, 2 cards |
|---|---|---|---|---|---|---|
| prose 1 / 4 / 8 / 12 (/ 16) streams | 124 / 333 / 522 / 671 | 109 / 309 / 485 / 640 / 699 | 113 / 313 / 503 / 653 | 102 / 275 / 472 / 606 / 801 | 72 / 132 / — / — | 68 / 128 / — / — |
| code 1 / 4 / 8 / 12 (/ 16) streams | 303 / 805 / 1,311 / 1,756 | 258 / 700 / 1,158 / 1,552 / 1,662 | 288 / 766 / 1,221 / 1,654 | 246 / 691 / 1,111 / 1,509 / 1,856 | 107 / 205 / — / — | 92 / 195 / — / — |
| prefill 32K / 128K | 6,758 / 4,972 | 6,309 / 4,732 | 5,944 / 5,586 | 5,308 / 5,057 | 2,865 / 2,412 | 2,820 / 2,893 |
| GPU KV tokens | 672,772 | 573,223 | 616,088 | 535,094 | 426,060 | 504,608 |
| burst · burst with prefix hits · structured output · token compare | 0/186 · 0/267 · pass · pass | 0/186 · 0/267 · pass · pass | — | — | 0/186 · 0/267 · pass · pass | 0/186 · 0/267 · pass · pass |
| needle retrieval | 128K 3/3, 240K 3/3 | 128K 3/3, 240K 3/3 | 128K 3/3 | 128K 3/3 | 128K 3/3 | 128K 3/3 |
| MMLU-300 / TruthfulQA-200, thinking on | 91.7 / 89.0 | 92.7 / 89.5 | — | — | — | — |

1. **Testing depth.** D8-6 and D16e-6 on 4 cards carry the full validation (gates, fidelity, quality, agentic); on D16e-6 that ran at 12 slots, and its speed row is read at 16. The 2-card presets had a short pass on this tree (gates, a short speed read, a needle); the 4-card sparse presets had a boot, a speed read and a needle. Other combinations carry the same switches and were not run.
2. **Fidelity.** Against D16e, INT8 execution moves token agreement (top-1 against a BF16 reference 86–88 % vs about 94 %). MMLU and TruthfulQA do not separate them (McNemar p 0.45 and 1.0), nor does the agentic test (36/50 vs 38/50, Fisher p 0.82). 8-bit KV against BF16 KV is at the repeat floor past 8K tokens, and visible below that, mostly on prose.
3. **2-card pairs.** Both 2-card columns are cards 0+3. The card pair matters on our host: S8-6 on cards 1+2 prefills 2,404 / 2,505 and decodes 67 / 92 (1 stream), 136 / 181 (4). Its gates and needle ran on 1+2, with D8-6 up and idle on 0+3.
4. **16 slots on D8-6:** 2K prompt, 512 tokens, total tok/s, two runs: 16 clients 848 / 851 at 16 slots against 698 / 687 at 12 with four queued; 12 clients 783 / 825 against 790 / 787. GPU KV 645,001 against 672,772.
5. Sparse against dense with earlier builds: [measurements](../../../docs/measurements/v0.32-qsa.md). Older presets and numbers: [earlier builds](../../../docs/measurements/flash-next-earlier-builds.md).

## Weights

1. **Intel AutoRound** `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` without the 102.4 GB PLE shard (78.8 GB to download): `tools/intel_snapshot.py snapshot`, into `/models/qwen3.8-flash-next/snapshot`.
2. **The INT8 PLE table:** devan-carlin's `ple_table_qwen4exp.pt` (`devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d`) through `tools/build_int8_ple.py build` ([docs/ple-int8.md](../../../docs/ple-int8.md)), into `/models/qwen3.8-flash-next/int8-ple/`; the BF16 table itself stays at `/models/qwen3.8-flash-next/W4A16/` for the boot cross-check.
3. **The drafter:** `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744` into `/models/qwen3.8-flash-next/DFlash2`.
4. **The W8A8 drafter** (INT8 presets): `python3 v0.32/models/qwen38-27b/quantize_dflash2_w8a8.py /models/qwen3.8-flash-next/DFlash2 /models/qwen3.8-flash-next/DFlash2-w8a8`. Ours has `model.safetensors` sha256 `6c3a1c2356bb350e6abbb402d0aaa9bf0e59ac4e7d858384acde74fb6a1df5df`.
5. **2-card presets:** the hot/cold rankings seed `configs/static_hot_cache_rankings.json` from [DominikBucko/qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090) @ `4dca58f` (Apache-2.0; sha256 starts `1230c61a`) into `/models/qwen3.8-flash-next/`.

Checkpoint details: [docs/weights.md](../../../docs/weights.md). Chat template: [`chat_template.jinja`](chat_template.jinja), the checkpoint's own plus the thinking switch.
