# Qwen3.8-Flash-Next

`Qwen4Exp`: a hybrid GDN/attention MoE with a QSA indexer and a PLE n-gram table. Served here with DFlash2 speculative decoding
(7 draft tokens) and an 8-bit KV cache.

## Presets

| preset | cards · slots | attention | execution | KV | drafter | PLE table | KV tier | files |
|---|---|---|---|---|---|---|---|---|
| D8-7 | 4 · 16 | dense FlashAttention | INT8 | INT8 | W8A8 | host RAM | 128 GiB | [`.args`](preset-d8-7-4card.args) · [`.env`](preset-d8-7-4card.env) |
| D16e-7 | 4 · 16 | dense FlashAttention | BF16 | INT8 | BF16 | host RAM | 128 GiB | [`.args`](preset-d16e-7-4card.args) · [`.env`](preset-d16e-7-4card.env) |
| D16-7, the dense reference | 4 · 16 | dense FlashAttention | BF16 | BF16 | BF16 | host RAM | 128 GiB | [`.args`](preset-d16-7-4card.args) · [`.env`](preset-d16-7-4card.env) |
| S8-7 | 4 · 16 | sparse QSA | INT8 | INT8 | W8A8 | host RAM | 128 GiB | [`.args`](preset-s8-7-4card.args) · [`.env`](preset-s8-7-4card.env) |
| S16e-7 | 4 · 16 | sparse QSA | BF16 | INT8 | BF16 | host RAM | 128 GiB | [`.args`](preset-s16e-7-4card.args) · [`.env`](preset-s16e-7-4card.env) |
| S16-7, the sparse reference | 4 · 16 | sparse QSA | BF16 | BF16 | BF16 | host RAM | 128 GiB | [`.args`](preset-s16-7-4card.args) · [`.env`](preset-s16-7-4card.env) |
| D8-7 | 2 · 4 | dense FlashAttention | INT8 | INT8 | W8A8 | NVMe | 32 GiB | [`.args`](preset-d8-7-2card.args) · [`.env`](preset-d8-7-2card.env) |
| S8-7 | 2 · 4 | sparse QSA | INT8 | INT8 | W8A8 | NVMe | 32 GiB | [`.args`](preset-s8-7-2card.args) · [`.env`](preset-s8-7-2card.env) |

The 2-card presets run hot/cold expert offload: 145 of 256 experts per layer per rank live in host RAM.

## Composing other modes

Each preset is a list of switches on one tree. D, S, 8, 16 and the card count are independent:

1. **Dense to sparse:** `SERVE_CONFIG=serve-config-qsa.json`, drop `--attention-backend FLASH_ATTN` and `B70_PTH_FA_*`, add the `B70_QSA_*` block.
2. **INT8 to BF16 execution:** `B70_DENSE_INT8=0`, `B70_HEAD_INT8=0`, `B70_XPU_INT8_ONEDNN=0`, `B70_MOE_W4A8=0`, drop `B70_DFLASH_KV_INT8` and take the BF16 drafter.
3. **BF16 KV:** `--kv-cache-dtype auto`, drop `B70_PTH_FA_*`.
4. **Slots:** `--max-num-seqs N`, capture sizes every multiple of 8 through N × 8, then 256 / 512 / 1024.
5. **4 cards to 2:** the hot/cold block, `B70_PLE_INT8_NVME*` in place of `B70_PLE_INT8`, TP 2, slots and capture list.

[Switches](../../../docs/switches.md) lists every key.

## Numbers

Measured on our host (AMD EPYC 7402, 256 GB, 4× B70), each preset as shipped, with the mask on. Total tok/s, thinking off, `ignore_eos`, 512 tokens, best of two runs. 4 cards at 16 slots, 2 cards at 4 slots on cards 0+3.

**Temperature 0.7**

| | D8-7, 4 cards | D16e-7, 4 cards | D16-7, 4 cards | S8-7, 4 cards | S16e-7, 4 cards | S16-7, 4 cards | D8-7, 2 cards | S8-7, 2 cards |
|---|---|---|---|---|---|---|---|---|
| short prompt, 1 / 4 / 8 / 12 / 16 streams | 120 / 333 / 514 / 667 / 769 | 112 / 299 / 474 / 625 / 682 | 114 / 306 / 482 / 628 / 698 | 115 / 320 / 498 / 650 / 741 | 105 / 284 / 465 / 600 / 739 | 101 / 292 / 460 / 610 / 676 | 69 / 136 / — / — / — | 69 / 133 / — / — / — |
| 2K prompt, 8 / 12 / 16 streams | 664 / 795 / 855 | 631 / 747 / 770 | 663 / 748 / 775 | 617 / 726 / 784 | 574 / 654 / 607 | 608 / 701 / 651 | — | — |
| 2K prompt, 2 / 4 streams | — | — | — | — | — | — | 172 / 170 | 167 / 181 |
| prefill while 16 streams load (2K prompt) | 6,950 | 6,409 | 6,399 | 6,186 | 2,903 | 2,830 | — | — |

**Temperature 0**

| | D8-7, 4 cards | D16e-7, 4 cards | D16-7, 4 cards | S8-7, 4 cards | S16e-7, 4 cards | S16-7, 4 cards | D8-7, 2 cards | S8-7, 2 cards |
|---|---|---|---|---|---|---|---|---|
| code, 1 / 4 / 8 / 12 / 16 streams | 302 / 811 / 1,270 / 1,698 / 1,941 | 252 / 704 / 1,131 / 1,502 / 1,661 | 254 / 710 / 1,141 / 1,532 / 1,680 | 266 / 755 / 1,208 / 1,546 / 1,817 | 243 / 668 / 1,081 / 1,442 / 1,749 | 248 / 649 / 1,092 / 1,443 / 1,638 | 100 / 230 / — / — / — | 96 / 181 / — / — / — |
| prose, 1 / 4 / 8 / 12 / 16 streams | 119 / 334 / 518 / 681 / 769 | 115 / 291 / 485 / 638 / 699 | 114 / 306 / 485 / 636 / 716 | 124 / 315 / 498 / 660 / 752 | 102 / 282 / 461 / 610 / 762 | 100 / 293 / 468 / 620 / 681 | — | — |

**Independent of temperature**

| | D8-7, 4 cards | D16e-7, 4 cards | D16-7, 4 cards | S8-7, 4 cards | S16e-7, 4 cards | S16-7, 4 cards | D8-7, 2 cards | S8-7, 2 cards |
|---|---|---|---|---|---|---|---|---|
| prefill 32K / 128K, uncached, mean of two | 6,772 / 5,004 | 6,366 / 4,726 | 6,358 / 4,718 | 6,000 / 5,583 | 5,395 / 5,050 | 5,664 / 5,051 | 2,860 / 2,418 | 2,834 / 2,892 |
| GPU KV tokens | 678,259 | 606,145 | 386,060 | 631,240 | 566,177 | 340,469 | 474,109 | 559,191 |
| burst · burst with prefix hits | 0/186 · 0/267 | 0/186 · 0/267 | 0/186 · 0/267 | 0/186 · 0/267 | 0/186 · 0/267 | 0/186 · 0/267 | 0/186 · 0/267 | 0/186 · 0/267 |
| structured output · token compare | pass · pass | pass · pass | pass · pass | pass · pass | pass · fail, see 2 | pass · pass | pass · pass | pass · pass |
| needle retrieval | 128K, 240K 6/6 | 128K, 240K 6/6 | 128K, 240K 6/6 | 128K, 240K 6/6 | 128K, 240K 6/6 | 128K, 240K 6/6 | 128K 3/3 | 128K 3/3 |

1. **Quality and fidelity,** D8 and D16e only, on the same engine code with 12 slots. MMLU-300 / TruthfulQA-200 (thinking on, medium, temperature 0): D8 91.7 / 89.0, D16e 92.7 / 89.5; McNemar p 0.45 and 1.0. Agentic test 36/50 and 38/50, Fisher p 0.82. Against a BF16 reference, INT8 execution moves token agreement (top-1 86–88 % vs about 94 %). 8-bit KV against BF16 KV is at the repeat floor past 8K tokens, and visible below that, mostly on prose. The sparse and 2-card presets have no quality run on this release.
2. **Sparse on BF16 execution at 16 streams.** S16e-7 and S16-7 halve prefill under a 16-stream 2K load, and decode drops with it; the other presets hold. The KV type does not change it. Token compare on S16e-7: 1 stream 8/8 identical; at 8 streams one prompt shows a token run of 8 or more; S16-7 passes. Neither is explained yet.
3. **Card pairs.** On our host cards 0+3 prefill about 15 % faster than 1+2.
4. Sparse against dense with earlier builds: [measurements](../../../docs/measurements/v0.32-qsa.md). Older presets and numbers: [earlier builds](../../../docs/measurements/flash-next-earlier-builds.md).

## Chinese characters on non-English prompts

Qwen3.8-Flash-Next can emit Han characters into answers to non-English prompts, on every precision we run and more on long agentic contexts than on short prompts. `B70_CJK_MASK=1`, set in every preset, removes them where the target logits are computed, so speculative verification reads the masked logits too.

1. **What it masks:** 55,659 of 248,320 token ids on this tokenizer (55,328 Han, 331 partial sequences). Kana stays. About 3,100 rare code points (Ext-A and compatibility ideographs) remain reachable through byte tokens.
2. **When it applies:** on every request, unless a system or user turn contains Han. A Chinese prompt gets a Chinese answer. `cjk_mask` in the request (top level or `vllm_xargs`) overrides per request.
3. **Measured,** 80 answers per replay of a long Bulgarian session: D16e 46 answers with Han to 0 on one session, 10 to 0 on another; D8 0 of 160 with the mask on. Throughput shows no difference beyond noise on D16e and D8. A Chinese test prompt on every preset here still answers in Chinese.
4. **What it does not do:** it cures Han only. Latin words inside non-English answers come from the model, and INT8 execution mixes in more of them than BF16. On our 48-answer short-prompt set, Latin words per 1,000 Cyrillic words read 0.59 on D8-7 and 0.88 on S8-7 against 0.29 on D16e-7, 0.44 on D16-7, 0.58 on S16e-7 and 0 on S16-7, a handful of words each. With the gate on, a legitimate Chinese gloss in a non-Chinese conversation is masked too; send `cjk_mask: false` when you want those.

## Weights

1. **Intel AutoRound** `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` without the 102.4 GB PLE shard (78.8 GB to download): `tools/intel_snapshot.py snapshot`, into `/models/qwen3.8-flash-next/snapshot`.
2. **The INT8 PLE table:** devan-carlin's `ple_table_qwen4exp.pt` (`devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d`) through `tools/build_int8_ple.py build` ([docs/ple-int8.md](../../../docs/ple-int8.md)), into `/models/qwen3.8-flash-next/int8-ple/`; the BF16 table itself stays at `/models/qwen3.8-flash-next/W4A16/` for the boot cross-check.
3. **The drafter:** `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744` into `/models/qwen3.8-flash-next/DFlash2`.
4. **The W8A8 drafter** (INT8 presets): `python3 v0.32/models/qwen38-27b/quantize_dflash2_w8a8.py /models/qwen3.8-flash-next/DFlash2 /models/qwen3.8-flash-next/DFlash2-w8a8`. Ours has `model.safetensors` sha256 `6c3a1c2356bb350e6abbb402d0aaa9bf0e59ac4e7d858384acde74fb6a1df5df`.
5. **2-card presets:** the hot/cold rankings seed `configs/static_hot_cache_rankings.json` from [DominikBucko/qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090) @ `4dca58f` (Apache-2.0; sha256 starts `1230c61a`) into `/models/qwen3.8-flash-next/`.

Checkpoint details: [docs/weights.md](../../../docs/weights.md). Chat template: [`chat_template.jinja`](chat_template.jinja), the checkpoint's own plus the thinking switch.
