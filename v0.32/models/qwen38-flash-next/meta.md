# Qwen3.8-Flash-Next

`Qwen4Exp`: a hybrid GDN/attention MoE with a QSA indexer and a PLE n-gram table. Served here with DFlash2 speculative decoding
(7 draft tokens) and an 8-bit KV cache.

## Presets

| preset | cards · slots | attention | execution | drafter | PLE table | KV tier | files |
|---|---|---|---|---|---|---|---|
| D8-6 | 4 · 16 | dense FlashAttention | INT8 | W8A8 | host RAM | 128 GiB | [`.args`](preset-d8-6-4card.args) · [`.env`](preset-d8-6-4card.env) |
| D16e-6 | 4 · 16 | dense FlashAttention | BF16 | BF16 | host RAM | 128 GiB | [`.args`](preset-d16e-6-4card.args) · [`.env`](preset-d16e-6-4card.env) |
| S8-6 | 4 · 16 | sparse QSA | INT8 | W8A8 | host RAM | 128 GiB | [`.args`](preset-s8-6-4card.args) · [`.env`](preset-s8-6-4card.env) |
| S16e-6 | 4 · 16 | sparse QSA | BF16 | BF16 | host RAM | 128 GiB | [`.args`](preset-s16e-6-4card.args) · [`.env`](preset-s16e-6-4card.env) |
| D8-6 | 2 · 4 | dense FlashAttention | INT8 | W8A8 | NVMe | 32 GiB | [`.args`](preset-d8-6-2card.args) · [`.env`](preset-d8-6-2card.env) |
| S8-6 | 2 · 4 | sparse QSA | INT8 | W8A8 | NVMe | 32 GiB | [`.args`](preset-s8-6-2card.args) · [`.env`](preset-s8-6-2card.env) |

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

Measured on our host (AMD EPYC 7402, 256 GB, 4× B70). Total tok/s, thinking off, `ignore_eos`, 512 tokens, best of two runs. 4 cards at 16 slots, 2 cards at 4 slots on cards 0+3.

**Temperature 0.7**

| | D8-6, 4 cards | D16e-6, 4 cards | S8-6, 4 cards | S16e-6, 4 cards | D8-6, 2 cards | S8-6, 2 cards |
|---|---|---|---|---|---|---|
| short prompt, 1 / 4 / 8 / 12 / 16 streams | 124 / 330 / 504 / 678 / 777 | 109 / 309 / 485 / 640 / 699 | 117 / 309 / 487 / 644 / 738 | 102 / 275 / 472 / 606 / 801 | 72 / 132 / — / — / — | 68 / 128 / — / — / — |
| 2K prompt, 8 / 12 / 16 streams | 681 / 778 / 880 | 630 / 742 / 685 | 629 / 757 / 715 | 578 / 678 / 607 | — | — |
| prefill while 16 streams load (2K prompt) | 6,868 | 3,207 | 3,356 | 2,892 | — | — |

**Temperature 0**

| | D8-6, 4 cards | D16e-6, 4 cards | S8-6, 4 cards | S16e-6, 4 cards | D8-6, 2 cards | S8-6, 2 cards |
|---|---|---|---|---|---|---|
| code, 1 / 4 / 8 / 12 / 16 streams | 296 / 830 / 1,316 / 1,729 / 1,830 | 258 / 700 / 1,158 / 1,552 / 1,662 | 271 / 757 / 1,198 / 1,631 / 1,880 | 246 / 691 / 1,111 / 1,509 / 1,856 | 107 / 205 / — / — / — | 92 / 195 / — / — / — |
| prose, 1 / 4 / 8 / 12 / 16 streams | 118 / 351 / 533 / 714 / 825 | — | 116 / 323 / 519 / 689 / 785 | — | — | — |

**Independent of temperature**

| | D8-6, 4 cards | D16e-6, 4 cards | S8-6, 4 cards | S16e-6, 4 cards | D8-6, 2 cards | S8-6, 2 cards |
|---|---|---|---|---|---|---|
| prefill 32K / 128K, uncached, mean of two | 6,671 / 5,016 | 6,309 / 4,732 | 5,947 / 5,586 | 5,308 / 5,057 | 2,865 / 2,412 | 2,820 / 2,893 |
| GPU KV tokens | 645,113 | 573,223 | 600,045 | 535,094 | 426,060 | 504,608 |
| needle retrieval, 128K | 3/3 at 12 slots | 3/3 | 3/3 at 12 slots | 3/3 | 3/3 | 3/3, on cards 1+2 |

1. **Validation at 12 slots.** D8-6 and D16e-6 had the full validation on this tree at 12 slots: burst 0/186, burst with prefix hits 0/267, structured output, token compare, needle to 240K, fidelity, quality. MMLU-300 / TruthfulQA-200 (thinking on, medium, temperature 0): D8-6 91.7 / 89.0, D16e-6 92.7 / 89.5; McNemar p 0.45 and 1.0. Agentic test 36/50 and 38/50, Fisher p 0.82.
2. **Short passes.** The 2-card presets: the same gates on this tree, on cards 1+2 for S8-6, a speed read and a needle. The 4-card sparse presets: a boot, a speed read and a needle. Other combinations carry the same switches and were not run.
3. **Chinese tokens on non-English prompts.** Replay of a Bulgarian-language session, thinking on, temperature 0.7, 80 samples, answers with Chinese characters: D8-6 17, S8-6 14, D16e-6 9, BF16 KV (D16-6) 15, D8-6 without a drafter 12. The gaps are small for this sample. The model's own answers in the original session already carried a few.
4. **Fidelity.** Against D16e, INT8 execution moves token agreement (top-1 against a BF16 reference 86–88 % vs about 94 %). 8-bit KV against BF16 KV is at the repeat floor past 8K tokens, and visible below that, mostly on prose.
5. **Card pairs.** On our host cards 0+3 prefill about 15 % faster than 1+2: S8-6 on cards 1+2 reads 2,404 / 2,505.
6. Sparse against dense with earlier builds: [measurements](../../../docs/measurements/v0.32-qsa.md). Older presets and numbers: [earlier builds](../../../docs/measurements/flash-next-earlier-builds.md).

## Weights

1. **Intel AutoRound** `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` without the 102.4 GB PLE shard (78.8 GB to download): `tools/intel_snapshot.py snapshot`, into `/models/qwen3.8-flash-next/snapshot`.
2. **The INT8 PLE table:** devan-carlin's `ple_table_qwen4exp.pt` (`devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d`) through `tools/build_int8_ple.py build` ([docs/ple-int8.md](../../../docs/ple-int8.md)), into `/models/qwen3.8-flash-next/int8-ple/`; the BF16 table itself stays at `/models/qwen3.8-flash-next/W4A16/` for the boot cross-check.
3. **The drafter:** `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744` into `/models/qwen3.8-flash-next/DFlash2`.
4. **The W8A8 drafter** (INT8 presets): `python3 v0.32/models/qwen38-27b/quantize_dflash2_w8a8.py /models/qwen3.8-flash-next/DFlash2 /models/qwen3.8-flash-next/DFlash2-w8a8`. Ours has `model.safetensors` sha256 `6c3a1c2356bb350e6abbb402d0aaa9bf0e59ac4e7d858384acde74fb6a1df5df`.
5. **2-card presets:** the hot/cold rankings seed `configs/static_hot_cache_rankings.json` from [DominikBucko/qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090) @ `4dca58f` (Apache-2.0; sha256 starts `1230c61a`) into `/models/qwen3.8-flash-next/`.

Checkpoint details: [docs/weights.md](../../../docs/weights.md). Chat template: [`chat_template.jinja`](chat_template.jinja), the checkpoint's own plus the thinking switch.
