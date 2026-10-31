# Qwen3.8-Flash-Next on the 0.32 engine

The model the engine was built for: Qwen3.8-Flash-Next (`Qwen4Exp`, a hybrid GDN/attention MoE with a QSA indexer and a
PLE n-gram table), served with sparse attention by default, DFlash2 speculative decoding (7 draft tokens), the INT8 PLE
table on NVMe and a host-RAM KV tier. Two presets: 4 cards for speed, 2 cards so the other pair stays free.

| | |
|---|---|
| weights | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` (recommended, the only build measured on 0.32), without its 102.4 GB BF16 PLE shard: snapshot from `tools/intel_snapshot.py`; INT8 PLE table from `tools/build_int8_ple.py` (built from devan-carlin's BF16 `ple_table_qwen4exp.pt`). Other builds: [Models](#other-weights) |
| drafter | `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744` (block diffusion, 0.6B) |
| presets | [`preset-4card`](preset-4card.args) / [`.env`](preset-4card.env) · [`preset-2card`](preset-2card.args) / [`.env`](preset-2card.env) |
| serve dir | the snapshot's files symlinked, with [`serve-config-qsa.json`](serve-config-qsa.json) as `config.json` (sparse attention, both presets) or [`v0.30/engines/serve-config-intel-autoround.json`](../../../v0.30/engines/serve-config-intel-autoround.json) (dense) |
| chat template | [`chat_template.jinja`](chat_template.jinja), both presets: generic template; behaviour and attribution belong to the client or gateway layer. It is the checkpoint's own template plus the thinking switch: off wins (`enable_thinking=false` or a `reasoning_effort` of none/off/false/0/disabled turns thinking off, so the prompt ends with an empty think block), and effort names outside xhigh/medium/low map instead of raising (`minimal` → low; `high`/`max`/`ultra` → xhigh). No system text is added. Per-effort thinking budgets are engine-side (`B70_THINKING_BUDGET` in the `.env`); when one runs out, the preset's `--reasoning-config` closes thinking with a short transition |
| attention | **sparse (QSA)** by default: the full-attention layers attend over the indexer's selection with our Triton kernels (`attn2`). **Dense** is the other value: the full-attention layers on the XPU FlashAttention backend (vLLM's XPU default). Both verified on this engine ([Sparse attention](#sparse-attention-qsa)). The DFlash2 drafter runs on `TRITON_ATTN` in both |
| KV cache dtype | no `--kv-cache-dtype` flag in either preset, so bf16 (`auto`) |
| tested | 4× Arc Pro B70; 4 cards (TP4 + expert parallel, 12 slots, 262,144 context) and 2 cards (TP2 + expert parallel, 4 slots, 262,144 context, hot/cold expert offload) |
| run | `v0.32/engine/serve.sh qwen38-flash-next preset-4card` (or `preset-2card`), after [Build and run](../../README.md#build-and-run) and [the weights](#weights) |

## Weights

1. **Intel AutoRound** @ `4c67bf68` without the 102.4 GB PLE shard (78.8 GB to download): `tools/intel_snapshot.py snapshot`.
2. **The INT8 PLE table:** devan-carlin's `ple_table_qwen4exp.pt` (`devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d`)
   through `tools/build_int8_ple.py build` ([docs/ple-int8.md](../../../docs/ple-int8.md)); the presets read it from NVMe.
3. **The serve dir:** symlink every snapshot file into it and copy [`serve-config-qsa.json`](serve-config-qsa.json) over its
   `config.json` (the dense config plus `indexer_n_heads`, `indexer_kv_heads`, `indexer_head_dim`, `indexer_budget`,
   `indexer_compress_ratio` in `text_config`, so the 39 indexer tensors load), or, for dense attention,
   [`v0.30/engines/serve-config-intel-autoround.json`](../../../v0.30/engines/serve-config-intel-autoround.json).
4. **The drafter:** `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744` into `/models/qwen3.8-flash-next/DFlash2`.
5. **2-card preset only:** the hot/cold rankings seed `configs/static_hot_cache_rankings.json` from
   [DominikBucko/qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090) @ `4dca58f` (Apache-2.0;
   sha256 starts `1230c61a`; an order of experts per layer, not counts) into `/models/qwen3.8-flash-next/`.

Checkpoint details for every build: [docs/weights.md](../../../docs/weights.md).

### Other weights

| build | on 0.32 | measured on |
|---|---|---|
| **Intel AutoRound** `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` | **recommended**; every number on this page | 0.32; and the 0.30.0-b70.2 recipe, incl. 8 days 6 hours in one engine ([0.30 page](../../../v0.30/README.md#intel-autoround)) |
| **AWQ** `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125` | not booted (vLLM 0.32 accepts it natively) | 0.30.0-b70.2 recipe: validated, 3 days 16 hours in one engine |
| **Minachist experts** `Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`, branch `new-main` @ `411505aa`, repacked into Intel's checkpoint with `tools/minachist_repack.py` | supported variant, not booted | 0.30 perf line: within noise of Intel except one-stream decode, 154 against 166 tok/s |
| **devan** `devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d` | reference, not booted | 0.30 line |

Why Intel over AWQ, measured on the 0.30.0-b70.2 recipe: speed is a tie (Intel 2.8 % faster at one short stream, level at 4
to 10 streams); MMLU, TruthfulQA, agentic coding and long context do not separate them; host RAM is the same; both have clean
multi-day runs. The full comparison is on the [0.30 page](../../../v0.30/README.md#models). Minachist's experts (int4, group
128, symmetric, F16 scales) go into Intel's checkpoint with `tools/minachist_repack.py convert <intel snapshot> <minachist
download> <new snapshot>`; the port of that tool has been tested on synthetic tensors only, so run its `verify` mode on your
snapshot.

## Presets

Both presets were measured on 2026-10-26 with the 0.32.0-b70.4 engine and kernel build (#660/#662), Intel AutoRound,
single runs, greedy (temperature 0, thinking off), with the harness of the earlier 12-slot measurements on the 0.1.15.4
build (`bench_det` for 1 and 4 streams, 8 parallel generations). What 0.32.0-b70.5 changes for each preset, and what was
measured on it, follows the table.

| as measured on 0.32.0-b70.4 | 4-card preset | 2-card preset (synchronous hot/cold swap) |
|---|---|---|
| cards, tensor parallel | 4, TP 4 (draft TP 4) | 2, TP 2 (draft TP 2); measured on cards 0 and 3 |
| attention, slots, context | sparse, 12, 262,144 | sparse, 4, 262,144 |
| expert offload | off (all experts in VRAM) | hot/cold, 145 of 256 experts per layer per rank in host RAM |
| KV tier (host RAM) | 128 GiB | 64 GiB (rows below measured at 16 GiB) |
| GPU KV cache (tokens) | 344,021 (344,145 on a second boot) | 273,169 |
| decode, 1 stream (tok/s; ms per step) | 156.4 (23.9) | 58.2 (61.5) |
| decode, 4 concurrent (total tok/s; ms per step) | 287.4 (46.4) | 69.4 (187.9) |
| decode, 8 streams (total tok/s, two runs) | 580.9 / 577.8 | 109.2 / 119.7 |
| prefill (tok/s) | 5,382 at 32K, 4,897 at 131K (12 slots, the 0.1.15.4 kernel build with identical engine code) | 2,768 at 32K (2,753 / 2,769 / 2,782) |
| host MemAvailable with it up (256 GB host) | 60.4 GiB | 154.9 GiB at a 16 GiB tier |
| quality: MMLU 300 / TruthfulQA MC1 200 | 86.7 / 89.0, thinking off, temperature 0 (the 0.32 4-card engine with dense attention, 2026-10-23) | 92.0 / 89.0, thinking on, medium (the b70.5 engine, 2026-10-29; TP2 against TP4 fidelity is at the floors, item 3) |
| gates | burst 0/150; token compare (det2) against the 0.1.15.4 build: PASS (c=1 3/8 identical + 5 near-tie; c=8 0/8 + 8 near-tie) | completion correct, journal clean; burst 0/150, structured output, mix probe and a 128K needle passed on the pre-release environment (below) |

How to read it:

1. **4-card per-step time is the clean signal.** 23.9 ms per step equals the 12-slot measurement on the 0.1.15.4 build (162.7 tok/s at
   one stream there); the one-stream difference is draft acceptance on four prompts (3.8 against 3.9 tokens per step), not
   kernel speed. 4 concurrent matches (287.4 against 287.1).
2. **The 2-card preset trades speed for the free pair.** Two cards carry the whole model: 145 experts per layer per rank live in
   host RAM, and every routed cold expert is read over PCIe. One-stream decode is 0.37× the 4-card preset.
3. **2-card fidelity** (pre-release environment with the same engine code, 48 GiB tier): teacher-forced on 8 prompts, TP2
   against TP4 KL 0.00132 / top-1 99.32 %, at the floors (TP2 against itself 0.00114 / 99.46 %, TP4 against itself
   0.00117 / 99.07 %). On a 12K-token document beyond 2K tokens it is at the TP4 floor; below 2K it is at its own TP2 floor
   (1.6× the TP4 floor).
4. **Not measured on the b70.4 environment:** the agent-session replay and long prefill (131K, 260K) for either preset,
   and the dense attention value at 12 slots. Those numbers come from the 0.1.15.4 build with the identical engine code and
   are labelled so where they appear ([Parameters](#parameters), [Sparse attention](#sparse-attention-qsa)).

### What 0.32.0-b70.5 changes

1. **Both presets: `B70_OFFLOAD_GDN_BACKSTEP=0`.** With the CPU tier and prefix caching the backstep restored a wrong GDN
   state on a prefix-cache hit (found on Qwen3.6-35B-A3B, [CHANGELOG](../../../CHANGELOG.md)); the engine also forces it off
   under `mamba_cache_mode=align`. What it bought, CPU-tier reuse of a state one block below the replay boundary, is lost: such
   a miss re-prefills the GDN tail (about 26 s at 94K tokens on Flash-Next). GPU prefix-cache hits are unaffected.
2. **2-card preset: the asynchronous hot/cold swap** (`B70_MOE_HOTCOLD_ASYNC_SWAP=1`, patches/vllm-models 0005, with a swap
   budget of 16, decay 0.85 and 16 spare cold rows per layer). Measured 2026-10-29 on cards 0 and 3, 64 GiB tier, against the
   synchronous swap on the same day:

   | 2-card preset | synchronous swap | asynchronous swap (shipped) |
   |---|---|---|
   | steady decode, 1 stream / 4 concurrent (`bench_det`, tok/s) | 55.7 / 68.9 | 57.1 / 72.6 |
   | after an 11-minute agent-session replay: essay 1 stream / story 4 concurrent (tok/s) | 33.0 / 60.7 | 38.2 / 77.5 |
   | host stall per swap tick | 160–940 ms | p50 82 ms |

   The swap follows the traffic, so a domain shift (an agent session after prose) leaves cold experts in the hot set for a
   while: decode is slower right after one, and the asynchronous swap recovers without stalling decode.
3. **The b70.5 engine and kernels.** The 2-card preset was re-gated on them (2026-10-29): GPU KV 267,319 tokens; burst 0/90;
   structured output with speculative decoding 54/54; a 10-turn agent loop with real tools, 50 completions with prompts up
   to 53K tokens: no repeated finished tool calls, 0 length and 0 repetition stops; quality, thinking on, MMLU 300 /
   TruthfulQA MC1 200: 92.0 / 89.0 against 92.7 / 89.0 on the kernels without #605 (McNemar p 0.68 / 1.0). Warm against
   cold prefix cache cannot be compared token for token on this preset (two cold runs already differ, from the hot/cold swap
   under traffic); the next-token distribution after a warm hit stays close to the cold runs (KL mean 0.29 against a
   cold-against-cold 0.20, top-1 equal to the cold majority 6 of 6), with no divergence at the scale of the fault. With the native-int8 kernels (before #605) `bench_det` read
   57.7 / 76.6 tok/s at 1 / 4 streams.
4. **Host RAM, 2-card preset at its shipped 64 GiB tier:** 118.6 GiB MemAvailable with no other engine on our 256 GB host
   (2026-10-30), with the 16 spare cold rows per layer included.
5. **4-card preset:** only the backstep key changes. It was not re-measured on the b70.5 kernels; the kernel changes are the
   native int8 ops (not on Flash-Next's path) and #605, which changes the grouped GEMM only for shapes whose scale pitch is
   under 64 bytes.

## Parameters

One engine; everything below is a parameter of it, set in the preset files (a `PARAMETER` comment marks each line). No
combination matrix: each row gives the default (the 4-card preset's value unless noted), the other working values and what
each was measured at.

| parameter | where | default | other values | measured at |
|---|---|---|---|---|
| **attention** | the serve dir's `config.json`, plus three `B70_QSA_*` keys in the env | **sparse (QSA)**: [`serve-config-qsa.json`](serve-config-qsa.json) with `B70_QSA_ATTN_IMPL=attn2`, `B70_QSA_TOPK=native_det`, `B70_QSA_TOPK_SELECT_IMPL=triton` (both presets) | **dense**: [`v0.30/engines/serve-config-intel-autoround.json`](../../../v0.30/engines/serve-config-intel-autoround.json) in the serve dir and the three keys removed. Working, measured below | sparse: [Presets](#presets). Dense, b70.3 build (0.1.15.4 environment, 8 slots, patched plugin): one stream 169.3 tok/s, 4 / 8 streams 306.2 / 605.5, prefill 8K / 32K / 131K 6,305 / 6,114 / 4,641, KV 397,309. Dense, 12 slots (0.1.15.4 environment): KV 385,685, one stream 169.9, 4 streams 297, 8 streams 639, agent replay 232 s. **Dense at 12 slots not re-measured on 0.1.16.2.** Sparse against dense on the same engine code: [Sparse attention](#sparse-attention-qsa) |
| **cards** | `--tensor-parallel-size`, `draft_tensor_parallel_size`, `ZE_AFFINITY_MASK` | 4 | 2 (the 2-card preset) | both: [Presets](#presets) |
| **drafter** | `--speculative-config` | DFlash2, K = 7 (incoai's drafter), both presets | MTP (`"method":"mtp"`, k = 3; the MTP draft-prefill patch 0021 and the uniform-drafts guard 0026 are in the series) | DFlash K = 7: every Flash-Next 0.32 number. MTP: only on the 0.30.0-b70.2 line (107.5 tok/s at one stream, 16 slots); not measured on 0.32 |
| **expert offload (hot/cold)** | `B70_MOE_HOTCOLD=1`, `B70_MOE_COLD_EXPERTS=N` (cold experts per layer per rank, of 128 at TP4 and 256 at TP2) and the keys of the hot/cold block in `preset-2card.env` | off (4-card) | on with N = 145 (2-card) | off: the 4-card preset. On, N = 145 at TP2: the 2-card preset. At TP4 (N = 16, before the capture fix, prefetch off): booted at KV 471,821 tokens; no speed or fidelity number |
| **KV offload size** | `--kv-offloading-size` (GiB of host RAM, split evenly over the ranks; keep it a power of two times the card count) | 128 (4-card) | 64 (2-card); 16 for a dev engine or with less host RAM | 128: the 4-card preset and every replay number. 16: the 2-card measurements. 48 at TP2 (pre-release environment): KV 274,069, host MemAvailable ~119 GiB, so 16 frees ~35 GiB more |
| **slots** | `--max-num-seqs` and the capture list (every multiple of 8 up to slots × 8, then 256 / 512 / 1024) | 12 (4-card) | 8 (within noise of 12 on the sparse engine); 4 (2-card); 16 was dropped (one preemption under 12-session load, −6 % at 4 streams) | sparse at 4 cards, 0.1.15.4 environment: 8 against 12 slots 280 against 287 tok/s at 4 streams, replay 204.5 against 206.6 s. 4: the 2-card preset |
| **GPU memory** | `--gpu-memory-utilization` | 0.89, both presets | lower for more headroom; 0.88 leaves too little KV for one 262,144-token request at TP4 | 0.89: every Flash-Next 0.32 number; the highest value vLLM's start-up check accepts on compute-runtime 26.35 |
| **oneCCL plugin** | `CCL_PLUGIN`, the five `CCL_SYCL_*_TMP_BUF` keys and the `expandable_segments` part of `PYTORCH_ALLOC_CONF` | patched legacy plugin ([build step 6](../../README.md#build-and-run); +~2 % KV), both presets | the stock plugin: remove those three groups of lines; never enable expandable segments with the shipped plugin | dense, 8 slots, b70.1 build: KV 388,872 (stock) against 397,309 (patched), speed within noise. Stock plugin with sparse: not run |
| **opt-out `B70_KV_SPLIT_GROUPS=0`** | env | on (66 KV groups of 905,216 B blocks) | 0: the old 7-group packing | sparse, 4 cards, 12 slots, 0.1.15.4 environment: KV 344,706 with the split; output unchanged (KL against unsplit sparse at its own floor); replay GPU prefix hit rate 0.37 → 0.46 |
| **opt-out `B70_KV_DRAFT_PAGE_FIT=0`** | env | on (patch 0038) | 0: the draft page sets every KV block as before | TP4: no effect (gate PASS, KV unchanged). TP2: needed for the 2-card preset's KV (before it: 77,885 tokens at a 64K context) |

Launch-time changes that are not parameters: the kernels, the patches and the oneAPI runtime are fixed per release.


## Host RAM (Flash-Next)

The 4-card preset takes ~164 GiB after start on our 256 GB host: the CPU KV tier is 128 GiB of pinned memory (32 GiB per
rank, which is why `AllowUnrestrictedSize=1` is needed), the PLE row cache 8 GiB, and the rest is the four workers.
MemAvailable was 35–39 GiB with ~20 GiB of other work on the host (b70.1) and 60.4 GiB with the 4-card preset up
(2026-10-26). The 2-card preset left 154.9 GiB available with a 16 GiB tier and 118.6 GiB with the shipped 64 GiB tier (2026-10-30, with
the asynchronous swap's spare rows): it holds the tier, a 4 GiB PLE row cache and the cold experts (145 per layer per rank,
pinned). Pinned memory does not count against a container memory limit, so watch the
host's MemAvailable, not the container's. With less RAM, lower `--kv-offloading-size` (keep it a power of two times the
card count) or move the PLE table into RAM only if you have the 49 GiB it then needs.


## Numbers, 0.32.0-b70.1 (how the line was validated)

Three configurations measured with the same harness on the same host, one after another, on an idle engine (each
timed cell waits for 20 s with no running or waiting requests). Single runs unless noted; read differences under
~5 % as noise: 0.32 matches the 0.30 perf line, and the gain over 0.30.0-b70.2 is the DFlash2/async/bundle stack, which
runs on both. These are **not** comparable with the per-build tables in [v0.30/README.md](../../../v0.30/README.md#models), which use a different
harness (the b70.2 Intel row there reads 90.4 tok/s at one stream; this harness reads 107.5 for a closely matching
configuration). The 0.32 column is dense attention at 8 slots on the 0.1.15.4 kernel build (the b70.1 code). The current presets' numbers are in [Presets](#presets).

| | 0.30.0-b70.2 (as released: MTP k=3, 16 slots) | 0.30 perf line (DFlash2 K=7, 8 slots) | **0.32.0-b70.1** |
|---|---|---|---|
| decode, 1 stream (tok/s) | 107.5 | 166 | **169.3** (172.3 before RC1 and the plugin) |
| decode, 4 concurrent requests (total tok/s) | 295 | 287 (repeat 296) | **306.2** |
| decode, 8 streams (total tok/s) | 370 | 588 / 601 | **605.5** (before RC1 and the plugin: 605 / 622) |
| prefill 8K / 32K / 131K (tok/s) | 4,052 / 3,833 / 2,953 | 6,125 / 6,092 / 4,648 | **6,305 / 6,114 / 4,641** |
| GPU KV cache (tokens) | ~311K | 406,432 | **388,872** (397,309 with the patched plugin) |
| agent replay: wall time, TTFT p50 / p90, preemptions | 401 s, 2.90 / 6.57 s, 0 | 302 s, 2.25 / 4.85 s, 0 | **268 s, 2.13 / 4.71 s, 0** |
| quality, MMLU 300 / TruthfulQA 200 | 259 / 177 | paired vs the left column: n.s. | paired vs the 0.30 perf line: 1+1 / 0+1 discordant, p = 1.0 / 1.0 |
| gates | pass | pass | burst 0/150, structured output clean, mix 7/7, greedy token compare: no garbage, first divergences at small baseline margins (not an equivalence check, see row 7) |
| host MemAvailable while serving (256 GB host, ~20 GB of other work on it) | — | ~44–46 GiB | **35–39 GiB** |

How each row was taken:

1. **Decode, 1 and 4.** `bench_det` (taken from Niluthon/qwen38-flash-tp2, Apache-2.0): steady decode milliseconds
   per step, converted with the measured tokens accepted per step. Greedy: temperature 0, thinking off. At temperature
   0 the K=7 configurations use the vocab-parallel greedy verify; sampled requests (production's default temperature
   is not 0) do not take that path, and no speed number at a sampling temperature is in this table.
   Rows 1–3 of the 0.32.0-b70.1 column were taken with the patched plugin (today's `preset-4card.env` minus its three
   `B70_QSA_*` keys, then `serve-plugin.env`); the stock-plugin configuration (then `serve.env`) was measured within noise
   of it, so the same numbers stand for both.
2. **8 streams.** Eight parallel generations, total generated tokens over wall time. The second 8-stream run of the
   served configuration overlapped other traffic and is not reported.
3. **Prefill.** One uncached, salted prompt of 8K, 32K or 131K tokens; prompt tokens over time to first token.
4. **KV.** The GPU KV cache size vLLM reports at start.
5. **Replay.** Six recorded agent sessions (243 requests, salted per run) replayed deterministically against the
   128 GiB CPU tier. The 0.32 row ran on the stock-plugin configuration (the KV offload code is the same; the plugin was added
   after it); the plain 0.32 def without the private pool needed 281 s with 2 preemptions.
6. **Quality.** MMLU 300 and TruthfulQA MC1 200, the same items for every configuration, paired McNemar test against
   the configuration it replaced. Every step of the chain was not significantly different: no difference was detected,
   which at these sample sizes (one or two discordant pairs) is not an equivalence bound; differences of a few points
   would not be detected.
7. **Gates.** Burst: 150 requests landing in the same steps at concurrency 3, 4 and 8, checked for NaN and garbage
   output. Structured output with speculative decoding. A mixed long/short probe. Greedy token compare at
   concurrency 1 and 8 against the previous configuration: no garbage markers, and where the outputs diverge, the
   first divergence sits at a small margin between the baseline's top two tokens. It does **not** show equivalence:
   the compare did not check that the candidate chose the baseline's runner-up token at a divergence, and it compared
   only the overlapping prefix of the two outputs (an empty or shorter output was not caught). The final captures for
   this release (8 prompts × 256 tokens at concurrency 1 and 8) have full-length outputs for every prompt. A stricter compare is being built; these rows will be re-run
   with it.
8. **Long context**, decode at one stream, two runs each: 8K / 64K / 128K prompts 209 / 200 / 172 tok/s on 0.32 vs
   217 / 220 / 169 on the 0.30 perf line (noise level at n = 2).

What 0.32 costs: fewer GPU KV tokens than the 0.30 perf line (about 9K, ~2 %, with the patched plugin; about
17.6K, ~4 %, with the stock plugin), and less host-RAM headroom (35–39 vs 44–46 GiB
on our host).


## Sparse attention (QSA)

**The default attention of both presets: Qwen3.8-Flash-Next with its own sparse attention, on Arc Pro B70, in vLLM.**
Release 0.32.0-b70.2 added it on top of the dense line; 0.32.0-b70.3 brought it near dense speed; 0.32.0-b70.4 fixed its KV packing and made it the preset
default. The checkpoint's QSA indexer is kept and the full-attention layers attend over the indexer's selection (a
2,048-token budget) with Triton kernels, instead of over the whole context. It boots on 4 and on 2 cards, answers correctly,
passes the gates and the long-context needles, and its output matches dense up to the selection budget. Dense attention (the
indexer dropped, full attention) is the other value of the attention parameter and works on the same engine.

**Sparse against dense, same engine code.** Measured on the 0.1.15.4 build with identical engine code (the 0.1.16.2
environment then passed the gate with the token compare identical, [Presets](#presets)); sparse and dense booted back to
back, one engine at a time, 4 cards, Intel AutoRound, both with the patched oneCCL plugin. The single source for these
numbers, with method, floors and caveats, is [docs/measurements/v0.32-qsa.md](../../../docs/measurements/v0.32-qsa.md) §0:

1. **Agent-session replay** (recorded multi-turn, 243 requests): sparse 204.7 s (b70.4 build, 12 slots, 0 preemptions,
   draft acceptance length 5.06) against 230–235 s for dense (8 slots); median decode 61–69 against 45–52 tok/s.
2. **Short context** (comparison drill, b70.3 sparse code, 8 slots): one-stream decode 0.92× dense, 4 streams level,
   8 streams 0.93×, prefill up to 32K 0.89×.
3. **Long prefill:** 131K 1.06×, 260K 1.31× (4,480 against 3,420 tok/s).
4. **Long-context tasks** (384 items, exact match): equal up to 8K; at 128K 80/78 % sparse against 72/72 % dense (two boots
   each); real-document next-token loss at 32–128K 1.399 against 1.417 nats/token.
5. **GPU KV:** 344,706 tokens sparse (12 slots, b70.4 build) against 397,309 dense (8 slots).

The b70.3 build, measured with the same harness, host and window as dense, single runs (the table the b70.3 release
published; the kernels did not change in b70.4, the KV packing and slot count did):

| end to end, b70.3 build, both at 8 slots, 0.1.15.4 environment | dense | sparse QSA (b70.3) | sparse / dense | sparse in b70.2 (superseded) |
|---|---|---|---|---|
| decode, 1 stream (tok/s) | 169.3 | 153.1 | 0.90 | 125.0 (0.74) |
| decode, 4 concurrent (total tok/s) | 306.2 | 306.1 | 1.00 | 161.2 (0.53) |
| decode, 8 streams (total tok/s) | 605.5 | 569.0 | 0.94 | 281.4 / 279.7 (0.46) |
| prefill 8K / 32K / 131K (tok/s) | 6,305 / 6,114 / 4,641 | 5,659 / 5,475 / 4,916 | 0.90 / 0.90 / 1.06 | 5,096 / 4,635 / 3,456 |
| GPU KV cache (tokens) | 397,309 | 342,144 | 0.86 | 329,287 |

Same harness and host, same window, one engine at a time, single runs, both with the patched oneCCL plugin
(the patched-plugin configuration, [`preset-4card.env`](preset-4card.env) with and without its three `B70_QSA_*` keys; with the stock plugin
dense has 388,872 KV tokens; sparse with the stock plugin has not been run). Fidelity (identical to dense up to 2K tokens;
beyond that KL 0.0224 against dense on a 12K-token document, where dense's own run-to-run floor is about 0.013–0.014), the
kernel-level comparison with Intel's SYCL kernel (labelled as kernel level), the gates, the stress stage, the levers that
were tried and not shipped, and what is not measured are in [docs/measurements/v0.32-qsa.md](../../../docs/measurements/v0.32-qsa.md).

**What b70.4 changed.** The sparse engine packed all 12 full-attention layers and their compressed-key caches into one KV
group, so every block was 10.9 MB and every small per-request state group (GDN, PLE conv, QSA ring, DFlash SWA) paid whole
fat blocks (about 425 MiB per running request against about 287 MiB on dense: 32 % less free KV at 6 running requests). Patches 0033 and 0036 split
that bucket into 66 groups of 905,216 B blocks (`B70_KV_SPLIT_GROUPS=0` restores the old packing; `B70_KV_PACK_REPEATS=<n>`
sets the repeat explicitly and wins). Outputs are unchanged (teacher-forced KL against unsplit sparse at its own noise
floor), the agent replay's GPU prefix hit rate goes from about 0.37 to about 0.46, and KV usage on it from 73 % to 62 %.
Patch 0034 adds `B70_QSA_OPTIMER` (a per-op timer around the QSA ops, off by default; it replaces the XPU profiler,
which hangs the engine on the first decode graph replay) and 0035 a cache-path lifecycle driver. The 4-card preset
runs 12 sequences with a capture list of every multiple of 8 up to 96; 12 against 8 slots on the sparse engine is within
boot-to-boot noise.

**What b70.3 fixed.** The b70.2 sparse build ran out of GPU memory on 131K prefills under load (about 800 MiB, inside the
deterministic top-k's tie repair) and was 0.46–0.53× dense at 4–8 streams. The tie repair is now device-only (no host
synchronisation, no full-width sort), as a fused Triton select, and the faster scorer launch configurations from the sweeps
are the defaults. The build passed a stress stage of repeated 131K prefills plus 8-stream decode.

**What is in the add-on.** [`patches/vllm-qsa/`](../../engine/patches/vllm-qsa/) is 38 patches, applied on top of
[`patches/vllm/`](../../engine/patches/vllm/) (fork tag `v0.32.0-b70.4-dense`); they end at fork tag `v0.32.0-b70.4-qsa`, and the
hot/cold directory follows them ([The series](../../README.md#the-series)). The b70.2/b70.3 branch `b70/v0.32.0-qsa-release` carries the
first 32 of them on the b70.1 base. `v0.32/engine/scripts/check-series.sh --qsa` fails if the directory differs from a fresh export. The
first 32 patches are unchanged in content from b70.3; their headers (`From <sha>`, `[PATCH n/38]`) are regenerated because
patch 0050 of the dense base now sits below them. Every patch is authored `Lumnus`; 0001 is the upstream
vllm-project/vllm#60602 cherry-pick.

| patches | what |
|---|---|
| 0001–0009 | bring-up: FP8 QSA main KV on Ampere (upstream #60602, B70-QSA-01); block top-k on XPU with one dispatch function per indexer kernel; the sparse GQA and side-cache kernels and Qwen4Exp QSA attention (fused prepare, portable unfused path) on XPU; the raw-key ring widened to a whole group; kernel tests running on XPU; the decode scorer pads `tl.dot` N to 16 (Triton 3.8.0 refuses N < 16 at compile time); a chunked torch top-k (it ran out of memory on a long prefill) |
| 0010–0014, 0020–0023, 0025 | tuning and measurement: one launch-override hook (`B70_QSA_TUNE`) for every QSA Triton kernel, step-trace wrapping for module-level functions, an XPU microbench and sweep, profiler bucket summary, `B70_QSA_EAGER_BREAK=0` (graph-capture arm), all-row checking against the torch reference, and the fixes the first sweeps exposed |
| 0015, 0016, 0024 | `attn2` (`B70_QSA_ATTN_IMPL=attn2`): a sparse GQA kernel with no register spills on Xe2, its comparison bench, and its defaults from the sweep (tile 32 × 8 chunks, 256 GRF, one warp; the kernel refuses two warps at tile ≥ 32, where the IGC compiler crashes) |
| 0017–0019, 0022 | qualification: off-by-default hooks for the QSA side caches, `B70_QSA_TOPK=native_det` (the native top-k with the torch selector's output), 148 side-cache state-contract tests and an engine lifecycle driver |
| 0026 | a one-time log line naming the sparse attention implementation in use |
| 0027–0032 | 0.32.0-b70.3: `native_det`'s tie repair without a host sync or a full-width sort (0027; `B70_QSA_TOPK=radix` as an alternative), top-k once per verify step and skip-when-fits (0028, both off by default), the scorer launch defaults from the sweeps (0029; `B70_QSA_SCORE_DEFAULTS=legacy` restores the old ones), CPU tests for them (0030), the fused Triton select and the radix schedule by width (0031; `B70_QSA_TOPK_SELECT_IMPL=triton`), profiler buckets for `attn2` and the select (0032) |
| 0033–0036 | 0.32.0-b70.4: the split packed full-attention KV bucket (0033, env-gated, and 0036, on by default), the `B70_QSA_OPTIMER` per-op timer and its report script (0034, off by default), the `qsa_engine_lifecycle2` driver and its tests (0035) |
| 0037–0038 | 0.32.0-b70.4, final trunk fixes: C1 target-graph capture snapshots kept in host RAM and accounted in the memory profile (0037; the TP2 warm-up ran out of memory without it; `VLLM_XPU_C1_TARGET_GRAPH=1` only), and the DFlash draft's sliding-window KV bucket split by the pool bytes a max-length request draws, so the draft page no longer sets every block at TP2 (0038; `B70_KV_DRAFT_PAGE_FIT=0` opts out; TP4 untouched) |

Off by default: `attn2` (`B70_QSA_ATTN_IMPL`, default `splitk`, the first kernel), `native_det` (`B70_QSA_TOPK`, default
`auto`), the Triton select (`B70_QSA_TOPK_SELECT_IMPL`, default `torch`), top-k once per verify step
(`B70_QSA_TOPK_PER_STEP`) and skip-when-fits (`B70_QSA_TOPK_SKIP_FIT`) (approximate and neutral respectively, see the
measurements), the tuning hook, the graph-capture arm (it degrades fidelity beyond 2K tokens: do not serve it), the debug
and dump hooks. Both presets turn on `attn2`, `native_det` and the Triton select; the scorer defaults are in the code.

**Run it.** The presets run sparse attention by default: build as in [Build and run](../../README.md#build-and-run) and serve either
preset with [`serve-config-qsa.json`](serve-config-qsa.json) as the serve dir's `config.json` ([Weights](#weights)). The
first start compiles the QSA Triton kernels cold (7 min in our first boot, ~5 min with a warm cache).

What to choose when, and what not to change: [Lessons learned](../../../docs/lessons-learned.md).
