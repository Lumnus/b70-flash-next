# 0.32.0-b70.4 — Qwen3.8-Flash-Next on vLLM 0.32, Intel Arc Pro B70

The current line (configuration first validated on 2026-10-24): vLLM **v0.32.0** with wu1ff's 1.0.5 B70 chain, our B70 layers
ported from the 0.30 line, the 0.32-specific fixes below, two of our upstream vLLM fixes cherry-picked, sparse attention (QSA)
and hot/cold expert offload. Speculative decoding is **DFlash2 with 7 draft tokens** (incoai's 0.6B block-diffusion drafter)
with async scheduling, 4096-token prefill chunks, a host-RAM KV tier in a private pinned pool and the INT8 PLE table on NVMe.

**One engine, one environment, two presets.** The engine is fork tag `v0.32.0-b70.4` (vLLM v0.32.0 + 101 commits, exported as
three patch directories, [The series](#the-series)). The environment is torch 2.14 + vllm-xpu-kernels 0.1.16.2 rebuilt with our
two kernel PRs + our `libgdn_index64.so` ([Software](#software)). On top of them:

1. **4-card preset** ([`engines/preset-4card.args`](engines/preset-4card.args) / [`.env`](engines/preset-4card.env)): the
   performance configuration. Tensor parallel 4, sparse attention, 12 slots, 128 GiB KV tier.
2. **2-card preset** ([`engines/preset-2card.args`](engines/preset-2card.args) / [`.env`](engines/preset-2card.env)): tensor
   parallel 2 on two cards, so the other pair is free for any other model. Sparse attention, 4 slots, the full 262,144-token
   context, hot/cold expert offload, 64 GiB KV tier.

Everything else is a parameter of the same engine ([Parameters](#parameters)), dense attention included.

| | |
|---|---|
| code | fork tag [`v0.32.0-b70.4`](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.4) (branch `b70/v0.32.0-b70.4`), 101 commits on vLLM v0.32.0 (`5df63a7203`): [`patches/vllm/`](patches/vllm/) (50, the dense base, tag `v0.32.0-b70.4-dense`) + [`patches/vllm-qsa/`](patches/vllm-qsa/) (38, sparse attention, ends at tag `v0.32.0-b70.4-qsa`) + [`patches/vllm-hotcold/`](patches/vllm-hotcold/) (13, hot/cold expert offload) |
| weights | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` (the only build measured on 0.32) + drafter `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744` |
| GPUs | 4× Arc Pro B70 (32 GB each); the 4-card preset uses all four, the 2-card preset two |
| attention | the target model on vLLM's XPU default backend (FlashAttention) plus the QSA Triton kernels for sparse; the drafter on `TRITON_ATTN` |
| software | [Software](#software); pins with hashes in [PROVENANCE.md](../PROVENANCE.md) §8 |
| tested host | AMD EPYC 7402, 256 GB RAM, WD_BLACK SN850X 4 TB NVMe |
| status | gates, quality, speed and an agent-session replay measured ([Presets](#presets), [Parameters](#parameters)). No multi-day endurance run on 0.32 yet. No container image. |

The 0.30.0-b70.2 recipe is in [`v0.30/`](../v0.30/README.md), unchanged, and is the stable fallback line.

## Presets

Measured on 2026-10-26 with the released engine and environment, Intel AutoRound, single runs, greedy (temperature 0,
thinking off), with the harness of the earlier 12-slot measurements on the 0.1.15.4 build (`bench_det` for 1 and 4 streams,
8 parallel generations):

| | 4-card preset | 2-card preset |
|---|---|---|
| cards, tensor parallel | 4, TP 4 (draft TP 4) | 2, TP 2 (draft TP 2); measured on cards 0 and 3 |
| attention, slots, context | sparse, 12, 262,144 | sparse, 4, 262,144 |
| expert offload | off (all experts in VRAM) | hot/cold, 145 of 256 experts per layer per rank in host RAM |
| KV tier (host RAM) | 128 GiB | 64 GiB (rows below measured at 16 GiB) |
| GPU KV cache (tokens) | 344,021 (344,145 on a second boot) | 273,169 |
| decode, 1 stream (tok/s; ms per step) | 156.4 (23.9) | 58.2 (61.5) |
| decode, 4 concurrent (total tok/s; ms per step) | 287.4 (46.4) | 69.4 (187.9) |
| decode, 8 streams (total tok/s, two runs) | 580.9 / 577.8 | 109.2 / 119.7 |
| prefill 32K (tok/s) | not re-measured (5,475 on the b70.3 build, 8 slots) | 2,768 (2,753 / 2,769 / 2,782) |
| host MemAvailable with it up (256 GB host) | 60.4 GiB | 154.9 GiB at a 16 GiB tier; about 107 GiB expected at 64 |
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
4. **Not measured on the released environment:** the agent-session replay and long prefill (131K, 260K) for either preset,
   and the dense attention value at 12 slots. Those numbers come from the 0.1.15.4 build with the identical engine code and
   are labelled so where they appear ([Parameters](#parameters), [Sparse attention](#sparse-attention-qsa)).

## Parameters

One engine; everything below is a parameter of it, set in the preset files (a `PARAMETER` comment marks each line). No
combination matrix: each row gives the default (the 4-card preset's value unless noted), the other working values and what
each was measured at.

| parameter | where | default | other values | measured at |
|---|---|---|---|---|
| **attention** | the serve dir's `config.json`, plus three `B70_QSA_*` keys in the env | **sparse (QSA)**: [`serve-config-qsa.json`](engines/serve-config-qsa.json) with `B70_QSA_ATTN_IMPL=attn2`, `B70_QSA_TOPK=native_det`, `B70_QSA_TOPK_SELECT_IMPL=triton` (both presets) | **dense**: [`v0.30/engines/serve-config-intel-autoround.json`](../v0.30/engines/serve-config-intel-autoround.json) in the serve dir and the three keys removed. Working, measured below | sparse: [Presets](#presets). Dense, b70.3 build (0.1.15.4 environment, 8 slots, patched plugin): one stream 169.3 tok/s, 4 / 8 streams 306.2 / 605.5, prefill 8K / 32K / 131K 6,305 / 6,114 / 4,641, KV 397,309. Dense, 12 slots (0.1.15.4 environment): KV 385,685, one stream 169.9, 4 streams 297, 8 streams 639, agent replay 232 s. **Dense at 12 slots not re-measured on 0.1.16.2.** Sparse against dense on the same engine code: [Sparse attention](#sparse-attention-qsa) |
| **cards** | `--tensor-parallel-size`, `draft_tensor_parallel_size`, `ZE_AFFINITY_MASK` | 4 | 2 (the 2-card preset) | both: [Presets](#presets) |
| **drafter** | `--speculative-config` | DFlash2, K = 7 (incoai's drafter), both presets | MTP (`"method":"mtp"`, k = 3; the MTP draft-prefill patch 0021 and the uniform-drafts guard 0026 are in the series) | DFlash K = 7: every 0.32 number. MTP: only on the 0.30.0-b70.2 line (107.5 tok/s at one stream, 16 slots); not measured on 0.32 |
| **expert offload (hot/cold)** | `B70_MOE_HOTCOLD=1`, `B70_MOE_COLD_EXPERTS=N` (cold experts per layer per rank, of 128 at TP4 and 256 at TP2) and the keys of the hot/cold block in `preset-2card.env` | off (4-card) | on with N = 145 (2-card) | off: the 4-card preset. On, N = 145 at TP2: the 2-card preset. At TP4 (N = 16, before the capture fix, prefetch off): booted at KV 471,821 tokens; no speed or fidelity number |
| **KV offload size** | `--kv-offloading-size` (GiB of host RAM, split evenly over the ranks; keep it a power of two times the card count) | 128 (4-card) | 64 (2-card); 16 for a dev engine or with less host RAM | 128: the 4-card preset and every replay number. 16: the 2-card measurements. 48 at TP2 (pre-release environment): KV 274,069, host MemAvailable ~119 GiB, so 16 frees ~35 GiB more |
| **slots** | `--max-num-seqs` and the capture list (every multiple of 8 up to slots × 8, then 256 / 512 / 1024) | 12 (4-card) | 8 (within noise of 12 on the sparse engine); 4 (2-card); 16 was dropped (one preemption under 12-session load, −6 % at 4 streams) | sparse at 4 cards, 0.1.15.4 environment: 8 against 12 slots 280 against 287 tok/s at 4 streams, replay 204.5 against 206.6 s. 4: the 2-card preset |
| **GPU memory** | `--gpu-memory-utilization` | 0.89, both presets | lower for more headroom; 0.88 leaves too little KV for one 262,144-token request at TP4 | 0.89: every 0.32 number; the highest value vLLM's start-up check accepts on compute-runtime 26.35 |
| **oneCCL plugin** | `CCL_PLUGIN`, the five `CCL_SYCL_*_TMP_BUF` keys and the `expandable_segments` part of `PYTORCH_ALLOC_CONF` | patched legacy plugin (build step 6; +~2 % KV), both presets | the stock plugin: remove those three groups of lines; never enable expandable segments with the shipped plugin | dense, 8 slots, b70.1 build: KV 388,872 (stock) against 397,309 (patched), speed within noise. Stock plugin with sparse: not run |
| **opt-out `B70_KV_SPLIT_GROUPS=0`** | env | on (66 KV groups of 905,216 B blocks) | 0: the old 7-group packing | sparse, 4 cards, 12 slots, 0.1.15.4 environment: KV 344,706 with the split; output unchanged (KL against unsplit sparse at its own floor); replay GPU prefix hit rate 0.37 → 0.46 |
| **opt-out `B70_KV_DRAFT_PAGE_FIT=0`** | env | on (patch 0038) | 0: the draft page sets every KV block as before | TP4: no effect (gate PASS, KV unchanged). TP2: needed for the 2-card preset's KV (before it: 77,885 tokens at a 64K context) |

Launch-time changes that are not parameters: the kernels, the patches and the oneAPI runtime are fixed per release.

## Numbers, 0.32.0-b70.1 (how the line was validated)

Three configurations measured with the same harness on the same host, one after another, on an idle engine (each
timed cell waits for 20 s with no running or waiting requests). Single runs unless noted; read differences under
~5 % as noise: 0.32 matches the 0.30 perf line, and the gain over 0.30.0-b70.2 is the DFlash2/async/bundle stack, which
runs on both. These are **not** comparable with the per-build tables in [v0.30/README.md](../v0.30/README.md#models), which use a different
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

## What changed against the 0.30 line, and why 0.32 was slower at first

Our first 0.32 build served correctly but was ~25 % slower at one stream. Measured causes, largest first:

1. **compute-runtime copy-through-lock.** Every small per-step host→device copy waited for the in-order queue to
   drain, so host preparation never overlapped the GPU: about 9.5 ms per step at one stream (found with phase timing
   and `py-spy --native`: the time was spent in `zeCommandListAppendMemoryCopy`). Fix: the compute-runtime keys
   `ExperimentalCopyThroughLock=0`, `ExperimentalH2DCpuCopyThreshold=0`, `ExperimentalD2HCpuCopyThreshold=0`.
2. **The DFlash2 fixed-order hyper-connection down GEMM** is 4.5–6.7× slower than `F.linear` above 32 rows. It was the
   whole prefill gap and most of the 8-stream gap. Fix: B70-V32-07, `B70_HC_ORDERED_MAX_ROWS=32`.
3. **The prefill hyper-connection graph regions** held ~0.7 GiB per card. Fix: B70-V32-08, `B70_PREFILL_HC_GRAPHS=0`.
   Together with the FP8 head (B70-V32-0053), the KV cache went from 337K to 389K tokens.
4. **PLE n-gram ids hashed on the host** (B70-V32-03): one small device→host copy replaces ~50 eager hash kernels
   (26.7 → 25.2 ms per step).
5. **K = 7 is the drafter's tuned shape.** Our earlier 0.32 runs used MTP k=3 or DFlash K=3.
6. A measurement lesson: GPU timing events on the host's critical path (B70-V32-02, diagnostic, off by default) cost
   ~3 ms per step themselves. Every number above comes from runs with them off.

Then two host-memory fixes:

7. **B70-V32-RC1, `B70_OFFLOAD_PRIVATE_PINNED=1`.** vLLM 0.32 keeps the CPU KV tier in a shared `/dev/shm` region
   registered with the device. SYCL reports registered memory as `usm::alloc::unknown`, so every CPU→GPU load was
   staged through torch's caching host allocator, and those blocks stayed cached: ~3.25 GiB per rank, ~13 GiB of host
   RAM. A private pinned pool per rank (as on 0.30) is copied directly. It also removes the need for a `/dev/shm` larger than
   the tier (≥ 140 GiB for 128 GiB). The upstream form of the kernel-side half is vllm-project/vllm-xpu-kernels#662.
8. **Expandable segments with a patched oneCCL plugin.** torch's `expandable_segments` puts XPU tensors in Level Zero
   virtual memory, which compute-runtime 26.35 reserves with a CPU mapping on Battlemage. oneCCL's legacy plugin
   classifies a buffer as host memory when `msync()` succeeds on it, sends it to the CPU collective without a stream,
   and libfabric then copies the device address on the host and segfaults. The plugin fix (uxlfoundation/oneCCL#229)
   asks SYCL for the pointer type when a queue is given. It buys ~8K KV tokens (~2 %) here, at the price of a hand-built `.so` tied to the oneCCL 2022.1.2 wheel and five
   `CCL_SYCL_*_TMP_BUF=1` copies, so it is a parameter. The presets use it; without the patched
   plugin, do not enable expandable segments (the stock-plugin configuration of b70.1–b70.3 did not; see [Parameters](#parameters)).

Upstream fixes folded in (all six are our posted PRs; #658 merged on main on 2026-10-26, the rest open):

| fix | upstream | here |
|---|---|---|
| GDN conv-state offset in 64 bits (an int32 wrap at block id ≥ 5,042 killed engines) | vllm-xpu-kernels #658 (fixes #657; merged on main 2026-10-26, in no release, so not in 0.1.16.2) | our `libgdn_index64.so` (the engine's GDN op; all conv-state sites 64-bit) |
| GDN kernels skip rows whose state index is the null block (graph padding rows wrote state into the shared null page) | vllm-xpu-kernels #660 (fixes #659; open) | our kernel build ([`patches/vllm-xpu-kernels/`](patches/vllm-xpu-kernels/) 0001) and `patches/libgdn-index64/` (our library) |
| `swap_blocks_batch`: copy pinned (and registered) host sources directly, `snapshot_src=False` | vllm-xpu-kernels #662 (fixes #661; open) | our kernel build ([`patches/vllm-xpu-kernels/`](patches/vllm-xpu-kernels/) 0002) + the caller, B70-V32-PR2 |
| Model Runner V2 prompt-logprobs chunks bounded by `max_num_reqs` (a 1024-row chunk spilled VRAM into host RAM) | vLLM #63564 (fixes #63563) | cherry-picked; replaces our 0029b |
| KV offload: pin a shared-prefix junction when a hybrid model's sparse group cuts the offload hit | vLLM #63566 (fixes #63565) | cherry-picked; replaces our 0014b |
| oneCCL legacy plugin: device USM that `msync()` sees as host takes the GPU path | oneCCL #229 (fixes #228) | plugin build, step 6 |


## The series

The engine is fork tag `v0.32.0-b70.4`, exported as three directories that apply in order on vLLM v0.32.0 (`git am`, or
`git apply` one after the other); `scripts/check-series.sh --all` fails if any of them differs from a fresh export of the
fork:

| directory | export of | commits |
|---|---|---|
| [`patches/vllm/`](patches/vllm/) | `v0.32.0..v0.32.0-b70.4-dense` | 50: the dense base (below) |
| [`patches/vllm-qsa/`](patches/vllm-qsa/) | `v0.32.0-b70.4-dense..v0.32.0-b70.4-qsa` | 38: sparse attention ([Sparse attention](#sparse-attention-qsa)) |
| [`patches/vllm-hotcold/`](patches/vllm-hotcold/) | `v0.32.0-b70.4-qsa..v0.32.0-b70.4` | 13: hot/cold expert offload ([below](#hotcold-expert-offload)) |

`tools/check-trunk-tree.sh` is the tree-level counterpart for maintainers: it applies the three directories to vanilla
v0.32.0 and fails if the result differs from a given trunk (scrubbed with a rules file) anywhere; see the header of the
script.

| patches (`patches/vllm/`) | what | author |
|---|---|---|
| 0001 | wu1ff's MIT license and pack notice, under `b70/third_party/wu1ff/` | Lumnus |
| 0002–0016 | wu1ff's 1.0.5 chain for vLLM 0.32 (target port, DFlash2, C1 verify/target graphs, astra500 MoE/HC/prefill paths, serving fixes), the `vllm/` hunks of his patch-chain taken unchanged, one commit per patch | wu1ff |
| 0017–0034 | our 0.30 layers ported: INT8 PLE from NVMe on wu1ff's host-prepared PLE (B70-L32-0013), thinking budget per effort, repetition stop, effort alias, dense-QSA indexer skip (main + MTP), MTP draft prefill graphs, uniform drafts guard, step trace, offload trace / junction / GDN backstep / empty-advance guard, prompt logprobs chunks, and their CPU tests | Lumnus |
| 0035–0037 | revert of our 0029b port, then vllm-project/vllm#63564 and #63566 | Lumnus |
| 0038–0049 | 0.32-specific: B70-V32-PR2, -01, -0053, -02/-02b (diagnostic), -03/-03b, -04, -07, -0054 (off), -08, -RC1 | Lumnus |
| 0050 | b70.4: backport of vllm-project/vllm#54163 (merged upstream 2026-10-23): `needs_eagle_block_drop()` excludes DFlash/DSpark, so the trailing prefix-cache block is no longer dropped for them and `disable_eagle_block_drop` is not needed. Applied by hand where the 0.32 line lacks `is_xpress`; its tests ported | Lumnus |

Off by default and not used in either preset: B70-V32-01 (`B70_C1_EAGER_ROUTE`), -02/-02b (`B70_PHASE_TIMING`), -04
(`B70_HC_PARALLEL_MAX_ROWS`, superseded by -07), -0054 (`B70_MOE_DECODE_OP_RESIDENT`; the astra 8-row path is faster at
decode; it loads the same library as the hot/cold decode op, build step 5).

### Hot/cold expert offload

[`patches/vllm-hotcold/`](patches/vllm-hotcold/), 13 commits: 0001–0010 are Niluthon's hot/cold patches from
[qwen38-flash-tp2](https://github.com/Niluthon/qwen38-flash-tp2) @ `e536e33` (Apache-2.0), ported to 0.32 through our 0.30
carry and kept with his authorship; 0011–0013 are ours.

| patches | what |
|---|---|
| 0001 | B70-0040: routed experts split between VRAM (hot) and pinned host RAM (cold); the cold ones are read by the same int4 grouped GEMM through a UVA view, and the two outputs are summed (bit-exact against all experts in VRAM) |
| 0002, 0004, 0006, 0007 | B70-0041 route counter, B70-0043 swapping (the VRAM hot set follows the traffic), B70-0046 route trace (diagnostic), B70-0047 decode-aware ranking and a learned hot set that survives restarts |
| 0003 | B70-0042: cold-expert prefetch for prefill (double-buffered DMA into two VRAM slots, eager batches of ≥ 512 tokens) |
| 0005 | B70-0044: input embedding table in pinned host RAM (`B70_EMBED_UVA`, off in both presets) |
| 0008, 0009 | B70-0048 fused hot+cold decode MoE op (`B70_MOE_DECODE_OP`, the library from build step 5) and an FP8 MTP draft head (off; the presets use DFlash); B70-0049 the op's CPU pool switch (`B70_MOE_CPU=0` in the 2-card preset) |
| 0010 | their CPU tests |
| 0011 | B70-V32-HC1: `B70_MOE_COLD_EXPERTS=N` (cold per layer per rank, so N means the same VRAM saving at TP4 and TP2), the 0.32 remap of unrouted ids as a helper, CPU tests |
| 0012 | B70-V32-HC2: drop two upstream tests that do not apply to 0.32 |
| 0013 | B70-V32-HC3: cold prefetch never runs under graph capture. It hung the 512/1024-token captures; capture now reads the cold experts over UVA at any size, and the prefetcher raises if it is called while the stream is capturing |

Off unless `B70_MOE_HOTCOLD=1`: with it unset nothing changes, which is how the 4-card preset runs.

## Software

| | version |
|---|---|
| GPU driver | Linux xe (kernel 6.18 on our host) |
| compute-runtime (Level Zero) | 26.35.39758.10 |
| oneAPI compiler (builds only) | 2026.1.1 (`icpx`) |
| Python | 3.12 |
| torch / triton | 2.14.0+xpu / 3.8.0+xpu |
| Intel runtime wheels | intel-sycl-rt, dpcpp-cpp-rt, intel-opencl-rt 2026.1.1; onemkl-sycl 2026.1.0 |
| oneCCL | 2022.1.2 (pip `oneccl`) core and loader; both presets use the legacy plugin rebuilt with oneCCL#229 (build step 6); the stock plugin is a parameter |
| vllm-xpu-kernels | **0.1.16.2 + our #660 and #662** (build step 3): the compiled libraries are built from upstream `6b30c9b` + [`patches/vllm-xpu-kernels/`](patches/vllm-xpu-kernels/) with FlashAttention-2, MQA-logits and Xe3P off; the attention libraries and the Python files come from the PyPI 0.1.16.2 wheel; the astra Python modules and hooks from [`patches/vllm-xpu-kernels-py/`](patches/vllm-xpu-kernels-py/) (step 4). Plus our `libgdn_index64.so` (step 5): the engine's GDN op, with 64-bit conv-state offsets at every site. Upstream #658 (the same 64-bit fix) is merged on main on 2026-10-26 and in no release |
| native libraries | wu1ff's B70-LLM-Controller 1.0.5 libraries, the Level Zero peer-residency shim, `libgdn_index64.so`, and Niluthon's `libb70_moe_decode.so` (2-card preset), all built in step 5 |
| other pins | transformers 5.17.0, auto-round-lib 0.16.0 |
| vLLM | v0.32.0 + the three directories above (fork tag `v0.32.0-b70.4`) |

**Which upstream commit 0.1.16.2 is.** There is no GitHub tag for 0.1.16.2. The PyPI wheel's seven `.py` files match upstream
byte for byte at `8b95ea6`, `6b30c9b` and `19c6ddc` and at nothing later; its paged-decode table has the rows that #566
(`6b30c9b`) adds, which rules out `8b95ea6`; `19c6ddc` changes only a CI file. So the source tree is `6b30c9b` (high
certainty for the tree, medium for the exact commit: upstream may have built from an untagged copy of it).

`/dev/shm`: no requirement (B70-V32-RC1). Without RC1, vLLM 0.32 maps the whole CPU KV tier into `/dev/shm`, so a
container would need `/dev/shm` ≥ 140 GiB for a 128 GiB tier.

## Host RAM

The 4-card preset takes ~164 GiB after start on our 256 GB host: the CPU KV tier is 128 GiB of pinned memory (32 GiB per
rank, which is why `AllowUnrestrictedSize=1` is needed), the PLE row cache 8 GiB, and the rest is the four workers.
MemAvailable was 35–39 GiB with ~20 GiB of other work on the host (b70.1) and 60.4 GiB with the released preset up
(2026-10-26). The 2-card preset left 154.9 GiB available with a 16 GiB tier (about 107 GiB expected with the shipped 64 GiB tier): it holds the tier, a 4 GiB PLE row cache and the cold
experts (145 per layer per rank, pinned). Pinned memory does not count against a container memory limit, so watch the
host's MemAvailable, not the container's. With less RAM, lower `--kv-offloading-size` (keep it a power of two times the
card count) or move the PLE table into RAM only if you have the 49 GiB it then needs.

## Build and run

The commands we ran, with our paths replaced by placeholders. The engine and environment were built and checked file by
file against these steps; the sequence has not been run end to end in this form from a clean clone.

1. **Base.** compute-runtime 26.35.39758.10 on the host, the xe driver, and oneAPI DPC++ 2026.1.1 for the builds.

2. **vLLM and the Python environment.**
   ```bash
   git clone -b v0.32.0-b70.4 https://github.com/Lumnus/vllm.git vllm-b70-0.32   # the engine: dense base + sparse + hot/cold
   uv venv --python 3.12 /opt/b70/venv
   uv pip install --python /opt/b70/venv/bin/python --index-strategy unsafe-best-match \
       --extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://wheels.vllm.ai/xpu/ \
       -r vllm-b70-0.32/requirements/xpu.txt torch==2.14.0+xpu oneccl==2022.1.2 auto-round-lib==0.16.0 transformers==5.17.0
   VLLM_TARGET_DEVICE=xpu uv pip install --python /opt/b70/venv/bin/python --no-build-isolation -e vllm-b70-0.32
   # triton's XPU launcher needs CL/cl.h; intel-sycl-rt ships only include/sycl. Check:
   printf '#include <sycl/detail/cl.h>\nint main(){}\n' | g++ -std=c++17 -I/opt/b70/venv/include -x c++ -fsyntax-only -
   ```
   Instead of cloning the tag you can apply the three patch directories to a v0.32.0 checkout in order
   (`git am v0.32/patches/vllm/*.patch v0.32/patches/vllm-qsa/*.patch v0.32/patches/vllm-hotcold/*.patch`).

3. **vllm-xpu-kernels 0.1.16.2 with #660 and #662.** Install the PyPI wheel (it brings the Python files and the attention
   libraries), then build the compiled libraries from source with our two PRs and lay them over the installed ones:
   ```bash
   uv pip install --python /opt/b70/venv/bin/python vllm-xpu-kernels==0.1.16.2      # PyPI wheel, sha256 starts 86bfa26abbb6
   git clone https://github.com/vllm-project/vllm-xpu-kernels.git && cd vllm-xpu-kernels
   git checkout 6b30c9b92cc525c32729df988b5f918e7c5661a5 && git am <kit>/v0.32/patches/vllm-xpu-kernels/*.patch
   source /opt/intel/oneapi/2026.1/oneapi-vars.sh   # plus gcc-toolset-13 on our builder
   VLLM_XPU_ENABLE_XE3P=OFF FA2_KERNELS_ENABLED=OFF MQA_LOGITS_KERNELS_ENABLED=OFF VLLM_XPU_ENABLE_ONEDNN=ON MAX_JOBS=6 \
     VLLM_VERSION_OVERRIDE=0.1.16.2+b70 /opt/b70/venv/bin/python setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
   # take only the eight compiled libraries; keep the wheel's attention libraries (not built here) and everything else
   K=/opt/b70/venv/lib/python3.12/site-packages/vllm_xpu_kernels; mkdir -p w && python3 -I -m zipfile -e dist/*.whl w
   for f in _C.abi3.so _moe_C.abi3.so _xpu_C.abi3.so xpumem_allocator.abi3.so libgdn_attn_kernels_xe_2.so \
            libgrouped_gemm_xe_2.so libgrouped_gemm_xe_default.so libmhc_kernels_xe_2.so; do install -m 0755 w/vllm_xpu_kernels/$f $K/$f; done
   ```
   Do not `pip install` the built wheel: with FlashAttention-2 and MQA-logits off it has no attention libraries
   (`libattn_kernels_xe_2/_xe_3`, `_vllm_fa2_C`, `libmqa_logits_kernels_xe_2`), and installing it would remove the PyPI
   ones. MAX_JOBS above 6 ran out of memory on a 64 GiB builder; the build takes ~36 min there. The kernel table must keep
   the paged-decode row `16,256,64,false,false,false` (a TP2 layout needs it; 0.1.16.2 has it).

4. **Kernels, Python side.** The astra modules the engine imports (`astra500_moe.py`, `astra_structural.py`,
   `astra_ownership.py`) and the astra hooks in `fused_moe_interface.py`, from
   [`patches/vllm-xpu-kernels-py/`](patches/vllm-xpu-kernels-py/), written against the 0.1.16.2 Python files:
   ```bash
   cd /opt/b70/venv/lib/python3.12/site-packages && for p in <kit>/v0.32/patches/vllm-xpu-kernels-py/*.patch; do git apply -p1 < "$p"; done
   ```
   The result is byte-identical to the validated environment (`fused_moe_interface.py` sha256 starts `3ef25b12`). It keeps
   0.1.16.2's own DP-padding mask, which sits after the astra early returns; both presets run DP = 1, so there are no padding
   rows.

5. **Native libraries.**
   1. **wu1ff's.** From [wu1ff/B70-LLM-Controller](https://github.com/wu1ff/B70-LLM-Controller) @ `5e5cf43` (release
      `qwen3.8-flash-next-v1.0.5`, MIT), `model-packs/Qwen3.8-Flash-Next/patches/production/native`: build the libraries with
      the pack's own build scripts (icpx 2026.1.1, torch 2.14 headers from the venv) and install `libastra_*.so`,
      `libbeyond_prefill.so`, `libdominant_w4_layout.so`, `liblong_hc.so`, `liblong_prefill_v3.so`, `libserving_prefill.so`
      into `site-packages/vllm_xpu_kernels/`. Put the Level Zero peer-residency shim at `/opt/b70/libl0_peer_residency_shim.so`.
      Our builds are not byte-identical to wu1ff's accepted hashes (different build paths); the shim is.
   2. **`libgdn_index64.so`.** Build it the same way after applying [`patches/libgdn-index64/`](patches/libgdn-index64/)
      to the pack's `dependencies/gdn-headers`, into `site-packages/vllm_xpu_kernels/`. vLLM 0.32 runs the GDN op from this
      library (`torch.ops._gdn_index64`; sha256 starts `aabd7ae6` in our environment). It imports `chunk_gated_delta_rule_xe2`
      and `l2norm` from the step-3 `libgdn_attn_kernels_xe_2.so`.
   3. **`libb70_moe_decode.so`** (the 2-card preset's hot/cold decode op). From
      [Niluthon/qwen38-flash-tp2](https://github.com/Niluthon/qwen38-flash-tp2) @ `e536e33`, `kernels/b70-moe-decode`:
      ```bash
      T=$(/opt/b70/venv/bin/python -I -c "import torch, os; print(os.path.dirname(torch.__file__))")
      icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg" -O3 -std=c++20 -shared -fPIC -ffast-math \
        -D_GLIBCXX_USE_CXX11_ABI=1 -I$T/include -I$T/include/torch/csrc/api/include -x c++ kernels/b70-moe-decode/b70_moe_decode.sycl \
        -o /opt/b70/libb70_moe_decode.so -L$T/lib -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_xpu -Wl,-rpath,$T/lib
      ```

6. **oneCCL legacy plugin** (used by both presets; skip it for the stock-plugin parameter value). One file, built against
   the venv's shipped oneCCL 2022.1.2:
   ```bash
   git clone -b master-v2 https://github.com/uxlfoundation/oneCCL.git && cd oneCCL
   git fetch -q origin pull/229/head && git checkout FETCH_HEAD   # caeb329, on master-v2 c5108e7
   source /opt/intel/oneapi/compiler/2026.1/env/vars.sh
   V=/opt/b70/venv; mkdir -p vinc /opt/b70/oneccl-legacy && ln -sfn $V/include/oneapi vinc/oneapi
   icpx -w -fsycl -O2 -fPIC -shared -std=c++17 -DNDEBUG -Ivinc -Isrc -Iinclude \
       plugins/legacy/ccl_legacy.cpp src/debug.cpp -o /opt/b70/oneccl-legacy/libccl_legacy.so \
       -L$V/lib -lccl -l:libccl.so.2 -ldl -Wl,--no-undefined -Wl,-soname,libccl_legacy.so
   ```
   `CCL_PLUGIN` in the preset `.env` points the shipped loader at it; the venv stays untouched.

7. **Weights.** Intel AutoRound @ `4c67bf68` without the 102.4 GB PLE shard, devan's `ple_table_qwen4exp.pt`, the INT8
   table from `tools/build_int8_ple.py`, and the snapshot from `tools/intel_snapshot.py` (as for 0.30, see
   [v0.30/engines/README.md](../v0.30/engines/README.md)). Make a serve dir: symlink every snapshot file into it and copy
   [`engines/serve-config-qsa.json`](engines/serve-config-qsa.json) (sparse attention, both presets' default) over its
   `config.json` (the dense config plus `indexer_n_heads`, `indexer_kv_heads`, `indexer_head_dim`, `indexer_budget`,
   `indexer_compress_ratio` in `text_config`, so the 39 indexer tensors load), or, for dense attention,
   [`v0.30/engines/serve-config-intel-autoround.json`](../v0.30/engines/serve-config-intel-autoround.json) (byte-identical to the one we
   serve). The drafter: `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744` into `/models/qwen3.8-flash-next/DFlash2`. For the
   2-card preset, the hot/cold rankings seed: `configs/static_hot_cache_rankings.json` from
   [DominikBucko/qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090) @ `4dca58f` (Apache-2.0;
   sha256 starts `1230c61a`; an order of experts per layer, not counts) into `/models/qwen3.8-flash-next/`.

8. **Run.** One preset at a time per card set; the 2-card preset needs only its two cards.
   ```bash
   P=preset-4card   # or preset-2card
   set -a; . v0.32/engines/$P.env; set +a
   mapfile -t args < <(grep -v '^#' v0.32/engines/$P.args | sed '/^$/d' | sed 's/^\(--[^ ]*\) \(.*\)$/\1\n\2/')
   LD_LIBRARY_PATH=/opt/b70/venv/lib /opt/b70/venv/bin/vllm serve /models/qwen3.8-flash-next/serve-dir "${args[@]}"
   ```
   The venv's own oneAPI 2026.1.1 runtime (`$VENV/lib`) must come first on `LD_LIBRARY_PATH`; a mixed runtime segfaults at
   exit. The first start compiles cold, the QSA Triton kernels included (7 min in our first boot); later boots take ~4–5 min.

Chat template: the examples use the checkpoint's own template. We serve a lightly modified one (thinking switches off
on `enable_thinking=false` or a "none/off" effort); it is not included.

## Sparse attention (QSA)

**Status: working, experimental; the default attention of both presets.** Release 0.32.0-b70.2 added sparse attention on
top of the dense line; 0.32.0-b70.3 brought it near dense speed; 0.32.0-b70.4 fixed its KV packing and made it the preset
default. The checkpoint's QSA indexer is kept and the full-attention layers attend over the indexer's selection (a
2,048-token budget) with Triton kernels, instead of over the whole context. It boots on 4 and on 2 cards, answers correctly,
passes the gates and the long-context needles, and its output matches dense up to the selection budget. Dense attention (the
indexer dropped, full attention) is the other value of the attention parameter and works on the same engine.

**Sparse against dense, same engine code.** Measured on the 0.1.15.4 build with identical engine code (the 0.1.16.2
environment then passed the gate with the token compare identical, [Presets](#presets)); sparse and dense booted back to
back, one engine at a time, 4 cards, Intel AutoRound, both with the patched oneCCL plugin. The single source for these
numbers, with method, floors and caveats, is [docs/measurements/v0.32-qsa.md](../docs/measurements/v0.32-qsa.md) §0:

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
(the patched-plugin configuration, [`preset-4card.env`](engines/preset-4card.env) with and without its three `B70_QSA_*` keys; with the stock plugin
dense has 388,872 KV tokens; sparse with the stock plugin has not been run). Fidelity (identical to dense up to 2K tokens;
beyond that KL 0.0224 against dense on a 12K-token document, where dense's own run-to-run floor is about 0.013–0.014), the
kernel-level comparison with Intel's SYCL kernel (labelled as kernel level), the gates, the stress stage, the levers that
were tried and not shipped, and what is not measured are in [docs/measurements/v0.32-qsa.md](../docs/measurements/v0.32-qsa.md).

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

**What is in the add-on.** [`patches/vllm-qsa/`](patches/vllm-qsa/) is 38 patches, applied on top of
[`patches/vllm/`](patches/vllm/) (fork tag `v0.32.0-b70.4-dense`); they end at fork tag `v0.32.0-b70.4-qsa`, and the
hot/cold directory follows them ([The series](#the-series)). The b70.2/b70.3 branch `b70/v0.32.0-qsa-release` carries the
first 32 of them on the b70.1 base. `scripts/check-series.sh --qsa` fails if the directory differs from a fresh export. The
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

**Run it.** The presets run sparse attention by default: build as in [Build and run](#build-and-run) and serve either
preset with [`engines/serve-config-qsa.json`](engines/serve-config-qsa.json) as the serve dir's `config.json` (step 7). The
first start compiles the QSA Triton kernels cold (7 min in our first boot, ~5 min with a warm cache).

**Known issues (sparse).**

1. **Slower than dense on short context** by 3–11 % on one stream, at prefill up to 32K and at 8 streams (above); at 4 streams
   it matches, and at 131K+ prefill and on agent-style session replays it beats it. No in-engine profile exists (the torch profiler hangs the engine on XPU), so where
   the remaining time goes is inferred.
2. **KV capacity about −13 %** against dense (344,706 tokens sparse at 12 slots against 397,309 dense at 8 slots, 4 cards, 0.1.15.4 environment), from the QSA side caches and the raw-key ring (vLLM warns that the QSA state backend's
   `kernel_block_size=16` is not fully optimised).
3. **Both variants are run-to-run nondeterministic from about token 512**, so bit-level comparison gates fail on either;
   fidelity was measured with KL against a same-engine floor instead.
4. **Prefill at 262K**: the first Triton kernel, run on Intel's main-branch Triton, beats `attn2` there (15.6 against
   17.6 ms per layer, kernel level); the release uses Triton 3.8.0.
5. **No multi-day run**, no generation or reasoning quality comparison with dense (retrieval-type tasks and next-token loss are measured), one host, Intel AutoRound only
   ([measurements, what this does not show](../docs/measurements/v0.32-qsa.md#5-what-this-does-not-show)).
6. **Plain `B70_QSA_TOPK=native` is not launch-repeatable** (equal scores ordered differently); do not serve it.
   `native_det` is the qualified native mode.
7. **Not run on the sparse engine:** the stock oneCCL plugin.


## Known issues (0.32)

1. **No multi-day run yet.** The 0.32 line has served since 2026-10-24, with planned stops for development windows;
   there is no endurance number for it yet.
2. **One hang in development.** Before the copy-through-lock fix, one boot with FULL graphs hung in `sample_tokens`
   (unit stopped, cards rebound). It did not recur in later boots or in the drill; the cause is not known.
3. **Dense KV (8 slots) is ~2 % below the 0.30 perf line with the patched plugin, ~4 % with the stock plugin.** Peak activation is 1.93 GiB per rank against 1.31 on 0.30, and it is not
   allocator fragmentation (expandable segments bought only ~0.1 GiB). Not yet attributed.
4. **GPU idles ~3 ms per step in the PLE sync at one stream.** A device→host probe takes 25 µs in isolation. Open.
5. **Only Intel AutoRound weights were measured on 0.32.** The AWQ build should work (vLLM 0.32 accepts it natively), but
   it was not booted on this line.
6. **AWQ / 0.30 items still apply** where the code is shared: card hangs after a GT reset ([v0.30/docs/known-issues.md](../v0.30/docs/known-issues.md)
   item 1), liveness must generate, and pinned memory escapes container limits.
7. **oneCCL `LL=twoshots`** corrupts output on this stack (burst gate 64/150 bad); every oneCCL change needs the burst gate and
   a token compare (garbage and divergence check, not an equivalence test) before serving.
8. **The released environment was measured for speed and gates, not for the replay and long prefill.** The agent-session
   replay, 131K/260K prefill and the long-context task set ran on the 0.1.15.4 kernel build with identical engine code; on
   the 0.1.16.2 environment the 4-card preset reproduces that build's per-step time (23.9 ms) and passes the token compare
   against it ([Presets](#presets)). Upstream #658 (64-bit GDN conv-state offsets) is merged on main and in no release, so
   the stock `_xpu_C` GDN op lacks it (the PyPI 0.1.16.2 build fails `tests/gdn_attn/test_causal_conv1d.py -k
   large_state_offset` 3 of 3 on a card; our rebuild has the same GDN source); the engine never calls it (its GDN op is `torch.ops._gdn_index64` from `libgdn_index64.so`, which passes the same tests 3 of 3).
9. **Our kernel build has no FlashAttention-2, MQA-logits or Xe3P code** (all off, as in every build we serve): the attention
   libraries come from the PyPI wheel, the fp8 MQA-logits ops are not registered (the engine does not call them; a model that
   needs them needs an MQA-on build, which ran out of memory on our 64 GiB builder), and B70 is Xe2.
10. **2-card preset: hot/cold costs speed.** One stream 58.2 tok/s against 156.4 on 4 cards. The fused hot+cold decode op
    with bf16 scales is not bit-exact against all experts in VRAM (relative L2 ≤ 2.9e-3; the two stock calls are); fidelity
    of the preset was measured at the floors ([Presets](#presets)).
11. **An occasional 4-card boot comes up at ~339K KV tokens** instead of ~344–345K (2 of 5 boots on 2026-10-26: one rank
    shows +0.08 GiB of non-torch memory). Harmless; cause unknown.
