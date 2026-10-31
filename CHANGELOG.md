# Changelog

Versions are `<vLLM base>-b70.<N>`; N increases whenever the patch series changes.

## Unreleased

Nothing yet.

## 0.32.0-b70.5 — 2026-10-31: sparse attention on B70 for Qwen3.8-Flash-Next, three models on one engine; the GDN backstep off in every preset

- **Fix: `B70_OFFLOAD_GDN_BACKSTEP=0` in every preset.** b70.4's two presets (and the 0.30 engine files `s16-kv128-mtp3`,
  `awq-s16-kv128-chunked`, `r8g-kv64-0014jb`, `nvme2-kv96-0014jb`) shipped `1`. With the CPU KV tier and prefix caching
  (`mamba_cache_mode=align`), the backstep's extra prefill stop is keyed on each request's own prompt length, so the request
  that cached a GDN state and the one that hits it build it through different chunk splits; the hit restores a state that does
  not match the history, and greedy output diverges from the first token. In an agent session it showed as the model
  re-issuing tool calls it had already finished. Found on Qwen3.6-35B-A3B with a warm-against-cold prefix-cache test
  (diverged at token 1 with the backstep on, with or without speculative decoding, on both model runners; identical with it
  off). The engine now also forces it off under align (`vllm-models` 0008, one log line; `B70_GDN_BACKSTEP_ALIGN_GUARD=0`
  opts out). What it bought, CPU-tier reuse of a state one block below the replay boundary, is lost: such a miss
  re-prefills the GDN tail. **b70.4 users: set `B70_OFFLOAD_GDN_BACKSTEP=0`.**
- **Three models.** Qwen3.8-Flash-Next (4 and 2 cards), Qwen3.8-27B (2 cards and 1 card) and Qwen3.6-35B-A3B (2 cards and 1
  card), each with a `meta.md` (pinned weights and drafter, tested cards, context, measured numbers) and its presets in
  `v0.32/models/<model>/`. Measured side by side on one 4-card host: Flash-Next 2-card, the 35B on card 1, the 27B on card 2.
- **Layout.** `v0.32/patches/` → `v0.32/engine/patches/`, `v0.32/scripts/` → `v0.32/engine/scripts/`,
  `tools/check-trunk-tree.sh` and `check_trunk_tree.py` → `v0.32/engine/tools/`, the Flash-Next presets and
  `serve-config-qsa.json` → `v0.32/models/qwen38-flash-next/`, `RELEASE-NOTES-b70.4.md` → `docs/releases/b70.4.md` (all
  `git mv`). New `v0.32/engine/serve.sh MODEL PRESET`. `v0.32/README.md` is the engine page; the Flash-Next presets,
  parameters, sparse attention and validation tables moved to its model page.
- **Engine: fork tag `v0.32.0-b70.5`**, b70.4's engine + 8 commits in the new `v0.32/engine/patches/vllm-models/`: the
  Qwen3.5/3.8 FP8 lm_head (`B70_HEAD_FP8`), the Triton W8A8 INT8 GEMM tiles of vllm-project/vllm#64190 and #64244, the DFlash
  draft-KV warning gate, the asynchronous hot/cold swap (`B70_MOE_HOTCOLD_ASYNC_SWAP`), the DFlash2 shortlist draft head
  (`B70_DFLASH_DRAFT_VOCAB`), native int8 activation quantisation and the oneDNN W8A8 GEMM (`B70_XPU_INT8_ONEDNN`), and the
  backstep guard. Every one is off unless its switch is set or its op is present. The first three directories are b70.4's,
  unchanged.
- **Kernels: five patches** on vllm-xpu-kernels `6b30c9b`: #660 and #662 as before, the native int8 ops (0003) with a
  precise-division build flag (0004), and vllm-project/vllm-xpu-kernels#605 (0005), the grouped GEMM scale-prefetch pitch guard
  the 35B's expert shapes need.
- **Flash-Next 2-card preset:** the asynchronous hot/cold swap (swap budget 16, decay 0.85, 16 spare rows): steady decode
  57.1 / 72.6 tok/s at 1 / 4 streams against 55.7 / 68.9 with the synchronous swap; after an agent-session replay 38.2 / 77.5
  against 33.0 / 60.7; host stall per tick p50 82 ms against 160–940 ms. Re-gated on the b70.5 engine and kernels: MMLU /
  TruthfulQA 92.0 / 89.0 (n.s. against 92.7 / 89.0). Host MemAvailable at the shipped 64 GiB tier: 118.6 GiB.
- **Flash-Next 4-card preset:** only the backstep key changes.
- **Generic chat templates**, one per model (`v0.32/models/<model>/chat_template.jinja`), passed by every preset
  (`--chat-template`): the checkpoint's own template plus the thinking switch ("off wins": `enable_thinking=false` or a
  none/off effort turns thinking off; on Qwen3.8 other effort names map onto xhigh/medium/low instead of raising). No serving
  line, no thinking guidance, no system text: behaviour and attribution belong to the client or gateway layer. Plain chats
  render byte-identical to the checkpoint's template. The per-card 27B and 35B template files are gone.
- **Qwen3.8-27B:** devan-carlin INT4 AutoRound run W4A8, DFlash2 K=7 with a 67,263-token shortlist draft head. 1 card: fp8 KV,
  a W8A8 int8 drafter (`quantize_dflash2_w8a8.py`), 147,456 context, 56–62 / 142–149 tok/s at one stream (prose / code).
  2 cards: 262,144 context, ~94 tok/s at one stream, MMLU / TruthfulQA 91.0 / 83.5.
- **Qwen3.6-35B-A3B:** Intel INT4 mixed AutoRound, DFlash K=7. 1 card: 196,608 context, 118–120 / 310–314 tok/s. 2 cards:
  262,144 context, 16 slots, 1,060,309 GPU KV tokens, 152–168 / 376 tok/s, MMLU / TruthfulQA 88.7 / 88.5.
- **Tooling.** `check-series.sh --models` and `--all` (four directories). `check-trunk-tree.sh` applies all four;
  `--no-models` for a trunk that ends at the hot/cold layer.
- **Docs.** The cover leads with sparse attention on Flash-Next, then the three models: a models table with peak decode,
  best prefill, quality, attention backend and KV dtype per preset, and a table of the verified attention backends. The
  engine page and three model pages carry the details. New `docs/lessons-learned.md` (what works, what does not, tuning and
  testing advice) replaces the known-issues lists; the 0.30 line is listed as the previous line. `docs/switches.md` lists the new switches. Paths in the older entries below name the
  layout of their release.
- Release notes: [RELEASE-NOTES-b70.5.md](RELEASE-NOTES-b70.5.md).

## 0.32.0-b70.4 — 2026-10-26: one engine, one environment, two presets; sparse attention and hot/cold in the engine; kernels 0.1.16.2 with our PRs

One release (no intermediate b70.4 builds were published).

- **One engine.** Fork tag `v0.32.0-b70.4` (101 commits on v0.32.0), exported to three directories that apply in order:
  `v0.32/patches/vllm/` (50, the dense base, tag `v0.32.0-b70.4-dense`), `v0.32/patches/vllm-qsa/` (38, sparse attention,
  ends at tag `v0.32.0-b70.4-qsa`) and the new `v0.32/patches/vllm-hotcold/` (13, hot/cold expert offload). Dense or sparse
  attention is a parameter of this engine, not a separate build.
- **One environment.** vllm-xpu-kernels 0.1.16.2 rebuilt from its source (`6b30c9b`) with our #660 and #662
  (`v0.32/patches/vllm-xpu-kernels/`, new), FlashAttention-2/MQA-logits/Xe3P off; the attention libraries and Python files from
  the PyPI wheel; the astra Python modules and hooks (`v0.32/patches/vllm-xpu-kernels-py/`, rebased onto the 0.1.16.2 files,
  content unchanged); our `libgdn_index64.so` for the 64-bit GDN conv-state offsets (upstream #658 is merged on main and in no
  release). 0.1.15.4 is retired.
- **Two presets.** `v0.32/engines/preset-4card.{args,env}` (TP4, sparse, 12 slots, 128 GiB KV tier): 156.4 tok/s at one
  stream, 287.4 at 4 concurrent, 580.9 at 8 streams, GPU KV 344,021. `v0.32/engines/preset-2card.{args,env}` (TP2 on two
  cards, the other pair free for any other model; sparse, 4 slots, 262,144 context, hot/cold with 145 cold experts per layer per
  rank, 64 GiB KV tier): 58.2 / 69.4 / 109.2 tok/s and GPU KV 273,169, measured at a 16 GiB tier (154.9 GiB host RAM left free; about 107 GiB expected at 64). They replace `serve.args`,
  `serve.env`, `serve-plugin.env`, `qsa-serve.args` and `qsa-serve.env`. Attention, cards, drafter, expert offload, KV offload
  size, slots, GPU memory, oneCCL plugin and two opt-outs are tabulated with their defaults and what each was measured at in
  [v0.32/models/qwen38-flash-next/meta.md](v0.32/models/qwen38-flash-next/meta.md#parameters).
- **Dense base: 50 patches.** 0001–0049 are the b70.1 series, unchanged in content (headers read `n/50`); **0050** backports
  vllm-project/vllm#54163 (merged upstream 2026-10-23): DFlash and DSpark no longer drop the trailing prefix-cache block, so
  `disable_eagle_block_drop` is no longer set (b70.1–b70.3 set it; agent-session replay 232 s against 266 s, dense, 8 slots).
- **Sparse: 38 patches** (0001–0032 unchanged in content from b70.3). The KV split patches are **0033** (the packed
  full-attention KV bucket split into 66 groups of 905,216 B blocks instead of 7 groups of 10,862,592 B, env-gated) and
  **0036** (the split on by default); `B70_KV_SPLIT_GROUPS=0` restores the old packing, `B70_KV_PACK_REPEATS=<n>` wins over
  both. GPU KV 344,706 tokens (12 slots, 0.1.15.4 build); the agent replay's GPU prefix hit rate about 0.37 → 0.46. **0034**
  is `B70_QSA_OPTIMER`, a per-op timer (off by default) with its report script; **0035** the cache-path lifecycle driver and
  its tests. **0037 (B70-C1-R1):** C1 target-graph capture snapshots live in host RAM and move through the device in chunks
  of at most 256 MiB; the remaining device footprint is checked inside the memory profile (the TP2 warm-up ran out of memory
  without it). **0038 (B70-K2-0003):** at TP2 the DFlash draft's sliding-window page is twice an attention page and set every
  KV block; its bucket is now split, the repeats chosen per group by the pool bytes a max-length request draws, as upstream
  #60996 does (no effect at TP4; `B70_KV_DRAFT_PAGE_FIT=0` opts out).
- **Hot/cold: 13 patches (new).** Niluthon's B70-0040..0049 from qwen38-flash-tp2 (routed experts split between VRAM and pinned
  host RAM, route counter, prefill prefetch, swapping, learned hot set, the fused hot+cold decode op, their tests), ported to
  0.32 with his authorship, and our HC1–HC3: `B70_MOE_COLD_EXPERTS=N` per layer per rank; two n/a tests dropped; cold prefetch
  never inside graph capture (it hung the 512/1024-token captures). Off unless `B70_MOE_HOTCOLD=1`.
- **Numbers.** The presets were measured on the released engine and environment (above). The sparse-against-dense comparison,
  the agent replay, long prefill and the long-context tasks were measured on the 0.1.15.4 kernel build with identical engine
  code; the 0.1.16.2 environment passed the gate with the token compare identical and reproduces that build's per-step decode
  time at 4 cards. Sparse against dense: agent replay 204.7 against 230–235 s; one-stream decode 0.92×, 4 streams level,
  8 streams 0.93×, prefill up to 32K 0.89×; prefill 131K 1.06×, 260K 1.31×; 128K long-context tasks 80/78 % against 72/72 %
  ([docs/measurements/v0.32-qsa.md](docs/measurements/v0.32-qsa.md) §0). Dense at 12 slots was measured once on the 0.1.15.4
  build (KV 385,685, replay 232 s), not on 0.1.16.2.
- **Tooling.** `v0.32/scripts/check-series.sh` checks the three directories (`--qsa`, `--hotcold`, `--all`).
  `tools/check-trunk-tree.sh` applies all three and requires the tree to equal the trunk exactly; `--no-hotcold` (with
  `--hotcold-count`, `--hotcold-tip`, `--hotcold-extra`) keeps the old mode that allows the trunk's hot/cold commits.
  `serve-config-qsa.json` re-indented to the dense file's style (JSON-equal).
- **Layout: the 0.30 line moved to `v0.30/`.** Everything that belongs only to the 0.30.0-b70.2 line left the repository
  root: `patches/`, `engines/`, `image/`, `scripts/`, the 0.30 known issues and measurements are now `v0.30/patches/`,
  `v0.30/engines/`, `v0.30/image/`, `v0.30/scripts/`, `v0.30/docs/known-issues.md`, `v0.30/docs/measurements/`, and
  `docs/line-0.30.md` is `v0.30/README.md`. The root holds the cover, `CHANGELOG.md`, `RELEASE-NOTES-b70.4.md`,
  `PROVENANCE.md`, `NOTICE`, `LICENSE`, the cross-line `docs/` (switches, weights, kernels, offload fix, PLE INT8 and NVMe,
  the 0.32 sparse-against-dense measurements) and `tools/`. `v0.32/` is unchanged. The 0.30 scripts run from `v0.30/`
  (`v0.30/scripts/check-series.sh`; the image builds with `v0.30/` as the context). File contents are unchanged apart
  from links; paths in the entries below name the layout of their release.
- Release notes: [docs/releases/b70.4.md](docs/releases/b70.4.md).

## 0.32.0-b70.3 — 2026-10-26: sparse attention (QSA) brought near dense speed; dense engine args

The dense series (`v0.32/patches/vllm/`, 49 patches, fork tag `v0.32.0-b70.1`) is unchanged.

- **Sparse (QSA) code.** `v0.32/patches/vllm-qsa/` grows from 26 to 32 patches (0027–0032), authored `Lumnus`; fork branch
  `b70/v0.32.0-qsa-release` (`be6a6e7d33`) on `Lumnus/vllm`; `v0.32/scripts/check-series.sh --qsa` checks it. The
  deterministic top-k (`native_det`) repairs ties without a host synchronisation and without a full-width sort (0027), as a
  fused Triton select (0031, `B70_QSA_TOPK_SELECT_IMPL=triton`); the faster scorer launch configurations from the sweeps
  are the defaults on XPU (0029, `B70_QSA_SCORE_DEFAULTS=legacy` restores the old ones); top-k once per verify step and
  skip-when-fits are in the series, off by default (0028); CPU tests (0030); profiler buckets (0032).
- **Fix.** The b70.2 sparse build ran out of GPU memory on rank 0 (about 800 MiB, inside the tie repair) during 131K
  prefills under load. The b70.3 build does not make that allocation and passed a stress stage of repeated 131K prefills
  plus 8-stream decode with 0 out-of-memory and 0 device errors.
- **Numbers**, same harness and window as dense ([docs/measurements/v0.32-qsa.md](docs/measurements/v0.32-qsa.md)): decode
  at 1 stream 0.90× dense (153.1 against 169.3 tok/s), 4 concurrent 1.00× (306 tok/s), 8 streams 0.94× (569 against 605
  tok/s), prefill 0.90× at 8K and 32K and faster than dense at 131K (4,916 against 4,641 tok/s); GPU KV 342,144 against
  397,309 tokens (with the patched plugin; dense default 388,872). b70.2 was 0.74 / 0.53 / 0.46× with 329,287 KV tokens;
  those numbers are kept in the measurements doc, labelled superseded. Beyond 2K tokens KL 0.0224 against dense (dense's
  own run-to-run floor about 0.013–0.014); identical to dense up to 2K.
- **Engine files.** `v0.32/engines/qsa-serve.env` gains `B70_QSA_TOPK_SELECT_IMPL=triton`; its header and `qsa-serve.args`'
  header are updated. `serve-config-qsa.json` is unchanged.
- **Not shipped as on:** top-k once per verify step (12 % slower at 4 streams, KL slightly worse), skip-when-fits (no
  measurable gain), graph capture of the sparse path (`B70_QSA_EAGER_BREAK=0`: fidelity degrades beyond 2K tokens, do not
  serve it).
- **Dense engine args: DFlash's trailing-block drop is off** (`"disable_eagle_block_drop": true` in
  `v0.32/engines/serve.args`). On this tree the DFlash2 draft KV group is not recognised as the draft's, so vLLM's
  EAGLE trailing-block drop applied to every group. Measured back to back on the dense configuration (one run each):
  agent-session replay 232 s against 266 s, median decode 49.4 against 40.7 tok/s, TTFT p50 1.76 against 2.03 s,
  0 preemptions against 1; gates and the greedy token compare pass; GPU KV −1.6K tokens; draft acceptance unchanged
  (5.02 against 5.07). The numbers tables were taken before this change. Not applied to `qsa-serve.args`: not yet run on
  the sparse engine.

## 0.32.0-b70.2 — 2026-10-26: sparse attention (QSA) variant, working and experimental

Adds a second attention variant to the 0.32 line next to the dense release. The dense series (`v0.32/patches/vllm/`, 49
patches, fork tag `v0.32.0-b70.1`) is unchanged.

Repository change shipped in this release:

- **The default recipe is now the stock-plugin configuration.** `v0.32/engines/serve.env` is the configuration without
  expandable segments and without the patched oneCCL plugin; the patched-plugin configuration moved to
  `v0.32/engines/serve-plugin.env` (formerly `serve.env`; the old `serve-no-plugin.env` is now `serve.env`). No variable
  line changed in either file.
- Why: the plugin buys ~2 % GPU KV (397,309 against 388,872 tokens) with speed within noise, and costs a hand-built `.so`
  tied to the pip oneCCL 2022.1.2 wheel, five `CCL_SYCL_*_TMP_BUF=1` buffer copies, and a second thing to rebuild on every
  torch/oneCCL bump.
- The default flips back to `serve-plugin.env` once uxlfoundation/oneCCL#229 ships in a oneCCL release that torch pins.

The sparse variant:

- **Code.** `v0.32/patches/vllm-qsa/`: 26 patches on top of the dense series, authored `Lumnus` (0001 is the
  vllm-project/vllm#60602 cherry-pick). Fork branch `b70/v0.32.0-qsa-release` (`4acd39891b`) on `Lumnus/vllm`.
  `v0.32/scripts/check-series.sh --qsa` checks the directory against it. QSA on XPU: block top-k, sparse GQA and
  side-cache kernels, fused prepare, `attn2` (a Triton sparse-attention kernel with no register spills on Xe2), a
  deterministic native top-k (`native_det`), a launch-override hook and sweep tools, 148 side-cache state-contract
  tests, and fixes the bring-up exposed (Triton 3.8.0 refuses `tl.dot` N < 16; the torch top-k ran out of memory on a
  long prefill).
- **Engine files.** `v0.32/engines/qsa-serve.args`, `qsa-serve.env` (the patched-plugin configuration plus
  `B70_QSA_ATTN_IMPL=attn2` and `B70_QSA_TOPK=native_det`) and `serve-config-qsa.json` (the dense config plus the five
  `indexer_*` keys).
- **Numbers**, same harness as the dense results ([docs/measurements/v0.32-qsa.md](docs/measurements/v0.32-qsa.md)):
  0.74–0.81× dense on prefill and one-stream decode, 0.46–0.53× at 4–8 streams, KV 329,287 against 397,309 tokens.
  Gates passed (needles 3/3, sanity 5/5, burst 0/150). Identical to dense up to 2K tokens; beyond 2K about 1.7–2× the
  engines' own run-to-run KL on a 12K-token document. Kernel level: `attn2` matches Intel's SYCL kernel on decode and
  beats it on prefill to 131K.
- **Status.** Working, experimental, slower than dense on multi-stream. Not run for days, no task-level comparison with
  dense. Not shipped: the engine-definition and card-sweep scripts we drove the measurements
  with (they run on our own host).
- `v0.32/README.md`: the "in preparation" placeholder is now the section [Sparse attention (QSA)](v0.32/models/qwen38-flash-next/meta.md#sparse-attention-qsa)
  (anchor renamed from `#sparse-attention-qsa-in-preparation`).

## Repository changes — 2026-10-25: attribution and scope of the headline numbers and gates

Documentation only; no series change, no new tag.

- **The cover table has the 0.30 perf line column** (DFlash2 K=7, 8 slots on vLLM v0.30.0, taken from
  `v0.32/README.md`). The gain over 0.30.0-b70.2 comes from DFlash2 K=7, async scheduling, 4096-token prefill chunks
  and bundles A/B, which also run on 0.30; 0.32 against the 0.30 perf line is within noise (it matches).
- **Speed numbers are greedy** (temperature 0, thinking off), where the K=7 greedy-verify path is active; stated on the
  cover and in `v0.32/README.md`.
- "The line we serve since 2026-10-24" → "the configuration validated on 2026-10-24": what the engine runs moves
  between this configuration and fallbacks during development windows.
- **The greedy token compare is described as what it tested**: no garbage markers, first divergences at small
  baseline margins. It did not check that the candidate chose the baseline's runner-up token at a divergence, and it
  compared only the overlapping prefix (the release's final captures are full length). It is not an equivalence check;
  a stricter compare will re-run these rows. MMLU / TruthfulQA paired tests: no difference detected, not an equivalence
  bound.
- Minachist: the "max |diff| 0" repack check was measured with the unpublished script `tools/minachist_repack.py` was
  ported from; the port is tested on synthetic tensors only, not yet on the real checkpoint (README, PROVENANCE, the
  tool's docstring).

## Repository changes — 2026-10-25: documentation reconciled to the current line

Documentation only; no series change, no new tag.

- **The cover describes the current line.** README's first table is now 0.32.0-b70.1 (vLLM v0.32.0, torch 2.14.0+xpu,
  triton 3.8.0+xpu, oneAPI 2026.1.1, oneCCL 2022.1.2 with the rebuilt legacy plugin, vllm-xpu-kernels
  `release/0.1.15.4` + our three PRs; DFlash2 K=7, async scheduling, 8 sequences, 397K GPU KV tokens, the private
  pinned CPU tier). Before, it still listed the 0.30 stack (vLLM v0.30.0, torch 2.13, MTP k=3, 16 sequences, ~311K)
  as if current. The 0.30.0-b70.2 line has a row of its own and a section labelled as the stable fallback; its
  endurance, validation, models and KV-tier numbers are labelled as 0.30 measurements.
- **"What we serve" corrected** wherever it still pointed at the 0.30 recipe or its kernels (`docs/kernels.md`,
  `docs/weights.md`, `docs/measurements/`, `engines/README.md`, engine file comments, PROVENANCE §6/§7).
- **Dense and sparse attention.** The 0.32 release is the dense variant; `v0.32/README.md` has a placeholder section
  for the sparse (QSA) variant and the paths it will use.
- `v0.32/README.md`: attention backends (target on the XPU default FlashAttention, drafter on `TRITON_ATTN`), KV and
  software rows; triton named as 3.8.0+xpu. `docs/known-issues.md` and `docs/switches.md` say which line they cover.
  `docs/switches.md`: the serving recipes set the presence penalty to 0, not 0.7.

## Repository changes — 2026-10-25: Intel AutoRound recommended, Minachist experts supported

Documentation and one tool; no series change, no new tag.

- **Intel AutoRound is the recommended weights**, AWQ the validated alternative. Intel's endurance on the 0.30.0-b70.2
  recipe (MTP k=3, 16 slots, 128 GiB CPU KV tier, kernels b70.3): about 15 days of serving from 2026-10-07 to 2026-10-23,
  the longest engine 8 days 6 h (40,912 requests, 0 errors, 0 aborts). One crash: on 2026-10-13 a card returned
  `UR_RESULT_ERROR_DEVICE_LOST` during a long memory-store re-ingest; host RAM was not the cause (31 GiB free); cause
  unknown; restarted the same day, not recurred. README (Validation status, Endurance, Models), `docs/weights.md`,
  `docs/known-issues.md` item 12, `docs/measurements/b70.2.md`, `engines/` comments and `PROVENANCE.md` updated; every
  "Intel verification in progress" note removed.
- **Correction:** the ~23 GiB host-RAM figure for Intel (against ~39 GiB for AWQ) was not the model; Intel leaves
  39.8 GiB at ready, the same as AWQ.
- **Minachist experts, a supported variant:** `Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound` branch `new-main` @
  `411505aa` (qwen-community-1.0), routed experts repacked into Intel's checkpoint by the new `tools/minachist_repack.py`.
  Measured on the 0.30 perf line against Intel: knowledge, long context and agent smoke within noise, coding 120 vs
  125 of 150 (n.s.), one-stream decode ~7 % slower. We keep Intel by preference.

## 0.32.0-b70.1 — 2026-10-25: vLLM 0.32, DFlash2 K=7

What we serve since 2026-10-24, published as a second line next to 0.30.0-b70.2 (unchanged). Everything is in `v0.32/`.

- **Code.** Fork branch `b70/v0.32.0` fast-forwarded to `103f2f2eb9`, tag `v0.32.0-b70.1`: 49 commits on vLLM v0.32.0
  (`5df63a7203`): wu1ff's 1.0.5 chain (15 patches, unchanged) and his notice; our 0.30 layers ported (INT8 PLE from NVMe
  on wu1ff's host-prepared PLE, chat defaults, dense-QSA indexer skip, offload trace/junction/backstep/guard, step trace,
  tests); vllm-project/vllm#63564 and #63566 cherry-picked (they replace our 0029b and 0014b); the 0.32 fixes
  B70-V32-PR2, -0053 (FP8 head), -03 (host PLE ids), -07 (HC ordered GEMM ≤ 32 rows), -08 (no prefill HC graphs), -RC1
  (private pinned KV offload pool), plus diagnostics and rejected arms kept off. Exported to `v0.32/patches/vllm/`;
  `v0.32/scripts/check-series.sh` checks it against the tag.
- **Serving shape.** Intel AutoRound weights, incoai DFlash2 drafter with 7 draft tokens, async scheduling, 8 sequences,
  prefill chunks of 4096, util 0.89, FULL_AND_PIECEWISE graphs at multiples of 8, 128 GiB CPU KV tier in a private
  pinned pool (no `/dev/shm` requirement), INT8 PLE from NVMe, compute-runtime copy-through-lock off, stock
  oneCCL plugin, no expandable segments (`v0.32/engines/serve.{args,env}`); the expandable-segments configuration with the
  oneCCL legacy plugin rebuilt with uxlfoundation/oneCCL#229 is `serve-plugin.env`.
- **Kernels.** vllm-xpu-kernels `release/0.1.15.4` + #658, #660 and #662 (we serve an earlier pinned-only form of #662),
  oneDNN on; wu1ff's Python hunks and
  B70-V32-05/06 in `v0.32/patches/vllm-xpu-kernels-py/`; wu1ff's native libraries built from his 1.0.5 sources, with
  `libgdn_index64.so` carrying #660 (`v0.32/patches/libgdn-index64/`). Build recipe in `v0.32/README.md`; no wheel
  attached.
- **Runtime.** torch 2.14.0+xpu, oneAPI runtime 2026.1.1, oneCCL 2022.1.2, compute-runtime 26.35.39758.10.
- **Measured** against 0.30.0-b70.2 with one harness: 1 stream 107.5 → 169.3 tok/s, 8 streams 370 → 605, prefill
  +51…57 %, KV ~311K → 397K tokens, agent replay 401 → 268 s; quality not significantly different at every step.
  Not yet: a multi-day run on 0.32, AWQ weights on 0.32, a container image.
- **Upstream.** README section listing the six fixes posted on 2026-10-25 (vllm-xpu-kernels #658, #660, #662; vLLM
  #63564, #63566; oneCCL #229). Credits: incoai and Niluthon added; NOTICE item 6.

## 0.30.0-b70.2 — 2026-10-07: the stable set

At release AWQ serving was fully validated and Intel's multi-day run had just started (it has since finished; see
the 2026-10-25 repository changes above). The configuration we served from 2026-10-04: the AWQ W4A16
checkpoint (wtdcode) with MTP (3 draft tokens) and a 128 GiB CPU KV tier, on our vllm-xpu-kernels 0.1.14.1+b70.3. It ran
3 days 16 h in one engine with 0 errors and 0 aborts (README, Endurance). Intel's AutoRound checkpoint on the same code
and flags is the alternative; a multi-day stability run of it started on 2026-10-07.

**One branch, the model as a parameter.** Source: fork branch `b70/v0.30.0-stable`, tag `v0.30.0-b70.2` (35 commits on
v0.30.0 `ced6857a`: 33 patch files + 3 test files). It consolidates the two working lines we tested on
(`b70/v0.30.0-mtp0020` for AWQ, `b70/v0.30.0-intel` for Intel) into one, with one copy of each fix (B70-0030 on the AWQ
line was the same change as 0029b). The engine recipe takes the model as a parameter: `engines/s16-kv128-mtp3.env` +
`engines/models/awq.env` or `engines/models/intel-autoround.env`. The release branch is not yet booted on a GPU; what
differs from what AWQ ran is listed in docs/known-issues.md item 12.

Compared with the working lines, the release branch has every commit rewritten with a neutral author and committer
(`Lumnus`; wu1ff's and the upstream backport author's commits keep their authors), commit messages without lab
references, the debug flag files under `/run/b70-probes/` instead of a lab path, and three code comments made neutral.
The code is otherwise byte-identical. `b70/v0.30.0`, `b70/v0.30.0-intel` and `b70/v0.30.0-mtp0020` keep their history
and are not updated any more.

The series (0020–0032):

- 0020: the MTP loaders skip `self_attn.indexer` tensors on dense-QSA configs (0019 for the draft model).
- 0021: MTP draft prefill with FULL graphs only on XPU (`B70_MTP_DRAFT_PREFILL_NO_PIECEWISE=1`).
- 0022: port of vllm-project/vllm#55506 (Mamba spec-decode block tables by request slot). It did not fix the MTP NaN
  it was ported for; it is kept because the served tree carries it.
- 0023, 0023c/0024t: debug tools (GDN op mode switch, NaN tracers, a null-block probe), off by default.
- 0026: MTP never schedules a partial draft list on XPU (structured output trimmed drafts and crashed the stock 0.1.14
  GDN kernel). On by default; harmless with kernels that carry #600.
- 0027: wu1ff's `libgdn_index64.so` is optional; without it the stock GDN op runs (64-bit with our kernels).
- 0028: the Qwen4Exp PLE embedding accepts Intel's AutoRound (INC) quantization config. The only change the Intel
  checkpoint needs.
- 0029, 0029b: prompt logprobs in 128-row chunks (V1 and Model Runner V2). Full-vocab logprobs spilled VRAM into host
  RAM. 0029 patches the V1 runner, which this model does not use; 0029b is the one that acts.
- 0031: CPU→GPU KV loads straight from the pinned offload pool (`B70_OFFLOAD_H2D_DIRECT=1`), for kernels without
  B70-K1. GPU-tested; gated off in what we serve.
- 0032: the 0027 fallback warning is logged once, not every 2 s (a log-only change; on the Intel trial since 2026-10-07).

Kernels: `patches/vllm-xpu-kernels/b70.3/` is the series of `Lumnus/vllm-xpu-kernels` tag `v0.1.14.1+b70.3` on
upstream 0.1.14.1: the 64-bit conv-state offset, upstream #600 #564 #563 #578 #586, and B70-K1 (direct H2D copies from
pinned sources). It replaces wu1ff's closed `libgdn_index64.so` on the stable line. Regenerate with
`git format-patch 0.1.14.1..v0.1.14.1+b70.3` in a clone of the fork; applied with `git am` on upstream tag `0.1.14.1`
it gives the tag's tree exactly. The `From:` author line of our own kernel commits is set to `Lumnus` in the exported
files (the tree is unaffected).

Repository: README (setup table, validation status, endurance, models ranked AWQ > Intel > devan with throughput per
concurrency, the host-RAM KV tier, credits last), `engines/s16-kv128-mtp3.env` with `engines/models/{awq,intel-autoround}.env`
(`run-example.sh` reads `MODEL=<name>`), `serve-s16-mtp3.args`, `serve-config-intel-autoround.json`,
`tools/intel_snapshot.py`, `docs/measurements/b70.2.md`; `scripts/export-series.sh` defaults to `b70/v0.30.0-stable`;
`scripts/make-tree.sh` maps `0.30.0-b70.1` to `b70/v0.30.0` and later releases to the fork tag `v<release>` (else
`b70/v0.30.0-stable`); `image/verify-overlay.sh` carries the final hashes of the 11 files 0020–0032 touch. Every file in
`patches/vllm/` and `patches/vllm-tests/` is re-exported (new commit ids and author lines; diffs of 0001–0022 and
0028 unchanged). The b70.1 files stay reproducible from this repository's b70.1 commit.

Release assets: the vllm-xpu-kernels `0.1.14.1+b70.3` wheel (manylinux_2_28 x86_64, sha256 `f104a3e5…d589`).

Not done for b70.2: an image build (it needs the kernel wheel and the removal of wu1ff's `libgdn_index64.so` from the
Dockerfile); a GPU boot of the release branch itself.

## 0.30.0-b70.1 — pre-release (2026-09-30, updated 2026-10-01)

First public release of the series. Image not yet built. No tag yet; the first public commit of this repository
(2026-09-30) carried 0001–0014f, and 0018/0019 were added on 2026-10-01.

Repository changes on 2026-10-01, besides 0018/0019:

- README: a navigation map, the two lines (stable vs edge), work in progress.
- `engines/awq-s16-kv128-chunked.env`, `serve-s16.args`, `serve-config-awq.json`: the configuration we serve (AWQ
  weights, 16 slots, 128 GiB CPU KV tier with 0018, INT8 PLE on NVMe, 0014jb). `run-example.sh` reads `MODEL_DIR`,
  `SERVE_ARGS` and `SERVE_CONFIG` from the engine file.
- `tools/awq_snapshot.py`: the AWQ snapshot (filtered weight index) and serve config.
- `docs/weights.md` (devan vs AWQ), `docs/kernels.md` (our vllm-xpu-kernels builds), `docs/known-issues.md`.
- `scripts/export-series.sh` pins `core.abbrev=7`, so a full vLLM clone reproduces the published patch files.

The series and the image:

- Base: vLLM v0.30.0 XPU image (`vllm/vllm-openai-xpu@sha256:fc0e112a…`), vllm-xpu-kernels 0.1.14.
- 0001–0005: wu1ff's B70-LLM-Controller Qwen3.8-Flash-Next pack 1.0.0, re-derived as source patches; byte-identical
  to wu1ff's files. P6/P7 binaries copied from wu1ff's image by digest.
- 0006: direct-to-pinned PLE load (`B70_PLE_DIRECT_PINNED`).
- 0007: FP8 PLE table (`B70_PLE_FP8`).
- 0008: INT8 per-row PLE table (`B70_PLE_INT8`), with `tools/build_int8_ple.py`.
- 0009: per-effort thinking budgets and a server default presence penalty.
- 0010: server default repetition stop.
- 0012: `reasoning.effort` alias, env-gated.
- 0013, 0013b: INT8 PLE table served from NVMe, native reader, optional prefill lookahead.
- 0014a–f: KV-offload trace, hybrid junction heal, GDN backstep, #56795 guard, vllm#51787 backport (gated).
- 0018: the pinned CPU KV-offload pool is split into power-of-two chunks when one pinned allocation is too large
  (Level Zero refuses a single host allocation of ~31 GiB or more; torch 2.13 then segfaulted at boot).
- 0019: dense-QSA configs skip `self_attn.indexer` tensors that some checkpoints (e.g. an AWQ export) still ship.
- All switches use the `B70_` prefix. The INT8 table format tag stays `lumnus-ple-int8-rowscale/v1`.
- `patches/vllm-xpu-kernels/0001`: the 64-bit conv-state offset fix as source. Not used by the image (torch 2.13);
  built into our vllm-xpu-kernels 0.1.15.4+b70.1 for the edge line (docs/kernels.md).
