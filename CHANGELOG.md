# Changelog

Versions are `<vLLM base>-b70.<N>`; N increases whenever the patch series changes.

## Unreleased

- **The default recipe is now the stock-plugin configuration.** `v0.32/engines/serve.env` is the configuration without
  expandable segments and without the patched oneCCL plugin; the patched-plugin configuration moved to
  `v0.32/engines/serve-plugin.env` (formerly `serve.env`; the old `serve-no-plugin.env` is now `serve.env`). No variable
  line changed in either file.
- Why: the plugin buys ~2 % GPU KV (397,309 against 388,872 tokens) with speed within noise, and costs a hand-built `.so`
  tied to the pip oneCCL 2022.1.2 wheel, five `CCL_SYCL_*_TMP_BUF=1` buffer copies, and a second thing to rebuild on every
  torch/oneCCL bump.
- The default flips back to `serve-plugin.env` once uxlfoundation/oneCCL#229 ships in a oneCCL release that torch pins.

## 0.32.0-b70.2 — 2026-10-26: sparse attention (QSA) variant, working and experimental

Adds a second attention variant to the 0.32 line next to the dense release. The dense series (`v0.32/patches/vllm/`, 49
patches, fork tag `v0.32.0-b70.1`) is unchanged.

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
- `v0.32/README.md`: the "in preparation" placeholder is now the section [Sparse attention (QSA)](v0.32/README.md#sparse-attention-qsa)
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
