# Changelog

Versions are `<vLLM base>-b70.<N>`; N increases whenever the patch series changes.

## 0.30.0-b70.2 — 2026-10-07: the stable set

AWQ serving fully validated; Intel verification in progress. The configuration we serve since 2026-10-04: the AWQ W4A16
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
