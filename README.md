# b70-flash-next

vLLM for **Qwen3.8-Flash-Next on 4× Intel Arc Pro B70**. This page describes the current line, **0.32**: release
**0.32.0-b70.2** (dense unchanged from b70.1; adds sparse). The line is vLLM v0.32.0 with wu1ff's 1.0.5 B70 chain, our patches on top, DFlash2 speculative decoding and a vllm-xpu-kernels
build carrying our upstream fixes. Its files are in [`v0.32/`](v0.32/README.md), which has the full build recipe and
measurement method. The previous line, **0.30.0-b70.2** (vLLM v0.30.0), stays as the stable fallback; its files are
the rest of this repository root (`patches/`, `engines/`, `image/`), and its page is
[docs/line-0.30.md](docs/line-0.30.md). Every number on this page names the line it was measured on.

The 0.32 line has two attention variants: **dense** (the release below: the checkpoint's QSA indexer is dropped and
the full-attention layers attend over the whole context) and **sparse** (QSA with the indexer kept; **working,
experimental, slower than dense on multi-stream**; added in 0.32.0-b70.2, see
[v0.32/README.md](v0.32/README.md#sparse-attention-qsa)). Both are listed as working.

**Latest, 2026-10-26: 0.32.0-b70.2 adds the sparse (QSA) variant of the 0.32 line, working and experimental.** Same
harness as the dense numbers: 0.74–0.81× dense on prefill and one-stream decode, 0.46–0.53× at 4–8 concurrent
streams, GPU KV 329K against 397K tokens for dense with the optional oneCCL plugin (`serve-plugin.env`; the default
`serve.env` has 389K); identical to dense up to 2K tokens, about 1.7–2× dense's own run-to-run KL beyond that on a
12K-token document. The dense series is unchanged and stays the recommended configuration when throughput matters.
[Measurements](docs/measurements/v0.32-qsa.md) · [section](v0.32/README.md#sparse-attention-qsa).

## Current line: 0.32.0-b70.2 (dense unchanged from b70.1; adds sparse)

| | |
|---|---|
| release | [`v0.32.0-b70.2`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.32.0-b70.2), 2026-10-26: dense unchanged from [`v0.32.0-b70.1`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.32.0-b70.1) (2026-10-25; the dense configuration validated on 2026-10-24); adds sparse |
| code | vLLM v0.32.0 + wu1ff's 1.0.5 chain + our layers: fork tag [`v0.32.0-b70.1`](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.1) (`103f2f2eb9`), exported to [`v0.32/patches/vllm/`](v0.32/patches/vllm/) (49 patches) |
| model | Qwen3.8-Flash-Next W4A16, **Intel AutoRound recommended** (the only build measured on 0.32); AWQ is validated on the 0.30 line only; Minachist's recalibrated experts are a supported variant, measured on the 0.30 perf line ([Models](#models)) |
| GPUs | 4× Intel Arc Pro B70 (32 GB each), tensor parallel 4 + expert parallel |
| decoding | DFlash2 speculative decoding, 7 draft tokens (incoai's drafter; the drafter on `TRITON_ATTN`, the target on vLLM's XPU default, FlashAttention); async scheduling; 8 sequences; 4096-token prefill chunks; 262K context; GPU memory fraction 0.89 |
| KV cache | GPU pool 388,872 tokens with the default `serve.env` (397,309 with the optional oneCCL plugin, `serve-plugin.env`) + **128 GiB in host RAM** (CPU tier, a private pinned pool per rank; no `/dev/shm` requirement) |
| PLE n-gram table | **INT8, served from NVMe** (8 GiB pinned row cache instead of 48.9 GiB in RAM) |
| software | torch 2.14.0+xpu, triton 3.8.0+xpu, oneAPI 2026.1.1, compute-runtime 26.35.39758.10, oneCCL 2022.1.2 (stock plugin; an optional legacy plugin rebuilt from oneCCL #229 adds ~2 % KV), vllm-xpu-kernels `release/0.1.15.4` + our three PRs ([v0.32/README.md](v0.32/README.md#software)) |
| tested host | AMD EPYC 7402 (24C/48T) on an ASRock Rack ROMED8-2T, 256 GB RAM, WD_BLACK SN850X 4 TB NVMe |
| status | gates, quality, speed and an agent-session replay measured; no multi-day run yet ([known issues](#known-issues-032)) |
| sparse variant | QSA add-on (26 patches on top of the dense series), added in 0.32.0-b70.2: working, experimental, slower than dense on multi-stream ([below](#sparse-attention-qsa-experimental)) |
| stable fallback | [`0.30.0-b70.2`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2), multi-day validated ([below](#the-030-line-stable-fallback)) |

## Numbers (0.32.0-b70.1)

Three configurations, same harness, same host, one after another, on an idle engine. Every cell is the Intel
AutoRound build.

| | 0.30.0-b70.2 (MTP k=3, 16 slots) | 0.30 perf line (DFlash2 K=7, 8 slots) | **0.32.0-b70.1** (DFlash2 K=7, 8 slots) |
|---|---|---|---|
| decode, 1 stream | 107.5 tok/s | 166 tok/s | **169.3 tok/s** |
| decode, 4 concurrent | 295 tok/s total | 287 tok/s total (repeat 296) | **306.2 tok/s** |
| decode, 8 streams | 370 tok/s total | 588 / 601 tok/s total | **605.5 tok/s** |
| prefill 8K / 32K / 131K | 4,052 / 3,833 / 2,953 tok/s | 6,125 / 6,092 / 4,648 tok/s | **6,305 / 6,114 / 4,641 tok/s** |
| GPU KV cache | ~311K tokens | 406K tokens | **389K tokens** (397K with the optional oneCCL plugin) |
| agent-session replay (243 requests): wall time, TTFT p50 / p90 | 401 s, 2.90 / 6.57 s | 302 s, 2.25 / 4.85 s | **268 s, 2.13 / 4.71 s**, 0 preemptions |
| MMLU 300 / TruthfulQA 200 | 259 / 177 | paired vs the left column: no difference detected | paired vs the 0.30 perf line: no difference detected |
| host MemAvailable while serving (256 GB host, ~20 GB of other work on it) | — | ~44–46 GiB | **35–39 GiB** |

How to read it:

1. **The gain over 0.30.0-b70.2 is not the version bump.** It comes from DFlash2 K=7, async scheduling, 4096-token
   prefill chunks and our bundles A/B, which also run on 0.30 (the middle column). 0.32 against that 0.30 perf line is
   within noise: it matches, it does not win. Read differences under ~5 % as noise.
2. **Speed rows are greedy** (temperature 0, thinking off), where the K=7 greedy-verify path is active. Requests that
   sample do not take that path; the served default temperature is 0.7, and no sampled speed number has been measured.
3. **Single runs** unless noted. The 0.32 8-stream number is one clean run (a second overlapped other traffic).
4. **Which engine file.** The 0.32 decode and prefill rows were taken with `serve-plugin.env`; the default `serve.env`
   measured within noise of them. The replay ran on `serve.env`.
5. **Quality** is a paired McNemar test on the same items at every step; with one or two discordant pairs this is
   not an equivalence bound, and differences of a few points would not be detected.
6. **Not comparable with the 0.30 per-build tables** in [docs/line-0.30.md](docs/line-0.30.md#models): those use a
   different harness (Intel at one stream reads 90.4 tok/s there, 107.5 here for a closely matching configuration).

Gates on 0.32: burst 0/150, structured output clean, a mixed long/short probe 7/7, and a greedy token compare with no
garbage and first divergences at small baseline margins (not an equivalence check). How each row was taken, long
context, and why our first 0.32 build was ~25 % slower: [v0.32/README.md](v0.32/README.md#numbers).

## Models

Intel AutoRound is recommended and is the only build measured on 0.32. The others:

| build | 0.32.0-b70.1 | measured on |
|---|---|---|
| **Intel AutoRound** — `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` | **recommended**; every number on this page | 0.32 (above); 0.30.0-b70.2 recipe, incl. 8 days 6 hours in one engine ([0.30 page](docs/line-0.30.md#intel-autoround)) |
| **AWQ** — `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125` | not booted (vLLM 0.32 accepts it natively; untested) | 0.30.0-b70.2 recipe: validated, 3 days 16 hours in one engine ([0.30 page](docs/line-0.30.md#awq--wtdcodeqwen38-flash-next-awq-w4a16--0939125-validated-alternative)) |
| **Minachist experts** — `Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`, branch `new-main` @ `411505aa`, repacked into Intel's checkpoint | supported variant, not booted | 0.30 perf line (DFlash2 K=7, 8 slots): within noise of Intel except one-stream decode, 154 vs 166 tok/s ([0.30 page](docs/line-0.30.md#minachist-experts--minachistqwen38-flash-next-int4-mixed-autoround-branch-new-main--411505aa-supported-variant)) |
| **devan** — `devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d` | reference, not booted | 0.30 line; most of [docs/measurements/](docs/measurements/README.md) |

Why Intel over AWQ (measured on the 0.30.0-b70.2 recipe): speed is a tie (Intel 2.8 % faster at one short stream,
level at 4 to 10 streams); MMLU, TruthfulQA, agentic coding and long context do not separate them; host RAM is the
same; both have clean multi-day runs; the rest is preference. The full comparison is on the
[0.30 page](docs/line-0.30.md#models).

**Intel on 0.32** needs the snapshot from `tools/intel_snapshot.py` (the checkpoint without its 102.4 GB BF16 PLE
shard, 78.8 GB to download) and [`engines/serve-config-intel-autoround.json`](engines/serve-config-intel-autoround.json)
as its `config.json` ([build step 7](v0.32/README.md#build-and-run)).

**Minachist's experts** (int4, group 128, symmetric, F16 scales) go into Intel's checkpoint with
`tools/minachist_repack.py convert <intel snapshot> <minachist download> <new snapshot>`: it repacks the experts from
compressed-tensors into Intel's GPTQ layout and keeps every other tensor Intel's, byte for byte. Minachist's own
INT6/INT8 layers are not used. The repack was exact (max |diff| 0 on every sampled layer) with the script this tool
was ported from; the port has been tested on synthetic tensors only, so run its `verify` mode on your snapshot. Engine
settings: the Intel files unchanged, only the snapshot path differs. License: qwen-community-1.0.

All builds serve the same INT8 PLE table from NVMe, built from devan's BF16 PLE file
([docs/ple-int8.md](docs/ple-int8.md)). Checkpoint details: [docs/weights.md](docs/weights.md).

## Host RAM and the KV tier (0.32)

Agent sessions resend long, growing prompts. The CPU tier keeps their KV cache in pinned host RAM and loads it back
instead of recomputing it. On 0.32.0-b70.1:

1. **Size.** `--kv-offloading-size 128`: 128 GiB, 32 GiB pinned per rank, in a private pool per rank
   (`B70_OFFLOAD_PRIVATE_PINNED=1`, B70-V32-RC1). It is copied to the GPU directly, and there is no `/dev/shm`
   requirement. Without RC1, vLLM 0.32 puts the tier in a shared `/dev/shm` region and needs `/dev/shm` ≥ 140 GiB.
2. **Measured.** The agent-session replay (six recorded sessions, 243 requests) against the tier: 268 s, TTFT p50 /
   p90 2.13 / 4.71 s, 0 preemptions. Without the private pool the same replay took 281 s with 2 preemptions.
3. **Host RAM.** The engine takes ~164 GiB after start (the 128 GiB tier, the 8 GiB PLE row cache, the four workers).
   On our 256 GB host MemAvailable stays at 35–39 GiB with ~20 GiB of other work on it. Pinned memory does not count
   against a container memory limit: watch the host's MemAvailable. With less RAM, lower `--kv-offloading-size`
   ([v0.32/README.md](v0.32/README.md#host-ram)).

The deeper measurement of the same mechanism (prompt-token reuse under real agent use, host-RAM vs GPU-pool split, load
bandwidth) was taken on the 0.30 line: [docs/line-0.30.md](docs/line-0.30.md#the-host-ram-kv-tier-030-measurements).

## Quick start (0.32, current line)

The short path; every step with its pinned commits and checks is in [v0.32/README.md](v0.32/README.md#build-and-run).
These are the commands we ran with our paths replaced by placeholders; the sequence has not been run end to end in this
form from a clean clone.

1. **Host.** 4× B70 on the xe driver, compute-runtime 26.35.39758.10, oneAPI DPC++ 2026.1.1 for the native builds.
2. **vLLM and the Python environment** (torch 2.14.0+xpu, oneCCL 2022.1.2 from pip):
   ```bash
   git clone -b v0.32.0-b70.1 https://github.com/Lumnus/vllm.git vllm-b70-0.32
   uv venv --python 3.12 /opt/b70/venv
   uv pip install --python /opt/b70/venv/bin/python --index-strategy unsafe-best-match \
       --extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://wheels.vllm.ai/xpu/ \
       -r vllm-b70-0.32/requirements/xpu.txt torch==2.14.0+xpu oneccl==2022.1.2 auto-round-lib==0.16.0 transformers==5.17.0
   VLLM_TARGET_DEVICE=xpu uv pip install --python /opt/b70/venv/bin/python --no-build-isolation -e vllm-b70-0.32
   ```
3. **Kernels.** vllm-xpu-kernels `release/0.1.15.4` + our PRs #660, #658, #662, built with oneDNN on, then our Python
   patches from [`v0.32/patches/vllm-xpu-kernels-py/`](v0.32/patches/vllm-xpu-kernels-py/) (build steps 3–4). No
   wheel is attached; the build takes ~26 min with flash-attention and MQA logits off.
4. **wu1ff's native libraries** from B70-LLM-Controller release `qwen3.8-flash-next-v1.0.5`, plus `libgdn_index64.so`
   with our patch, and the Level Zero peer-residency shim (build step 5).
5. **Weights.** Intel AutoRound @ `4c67bf68` (without the 102.4 GB PLE shard), devan's `ple_table_qwen4exp.pt`, the
   INT8 PLE table (`tools/build_int8_ple.py build`), the snapshot (`tools/intel_snapshot.py snapshot`), a serve dir
   with [`engines/serve-config-intel-autoround.json`](engines/serve-config-intel-autoround.json) as its `config.json`, and
   the drafter `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744` (build step 7).
6. **Serve.**
   ```bash
   set -a; . v0.32/engines/serve.env; set +a
   mapfile -t args < <(grep -v '^#' v0.32/engines/serve.args | sed '/^$/d' | sed 's/^\(--[^ ]*\) \(.*\)$/\1\n\2/')
   LD_LIBRARY_PATH=/opt/b70/venv/lib /opt/b70/venv/bin/vllm serve /models/qwen3.8-flash-next/serve-dir "${args[@]}"
   ```
   The venv's own oneAPI runtime must come first on `LD_LIBRARY_PATH`. The first start compiles cold; ours take ~4 min.
   `serve-plugin.env` instead of `serve.env` adds ~2 % GPU KV after building the optional oneCCL plugin (build step 6).

The chat template is the checkpoint's own in these commands. The serving prompt is yours to choose; ours is not in this
repository.

## What is in the 0.32 series

[`v0.32/patches/vllm/`](v0.32/patches/vllm/) is `git format-patch v0.32.0..v0.32.0-b70.1` of the fork, 49 commits:

| patches | what |
|---|---|
| 0001 | wu1ff's MIT license and pack notice |
| 0002–0016 | wu1ff's 1.0.5 chain for vLLM 0.32 (target port, DFlash2, C1 verify/target graphs, astra500 MoE/HC/prefill paths, serving fixes), taken unchanged with his authorship |
| 0017–0034 | our 0.30 layers ported: INT8 PLE from NVMe, per-effort thinking budget, repetition stop, effort alias, dense-QSA indexer skip, MTP draft prefill graphs, uniform-drafts guard, step trace, offload fixes, prompt-logprobs chunks, and their CPU tests |
| 0035–0037 | our upstream vLLM fixes #63564 and #63566 cherry-picked (replacing our earlier forms) |
| 0038–0049 | 0.32-specific: copy-path, hyper-connection, PLE-hash and FP8-head changes, the private pinned KV pool (RC1); a few off by default |

Beside it: [`v0.32/patches/vllm-xpu-kernels-py/`](v0.32/patches/vllm-xpu-kernels-py/) (the Python side of
vllm-xpu-kernels 0.1.15.4) and [`v0.32/patches/libgdn-index64/`](v0.32/patches/libgdn-index64/). The engine files
[`v0.32/engines/`](v0.32/engines/) explain each switch inline. Per-patch table and what is off by default:
[v0.32/README.md](v0.32/README.md#the-series).

## Known issues (0.32)

1. **No multi-day run yet.** The line has served since 2026-10-24 with planned stops for development windows.
2. **One hang in development**, before the copy-through-lock fix; it did not recur; cause unknown.
3. **GPU KV below the 0.30 perf line**: ~2 % with `serve-plugin.env`, ~4 % with the default `serve.env`; not yet
   attributed.
4. **The GPU idles ~3 ms per step in the PLE sync at one stream.** Open.
5. **Only Intel AutoRound was measured on 0.32.**
6. **0.30 items that still apply** where the code is shared: card hangs after a GT reset, liveness must generate,
   pinned memory escapes container limits ([docs/known-issues.md](docs/known-issues.md) items 1–3).
7. **oneCCL `LL=twoshots` corrupts output** on this stack; every oneCCL change needs the burst gate and a token compare.
8. **The kernels build is not attached**; build it yourself (Quick start step 3).

Details and workarounds: [v0.32/README.md](v0.32/README.md#known-issues-032).

## Sparse attention (QSA, experimental)

Added in release 0.32.0-b70.2 on top of the dense 0.32 line, which is unchanged. The checkpoint's QSA indexer is
kept, and the full-attention layers attend over the indexer's selection (a 2,048-token budget) with Triton kernels
instead of over the whole context. It boots on 4× B70, passes the gates and the long-context needles, and its output
matches dense up to the selection budget. It is slower than dense. Measured on 0.32 with the same harness and host,
single runs, Intel AutoRound, both variants with the patched oneCCL plugin:

| | dense (`serve-plugin.env`) | sparse QSA (`qsa-serve.env`) | sparse / dense |
|---|---|---|---|
| decode, 1 stream (tok/s) | 169.3 | 125.0 | 0.74 |
| decode, 4 concurrent (total tok/s) | 306.2 | 161.2 | 0.53 |
| decode, 8 streams (total tok/s) | 605.5 | 281.4 / 279.7 | 0.46 |
| prefill 8K / 32K / 131K (tok/s) | 6,305 / 6,114 / 4,641 | 5,096 / 4,635 / 3,456 | 0.81 / 0.76 / 0.74 |
| GPU KV cache (tokens) | 397,309 | 329,287 | 0.83 |

Use dense when throughput matters. Not measured for sparse: a multi-day run and a task-level quality comparison with
dense. Its files: [`v0.32/patches/vllm-qsa/`](v0.32/patches/vllm-qsa/), [`v0.32/engines/qsa-serve.args`](v0.32/engines/qsa-serve.args),
[`v0.32/engines/qsa-serve.env`](v0.32/engines/qsa-serve.env), [`v0.32/engines/serve-config-qsa.json`](v0.32/engines/serve-config-qsa.json).
How to build and run it, its known issues: [v0.32/README.md](v0.32/README.md#sparse-attention-qsa); fidelity and
what the measurement does not show: [docs/measurements/v0.32-qsa.md](docs/measurements/v0.32-qsa.md).

## Work in progress

1. **Sparse attention (QSA) on 0.32: working, experimental** (0.32.0-b70.2, [v0.32/README.md](v0.32/README.md#sparse-attention-qsa)). Open: multi-stream decode (0.46–0.53× dense: one top-k per verify step, a histogram top-k, the tuned scorer configurations as defaults), an in-engine profile, KV −17 %, a multi-day run, and a task-level quality comparison with dense.
2. **A stricter token compare**: the greedy compare above checks for garbage, not equivalence; the gate rows will be
   re-run with the stricter one.

The 0.30 line's open items: [docs/line-0.30.md](docs/line-0.30.md#work-in-progress-on-the-030-line-not-in-the-series).

## The 0.30 line (stable fallback)

**[`0.30.0-b70.2`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2), 2026-10-07**: stock vLLM
v0.30.0 + the series in [`patches/`](patches/), torch 2.13.0+xpu, vllm-xpu-kernels 0.1.14.1+b70.3, MTP speculative
decoding with 3 draft tokens, 16 sequences, ~311K GPU KV tokens + the same 128 GiB CPU tier.

Use it when you need a line with multi-day runs behind it. On its recipe: **Intel AutoRound** served about **15 days**
(2026-10-07 to 2026-10-23), the longest engine **8 days 6 hours** with 0 errors and 0 aborts, and one crash (a GPU
device-lost, cause unknown); **AWQ** ran **3 days 16 hours** in one engine with 0 errors. Its speed, on the harness of
the [Numbers](#numbers-0320-b701) table: 107.5 tok/s at one stream, 370 tok/s total at 8 streams, prefill 4,052 tok/s
at 8K.

Everything else about it (validation status, endurance tables, the per-build Models tables with the Intel-vs-AWQ
comparison, the KV-tier measurements, its quick start, its patch series, its open items):
**[docs/line-0.30.md](docs/line-0.30.md)**.

## Lines

| line | what | state |
|---|---|---|
| **current `0.32.0-b70.N`** | vLLM v0.32.0 + wu1ff's 1.0.5 chain + our layers (fork branch [`b70/v0.32.0`](https://github.com/Lumnus/vllm/tree/b70/v0.32.0), tag `v0.32.0-b70.1`, dense; sparse add-on on [`b70/v0.32.0-qsa-release`](https://github.com/Lumnus/vllm/tree/b70/v0.32.0-qsa-release)), release `v0.32.0-b70.2`, torch 2.14, vllm-xpu-kernels `release/0.1.15.4` + our three kernel PRs ([v0.32/](v0.32/README.md)) | **the configuration validated on 2026-10-24**; gates, quality and speed measured, no multi-day run yet. Dense and sparse (QSA) attention both working; sparse is experimental and slower than dense on multi-stream ([measurements](docs/measurements/v0.32-qsa.md)) |
| **stable fallback `0.30.0-b70.N`** | stock vLLM v0.30.0 + the series in `patches/` (fork branch [`b70/v0.30.0-stable`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0-stable), tag `v0.30.0-b70.2`; b70.1 was [`b70/v0.30.0`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0)), torch 2.13, vllm-xpu-kernels 0.1.14.1+b70.3 ([docs/line-0.30.md](docs/line-0.30.md)) | the multi-day validated line (Intel and AWQ); the fallback |
| edge `main-YYYYMMDD-b70.N` | vLLM main + the series (fork branch [`b70/main`](https://github.com/Lumnus/vllm/tree/b70/main)), torch 2.14, our 0.1.15.4 kernel build | **not recommended**: a progressive decode corruption under concurrency ([known issues](docs/known-issues.md) item 5) |

## Where to find what

| path / repo | what |
|---|---|
| [`v0.32/`](v0.32/README.md) | the 0.32 line: `patches/vllm/` (49 commits on v0.32.0, dense), `patches/vllm-qsa/` (the sparse add-on), `patches/vllm-xpu-kernels-py/`, `patches/libgdn-index64/`, `engines/` (`serve.args`, `serve.env`, `serve-plugin.env`; `qsa-serve.args`, `qsa-serve.env`, `serve-config-qsa.json`), `scripts/check-series.sh` (`--qsa` for the add-on), build recipe and numbers |
| [`docs/line-0.30.md`](docs/line-0.30.md) | the 0.30.0-b70.2 line: validation, endurance, per-build numbers, KV-tier measurements, quick start, series |
| `patches/vllm/`, `patches/vllm-tests/` | the 0.30 series 0001–0032, exported from the fork branch (one commit per patch on v0.30.0); CPU tests (never applied to the image) |
| `patches/vllm-xpu-kernels/` | `b70.3/`: our kernel series on vllm-xpu-kernels 0.1.14.1 (the 64-bit GDN conv-state offset, five upstream fixes, B70-K1); `0001-…` at the top: the b70.1 form of the 64-bit fix ([docs/kernels.md](docs/kernels.md)) |
| `patches/series.txt` | which fork commit goes to which file (0.30 series) |
| `scripts/` | `export-series.sh`, `check-series.sh` (patches/ == fork branch), `make-tree.sh` (a verified source tree); 0.30 line |
| `image/` | `Dockerfile`, `verify-overlay.sh`, wu1ff's pack files, `derive-serve-config.py` (the 0.30.0-b70.1 image recipe) |
| `engines/` | 0.30 line: environments and `vllm serve` flags ([engines/README.md](engines/README.md)); the two serve configs are shared with 0.32 |
| `docs/` | [0.30 line](docs/line-0.30.md) · [switches](docs/switches.md) · [weights](docs/weights.md) · [kernels](docs/kernels.md) · [known issues](docs/known-issues.md) · [offload fix](docs/offload-fix.md) · [PLE INT8](docs/ple-int8.md) · [PLE on NVMe](docs/ple-nvme.md) · [measurements](docs/measurements/README.md) |
| `tools/` | `build_int8_ple.py` (INT8 PLE table), `intel_snapshot.py` and `awq_snapshot.py` (snapshots + serve configs), `minachist_repack.py` (Minachist experts into the Intel snapshot), `repro_band.py` (reproduce the offload miss); used by both lines |
| [Lumnus/vllm](https://github.com/Lumnus/vllm) | the code: `b70/v0.32.0` (tag `v0.32.0-b70.1`, the 0.32 line, dense), `b70/v0.32.0-qsa-release` (the sparse add-on), `b70/v0.30.0-stable` (the release branch from b70.2 on), `b70/v0.30.0` (b70.1), `b70/main` (edge), `offload-hybrid-junction` (the offload fix in upstream style). `b70/v0.30.0-intel` and `b70/v0.30.0-mtp0020` are the two working lines that b70.2 consolidates; they are not updated any more |
| [Lumnus/vllm-xpu-kernels](https://github.com/Lumnus/vllm-xpu-kernels) | `b70/v0.1.14` (tag `v0.1.14.1+b70.3`, the 0.30 line's kernels), `b70/v0.1.15` (edge). The 0.32 line builds upstream `release/0.1.15.4` + our PRs instead ([v0.32/README.md](v0.32/README.md#build-and-run), step 3) |
| [Lumnus/b70-fan-control](https://github.com/Lumnus/b70-fan-control) | sibling repo: a user-space fan table for the B70 |

`PROVENANCE.md` records where every piece comes from, with hashes (§8 for 0.32). `CHANGELOG.md` lists releases.

## The code lives in the fork

Changes are made on the fork branch, then exported here: for 0.32 with `git format-patch v0.32.0..v0.32.0-b70.1`,
checked by `v0.32/scripts/check-series.sh`; for 0.30 with `scripts/export-series.sh`, checked by
`scripts/check-series.sh`. Both checks fail on any drift.

## Upstream

Fixes from this work, posted upstream on 2026-10-25 (open as of that day). The 0.32 release carries all six.

| issue → PR | what |
|---|---|
| vllm-xpu-kernels [#657](https://github.com/vllm-project/vllm-xpu-kernels/issues/657) → [#658](https://github.com/vllm-project/vllm-xpu-kernels/pull/658) | GDN `causal_conv1d` conv-state offsets in 64 bits |
| vllm-xpu-kernels [#659](https://github.com/vllm-project/vllm-xpu-kernels/issues/659) → [#660](https://github.com/vllm-project/vllm-xpu-kernels/pull/660) | GDN kernels skip rows whose state index is `NULL_BLOCK_ID` |
| vllm-xpu-kernels [#661](https://github.com/vllm-project/vllm-xpu-kernels/issues/661) → [#662](https://github.com/vllm-project/vllm-xpu-kernels/pull/662) | `swap_blocks_batch`: `snapshot_src=False` for pinned and registered host sources |
| vLLM [#63563](https://github.com/vllm-project/vllm/issues/63563) → [#63564](https://github.com/vllm-project/vllm/pull/63564) | Model Runner V2: bound prompt-logprobs chunks by `max_num_reqs` |
| vLLM [#63565](https://github.com/vllm-project/vllm/issues/63565) → [#63566](https://github.com/vllm-project/vllm/pull/63566) | KV offload: pin a shared-prefix junction when a sparse group cuts the offload hit |
| oneCCL [#228](https://github.com/uxlfoundation/oneCCL/issues/228) → [#229](https://github.com/uxlfoundation/oneCCL/pull/229) | legacy plugin: device USM that `msync()` sees as host takes the GPU path |

Issues filed the same day, without a PR yet: vLLM [#63620](https://github.com/vllm-project/vllm/issues/63620) (hybrid
models: padding rows of replayed graphs point recurrent state at the shared null block; a scratch-block fix is
offered) and [#63621](https://github.com/vllm-project/vllm/issues/63621) (XPU KV offload: the registered host region is
not recognised as USM, so every CPU→GPU load is staged; the caller change follows vllm-xpu-kernels #662).

## Credits

This work stands on other people's:

1. **wu1ff — [B70-LLM-Controller](https://github.com/wu1ff/B70-LLM-Controller)** (MIT). The Qwen3.8-Flash-Next B70
   pack: the XPU gate, the pinned-host PLE table, the HC down-GEMM K-split, the GDN spec-metadata fusion, the 64-bit
   GDN conv-state offset fix, the Level Zero peer-residency shim and the dense-QSA serve config. Patches 0001–0005 are
   wu1ff's changes re-derived as source diffs; two binaries are copied from wu1ff's public image by digest. The 0.32
   line is built on wu1ff's 1.0.5 chain for vLLM 0.32 (DFlash2 path, C1 verify/target graphs, the astra500 MoE,
   hyper-connection and prefill kernels), taken unchanged with his authorship and MIT notice. Thank you.
2. **Intel** — the [`Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
   weights we serve and recommend, made with [AutoRound](https://github.com/intel/auto-round);
   [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels), whose upstream fixes #600, #563, #564, #578 and
   #586 are in our 0.30 kernel build (by Guancheng Fu, Chaojun Zhang, Qiming Zhang and Tony Lin);
   [llm-scaler](https://github.com/intel/llm-scaler) and the XPU work in
   [vLLM #55068](https://github.com/vllm-project/vllm/pull/55068).
3. **wtdcode** — the [`Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16)
   weights, the validated alternative (we served them 2026-09-30 to 2026-10-07).
   **Minachist** — [`Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`](https://huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound),
   whose recalibrated experts are the supported variant.
4. **devan-carlin** — the [`Qwen3.8-Flash-Next-W4A16`](https://huggingface.co/devan-carlin/Qwen3.8-Flash-Next-W4A16)
   weights, its BF16 PLE table (every INT8 table here is built from it), and the early community vLLM port of
   the model to XPU (`devan-carlin/vllm@xpu-qwen4exp`).
5. **TSUMUGI-XE** — [intel/compute-runtime#968](https://github.com/intel/compute-runtime/issues/968), the analysis of
   Level Zero peer residency mirroring device memory into host memory, and the first shim that works around it.
6. **incoai** — the [`Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
   block-diffusion drafter (Apache-2.0) behind the 0.32 release's speculative decoding.
7. **Niluthon** — [qwen38-flash-tp2](https://github.com/Niluthon/qwen38-flash-tp2) (Apache-2.0), a two-card fork of
   our b70.2: the FP8 `lm_head`, the resident decode MoE op, the step trace and the `bench_det` harness came from there;
   the 0.32 FP8 head (B70-V32-0053) is a port of its 0053, and the 1- and 4-stream numbers in the 0.32 table use
   `bench_det`.
8. **DominikBucko** — [qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090), the
   two-card CUDA port. No code is taken from it; its published measurements and design notes (async scheduling and the
   Mamba state-block fix, prefill chunk sizes, the AVX2 cold-expert path) shaped which levers we measured.
9. **The Qwen team** — Qwen3.8-Flash-Next itself (license: qwen-community-1.0, see each checkpoint's card).
10. **[vLLM](https://github.com/vllm-project/vllm)**, on which everything here is built. Patch 0014e is a backport of
    vllm-project/vllm#51787; patch 0022 ports vllm-project/vllm#55506.

## License

Apache-2.0 (LICENSE). wu1ff's files and binaries keep their MIT license; the vLLM and vllm-xpu-kernels patches modify
Apache-2.0 code. Model weights are not included and keep their own licenses. See NOTICE.
