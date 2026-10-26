# b70-flash-next

vLLM for **Qwen3.8-Flash-Next on 4× Intel Arc Pro B70**. The current line, **0.32.0-b70.1**, is vLLM v0.32.0 with
wu1ff's 1.0.5 B70 chain, our patches on top, DFlash2 speculative decoding and a vllm-xpu-kernels build carrying our
upstream fixes; everything for it is in [`v0.32/`](v0.32/README.md). The previous line, **0.30.0-b70.2** (vLLM
v0.30.0), stays as the stable fallback; it is the rest of this repository root (`patches/`, `engines/`, `image/`).

The 0.32 line has two attention variants: **dense** (the release below: the checkpoint's QSA indexer is dropped and
the full-attention layers attend over the whole context) and **sparse** (QSA with the indexer kept; in preparation, see
[v0.32/README.md](v0.32/README.md#sparse-attention-qsa-in-preparation)).

## Current line: 0.32.0-b70.1 (dense attention)

| | |
|---|---|
| release | [`v0.32.0-b70.1`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.32.0-b70.1), 2026-10-25; the configuration validated on 2026-10-24 |
| code | vLLM v0.32.0 + wu1ff's 1.0.5 chain + our layers: fork tag [`v0.32.0-b70.1`](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.1) (`103f2f2eb9`), exported to [`v0.32/patches/vllm/`](v0.32/patches/vllm/) (49 patches) |
| model | Qwen3.8-Flash-Next W4A16, **Intel AutoRound recommended** (the only build measured on 0.32); AWQ is validated on the 0.30 line; Minachist's recalibrated experts are a supported variant ([Models](#models)) |
| GPUs | 4× Intel Arc Pro B70 (32 GB each), tensor parallel 4 + expert parallel |
| decoding | DFlash2 speculative decoding, 7 draft tokens (incoai's drafter; the drafter on `TRITON_ATTN`, the target on vLLM's XPU default, FlashAttention); async scheduling; 8 sequences; 4096-token prefill chunks; 262K context; GPU memory fraction 0.89 |
| KV cache | GPU pool 389K tokens (397K with the optional oneCCL plugin) + **128 GiB in host RAM** (CPU tier, a private pinned pool per rank; no `/dev/shm` requirement) |
| PLE n-gram table | **INT8, served from NVMe** (8 GiB pinned row cache instead of 48.9 GiB in RAM) |
| software | torch 2.14.0+xpu, triton 3.8.0+xpu, oneAPI 2026.1.1, compute-runtime 26.35.39758.10, oneCCL 2022.1.2 (stock plugin; an optional legacy plugin rebuilt from oneCCL #229 adds ~2 % KV), vllm-xpu-kernels `release/0.1.15.4` + our three PRs ([v0.32/README.md](v0.32/README.md#software)) |
| tested host | AMD EPYC 7402 (24C/48T) on an ASRock Rack ROMED8-2T, 256 GB RAM, WD_BLACK SN850X 4 TB NVMe |
| status | gates, quality, speed and an agent-session replay measured; no multi-day run yet ([known issues](v0.32/README.md#known-issues-032)) |
| **stable fallback** | [`0.30.0-b70.2`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2): vLLM v0.30.0, torch 2.13, vllm-xpu-kernels 0.1.14.1+b70.3, MTP k=3, 16 sequences, ~311K GPU KV tokens + the same 128 GiB CPU tier; multi-day validated on Intel and AWQ ([the 0.30 line](#the-030-line-stable-fallback)) |

Measured against the 0.30 fallback, same harness, same host, one after another:

| | 0.30.0-b70.2 (MTP k=3, 16 slots) | 0.30 perf line (DFlash2 K=7, 8 slots) | **0.32.0-b70.1** (DFlash2 K=7, 8 slots) |
|---|---|---|---|
| decode, 1 stream | 107.5 tok/s | 166 tok/s | **169.3 tok/s** |
| decode, 4 concurrent | 295 tok/s total | 287 tok/s total (repeat 296) | **306.2 tok/s** |
| decode, 8 streams | 370 tok/s total | 588 / 601 tok/s total | **605.5 tok/s** |
| prefill 8K / 32K / 131K | 4,052 / 3,833 / 2,953 tok/s | 6,125 / 6,092 / 4,648 tok/s | **6,305 / 6,114 / 4,641 tok/s** |
| GPU KV cache | ~311K tokens | 406K tokens | **389K tokens** (397K with the optional oneCCL plugin) |
| agent-session replay (243 requests) | 401 s | 302 s | **268 s** |
| MMLU / TruthfulQA | — | not significantly different | no difference detected (paired, at every step; not an equivalence bound) |

The gain over 0.30.0-b70.2 comes from DFlash2 K=7, async scheduling, 4096-token prefill chunks and our bundles A/B,
which also run on 0.30 (the middle column); 0.32 against that 0.30 perf line is within noise: it matches, it does not
win. Speed rows are greedy (temperature 0, thinking off), where the K=7 greedy-verify path is active; requests that
sample (production's default temperature is not 0) do not take that path.

Single runs on an idle engine; this harness is not the one behind the 0.30 per-build tables further down. Method, the
intermediate 0.30 perf line, and the caveats: [v0.32/README.md](v0.32/README.md#numbers). How to build and run it:
[v0.32/README.md](v0.32/README.md#build-and-run).

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
   patches from [`v0.32/patches/vllm-xpu-kernels-py/`](v0.32/patches/vllm-xpu-kernels-py/) (build steps 3–4).
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

## The 0.30 line (stable fallback)

**Release [`0.30.0-b70.2`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2), 2026-10-07.** Both
weight builds are validated on its code ([known issues](docs/known-issues.md) item 12): Intel AutoRound (8 days 6 hours
in one engine, 0 errors) and AWQ (3 days 16 hours, 0 errors). One code branch, the model is a serving parameter. It
runs from source trees; the vllm-xpu-kernels wheel is attached to the release, and no container image is built for
b70.2 ([CHANGELOG](CHANGELOG.md)). Everything from here to [Quick start](#quick-start-030-source-tree) is about this
line unless a row says otherwise.

| | |
|---|---|
| code | stock vLLM v0.30.0 + the series in [`patches/`](patches/) (fork branch [`b70/v0.30.0-stable`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0-stable), tag `v0.30.0-b70.2`) |
| decoding | MTP speculative decoding, 3 draft tokens; 16 sequences; 262K context; GPU memory fraction 0.88; `--no-async-scheduling`; cudagraph capture sizes at every multiple of 4 up to 64 (plus 256/512/1024) |
| KV cache | GPU pool ~311K tokens + 128 GB in host RAM (CPU tier) |
| PLE n-gram table | INT8, served from NVMe (as on 0.32) |
| software | torch 2.13.0+xpu, compute-runtime 26.35.39758.10, vllm-xpu-kernels 0.1.14.1+b70.3 |

### Validation status (0.30 line)

| | |
|---|---|
| **Intel AutoRound** (recommended) | **fully validated**: gates, quality, speed, and about **15 days of serving** (2026-10-07 to 2026-10-23), the longest single engine **8 days 6 hours** with 0 errors and 0 aborts ([Endurance](#endurance-030-line)). One crash in that time: a GPU device-lost, described there |
| **AWQ** | **fully validated** on an earlier branch of the same code: gates, quality, speed, and one engine running **3 days 16 hours** without an error ([Endurance](#endurance-030-line)) |
| the release branch [`b70/v0.30.0-stable`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0-stable) | not booted on a GPU as such. It is the union of the two lines we tested, with one copy of each fix. Intel's endurance ran on the release code except for the directory of the debug flag files; AWQ's on an earlier branch. Every difference is listed in [known issues](docs/known-issues.md) item 12 |

In day-to-day use on this line (an operator observation under real mixed agent load, not a benchmark), aggregate
generation peaked at about 500 tok/s and held at 300+ tok/s sustained; it depends on the jobs.

## Endurance (0.30 line)

### Intel AutoRound

The recommended setup (Intel AutoRound weights, MTP k=3, 16 slots, 128 GiB host-RAM KV tier, kernels b70.3) served
from 2026-10-07 16:52 UTC until 2026-10-23 06:50 UTC, when we took the engine for performance work: about 15 days
6 hours of serving in 15 days 14 hours, in four engine runs.

| engine run (UTC) | length | how it ended |
|---|---|---|
| 10-07 16:52 → 10-08 06:43 | 13 h 51 min | planned stop (an 8-minute test of another model) |
| 10-08 06:51 → 10-13 08:23 | 5 d 1 h 32 min | **the one crash**: a GPU device-lost (below) |
| 10-13 14:16 → 10-14 22:50 | 1 d 8 h 35 min | planned stop (the host was powered off for a disk change) |
| 10-15 00:26 → 10-23 06:50 | **8 d 6 h 24 min** | planned stop (start of the performance work) |

The 8-day run, from Prometheus:

| | |
|---|---|
| requests | 40,912 (39,392 ended on a stop token, 1,520 on the length limit, 0 on the repetition stop) |
| errors / aborts | **0 / 0** |
| prompt tokens processed | 80.6 million |
| tokens generated | 2.87 million |
| preemptions | 10 |
| host RAM | MemAvailable never below 53 GiB on our 256 GiB host |
| host-RAM guard stops, GPU errors | none |

**The crash, 2026-10-13 08:23 UTC.** One card returned `UR_RESULT_ERROR_DEVICE_LOST` in the middle of a decode step,
and the engine stopped on all four workers. A rebuild of our memory store was feeding the engine at the time: a
two-day re-ingest that kept several LLM calls in flight (7 were running, GPU KV cache 83 % full). It was not a memory
stop: host MemAvailable was 31 GiB, far above the guard. We do not know the cause; the driver's log of that moment
had rotated before it was read. The engine was restarted on the same configuration the same day, and the error has not
recurred.

### AWQ

The AWQ setup (AWQ weights, otherwise the same) ran as **one continuous engine for 3 days 16 hours** (2026-10-04
00:15 UTC to 2026-10-07 16:30 UTC). It ended only in a planned restart (a change of the default serving prompt);
nothing failed. Prometheus over the 86 h window:

| | |
|---|---|
| requests | ~52,500 (51,312 ended on a stop token, 1,179 on the length limit, 1 on the repetition stop) |
| errors / aborts | **0 / 0** |
| prompt tokens processed | 566 million |
| tokens generated | 39.4 million |
| preemptions | 134 |
| host-RAM guard stops, GPU errors | none |

The load on both was real mixed use, not a benchmark: interactive coding agents plus memory-engine ingests (on AWQ up
to 8 concurrent LLM calls with prompts up to ~22K tokens, host memory flat; on Intel's 8-day run up to 11 requests
running at once). These are runs on one host; they show the stack holding under that mix, not a guarantee for other
workloads.

## Models

Two calibrated 4-bit checkpoints run on this stack and both are validated; a third, round-to-nearest one is the
reference, and a fourth variant (Minachist's recalibrated experts in Intel's checkpoint) is supported. **We recommend
Intel AutoRound, and it is what we serve.** Only Intel has been measured on the current 0.32 line (its numbers are in
the Intel section below and in the table at the top). Everything else in this section, including the Intel-vs-AWQ
comparison, was measured on the 0.30.0-b70.2 recipe (MTP k=3, 16 slots, kernels b70.2/b70.3) with a different harness;
those numbers are much lower than the 0.32 line's because that recipe has no DFlash2, no async scheduling and 1024-token
prefill chunks, not because of the weights. The two calibrated builds are close; the reasons for the order:

1. **Speed is a tie, Intel slightly ahead at one stream.** Measured side by side, Intel decodes 2.8 % faster at one
   short stream (90.4 vs 87.9 tok/s) and 5 % faster at one 16K stream (89.8 vs 85.0), and the two are level at 4 to 10
   streams (within the ~10 % noise). The benchmarks we ran (MMLU, TruthfulQA, agentic coding, long context) do not
   separate them significantly; they sit 0.082 KL apart (top-1 agreement 90.2 %) and give the same answer on 97 % of
   items.
2. **Both have clean multi-day runs** ([Endurance](#endurance-030-line)): Intel 8 days 6 hours in one engine and about 15 days of
   serving with one crash (a GPU device-lost under a long memory-store re-ingest); AWQ 3 days 16 hours.
3. **Host RAM is the same.** An earlier reading of ~23 GiB left free with Intel against ~39 GiB with AWQ was not the
   model: at ready Intel leaves 39.8 GiB, the same as AWQ, and over its 8-day run MemAvailable never went below 53 GiB.
4. **Digit loops: too few to call.** In 150 coding samples Intel has 2 digit-run repetition stops, AWQ 0
   ([known issues](docs/known-issues.md) item 7).
5. **The rest is preference.** We run Intel day to day and keep it on a subjective preference; AWQ is a fully validated
   alternative with the same code and flags.

All of them serve the same INT8 PLE table from NVMe, built from devan's BF16 PLE file
([docs/ple-int8.md](docs/ple-int8.md)). Details: [docs/weights.md](docs/weights.md),
[docs/measurements/b70.2.md](docs/measurements/b70.2.md).

### Intel AutoRound — `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` (recommended; what we serve)

AutoRound (tuned rounding and clipping), int4 routed experts; attention, GDN, MTP and the rest BF16.

**On the current line, 0.32.0-b70.1** (DFlash2 K=7, async, 8 slots, 4096-token chunks; greedy, thinking off; single runs
on an idle engine; method in [v0.32/README.md](v0.32/README.md#numbers)):

| | |
|---|---|
| decode, 1 stream | **169.3 tok/s** |
| decode, 4 concurrent / 8 streams | 306.2 / 605.5 tok/s total |
| prefill 8K / 32K / 131K | 6,305 / 6,114 / 4,641 tok/s |
| GPU KV cache | 388,872 tokens (397,309 with the optional oneCCL plugin), plus the 128 GiB host-RAM tier |
| agent-session replay (243 requests) | 268 s, TTFT p50 / p90 2.13 / 4.71 s, 0 preemptions |
| quality | MMLU 300 / TruthfulQA 200 paired against the 0.30 perf line: no difference detected (not an equivalence bound) |
| host RAM | 35–39 GiB MemAvailable while serving on our 256 GB host |
| endurance | no multi-day run on 0.32 yet |

**On the 0.30.0-b70.2 recipe** (MTP k=3, 16 slots; the stable fallback). These rows are the Intel-vs-AWQ comparison;
single runs on an idle engine, thinking off, 400 (short) or 256 (16K) generated tokens per request; "prefill" = prompt
tokens over time to first token; differences under ~10 % are noise:

| | |
|---|---|
| decode, 1 stream | **90.4 tok/s with MTP** (89.8 tok/s at a 16K prompt); without MTP: not measured |
| decode, 4 streams | 67.9 tok/s per session, 250 tok/s total, with MTP |
| decode, 8 streams | 56.1 tok/s per session, 403 tok/s total, with MTP |
| decode, 10 streams | 48.8 tok/s per session, 436 tok/s total, with MTP |
| prefill | ~3,600 tok/s for one 16K prompt with MTP; 4 × 16K at once: ~3,000 tok/s total |
| degradation breakpoint | short prompts: none up to 10 streams; 16K prompts: 2 → 4 streams (per-session 90 → 23.5 tok/s with MTP); 48K: not measured |
| MTP acceptance length | 2.20–2.35 |
| coding (10 agentic tasks × 5; effort none / medium / high) | 38 / 45 / 42 of 50 with MTP (AWQ with MTP 36 / 47 / 40, p 0.82 / 0.71 / 0.80); 2 of 150 samples end in a digit-run repetition stop (AWQ 0) |
| knowledge | MMLU 280/300, TruthfulQA 178/200 (medium, MTP); AWQ with MTP 275 / 176 (p 0.18 / 0.50) |
| per card | 20.29 GiB model weights; 311,299 GPU KV tokens at memory fraction 0.88 |
| host RAM | 39.8 GiB MemAvailable at ready on our 256 GiB host, the same as AWQ; never below 53 GiB over the 8-day run |
| endurance | **8 days 6 h in one engine**, 40,912 requests, 0 errors, 0 aborts; about 15 days of serving with one crash, a GPU device-lost ([Endurance](#endurance-030-line)). On the earlier b70.2 kernels its longest run was 21.5 h, ended by two guard stops that 0029b fixed |
| setup | needs patch 0028 and `tools/intel_snapshot.py` ([engines/README.md](engines/README.md)) |
| download | 78.8 GB (the 102.4 GB BF16 PLE shard in the checkpoint is not used) |

### AWQ — `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125` (validated alternative)

AWQ (llm-compressor, calibrated), int4 routed experts only; the rest BF16. Not measured on the 0.32 line; every row
below is from the 0.30.0-b70.2 recipe (MTP k=3, 16 slots), the same as Intel's second table.

| | |
|---|---|
| decode, 1 stream | **87.9 tok/s with MTP** (85.0 tok/s at a 16K prompt); 54.8 tok/s without MTP (earlier matrix) |
| decode, 4 streams | 71.5 tok/s per session, 252 tok/s total, with MTP; 50.3 / 193 without MTP |
| decode, 8 streams | 55.6 tok/s per session, 396 tok/s total, with MTP; 43.9 / 333 without MTP |
| decode, 10 streams | 47.9 tok/s per session, 422 tok/s total, with MTP; 40.4 / 379 without MTP |
| prefill | ~3,500 tok/s for one 16K prompt with MTP; 4 × 16K at once: ~3,100 tok/s total with MTP; 48K with MTP: not measured. Without MTP (earlier matrix): ~4,000 for one 16K, ~3,900 for one 48K, ~4,100 total for 4 × 16K |
| degradation breakpoint | short prompts: none up to 10 streams; 16K prompts with MTP: 85.0 tok/s per session at 1 stream, 24.5 at 4 (2 streams: not measured); without MTP (earlier matrix): 2 → 4 streams (55 → 24), 48K prompts 1 → 2 streams |
| MTP acceptance length | 2.18–2.35 (kernels b70.3; per draft position 0.69 / 0.42 / 0.24 at one short stream, Intel 0.70 / 0.40 / 0.19 on b70.2); 2.05–2.20 on the b70.2 kernels |
| coding (10 agentic tasks × 5; effort none / medium / high) | **36 / 47 / 40 of 50 with MTP** (Intel 38 / 45 / 42, p 0.82 / 0.71 / 0.80); 48/50 at medium without MTP, vs devan 39/50 (Fisher p 0.015); 0 repetition stops in 150 |
| knowledge | MMLU 275/300, TruthfulQA 176/200 (medium, MTP); 276 / 176 without MTP |
| token-level distance to Intel (both MTP) | KL 0.082, top-1 90.2 %; same answer on 97 % of items |
| per card | 309,467 GPU KV tokens at memory fraction 0.88; 30.5–31.1 GiB VRAM in use; model weights per card: not measured |
| endurance | 3 days 16 h in one run, ~52,500 requests, 0 errors, 0 aborts ([Endurance](#endurance-030-line)) |
| host RAM | ~39 GiB MemAvailable left in steady state on our 256 GiB host (the same as Intel) |
| download | 180.8 GB |

### Minachist experts — `Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`, branch `new-main` @ `411505aa` (supported variant)

Minachist's recalibrated int4 routed experts (group 128, symmetric, F16 scales; its card reports a lower KL than its own
`main`), placed into Intel's checkpoint: `tools/minachist_repack.py convert <intel snapshot> <minachist download>
<new snapshot>` repacks the experts from compressed-tensors into Intel's GPTQ layout and keeps every other tensor Intel's, byte for byte (attention, GDN, shared expert, HC, embed/head,
MTP). Minachist's own INT6/INT8 layers are not used: the XPU path does not load them. Download: the checkpoint without
its BF16 n-gram table, ~73 GB. License: qwen-community-1.0, as the base model. The repack was exact (max |diff| 0
on every sampled layer) when measured with the script it was ported from; this port, `tools/minachist_repack.py`, has
been tested on synthetic tensors only, not yet run on the real checkpoint. Run its `verify` mode on your snapshot.

Engine settings: the Intel model file and serve config unchanged, only the snapshot path points at the new snapshot.
Measured on the 0.30 perf line (fork branch
[`b70/v0.30.0-intel-dflashc`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0-intel-dflashc): DFlash2 K=7, async
scheduling, 8 slots, kernels b70.3). Knowledge, agent smoke, gates and speed are paired against Intel on the same
engine and prompts; long context and coding are compared with Intel's stored results from the earlier MTP engine (same
harness and arguments):

| | Minachist | Intel |
|---|---|---|
| MMLU 300 / TruthfulQA 200 (paired) | 1+1 / 0+1 discordant, p = 1.0 / 1.0 | — |
| long context: needles, 64K document (none / medium) | 6/6, 2/2; 12/15, 15/15 | the same |
| agent smoke (10 tool tasks × 3) | 30/30, dark-turn rate 0.88 | 30/30, 0.85 |
| agentic coding, 150 samples | 120 | 125 (within sampling noise at 5 per task) |
| decode, 1 stream | 154 tok/s, 3.6 tokens per step | 166 tok/s, 3.9 |
| decode, 8 streams / prefill 8K–131K | 578–598 / 6,025–4,651 tok/s | 588–601 / 6,125–4,648 tok/s |
| gates (burst, structured output, mix) | 0/150, clean, 7/7 | pass |

The results are within noise of Intel's except the one-stream decode, about 7 % slower: the drafter accepts fewer
tokens per step with these experts (3.6 vs 3.9). Since the rest is a tie, we keep Intel running by subjective
preference. Not measured on
the 0.30.0-b70.2 recipe (MTP) or on 0.32.

### devan — `devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d` (reference)

Most likely round-to-nearest; int4 routed experts **and** the full-attention projections. Most of
[docs/measurements/](docs/measurements/README.md) is measured on it (0.30 line, 8 slots, earlier settings; not on 0.32).

| | |
|---|---|
| decode, 1 stream | ~59 tok/s without MTP; 78.5 tok/s with MTP |
| decode, 4 streams | ~49 tok/s per session, ~197 tok/s total, without MTP |
| prefill | **4,711 tok/s at an 18.7K prompt, 3,711 at 98K**, without MTP |
| coding (same set) | 39/50 (AWQ 48/50, p 0.015); knowledge level with AWQ |
| token-level distance to Intel | KL 0.20, prose top-1 77 % |

**Benchmark status (0.30 tables).** The Intel and AWQ speed rows are not on the same kernels: Intel's are from the b70.2 kernels, AWQ's from b70.3. AWQ's with-MTP speed rows are measured on the b70.3 kernels with B70-0030 (2026-10-04, one run
each; the first short one-stream cell after boot ran cold at 68.6 tok/s, the repeat gave 87.9); Intel's speed rows are
from the b70.2 kernels (2026-10-02, same harness), and Intel on b70.3 is not re-benched (b70.3 changes only the
CPU → GPU KV load path, which these cells do not use). AWQ's without-MTP rows come from an earlier matrix on stock
kernels. Quality rows: AWQ with MTP (2026-10-03, MTP on both; the engine ran the 64K-KV test definition); the earlier
48/50 AWQ coding row ran without MTP. The b70.2 gap (Intel +10–18 % at one stream) is ~0–5 % on b70.3, and the
acceptance per draft position is now similar for both builds (AWQ 0.69 / 0.42 / 0.24, Intel 0.70 / 0.40 / 0.19 at one
short stream). Gates on AWQ b70.3: sweep 0 bad at 1–10 streams, burst gate 0/90, structured-output gate 36 ok + 18 plain ok. All AWQ
numbers come from an earlier branch of the same code; how it differs from the release branch:
[known issues](docs/known-issues.md) item 12.

## The host-RAM KV tier

Agent sessions resend long, growing prompts. The CPU tier keeps their KV cache in 128 GB of pinned host RAM and loads
it back instead of recomputing it. Both lines run it at 128 GiB. The numbers in this table were measured on the 0.30
line; the 0.32 line's replay, KV and host-RAM numbers are in [v0.32/README.md](v0.32/README.md#numbers).

| | |
|---|---|
| prompt tokens reused from cache, real agent use (7.3 h, 1,955 requests, mean prompt 93K tokens) | **96.6 %** |
| … of which from host RAM / from the GPU pool / computed | **65 % / 31 % / 3.4 %** |
| prompt tokens served, mean over that window | 6,899 tok/s (peak 32,553 tok/s over 1 min), against 230 tok/s computed |
| KV load bandwidth, host RAM → GPU | **19.6–19.9 GB/s** |
| host memory under KV reload load (two replays of 6 real sessions) | **flat**: +0.058 GiB per rank once, no RAM-guard stop ([B70-K1 test](docs/measurements/b70.2.md)) |
| the same on the AWQ build (2.7 h, mean prompt 59K tokens) | 95.7 % from cache: GPU pool 54 %, host RAM 42 % |
| GPU vs RAM split | set by the workload, not the weights: with 6 sessions in flight the GPU pool serves 46–55 % of prompt tokens and RAM the rest; with 3 sessions the GPU pool serves 91 % |

With ~6 agent sessions in flight the GPU pool serves about half of the reusable prefix; the RAM tier serves the
rest, so a session that comes back after others have filled the GPU pool still gets its prefix in seconds instead of a
full re-prefill. The engine's working range is ~6 concurrent sessions; beyond that it holds, and everything slows.
Sources: [docs/measurements/b70.2.md](docs/measurements/b70.2.md) §3.

## Lines

| line | what | state |
|---|---|---|
| **current `0.32.0-b70.N`** | vLLM v0.32.0 + wu1ff's 1.0.5 chain + our layers (fork branch [`b70/v0.32.0`](https://github.com/Lumnus/vllm/tree/b70/v0.32.0), tag `v0.32.0-b70.1`), torch 2.14, vllm-xpu-kernels `release/0.1.15.4` + our three kernel PRs ([v0.32/](v0.32/README.md)) | **the configuration validated on 2026-10-24**; gates, quality and speed measured, no multi-day run yet. Dense attention released; sparse (QSA) in preparation |
| **stable fallback `0.30.0-b70.N`** | stock vLLM v0.30.0 + the series in `patches/` (fork branch [`b70/v0.30.0-stable`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0-stable), tag `v0.30.0-b70.2`; b70.1 was [`b70/v0.30.0`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0)), torch 2.13, vllm-xpu-kernels 0.1.14.1+b70.3 | the multi-day validated line (Intel and AWQ); the fallback |
| edge `main-YYYYMMDD-b70.N` | vLLM main + the series (fork branch [`b70/main`](https://github.com/Lumnus/vllm/tree/b70/main)), torch 2.14, our 0.1.15.4 kernel build | **not recommended**: a progressive decode corruption under concurrency ([known issues](docs/known-issues.md) item 5) |

## Quick start (0.30, source tree)

The stable fallback. For the current line, see [Quick start (0.32)](#quick-start-032-current-line).

```bash
git clone https://github.com/Lumnus/b70-flash-next && cd b70-flash-next
scripts/check-series.sh                          # patches/ == the fork branch (clones it into ./vllm-src)
scripts/make-tree.sh 0.30.0-b70.2 /srv/vllm-b70  # the source tree, hash-checked
```

1. **Environment.** The vLLM v0.30.0 XPU environment (torch 2.13.0+xpu; the `vllm/vllm-openai-xpu:v0.30.0` image's
   `/opt/venv` is one), with vllm-xpu-kernels **0.1.14.1+b70.3** built from
   [`Lumnus/vllm-xpu-kernels` tag `v0.1.14.1+b70.3`](https://github.com/Lumnus/vllm-xpu-kernels/tree/v0.1.14.1%2Bb70.3)
   ([docs/kernels.md](docs/kernels.md)); the wheel is attached to the
   [release](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2) (sha256 `f104a3e5…d589`). Put the
   tree first on `PYTHONPATH`. We run compute-runtime 26.35.39758.10.
2. **Weights (recommended: Intel AutoRound).** `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` (all files
   except `model-00016-of-00017.safetensors`), plus devan's `ple_table_qwen4exp.pt`; then
   `tools/intel_snapshot.py snapshot <download> <devan ple_table_qwen4exp.pt> <snapshot>`. Build the INT8 PLE table once
   with `tools/build_int8_ple.py build` ([docs/ple-int8.md](docs/ple-int8.md)).
   *Alternative (AWQ):* `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125`, then
   `tools/awq_snapshot.py snapshot <download> <devan ple_table_qwen4exp.pt> <snapshot>`.
   *Variant (Minachist experts):* the Intel snapshot, then `tools/minachist_repack.py convert` ([Models](#models)).
3. **Serve.** The model is a parameter: `common.env` + `engines/s16-kv128-mtp3.env` + one model file,
   `engines/models/intel-autoround.env` (recommended) or `engines/models/awq.env`; `vllm serve <snapshot>` with the
   flags in `engines/serve-s16-mtp3.args`, `--kv-offloading-size 128`, and the model file's serve config
   (`engines/serve-config-intel-autoround.json` or `engines/serve-config-awq.json`) as the snapshot's `config.json`
   ([engines/README.md](engines/README.md)). Same code and flags for both.
4. **Serving prompt.** The chat template and the default serving prompt are yours to choose. We keep the checkpoint's
   template and put the serving context (what the model runs on, nothing personal) in a short default system prompt;
   we may move to an empty engine default and let a gateway add the preamble. None of that is in this repository.

The image path (`image/Dockerfile`, `engines/run-example.sh`) is the b70.1 recipe: it pulls only public images by
digest and fails unless every patched file carries its recorded sha256 (`image/verify-overlay.sh`). For b70.2 it
still needs the kernel wheel installed and wu1ff's `libgdn_index64.so` removed; see the CHANGELOG.

## Where to find what

| path / repo | what |
|---|---|
| [`v0.32/`](v0.32/README.md) | the 0.32.0-b70.1 release: `patches/vllm/` (49 commits on v0.32.0), `patches/vllm-xpu-kernels-py/`, `patches/libgdn-index64/`, `engines/` (`serve.args`, `serve.env`, `serve-plugin.env`), `scripts/check-series.sh`, build recipe and numbers |
| `patches/vllm/`, `patches/vllm-tests/` | the series 0001–0032, exported from the fork branch (one commit per patch on v0.30.0); CPU tests (never applied to the image) |
| `patches/vllm-xpu-kernels/` | `b70.3/`: our kernel series on vllm-xpu-kernels 0.1.14.1 (the 64-bit GDN conv-state offset, five upstream fixes, B70-K1); `0001-…` at the top: the b70.1 form of the 64-bit fix ([docs/kernels.md](docs/kernels.md)) |
| `patches/series.txt` | which fork commit goes to which file |
| `scripts/` | `export-series.sh`, `check-series.sh` (patches/ == fork branch), `make-tree.sh` (a verified source tree) |
| `image/` | `Dockerfile`, `verify-overlay.sh`, wu1ff's pack files, `derive-serve-config.py` |
| `engines/` | 0.30 line: environments and `vllm serve` flags, incl. its recommended configuration (`s16-kv128-mtp3` + `models/intel-autoround.env`; [engines/README.md](engines/README.md)); the serve configs are shared with 0.32 |
| `docs/` | [switches](docs/switches.md) · [weights](docs/weights.md) · [kernels](docs/kernels.md) · [known issues](docs/known-issues.md) · [offload fix](docs/offload-fix.md) · [PLE INT8](docs/ple-int8.md) · [PLE on NVMe](docs/ple-nvme.md) · [measurements](docs/measurements/README.md) |
| `tools/` | `build_int8_ple.py` (INT8 PLE table), `intel_snapshot.py` and `awq_snapshot.py` (snapshots + serve configs), `minachist_repack.py` (Minachist experts into the Intel snapshot), `repro_band.py` (reproduce the offload miss) |
| [Lumnus/vllm](https://github.com/Lumnus/vllm) | the code: `b70/v0.32.0` (tag `v0.32.0-b70.1`, the 0.32 line), `b70/v0.30.0-stable` (the release branch from b70.2 on), `b70/v0.30.0` (b70.1), `b70/main` (edge), `offload-hybrid-junction` (the offload fix in upstream style). `b70/v0.30.0-intel` and `b70/v0.30.0-mtp0020` are the two working lines that b70.2 consolidates; they are not updated any more |
| [Lumnus/vllm-xpu-kernels](https://github.com/Lumnus/vllm-xpu-kernels) | `b70/v0.1.14` (tag `v0.1.14.1+b70.3`, the 0.30 line's kernels), `b70/v0.1.15` (edge). The 0.32 line builds upstream `release/0.1.15.4` + our PRs instead ([v0.32/README.md](v0.32/README.md#build-and-run), step 3) |
| [Lumnus/b70-fan-control](https://github.com/Lumnus/b70-fan-control) | sibling repo: a user-space fan table for the B70 |

`PROVENANCE.md` records where every piece comes from, with hashes. `CHANGELOG.md` lists releases.

## What is in the 0.30 series

The 0.32 series is listed in [v0.32/README.md](v0.32/README.md#the-series).

| patch | what | switch |
|---|---|---|
| 0001–0005 | wu1ff's pack (XPU gate, pinned-host PLE, HC K-split, GDN spec fusion, GDN index64 hook) | always on |
| 0006 | load the PLE table straight into pinned memory (boot host-RAM peak ~191 → ~158 GiB) | `B70_PLE_DIRECT_PINNED=1` |
| 0007 | PLE table in FP8 (47.7 GiB pinned instead of 95.4) | `B70_PLE_FP8=1` |
| 0008 | PLE table in INT8 with a per-row scale (48.9 GiB, 4× lower error than FP8) | `B70_PLE_INT8=1` |
| 0009 | per-effort thinking budgets; server default presence penalty | `B70_THINKING_BUDGET`, `B70_DEFAULT_PRESENCE_PENALTY` |
| 0010 | server default repetition stop (bounds the token-1023 NaN loop) | `B70_DEFAULT_REPETITION_DETECTION` |
| 0012 | accept OpenRouter-style `{"reasoning":{"effort":…}}` | `B70_REASONING_EFFORT_ALIAS=1` |
| 0013, 0013b | serve the INT8 PLE table from NVMe (~39 GiB host RAM freed) | `B70_PLE_INT8_NVME=1` |
| 0014a–f | fix hybrid-model KV-offload misses for "same document, new question"; trace; #51787 backport | `B70_OFFLOAD_*` |
| 0018 | split the pinned CPU KV pool into chunks when one pinned allocation (≥ ~31 GiB) is refused | ungated, acts only then |
| 0019, 0020 | dense-QSA configs skip `self_attn.indexer` tensors, in the model and in the MTP loaders | ungated |
| 0021 | MTP draft prefill with FULL graphs only on XPU | `B70_MTP_DRAFT_PREFILL_NO_PIECEWISE=1` |
| 0022 | port of vllm-project/vllm#55506 (Mamba spec-decode block tables by request slot) | ungated |
| 0023, 0023c/0024t | debug: GDN op mode switch, NaN tracers, a null-block probe | debug, off by default |
| 0026 | MTP: never schedule a partial draft list on XPU (stock 0.1.14 kernels crash on one) | on by default; `B70_MTP_UNIFORM_DRAFTS=0` |
| 0027, 0032 | wu1ff's `libgdn_index64.so` optional: without it, the stock (64-bit with our kernels) GDN op | automatic; `B70_GDN_MODE` |
| 0028 | the PLE embedding accepts Intel's AutoRound (INC) quantization config | ungated, acts only then |
| 0029, 0029b | prompt logprobs in 128-row chunks (no VRAM spill into host RAM) | on; `B70_PROMPT_LOGPROBS_CHUNK` |
| 0031 | CPU→GPU KV loads straight from the pinned pool (for kernels without B70-K1) | `B70_OFFLOAD_H2D_DIRECT=1` |

Every switch of the 0.30 series: [docs/switches.md](docs/switches.md); the 0.32 engine files explain theirs inline.

## Work in progress (not in the series)

1. **Sparse attention (QSA) on 0.32**: being brought up; its release slot is described in
   [v0.32/README.md](v0.32/README.md#sparse-attention-qsa-in-preparation).
2. **The proper MTP fixes** (0.30 line): padding rows that write no state (so capture sizes no longer have to avoid padding) and
   the spec + async-scheduling race (so `--no-async-scheduling` can go). Until then the serving flags work around both
   ([known issues](docs/known-issues.md) item 6).
3. **A few upstream API fixes on the 0.30 line** (request validation, streaming tool-call finish reasons, logprobs
   of parser-suppressed chunks): cherry-picked and CPU-tested, not published yet.

## The code lives in the fork

Changes are made on the fork branch, then exported here: for 0.30 with `scripts/export-series.sh`, checked by
`scripts/check-series.sh`; for 0.32 with `git format-patch v0.32.0..v0.32.0-b70.1`, checked by
`v0.32/scripts/check-series.sh`. Both checks fail on any drift.

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
