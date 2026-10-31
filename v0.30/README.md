# The 0.30 line (previous, archived): 0.30.0-b70.2

This directory is the 0.30 line of b70-flash-next: vLLM v0.30.0 + the 0.30 patch series, MTP speculative decoding,
torch 2.13, vllm-xpu-kernels 0.1.14.1+b70.3. This page is the line in full: its validation and endurance, the per-build
numbers (the Intel-vs-AWQ comparison), the host-RAM KV tier measurements, how to run it, and its patch series. Every
number on this page was measured on the 0.30 line; the row or paragraph says which recipe (the released MTP k=3 recipe,
or the 0.30 perf line with DFlash2 K=7). The current line is 0.32.0-b70.5: [the cover](../README.md) and
[v0.32/README.md](../v0.32/README.md).

| path (under `v0.30/`) | what |
|---|---|
| `patches/vllm/`, `patches/vllm-tests/` | the 0.30 series 0001–0032, exported from the fork branch (one commit per patch on v0.30.0); CPU tests (never applied to the image) |
| `patches/vllm-xpu-kernels/` | `b70.3/`: our kernel series on vllm-xpu-kernels 0.1.14.1 (the 64-bit GDN conv-state offset, five upstream fixes, B70-K1); `0001-…` at the top: the b70.1 form of the 64-bit fix ([docs/kernels.md](../docs/kernels.md)) |
| `patches/series.txt` | which fork commit goes to which file |
| `scripts/` | `export-series.sh`, `check-series.sh` (`patches/` == fork branch), `make-tree.sh` (a verified source tree) |
| `image/` | `Dockerfile`, `verify-overlay.sh`, wu1ff's pack files, `derive-serve-config.py` (the 0.30.0-b70.1 image recipe) |
| `engines/` | environments and `vllm serve` flags ([engines/README.md](engines/README.md)); `serve-config-intel-autoround.json` is also the 0.32 line's dense serve config |
| `docs/` | [known issues](docs/known-issues.md) of this line (items 1–3 apply to 0.32 too) · [measurements](docs/measurements/README.md) (b70.1 baseline, KV tier, PLE; b70.2 Intel vs AWQ) |

The cross-line pages stay in the repository's [`docs/`](../docs/): [switches](../docs/switches.md), [weights](../docs/weights.md),
[kernels](../docs/kernels.md), [offload fix](../docs/offload-fix.md), [PLE INT8](../docs/ple-int8.md), [PLE on NVMe](../docs/ple-nvme.md);
the tools in [`tools/`](../tools/) serve both lines.

## What it is

**Release [`0.30.0-b70.2`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2), 2026-10-07.** Both
weight builds are validated on its code ([known issues](docs/known-issues.md) item 12): Intel AutoRound (8 days 6 hours
in one engine, 0 errors) and AWQ (3 days 16 hours, 0 errors). One code branch, the model is a serving parameter. It
runs from source trees; the vllm-xpu-kernels wheel is attached to the release, and no container image is built for
b70.2 ([CHANGELOG](../CHANGELOG.md)).

| | |
|---|---|
| code | stock vLLM v0.30.0 + the series in [`patches/`](patches/) (fork branch [`b70/v0.30.0-stable`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0-stable), tag `v0.30.0-b70.2`) |
| decoding | MTP speculative decoding, 3 draft tokens; 16 sequences; 262K context; GPU memory fraction 0.88; `--no-async-scheduling`; cudagraph capture sizes at every multiple of 4 up to 64 (plus 256/512/1024) |
| KV cache | GPU pool ~311K tokens + 128 GiB in host RAM (CPU tier) |
| PLE n-gram table | INT8, served from NVMe (as on 0.32) |
| software | torch 2.13.0+xpu, compute-runtime 26.35.39758.10, vllm-xpu-kernels 0.1.14.1+b70.3 |

## Validation status

| | |
|---|---|
| **Intel AutoRound** (recommended) | **fully validated**: gates, quality, speed, and about **15 days of serving** (2026-10-07 to 2026-10-23), the longest single engine **8 days 6 hours** with 0 errors and 0 aborts ([Endurance](#endurance)). One crash in that time: a GPU device-lost, described there |
| **AWQ** | **fully validated** on an earlier branch of the same code: gates, quality, speed, and one engine running **3 days 16 hours** without an error ([Endurance](#endurance)) |
| the release branch [`b70/v0.30.0-stable`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0-stable) | not booted on a GPU as such. It is the union of the two lines we tested, with one copy of each fix. Intel's endurance ran on the release code except for the directory of the debug flag files; AWQ's on an earlier branch. Every difference is listed in [known issues](docs/known-issues.md) item 12 |

In day-to-day use on this line (an operator observation under real mixed agent load, not a benchmark), aggregate
generation peaked at about 500 tok/s and held at 300+ tok/s sustained; it depends on the jobs.

## Endurance

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
| host RAM | MemAvailable never below 53 GiB on our 256 GB host |
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

The per-build numbers of the 0.30 line. Two calibrated 4-bit checkpoints run on it and both are validated; a third,
round-to-nearest one is the reference, and a fourth variant (Minachist's recalibrated experts in Intel's checkpoint)
is supported. Intel AutoRound is the recommended build on both lines. Unless a table says otherwise, the rows below
were measured on the 0.30.0-b70.2 recipe (MTP k=3, 16 slots, kernels b70.2/b70.3) with a different harness from the
one behind the 0.32 numbers on the [cover](../README.md#models); they are much lower than the 0.32 line's
because that recipe has no DFlash2, no async scheduling and 1024-token prefill chunks, not because of the weights. The
Minachist table is the exception: it was measured on the 0.30 perf line (DFlash2 K=7). Intel's 0.32 numbers are on the
cover. The two calibrated builds are close; the reasons for the order:

1. **Speed is a tie, Intel slightly ahead at one stream.** Measured side by side, Intel decodes 2.8 % faster at one
   short stream (90.4 vs 87.9 tok/s) and 5 % faster at one 16K stream (89.8 vs 85.0), and the two are level at 4 to 10
   streams (within the ~10 % noise). The benchmarks we ran (MMLU, TruthfulQA, agentic coding, long context) do not
   separate them significantly; they sit 0.082 KL apart (top-1 agreement 90.2 %) and give the same answer on 97 % of
   items.
2. **Both have clean multi-day runs** ([Endurance](#endurance)): Intel 8 days 6 hours in one engine and about 15 days of
   serving with one crash (a GPU device-lost under a long memory-store re-ingest); AWQ 3 days 16 hours.
3. **Host RAM is the same.** An earlier reading of ~23 GiB left free with Intel against ~39 GiB with AWQ was not the
   model: at ready Intel leaves 39.8 GiB, the same as AWQ, and over its 8-day run MemAvailable never went below 53 GiB.
4. **Digit loops: too few to call.** In 150 coding samples Intel has 2 digit-run repetition stops, AWQ 0
   ([known issues](docs/known-issues.md) item 7).
5. **The rest is preference.** We run Intel day to day and keep it on a subjective preference; AWQ is a fully validated
   alternative with the same code and flags.

All of them serve the same INT8 PLE table from NVMe, built from devan's BF16 PLE file
([docs/ple-int8.md](../docs/ple-int8.md)). Details: [docs/weights.md](../docs/weights.md),
[docs/measurements/b70.2.md](docs/measurements/b70.2.md).

### Intel AutoRound — `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` (recommended)

AutoRound (tuned rounding and clipping), int4 routed experts; attention, GDN, MTP and the rest BF16.

**On the 0.30.0-b70.2 recipe** (MTP k=3, 16 slots). These rows are the Intel-vs-AWQ comparison;
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
| host RAM | 39.8 GiB MemAvailable at ready on our 256 GB host, the same as AWQ; never below 53 GiB over the 8-day run |
| endurance | **8 days 6 h in one engine**, 40,912 requests, 0 errors, 0 aborts; about 15 days of serving with one crash, a GPU device-lost ([Endurance](#endurance)). On the earlier b70.2 kernels its longest run was 21.5 h, ended by two guard stops that 0029b fixed |
| setup | needs patch 0028 and `tools/intel_snapshot.py` ([engines/README.md](engines/README.md)) |
| download | 78.8 GB (the 102.4 GB BF16 PLE shard in the checkpoint is not used) |

### AWQ — `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125` (validated alternative)

AWQ (llm-compressor, calibrated), int4 routed experts only; the rest BF16. Not measured on the 0.32 line; every row
below is from the 0.30.0-b70.2 recipe (MTP k=3, 16 slots), the same as Intel's table.

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
| endurance | 3 days 16 h in one run, ~52,500 requests, 0 errors, 0 aborts ([Endurance](#endurance)) |
| host RAM | ~39 GiB MemAvailable left in steady state on our 256 GB host (the same as Intel) |
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

## The host-RAM KV tier (0.30 measurements)

Agent sessions resend long, growing prompts. The CPU tier keeps their KV cache in 128 GiB of pinned host RAM and loads
it back instead of recomputing it. Both lines run it at 128 GiB in a private pinned pool per rank. Every number in this
table was measured on the 0.30 line (the 0.30.0-b70.2 recipe, kernels b70.3); it is the deeper measurement of the same
mechanism the 0.32 line uses. The 0.32 line's own KV-tier numbers (an agent-session replay, host RAM) are on the
[cover](../v0.32/models/qwen38-flash-next/meta.md#host-ram-flash-next).

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

## Quick start

For the current line, see [Quick start (0.32)](../README.md#quick-start).

```bash
git clone https://github.com/Lumnus/b70-flash-next && cd b70-flash-next/v0.30
scripts/check-series.sh                          # patches/ == the fork branch (clones it into ./vllm-src)
scripts/make-tree.sh 0.30.0-b70.2 /srv/vllm-b70  # the source tree, hash-checked
```

1. **Environment.** The vLLM v0.30.0 XPU environment (torch 2.13.0+xpu; the `vllm/vllm-openai-xpu:v0.30.0` image's
   `/opt/venv` is one), with vllm-xpu-kernels **0.1.14.1+b70.3** built from
   [`Lumnus/vllm-xpu-kernels` tag `v0.1.14.1+b70.3`](https://github.com/Lumnus/vllm-xpu-kernels/tree/v0.1.14.1%2Bb70.3)
   ([docs/kernels.md](../docs/kernels.md)); the wheel is attached to the
   [release](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2) (sha256 `f104a3e5…d589`). Put the
   tree first on `PYTHONPATH`. We run compute-runtime 26.35.39758.10.
2. **Weights (recommended: Intel AutoRound).** `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` (all files
   except `model-00016-of-00017.safetensors`), plus devan's `ple_table_qwen4exp.pt`; then
   `tools/intel_snapshot.py snapshot <download> <devan ple_table_qwen4exp.pt> <snapshot>`. Build the INT8 PLE table once
   with `tools/build_int8_ple.py build` ([docs/ple-int8.md](../docs/ple-int8.md)).
   *Alternative (AWQ):* `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125`, then
   `tools/awq_snapshot.py snapshot <download> <devan ple_table_qwen4exp.pt> <snapshot>`.
   *Variant (Minachist experts):* the Intel snapshot, then `tools/minachist_repack.py convert` ([Models](#models)).
3. **Serve.** The model is a parameter: `engines/common.env` + `engines/s16-kv128-mtp3.env` + one model file,
   `engines/models/intel-autoround.env` (recommended) or `engines/models/awq.env`; `vllm serve <snapshot>` with the
   flags in `engines/serve-s16-mtp3.args`, `--kv-offloading-size 128`, and the model file's serve config
   (`engines/serve-config-intel-autoround.json` or `engines/serve-config-awq.json`) as the snapshot's `config.json`
   ([engines/README.md](engines/README.md)). Same code and flags for both.
4. **Serving prompt.** The chat template and the default serving prompt are yours to choose. We keep the checkpoint's
   template and put the serving context (what the model runs on, nothing personal) in a short default system prompt;
   we may move to an empty engine default and let a gateway add the preamble. None of that is in this repository.

The image path (`image/Dockerfile`, built from this directory; `engines/run-example.sh`) is the b70.1 recipe: it pulls only public images by
digest and fails unless every patched file carries its recorded sha256 (`image/verify-overlay.sh`). For b70.2 it
still needs the kernel wheel installed and wu1ff's `libgdn_index64.so` removed; see the [CHANGELOG](../CHANGELOG.md).

## What is in the 0.30 series

The 0.32 series is listed in [v0.32/README.md](../v0.32/README.md#the-series).

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

Every switch of the 0.30 series: [docs/switches.md](../docs/switches.md); the 0.32 engine files explain theirs inline.

## Work in progress on the 0.30 line (not in the series)

1. **The proper MTP fixes**: padding rows that write no state (so capture sizes no longer have to avoid padding) and
   the spec + async-scheduling race (so `--no-async-scheduling` can go). Until then the serving flags work around both
   ([known issues](docs/known-issues.md) item 6).
2. **A few upstream API fixes** (request validation, streaming tool-call finish reasons, logprobs
   of parser-suppressed chunks): cherry-picked and CPU-tested, not published yet.
