# b70-flash-next

vLLM for **Qwen3.8-Flash-Next on 4× Intel Arc Pro B70**: vLLM v0.30.0 for XPU, wu1ff's B70 patch set rebuilt from
source patches, a series of patches for host memory, long-context reuse, MTP speculative decoding and agent clients,
and a vllm-xpu-kernels build with the fixes this model needs.

| | |
|---|---|
| model | Qwen3.8-Flash-Next W4A16 (Intel AutoRound or AWQ; [Models](#models)) |
| GPUs | 4× Intel Arc Pro B70 (32 GB each), tensor parallel 4 + expert parallel |
| KV cache | GPU pool ~311K tokens + **128 GB in host RAM** (CPU tier) |
| PLE n-gram table | **INT8, served from NVMe** (8 GB pinned row cache instead of 48.9 GB in RAM) |
| decoding | MTP speculative decoding, 3 draft tokens; 16 sequences; 262K context |
| tested host | AMD EPYC 7402 (24C/48T) on an ASRock Rack ROMED8-2T, 256 GB RAM, WD_BLACK SN850X 4 TB NVMe |
| software | torch 2.13.0+xpu, compute-runtime 26.35.39758.10, vllm-xpu-kernels 0.1.14.1+b70.3 |

**Release `0.30.0-b70.2` (draft).** It runs from source trees; the container image of this repository is not built for
b70.2 ([CHANGELOG](CHANGELOG.md)).

## Models

Two calibrated 4-bit checkpoints run on this stack; a third, round-to-nearest one is the reference. **Intel AutoRound
is the fastest, AWQ has the best coding fidelity; the choice between them is pending a comparable run** (Benchmark
status below). All three serve the same INT8 PLE table from NVMe, built from devan's BF16 PLE file
([docs/ple-int8.md](docs/ple-int8.md)). Details: [docs/weights.md](docs/weights.md),
[docs/measurements/b70.2.md](docs/measurements/b70.2.md).

Throughput below: single runs on an idle engine, thinking off, 400 (short) or 256 (16K) generated tokens per request;
"prefill" = prompt tokens over time to first token; differences under ~10 % are noise.

### Intel AutoRound — `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` (fastest)

AutoRound (tuned rounding and clipping), int4 routed experts; attention, GDN, MTP and the rest BF16.

| | |
|---|---|
| decode, 1 stream | **90.4 tok/s with MTP** (89.8 tok/s at a 16K prompt); without MTP: pending the comparable run |
| decode, 4 streams | 67.9 tok/s per session, 250 tok/s total, with MTP |
| decode, 8 streams | 56.1 tok/s per session, 403 tok/s total, with MTP |
| decode, 10 streams | 48.8 tok/s per session, 436 tok/s total, with MTP |
| prefill | ~3,600 tok/s for one 16K prompt with MTP; 4 × 16K at once: ~3,000 tok/s total |
| degradation breakpoint | short prompts: none up to 10 streams; 16K prompts: 2 → 4 streams (per-session 90 → 23.5 tok/s with MTP); 48K: pending the comparable run |
| MTP acceptance length | 2.20–2.35 |
| coding (10 agentic tasks × 5, medium effort) | 45/50 with MTP (AWQ 48/50, p 0.44); 2 of 150 samples end in a digit-run repetition stop (AWQ 0) |
| knowledge | MMLU 280/300, TruthfulQA 178/200 (medium); not different from AWQ |
| per card | 20.29 GiB model weights; 311,299 GPU KV tokens at memory fraction 0.88 |
| download | 78.8 GB (the 102.4 GB BF16 PLE shard in the checkpoint is not used) |

### AWQ — `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125` (best coding fidelity)

AWQ (llm-compressor, calibrated), int4 routed experts only; the rest BF16.

| | |
|---|---|
| decode, 1 stream | **81.8 tok/s with MTP** (76.2 tok/s at a 16K prompt); 54.8 tok/s without MTP |
| decode, 4 streams | 66.0 tok/s per session, 244 tok/s total, with MTP; 50.3 / 193 without MTP |
| decode, 8 streams | 54.1 tok/s per session, 386 tok/s total, with MTP; 43.9 / 333 without MTP |
| decode, 10 streams | 45.5 tok/s per session, 405 tok/s total, with MTP; 40.4 / 379 without MTP |
| prefill | ~4,000 tok/s for one 16K prompt and ~3,900 for one 48K prompt without MTP; ~3,500 / ~3,400 with MTP; 4 × 16K at once: ~4,100 tok/s total without MTP |
| degradation breakpoint | short prompts: none up to 10 streams; 16K prompts: 2 → 4 streams (per-session 55 → 24 tok/s without MTP); 48K prompts: 1 → 2 streams |
| MTP acceptance length | 2.05–2.20 |
| coding (same set) | **48/50 without MTP**; vs devan 39/50 (Fisher p 0.015); 0 repetition stops |
| knowledge | MMLU 276/300, TruthfulQA 176/200 (medium) |
| token-level distance to Intel (both MTP) | KL 0.082, top-1 90.2 %; same answer on 97 % of items |
| per card | 309,162 GPU KV tokens; model size per card: pending the comparable run |
| download | 180.8 GB |

### devan — `devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d` (reference)

Most likely round-to-nearest; int4 routed experts **and** the full-attention projections. Most of
[docs/measurements/](docs/measurements/README.md) is measured on it (8 slots, earlier settings).

| | |
|---|---|
| decode, 1 stream | ~59 tok/s without MTP; 78.5 tok/s with MTP |
| decode, 4 streams | ~49 tok/s per session, ~197 tok/s total, without MTP |
| prefill | **4,711 tok/s at an 18.7K prompt, 3,711 at 98K**, without MTP |
| coding (same set) | 39/50 (AWQ 48/50, p 0.015); knowledge level with AWQ |
| token-level distance to Intel | KL 0.20, prose top-1 77 % |

**Benchmark status.** The Intel and AWQ rows above with MTP come from one matrix on kernels 0.1.14.1+b70.2; AWQ's
without-MTP rows come from an earlier matrix on stock kernels, and its quality rows ran without MTP. AWQ is being measured on the same stack as Intel (kernels b70.3, B70-0030, MTP). The open
question: why AWQ decodes slower than Intel with MTP, when both use the same int4 kernel (Intel's drafts are accepted
more often, ~0.68/0.39/0.19 per position vs ~0.63/0.36/0.19).

## The host-RAM KV tier

Agent sessions resend long, growing prompts. The CPU tier keeps their KV cache in 128 GB of pinned host RAM and loads
it back instead of recomputing it.

| | |
|---|---|
| prompt tokens reused from cache, real agent use (7.3 h, 1,955 requests, mean prompt 93K tokens) | **96.6 %** |
| … of which from host RAM / from the GPU pool / computed | **65 % / 31 % / 3.4 %** |
| prompt tokens served, mean over that window | 6,899 tok/s (peak 32,553 tok/s over 1 min), against 230 tok/s computed |
| KV load bandwidth, host RAM → GPU | **19.6–19.9 GB/s** |
| host memory under KV reload load (two replays of 6 real sessions) | **flat**: +0.058 GiB per rank once, no RAM-guard stop ([B70-K1 test](docs/measurements/b70.2.md)) |
| GPU vs RAM split | set by the workload, not the weights: with 6 sessions in flight the GPU pool serves 46–55 % of prompt tokens and RAM the rest; with 3 sessions the GPU pool serves 91 % |

With ~6 agent sessions in flight the GPU pool serves about half of the reusable prefix; the RAM tier serves the
rest, so a session that comes back after others have filled the GPU pool still gets its prefix in seconds instead of a
full re-prefill. The engine's working range is ~6 concurrent sessions; beyond that it holds, and everything slows.
Sources: [docs/measurements/b70.2.md](docs/measurements/b70.2.md) §3.

## Two lines

| line | what | state |
|---|---|---|
| **stable `0.30.0-b70.N`** | stock vLLM v0.30.0 + this series (fork branch [`b70/v0.30.0-intel`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0-intel) for b70.2; [`b70/v0.30.0`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0) for b70.1), torch 2.13, vllm-xpu-kernels 0.1.14.1+b70.3 | **recommended**; what we serve |
| edge `main-YYYYMMDD-b70.N` | vLLM main + the series (fork branch [`b70/main`](https://github.com/Lumnus/vllm/tree/b70/main)), torch 2.14, our 0.1.15.4 kernel build | **not recommended**: a progressive decode corruption under concurrency ([known issues](docs/known-issues.md) item 5) |

## Quick start (source tree, how we run it)

```bash
git clone https://github.com/Lumnus/b70-flash-next && cd b70-flash-next
scripts/check-series.sh                          # patches/ == the fork branch (clones it into ./vllm-src)
scripts/make-tree.sh 0.30.0-b70.2 /srv/vllm-b70  # the source tree, hash-checked
```

1. **Environment.** The vLLM v0.30.0 XPU environment (torch 2.13.0+xpu; the `vllm/vllm-openai-xpu:v0.30.0` image's
   `/opt/venv` is one), with vllm-xpu-kernels **0.1.14.1+b70.3** built from
   [`Lumnus/vllm-xpu-kernels` tag `v0.1.14.1+b70.3`](https://github.com/Lumnus/vllm-xpu-kernels/tree/v0.1.14.1%2Bb70.3)
   ([docs/kernels.md](docs/kernels.md); no wheel is published yet). Put the tree first on `PYTHONPATH`. We run
   compute-runtime 26.35.39758.10.
2. **Weights.** Intel @ `4c67bf68` (all files except `model-00016-of-00017.safetensors`), then
   `tools/intel_snapshot.py snapshot <download> <devan ple_table_qwen4exp.pt> <snapshot>`. Build the INT8 PLE table once
   with `tools/build_int8_ple.py build` ([docs/ple-int8.md](docs/ple-int8.md)).
3. **Serve.** `common.env` + `engines/intel-autoround-s16-kv128-mtp3.env`, `vllm serve <snapshot>` with the flags in
   `engines/serve-s16-mtp3.args`, `--kv-offloading-size 128`, and `engines/serve-config-intel-autoround.json` as the
   snapshot's `config.json` ([engines/README.md](engines/README.md)).

The image path (`image/Dockerfile`, `engines/run-example.sh`) is the b70.1 recipe: it pulls only public images by
digest and fails unless every patched file carries its recorded sha256 (`image/verify-overlay.sh`). For b70.2 it still
needs the kernel wheel; see the CHANGELOG.

## Where to find what

| path / repo | what |
|---|---|
| `patches/vllm/`, `patches/vllm-tests/` | the series 0001–0032, exported from the fork branch (one commit per patch on v0.30.0); CPU tests (never applied to the image) |
| `patches/vllm-xpu-kernels/` | `b70.3/`: our kernel series on vllm-xpu-kernels 0.1.14.1 (the 64-bit GDN conv-state offset, five upstream fixes, B70-K1); `0001-…` at the top: the b70.1 form of the 64-bit fix ([docs/kernels.md](docs/kernels.md)) |
| `patches/series.txt` | which fork commit goes to which file |
| `scripts/` | `export-series.sh`, `check-series.sh` (patches/ == fork branch), `make-tree.sh` (a verified source tree) |
| `image/` | `Dockerfile`, `verify-overlay.sh`, wu1ff's pack files, `derive-serve-config.py` |
| `engines/` | environments and `vllm serve` flags, incl. the configuration we serve ([engines/README.md](engines/README.md)) |
| `docs/` | [switches](docs/switches.md) · [weights](docs/weights.md) · [kernels](docs/kernels.md) · [known issues](docs/known-issues.md) · [offload fix](docs/offload-fix.md) · [PLE INT8](docs/ple-int8.md) · [PLE on NVMe](docs/ple-nvme.md) · [measurements](docs/measurements/README.md) |
| `tools/` | `build_int8_ple.py` (INT8 PLE table), `intel_snapshot.py` and `awq_snapshot.py` (snapshots + serve configs), `repro_band.py` (reproduce the offload miss) |
| [Lumnus/vllm](https://github.com/Lumnus/vllm) | the code: `b70/v0.30.0-intel` (b70.2), `b70/v0.30.0` (b70.1), `b70/main` (edge), `offload-hybrid-junction` (the offload fix in upstream style) |
| [Lumnus/vllm-xpu-kernels](https://github.com/Lumnus/vllm-xpu-kernels) | `b70/v0.1.14` (tag `v0.1.14.1+b70.3`, what we serve), `b70/v0.1.15` (edge) |
| [Lumnus/b70-fan-control](https://github.com/Lumnus/b70-fan-control) | sibling repo: a user-space fan table for the B70 |

`PROVENANCE.md` records where every piece comes from, with hashes. `CHANGELOG.md` lists releases.

## What is in the series

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

Every switch: [docs/switches.md](docs/switches.md).

## Work in progress (not in the series)

1. **The proper MTP fixes**: padding rows that write no state (so capture sizes no longer have to avoid padding) and
   the spec + async-scheduling race (so `--no-async-scheduling` can go). Until then the serving flags work around both
   ([known issues](docs/known-issues.md) item 6).
2. **A few upstream API fixes on the stable line** (request validation, streaming tool-call finish reasons, logprobs
   of parser-suppressed chunks): cherry-picked and CPU-tested, not published yet.

## The code lives in the fork

Changes are made on the fork branch, then exported here with `scripts/export-series.sh`; `scripts/check-series.sh`
fails on any drift.

## Credits

This work stands on other people's:

1. **wu1ff — [B70-LLM-Controller](https://github.com/wu1ff/B70-LLM-Controller)** (MIT). The Qwen3.8-Flash-Next B70
   pack: the XPU gate, the pinned-host PLE table, the HC down-GEMM K-split, the GDN spec-metadata fusion, the 64-bit
   GDN conv-state offset fix, the Level Zero peer-residency shim and the dense-QSA serve config. Patches 0001–0005 are
   wu1ff's changes re-derived as source diffs; two binaries are copied from wu1ff's public image by digest. Thank you.
2. **Intel** — the [`Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
   weights we serve, made with [AutoRound](https://github.com/intel/auto-round);
   [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels), whose upstream fixes #600, #563, #564, #578 and
   #586 are in our kernel build (by Guancheng Fu, Chaojun Zhang, Qiming Zhang and Tony Lin);
   [llm-scaler](https://github.com/intel/llm-scaler) and the XPU work in
   [vLLM #55068](https://github.com/vllm-project/vllm/pull/55068).
3. **wtdcode** — the [`Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16)
   weights.
4. **devan-carlin** — the [`Qwen3.8-Flash-Next-W4A16`](https://huggingface.co/devan-carlin/Qwen3.8-Flash-Next-W4A16)
   weights, its BF16 PLE table (every INT8 table here is built from it), and the early community vLLM port of
   the model to XPU (`devan-carlin/vllm@xpu-qwen4exp`).
5. **TSUMUGI-XE** — [intel/compute-runtime#968](https://github.com/intel/compute-runtime/issues/968), the analysis of
   Level Zero peer residency mirroring device memory into host memory, and the first shim that works around it.
6. **The Qwen team** — Qwen3.8-Flash-Next itself (license: qwen-community-1.0, see each checkpoint's card).
7. **[vLLM](https://github.com/vllm-project/vllm)**, on which everything here is built. Patch 0014e is a backport of
   vllm-project/vllm#51787; patch 0022 ports vllm-project/vllm#55506.

## License

Apache-2.0 (LICENSE). wu1ff's files and binaries keep their MIT license; the vLLM and vllm-xpu-kernels patches modify
Apache-2.0 code. Model weights are not included and keep their own licenses. See NOTICE.
