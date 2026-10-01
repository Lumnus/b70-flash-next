# b70-flash-next

vLLM for **Qwen3.8-Flash-Next on 4× Intel Arc Pro B70**: the vLLM v0.30.0 XPU image, wu1ff's B70 patch set rebuilt
from source patches, and a series of opt-in improvements for host memory, long-context reuse and agent clients.

**Status: pre-release `0.30.0-b70.1`.** The series serves daily on 4× B70 from source trees; the first image build from
this repository is next.

## Credits first

This work stands on other people's:

1. **wu1ff — [B70-LLM-Controller](https://github.com/wu1ff/B70-LLM-Controller)** (MIT). The Qwen3.8-Flash-Next B70
   pack: the XPU gate, the pinned-host PLE table, the HC down-GEMM K-split, the GDN spec-metadata fusion, the 64-bit
   GDN conv-state offset fix, the Level Zero peer-residency shim and the dense-QSA serve config. Patches 0001–0005 are
   wu1ff's changes re-derived as source diffs; two binaries are copied from wu1ff's public image by digest. Thank you.
2. **devan-carlin** — the [`Qwen3.8-Flash-Next-W4A16`](https://huggingface.co/devan-carlin/Qwen3.8-Flash-Next-W4A16)
   weights, and the early community vLLM port of the model to XPU (`devan-carlin/vllm@xpu-qwen4exp`).
3. **wtdcode** — the [`Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16)
   weights we serve.
4. **TSUMUGI-XE** — [intel/compute-runtime#968](https://github.com/intel/compute-runtime/issues/968), the analysis of
   Level Zero peer residency mirroring device memory into host memory, and the first shim that works around it.
5. **Intel** — [llm-scaler](https://github.com/intel/llm-scaler) and the XPU work in
   [vLLM #55068](https://github.com/vllm-project/vllm/pull/55068) (Qwen3.8-Next on XPU).
6. **[vLLM](https://github.com/vllm-project/vllm)** and **[vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels)**,
   on which everything here is built. Patch 0014e is a backport of vllm-project/vllm#51787.

## Two lines

| line | what | state |
|---|---|---|
| **stable `0.30.0-b70.N`** | stock vLLM v0.30.0 + this series (fork branch [`b70/v0.30.0`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0)), torch 2.13, vllm-xpu-kernels 0.1.14.1 | **recommended**; what we serve |
| edge `main-YYYYMMDD-b70.N` | vLLM main + the series (fork branch [`b70/main`](https://github.com/Lumnus/vllm/tree/b70/main)), torch 2.14, [our kernel build](docs/kernels.md) | **not recommended**: a progressive decode corruption under concurrency is under investigation ([known issues](docs/known-issues.md) item 5) |

`patches/` and the image are the stable line. The edge line has no release yet.

## Quick start

```bash
git clone https://github.com/Lumnus/b70-flash-next && cd b70-flash-next
scripts/check-series.sh                      # patches/ == the fork branch (clones it into ./vllm-src)
docker buildx build -f image/Dockerfile -t b70-flash-next:0.30.0-b70.1 \
  --build-arg GIT_SHA=0.30.0-b70.1 --build-arg SOURCE_SHA=$(git rev-parse HEAD) .
```

The build pulls only public images (`docker.io/vllm/vllm-openai-xpu` and `ghcr.io/wu1ff/qwen38-flashnext-b70`, both by
digest) and fails unless every patched file carries its recorded sha256 (`image/verify-overlay.sh`).

Weights: `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125` (what we serve) or `devan-carlin/Qwen3.8-Flash-Next-W4A16`
@ `40b8f18d`; the PLE table `ple_table_qwen4exp.pt` comes from the devan repository either way. Build the INT8 PLE table
once with `tools/build_int8_ple.py build` ([docs/ple-int8.md](docs/ple-int8.md)); for AWQ, make the snapshot with
`tools/awq_snapshot.py` ([docs/weights.md](docs/weights.md)). Then:

```bash
MODELS=/srv/models CACHE=/srv/cache engines/run-example.sh awq-s16-kv128-chunked b70-flash-next:0.30.0-b70.1
```

## Where to find what

| path / repo | what |
|---|---|
| `patches/vllm/`, `patches/vllm-tests/` | the series 0001–0019, exported from the fork branch (one commit per patch on v0.30.0); CPU tests (never applied to the image) |
| `patches/vllm-xpu-kernels/` | the 64-bit GDN conv-state offset fix as source; built into our kernels ([docs/kernels.md](docs/kernels.md)) |
| `patches/series.txt` | which fork commit goes to which file |
| `scripts/` | `export-series.sh`, `check-series.sh` (patches/ == fork branch), `make-tree.sh` (a verified source tree) |
| `image/` | `Dockerfile`, `verify-overlay.sh`, wu1ff's pack files, `derive-serve-config.py` |
| `engines/` | example environments and `vllm serve` flags, incl. the recipe we serve ([engines/README.md](engines/README.md)) |
| `docs/` | [switches](docs/switches.md) · [weights](docs/weights.md) · [kernels](docs/kernels.md) · [known issues](docs/known-issues.md) · [offload fix](docs/offload-fix.md) · [PLE INT8](docs/ple-int8.md) · [PLE on NVMe](docs/ple-nvme.md) · [measurements](docs/measurements/README.md) |
| `tools/` | `build_int8_ple.py` (INT8 PLE table), `awq_snapshot.py` (AWQ snapshot + serve config), `repro_band.py` (reproduce the offload miss) |
| [Lumnus/vllm](https://github.com/Lumnus/vllm) | the code: `b70/v0.30.0` (stable), `b70/main` (edge), `offload-hybrid-junction` (the offload fix in upstream style) |
| [Lumnus/vllm-xpu-kernels](https://github.com/Lumnus/vllm-xpu-kernels) | `b70/v0.1.15` = our kernel build (release/0.1.15.4 + the int64 fix) |
| [Lumnus/b70-fan-control](https://github.com/Lumnus/b70-fan-control) | sibling repo: a user-space fan table for the B70 |

`PROVENANCE.md` records where every piece comes from, with hashes. `CHANGELOG.md` lists releases.

## What is in the series

| patch | what | switch (default off) |
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
| 0019 | dense-QSA configs skip `self_attn.indexer` tensors (e.g. in the AWQ checkpoint) | ungated |

Every switch: [docs/switches.md](docs/switches.md). With none set, the image behaves as wu1ff's pack image.

## Headline numbers (4× Arc Pro B70, TP4 + EP, measured 2026-09-28 … 30)

| | |
|---|---|
| decode, 1 stream / 4 streams aggregate | ~59 / ~197 tok/s (llama.cpp layer split on the same cards: 19.5 / 26.1) |
| prefill at 18.7K / 98K tokens | 4,711 / 3,711 tok/s (llama.cpp: 548 / 354) |
| PLE table in host RAM | 95.4 GiB (BF16) → 48.9 GiB (INT8, no measurable quality loss) → 8 GiB cache (NVMe) |
| NVMe PLE cost | decode −2 … −4 %, prefill unchanged, KL at the INT8 noise floor |
| 8 growing agent sessions past the GPU KV pool, 64 GiB CPU KV tier | TTFT median 18 s (161 s without the tier) |
| "same document, new question" revisits in the failing length band | unpatched 0/3 hits forever → 3/3 on the first revisit with the fix |
| AWQ vs devan weights, agentic coding at effort medium | 48/50 vs 39/50 (Fisher p 0.015); MMLU/TruthfulQA not different |

Numbers above are with devan's weights and 8 slots, except the AWQ row. Details and caveats:
[docs/measurements/](docs/measurements/README.md), [docs/weights.md](docs/weights.md).

## Work in progress (not in the series)

1. **MTP speculative decoding** on the stable line: fork branch
   [`b70/v0.30.0-mtp0020`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0-mtp0020) = `b70/v0.30.0` + 0020 (the MTP
   loader skips the indexer tensors) + 0021 (FULL-only draft prefill graphs on XPU). It boots and is fast for one
   request, but not servable: NaN with ≥3 concurrent requests ([known issues](docs/known-issues.md) item 6). A port of
   vllm-project/vllm#55506 (0022) did not fix it and is not published.
2. **A few upstream API fixes on the stable line** (request validation, streaming tool-call finish reasons, logprobs
   of parser-suppressed chunks): cherry-picked and CPU-tested, not published yet.
3. **Kernels for torch 2.13**: vllm-xpu-kernels 0.1.14.1 + the int64 fix, to replace wu1ff's closed library on the
   stable line ([docs/kernels.md](docs/kernels.md)).

## The code lives in the fork

Changes are made on the fork branch, then exported here with `scripts/export-series.sh`; `scripts/check-series.sh`
fails on any drift.

## License

Apache-2.0 (LICENSE). wu1ff's files and binaries keep their MIT license; the vLLM patches modify Apache-2.0 code.
See NOTICE.
