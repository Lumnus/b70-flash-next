# b70-flash-next

vLLM for **Qwen3.8-Flash-Next on 4× Intel Arc Pro B70**: the vLLM v0.30.0 XPU image, wu1ff's B70 patch set rebuilt
from source patches, and a series of opt-in improvements for host memory, long-context reuse and agent clients.

**Status: pre-release `0.30.0-b70.1` — the image has not been built yet.** The patches have been validated on 4× B70
from source trees (numbers below); the first image build from this repository is next.

## Credits first

This work stands on other people's:

1. **wu1ff — [B70-LLM-Controller](https://github.com/wu1ff/B70-LLM-Controller)** (MIT). The Qwen3.8-Flash-Next B70
   pack: the XPU gate, the pinned-host PLE table, the HC down-GEMM K-split, the GDN spec-metadata fusion, the 64-bit
   GDN conv-state offset fix, the Level Zero peer-residency shim and the dense-QSA serve config. Patches 0001–0005 are
   wu1ff's changes re-derived as source diffs; two binaries are copied from wu1ff's public image by digest. Thank you.
2. **devan-carlin** — the [`Qwen3.8-Flash-Next-W4A16`](https://huggingface.co/devan-carlin/Qwen3.8-Flash-Next-W4A16)
   weights this targets, and the early community vLLM port of the model to XPU (`devan-carlin/vllm@xpu-qwen4exp`).
3. **TSUMUGI-XE** — [intel/compute-runtime#968](https://github.com/intel/compute-runtime/issues/968), the analysis of
   Level Zero peer residency mirroring device memory into host memory, and the first shim that works around it.
4. **Intel** — [llm-scaler](https://github.com/intel/llm-scaler) and the XPU work in
   [vLLM #55068](https://github.com/vllm-project/vllm/pull/55068) (Qwen3.8-Next on XPU).
5. **[vLLM](https://github.com/vllm-project/vllm)** and **[vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels)**,
   on which everything here is built. Patch 0014e is a backport of vllm-project/vllm#51787.

## What is in it

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

Details and caveats: [docs/measurements/](docs/measurements/README.md).

## Quick start

```bash
git clone https://github.com/Lumnus/b70-flash-next && cd b70-flash-next
scripts/check-series.sh                      # patches/ == the fork branch (clones it into ./vllm-src)
docker buildx build -f image/Dockerfile -t b70-flash-next:0.30.0-b70.1 \
  --build-arg GIT_SHA=0.30.0-b70.1 --build-arg SOURCE_SHA=$(git rev-parse HEAD) .
```

The build pulls only public images (`docker.io/vllm/vllm-openai-xpu` and `ghcr.io/wu1ff/qwen38-flashnext-b70`, both by
digest) and fails unless every patched file carries its recorded sha256 (`image/verify-overlay.sh`).

Weights: `devan-carlin/Qwen3.8-Flash-Next-W4A16` at revision `40b8f18d` (it includes `ple_table_qwen4exp.pt`). For the
INT8 table run `tools/build_int8_ple.py build` once ([docs/ple-int8.md](docs/ple-int8.md)). Then:

```bash
MODELS=/srv/models CACHE=/srv/cache engines/run-example.sh r8g-kv64-0014jb b70-flash-next:0.30.0-b70.1
```

See [engines/README.md](engines/README.md) for the three example configurations and their host-RAM needs.

## Layout

| path | what |
|---|---|
| `patches/vllm/` | the series, exported from [Lumnus/vllm `b70/v0.30.0`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0) (one commit per patch on v0.30.0) |
| `patches/vllm-tests/` | the CPU tests for 0013 and 0014 (never applied to the image) |
| `patches/vllm-xpu-kernels/` | the 64-bit conv-state offset fix as source ([Lumnus/vllm-xpu-kernels `gdn-int64-conv-offset`](https://github.com/Lumnus/vllm-xpu-kernels/tree/gdn-int64-conv-offset)); reconstructed from wu1ff's description, untested, not used by the build |
| `patches/series.txt` | which fork commit goes to which file |
| `scripts/` | `export-series.sh`, `check-series.sh`, `make-tree.sh` (a verified source tree from the fork or from `patches/`) |
| `image/` | `Dockerfile`, `verify-overlay.sh`, wu1ff's pack files, `derive-serve-config.py` |
| `engines/` | example environment and `vllm serve` flags |
| `docs/` | switches, the offload fix, PLE INT8 and NVMe, measurements |
| `tools/` | `build_int8_ple.py` (INT8 PLE table), `repro_band.py` (reproduce the offload miss) |

`PROVENANCE.md` records where every piece comes from, with hashes. `CHANGELOG.md` lists releases.

## The code lives in the fork

Changes are made on the fork branch, then exported here with `scripts/export-series.sh`; `scripts/check-series.sh`
fails on any drift. An upstream-style version of the offload fix for vLLM main is on the fork's
`offload-hybrid-junction` branch.

## License

Apache-2.0 (LICENSE). wu1ff's files and binaries keep their MIT license; the vLLM patches modify Apache-2.0 code.
See NOTICE.
