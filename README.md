# b70-flash-next

vLLM for **Qwen3.8-Flash-Next on Intel Arc Pro B70** cards. This page describes the current line, **0.32**, release
**0.32.0-b70.4**: vLLM v0.32.0 with wu1ff's 1.0.5 B70 chain, our patches on top, sparse attention (QSA), hot/cold expert
offload, DFlash2 speculative decoding and a vllm-xpu-kernels build carrying our upstream fixes. Its files are in
[`v0.32/`](v0.32/README.md), which has the build recipe, the measurements and the parameter table. The previous line,
**0.30.0-b70.2** (vLLM v0.30.0), stays as the stable fallback; its files are the rest of this repository root (`patches/`,
`engines/`, `image/`), and its page is [docs/line-0.30.md](docs/line-0.30.md). Every number on this page names the build
and configuration it was measured on.

**0.32.0-b70.4: one engine, one environment, two presets.** The engine is fork tag `v0.32.0-b70.4`; the environment is torch
2.14 with vllm-xpu-kernels 0.1.16.2 rebuilt with our kernel PRs #660 and #662, plus our `libgdn_index64.so`. Two tuned presets
run it:

1. **4-card** ([`preset-4card`](v0.32/engines/preset-4card.args)): the performance preset. Sparse attention, 12 slots, 128 GiB
   KV tier. 156.4 tok/s at one stream, 287 at 4 concurrent, 581 at 8 streams, 344K GPU KV tokens.
2. **2-card** ([`preset-2card`](v0.32/engines/preset-2card.args)): two cards, so the other pair stays free for any other
   model. Sparse attention, 4 slots, the full 262K context, hot/cold expert offload (145 of 256 experts per layer per rank in
   host RAM), 64 GiB KV tier. 58.2 tok/s at one stream, 69 at 4 concurrent, 273K GPU KV tokens (measured at a 16 GiB tier, which left
   155 GiB of host RAM free on our 256 GB host; about 107 GiB expected at 64).

Attention (sparse or dense), cards, drafter, expert offload, KV offload size, slots and GPU memory are parameters of the same
engine, each with its default and what it was measured at: [Parameters](v0.32/README.md#parameters). Both attention values
work; sparse is the default. [Release notes](RELEASE-NOTES-b70.4.md) · [measurements](docs/measurements/v0.32-qsa.md).

## Current line: 0.32.0-b70.4

| | |
|---|---|
| release | `v0.32.0-b70.4`, 2026-10-26 ([release notes](RELEASE-NOTES-b70.4.md), [CHANGELOG](CHANGELOG.md)). Earlier: [`v0.32.0-b70.3`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.32.0-b70.3) (sparse near dense speed), [`v0.32.0-b70.2`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.32.0-b70.2) (sparse added), [`v0.32.0-b70.1`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.32.0-b70.1) (2026-10-25; the dense configuration validated on 2026-10-24) |
| code | fork tag [`v0.32.0-b70.4`](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.4): vLLM v0.32.0 + 101 commits, exported to [`v0.32/patches/vllm/`](v0.32/patches/vllm/) (50, the dense base incl. wu1ff's chain), [`vllm-qsa/`](v0.32/patches/vllm-qsa/) (38, sparse attention) and [`vllm-hotcold/`](v0.32/patches/vllm-hotcold/) (13, hot/cold expert offload, from Niluthon's qwen38-flash-tp2) |
| model | Qwen3.8-Flash-Next W4A16, **Intel AutoRound recommended** (the only build measured on 0.32); AWQ is validated on the 0.30 line only; Minachist's recalibrated experts are a supported variant, measured on the 0.30 perf line ([Models](#models)) |
| GPUs | 4× Intel Arc Pro B70 (32 GB each): all four with the 4-card preset (tensor parallel 4 + expert parallel), two with the 2-card preset (tensor parallel 2) |
| decoding | DFlash2 speculative decoding, 7 draft tokens (incoai's drafter, on `TRITON_ATTN`); async scheduling; 4096-token prefill chunks; 262K context; GPU memory fraction 0.89 |
| KV cache | 4-card preset: 344,021 GPU tokens + 128 GiB in host RAM. 2-card preset: 273,169 GPU tokens + 64 GiB in host RAM. The host tier is a private pinned pool per rank; no `/dev/shm` requirement |
| PLE n-gram table | **INT8, served from NVMe** (a pinned row cache of 8 GiB at 4 cards, 4 GiB at 2, instead of 48.9 GiB in RAM) |
| software | torch 2.14.0+xpu, triton 3.8.0+xpu, oneAPI 2026.1.1, compute-runtime 26.35.39758.10, oneCCL 2022.1.2 (both presets use the legacy plugin rebuilt from oneCCL #229, +~2 % KV; the stock plugin is a parameter), vllm-xpu-kernels 0.1.16.2 + our #660/#662, with #658 merged upstream (main only) and carried by our `libgdn_index64` ([v0.32/README.md](v0.32/README.md#software)) |
| tested host | AMD EPYC 7402 (24C/48T) on an ASRock Rack ROMED8-2T, 256 GB RAM, WD_BLACK SN850X 4 TB NVMe |
| status | working: gates, quality, speed and an agent-session replay measured; sparse attention is experimental; no multi-day run yet ([known issues](#known-issues-032)) |
| stable fallback | [`0.30.0-b70.2`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2), multi-day validated ([below](#the-030-line-stable-fallback)) |

## Numbers (0.32.0-b70.4)

The two presets' numbers above were measured on 2026-10-26 with the released engine and environment, Intel AutoRound,
single runs, greedy (temperature 0, thinking off); the full table (8 streams, prefill, per-step times, gates, host RAM) is
[v0.32/README.md §Presets](v0.32/README.md#presets).

Sparse against dense attention, on identical engine code (measured on the 0.1.15.4 kernel build; the 0.1.16.2 environment
passed the gate with the token compare identical), 4 cards: an agent-session replay finishes in 204.7 s sparse (12 slots)
against 230–235 s dense (8 slots); on short context one-stream decode is 0.92× dense, 4 streams level, 8 streams
0.93×, prefill up to 32K 0.89×; long prefill 1.06× at 131K and 1.31× at 260K; long-context
tasks equal up to 8K and 80/78 % against 72/72 % at 128K. The dense numbers and what each was measured at are in
[v0.32/README.md §Parameters](v0.32/README.md#parameters), the sparse-against-dense comparison in
[docs/measurements/v0.32-qsa.md](docs/measurements/v0.32-qsa.md) §0, and see [v0.32/README.md](v0.32/README.md#numbers-0320-b701-how-the-line-was-validated) for how the
0.32 line compared with 0.30 when it was first validated (0.32 matched the 0.30 perf line; the gain over 0.30.0-b70.2 is
DFlash2, async scheduling and prefill chunking, not the version bump).

## Models

Intel AutoRound is recommended and is the only build measured on 0.32. The others:

| build | 0.32 line | measured on |
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
instead of recomputing it: 128 GiB in the 4-card preset (32 GiB per rank), 64 GiB in the 2-card preset (32 GiB per rank), each a private
pinned pool per rank (`B70_OFFLOAD_PRIVATE_PINNED=1`), copied to the GPU directly, with no `/dev/shm` requirement. Pinned
memory does not count against a container memory limit: watch the host's MemAvailable, and lower `--kv-offloading-size` with
less RAM. What each preset takes and leaves free: [v0.32/README.md](v0.32/README.md#host-ram). The deeper measurement of the same mechanism (prompt-token reuse under real
agent use, host-RAM vs GPU-pool split, load bandwidth) was taken on the 0.30 line:
[docs/line-0.30.md](docs/line-0.30.md#the-host-ram-kv-tier-030-measurements).

## Quick start (0.32, current line)

The short path; every step with its pinned commits and checks is in [v0.32/README.md](v0.32/README.md#build-and-run).
These are the commands we ran with our paths replaced by placeholders; the sequence has not been run end to end in this
form from a clean clone.

1. **Host.** B70 cards on the xe driver, compute-runtime 26.35.39758.10, oneAPI DPC++ 2026.1.1 for the native builds.
2. **vLLM (the engine) and the Python environment** (torch 2.14.0+xpu, oneCCL 2022.1.2 from pip):
   ```bash
   git clone -b v0.32.0-b70.4 https://github.com/Lumnus/vllm.git vllm-b70-0.32   # the engine: dense base + sparse + hot/cold
   uv venv --python 3.12 /opt/b70/venv
   uv pip install --python /opt/b70/venv/bin/python --index-strategy unsafe-best-match \
       --extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://wheels.vllm.ai/xpu/ \
       -r vllm-b70-0.32/requirements/xpu.txt torch==2.14.0+xpu oneccl==2022.1.2 auto-round-lib==0.16.0 transformers==5.17.0
   VLLM_TARGET_DEVICE=xpu uv pip install --python /opt/b70/venv/bin/python --no-build-isolation -e vllm-b70-0.32
   ```
3. **Kernels.** `uv pip install vllm-xpu-kernels==0.1.16.2` (the PyPI wheel: Python files and attention libraries), then the
   compiled libraries rebuilt from upstream `6b30c9b` + [`v0.32/patches/vllm-xpu-kernels/`](v0.32/patches/vllm-xpu-kernels/)
   (our #660 and #662) laid over the installed ones, then the astra Python modules from
   [`v0.32/patches/vllm-xpu-kernels-py/`](v0.32/patches/vllm-xpu-kernels-py/) (build steps 3–4).
4. **Native libraries:** wu1ff's from B70-LLM-Controller release `qwen3.8-flash-next-v1.0.5`, `libgdn_index64.so` with our
   patch, the Level Zero peer-residency shim, and, for the 2-card preset, Niluthon's hot/cold decode op; then the patched
   oneCCL legacy plugin (build steps 5–6).
5. **Weights.** Intel AutoRound @ `4c67bf68` (without the 102.4 GB PLE shard), devan's `ple_table_qwen4exp.pt`, the INT8
   PLE table (`tools/build_int8_ple.py build`), the snapshot (`tools/intel_snapshot.py snapshot`), a serve dir with
   [`v0.32/engines/serve-config-qsa.json`](v0.32/engines/serve-config-qsa.json) as its `config.json` (sparse; for dense
   attention [`engines/serve-config-intel-autoround.json`](engines/serve-config-intel-autoround.json)), the drafter
   `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744`, and for the 2-card preset DominikBucko's hot/cold rankings seed
   (build step 7).
6. **Serve** one preset:
   ```bash
   P=preset-4card   # or preset-2card
   set -a; . v0.32/engines/$P.env; set +a
   mapfile -t args < <(grep -v '^#' v0.32/engines/$P.args | sed '/^$/d' | sed 's/^\(--[^ ]*\) \(.*\)$/\1\n\2/')
   LD_LIBRARY_PATH=/opt/b70/venv/lib /opt/b70/venv/bin/vllm serve /models/qwen3.8-flash-next/serve-dir "${args[@]}"
   ```
   The venv's own oneAPI runtime must come first on `LD_LIBRARY_PATH`. The first start compiles cold (7 min in our first
   boot, ~4–5 min after). Every other capability is a parameter ([Parameters](v0.32/README.md#parameters)).

The chat template is the checkpoint's own in these commands. The serving prompt is yours to choose; ours is not in this
repository.

## What is in the 0.32 series

The engine, fork tag `v0.32.0-b70.4`, is exported to three directories that apply in order on vLLM v0.32.0:

| directory | what |
|---|---|
| [`v0.32/patches/vllm/`](v0.32/patches/vllm/), 50 | the dense base (fork tag `v0.32.0-b70.4-dense`): 0001 wu1ff's MIT notice; 0002–0016 wu1ff's 1.0.5 chain for vLLM 0.32 (target port, DFlash2, C1 verify/target graphs, astra500 MoE/HC/prefill paths, serving fixes), unchanged with his authorship; 0017–0034 our 0.30 layers ported (INT8 PLE from NVMe, per-effort thinking budget, repetition stop, effort alias, dense-QSA indexer skip, MTP draft prefill graphs, uniform-drafts guard, step trace, offload fixes, prompt-logprobs chunks, and their CPU tests); 0035–0037 our upstream vLLM fixes #63564 and #63566 cherry-picked; 0038–0049 0.32-specific fixes, the private pinned KV pool among them; 0050 the backport of vllm-project/vllm#54163 (DFlash keeps its trailing prefix-cache block) |
| [`v0.32/patches/vllm-qsa/`](v0.32/patches/vllm-qsa/), 38 | sparse attention (QSA): XPU bring-up, the `attn2` kernel, the deterministic top-k with a fused Triton select, tuning hooks, the split KV groups (0033 and 0036), the `B70_QSA_OPTIMER` per-op timer (0034), the cache-path lifecycle driver (0035), C1 snapshots in host RAM (0037), the DFlash draft page at TP2 (0038); ends at fork tag `v0.32.0-b70.4-qsa` |
| [`v0.32/patches/vllm-hotcold/`](v0.32/patches/vllm-hotcold/), 13 | hot/cold expert offload: Niluthon's B70-0040..0049 from qwen38-flash-tp2 (with his authorship) and our HC1–HC3 (cold count per layer per rank, and cold prefetch never inside graph capture) |

Beside them: [`v0.32/patches/vllm-xpu-kernels/`](v0.32/patches/vllm-xpu-kernels/) (our #660 and #662 on vllm-xpu-kernels
`6b30c9b`, the source of 0.1.16.2), [`v0.32/patches/vllm-xpu-kernels-py/`](v0.32/patches/vllm-xpu-kernels-py/) (the astra
Python modules and hooks, against the 0.1.16.2 Python files) and [`v0.32/patches/libgdn-index64/`](v0.32/patches/libgdn-index64/).
The engine files in [`v0.32/engines/`](v0.32/engines/) explain each switch inline. Per-patch tables and what is off by default:
[v0.32/README.md](v0.32/README.md#the-series).

## Known issues (0.32)

1. **No multi-day run yet.** Since 2026-10-24 the line has run in serving and development windows; no engine has run for days in one go.
2. **The agent replay, long prefill and the long-context task set were not re-run on the released environment**; they come
   from the 0.1.15.4 kernel build with identical engine code; at 4 cards the released environment matches that build's
   per-step decode time and passes the token compare against it.
3. **The 2-card preset is 0.37× the 4-card preset at one stream**: the cold experts are read from host RAM every step.
4. **One hang in development**, before the copy-through-lock fix; it did not recur; cause unknown.
5. **Dense GPU KV (8 slots) is below the 0.30 perf line**: ~2 % with the patched oneCCL plugin, ~4 % with the stock plugin; not yet attributed.
6. **The GPU idles ~3 ms per step in the PLE sync at one stream.** Open.
7. **Only Intel AutoRound was measured on 0.32.**
8. **0.30 items that still apply** where the code is shared: card hangs after a GT reset, liveness must generate,
   pinned memory escapes container limits ([docs/known-issues.md](docs/known-issues.md) items 1–3).
9. **oneCCL `LL=twoshots` corrupts output** on this stack; every oneCCL change needs the burst gate and a token compare.

The full list, with the sparse-attention issues and workarounds: [v0.32/README.md](v0.32/README.md#known-issues-032).

## Sparse attention (QSA, experimental)

**Status: working, experimental, the default of both presets.** The checkpoint's QSA indexer is kept, and the
full-attention layers attend over the indexer's selection (a 2,048-token budget) with Triton kernels instead of over the
whole context. Added in 0.32.0-b70.2, near dense speed in 0.32.0-b70.3, KV packing fixed and made the preset default in
0.32.0-b70.4. It passes the gates and the long-context needles, and its output matches dense up to the selection budget;
beyond 2K tokens it drifts from dense (KL 0.0224 on a 12K-token document, against dense's own run-to-run floor of about
0.013–0.014), and on long-context tasks it scores higher than dense at 128K. Graph-capturing the sparse path degrades output
beyond 2K and must not be served. Against dense, its build history, what is in the add-on and its known issues:
[v0.32/README.md](v0.32/README.md#sparse-attention-qsa); fidelity and what the measurement does not show:
[docs/measurements/v0.32-qsa.md](docs/measurements/v0.32-qsa.md).

## Work in progress

1. **Open on 0.32.0-b70.4:** the agent replay and long prefill on the released environment; dense attention at 12 slots on
   it; the stock oneCCL plugin with sparse attention (not run); why graph-capturing the sparse path breaks fidelity beyond
   2K; generation and reasoning quality of sparse against dense (only retrieval-type tasks and next-token prediction are
   measured); a multi-day run.
2. **A stricter token compare**: the greedy compare checks for garbage, not equivalence; the gate rows will be re-run with
   the stricter one.

The 0.30 line's open items: [docs/line-0.30.md](docs/line-0.30.md#work-in-progress-on-the-030-line-not-in-the-series).

## The 0.30 line (stable fallback)

**[`0.30.0-b70.2`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2), 2026-10-07**: stock vLLM
v0.30.0 + the series in [`patches/`](patches/), torch 2.13.0+xpu, vllm-xpu-kernels 0.1.14.1+b70.3, MTP speculative
decoding with 3 draft tokens, 16 sequences, ~311K GPU KV tokens + the same 128 GiB CPU tier.

Use it when you need a line with multi-day runs behind it. On its recipe: **Intel AutoRound** served about **15 days**
(2026-10-07 to 2026-10-23), the longest engine **8 days 6 hours** with 0 errors and 0 aborts, and one crash (a GPU
device-lost, cause unknown); **AWQ** ran **3 days 16 hours** in one engine with 0 errors. Its speed, on the harness of
the [0.32 validation table](v0.32/README.md#numbers-0320-b701-how-the-line-was-validated): 107.5 tok/s at one stream, 370 tok/s total at 8 streams, prefill 4,052 tok/s
at 8K.

Everything else about it (validation status, endurance tables, the per-build Models tables with the Intel-vs-AWQ
comparison, the KV-tier measurements, its quick start, its patch series, its open items):
**[docs/line-0.30.md](docs/line-0.30.md)**.

## Lines

| line | what | state |
|---|---|---|
| **current `0.32.0-b70.N`** | vLLM v0.32.0 + wu1ff's 1.0.5 chain + our layers, sparse attention and hot/cold expert offload (fork tag `v0.32.0-b70.4`, branch [`b70/v0.32.0-b70.4`](https://github.com/Lumnus/vllm/tree/b70/v0.32.0-b70.4)), release `v0.32.0-b70.4`, torch 2.14, vllm-xpu-kernels 0.1.16.2 + our #660/#662, with #658 merged upstream (main only) and carried by our `libgdn_index64` ([v0.32/](v0.32/README.md)) | **the current line**: one engine, a 4-card and a 2-card preset; gates, quality, speed and an agent replay measured, no multi-day run yet. Sparse attention (the preset default, experimental) against dense on the same engine code: agent replay 204.7 against 230–235 s, short context 0.89–1.00×, long prefill 1.06–1.31× ([measurements](docs/measurements/v0.32-qsa.md) §0) |
| **stable fallback `0.30.0-b70.N`** | stock vLLM v0.30.0 + the series in `patches/` (fork branch [`b70/v0.30.0-stable`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0-stable), tag `v0.30.0-b70.2`; b70.1 was [`b70/v0.30.0`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0)), torch 2.13, vllm-xpu-kernels 0.1.14.1+b70.3 ([docs/line-0.30.md](docs/line-0.30.md)) | the multi-day validated line (Intel and AWQ); the fallback |
| edge `main-YYYYMMDD-b70.N` | vLLM main + the series (fork branch [`b70/main`](https://github.com/Lumnus/vllm/tree/b70/main)), torch 2.14, our `0.1.15.4+b70.1` kernel build ([docs/kernels.md](docs/kernels.md)) | **not recommended**: a progressive decode corruption under concurrency ([known issues](docs/known-issues.md) item 5) |

## Where to find what

| path / repo | what |
|---|---|
| [`v0.32/`](v0.32/README.md) | the 0.32 line: `patches/vllm/` (50 commits on v0.32.0, the dense base), `patches/vllm-qsa/` (38, sparse attention), `patches/vllm-hotcold/` (13, hot/cold expert offload), `patches/vllm-xpu-kernels/` (our #660/#662), `patches/vllm-xpu-kernels-py/`, `patches/libgdn-index64/`, `engines/` (`preset-4card.args`/`.env`, `preset-2card.args`/`.env`, `serve-config-qsa.json`), `scripts/check-series.sh` (`--all` for the three directories), build recipe, presets, parameters and numbers |
| [`docs/line-0.30.md`](docs/line-0.30.md) | the 0.30.0-b70.2 line: validation, endurance, per-build numbers, KV-tier measurements, quick start, series |
| `patches/vllm/`, `patches/vllm-tests/` | the 0.30 series 0001–0032, exported from the fork branch (one commit per patch on v0.30.0); CPU tests (never applied to the image) |
| `patches/vllm-xpu-kernels/` | `b70.3/`: our kernel series on vllm-xpu-kernels 0.1.14.1 (the 64-bit GDN conv-state offset, five upstream fixes, B70-K1); `0001-…` at the top: the b70.1 form of the 64-bit fix ([docs/kernels.md](docs/kernels.md)) |
| `patches/series.txt` | which fork commit goes to which file (0.30 series) |
| `scripts/` | `export-series.sh`, `check-series.sh` (patches/ == fork branch), `make-tree.sh` (a verified source tree); 0.30 line |
| `image/` | `Dockerfile`, `verify-overlay.sh`, wu1ff's pack files, `derive-serve-config.py` (the 0.30.0-b70.1 image recipe) |
| `engines/` | 0.30 line: environments and `vllm serve` flags ([engines/README.md](engines/README.md)); the two serve configs are shared with 0.32 |
| `docs/` | [0.30 line](docs/line-0.30.md) · [switches](docs/switches.md) · [weights](docs/weights.md) · [kernels](docs/kernels.md) · [known issues](docs/known-issues.md) · [offload fix](docs/offload-fix.md) · [PLE INT8](docs/ple-int8.md) · [PLE on NVMe](docs/ple-nvme.md) · [measurements](docs/measurements/README.md) |
| `tools/` | `build_int8_ple.py` (INT8 PLE table), `intel_snapshot.py` and `awq_snapshot.py` (snapshots + serve configs), `minachist_repack.py` (Minachist experts into the Intel snapshot), `repro_band.py` (reproduce the offload miss); used by both lines |
| [Lumnus/vllm](https://github.com/Lumnus/vllm) | the code: `b70/v0.32.0-b70.4` (tag `v0.32.0-b70.4`, the 0.32 engine; tags `v0.32.0-b70.4-dense` and `v0.32.0-b70.4-qsa` mark where the dense base and the sparse layer end); older: `b70/v0.32.0` (tag `v0.32.0-b70.1`), `b70/v0.32.0-qsa-release` (b70.2/b70.3 sparse), `b70/v0.30.0-stable` (the release branch from b70.2 on), `b70/v0.30.0` (b70.1), `b70/main` (edge), `offload-hybrid-junction` (the offload fix in upstream style). `b70/v0.30.0-intel` and `b70/v0.30.0-mtp0020` are the two working lines that b70.2 consolidates; they are not updated any more |
| [Lumnus/vllm-xpu-kernels](https://github.com/Lumnus/vllm-xpu-kernels) | `b70/v0.1.14` (tag `v0.1.14.1+b70.3`, the 0.30 line's kernels), `b70/v0.1.15` (edge). The 0.32 line's kernels are upstream `6b30c9b` (0.1.16.2) + `v0.32/patches/vllm-xpu-kernels/` ([v0.32/README.md](v0.32/README.md#build-and-run), step 3) |
| [Lumnus/b70-fan-control](https://github.com/Lumnus/b70-fan-control) | sibling repo: a user-space fan table for the B70 |

`PROVENANCE.md` records where every piece comes from, with hashes (§8 for 0.32). `CHANGELOG.md` lists releases.

## The code lives in the fork

Changes are made on the fork branch, then exported here: for 0.32 with `git format-patch` of the three ranges
(`v0.32.0..v0.32.0-b70.4-dense`, `..-qsa`, `..v0.32.0-b70.4`), checked by `v0.32/scripts/check-series.sh --all`; for 0.30 with
`scripts/export-series.sh`, checked by `scripts/check-series.sh`. Both checks fail on any drift.

## Upstream

Fixes from this work, posted upstream on 2026-10-25 (open as of that day; #658 was merged on main on 2026-10-26 and is in no release yet). b70.4 carries all six: #660 and #662 in our kernel build (`v0.32/patches/vllm-xpu-kernels/`), #658 through our `libgdn_index64.so`, the two vLLM fixes cherry-picked in the dense base, oneCCL #229 in the plugin build.

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
   GDN conv-state offset fix, the Level Zero peer-residency shim and the dense-QSA serve config. In the 0.30 series, patches
   0001–0005 are wu1ff's changes re-derived as source diffs, and two binaries are copied from wu1ff's public image by
   digest. In the 0.32 series, patches 0001–0016 are his. The 0.32
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
   the 0.32 FP8 head (B70-V32-0053) is a port of its 0053, and the 1- and 4-stream numbers on this page use `bench_det`.
   **Hot/cold expert offload**, which makes the 2-card preset possible, is his: `v0.32/patches/vllm-hotcold/` 0001–0010
   are his B70-0040..0049 ported to 0.32 with his authorship, and the preset's fused hot+cold decode op is built from his
   `kernels/b70-moe-decode`.
8. **DominikBucko** — [qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090) (Apache-2.0),
   the two-card CUDA port. No code is taken from it; its published measurements and design notes (async scheduling and the
   Mamba state-block fix, prefill chunk sizes, the AVX2 cold-expert path) shaped which levers we measured, and the 2-card
   preset starts its hot set from his `configs/static_hot_cache_rankings.json` (@ `4dca58f`).
9. **The Qwen team** — Qwen3.8-Flash-Next itself (license: qwen-community-1.0, see each checkpoint's card).
10. **[vLLM](https://github.com/vllm-project/vllm)**, on which everything here is built. Patch 0014e is a backport of
    vllm-project/vllm#51787; patch 0022 ports vllm-project/vllm#55506.

## License

Apache-2.0 (LICENSE). wu1ff's files and binaries keep their MIT license; the vLLM and vllm-xpu-kernels patches modify
Apache-2.0 code. Model weights are not included and keep their own licenses. See NOTICE.
