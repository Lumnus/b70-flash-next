# b70-flash-next

**Sparse attention (QSA) for Qwen3.8-Flash-Next on Intel Arc Pro B70, in vLLM.** The model's own sparse attention runs
end to end on B70 cards: the QSA indexer picks a 2,048-token selection for each full-attention layer and Triton kernels
on XPU attend over it, on 4 cards and on 2. It is the default of both Flash-Next presets. Against dense attention on the
same engine, sparse prefills 1.31× faster at 260K tokens (4,480 against 3,420 tok/s), finishes an agent-session replay
in 204.7 s against 230–235 s, and answers 80/78 % against 72/72 % of long-context tasks at 128K. Dense attention
(FlashAttention) is a parameter of the same engine; both are verified. This is the first released, gated vLLM serving
recipe for Qwen3.8-Flash-Next with sparse attention on B70. Prior public work: Intel's SYCL QSA kernels for Flash-Next
on B70, and TSUMUGI-XE's Triton QSA patches
([vllm-project/vllm#54595](https://github.com/vllm-project/vllm/issues/54595)).

The same engine serves **Qwen3.8-27B** and **Qwen3.6-35B-A3B**, from the same Python environment, with one preset per model
and card count. On our 4-card host the three run side by side: Flash-Next on two cards, the 35B and the 27B on one card
each. The repository is named after the model it was built for.

The engine is vLLM v0.32.0 with wu1ff's 1.0.5 B70 chain, our patches on top, sparse attention and hot/cold expert offload
for Flash-Next, DFlash/DFlash2 speculative decoding, a host-RAM KV tier, and a vllm-xpu-kernels build carrying our upstream
fixes and native int8 ops. Its files are in [`v0.32/`](v0.32/README.md): the engine in `v0.32/engine/`, one directory per
model in `v0.32/models/`, each with a page that names the build and configuration every number was measured on.

## Models

| model, weights | preset | attention | KV | context | decode, 1 stream (tok/s) | peak decode, total (tok/s, streams) | best prefill (tok/s, prompt) | MMLU / TruthfulQA |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.8-Flash-Next**, Intel AutoRound W4A16 ([page](v0.32/models/qwen38-flash-next/meta.md)) | 4 cards | **sparse** | bf16 | 262,144 | 156.4 | **580.9** (8) | **5,382** (32K) | 92.0 / 89.0 on 2 cards |
| | 4 cards | dense | bf16 | 262,144 | 169.3 | 605.5 (8) | **6,305** (8K) | 86.7 / 89.0, thinking off |
| | 2 cards | **sparse** | bf16 | 262,144 | 57.1 | 119.7 (8) | 2,768 (32K) | 92.0 / 89.0 |
| **Qwen3.8-27B**, INT4 AutoRound run W4A8 ([page](v0.32/models/qwen38-27b/meta.md)) | 2 cards | dense | bf16 | 262,144 | ~94 prose | 370 (8) | 3,504 (32K) | 91.0 / 83.5 |
| | 1 card | dense | fp8 | 147,456 | 56–62 prose, 142–149 code | 413–419 code, 175–177 prose (4) | 1,947 (32K) | — |
| **Qwen3.6-35B-A3B**, Intel INT4 mixed AutoRound ([page](v0.32/models/qwen36-35b-a3b/meta.md)) | 2 cards | dense | bf16 | 262,144 | 152–168 prose, 376 code | **808** prose (16) | **9,997** (32K) | 88.7 / 88.5 |
| | 1 card | dense | bf16 | 196,608 | 118–120 prose, 310–314 code | 756–798 code, 308–316 prose (4) | 8,386 (8K) | — |

How to read the rows: every number is on its model's page with the build, date and settings it was measured on. Peak decode
is the highest concurrency measured on that preset; prose and code differ because the drafter is accepted more often on
code. Quality is 300 MMLU and 200 TruthfulQA MC1 items, the same items for every model, thinking on at medium effort unless
marked, taken on the engine runs that produced the speed numbers; the Flash-Next 4-card sparse cell carries its 2-card
numbers (same engine and weights; teacher-forced TP4 against TP2 fidelity is at the noise floor); a dash means the 1-card
preset was not run for quality. Choosing between them: Flash-Next scores higher than the 35B on MMLU (McNemar p 0.006 on the
same items) and needs two or four cards; the 35B is the fastest on two cards and on one, and holds four full 262K requests
in GPU KV on two; the 27B is below Flash-Next on TruthfulQA (p 0.016), not different from the 35B.

| model | attention backends (target · drafter), verified | KV cache dtype |
|---|---|---|
| Qwen3.8-Flash-Next | sparse QSA on Triton kernels (default) and dense on FlashAttention · DFlash2 on `TRITON_ATTN` | bf16 on both presets |
| Qwen3.8-27B | FlashAttention · DFlash2 on `TRITON_ATTN` | fp8 on 1 card (what fits 147K context there), bf16 on 2 |
| Qwen3.6-35B-A3B | FlashAttention · DFlash on `FLASH_ATTN` | bf16 on both presets |

What works, what does not and how to tune it: [Lessons learned](docs/lessons-learned.md).

## Current release: 0.32.0-b70.5

| | |
|---|---|
| release | `v0.32.0-b70.5`, 2026-10-31 ([release notes](RELEASE-NOTES-b70.5.md), [CHANGELOG](CHANGELOG.md)). Earlier: `v0.32.0-b70.4` (sparse attention as the default of two Flash-Next presets; [notes](docs/releases/b70.4.md)), `v0.32.0-b70.3`, `v0.32.0-b70.2`, `v0.32.0-b70.1` |
| **fix** | `B70_OFFLOAD_GDN_BACKSTEP` is 0 in every preset (b70.4 shipped 1): with the CPU KV tier and prefix caching it restored a wrong GDN state on a prefix-cache hit, so a model in an agent session re-did finished work. The engine also forces it off under `mamba_cache_mode=align`. Users of b70.4: set it to 0 |
| code | fork tag [`v0.32.0-b70.5`](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.5): vLLM v0.32.0 + 109 commits, exported to four directories in [`v0.32/engine/patches/`](v0.32/engine/patches/): `vllm/` (50, the dense base incl. wu1ff's chain), `vllm-qsa/` (38, sparse attention), `vllm-hotcold/` (13, hot/cold expert offload), `vllm-models/` (8, the 27B/35B layer and the backstep guard) |
| GPUs | Intel Arc Pro B70 (32 GB); presets for 1, 2 and 4 cards |
| software | torch 2.14.0+xpu, triton 3.8.0+xpu, oneAPI 2026.1.1, compute-runtime 26.35.39758.10, oneCCL 2022.1.2 with the legacy plugin rebuilt from oneCCL #229, vllm-xpu-kernels 0.1.16.2 + five patches (#660, #662, native int8 ops, #605) and our `libgdn_index64` ([v0.32/README.md](v0.32/README.md#software)) |
| tested host | AMD EPYC 7402 (24C/48T) on an ASRock Rack ROMED8-2T, 256 GB RAM, WD_BLACK SN850X 4 TB NVMe |
| validation | every preset gated and measured for speed, quality measured per model; each model page lists its gates |
| previous | [`0.30.0-b70.2`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2), Flash-Next only, archived in [`v0.30/`](v0.30/README.md) |

## Quick start

The short path; every step with its pinned commits and checks is in [v0.32/README.md](v0.32/README.md#build-and-run). These
are the commands we ran, with our paths replaced by placeholders.

1. **Host.** B70 cards on the xe driver, compute-runtime 26.35.39758.10, oneAPI DPC++ 2026.1.1 for the native builds.
2. **vLLM (the engine) and the Python environment** (torch 2.14.0+xpu, oneCCL 2022.1.2 from pip):
   ```bash
   git clone -b v0.32.0-b70.5 https://github.com/Lumnus/vllm.git vllm-b70-0.32
   uv venv --python 3.12 /opt/b70/venv
   uv pip install --python /opt/b70/venv/bin/python --index-strategy unsafe-best-match \
       --extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://wheels.vllm.ai/xpu/ \
       -r vllm-b70-0.32/requirements/xpu.txt torch==2.14.0+xpu oneccl==2022.1.2 auto-round-lib==0.16.0 transformers==5.17.0
   VLLM_TARGET_DEVICE=xpu uv pip install --python /opt/b70/venv/bin/python --no-build-isolation -e vllm-b70-0.32
   ```
3. **Kernels.** `uv pip install vllm-xpu-kernels==0.1.16.2` (the PyPI wheel: Python files and attention libraries), then the
   compiled libraries rebuilt from upstream `6b30c9b` + [`v0.32/engine/patches/vllm-xpu-kernels/`](v0.32/engine/patches/vllm-xpu-kernels/)
   laid over the installed ones, then the astra Python modules from
   [`v0.32/engine/patches/vllm-xpu-kernels-py/`](v0.32/engine/patches/vllm-xpu-kernels-py/) (build steps 3–4).
4. **Native libraries:** wu1ff's from B70-LLM-Controller release `qwen3.8-flash-next-v1.0.5`, `libgdn_index64.so` with our
   patch, the Level Zero peer-residency shim, Niluthon's hot/cold decode op (Flash-Next 2-card), and the patched oneCCL
   legacy plugin (build steps 5–6).
5. **Weights**, per model: the pins and the preparation are on each model's page.
6. **Serve** one preset:
   ```bash
   VENV=/opt/b70/venv v0.32/engine/serve.sh qwen38-flash-next preset-4card
   VENV=/opt/b70/venv MODEL_PATH=/models/qwen3.8-27b/int4-autoround v0.32/engine/serve.sh qwen38-27b preset-1card
   ```

## Previous line

[`0.30.0-b70.2`](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2) (2026-10-07, Qwen3.8-Flash-Next only,
stock vLLM v0.30.0 + its series, MTP speculative decoding): the recipe is archived in [`v0.30/`](v0.30/README.md).

## Lines

| line | what | state |
|---|---|---|
| **current** `0.32.0-b70.N` | vLLM v0.32.0 + wu1ff's 1.0.5 chain + our layers (fork tag `v0.32.0-b70.5`, branch [`b70/v0.32.0-b70.5`](https://github.com/Lumnus/vllm/tree/b70/v0.32.0-b70.5)), torch 2.14, vllm-xpu-kernels 0.1.16.2 + five patches ([v0.32/](v0.32/README.md)) | **the current line**: Qwen3.8-Flash-Next (4 and 2 cards), Qwen3.8-27B and Qwen3.6-35B-A3B (2 and 1 card); gates, quality and speed measured per preset |
| previous `0.30.0-b70.N` | stock vLLM v0.30.0 + the series in `v0.30/patches/` (fork branch [`b70/v0.30.0-stable`](https://github.com/Lumnus/vllm/tree/b70/v0.30.0-stable), tag `v0.30.0-b70.2`), torch 2.13, vllm-xpu-kernels 0.1.14.1+b70.3 ([v0.30/README.md](v0.30/README.md)) | archived; Flash-Next only |
| edge `main-YYYYMMDD-b70.N` | vLLM main + the series (fork branch [`b70/main`](https://github.com/Lumnus/vllm/tree/b70/main)), torch 2.14, our `0.1.15.4+b70.1` kernel build ([docs/kernels.md](docs/kernels.md)) | archived; not for serving (a progressive decode corruption under concurrency, [v0.30/docs/known-issues.md](v0.30/docs/known-issues.md) item 5) |

## Where to find what

| path / repo | what |
|---|---|
| [`v0.32/`](v0.32/README.md) | the current line: `README.md` (the engine: series, software, build and run, tuning), `engine/` (`patches/` with the four vLLM directories, the kernel patches, the astra Python patches and `libgdn-index64/`; `scripts/check-series.sh`; `tools/check-trunk-tree.sh`; `serve.sh`), `models/<model>/` (`meta.md`, presets per card count, chat templates, weight file lists, model tools) |
| [`v0.30/`](v0.30/README.md) | the 0.30.0-b70.2 line (previous, archived): its page, patches, engines, image recipe, scripts, known issues and measurements |
| `docs/` | cross-line pages: [lessons learned](docs/lessons-learned.md) · [switches](docs/switches.md) · [weights](docs/weights.md) · [kernels](docs/kernels.md) · [offload fix](docs/offload-fix.md) · [PLE INT8](docs/ple-int8.md) · [PLE on NVMe](docs/ple-nvme.md) · [measurements](docs/measurements/README.md) · [release notes of earlier releases](docs/releases/) |
| `tools/` | Flash-Next tools used by both lines: `build_int8_ple.py` (INT8 PLE table), `intel_snapshot.py` and `awq_snapshot.py` (snapshots + serve configs), `minachist_repack.py` (Minachist experts into the Intel snapshot), `repro_band.py` (reproduce the offload miss) |
| [Lumnus/vllm](https://github.com/Lumnus/vllm) | the code: `b70/v0.32.0-b70.5` (tag `v0.32.0-b70.5`, the 0.32 engine; tags `v0.32.0-b70.4-dense`, `v0.32.0-b70.4-qsa` and `v0.32.0-b70.4` mark where the dense base, the sparse layer and the hot/cold layer end); older: `b70/v0.32.0-b70.4`, `b70/v0.32.0` (tag `v0.32.0-b70.1`), `b70/v0.32.0-qsa-release` (b70.2/b70.3 sparse), `b70/v0.30.0-stable`, `b70/v0.30.0` (b70.1), `b70/main` (edge), `offload-hybrid-junction` (the offload fix in upstream style) |
| [Lumnus/vllm-xpu-kernels](https://github.com/Lumnus/vllm-xpu-kernels) | `b70/v0.1.14` (tag `v0.1.14.1+b70.3`, the 0.30 line's kernels), `b70/v0.1.15` (edge). The 0.32 line's kernels are upstream `6b30c9b` (0.1.16.2) + `v0.32/engine/patches/vllm-xpu-kernels/` |
| [Lumnus/b70-fan-control](https://github.com/Lumnus/b70-fan-control) | sibling repo: a user-space fan table for the B70 |

`PROVENANCE.md` records where every piece comes from, with hashes (§8 for 0.32.0-b70.4, §9 for b70.5). `CHANGELOG.md` lists
releases.

## The code lives in the fork

Changes are made on the fork branch, then exported here: for 0.32 with `git format-patch` of the four ranges
(`v0.32.0..v0.32.0-b70.4-dense`, `..-qsa`, `..v0.32.0-b70.4`, `..v0.32.0-b70.5`), checked by
`v0.32/engine/scripts/check-series.sh --all`; for 0.30 with `v0.30/scripts/export-series.sh`, checked by
`v0.30/scripts/check-series.sh`. Both checks fail on any drift.

## Upstream

Fixes from this work, posted upstream on 2026-10-25. As of 2026-10-31, #658 is merged on vllm-xpu-kernels main (2026-10-26, in no
release yet) and #63566 on vLLM main (2026-10-29); the others are open. The engine carries all six: #660 and #662 in our kernel
build (`v0.32/engine/patches/vllm-xpu-kernels/`), #658 through our `libgdn_index64.so`, the two vLLM fixes cherry-picked in the
dense base, oneCCL #229 in the plugin build. It also carries three open PRs by others: vllm-project/vllm#64190 and #64244
(Triton W8A8 INT8 GEMM tiles on XPU) and vllm-project/vllm-xpu-kernels#605 (grouped GEMM scale-prefetch pitch).

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
   **Hot/cold expert offload**, which makes the Flash-Next 2-card preset possible, is his: `v0.32/engine/patches/vllm-hotcold/` 0001–0010
   are his B70-0040..0049 ported to 0.32 with his authorship, and the preset's fused hot+cold decode op is built from his
   `kernels/b70-moe-decode`.
8. **DominikBucko** — [qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090) (Apache-2.0),
   the two-card CUDA port. No code is taken from it; its published measurements and design notes (async scheduling and the
   Mamba state-block fix, prefill chunk sizes, the AVX2 cold-expert path) shaped which levers we measured, and the 2-card
   preset starts its hot set from his `configs/static_hot_cache_rankings.json` (@ `4dca58f`).
9. **The Qwen team** — Qwen3.8-Flash-Next, Qwen3.8-27B and Qwen3.6-35B-A3B (licenses on each checkpoint's card).
10. **For the 27B and 35B**: **devan-carlin**'s [`Qwen3.8-27B-int4-AutoRound`](https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound)
    and **Intel**'s [`Qwen3.6-35B-A3B-int4-mixed-AutoRound`](https://huggingface.co/Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound) weights;
    **incoai**'s [`Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) and **z-lab**'s
    [`Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) drafters (Apache-2.0); **Katsumi Takeuchi**
    (recutita), vllm-project/vllm#64190 and #64244, the Triton W8A8 INT8 GEMM tiles on XPU, carried with his authorship;
    **Guancheng Fu** (Intel), vllm-xpu-kernels#605, the grouped GEMM pitch guard the 35B needs, carried with his authorship;
    **steveseguin** — [b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab) (The Unlicense): the
    shortlist draft head idea and the token list behind the 27B's draft vocab.
11. **[vLLM](https://github.com/vllm-project/vllm)**, on which everything here is built. Patch 0014e is a backport of
    vllm-project/vllm#51787; patch 0022 ports vllm-project/vllm#55506.

## License

Apache-2.0 (LICENSE). wu1ff's files and binaries keep their MIT license; the vLLM and vllm-xpu-kernels patches modify
Apache-2.0 code. Model weights are not included and keep their own licenses. See NOTICE.
