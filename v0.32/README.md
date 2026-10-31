# The 0.32 engine (0.32.0-b70.5) — vLLM 0.32 for Intel Arc Pro B70

One engine, three models. vLLM **v0.32.0** with wu1ff's 1.0.5 B70 chain, our B70 layers ported from the 0.30 line, the
0.32-specific fixes, two of our upstream vLLM fixes cherry-picked, sparse attention (QSA) and hot/cold expert offload for
Qwen3.8-Flash-Next, and the layer that serves Qwen3.8-27B and Qwen3.6-35B-A3B next to it (FP8 lm_head, native int8
activation quantisation and an oneDNN W8A8 GEMM, a shortlist draft head) together with the GDN backstep guard. The engine
is fork tag `v0.32.0-b70.5`; the environment is torch 2.14 + vllm-xpu-kernels 0.1.16.2 rebuilt with five kernel patches +
our `libgdn_index64.so` ([Software](#software)). Model, card count, drafter, attention and offload are run-time parameters:
each model has a directory with its presets, one per card count.

| model | presets | attention (target · drafter) | KV cache dtype | page |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (Intel AutoRound W4A16) | 4 cards, 2 cards | **sparse QSA on our Triton kernels** (default) or dense on FlashAttention, both verified · DFlash2 on `TRITON_ATTN` | bf16, both presets | [models/qwen38-flash-next/](models/qwen38-flash-next/meta.md) |
| Qwen3.8-27B (INT4 AutoRound, run W4A8) | 2 cards, 1 card | FlashAttention · DFlash2 on `TRITON_ATTN` | 2 cards bf16; 1 card **fp8** | [models/qwen38-27b/](models/qwen38-27b/meta.md) |
| Qwen3.6-35B-A3B (Intel INT4 mixed AutoRound) | 2 cards, 1 card | FlashAttention · DFlash on `FLASH_ATTN` | bf16, both presets | [models/qwen36-35b-a3b/](models/qwen36-35b-a3b/meta.md) |

FlashAttention here is vLLM's XPU default backend for the target (no `--attention-backend` flag); the drafter's backend is
named in its `--speculative-config`. bf16 KV means no `--kv-cache-dtype` flag (`auto`).

| | |
|---|---|
| code | fork tag [`v0.32.0-b70.5`](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.5) (branch `b70/v0.32.0-b70.5`), 109 commits on vLLM v0.32.0 (`5df63a7203`), exported as four directories ([The series](#the-series)) |
| GPUs | Intel Arc Pro B70 (32 GB); presets for 1, 2 and 4 cards; three engines on one 4-card host measured (Flash-Next on 2 cards, the 35B and the 27B on one card each) |
| speculative decoding | DFlash2 (Flash-Next, 27B) or DFlash (35B), 7 draft tokens, async scheduling, Model Runner V2 |
| KV | GPU KV plus a host-RAM tier in a private pinned pool per rank (`--kv-offloading-size`) |
| software | [Software](#software); pins with hashes in [PROVENANCE.md](../PROVENANCE.md) §8 and §9 |
| tested host | AMD EPYC 7402, 256 GB RAM, WD_BLACK SN850X 4 TB NVMe, 4× B70 |
| validation | every preset gated (burst, structured output, token compare, agent loop) and measured for speed; quality measured per model; runs from a Python environment, no container image |

The previous line, 0.30.0-b70.2 (Flash-Next only), is archived in [`v0.30/`](../v0.30/README.md).

## Layout

```
v0.32/
  README.md                this page: the engine, its series, software, build and run
  engine/
    patches/               the series: vllm/ (dense base), vllm-qsa/ (sparse), vllm-hotcold/ (hot/cold), vllm-models/ (27B/35B
                           layer + backstep guard), vllm-xpu-kernels/ (5 kernel patches), vllm-xpu-kernels-py/, libgdn-index64/
    scripts/check-series.sh  each patch directory against a fresh export of the fork (--all for the four vLLM directories)
    tools/check-trunk-tree.sh  the four directories against a trunk at tree level (maintainers)
    serve.sh               serve.sh MODEL PRESET: sources the preset .env, reads the .args, runs vllm serve
  models/
    qwen38-flash-next/     meta.md, preset-4card.{args,env}, preset-2card.{args,env}, chat_template.jinja, serve-config-qsa.json
    qwen38-27b/            meta.md, preset-1card / preset-2card, chat templates, weights.oid, drafter.oid, draft vocab,
                           build_draft_vocab.py, quantize_dflash2_w8a8.py
    qwen36-35b-a3b/        meta.md, preset-1card / preset-2card, chat templates, weights.oid, drafter.oid
```

A new model is a new directory under `models/`: its presets as `.args`/`.env` pairs, a `meta.md` with the pinned weights,
drafter, tested cards, context and the measured numbers, and its chat template if it has one.

## The series

The engine is fork tag `v0.32.0-b70.5`, exported as four directories that apply in order on vLLM v0.32.0 (`git am`, or
`git apply` one after the other); `engine/scripts/check-series.sh --all` fails if any of them differs from a fresh export of the
fork. The first three are 0.32.0-b70.4's engine unchanged (its three tags mark where each layer ends):

| directory | export of | commits |
|---|---|---|
| [`patches/vllm/`](engine/patches/vllm/) | `v0.32.0..v0.32.0-b70.4-dense` | 50: the dense base (below) |
| [`patches/vllm-qsa/`](engine/patches/vllm-qsa/) | `v0.32.0-b70.4-dense..v0.32.0-b70.4-qsa` | 38: sparse attention ([Sparse attention](models/qwen38-flash-next/meta.md#sparse-attention-qsa)) |
| [`patches/vllm-hotcold/`](engine/patches/vllm-hotcold/) | `v0.32.0-b70.4-qsa..v0.32.0-b70.4` | 13: hot/cold expert offload ([below](#hotcold-expert-offload)) |
| [`patches/vllm-models/`](engine/patches/vllm-models/) | `v0.32.0-b70.4..v0.32.0-b70.5` | 8: the 27B/35B layer and the backstep guard ([below](#the-models-layer)) |

`engine/tools/check-trunk-tree.sh` is the tree-level counterpart for maintainers: it applies the four directories to vanilla
v0.32.0 and fails if the result differs from a given trunk (scrubbed with a rules file) anywhere; see the header of the
script.

| patches (`patches/vllm/`) | what | author |
|---|---|---|
| 0001 | wu1ff's MIT license and pack notice, under `b70/third_party/wu1ff/` | Lumnus |
| 0002–0016 | wu1ff's 1.0.5 chain for vLLM 0.32 (target port, DFlash2, C1 verify/target graphs, astra500 MoE/HC/prefill paths, serving fixes), the `vllm/` hunks of his patch-chain taken unchanged, one commit per patch | wu1ff |
| 0017–0034 | our 0.30 layers ported: INT8 PLE from NVMe on wu1ff's host-prepared PLE (B70-L32-0013), thinking budget per effort, repetition stop, effort alias, dense-QSA indexer skip (main + MTP), MTP draft prefill graphs, uniform drafts guard, step trace, offload trace / junction / GDN backstep / empty-advance guard, prompt logprobs chunks, and their CPU tests | Lumnus |
| 0035–0037 | revert of our 0029b port, then vllm-project/vllm#63564 and #63566 | Lumnus |
| 0038–0049 | 0.32-specific: B70-V32-PR2, -01, -0053, -02/-02b (diagnostic), -03/-03b, -04, -07, -0054 (off), -08, -RC1 | Lumnus |
| 0050 | b70.4: backport of vllm-project/vllm#54163 (merged upstream 2026-10-23): `needs_eagle_block_drop()` excludes DFlash/DSpark, so the trailing prefix-cache block is no longer dropped for them and `disable_eagle_block_drop` is not needed. Applied by hand where the 0.32 line lacks `is_xpress`; its tests ported | Lumnus |

Off by default and not used in either preset: B70-V32-01 (`B70_C1_EAGER_ROUTE`), -02/-02b (`B70_PHASE_TIMING`), -04
(`B70_HC_PARALLEL_MAX_ROWS`, superseded by -07), -0054 (`B70_MOE_DECODE_OP_RESIDENT`; the astra 8-row path is faster at
decode; it loads the same library as the hot/cold decode op, build step 5).

### Hot/cold expert offload

[`patches/vllm-hotcold/`](engine/patches/vllm-hotcold/), 13 commits: 0001–0010 are Niluthon's hot/cold patches from
[qwen38-flash-tp2](https://github.com/Niluthon/qwen38-flash-tp2) @ `e536e33` (Apache-2.0), ported to 0.32 through our 0.30
carry and kept with his authorship; 0011–0013 are ours.

| patches | what |
|---|---|
| 0001 | B70-0040: routed experts split between VRAM (hot) and pinned host RAM (cold); the cold ones are read by the same int4 grouped GEMM through a UVA view, and the two outputs are summed (bit-exact against all experts in VRAM) |
| 0002, 0004, 0006, 0007 | B70-0041 route counter, B70-0043 swapping (the VRAM hot set follows the traffic), B70-0046 route trace (diagnostic), B70-0047 decode-aware ranking and a learned hot set that survives restarts |
| 0003 | B70-0042: cold-expert prefetch for prefill (double-buffered DMA into two VRAM slots, eager batches of ≥ 512 tokens) |
| 0005 | B70-0044: input embedding table in pinned host RAM (`B70_EMBED_UVA`, off in both presets) |
| 0008, 0009 | B70-0048 fused hot+cold decode MoE op (`B70_MOE_DECODE_OP`, the library from build step 5) and an FP8 MTP draft head (off; the presets use DFlash); B70-0049 the op's CPU pool switch (`B70_MOE_CPU=0` in the 2-card preset) |
| 0010 | their CPU tests |
| 0011 | B70-V32-HC1: `B70_MOE_COLD_EXPERTS=N` (cold per layer per rank, so N means the same VRAM saving at TP4 and TP2), the 0.32 remap of unrouted ids as a helper, CPU tests |
| 0012 | B70-V32-HC2: drop two upstream tests that do not apply to 0.32 |
| 0013 | B70-V32-HC3: cold prefetch never runs under graph capture. It hung the 512/1024-token captures; capture now reads the cold experts over UVA at any size, and the prefetcher raises if it is called while the stream is capturing |

Off unless `B70_MOE_HOTCOLD=1`: with it unset nothing changes, which is how the 4-card preset runs.

### The models layer

[`patches/vllm-models/`](engine/patches/vllm-models/), 8 commits. Every switch is off unless its key is set (0008 acts only
when the backstep is on), so the Flash-Next presets run the b70.4 code paths apart from the asynchronous swap they turn on.

| patch | what | switch |
|---|---|---|
| 0001 | B70-Q27-0001: the Qwen3.5/3.8 lm_head as FP8 W8A16 (row scale), converted at the first forward; a DFlash2/MTP drafter that shares the head reads the same FP8 weights. Covers Qwen3.8-27B and Qwen3.6-35B-A3B (`Qwen3_5ForCausalLMBase`); Flash-Next has its own | `B70_HEAD_FP8=1` |
| 0002, 0003 | backports of vllm-project/vllm#64190 (small-M tiles for the Triton W8A8 INT8 GEMM on XPU: 408 → 555 GB/s at M = 1 on a B70) and #64244 (no register spills above M = 64), by Katsumi Takeuchi | path (int8 W8A8 only) |
| 0004 | the "unannotated draft KV group" warning only when block drop needs it (it fired on every DFlash boot over a hybrid target) | log only |
| 0005 | B70-Q27-0003: asynchronous hot/cold expert swap: side-stream copies and routing flips, one phase per step, no host synchronisation; spare cold rows per layer; CPU test step-for-step identical to the synchronous swap when off | `B70_MOE_HOTCOLD_ASYNC_SWAP=1` (+ `_SPARE`) |
| 0006 | B70-Q27-0006: a shortlist draft head for DFlash2 on Qwen3.5/3.8 targets: the drafter's candidate top-k over a token list; the target verifies against the full vocabulary, so the output distribution is unchanged (CPU proofs: exactness, chi-square, TP1/TP2) | `B70_DFLASH_DRAFT_VOCAB=<file>` |
| 0007 | B70-Q27-0004: native XPU int8 activation quantisation (W8A8 and INC w4a8) when the kernels have the op, and an oneDNN W8A8 int8 GEMM ahead of Triton | op present; `B70_XPU_INT8_ONEDNN=1` for the GEMM |
| 0008 | B70-XC: the GDN backstep (0014c) is a no-op under `mamba_cache_mode=align`, with a one-time log line | `B70_GDN_BACKSTEP_ALIGN_GUARD=0` opts out |

## Software

| | version |
|---|---|
| GPU driver | Linux xe (kernel 6.18 on our host) |
| compute-runtime (Level Zero) | 26.35.39758.10 |
| oneAPI compiler (builds only) | 2026.1.1 (`icpx`) |
| Python | 3.12 |
| torch / triton | 2.14.0+xpu / 3.8.0+xpu |
| Intel runtime wheels | intel-sycl-rt, dpcpp-cpp-rt, intel-opencl-rt 2026.1.1; onemkl-sycl 2026.1.0 |
| oneCCL | 2022.1.2 (pip `oneccl`) core and loader; both presets use the legacy plugin rebuilt with oneCCL#229 (build step 6); the stock plugin is a parameter |
| vllm-xpu-kernels | **0.1.16.2 + five patches** (build step 3): the compiled libraries are built from upstream `6b30c9b` + [`patches/vllm-xpu-kernels/`](engine/patches/vllm-xpu-kernels/) (our #660 and #662; the native per-token int8 activation quantisation and oneDNN W8A8 GEMM ops with their precise-division build flag; vllm-project/vllm-xpu-kernels#605, the grouped-GEMM scale-prefetch pitch guard the 35B needs) with FlashAttention-2, MQA-logits and Xe3P off; the attention libraries and the Python files come from the PyPI 0.1.16.2 wheel; the astra Python modules and hooks from [`patches/vllm-xpu-kernels-py/`](engine/patches/vllm-xpu-kernels-py/) (step 4). Plus our `libgdn_index64.so` (step 5): the engine's GDN op, with 64-bit conv-state offsets at every site. Upstream #658 (the same 64-bit fix) is merged on main on 2026-10-26 and in no release |
| native libraries | wu1ff's B70-LLM-Controller 1.0.5 libraries, the Level Zero peer-residency shim, `libgdn_index64.so`, and Niluthon's `libb70_moe_decode.so` (Flash-Next 2-card preset), all built in step 5 |
| other pins | transformers 5.17.0, auto-round-lib 0.16.0 |
| vLLM | v0.32.0 + the four directories above (fork tag `v0.32.0-b70.5`) |

**Which upstream commit 0.1.16.2 is.** There is no GitHub tag for 0.1.16.2. The PyPI wheel's seven `.py` files match upstream
byte for byte at `8b95ea6`, `6b30c9b` and `19c6ddc` and at nothing later; its paged-decode table has the rows that #566
(`6b30c9b`) adds, which rules out `8b95ea6`; `19c6ddc` changes only a CI file. So the source tree is `6b30c9b` (high
certainty for the tree, medium for the exact commit: upstream may have built from an untagged copy of it).

`/dev/shm`: no requirement (B70-V32-RC1). Without RC1, vLLM 0.32 maps the whole CPU KV tier into `/dev/shm`, so a
container would need `/dev/shm` ≥ 140 GiB for a 128 GiB tier.

## Host RAM with three engines

Measured on our 256 GB host (251.5 GiB MemTotal) on 2026-10-30, Flash-Next 2-card with its 64 GiB tier, the 35B on one card and
the 27B on one card, each with a 32 GiB tier:

| | GiB |
|---|---|
| MemAvailable with Flash-Next 2-card alone | 118.6 |
| the 35B 1-card engine takes | ~42 |
| the 27B 1-card engine takes | ~36 |
| MemAvailable with all three serving | 41–43 |
| boot dips | ≤ 0.5 (35B), ~1.5 (27B) |

Every tier is pinned memory: it does not count against a container memory limit, so watch the host's MemAvailable.

## Build and run

The commands we ran, with our paths replaced by placeholders; the engine and environment were checked file by file against
these steps.

1. **Base.** compute-runtime 26.35.39758.10 on the host, the xe driver, and oneAPI DPC++ 2026.1.1 for the builds.

2. **vLLM and the Python environment.**
   ```bash
   git clone -b v0.32.0-b70.5 https://github.com/Lumnus/vllm.git vllm-b70-0.32   # the engine: dense base + sparse + hot/cold + models
   uv venv --python 3.12 /opt/b70/venv
   uv pip install --python /opt/b70/venv/bin/python --index-strategy unsafe-best-match \
       --extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://wheels.vllm.ai/xpu/ \
       -r vllm-b70-0.32/requirements/xpu.txt torch==2.14.0+xpu oneccl==2022.1.2 auto-round-lib==0.16.0 transformers==5.17.0
   VLLM_TARGET_DEVICE=xpu uv pip install --python /opt/b70/venv/bin/python --no-build-isolation -e vllm-b70-0.32
   # triton's XPU launcher needs CL/cl.h; intel-sycl-rt ships only include/sycl. Check:
   printf '#include <sycl/detail/cl.h>\nint main(){}\n' | g++ -std=c++17 -I/opt/b70/venv/include -x c++ -fsyntax-only -
   ```
   Instead of cloning the tag you can apply the four patch directories to a v0.32.0 checkout in order
   (`git am v0.32/engine/patches/{vllm,vllm-qsa,vllm-hotcold,vllm-models}/*.patch`, in that order).

3. **vllm-xpu-kernels 0.1.16.2 with five patches.** Install the PyPI wheel (it brings the Python files and the attention
   libraries), then build the compiled libraries from source with [`patches/vllm-xpu-kernels/`](engine/patches/vllm-xpu-kernels/)
   (#660, #662, the native int8 ops, #605) and lay them over the installed ones:
   ```bash
   uv pip install --python /opt/b70/venv/bin/python vllm-xpu-kernels==0.1.16.2      # PyPI wheel, sha256 starts 86bfa26abbb6
   git clone https://github.com/vllm-project/vllm-xpu-kernels.git && cd vllm-xpu-kernels
   git checkout 6b30c9b92cc525c32729df988b5f918e7c5661a5 && git am <kit>/v0.32/engine/patches/vllm-xpu-kernels/*.patch
   source /opt/intel/oneapi/2026.1/oneapi-vars.sh   # plus gcc-toolset-13 on our builder
   VLLM_XPU_ENABLE_XE3P=OFF FA2_KERNELS_ENABLED=OFF MQA_LOGITS_KERNELS_ENABLED=OFF VLLM_XPU_ENABLE_ONEDNN=ON MAX_JOBS=6 \
     VLLM_VERSION_OVERRIDE=0.1.16.2+b70 /opt/b70/venv/bin/python setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
   # take only the eight compiled libraries; keep the wheel's attention libraries (not built here) and everything else
   K=/opt/b70/venv/lib/python3.12/site-packages/vllm_xpu_kernels; mkdir -p w && python3 -I -m zipfile -e dist/*.whl w
   for f in _C.abi3.so _moe_C.abi3.so _xpu_C.abi3.so xpumem_allocator.abi3.so libgdn_attn_kernels_xe_2.so \
            libgrouped_gemm_xe_2.so libgrouped_gemm_xe_default.so libmhc_kernels_xe_2.so; do install -m 0755 w/vllm_xpu_kernels/$f $K/$f; done
   ```
   Do not `pip install` the built wheel: with FlashAttention-2 and MQA-logits off it has no attention libraries
   (`libattn_kernels_xe_2/_xe_3`, `_vllm_fa2_C`, `libmqa_logits_kernels_xe_2`), and installing it would remove the PyPI
   ones. MAX_JOBS above 6 ran out of memory on a 64 GiB builder; the build takes ~36 min there. The kernel table must keep
   the paged-decode row `16,256,64,false,false,false` (a TP2 layout needs it; 0.1.16.2 has it).

4. **Kernels, Python side.** The astra modules the engine imports (`astra500_moe.py`, `astra_structural.py`,
   `astra_ownership.py`) and the astra hooks in `fused_moe_interface.py`, from
   [`patches/vllm-xpu-kernels-py/`](engine/patches/vllm-xpu-kernels-py/), written against the 0.1.16.2 Python files:
   ```bash
   cd /opt/b70/venv/lib/python3.12/site-packages && for p in <kit>/v0.32/engine/patches/vllm-xpu-kernels-py/*.patch; do git apply -p1 < "$p"; done
   ```
   The result is byte-identical to the validated environment (`fused_moe_interface.py` sha256 starts `3ef25b12`). It keeps
   0.1.16.2's own DP-padding mask, which sits after the astra early returns; every preset runs DP = 1, so there are no padding
   rows.

5. **Native libraries.**
   1. **wu1ff's.** From [wu1ff/B70-LLM-Controller](https://github.com/wu1ff/B70-LLM-Controller) @ `5e5cf43` (release
      `qwen3.8-flash-next-v1.0.5`, MIT), `model-packs/Qwen3.8-Flash-Next/patches/production/native`: build the libraries with
      the pack's own build scripts (icpx 2026.1.1, torch 2.14 headers from the venv) and install `libastra_*.so`,
      `libbeyond_prefill.so`, `libdominant_w4_layout.so`, `liblong_hc.so`, `liblong_prefill_v3.so`, `libserving_prefill.so`
      into `site-packages/vllm_xpu_kernels/`. Put the Level Zero peer-residency shim at `/opt/b70/libl0_peer_residency_shim.so`.
      Our builds are not byte-identical to wu1ff's accepted hashes (different build paths); the shim is.
   2. **`libgdn_index64.so`.** Build it the same way after applying [`patches/libgdn-index64/`](engine/patches/libgdn-index64/)
      to the pack's `dependencies/gdn-headers`, into `site-packages/vllm_xpu_kernels/`. vLLM 0.32 runs the GDN op from this
      library (`torch.ops._gdn_index64`; sha256 starts `aabd7ae6` in our environment). It imports `chunk_gated_delta_rule_xe2`
      and `l2norm` from the step-3 `libgdn_attn_kernels_xe_2.so`.
   3. **`libb70_moe_decode.so`** (the Flash-Next 2-card preset's hot/cold decode op). From
      [Niluthon/qwen38-flash-tp2](https://github.com/Niluthon/qwen38-flash-tp2) @ `e536e33`, `kernels/b70-moe-decode`:
      ```bash
      T=$(/opt/b70/venv/bin/python -I -c "import torch, os; print(os.path.dirname(torch.__file__))")
      icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg" -O3 -std=c++20 -shared -fPIC -ffast-math \
        -D_GLIBCXX_USE_CXX11_ABI=1 -I$T/include -I$T/include/torch/csrc/api/include -x c++ kernels/b70-moe-decode/b70_moe_decode.sycl \
        -o /opt/b70/libb70_moe_decode.so -L$T/lib -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_xpu -Wl,-rpath,$T/lib
      ```

6. **oneCCL legacy plugin** (used by every preset; at one card the drafter still calls a collective, and without the plugin
   block the first DFlash request segfaulted in oneCCL). For Flash-Next the stock plugin is a parameter. One file, built against
   the venv's shipped oneCCL 2022.1.2:
   ```bash
   git clone -b master-v2 https://github.com/uxlfoundation/oneCCL.git && cd oneCCL
   git fetch -q origin pull/229/head && git checkout FETCH_HEAD   # caeb329, on master-v2 c5108e7
   source /opt/intel/oneapi/compiler/2026.1/env/vars.sh
   V=/opt/b70/venv; mkdir -p vinc /opt/b70/oneccl-legacy && ln -sfn $V/include/oneapi vinc/oneapi
   icpx -w -fsycl -O2 -fPIC -shared -std=c++17 -DNDEBUG -Ivinc -Isrc -Iinclude \
       plugins/legacy/ccl_legacy.cpp src/debug.cpp -o /opt/b70/oneccl-legacy/libccl_legacy.so \
       -L$V/lib -lccl -l:libccl.so.2 -ldl -Wl,--no-undefined -Wl,-soname,libccl_legacy.so
   ```
   `CCL_PLUGIN` in the preset `.env` points the shipped loader at it; the venv stays untouched.

7. **Weights.** Per model, on its page: [Flash-Next](models/qwen38-flash-next/meta.md#weights) (snapshot, INT8 PLE table,
   serve dir, drafter, rankings seed), [Qwen3.8-27B](models/qwen38-27b/meta.md) (weights, drafter and its W8A8 conversion,
   draft vocab), [Qwen3.6-35B-A3B](models/qwen36-35b-a3b/meta.md) (weights, drafter). Each page pins the revisions; the
   27B and 35B directories carry the file lists with their object ids.

8. **Run.** One preset per card set; presets on disjoint cards run side by side (set `ZE_AFFINITY_MASK` and the port per
   engine).
   ```bash
   VENV=/opt/b70/venv v0.32/engine/serve.sh qwen38-flash-next preset-4card                  # or preset-2card
   VENV=/opt/b70/venv MODEL_PATH=/models/qwen3.8-27b/int4-autoround v0.32/engine/serve.sh qwen38-27b preset-1card
   DRY=1 v0.32/engine/serve.sh qwen36-35b-a3b preset-2card                                  # print the command only
   ```
   `serve.sh` sources the preset's `.env`, reads its `.args` (one `--flag value` per line), runs from the kit root so the chat
   template paths resolve, and puts the venv's own oneAPI 2026.1.1 runtime (`$VENV/lib`) first on `LD_LIBRARY_PATH` (a mixed
   runtime segfaults at exit). Extra arguments are appended and win. The first start compiles cold (Flash-Next with the QSA
   Triton kernels: 7 min in our first boot, ~4–5 min after; the 27B and 35B on one card: 2–3 min).

## What the 0.32 engine tunes (Flash-Next)

The levers below are set in the presets. Each was found and measured on Flash-Next (a 0.32 build without them was ~25 %
slower at one stream), largest first:

1. **compute-runtime copy-through-lock.** Every small per-step host→device copy waited for the in-order queue to
   drain, so host preparation never overlapped the GPU: about 9.5 ms per step at one stream (found with phase timing
   and `py-spy --native`: the time was spent in `zeCommandListAppendMemoryCopy`). Fix: the compute-runtime keys
   `ExperimentalCopyThroughLock=0`, `ExperimentalH2DCpuCopyThreshold=0`, `ExperimentalD2HCpuCopyThreshold=0`.
2. **The DFlash2 fixed-order hyper-connection down GEMM** is 4.5–6.7× slower than `F.linear` above 32 rows. It was the
   whole prefill gap and most of the 8-stream gap. Fix: B70-V32-07, `B70_HC_ORDERED_MAX_ROWS=32`.
3. **The prefill hyper-connection graph regions** held ~0.7 GiB per card. Fix: B70-V32-08, `B70_PREFILL_HC_GRAPHS=0`.
   Together with the FP8 head (B70-V32-0053), the KV cache went from 337K to 389K tokens.
4. **PLE n-gram ids hashed on the host** (B70-V32-03): one small device→host copy replaces ~50 eager hash kernels
   (26.7 → 25.2 ms per step).
5. **K = 7 is the drafter's tuned shape.** Our earlier 0.32 runs used MTP k=3 or DFlash K=3.
6. A measurement lesson: GPU timing events on the host's critical path (B70-V32-02, diagnostic, off by default) cost
   ~3 ms per step themselves. Every number above comes from runs with them off.

Then two host-memory fixes:

7. **B70-V32-RC1, `B70_OFFLOAD_PRIVATE_PINNED=1`.** vLLM 0.32 keeps the CPU KV tier in a shared `/dev/shm` region
   registered with the device. SYCL reports registered memory as `usm::alloc::unknown`, so every CPU→GPU load was
   staged through torch's caching host allocator, and those blocks stayed cached: ~3.25 GiB per rank, ~13 GiB of host
   RAM. A private pinned pool per rank (as on 0.30) is copied directly. It also removes the need for a `/dev/shm` larger than
   the tier (≥ 140 GiB for 128 GiB). The upstream form of the kernel-side half is vllm-project/vllm-xpu-kernels#662.
8. **Expandable segments with a patched oneCCL plugin.** torch's `expandable_segments` puts XPU tensors in Level Zero
   virtual memory, which compute-runtime 26.35 reserves with a CPU mapping on Battlemage. oneCCL's legacy plugin
   classifies a buffer as host memory when `msync()` succeeds on it, sends it to the CPU collective without a stream,
   and libfabric then copies the device address on the host and segfaults. The plugin fix (uxlfoundation/oneCCL#229)
   asks SYCL for the pointer type when a queue is given. It buys ~8K KV tokens (~2 %) here, at the price of a hand-built `.so` tied to the oneCCL 2022.1.2 wheel and five
   `CCL_SYCL_*_TMP_BUF=1` copies, so it is a parameter. The presets use it; without the patched
   plugin, do not enable expandable segments (the stock-plugin configuration of b70.1–b70.3 did not; see [Parameters](models/qwen38-flash-next/meta.md#parameters)).

Upstream fixes folded in (all six are our posted PRs; #658 merged on main on 2026-10-26, the rest open):

| fix | upstream | here |
|---|---|---|
| GDN conv-state offset in 64 bits (an int32 wrap at block id ≥ 5,042 killed engines) | vllm-xpu-kernels #658 (fixes #657; merged on main 2026-10-26, in no release, so not in 0.1.16.2) | our `libgdn_index64.so` (the engine's GDN op; all conv-state sites 64-bit) |
| GDN kernels skip rows whose state index is the null block (graph padding rows wrote state into the shared null page) | vllm-xpu-kernels #660 (fixes #659; open) | our kernel build ([`patches/vllm-xpu-kernels/`](engine/patches/vllm-xpu-kernels/) 0001) and `patches/libgdn-index64/` (our library) |
| `swap_blocks_batch`: copy pinned (and registered) host sources directly, `snapshot_src=False` | vllm-xpu-kernels #662 (fixes #661; open) | our kernel build ([`patches/vllm-xpu-kernels/`](engine/patches/vllm-xpu-kernels/) 0002) + the caller, B70-V32-PR2 |
| Model Runner V2 prompt-logprobs chunks bounded by `max_num_reqs` (a 1024-row chunk spilled VRAM into host RAM) | vLLM #63564 (fixes #63563) | cherry-picked; replaces our 0029b |
| KV offload: pin a shared-prefix junction when a hybrid model's sparse group cuts the offload hit | vLLM #63566 (fixes #63565) | cherry-picked; replaces our 0014b |
| oneCCL legacy plugin: device USM that `msync()` sees as host takes the GPU path | oneCCL #229 (fixes #228) | plugin build, step 6 |

## Lessons learned

What works, what does not, and how to tune and test this engine: [docs/lessons-learned.md](../docs/lessons-learned.md).
