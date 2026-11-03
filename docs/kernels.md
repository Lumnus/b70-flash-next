# Our vllm-xpu-kernels builds

Fork: [Lumnus/vllm-xpu-kernels](https://github.com/Lumnus/vllm-xpu-kernels). Five builds:

| build | for | source | state |
|---|---|---|---|
| **`0.1.16.2+b70.6`, a whole wheel** | torch 2.14, oneAPI 2026.1.0 (the current 0.32 line, b70.6) | upstream `6b30c9b` + [`v0.32/engine/patches/vllm-xpu-kernels/`](../v0.32/engine/patches/vllm-xpu-kernels/), 15 commits (below); the same 15 as commits: branch [`b70/v0.1.16-b70.6`](https://github.com/Lumnus/vllm-xpu-kernels/tree/b70/v0.1.16-b70.6) (tree `d87547a135efb1fb2f1a5d4388af82dfeb708c79`, identical to `git am` of the patches); built with sycl-tla pull 859 ([`v0.32/engine/patches/sycl-tla/`](../v0.32/engine/patches/sycl-tla/)) for the fast fp16 + e4m3 KV path; FlashAttention-2, MQA-logits, oneDNN on, Xe3P off; installed whole, with its own attention libraries | **the current line's kernels**; recipe in [the engine page](../v0.32/README.md#build-and-run) step 3 |
| `0.1.16.2` + five patches, + `libgdn_index64` (b70.5) | torch 2.14, oneAPI 2026.1.1 (b70.5; b70.4 had the first two patches) | upstream `6b30c9b` + the first five of the 15 patches above (#660, #662, the native per-token int8 activation quantisation and oneDNN W8A8 GEMM ops with their precise-division flag, vllm-project/vllm-xpu-kernels#605); the same five as commits: branch [`b70/v0.1.16`](https://github.com/Lumnus/vllm-xpu-kernels/tree/b70/v0.1.16) @ `4473199c` (tree `49952057`); FlashAttention-2, MQA-logits and Xe3P off, the eight compiled libraries laid over the PyPI wheel | the build behind the 27B and 35B numbers |
| `release/0.1.15.4` + our PRs (retired in b70.4) | torch 2.14, oneAPI 2026.1.1 (the 0.32 line through b70.3) | upstream `release/0.1.15.4` (`ddf336d`) + vllm-xpu-kernels #660, #658, #662 from their pull refs, oneDNN on, plus wu1ff's Python hunks | served through b70.3; the replay, long-prefill and sparse-against-dense numbers of b70.1–b70.4 were taken on it |
| **`0.1.14.1+b70.3`** | torch 2.13, oneAPI 2026.0 (the previous 0.30 line) | tag [`v0.1.14.1+b70.3`](https://github.com/Lumnus/vllm-xpu-kernels/tree/v0.1.14.1%2Bb70.3) on branch `b70/v0.1.14` = upstream tag `0.1.14.1` (`6d92b1bf`) + 7 commits | the 0.30 line's kernels (served 2026-10-03 to 2026-10-23); wheel attached to the [0.30.0-b70.2 release](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2) |
| `0.1.15.4+b70.1` | torch 2.14, oneAPI 2026.1 (the edge line) | [`b70/v0.1.15`](https://github.com/Lumnus/vllm-xpu-kernels/tree/b70/v0.1.15) @ `69b823f9` = upstream `release/0.1.15.4` + the 64-bit fix | ran on the edge line; not recommended ([known issues](../v0.30/docs/known-issues.md) item 5) |

## 0.1.16.2+b70.6: the series

[`v0.32/engine/patches/vllm-xpu-kernels/`](../v0.32/engine/patches/vllm-xpu-kernels/), one file per commit, 15 commits on upstream `6b30c9b`.
`git am` of the 15 files on `6b30c9b` gives the tree of the branch above. The first five are b70.5's.

| # | what | for |
|---|---|---|
| 1 | GDN: skip rows whose state index is `NULL_BLOCK_ID` (#660, merged upstream) | graph padding rows no longer write recurrent state into the shared null block |
| 2 | `swap_blocks_batch`: `snapshot_src=False` (#662) | pinned host sources copy directly |
| 3, 4 | native per-token int8 activation quantisation, oneDNN int8 W8A8 GEMM, precise fp32 division | INT8 execution |
| 5 | grouped GEMM scale-prefetch pitch guard (#605, Guancheng Fu) | the 35B's expert shapes |
| 6 | W4A8 grouped GEMM (s8 x s4 DPAS) | `B70_MOE_W4A8=2` |
| 7 | GDN conv-state offsets in 64 bits (#658, merged upstream) | block ids above 5,042 |
| 8 | spec-decode adaptive verification in FlashAttention (#647, Yan Ma, merged upstream) | verify batches with unequal query lengths |
| 9, 10 | INT8 per-token-head KV for paged decode, and its tests (wiesion) | `int8_per_token_head` KV |
| 11 | INT8 KV: head-size guard and a B70 check | |
| 12, 13 | int4 GEMV for decode-sized grouped GEMMs, and its test (riceharvest) | MoE decode |
| 14 | fast e4m3 to half KV conversion on Xe2 (riceharvest) | active only with sycl-tla #859 built in, fp16 Q and e4m3 K/V; the presets run bf16 Q, so it is inactive for them |
| 15 | no fallback to the reference attention during graph capture | a missing kernel configuration raises at capture; the whole wheel compiles every configuration, including the TP2 row `16,256,64,false,false,false` |

**Behaviour that changes without a flag.** (1) The int4 GEMV is on by default: int4 grouped GEMMs with 8 rows or fewer
(`K % 512 == 0`, group size a multiple of 32, no bias) take `moe_int4_gemv`; `VLLM_XPU_FUSED_MOE_INT4_GEMV=0` turns it off.
(2) The spec-decode fast path (#647) no longer requires a uniform query length. (3) #658 is on, with the same values where the
32-bit product did not overflow. The Qwen3.8-27B and Qwen3.6-35B-A3B numbers were taken before these, so they do not apply to
this build: see their pages.

**Build.** `setup.py bdist_wheel --py-limited-api=cp38` with oneAPI DPC++/C++ 2026.1.0, gcc-toolset-13, Python 3.12,
`VLLM_XPU_ENABLE_XE3P=OFF VLLM_XPU_ENABLE_ONEDNN=ON VLLM_XPU_XE2_AOT_DEVICES=bmg-g31-a0 VLLM_XPU_AOT_DEVICES=bmg-g31-a0
VLLM_VERSION_OVERRIDE=0.1.16.2+b70.6`, `VLLM_CUTLASS_SRC_DIR` pointing at a sycl-tla `87f6850` checkout with the two #859 patches. A cold
build took 5 h 27 min at 16 jobs and peaked near 120 GiB of RAM (1,360 compile calls); the wheel is about 205 MB. Without
#859 the build succeeds and the fast conversion stays off. The native libraries of wu1ff stay where the engine page's step 5 puts
them: they are not in the wheel.

## 0.1.14.1+b70.3: the series

`v0.30/patches/vllm-xpu-kernels/b70.3/`, one file per commit, exported with `git format-patch 0.1.14.1..v0.1.14.1+b70.3`.
`git am` of the seven files on upstream tag `0.1.14.1` gives exactly the tag's tree.

| # | commit | what | why we need it |
|---|---|---|---|
| 1 | `fa542f6` | GDN `causal_conv1d`: the conv-state pointer offset in 64 bits (7 sites) | Qwen3.8-Flash-Next keeps one conv state per cache block; `states_id * conv_states_stride_0` passes 2^31 at block id 5,042 (× 425,984) on 4× B70 and the engine dies with `DEVICE_LOST`. wu1ff's pack works around it with a closed library (`libgdn_index64.so`, selected by patch 0005); this is the same fix in source, so 0027 can run without that library |
| 2 | `ba19ef4` | upstream #600: ragged speculative token traversal | MTP + structured output: grammar-trimmed drafts give rows shorter than 1+k, which the 0.1.14 GDN kernel rejected (an engine crash on the first `json_schema` request) |
| 3 | `dd87485` | upstream #564: keep tensor strides when pinning non-pinned CPU tensors for a UVA view | UVA views of pinned host tensors keep their strides |
| 4 | `d00fbf7` | upstream #563: top-k/top-p sampler with an unaligned vocab size | sampler correctness |
| 5 | `5fcc722` | upstream #578: negative expert ids in the XPU MoE remap | with expert parallelism, padding routes read `expert_map[-1]` out of bounds |
| 6 | `e68951b` | upstream #586: initialise `atomic_buffer` to 0 | Xe2 grouped-GEMM tile-counter race (our MoE path) |
| 7 | `493364a` | **B70-K1**: `swap_blocks_batch` copies H2D straight from pinned (USM host) sources | every CPU→GPU KV load used to be staged through a pinned buffer the size of the load, which torch keeps in its host cache: host memory grew under a 128 GiB CPU KV tier until our RAM guard stopped the engine. Measured with it: +0.058 GiB per rank once, then flat; 19.6–19.9 GB/s (was 8–12); no guard stop ([measurements](../v0.30/docs/measurements/b70.2.md) §3). `VLLM_XPU_H2D_BATCH_STAGING=1` restores staging |

Commits 2–6 are cherry-picks of upstream merges (`-x`, so each carries its upstream commit id); credit goes to their
authors. B70-K1 changes the op's contract for pinned sources: the caller keeps the source unchanged until the copies
complete on the current stream (vLLM holds a reference on loaded blocks until the transfer event completes; CUDA's
batched copy assumes the same). We intend to propose it upstream as an opt-in.

**Build.** B70-only (`VLLM_XPU_ENABLE_XE3P=OFF`), oneAPI 2026.0 to match torch 2.13's SYCL runtime, `setup.py
bdist_wheel --py-limited-api=cp38`, `VLLM_VERSION_OVERRIDE=0.1.14.1+b70.3`. Our wheel: 351,983,263 B, sha256
`f104a3e5f61480681284ec3632e1a6692358b72a1fbdbda7406db0d7b7d1d589`, attached to the [0.30.0-b70.2 release](https://github.com/Lumnus/b70-flash-next/releases/tag/v0.30.0-b70.2)
(manylinux_2_28 x86_64, cp38-abi3; built against torch 2.13.0+xpu). Against the b70.2 build it differs
only in `_C.abi3.so`; its 106 op schemas match b70.2 and b70.1, and b70.1 was checked against the official 0.1.14.1 wheel. A full build takes ~2 h 40 min
at 12 jobs and peaks at ~100 GB of RAM; an incremental rebuild from b70.2 took under 3 min.

**In use.** With it there is no wu1ff library in the environment: 0027 then selects the stock `gdn_attention` op, which
is 64-bit with commit 1, and logs the fallback once (0032). Keep `B70_OFFLOAD_H2D_DIRECT=0` (commit 7 does the copy).

## 0.1.15.4+b70.1 (edge line)

1. **Source.** `release/0.1.15.4` (`ddf336d`), the version vLLM main pins, plus the 64-bit fix cherry-picked cleanly.
2. **Build.** Upstream's builder image (`pytorch/manylinux2_28-builder:xpu-v2.14.0-rc10`), B70-only, the full kernel
   presets (`libattn_kernels_xe_2.so` 1.63 GB against the official 307 MB). Peak ~94 GiB RAM at 12 jobs.
3. **Same API.** All 111 registered ops have identical schemas to the official 0.1.15.4 wheel.
4. **The fix is in the binary.** Disassembly of the BMG device code: 36 of 40 `causal_conv1d` variants gain the 64-bit
   product; the 4 unchanged ones have no conv-state access.
5. **In use.** Edge-line engines without it died with `DEVICE_LOST` on all four cards; with it they did not, and fail
   differently (known issues item 5).

`v0.30/patches/vllm-xpu-kernels/0001-gdn-causal-conv1d-int64-state-offset.patch` is the b70.1 form of commit 1, kept because
patch 0005 refers to it.
