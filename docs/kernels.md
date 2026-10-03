# Our vllm-xpu-kernels builds

Fork: [Lumnus/vllm-xpu-kernels](https://github.com/Lumnus/vllm-xpu-kernels). Two lines:

| build | for | source | state |
|---|---|---|---|
| **`0.1.14.1+b70.3`** | torch 2.13, oneAPI 2026.0 (the stable line) | tag [`v0.1.14.1+b70.3`](https://github.com/Lumnus/vllm-xpu-kernels/tree/v0.1.14.1%2Bb70.3) on branch `b70/v0.1.14` = upstream tag `0.1.14.1` (`6d92b1bf`) + 7 commits | **what we serve** (since 2026-10-03); no wheel published |
| `0.1.15.4+b70.1` | torch 2.14, oneAPI 2026.1 (the edge line) | [`b70/v0.1.15`](https://github.com/Lumnus/vllm-xpu-kernels/tree/b70/v0.1.15) @ `69b823f9` = upstream `release/0.1.15.4` + the 64-bit fix | ran on the edge line; not recommended (known issues item 5) |

## 0.1.14.1+b70.3: the series

`patches/vllm-xpu-kernels/b70.3/`, one file per commit, exported with `git format-patch 0.1.14.1..v0.1.14.1+b70.3`.
`git am` of the seven files on upstream tag `0.1.14.1` gives exactly the tag's tree.

| # | commit | what | why we need it |
|---|---|---|---|
| 1 | `fa542f6` | GDN `causal_conv1d`: the conv-state pointer offset in 64 bits (7 sites) | Qwen3.8-Flash-Next keeps one conv state per cache block; `states_id * conv_states_stride_0` passes 2^31 at block id 5,042 (× 425,984) on 4× B70 and the engine dies with `DEVICE_LOST`. wu1ff's pack works around it with a closed library (`libgdn_index64.so`, selected by patch 0005); this is the same fix in source, so 0027 can run without that library |
| 2 | `ba19ef4` | upstream #600: ragged speculative token traversal | MTP + structured output: grammar-trimmed drafts give rows shorter than 1+k, which the 0.1.14 GDN kernel rejected (an engine crash on the first `json_schema` request) |
| 3 | `dd87485` | upstream #564: keep tensor strides when pinning non-pinned CPU tensors for a UVA view | UVA views of pinned host tensors keep their strides |
| 4 | `d00fbf7` | upstream #563: top-k/top-p sampler with an unaligned vocab size | sampler correctness |
| 5 | `5fcc722` | upstream #578: negative expert ids in the XPU MoE remap | with expert parallelism, padding routes read `expert_map[-1]` out of bounds |
| 6 | `e68951b` | upstream #586: initialise `atomic_buffer` to 0 | Xe2 grouped-GEMM tile-counter race (our MoE path) |
| 7 | `493364a` | **B70-K1**: `swap_blocks_batch` copies H2D straight from pinned (USM host) sources | every CPU→GPU KV load used to be staged through a pinned buffer the size of the load, which torch keeps in its host cache: host memory grew under a 128 GiB CPU KV tier until our RAM guard stopped the engine. Measured with it: +0.058 GiB per rank once, then flat; 19.6–19.9 GB/s (was 8–12); no guard stop ([measurements](measurements/b70.2.md) §3). `VLLM_XPU_H2D_BATCH_STAGING=1` restores staging |

Commits 2–6 are cherry-picks of upstream merges (`-x`, so each carries its upstream commit id); credit goes to their
authors. B70-K1 changes the op's contract for pinned sources: the caller keeps the source unchanged until the copies
complete on the current stream (vLLM holds a reference on loaded blocks until the transfer event completes; CUDA's
batched copy assumes the same). We intend to propose it upstream as an opt-in.

**Build.** B70-only (`VLLM_XPU_ENABLE_XE3P=OFF`), oneAPI 2026.0 to match torch 2.13's SYCL runtime, `setup.py
bdist_wheel --py-limited-api=cp38`, `VLLM_VERSION_OVERRIDE=0.1.14.1+b70.3`. Our wheel: 351,983,263 B, sha256
`f104a3e5f61480681284ec3632e1a6692358b72a1fbdbda7406db0d7b7d1d589` (not published). Against the b70.2 build it differs
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

`patches/vllm-xpu-kernels/0001-gdn-causal-conv1d-int64-state-offset.patch` is the b70.1 form of commit 1, kept because
patch 0005 refers to it.
