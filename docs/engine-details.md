# Engine details: the series patch by patch

What each directory of [`v0.32/engine/patches/`](../v0.32/engine/patches/) contains. The engine page ([v0.32/README.md](../v0.32/README.md)) has the
software versions, the build and the run commands; every switch is in [switches](switches.md).

## The series

The engine is fork tag `v0.32.0-b70.6`, exported as five directories that apply in order on vLLM v0.32.0 (`git am`, or
`git apply` one after the other); `v0.32/engine/scripts/check-series.sh --all` fails if any of them differs from a fresh export of the
fork. The fork's other tags mark where each layer ends:

| directory | export of | commits |
|---|---|---|
| [`patches/vllm/`](../v0.32/engine/patches/vllm/) | `v0.32.0..v0.32.0-b70.4-dense` | 50: the dense base (below) |
| [`patches/vllm-qsa/`](../v0.32/engine/patches/vllm-qsa/) | `v0.32.0-b70.4-dense..v0.32.0-b70.4-qsa` | 38: sparse attention ([Sparse attention](measurements/flash-next-earlier-builds.md#sparse-attention-qsa)) |
| [`patches/vllm-hotcold/`](../v0.32/engine/patches/vllm-hotcold/) | `v0.32.0-b70.4-qsa..v0.32.0-b70.4` | 13: hot/cold expert offload ([below](#hotcold-expert-offload)) |
| [`patches/vllm-models/`](../v0.32/engine/patches/vllm-models/) | `v0.32.0-b70.4..v0.32.0-b70.5` | 8: the 27B/35B layer and the backstep guard ([below](#the-models-layer)) |
| [`patches/vllm-v6/`](../v0.32/engine/patches/vllm-v6/) | `v0.32.0-b70.5..v0.32.0-b70.6` | 51: INT8 execution and KV, in-graph PLE gather, kernel cuts, fast sampler ([below](#the-v6-layer)) |

`v0.32/engine/tools/check-trunk-tree.sh` is the tree-level counterpart for maintainers: it applies the five directories to vanilla
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

Off by default and not used in any preset: B70-V32-01 (`B70_C1_EAGER_ROUTE`), -02/-02b (`B70_PHASE_TIMING`), -04
(`B70_HC_PARALLEL_MAX_ROWS`, superseded by -07), -0054 (`B70_MOE_DECODE_OP_RESIDENT`; the astra 8-row path is faster at
decode; it loads the same library as the hot/cold decode op, [build step 5](../v0.32/README.md#build-and-run)).

### Hot/cold expert offload

[`patches/vllm-hotcold/`](../v0.32/engine/patches/vllm-hotcold/), 13 commits: 0001–0010 are Niluthon's hot/cold patches from
[qwen38-flash-tp2](https://github.com/Niluthon/qwen38-flash-tp2) @ `e536e33` (Apache-2.0), ported to 0.32 through our 0.30
carry and kept with his authorship; 0011–0013 are ours.

| patches | what |
|---|---|
| 0001 | B70-0040: routed experts split between VRAM (hot) and pinned host RAM (cold); the cold ones are read by the same int4 grouped GEMM through a UVA view, and the two outputs are summed (bit-exact against all experts in VRAM) |
| 0002, 0004, 0006, 0007 | B70-0041 route counter, B70-0043 swapping (the VRAM hot set follows the traffic), B70-0046 route trace (diagnostic), B70-0047 decode-aware ranking and a learned hot set that survives restarts |
| 0003 | B70-0042: cold-expert prefetch for prefill (double-buffered DMA into two VRAM slots, eager batches of ≥ 512 tokens) |
| 0005 | B70-0044: input embedding table in pinned host RAM (`B70_EMBED_UVA`, off in every preset) |
| 0008, 0009 | B70-0048 fused hot+cold decode MoE op (`B70_MOE_DECODE_OP`, the library from [build step 5](../v0.32/README.md#build-and-run)) and an FP8 MTP draft head (off; the presets use DFlash); B70-0049 the op's CPU pool switch (`B70_MOE_CPU=0` in the 2-card preset) |
| 0010 | their CPU tests |
| 0011 | B70-V32-HC1: `B70_MOE_COLD_EXPERTS=N` (cold per layer per rank, so N means the same VRAM saving at TP4 and TP2), the 0.32 remap of unrouted ids as a helper, CPU tests |
| 0012 | B70-V32-HC2: drop two upstream tests that do not apply to 0.32 |
| 0013 | B70-V32-HC3: cold prefetch never runs under graph capture. It hung the 512/1024-token captures; capture now reads the cold experts over UVA at any size, and the prefetcher raises if it is called while the stream is capturing |

Off unless `B70_MOE_HOTCOLD=1`: with it unset nothing changes, which is how the 4-card presets run.

### The models layer

[`patches/vllm-models/`](../v0.32/engine/patches/vllm-models/), 8 commits. Every switch is off unless its key is set (0008 acts only
when the backstep is on), so a preset that sets none of them runs the b70.4 code paths.

| patch | what | switch |
|---|---|---|
| 0001 | B70-Q27-0001: the Qwen3.5/3.8 lm_head as FP8 W8A16 (row scale), converted at the first forward; a DFlash2/MTP drafter that shares the head reads the same FP8 weights. Covers Qwen3.8-27B and Qwen3.6-35B-A3B (`Qwen3_5ForCausalLMBase`); Flash-Next has its own | `B70_HEAD_FP8=1` |
| 0002, 0003 | backports of vllm-project/vllm#64190 (small-M tiles for the Triton W8A8 INT8 GEMM on XPU: 408 → 555 GB/s at M = 1 on a B70) and #64244 (no register spills above M = 64), by Katsumi Takeuchi | path (int8 W8A8 only) |
| 0004 | the "unannotated draft KV group" warning only when block drop needs it (it fired on every DFlash boot over a hybrid target) | log only |
| 0005 | B70-Q27-0003: asynchronous hot/cold expert swap: side-stream copies and routing flips, one phase per step, no host synchronisation; spare cold rows per layer; CPU test step-for-step identical to the synchronous swap when off | `B70_MOE_HOTCOLD_ASYNC_SWAP=1` (+ `_SPARE`) |
| 0006 | B70-Q27-0006: a shortlist draft head for DFlash2 on Qwen3.5/3.8 targets: the drafter's candidate top-k over a token list; the target verifies against the full vocabulary, so the output distribution is unchanged (CPU proofs: exactness, chi-square, TP1/TP2) | `B70_DFLASH_DRAFT_VOCAB=<file>` |
| 0007 | B70-Q27-0004: native XPU int8 activation quantisation (W8A8 and INC w4a8) when the kernels have the op, and an oneDNN W8A8 int8 GEMM ahead of Triton | op present; `B70_XPU_INT8_ONEDNN=1` for the GEMM |
| 0008 | B70-XC: the GDN backstep (0014c) is a no-op under `mamba_cache_mode=align`, with a one-time log line | `B70_GDN_BACKSTEP_ALIGN_GUARD=0` opts out |

### The v6 layer

[`patches/vllm-v6/`](../v0.32/engine/patches/vllm-v6/), 51 commits. Every lever is off unless a preset sets it, except the three KV defaults marked
below; [switches](switches.md) lists each key.

| patches | what | switch |
|---|---|---|
| 0001–0004 | B70-IP-K1 to K3: per-token-head KV with pointer loads instead of tensor descriptors, no fill of the scale views (backport of vllm-project/vllm#64141), the KV page unify takes the integer block multiple, CPU tests | defaults on: `B70_PTH_TD_GUARD`, `B70_KV_UNIFY_MULTIPLE` |
| 0005, 0007 | W1: BF16 dense linears as W8A8 int8 (RTN per channel, per-token activations; Triton up to 64 rows, oneDNN above), tests | `B70_DENSE_INT8=1` |
| 0006, 0008 | K4, K5: INT8 KV prefill through the XPU FlashAttention kernel (dequantised to a bf16 scratch above 64 query rows) | `B70_PTH_FA_PREFILL` (on) |
| 0009–0014 | K6 to K8: the QSA kernels read and write `int8_per_token_head` KV; native int8 Q.K and P.V DPAS; int8-dot attn2 faster than bf16 | `B70_QSA_I8DOT=1/2` |
| 0015 | H1: lm_head as INT8 W8A8, native int8 DPAS | `B70_HEAD_INT8=1` |
| 0016–0018 | W4A8 routed experts for the XPU int4 MoE: Triton form, measured kernel shape, then the native cutlass W4A8 grouped GEMM | `B70_MOE_W4A8=1/2` |
| 0019–0021 | upstream picks: Triton draft-budget kernel (vllm-project/vllm#63421), DFlash/DSpark JIT warmup gaps (#63819), XPU adaptive verification without a confidence head (#62244); they act only with adaptive verification, which is off | none |
| 0022–0024 | D2: native INT8 per-token-head KV decode on XPU FlashAttention, tests, a local-import fix | `B70_PTH_FA_DECODE=1` |
| 0025, 0027, 0028 | riceharvest's draft int4 MTP lane and 64K draft vocab, the asymmetric-GPTQ loader (inert unless such a checkpoint loads), mixed GDN steps without a host sync | all off |
| 0026 | warm-up of the top-k/top-p and greedy rejection kernels for speculative decoding (vllm-project/vllm#59260, qizixi) | none |
| 0029–0032 | backport of vllm-project/vllm#62113 (16-byte aligned per-token-head KV slots) and the PAD16 follow-ups in TRITON_ATTN, the QSA views and the split views | default on: `B70_PTH_PAD16` |
| 0033, 0034, 0039 | INT8 PLE: preload of the NVMe row cache, the table in pinned host RAM, the producer on the device inside the graph | `B70_PLE_INT8_NVME_PRELOAD`, `B70_PLE_INT8`, `B70_PLE_DEVICE_GATHER` |
| 0035–0038, 0040 | V7: drafter context K/V through one KV-only int8 GEMM, drafter attention on the INT8 decode, int8 KV prefill dequant without host reads, DFlash2 aux features reuse the next layer's combine, tests | `B70_DFLASH_KV_INT8`, `B70_DRAFT_FA_INT8`, `B70_PTH_FA_PREFILL_FAST`, `B70_DFLASH2_AUX_REUSE` |
| 0041 | torch.compile for XPU Qwen4Exp (experiment, carried unchanged) | `B70_TORCH_COMPILE=1` |
| 0042–0045 | the kernel cut: EP sentinel and rows-per-expert zero in one Triton kernel, in-place XCCL all-reduce at caller-owned sites, `hc_silu` folded into the HC down finish, CPU equivalence tests | `B70_MOE_KCUT`, `B70_AR_INPLACE`, `B70_HC_SILU_FUSE` |
| 0046–0050 | glue for the whole-wheel kernels: fused small-M int8 dense, HC down on the W8A8 GEMM with deterministic split-K, W4A8 apply with the fused gemm1 epilogue, int8 activation quantisation folded into the producers (K2), per-(token, stream) HC scales | `B70_DENSE_INT8_FUSED`, `B70_HC_DOWN_INT8`, `B70_I8_PRODUCER_QUANT` (all off) |
| 0051 | top-k-first temperature sampling and speculative verification | `B70_SAMPLER_FAST=1` |

**Which upstream commit 0.1.16.2 is.** There is no GitHub tag for 0.1.16.2. The PyPI wheel's seven `.py` files match upstream
byte for byte at `8b95ea6`, `6b30c9b` and `19c6ddc` and at nothing later; its paged-decode table has the rows that #566
(`6b30c9b`) adds, which rules out `8b95ea6`; `19c6ddc` changes only a CI file. So the source tree is `6b30c9b` (high
certainty for the tree, medium for the exact commit: upstream may have built from an untagged copy of it).

`/dev/shm`: no requirement (B70-V32-RC1). Without RC1, vLLM 0.32 maps the whole CPU KV tier into `/dev/shm`, so a
container would need `/dev/shm` ≥ 140 GiB for a 128 GiB tier.

## Host RAM with three engines

Measured on our 256 GB host (251.5 GiB MemTotal) on 2026-10-30 with the b70.5 presets, Flash-Next 2-card with a 64 GiB tier, the 35B on one card and
the 27B on one card, each with a 32 GiB tier:

| | GiB |
|---|---|
| MemAvailable with Flash-Next 2-card alone | 118.6 |
| the 35B 1-card engine takes | ~42 |
| the 27B 1-card engine takes | ~36 |
| MemAvailable with all three serving | 41–43 |
| boot dips | ≤ 0.5 (35B), ~1.5 (27B) |

Every tier is pinned memory: it does not count against a container memory limit, so watch the host's MemAvailable.

## What the engine tunes (Flash-Next)

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
   plugin, do not enable expandable segments (the stock-plugin configuration of b70.1–b70.3 did not; see [earlier builds](measurements/flash-next-earlier-builds.md)).

Upstream fixes folded in (the six that are our posted PRs, as of 2026-11-03; [upstream](upstream.md) has the status of each):

| fix | upstream | here |
|---|---|---|
| GDN conv-state offset in 64 bits (an int32 wrap at block id ≥ 5,042 killed engines) | vllm-xpu-kernels #658 (fixes #657; merged on main 2026-10-26, in no release, so not in 0.1.16.2) | our `libgdn_index64.so` (the engine's GDN op; all conv-state sites 64-bit) |
| GDN kernels skip rows whose state index is the null block (graph padding rows wrote state into the shared null page) | vllm-xpu-kernels #660 (fixes #659; merged on main) | our kernel build ([`patches/vllm-xpu-kernels/`](../v0.32/engine/patches/vllm-xpu-kernels/) 0001) and `patches/libgdn-index64/` (our library) |
| `swap_blocks_batch`: copy pinned (and registered) host sources directly, `snapshot_src=False` | vllm-xpu-kernels #662 (fixes #661; open) | our kernel build ([`patches/vllm-xpu-kernels/`](../v0.32/engine/patches/vllm-xpu-kernels/) 0002) + the caller, B70-V32-PR2 |
| Model Runner V2 prompt-logprobs chunks bounded by `max_num_reqs` (a 1024-row chunk spilled VRAM into host RAM) | vLLM #63564 (fixes #63563) | cherry-picked; replaces our 0029b |
| KV offload: pin a shared-prefix junction when a hybrid model's sparse group cuts the offload hit | vLLM #63566 (fixes #63565) | cherry-picked; replaces our 0014b |
| oneCCL legacy plugin: device USM that `msync()` sees as host takes the GPU path | oneCCL #229 (fixes #228) | plugin build, [build step 6](../v0.32/README.md#build-and-run) |

## Lessons learned

What works, what does not, and how to tune and test this engine: [lessons-learned.md](lessons-learned.md).
