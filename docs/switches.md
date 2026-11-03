# Runtime switches (`B70_*`)

The first sections cover the 0.30 series in `v0.30/patches/`, which the 0.32 line carries as ports; the last two cover the 0.32 line's own
switches, those of the b70.5 models layer and of the b70.6 `vllm-v6/` series. The preset files
([e.g. preset-d8-6-4card.env](../v0.32/models/qwen38-flash-next/preset-d8-6-4card.env)) explain each key they set inline, and
[the engine page](../v0.32/README.md#the-series) lists the series.

Every feature the b70 series adds is behind an environment variable and **off by default**. With none of them set,
the image behaves as vLLM v0.30.0 plus wu1ff's patch set (0001–0005), apart from the ungated parts listed at
the end. Set the variables in the serving container's environment; `vllm serve` flags are unchanged.

A variable that is not listed here is ignored. Older builds used a `LUMNUS_` prefix for the same switches; those
names are no longer read, so a deployment that still sets them runs with the feature off.

## PLE n-gram table (0006, 0007, 0008, 0013, 0013b)

| variable | default | patch | effect |
|---|---|---|---|
| `B70_PLE_DIRECT_PINNED` | unset (off) | 0006 | `1`: fill the pinned PLE slabs straight from the memory-mapped table file, per TP rank, instead of first loading a pageable copy per rank. Measured boot host-RAM peak on 4 ranks with the BF16 table: ~191 GiB → ~158 GiB. |
| `B70_PLE_FP8` | unset (off) | 0007 | `1`: keep the table as FP8 E4M3 (one global scale) in the pinned slabs; needs `B70_PLE_FP8_PATH`. Exclusive with `B70_PLE_INT8`. |
| `B70_PLE_FP8_PATH` | — | 0007 | `.safetensors` with a `table` tensor (or the checkpoint's `…shard_<i>.weight` keys) plus one `weight_scale`. |
| `B70_PLE_FP8_CROSSCHECK` | `1` | 0007 | `0` skips the boot cross-check against the BF16 table (only runs when `PLE_TABLE_PATH` is set). |
| `B70_PLE_FP8_MAX_REL_ERR` | `0.25` | 0007 | refuse to start above this relative L2 error on 4,096 random rows. |
| `B70_PLE_FP8_DEQUANT` | `lut` | 0007 | `cast` uses the stock cast-and-multiply instead of the 256-entry lookup table (bitwise equal). |
| `B70_PLE_INT8` | unset (off) | 0008 | `1`: keep the table as INT8, one fp32 scale per row (164 B/row, 48.9 GiB over 4 ranks instead of 95.4 GiB BF16); needs `B70_PLE_INT8_PATH`. Build the table with `tools/build_int8_ple.py` ([ple-int8.md](ple-int8.md)). |
| `B70_PLE_INT8_PATH` | — | 0008 | the INT8 table `.safetensors`. Its metadata must carry the format tag `lumnus-ple-int8-rowscale/v1` (see below). |
| `B70_PLE_INT8_CROSSCHECK` | `1` | 0008 | `0` skips the boot cross-check against the BF16 table. |
| `B70_PLE_INT8_MAX_REL_ERR` | `0.02` | 0008 | refuse to start above this relative L2 error (expected ~0.0066). |
| `B70_PLE_INT8_NVME` | unset (off) | 0013 | `1`: serve the INT8 table from NVMe through a small pinned row cache instead of pinning all of it ([ple-nvme.md](ple-nvme.md)). On the 0.32 line it replaces `B70_PLE_INT8` (setting both raises at init). |
| `B70_PLE_INT8_NVME_PATH` | `B70_PLE_INT8_PATH` | 0013 | the table file to read with `O_DIRECT`. |
| `B70_PLE_INT8_NVME_CACHE_GIB` | `8` | 0013 | pinned row cache, GiB **in total over all TP ranks**. The 2-card presets run `4` (2 GiB per rank); the hit rate on agent traffic was 97–98 %. |
| `B70_PLE_INT8_NVME_IO_THREADS` | `16` | 0013 | read threads per rank (also the io_uring queue depth with `READER=uring`). |
| `B70_PLE_INT8_NVME_STATS` | `0` | 0013 | `1`: per-rank stats line (hits, misses, reads, host bubble). |
| `B70_PLE_INT8_NVME_STATS_S` | `60` | 0013 | stats period, seconds. |
| `B70_PLE_INT8_NVME_SYNC_ONLY` | `0` | 0013 | diagnostic: keep the table in RAM but add the per-step host sync the NVMe path needs, to measure that cost alone. |
| `B70_PLE_INT8_NVME_BOOT_SAMPLE` | `65536` | 0013 | rows compared byte-for-byte against the memory-mapped file at boot. |
| `B70_PLE_INT8_NVME_READER` | `py` | 0013b | `native` = a C thread pool (one call per batch, GIL released); `uring` = io_uring via raw syscalls. `native` is the one we run. |
| `B70_PLE_INT8_NVME_LOOKAHEAD` | `0` | 0013b | `1`: read the next prefill chunk's rows in the background during the current step. |
| `B70_PLE_INT8_NVME_LOOKAHEAD_TOKENS` | scheduler's `max_num_batched_tokens` | 0013b | tokens predicted per step (unset or `0` = the default). |
| `B70_PLE_INT8_NVME_NATIVE_DIR` | `$TMPDIR/b70-ple-nvme` | 0013b | where the C reader is compiled at first use (needs `gcc`, present in the base image). |
| `B70_PLE_INT8_NVME_NATIVE_LIB` | — | 0013b | use this prebuilt reader `.so` instead of compiling. |

`PLE_TABLE_PATH` (wu1ff's variable, the BF16 table) is still read: it is the source table for BF16 serving and the
reference for the FP8/INT8 cross-checks.

**The INT8 table format tag stays `lumnus-ple-int8-rowscale/v1`.** It is a data-contract string written into the
table file's metadata by `tools/build_int8_ple.py` and compared by equality in 0008 and 0013; renaming it would make
every existing table unloadable. It is not a switch.

## Chat API defaults (0009, 0010, 0012)

| variable | default | patch | effect |
|---|---|---|---|
| `B70_THINKING_BUDGET` | unset (off) | 0009 | per-effort cap on reasoning tokens, keyed on the effort the **client** asked for: `minimal=512,low=512,medium=2048,high=4096,xhigh=8192,max=12288,ultra=12288,default=8192`. `default` applies when no effort was sent. Nothing applies when thinking is off (`enable_thinking=false` or an effort of `none`/`off`/`false`/`0`/`disabled`), and a request's own `thinking_token_budget` wins. |
| `B70_DEFAULT_PRESENCE_PENALTY` | unset | 0009 | server default for `presence_penalty`; a request's own value wins. The serving recipes set `0` (`common.env` sets `0.7` for the devan examples). |
| `B70_DEFAULT_REPETITION_DETECTION` | unset | 0010 | server default for vLLM's built-in n-gram repetition stop, `max=<n>,min=<n>,count=<n>` (we use `max=1,min=1,count=128`: stop when one token repeats 128 times). Catches the token-1023 "duct" NaN loop. A request's own `repetition_detection` wins. Chat endpoint only. |
| `B70_REASONING_EFFORT_ALIAS` | unset (off) | 0012 | `1`: map an OpenRouter-style `{"reasoning": {"effort": X}}` onto `reasoning_effort` when the request did not set that field. |

Effort-alias map (0012): `reasoning.effort` is lower-cased and trimmed; `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, `max` and `ultra` become `reasoning_effort` unchanged; any other value is ignored. The budget table (0009)
is then looked up with that effort.

## KV offload (0014a–f)

| variable | default | patch | effect |
|---|---|---|---|
| `B70_OFFLOAD_TRACE` | unset (off) | 0014a/f | `1`: boot line, a state line every period, and the mechanism events only (`junction-set`, `zeroed_by`), once per request per value. `2`: also every per-request lookup/store/load/hand-off/req-done line. Log prefix `B70-OFFLOAD`, logger `vllm.b70_offload`. |
| `B70_OFFLOAD_TRACE_PERIOD_S` | `60` | 0014a | state-line period (a daemon thread, so it also fires when idle). |
| `B70_OFFLOAD_JUNCTION` | unset (off) | 0014b | `1`: when a Mamba/GDN group cuts an offload hit below what the full-attention groups proved, pin a shared-prefix junction there so the recompute stores that state. A "same document, new question" revisit misses once, then hits (docs/offload-fix.md). |
| `B70_OFFLOAD_GDN_BACKSTEP` | `0` | 0014c | `N`: also retain the Mamba/GDN states at the N block boundaries below the replay boundary, so the first revisit hits too. Cost per request: N × one Mamba state set (~116 MB for Qwen3.8-Flash-Next) in the CPU tier. **Keep it 0 with prefix caching on a hybrid model:** a prefix-cache hit can restore a GDN state built through a different chunk split, and output diverges from the first token (CHANGELOG, 0.32.0-b70.5). |
| `B70_OFFLOAD_EMPTY_ADVANCE_GUARD` | unset (off) | 0014d | `1`: guard for vllm#56795 (do not advance past keys still write-pending under another request). |
| `B70_OFFLOAD_GROUP_EVICT` | unset (off) | 0014e/f | `1`: select the vllm#51787 backport (request-scoped recency, tail-before-head eviction across all KV groups). Unset = upstream v0.30.0 LRU/ARC, byte for byte. |

The offload-fix arm ("0014jb") was measured with `B70_OFFLOAD_TRACE=1 B70_OFFLOAD_JUNCTION=1 B70_OFFLOAD_GDN_BACKSTEP=1
B70_OFFLOAD_EMPTY_ADVANCE_GUARD=0 B70_OFFLOAD_GROUP_EVICT=0`; since 0.32.0-b70.5 every engine file ships
`B70_OFFLOAD_GDN_BACKSTEP=0` (row above).

## Ungated parts (active even with every switch unset)

1. 0009: `reasoning_effort` also accepts `"ultra"` (API widening only).
2. 0013: the KV-offload host-tensor allocation log line moves from DEBUG to INFO and prints `is_pinned()`.
3. 0014f: `ReqContext` records key positions (a dict write per key, never read with `B70_OFFLOAD_GROUP_EVICT` unset),
   and the policy base class gains optional hooks that are never called with the flag unset.
4. 0018: when one pinned allocation for the CPU KV pool is refused (≥ ~31 GiB on a B70), the pool is stored as
   equal chunks instead of failing at boot. A pool that fits in one allocation is unchanged.
5. 0019: on dense-QSA configs, checkpoint tensors under `self_attn.indexer` are skipped at load (checkpoints without
   them, such as devan's, are unaffected).

## Added in 0.30.0-b70.2 (0020–0032)

| variable | default | patch | effect |
|---|---|---|---|
| `B70_MTP_DRAFT_PREFILL_NO_PIECEWISE` | unset (off) | 0021 | `1`: on XPU, the MTP draft model keeps only FULL (uniform-decode) graphs; mixed prefill batches run the draft eagerly. We serve with `1` (the PIECEWISE draft prefill capture failed on XPU). |
| `B70_MTP_UNIFORM_DRAFTS` | `1` (on) | 0026 | never schedule a partial draft list on XPU (grammar-trimmed drafts crashed the stock 0.1.14 GDN kernel). `0` turns it off; with kernels that carry upstream #600 (our b70.2/b70.3) it is not needed. |
| `B70_GDN_MODE` | `index64` when `libgdn_index64.so` is present, else `official` | 0023, 0027 | which GDN op runs: `index64` (wu1ff's library), `official` (the stock op: 64-bit with our kernels, 32-bit with stock 0.1.14.1), `split-index64`, `split-official` (debug). Re-read every 2 s. |
| `B70_PROMPT_LOGPROBS_CHUNK` | `128` | 0029, 0029b | rows per prompt-logprobs chunk (bounds the [rows, vocab] fp32 transient). `0` = vLLM's own chunking (one pass on V1, 1,024 rows on V2). |
| `B70_OFFLOAD_H2D_DIRECT` | unset (off) | 0031 | `1`: CPU→GPU KV loads copy straight from the pinned offload pool instead of the kernel's staged batch copy. For kernels without B70-K1; with our 0.1.14.1+b70.3, leave it off. |

Debug tools in 0023/0023c (off unless their flag file exists): a NaN tracer in the GDN wrapper and at the sampling
boundary, and a probe that zeroes block 0 around FULL replays. The flag files are fixed paths
(`/run/b70-probes/gdn-mode`, `/run/b70-probes/gdn-nancheck`, `/run/b70-probes/zero-null-block`,
`/run/b70-probes/uniform-drafts-off`), re-read every 2 s; with no such files the code does nothing. (Before the
0.30.0-b70.2 release the fork branches used another directory; only the path changed.)

Ungated in b70.2: 0020 and 0022 (MTP loader indexer skip; vllm#55506 port), 0028 (acts only on an INC config), 0032
(log once). Every patch that adds a log line logs once per process when it first acts (e.g. `B70-0029b`, `B70-0031`),
so a deployment can confirm what is active.

## Switches added in 0.32.0-b70.5 (0.32 line, `v0.32/engine/patches/vllm-models/`)

| variable | default | patch | effect |
|---|---|---|---|
| `B70_HEAD_FP8` | unset (off) | 0001 | `1` on XPU: the Qwen3.5/3.8 lm_head (Qwen3.8-27B, Qwen3.6-35B-A3B) as FP8 W8A16 with a row scale, converted at the first forward; a drafter that shares the head reads the same FP8 weights. Flash-Next has its own form of the switch in the dense base. Skipped with a warning when the head is tied or not BF16 |
| `B70_MOE_HOTCOLD_ASYNC_SWAP` | unset (off) | 0005 | `1` (with `B70_MOE_HOTCOLD_SWAP=1`): the hot/cold swap as side-stream copies and routing flips, one phase per runner step, without a host synchronisation |
| `B70_MOE_HOTCOLD_ASYNC_SPARE` | the swap budget | 0005 | free cold rows per layer for the asynchronous swap (pinned host RAM); also caps the swaps per layer and cycle |
| `B70_DFLASH_DRAFT_VOCAB` | unset (off) | 0006 | `<file of token ids>`: the DFlash2 drafter of a Qwen3.5/3.8 target scores only these tokens for its candidates; the target verifies against the full vocabulary, so the output distribution is unchanged |
| `B70_XPU_INT8_QUANT_NATIVE` | on when the kernels have the op | 0007 | `0`: the torch.compile reference for per-token int8 activation quantisation instead of the native op (same bits) |
| `B70_XPU_INT8_ONEDNN` | unset (off) | 0007 | `1`: W8A8 int8 GEMMs (compressed-tensors) on oneDNN `int8_scaled_mm` ahead of Triton, symmetric activations only. `B70_INT8_GEMM_SCALE_MODE=postop` moves the per-token scale into a post-op |
| `B70_GDN_BACKSTEP_ALIGN_GUARD` | on | 0008 | `0`: let `B70_OFFLOAD_GDN_BACKSTEP` act under `mamba_cache_mode=align` again (not recommended) |

## Switches added in 0.32.0-b70.6 (`v0.32/engine/patches/vllm-v6/`)

Every one is off unless a preset sets it, except the four marked "on". The preset columns: D8 and S8 are the INT8-execution presets, D16e and S16e the BF16-execution ones; the 2-card presets are D8 and S8.

**Set by the presets**

| variable | default | D8 / S8 / D16e / S16e | effect |
|---|---|---|---|
| `B70_DENSE_INT8` | `0` | 1 / 1 / 0 / 0 | BF16 dense linears converted at load to int8 per channel, run W8A8 with per-token int8 activations (Triton up to 64 rows, oneDNN above) |
| `B70_HEAD_INT8` | `0` | 1 / 1 / 0 / 0 | the lm_head as INT8 W8A8 on native int8 DPAS; wins over `B70_HEAD_FP8`, and a drafter sharing the head uses it |
| `B70_XPU_INT8_ONEDNN` | `0` | 1 / 1 / 0 / 0 | the oneDNN W8A8 GEMM ahead of Triton (b70.5 key) |
| `B70_MOE_W4A8` | `0` | 2 / 2 / 0 / 0 | routed experts as W4A8: `1` Triton kernels, `2` the native cutlass grouped GEMM in the kernels wheel. `B70_W4A8_LIB_DIR` is not read in mode 2 |
| `B70_MOE_W4A8_MIN_TOKENS` | `64` | 1 / 1 / unset / unset | batches with fewer tokens stay on W4A16 |
| `B70_DFLASH_KV_INT8` | `0` | 1 / 1 / unset / unset | drafter context K/V through one KV-only int8 GEMM (bit-identical K/V) |
| `B70_PTH_FA_DECODE` | `0` | 1 / unset / 1 / unset | FlashAttention on XPU takes `int8_per_token_head` KV with a native INT8 paged decode; without it that KV type raises on FlashAttention |
| `B70_PTH_FA_PREFILL_FAST` | `0` | 1 / unset / 1 / unset | INT8 KV prefill dequantises with one Triton launch per group, no host reads |
| `B70_QSA_I8DOT` | `0` | unset / 2 / unset / 0 | int8 Q.K (`1`) and P.V (`2`) DPAS in the QSA kernels over `int8_per_token_head` KV |
| `B70_PLE_INT8` | `0` | 1 / 1 / 1 / 1 on 4 cards | the INT8 PLE table native in pinned host RAM, about 49 GiB (the 2-card presets use `B70_PLE_INT8_NVME` instead) |
| `B70_PLE_DEVICE_GATHER` | `0` | 1 on all | the RAM-native PLE producer on the device, inside the graph; the first step compares its bytes with the host path |
| `B70_SAMPLER_FAST` | `0` | 1 on all | top-k-first temperature sampling and speculative verification; falls back to the stock path for min_p, processed logprobs, k above 32, grammar and similar |
| `B70_MOE_KCUT` | `0` | 1 on all | the expert-parallel sentinel and the rows-per-expert zero in one Triton kernel (7 kernels fewer per MoE layer) |
| `B70_AR_INPLACE` | `0` | 1 on all | all-reduce caller-owned outputs in place, no clone |
| `B70_HC_SILU_FUSE` | `0` | 1 on all | `hc_silu` folded into the parallel hyper-connection down finish kernel |
| `B70_DFLASH2_AUX_REUSE` | `0` | 1 on all | the DFlash2 aux features reuse the next layer's hyper-connection combine |
| `CCL_SYCL_ALLREDUCE_ARC` | unset | 1 on all | not an engine switch: oneCCL's core reads it for its Arc all-reduce path; the engine and the legacy plugin do not |

**Other switches of the series**

| variable | default | effect |
|---|---|---|
| `B70_PTH_PAD16` | `1` (on) | per-token-head KV halves padded to 16 bytes (544 B at head size 256) so Triton vectorises loads; `0` = tight. Int8 and fp8 per-token-head KV only |
| `B70_PTH_TD_GUARD` | `1` (on) | per-token-head KV loads K/V by pointer, not by tensor descriptor |
| `B70_KV_UNIFY_MULTIPLE` | `1` (on) | page unify takes block x floor(ratio) when the draft page is at least twice the target page and not divisible; `0` = plain padding |
| `B70_PTH_FA_PREFILL`, `B70_PTH_FA_MIN_Q` | `1` (on), `64` | batches with max q above `MIN_Q` dequantise int8 or fp8 KV blocks to a bf16 scratch and run bf16 FlashAttention |
| `B70_DENSE_INT8_INCLUDE`, `_EXCLUDE`, `_MIN_PARAMS`, `_TRITON_MAX_M` | all / router, GDN a/b, indexer, heads, embeddings and HC regex / `1<<20` / `64` | which linears convert, and the Triton-to-oneDNN row threshold |
| `B70_DENSE_INT8_FUSED` (+ `_FUSED_MAX_M` 64, `_W8A16_MAX_M` 8) | `0` | small-M dense linears on the fused W8A16 and W8A8 ops of the kernels wheel |
| `B70_MOE_W4A8_CFG1`, `_CFG2`, `_WLOAD`, `_FUSED_ACT`, `_UVA` | `32,256,8,0,2` / same / `2` / `0` / `0` | W4A8 tile configurations "BM,BN,warps,grf[,wload]", the weight-load scheme, the fused SiLU epilogue, cold experts over UVA |
| `B70_HC_DOWN_INT8` (+ `_MAX_ROWS`, `_SPS` 4, `_STREAM_SCALE` 0) | `0` | the HC down projection as a deterministic int8 split-K GEMM |
| `B70_I8_PRODUCER_QUANT` (+ `B70_I8_PQ_SITES`, `_MAX_M`, `_GEMM`, `_K1_MAX_M` 64, `_K1_INCLUDE`, `_NUM_WARPS` 4, `B70_I8D_TILE`) | `0` | per-token int8 activation quantisation folded into the producers (hyper-connection, gated RMSNorm, attention gate) |
| `B70_K2_HC_CAPTURE_DIR` (+ `_DECODE`, `_PREFILL`, `_EVERY`) | unset | diagnostic dump of HC-down inputs |
| `B70_DRAFT_FA_INT8` | `0` | the DFlash drafter's attention on the INT8 FlashAttention decode (needs `B70_PTH_FA_DECODE=1` and int8 draft KV) |
| `B70_PLE_INT8_NVME_PRELOAD` (+ `_THREADS` 4, `_CHUNK_MIB` 64) | `0` | preload the rank's rows into the pinned NVMe row cache |
| `B70_PLE_INT8_LOAD_THREADS`, `_LOAD_CHUNK_MIB`, `_BOOT_SAMPLE` | `4`, `64`, `4096` | the RAM table's boot read |
| `B70_SPEC_GREEDY_WARMUP` | `0` | an extra greedy warm-up round for speculative decoding |
| `B70_TORCH_COMPILE` | `0` | torch.compile for XPU Qwen4Exp with opaque custom ops (experiment) |
| `B70_GDN_MIXED_NO_SYNC` | `0` | device row indices for mixed speculative and prefill GDN steps, one asynchronous H2D copy |
| `B70_MTP_BF16_DRAFT`, `B70_DRAFT_LMHEAD_INT4`, `B70_DRAFT_MTP_INT4`, `B70_DRAFT_VOCAB` | off | riceharvest's MTP draft lanes (BF16 MTP layers, int4 draft head, int4 MTP linears, vocab subset) |

Kernel-side: `VLLM_XPU_FUSED_MOE_INT4_GEMV=0` turns off the int4 GEMV the whole-wheel kernels take by default for decode-sized int4 grouped GEMMs ([kernels](kernels.md)).
