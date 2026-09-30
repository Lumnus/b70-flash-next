# Runtime switches (`B70_*`)

Every feature the b70 series adds is behind an environment variable and **off by default**. With none of them set,
the image behaves as vLLM v0.30.0 plus wu1ff's patch set (0001–0005), apart from three small ungated parts listed at
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
| `B70_PLE_INT8` | unset (off) | 0008 | `1`: keep the table as INT8, one fp32 scale per row (164 B/row, 48.9 GiB over 4 ranks instead of 95.4 GiB BF16); needs `B70_PLE_INT8_PATH`. Build the table with `tools/build_int8_ple.py` (docs/ple-int8.md). |
| `B70_PLE_INT8_PATH` | — | 0008 | the INT8 table `.safetensors`. Its metadata must carry the format tag `lumnus-ple-int8-rowscale/v1` (see below). |
| `B70_PLE_INT8_CROSSCHECK` | `1` | 0008 | `0` skips the boot cross-check against the BF16 table. |
| `B70_PLE_INT8_MAX_REL_ERR` | `0.02` | 0008 | refuse to start above this relative L2 error (expected ~0.0066). |
| `B70_PLE_INT8_NVME` | unset (off) | 0013 | `1` (with `B70_PLE_INT8=1`): serve the INT8 table from NVMe through a small pinned row cache instead of pinning all of it (docs/ple-nvme.md). |
| `B70_PLE_INT8_NVME_PATH` | `B70_PLE_INT8_PATH` | 0013 | the table file to read with `O_DIRECT`. |
| `B70_PLE_INT8_NVME_CACHE_GIB` | `8` | 0013 | pinned row cache, GiB **in total over all TP ranks**. We run `8` (2 GiB per rank on 4 ranks); the hit rate on agent traffic was 97–98 %. |
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
| `B70_DEFAULT_PRESENCE_PENALTY` | unset | 0009 | server default for `presence_penalty` (we use `0.7`); a request's own value wins. |
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
| `B70_OFFLOAD_GDN_BACKSTEP` | `0` | 0014c | `N`: also retain the Mamba/GDN states at the N block boundaries below the replay boundary, so the first revisit hits too. Cost per request: N × one Mamba state set (~116 MB for Qwen3.8-Flash-Next) in the CPU tier. |
| `B70_OFFLOAD_EMPTY_ADVANCE_GUARD` | unset (off) | 0014d | `1`: guard for vllm#56795 (do not advance past keys still write-pending under another request). |
| `B70_OFFLOAD_GROUP_EVICT` | unset (off) | 0014e/f | `1`: select the vllm#51787 backport (request-scoped recency, tail-before-head eviction across all KV groups). Unset = upstream v0.30.0 LRU/ARC, byte for byte. |

Candidate default for the offload fix ("0014jb"): `B70_OFFLOAD_TRACE=1 B70_OFFLOAD_JUNCTION=1
B70_OFFLOAD_GDN_BACKSTEP=1 B70_OFFLOAD_EMPTY_ADVANCE_GUARD=0 B70_OFFLOAD_GROUP_EVICT=0`.

## Ungated parts (active even with every switch unset)

1. 0009: `reasoning_effort` also accepts `"ultra"` (API widening only).
2. 0013: the KV-offload host-tensor allocation log line moves from DEBUG to INFO and prints `is_pinned()`.
3. 0014f: `ReqContext` records key positions (a dict write per key, never read with `B70_OFFLOAD_GROUP_EVICT` unset),
   and the policy base class gains optional hooks that are never called with the flag unset.
