# CPU KV offload tier

`--kv-offloading-size 64 --kv-offloading-backend native` (64 GiB in total, 16 GiB per rank, a power of two), INT8 PLE
in RAM, 8 sequences, GPU KV pool ~454K tokens.

## What the tier buys (8 growing agent sessions, same seed)

| | no CPU tier | 64 GiB CPU tier |
|---|---|---|
| working set below the GPU pool: aggregate decode | 74 tok/s | 68 tok/s |
| working set past the GPU pool: TTFT median (p90) | 161 s (269 s) | **18 s** |
| past the GPU pool: decode per stream / aggregate | 3.7 / 8 tok/s | 13.7 / 39 tok/s |
| history recomputed | 7.6 M tokens | 0.7 M tokens |
| serial re-send of an 80K / 120K prompt | 20.4 / 33.9 s | 0.77 / 1.93 s |

Read-back correctness: 8 documents of ~60K tokens revisited twice, 24/24 answers correct; revisit TTFT 18.2 s → 3–4 s
median (0.7 s best); the server computed only 439–811 of ~59.5K prompt tokens per revisit. For this hybrid model the
GPU prefix cache gave 0 hits on these revisits; reuse came from the CPU tier only.

Pinned-memory notes (torch 2.13 on XPU):

1. torch's caching host allocator rounds pinned allocations up to a power of two by default. Size the pool per rank
   as a power of two, or set `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1024` to allocate exact sizes above
   1 GiB.
2. With exact sizes, also set `pinned_max_cached_size_mb:1024`: without it a cached, undersized staging block can be
   handed to a >1 GiB CPU→GPU reload (seen as a segfault in `xpuAsyncMemcpyBatch`; fixed upstream in pytorch#192722).
3. `vllm:kv_offload_cpu_cache_usage_perc` is not the pool fill level.

## The offload fix (0014)

See [../offload-fix.md](../offload-fix.md) for the band test. Summary: unpatched, "same document, new question"
revisits whose last full block reaches into the question miss forever (0/3, then 0/3); with `B70_OFFLOAD_JUNCTION=1`
they miss once and then hit (0/3, then 3/3); adding `B70_OFFLOAD_GDN_BACKSTEP=1` makes the first revisit hit too (3/3,
3/3) for +3.8 % CPU-tier bytes per document.
