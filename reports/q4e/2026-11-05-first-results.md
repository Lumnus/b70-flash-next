# Qwen3.8-Flash-Next on zhaohb's q4e branch: test report

Updated as results land; last update 2026-11-05. Hardware: 4 x Arc Pro B70 (32 GB each), EPYC 7402. Stack on q4e: torch 2.15.0, Triton 3.9.0, compute-runtime 26.35, IGC 2.41.5, both pinned on this host rather than from distro packages. Model: Intel W4A16 AutoRound. Same prompts and tools throughout. Single-run differences under about 10% are noise. Testing is ongoing.

## Engines

| Name | What it is |
|---|---|
| S16-7, D16-7 | Released kit on vLLM 0.32, sparse and dense, DFlash |
| S16-Z1, D16-Z1 | q4e branch at `6515212683` + #64925 + #64926 + strict fixes, kernels 0.1.17.1 |
| S16-Z2 | q4e branch at `a8cdbee221` + zhaohb/vllm#1 + #2 + PLE row staging (not yet in a PR), kernels 0.1.17.1 |
| D16-Z2 | Same tree as S16-Z2, dense |
| S16-q4e-8 | The kit's engine merged onto the q4e branch, its own kernels on 0.1.16.3 + #658 |

## Main numbers (tok/s)

| | S16-7 | S16-Z1 | S16-Z2 | S16-Z2 best | S16-q4e-8 | D16-7 | D16-Z1 | D16-Z1 MTP |
|---|---|---|---|---|---|---|---|---|
| Decode, prose, 1 stream | 105 | 35 | 40 | 32-44 | 105.8 | 106 | 37 | 89-92 |
| Prefill, about 8K | 5,540 | 943 | 3,050-3,200 | 3,450-3,560 | 5,560 | 6,350 | 3,300 | 3,550 |
| Prefill, about 125K | 5,320 | 661 | about 2,100 | TTFT 42.3 s | TTFT 23.1 s | 4,760 | 2,580 | |
| 3 streams at 8K, total | | 32 | 66 | 75 | 214 | | 64 | |
| Speculative | DFlash | none | none | MTP K=3, 0.56 of 3 | DFlash K=7, 2.87 of 7 | DFlash | none | MTP K=3, 2.14 of 3 |

## Best Z2 formula against the kit's engine on q4e (`cpu_offload` exact pin, FULL_AND_PIECEWISE)

| | S16-Z2, MTP K=3 | D16-Z2, MTP K=3 | S16-q4e-8, DFlash K=7 |
|---|---|---|---|
| Prefill, 8K | 3,550 | 3,950 | 5,700 |
| Prefill, 42K | 3,410 | 3,800 | 5,600 |
| TTFT at 125K | 42.3 s | 39.3 s | 23.1 s |
| Decode, prose | 44 | 71 | 106 |
| 3 streams at 8K, total | 75 | 137 | 214 |
| Acceptance | 0.56 of 3 | 2.12 of 3 | 2.87 of 7 |
| PLE per chunk | 2.4 ms | 2.5 ms | 2.8-4.7 ms (INT8) |
| Host memory used | 215 GiB | 181 GiB | 214 GiB |
| Gates | RAM guard stop at 12 streams x 32K | all pass | all pass |

D16-Z2 at 8K, 1/2/3/8/12 streams: 100/152/137/126/124 tok/s total. The same MTP layer gets 2.12 on dense and 0.56 on sparse in the same tree, so low acceptance comes from the draft on a sparse target.

With a sparse MTP draft layer on S16-Z2 (`attn2` on the draft, the checkpoint's own `mtp...indexer` weights, chunk 4,096), acceptance goes from 0.56 to 2.07 of 3 at the same 31 ms step. Decode, prose 44 to 79, and at 16K to 125K from 70/33/32 to 80-85. Prefill 3,420 at 8K, about 3,400 at 42K, TTFT at 125K 42.6 s, host memory 186 GiB, needle 4/4, tools pass, burst 0/150, 0 resets. 3 streams at 8K total 131, up from 75; at 32K 38, up from 31. 1 stream at 8K 100, at 32K 109. In [zhaohb/vllm#4](https://github.com/zhaohb/vllm/pull/4).

"S16-Z2 best" is `cpu_offload` with exact pinning, MTP K=3 and `attn2`, FULL_AND_PIECEWISE: boot 225 s, needle 4/4, tools pass, burst 0/150, 0 resets. Decode there is 85 at 8K and 65 at 32K.

## Fused hyper-connections (origin wu1ff), D16-Z2h

The Z formula above plus fused hyper-connections, on the same branch. In [zhaohb/vllm#5](https://github.com/zhaohb/vllm/pull/5).

| | D16-Z2 | D16-Z2h | D16-7 |
|---|---|---|---|
| Forward per 4,096 chunk, rank 0, ms | 852 | 561 | 539 |
| Hyper-connections per chunk, ms | 409 | 130 | 132-159, by subtraction |
| Prefill, 8K | 3,950 | 6,130 | 6,350 |
| Prefill, 42K | 3,800 | 5,720 | 6,150 |
| TTFT at 125K | 39.3 s | 28.4 s | 26.3 s |
| Decode, prose | 71 | 96 | 106 |
| 3 streams at 8K, total | 137 | 184 | 232 |
| Acceptance, MTP K=3 | 2.12 of 3 | 2.15 of 3 | |
| Step | 32-37 ms | 25.5-29.9 ms | |

D16-7 runs DFlash K=7, so its decode rows aren't a like-for-like speculative comparison. D16-Z2h gates: needle 4/4, tools pass, burst 0/150, 12/12 throughput twice, 0 resets, temperature-0 one text on all three prompts. Host memory 187 GiB, KV 258,748 tokens. Streams at 8K, 1/2/3/8/12: 124/184/184/184/181 tok/s total; at 32K: 133/73/60/48/46.

S16-Z2h, the same with sparse MTP, same gates pass: forward 677 ms per 4,096 chunk, hyper-connections 129, MoE 223, GDN 110, QSA kernel 94, indexer 97, PLE 4.5. Prefill 5,300 at 8K, 4,840 at 42K, TTFT at 125K 31.6 s, decode prose 102, code 159, at 16K/61K/125K 108/107/117, acceptance 2.11 of 3, step 23.6-23.9 ms. Streams at 8K, 1/2/3/8/12: 137/187/182/165/164 tok/s total; at 32K: 135/66/54/42/41. Host memory 190 GiB, KV 246,034 tokens. The next sparse cost is the indexer: 97 ms against 42 on S16-7, with `_prefill_logits` 81 against 10.

## PLE paths on S16-Z2 (BF16 table, no speculation, all gates pass)

| | mmap | `disk_offload` | `cpu_offload`, exact pin |
|---|---|---|---|
| Prefill, 8K | 3,340 | 3,000 | 3,650 |
| Prefill, 42K | 3,050 | 2,790 | 3,250 |
| TTFT at 125K, repeat round | 48 s | 52 s | 44.4 s |
| PLE per 1,664-token chunk, warm | 33 ms | 77 ms | 2.4 ms |
| Cold first 8K prompt | 9.3 s | 3.1 s | |
| Host memory | page cache | 0.24% of shard resident | +94 GiB pinned |

Without `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1024,pinned_max_cached_size_mb:1024`, `cpu_offload` pins 128 GiB. The `disk_offload` gap is Python syscalls per row and run, so batching should remove it.

The kit's PLE on S16-q4e-8: INT8 rows, 12.22 GiB per rank in 6 pinned slabs, in-graph gather 2.8-4.7 ms per chunk. The gather also passed a device test on real rows, 1,200 iterations in eager with bitwise-equal hash and gather. 

The kit's INT8 table from NVMe on S16-q4e-8, 8 GiB row cache, native reader, DFlash K=7: prefill 5,600 at 8K, 5,550 at 42K, TTFT at 125K 24.7 s, decode 103, 3 streams at 8K total 207, PLE 4.3 ms per chunk, all gates pass. Within noise of INT8 in RAM, at 40 GiB less host RAM. These prompts reach a 1.000 hit rate once warm, so they don't stress the tier.

Random-token stress on the same NVMe path, 4 GiB row cache, every row new:

| | Lookahead off | Lookahead on |
|---|---|---|
| 16K prompt, s | 3.33 | 2.90 |
| 64K prompt, s | 13.78 | 11.84 |
| Step bubble, p50 per 4,096 chunk, ms | 107 | 2.5 |

With lookahead on the ordinary prompts stay within noise, since few rows are new: prefill 5,580 at 8K and 5,550 at 42K, TTFT at 125K 23.4 s, decode 103, 3 streams at 8K total 203, burst 0/150, about 170 GiB host memory.

## Findings

0. **Dense prefill gap, per 4,096-token chunk, rank 0:** D16-Z2 852 ms, D16-7 539 ms. FA 71 vs 65, MoE 222 vs 203, GDN 109 vs 107, PLE 4.4 vs 4.9. The gap is hyper-connections, 409 ms (mix 331, combine 78, 136 of all-reduce) against about 159 ms by subtraction on the kit's fused path. Chunk size matters separately: stock 1,664 vs 4,096 took D16 prefill at 8K from 3,300 to 3,950, and S16 from 3,340 to 3,550, both with MTP on at 4,096.

1. **Stall under load:** an int32 GDN conv-state offset overflows at block 396 at this pool size, and the engine dies with `RPC call to sample_tokens timed out`. Dense hits it too, so it isn't specific to the branch. Kernels 0.1.17.1 ran two full 1-3 stream loads with 0 resets on the full KV pool. The fix is vllm-xpu-kernels#658; 0.1.17.1 carries other changes too, so #658 isn't isolated.
2. **Sparse prefill:** the stock `qsa_sparse_paged_attention` kernel takes 1,545 ms per 1,664-token chunk, against 46 ms for dense FA. The `attn2` kernel takes 97 ms per 4,096-token chunk, about 39x per token.
3. **Temperature 0:** the kit and S16-q4e-8 are bit-identical on 5-word and 30-word prompts. The Z arms jitter there without speculation, and are bit-identical with MTP on. Cause not found yet.
4. **BF16 experts on kernels 0.1.17:** the layout rewrite moved into `XpuFusedMoe`, so a host-side transpose undoes it and BF16 layers die; fixed in zhaohb/vllm#1.
5. **2 cards:** Intel W4A16 needs 73.3 GiB on the cards against 64 GiB, so 2 cards can't boot without expert offload. D16e on the q4e line with the hot/cold split behind `--cpu-offload-gb`: 111 hot, 145 cold, KV 343,037 tokens, needle 4/4, burst 0/150, decode 54-61, prefill about 2,700. Prefix caching shows 0 hits on this arm so far; under investigation.

## PRs

1. [zhaohb/vllm#1](https://github.com/zhaohb/vllm/pull/1): fixes.
2. [zhaohb/vllm#2](https://github.com/zhaohb/vllm/pull/2): QSA shared-op fixes and `attn2`.
3. [zhaohb/vllm#3](https://github.com/zhaohb/vllm/pull/3): PLE `--engram-config`, the three paths in the table above.
4. [zhaohb/vllm#4](https://github.com/zhaohb/vllm/pull/4): sparse MTP draft layer.
5. [zhaohb/vllm#5](https://github.com/zhaohb/vllm/pull/5): fused hyper-connections, from wu1ff's work.
