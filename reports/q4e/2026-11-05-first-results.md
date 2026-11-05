# Qwen3.8-Flash-Next on zhaohb's q4e branch: first results, 2026-11-05

Hardware: 4 x Arc Pro B70 (32 GB each), EPYC 7402. Stack: torch 2.15.0, Triton 3.9.0. Model: Intel W4A16 AutoRound. Same prompts and tools for every column. Single-run differences under about 10% are noise. Testing is ongoing.

## Engines

| Name | What it is |
|---|---|
| S16-7, D16-7 | Released kit on vLLM 0.32, sparse and dense, 16-bit |
| S16-Z1, D16-Z1 | q4e branch at `6515212683` + #64925 + #64926 + strict fixes, kernels 0.1.17.1 |
| S16-Z2 | q4e branch at `a8cdbee221` + zhaohb/vllm#1 + #2 + PLE row staging (not yet in a PR) |

## Numbers (tok/s)

| | S16-7 | S16-Z1 | S16-Z2 | D16-7 | D16-Z1 |
|---|---|---|---|---|---|
| Decode, 1 stream, no speculation on Z | 105 | 35 | 40 | 106 | 37 |
| Prefill, about 8K | 5,540 | 943 | 3,050-3,200 | 6,350 | 3,300 |
| Prefill, about 125K | 5,320 | 661 | about 2,100 | 4,760 | 2,580 |

The kit columns run speculative decoding. Native MTP with K=3 on Z1 gives decode 58 (sparse) and 89-92 (dense), burst 0/150.

## Findings

1. **Stall under load:** an int32 GDN conv-state offset overflows at block 396 at this pool size, and the engine dies with `RPC call to sample_tokens timed out`. Dense hits it too, so it isn't specific to the branch. Kernels 0.1.17.1 ran two full 1-3 stream loads with 0 resets on the full KV pool. The fix is vllm-xpu-kernels#658; 0.1.17.1 carries other changes too, so #658 isn't isolated.
2. **Sparse prefill:** the stock `qsa_sparse_paged_attention` kernel takes 1,545 ms per 1,664-token chunk, against 46 ms for dense FA. The `attn2` kernel takes 97 ms per 4,096-token chunk, about 39x per token.
3. **Temperature 0:** the kit is bit-identical on 5-word and 30-word prompts. The Z arms jitter there without speculation, and are bit-identical with MTP on. Cause not found yet.
4. **2 cards:** Intel W4A16 needs 73.3 GiB on the cards against 64 GiB, so 2 cards can't boot without expert offload. D16e on this branch with the hot/cold split behind `--cpu-offload-gb`: 111 hot, 145 cold, KV 343,037 tokens, needle 4/4, burst 0/150, decode 54-61, prefill about 2,700.

## PRs

1. [zhaohb/vllm#1](https://github.com/zhaohb/vllm/pull/1): fixes.
2. [zhaohb/vllm#2](https://github.com/zhaohb/vllm/pull/2): QSA shared-op fixes and `attn2`.
