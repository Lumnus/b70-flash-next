# Upstream

Fixes from this work, posted upstream on 2026-10-25. Status on 2026-11-03: on vllm-xpu-kernels main, #658 is merged (`daf6240`) and
#660 merged on 2026-11-02 (`b73bdc3`), neither in a release yet; vLLM #63566 is merged on main (2026-10-29). #662 is not on main (checked by title).
#63564 and oneCCL #229 were open on 2026-10-31 and were not checked again. The engine carries all six: #660 and #662 in our kernel
build (`v0.32/engine/patches/vllm-xpu-kernels/`), #658 in the same build and through our `libgdn_index64.so`, the two vLLM fixes
cherry-picked in the dense base, oneCCL #229 in the plugin build.

It also carries pull requests by others, with their authorship:

1. vllm-project/vllm#62113 (16-byte aligned per-token-head KV slots, `vllm-v6/` 0029), #64190 and #64244 (Triton W8A8 INT8 GEMM tiles on XPU, `vllm-models/` 0002 and 0003).
2. vllm-project/vllm-xpu-kernels#605 (grouped GEMM scale-prefetch pitch guard, kernels 0005), still open when last checked on 2026-10-31.
3. vllm-xpu-kernels#647 (spec-decode adaptive verification in FlashAttention, kernels 0008, merged as `470c19f`).
4. Picks in `vllm-v6/`: vllm-project/vllm#63421, #63819, #62244, #59260 and a backport of #64141.

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

