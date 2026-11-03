# Credits

What this work builds on, and who to thank.

This work stands on other people's:

1. **wu1ff — [B70-LLM-Controller](https://github.com/wu1ff/B70-LLM-Controller)** (MIT). The Qwen3.8-Flash-Next B70
   pack: the XPU gate, the pinned-host PLE table, the HC down-GEMM K-split, the GDN spec-metadata fusion, the 64-bit
   GDN conv-state offset fix, the Level Zero peer-residency shim and the dense-QSA serve config. In the 0.30 series, patches
   0001–0005 are wu1ff's changes re-derived as source diffs, and two binaries are copied from wu1ff's public image by
   digest. In the 0.32 series, patches 0001–0016 are his. The 0.32
   line is built on wu1ff's 1.0.5 chain for vLLM 0.32 (DFlash2 path, C1 verify/target graphs, the astra500 MoE,
   hyper-connection and prefill kernels), taken unchanged with his authorship and MIT notice. Thank you.
2. **Intel** — the [`Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
   weights we serve and recommend, made with [AutoRound](https://github.com/intel/auto-round);
   [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels), whose upstream fixes #600, #563, #564, #578 and
   #586 are in our 0.30 kernel build (by Guancheng Fu, Chaojun Zhang, Qiming Zhang and Tony Lin);
   [llm-scaler](https://github.com/intel/llm-scaler) and the XPU work in
   [vLLM #55068](https://github.com/vllm-project/vllm/pull/55068).
3. **wtdcode** — the [`Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16)
   weights, the validated alternative (we served them 2026-09-30 to 2026-10-07).
   **Minachist** — [`Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`](https://huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound),
   whose recalibrated experts are the supported variant.
4. **devan-carlin** — the [`Qwen3.8-Flash-Next-W4A16`](https://huggingface.co/devan-carlin/Qwen3.8-Flash-Next-W4A16)
   weights, its BF16 PLE table (every INT8 table here is built from it), and the early community vLLM port of
   the model to XPU (`devan-carlin/vllm@xpu-qwen4exp`).
5. **TSUMUGI-XE** — [intel/compute-runtime#968](https://github.com/intel/compute-runtime/issues/968), the analysis of
   Level Zero peer residency mirroring device memory into host memory, and the first shim that works around it.
6. **incoai** — the [`Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
   block-diffusion drafter (Apache-2.0) behind the 0.32 release's speculative decoding.
7. **Niluthon** — [qwen38-flash-tp2](https://github.com/Niluthon/qwen38-flash-tp2) (Apache-2.0), a two-card fork of
   our b70.2: the FP8 `lm_head`, the resident decode MoE op, the step trace and the `bench_det` harness came from there;
   the 0.32 FP8 head (B70-V32-0053) is a port of its 0053, and the 1- and 4-stream numbers on this page use `bench_det`.
   **Hot/cold expert offload**, which makes the Flash-Next 2-card presets possible, is his: `v0.32/engine/patches/vllm-hotcold/` 0001–0010
   are his B70-0040..0049 ported to 0.32 with his authorship, and the 2-card presets' fused hot+cold decode op is built from his
   `kernels/b70-moe-decode`.
8. **DominikBucko** — [qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090) (Apache-2.0),
   the two-card CUDA port. No code is taken from it; its published measurements and design notes (async scheduling and the
   Mamba state-block fix, prefill chunk sizes, the AVX2 cold-expert path) shaped which levers we measured, and the 2-card
   presets start their hot set from his `configs/static_hot_cache_rankings.json` (@ `4dca58f`).
9. **The Qwen team** — Qwen3.8-Flash-Next, Qwen3.8-27B and Qwen3.6-35B-A3B (licenses on each checkpoint's card).
10. **For the 27B and 35B**: **devan-carlin**'s [`Qwen3.8-27B-int4-AutoRound`](https://huggingface.co/devan-carlin/Qwen3.8-27B-int4-AutoRound)
    and **Intel**'s [`Qwen3.6-35B-A3B-int4-mixed-AutoRound`](https://huggingface.co/Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound) weights;
    **incoai**'s [`Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) and **z-lab**'s
    [`Qwen3.6-35B-A3B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) drafters (Apache-2.0); **Katsumi Takeuchi**
    (recutita), vllm-project/vllm#64190 and #64244, the Triton W8A8 INT8 GEMM tiles on XPU, carried with his authorship;
    **Guancheng Fu** (Intel), vllm-xpu-kernels#605, the grouped GEMM pitch guard the 35B needs, carried with his authorship;
    **steveseguin** — [b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab) (The Unlicense): the
    shortlist draft head idea and the token list behind the 27B's draft vocab.
11. **[vLLM](https://github.com/vllm-project/vllm)**, on which everything here is built. Patch 0014e is a backport of
    vllm-project/vllm#51787; patch 0022 ports vllm-project/vllm#55506.
12. **For 0.32.0-b70.6**, carried with their authorship: **riceharvest**, the int4 GEMV and the fast e4m3 conversion in the
    kernels, [intel/sycl-tla#859](https://github.com/intel/sycl-tla/pull/859), and the draft int4 MTP lane, the asymmetric-GPTQ
    loader and `gdn_mixed_no_sync` in the engine; **wiesion**, the INT8 per-token-head KV cache for paged decode in the kernels;
    **Yan Ma** (Intel), vllm-xpu-kernels#647 and vllm-project/vllm#62244, spec-decode adaptive verification on XPU; **Misha
    Goin**, vllm-project/vllm#63421 and #63819; **qizixi**, vllm-project/vllm#59260; **noon-at-cgn**, the 16-byte aligned
    per-token-head KV slots (vllm-project/vllm#62113).
