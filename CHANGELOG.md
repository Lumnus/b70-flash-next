# Changelog

Versions are `<vLLM base>-b70.<N>`; N increases whenever the patch series changes.

## 0.30.0-b70.1 — pre-release (2026-09-30)

First public release of the series. Image not yet built.

- Base: vLLM v0.30.0 XPU image (`vllm/vllm-openai-xpu@sha256:fc0e112a…`), vllm-xpu-kernels 0.1.14.
- 0001–0005: wu1ff's B70-LLM-Controller Qwen3.8-Flash-Next pack 1.0.0, re-derived as source patches; byte-identical
  to wu1ff's files. P6/P7 binaries copied from wu1ff's image by digest.
- 0006: direct-to-pinned PLE load (`B70_PLE_DIRECT_PINNED`).
- 0007: FP8 PLE table (`B70_PLE_FP8`).
- 0008: INT8 per-row PLE table (`B70_PLE_INT8`), with `tools/build_int8_ple.py`.
- 0009: per-effort thinking budgets and a server default presence penalty.
- 0010: server default repetition stop.
- 0012: `reasoning.effort` alias, env-gated.
- 0013, 0013b: INT8 PLE table served from NVMe, native reader, optional prefill lookahead.
- 0014a–f: KV-offload trace, hybrid junction heal, GDN backstep, #56795 guard, vllm#51787 backport (gated).
- All switches use the `B70_` prefix. The INT8 table format tag stays `lumnus-ple-int8-rowscale/v1`.
- `patches/vllm-xpu-kernels/0001`: the 64-bit conv-state offset fix as source (untested, not used by the build).
