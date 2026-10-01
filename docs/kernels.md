# Our vllm-xpu-kernels builds

Qwen3.8-Flash-Next's GDN layers keep one conv state per cache block. Upstream's `causal_conv1d` kernels compute that
state's address as `states_id * conv_states_stride_0` in 32 bits. On 4× B70 the product passes 2^31 at block id 5,042
(× 425,984), and the engine dies with `DEVICE_LOST` on the first step that uses such a block. wu1ff's pack works around
it with a separate closed library (`libgdn_index64.so`, selected by patch 0005 with `B70_GDN_INDEX64=1`). We fix it in
the kernels' source instead: one commit, 7 sites, `static_cast<int64_t>(states_id) * conv_states_stride_0`
(`patches/vllm-xpu-kernels/0001-gdn-causal-conv1d-int64-state-offset.patch`).

Fork: [Lumnus/vllm-xpu-kernels](https://github.com/Lumnus/vllm-xpu-kernels).

| build | for | branch | state |
|---|---|---|---|
| `0.1.15.4+b70.1` | torch 2.14, oneAPI 2026.1 (the edge line) | [`b70/v0.1.15`](https://github.com/Lumnus/vllm-xpu-kernels/tree/b70/v0.1.15) = upstream `release/0.1.15.4` + the fix | built and verified; ran on the edge line |
| `0.1.14.1+b70.1` | torch 2.13, oneAPI 2026.0 (the stable line) | upstream tag `0.1.14.1` + the fix | building; not published yet |

## 0.1.15.4+b70.1

1. **Source.** `release/0.1.15.4` (`ddf336d`), the version vLLM main pins, plus the fix cherry-picked cleanly:
   `b70/v0.1.15` @ `69b823f9`. (The `v0.1.15` tag is still a torch 2.13 tree; the torch 2.14 bump landed after it.)
2. **B70-only.** Built with `VLLM_XPU_ENABLE_XE3P=OFF` (no Xe3/Xe3P kernels; the B70 never runs them) and CMake's
   default kernel presets. `libattn_kernels_xe_2.so` is 1.63 GB against the official 307 MB: the full presets, a
   superset. Toolchain: upstream's own builder image (`pytorch/manylinux2_28-builder:xpu-v2.14.0-rc10`),
   `setup.py bdist_wheel --py-limited-api=cp38`, `VLLM_VERSION_OVERRIDE=0.1.15.4+b70.1`. Our build peaked at ~94 GiB
   RAM with 12 compile jobs; each AOT compile is single-threaded and the whole build takes hours.
3. **Same API.** All 111 registered ops (`_xpu_C`, `_C`, `_moe_C`, `_C_cache_ops`, `_vllm_fa2_C`) have identical
   schemas to the official 0.1.15.4 wheel.
4. **The fix is in the binary.** Disassembly of the BMG device code: the official kernel multiplies in 32 bits and
   sign-extends the wrapped product; ours computes the full 64-bit product. 36 of 40 `causal_conv1d` kernel variants
   gain the high-word multiply; the 4 unchanged ones have no conv-state access.
5. **In use.** With it, the stock `gdn_attention` op is already 64-bit: no wu1ff library, `B70_GDN_INDEX64=0`, and patch
   0005 is redundant. Edge-line engines without it died with `DEVICE_LOST` on all four cards; with it they did not
   (they fail differently: see [known-issues.md](known-issues.md) item 5, the reason the edge line is not recommended).
6. **Not published as a wheel yet.** Build it from the branch.

## 0.1.14.1+b70.1 (in progress)

The stable line runs vLLM v0.30.0 on torch 2.13 with the official vllm-xpu-kernels 0.1.14.1 plus wu1ff's
`libgdn_index64.so`. This build replaces that closed library with the same source fix: upstream tag `0.1.14.1`
(`6d92b1bf`, the commit the PyPI 0.1.14.1 wheel was built from) + the fix, compiled with oneAPI 2026.0 to match torch 2.13's
SYCL runtime. When it is verified, the stable line can drop 0005 and wu1ff's library.
