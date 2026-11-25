# l34 manifest

The stack of the l34 reference engine as it served, read from the running host on 2026-11-25. Hashes are sha256.

## 1. Code

| component | ref | commit or version |
|---|---|---|
| vLLM | `Lumnus/vllm` `l34`, tag `l34-final` | `e1f970f791ccc96f1904509748c42658c813c83a` |
| vllm-xpu-kernels | `Lumnus/vllm-xpu-kernels` `l34`, tag `l34-final` | `e245d929a5d212d4a273cd0ba19510d0d0d20ed8` |
| kernels wheel | `vllm_xpu_kernels-0.1.18.dev0+l34.k17k.e245d929a-cp38-abi3-manylinux_2_28_x86_64.whl` | `514d87e5df2602d4d188419301928d0de70c9a2d252ce501a53d2fa64a3784bd` |
| torch | `torch 2.15.0+xpu` (official wheel, git `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`) | two libraries replaced, below |
| torch patch 1 | `patches/pytorch/0001-XPU-Do-not-create-the-stream-pool-to-check-the-captu.patch` (on release/2.15 `d0f2af8b66`) | `libtorch_xpu.so` is built from this commit |
| torch patch 2 | `patches/pytorch/0002-XPU-Create-a-reserved-stream-queue-when-it-is-first-.patch` (on patch 1) | `libc10_xpu.so` is built from this commit |
| `libc10_xpu.so` (patched; rpath `$ORIGIN/../../../..:$ORIGIN`) | 352,040 bytes | `89c152775bfe684ff8947ab04e8caf6907cec12f506673fae4637b679f32042b` |
| `libtorch_xpu.so` (patched) | 1,033,429,129 bytes | `4abc583e634dcebbafbd9c15136de218514d9fda0100b28f97ff247f2111c467` |
| official `libc10_xpu.so` / `libtorch_xpu.so` it replaces | | `4afc23555bf454cfdcb37363831a488f6c8d8bbdaadcab3f1ac270a3283f264c` / `5ea1082357b1929c9192930411837bf4b1f48eaf674cd911ae037c5c83e59df4` |
| Triton | `triton_xpu 3.9.1+git13963861.l34` = intel-xpu-backend-for-triton main `9b4b1c768` + PRs #8769, #8736, #8761, #8391, #8740, #8122, #8765, #8711, #8660, #8289 merged by head | wheel `37c840e5b54b895b96670c16f9bcc6cffaebdac0942cfcef4ba4b74b935be3f0` |
| Python | CPython 3.12.14 | |

## 2. Runtime libraries (pip, in the same venv)

`intel-sycl-rt 2026.2.0` · `intel-cmplr-lib-rt 2026.2.0` · `intel-cmplr-lib-ur 2026.2.0` · `intel-opencl-rt 2026.2.0` · `dpcpp-cpp-rt 2026.2.0` · `intel-openmp 2026.2.0` · `oneccl 2022.1.2` · `impi-rt 2021.18.1` · `mkl 2026.1.0` · `onemkl-sycl-{blas,dft,lapack,rng,sparse} 2026.1.0` · `umf 1.2.0` · `tcmlib 1.5.0` · `intel-pti 1.1.0`.

## 3. Driver stack

| component | version |
|---|---|
| compute-runtime (Level Zero / OpenCL) | 26.35.39758.10 |
| IGC | 2.41.5 (2_2.41.5+22716) |
| gmmlib | 22.10.0 |
| Linux kernel, `xe` driver | 6.18.63 |
| GuC firmware | `xe/bmg_guc_70.bin` 70.72.1 |

The engine runs with `NEOReadDebugKeys=1 AllowUnrestrictedSize=1 EnableDeferBacking=0` and the copy thresholds in the preset (pinned host allocations above 30 GiB are otherwise refused).

## 4. Weights

| part | source | identity |
|---|---|---|
| target | `qfn-w4a8-x2`: Qwen3.8-Flash-Next (`Qwen/Qwen3.8-Flash-Next` revision `de4b8e4`), calibrated W4A8 experts and W8A8 dense layers, INT8 head; not published yet | its `MANIFEST.sha256` (29 own files; 31 unchanged shards linked from the base): manifest `0f62a5a8699dde502f7973d2263a1db5be8b711614b6f1e4cf1030e28a3f7ab8`, `config.json` `ad3c623bb75ac7cdb577c63fdff23038a3c612b4709b1c1ea3b2b23389b09d1b`, `model.safetensors.index.json` `461ee2d1827d2aa612ebcf8dbf01fda64ac230df5b0723b5b638cc2b784cec27` |
| drafter | `incoai/Qwen3.8-Flash-Next-DFlash2` revision `a22a1744e85c17364dcc54a6a420af7135d4c7f6` (source `model.safetensors` `d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f`), quantized W8A8 with llm-compressor 0.15.0 + compressed-tensors 0.20.1, `QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["candidate_selector.hidden_projection"])`, data-free | `model.safetensors` `6c3a1c2356bb350e6abbb402d0aaa9bf0e59ac4e7d858384acde74fb6a1df5df`, `config.json` `ac1d0142012bdf1c9b703479b81ac929914b43510f7bca4ad0e5ccf55ac71461` |
| n-gram (PLE) table, INT8 with row scales | built from the target with `tools/build_int8_ple.py` (on `main`) | `ple_ngram_int8_rowscale.safetensors` 52,480,256,280 bytes, `76427c71e45bfd4cf0815b2c6d63bf00a280515b561faac74ebf29e7e9db384c` |

## 5. Engine definition

[`presets/qfn4-l34d8di8-8.sh`](presets/qfn4-l34d8di8-8.sh): the serving command line, with host paths replaced by placeholders.
