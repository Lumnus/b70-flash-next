# The engine

vLLM v0.32.0 with wu1ff's 1.0.5 B70 chain and our B70 work on top, for Intel Arc Pro B70 (32 GB). One tree serves every
preset: card count, attention, execution precision, drafter and offload are flags and environment switches. The engine is fork tag
[`v0.32.0-b70.6`](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.6). Tested host: AMD EPYC 7402, 256 GB RAM, WD_BLACK SN850X 4 TB NVMe, 4× B70.

## The series

Five directories in [`engine/patches/`](engine/patches/) apply in order on vLLM v0.32.0 and give the tag's tree.
`engine/scripts/check-series.sh --all` compares them with a fresh export of the fork; `engine/tools/check-trunk-tree.sh`
compares them with a trunk at tree level.

| directory | commits | what |
|---|---|---|
| `vllm/` | 50 | dense base: wu1ff's chain, our B70 layers, 0.32-specific fixes |
| `vllm-qsa/` | 38 | sparse attention (QSA) |
| `vllm-hotcold/` | 13 | hot/cold expert offload (Niluthon's patches and ours) |
| `vllm-models/` | 8 | FP8 head, Triton W8A8 tiles, asynchronous swap, shortlist draft head, native int8 activation quantisation, backstep guard |
| `vllm-v6/` | 51 | INT8 execution (dense linears, head, W4A8 experts, drafter), INT8 KV on FlashAttention and QSA, in-graph PLE gather, kernel cuts, fast sampler |

Also here: `vllm-xpu-kernels/` (15 commits on upstream `6b30c9b`), `vllm-xpu-kernels-py/` (the astra Python modules),
`libgdn-index64/`, `sycl-tla/`. Patch by patch: [engine details](../docs/engine-details.md). Every switch: [switches](../docs/switches.md).

## Software

| | version |
|---|---|
| GPU driver | Linux xe (kernel 6.18 on our host) |
| compute-runtime (Level Zero) | 26.35.39758.10 |
| oneAPI compiler (builds only) | 2026.1.0 for the kernels; 2026.1.1 for the native libraries |
| Python, torch, triton | 3.12, 2.14.0+xpu, 3.8.0+xpu |
| oneCCL | 2022.1.2 (pip), with the legacy plugin rebuilt from oneCCL#229 |
| vllm-xpu-kernels | 0.1.16.2 + our 15 commits, built as a whole wheel |
| other pins | transformers 5.17.0, auto-round-lib 0.16.0 |

## Build and run

The commands behind our environment, with our paths as placeholders. They have been checked against that environment file by file, and have not been run end to end from a clean machine.

1. **Base.** compute-runtime 26.35.39758.10 on the host, the xe driver, oneAPI DPC++ 2026.1.

2. **vLLM and the Python environment.**
   ```bash
   git clone -b v0.32.0-b70.6 https://github.com/Lumnus/vllm.git vllm-b70-0.32
   uv venv --python 3.12 /opt/b70/venv
   uv pip install --python /opt/b70/venv/bin/python --index-strategy unsafe-best-match \
       --extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://wheels.vllm.ai/xpu/ \
       -r vllm-b70-0.32/requirements/xpu.txt torch==2.14.0+xpu oneccl==2022.1.2 auto-round-lib==0.16.0 transformers==5.17.0
   VLLM_TARGET_DEVICE=xpu uv pip install --python /opt/b70/venv/bin/python --no-build-isolation -e vllm-b70-0.32
   # triton's XPU launcher needs CL/cl.h; intel-sycl-rt ships only include/sycl. Check:
   printf '#include <sycl/detail/cl.h>\nint main(){}\n' | g++ -std=c++17 -I/opt/b70/venv/include -x c++ -fsyntax-only -
   ```

3. **vllm-xpu-kernels, a whole wheel.** A build venv with `torch==2.14.0+xpu cmake ninja 'setuptools<80' setuptools-scm wheel packaging jinja2 regex psutil build`. A cold build took 5 h 27 min and peaked near 120 GiB at 16 jobs; lower `MAX_JOBS` on a smaller builder, and put ccache on `PATH` so that later builds reuse it.
   ```bash
   git clone -b b70/v0.1.16-b70.6 https://github.com/Lumnus/vllm-xpu-kernels.git
   # the fast fp16 + e4m3 KV path needs intel/sycl-tla#859:
   git clone https://github.com/intel/sycl-tla.git sycl-tla-b70 && git -C sycl-tla-b70 checkout 87f6850680a580654b9ea2c80dbc01aeb36ad231
   git -C sycl-tla-b70 am <kit>/v0.32/engine/patches/sycl-tla/*.patch
   source /opt/intel/oneapi/2026.1/oneapi-vars.sh
   export VLLM_CUTLASS_SRC_DIR=$PWD/sycl-tla-b70 VLLM_XPU_ENABLE_XE3P=OFF VLLM_XPU_ENABLE_ONEDNN=ON MAX_JOBS=16
   export VLLM_XPU_XE2_AOT_DEVICES=bmg-g31-a0 VLLM_XPU_AOT_DEVICES=bmg-g31-a0 VLLM_VERSION_OVERRIDE=0.1.16.2+b70.6
   cd vllm-xpu-kernels && python setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
   uv pip install --python /opt/b70/venv/bin/python --no-deps dist/*.whl
   ```

4. **The astra Python modules** (the wheel's `fused_moe_interface.py` stays as built):
   ```bash
   cd /opt/b70/venv/lib/python3.12/site-packages
   for p in <kit>/v0.32/engine/patches/vllm-xpu-kernels-py/*.patch; do git apply -p1 --include='*/astra*' "$p"; done
   ```

5. **Native libraries.**
   1. **wu1ff's**, from [wu1ff/B70-LLM-Controller](https://github.com/wu1ff/B70-LLM-Controller) @ `5e5cf43` (MIT), `model-packs/Qwen3.8-Flash-Next/patches/production/native`: build with the pack's scripts (icpx 2026.1.1, torch headers from the venv) and install `libastra_*.so`, `libbeyond_prefill.so`, `libdominant_w4_layout.so`, `liblong_hc.so`, `liblong_prefill_v3.so`, `libserving_prefill.so` into `site-packages/vllm_xpu_kernels/`.
   2. **`libgdn_index64.so`**: the same build after applying [`patches/libgdn-index64/`](engine/patches/libgdn-index64/) to the pack's `dependencies/gdn-headers`, into the same directory.
   3. **`libb70_moe_decode.so`** (2-card presets), from [Niluthon/qwen38-flash-tp2](https://github.com/Niluthon/qwen38-flash-tp2) @ `e536e33`:
      ```bash
      T=$(/opt/b70/venv/bin/python -I -c "import torch, os; print(os.path.dirname(torch.__file__))")
      icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg" -O3 -std=c++20 -shared -fPIC -ffast-math \
        -D_GLIBCXX_USE_CXX11_ABI=1 -I$T/include -I$T/include/torch/csrc/api/include -x c++ kernels/b70-moe-decode/b70_moe_decode.sycl \
        -o /opt/b70/libb70_moe_decode.so -L$T/lib -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_xpu -Wl,-rpath,$T/lib
      ```

6. **oneCCL legacy plugin**, built against the venv's oneCCL:
   ```bash
   git clone -b master-v2 https://github.com/uxlfoundation/oneCCL.git && cd oneCCL
   git fetch -q origin pull/229/head && git checkout FETCH_HEAD   # caeb329, on master-v2 c5108e7
   source /opt/intel/oneapi/compiler/2026.1/env/vars.sh
   V=/opt/b70/venv; mkdir -p vinc /opt/b70/oneccl-legacy && ln -sfn $V/include/oneapi vinc/oneapi
   icpx -w -fsycl -O2 -fPIC -shared -std=c++17 -DNDEBUG -Ivinc -Isrc -Iinclude \
       plugins/legacy/ccl_legacy.cpp src/debug.cpp -o /opt/b70/oneccl-legacy/libccl_legacy.so \
       -L$V/lib -lccl -l:libccl.so.2 -ldl -Wl,--no-undefined -Wl,-soname,libccl_legacy.so
   ```

7. **Weights:** [Flash-Next](models/qwen38-flash-next/meta.md#weights).

8. **Run.** Presets on disjoint cards run side by side: set `ZE_AFFINITY_MASK`, the port and the `/cache` paths per engine.
   ```bash
   VENV=/opt/b70/venv v0.32/engine/serve.sh qwen38-flash-next preset-d8-6-4card
   DRY=1 v0.32/engine/serve.sh qwen38-flash-next preset-s8-6-2card     # print the command only
   ```
   `serve.sh` sources the preset's `.env`, reads its `.args`, links the weights with the preset's `config.json` into a serve dir, and puts `$VENV/lib` first on `LD_LIBRARY_PATH`.

## Host RAM

Pinned memory does not count against a container limit, so watch the host's MemAvailable.

| preset | host RAM held |
|---|---|
| 2 cards, 32 GiB KV tier | about 90 GiB per engine (two engines on our 251 GiB host left 61 GiB) |
| 4 cards, 128 GiB KV tier | about 200 GiB (MemAvailable 40 GiB on our 251 GiB host) |

Each GiB of KV tier is one GiB of pinned host RAM.
