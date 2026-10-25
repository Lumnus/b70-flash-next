# 0.32.0-b70.1 — Qwen3.8-Flash-Next on vLLM 0.32, 4× Intel Arc Pro B70

The configuration we serve since 2026-10-24: vLLM **v0.32.0** with wu1ff's 1.0.5 B70 chain, our B70 layers ported
from the 0.30 line, the 0.32-specific fixes below, and two of our upstream vLLM fixes cherry-picked. Speculative
decoding is **DFlash2 with 7 draft tokens** (incoai's 0.6B block-diffusion drafter), async scheduling, 8 sequences,
4096-token prefill chunks, a 128 GiB host-RAM KV tier and the INT8 PLE table on NVMe.

| | |
|---|---|
| code | fork tag [`v0.32.0-b70.1`](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.1) = branch [`b70/v0.32.0`](https://github.com/Lumnus/vllm/tree/b70/v0.32.0) @ `103f2f2eb9`, 49 commits on vLLM v0.32.0 (`5df63a7203`), exported to [`patches/vllm/`](patches/vllm/) |
| weights | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68` (the only build measured on 0.32) + drafter `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744` |
| GPUs | 4× Arc Pro B70 (32 GB each), tensor parallel 4 + expert parallel |
| tested host | AMD EPYC 7402, 256 GB RAM, WD_BLACK SN850X 4 TB NVMe |
| engine files | [`engines/serve.args`](engines/serve.args), [`engines/serve.env`](engines/serve.env); fallback [`engines/serve-no-plugin.env`](engines/serve-no-plugin.env) |
| status | serving since 2026-10-24; gates, quality, speed and an agent-session replay measured (below). No multi-day endurance run on 0.32 yet. No container image. |

The 0.30.0-b70.2 recipe at the top of this repository stays as it was and is the long-term fallback line.

## Numbers

Three configurations measured with the same harness on the same host, one after another, on an idle engine (each
timed cell waits for 20 s with no running or waiting requests). Single runs unless noted; read differences under
~5 % as noise. These are **not** comparable with the tables in the main README, which use a different harness
(the b70.2 Intel row there reads 90.4 tok/s at one stream; this harness reads 107.5 for a closely matching configuration).

| | 0.30.0-b70.2 (as released: MTP k=3, 16 slots) | 0.30 perf line (DFlash2 K=7, 8 slots) | **0.32.0-b70.1** |
|---|---|---|---|
| decode, 1 stream (tok/s) | 107.5 | 166 | **169.3** (172.3 before RC1 and the plugin) |
| decode, 4 concurrent requests (total tok/s) | 295 | 287 (repeat 296) | **306.2** |
| decode, 8 streams (total tok/s) | 370 | 588 / 601 | **605.5** (before RC1 and the plugin: 605 / 622) |
| prefill 8K / 32K / 131K (tok/s) | 4,052 / 3,833 / 2,953 | 6,125 / 6,092 / 4,648 | **6,305 / 6,114 / 4,641** |
| GPU KV cache (tokens) | ~311K | 406,432 | **397,309** |
| agent replay: wall time, TTFT p50 / p90, preemptions | 401 s, 2.90 / 6.57 s, 0 | 302 s, 2.25 / 4.85 s, 0 | **268 s, 2.13 / 4.71 s, 0** |
| quality, MMLU 300 / TruthfulQA 200 | 259 / 177 | paired vs the left column: n.s. | paired vs the 0.30 perf line: 1+1 / 0+1 discordant, p = 1.0 / 1.0 |
| gates | pass | pass | burst 0/150, structured output clean, mix 7/7, greedy compare pass |
| host MemAvailable while serving (256 GB host, ~20 GB of other work on it) | — | ~44–46 GiB | **35–39 GiB** |

How each row was taken:

1. **Decode, 1 and 4.** `bench_det` (taken from Niluthon/qwen38-flash-tp2, Apache-2.0): steady decode milliseconds
   per step, converted with the measured tokens accepted per step.
2. **8 streams.** Eight parallel generations, total generated tokens over wall time. The second 8-stream run of the
   served configuration overlapped other traffic and is not reported.
3. **Prefill.** One uncached, salted prompt of 8K, 32K or 131K tokens; prompt tokens over time to first token.
4. **KV.** The GPU KV cache size vLLM reports at start.
5. **Replay.** Six recorded agent sessions (243 requests, salted per run) replayed deterministically against the
   128 GiB CPU tier. The 0.32 row ran on `serve-no-plugin.env` (the KV offload code is the same; the plugin was added
   after it); the plain 0.32 def without the private pool needed 281 s with 2 preemptions.
6. **Quality.** MMLU 300 and TruthfulQA MC1 200, the same items for every configuration, paired McNemar test against
   the configuration it replaced. Every step of the chain was not significantly different.
7. **Gates.** Burst: 150 requests landing in the same steps at concurrency 3, 4 and 8, checked for NaN and garbage
   output. Structured output with speculative decoding. A mixed long/short probe. Greedy token compare at
   concurrency 1 and 8 against the previous configuration (near-tie flips only).
8. **Long context**, decode at one stream, two runs each: 8K / 64K / 128K prompts 209 / 200 / 172 tok/s on 0.32 vs
   217 / 220 / 169 on the 0.30 perf line (noise level at n = 2).

What 0.32 costs: about 9K fewer GPU KV tokens than the 0.30 perf line, and less host-RAM headroom (35–39 vs 44–46 GiB
on our host).

## What changed against the 0.30 line, and why 0.32 was slower at first

Our first 0.32 build served correctly but was ~25 % slower at one stream. Measured causes, largest first:

1. **compute-runtime copy-through-lock.** Every small per-step host→device copy waited for the in-order queue to
   drain, so host preparation never overlapped the GPU: about 9.5 ms per step at one stream (found with phase timing
   and `py-spy --native`: the time was spent in `zeCommandListAppendMemoryCopy`). Fix: the compute-runtime keys
   `ExperimentalCopyThroughLock=0`, `ExperimentalH2DCpuCopyThreshold=0`, `ExperimentalD2HCpuCopyThreshold=0`.
2. **The DFlash2 fixed-order hyper-connection down GEMM** is 4.5–6.7× slower than `F.linear` above 32 rows. It was the
   whole prefill gap and most of the 8-stream gap. Fix: B70-V32-07, `B70_HC_ORDERED_MAX_ROWS=32`.
3. **The prefill hyper-connection graph regions** held ~0.7 GiB per card. Fix: B70-V32-08, `B70_PREFILL_HC_GRAPHS=0`.
   Together with the FP8 head (B70-V32-0053), the KV cache went from 337K to 389K tokens.
4. **PLE n-gram ids hashed on the host** (B70-V32-03): one small device→host copy replaces ~50 eager hash kernels
   (26.7 → 25.2 ms per step).
5. **K = 7 is the drafter's tuned shape.** Our earlier 0.32 runs used MTP k=3 or DFlash K=3.
6. A measurement lesson: GPU timing events on the host's critical path (B70-V32-02, diagnostic, off by default) cost
   ~3 ms per step themselves. Every number above comes from runs with them off.

Then two host-memory fixes:

7. **B70-V32-RC1, `B70_OFFLOAD_PRIVATE_PINNED=1`.** vLLM 0.32 keeps the CPU KV tier in a shared `/dev/shm` region
   registered with the device. SYCL reports registered memory as `usm::alloc::unknown`, so every CPU→GPU load was
   staged through torch's caching host allocator, and those blocks stayed cached: ~3.25 GiB per rank, ~13 GiB of host
   RAM. A private pinned pool per rank (as on 0.30) is copied directly. It also removes the need for a ≥128 GiB
   `/dev/shm`. The upstream form of the kernel-side half is vllm-project/vllm-xpu-kernels#662.
8. **Expandable segments with a patched oneCCL plugin.** torch's `expandable_segments` puts XPU tensors in Level Zero
   virtual memory, which compute-runtime 26.35 reserves with a CPU mapping on Battlemage. oneCCL's legacy plugin
   classifies a buffer as host memory when `msync()` succeeds on it, sends it to the CPU collective without a stream,
   and libfabric then copies the device address on the host and segfaults. The plugin fix (uxlfoundation/oneCCL#229)
   asks SYCL for the pointer type when a queue is given. It buys ~8K KV tokens here. Without the patched plugin, do not
   enable expandable segments; use `serve-no-plugin.env`.

Upstream fixes folded in (all six are our posted PRs; none merged as of 2026-10-25):

| fix | upstream | here |
|---|---|---|
| GDN conv-state offset in 64 bits (an int32 wrap at block id ≥ 5,042 killed engines) | vllm-xpu-kernels #658 (fixes #657) | kernels build |
| GDN kernels skip rows whose state index is the null block (graph padding rows wrote state into the shared null page) | vllm-xpu-kernels #660 (fixes #659) | kernels build + `patches/libgdn-index64/` |
| `swap_blocks_batch`: copy pinned (and registered) host sources directly, `snapshot_src=False` | vllm-xpu-kernels #662 (fixes #661) | kernels build (pinned half) + caller B70-V32-PR2 |
| Model Runner V2 prompt-logprobs chunks bounded by `max_num_reqs` (a 1024-row chunk spilled VRAM into host RAM) | vLLM #63564 (fixes #63563) | cherry-picked; replaces our 0029b |
| KV offload: pin a shared-prefix junction when a hybrid model's sparse group cuts the offload hit | vLLM #63566 (fixes #63565) | cherry-picked; replaces our 0014b |
| oneCCL legacy plugin: device USM that `msync()` sees as host takes the GPU path | oneCCL #229 (fixes #228) | plugin build, step 6 |

## The series

[`patches/vllm/`](patches/vllm/) is `git format-patch v0.32.0..v0.32.0-b70.1` of the fork; `scripts/check-series.sh`
fails if it differs from a fresh export.

| patches | what | author |
|---|---|---|
| 0001 | wu1ff's MIT license and pack notice, under `b70/third_party/wu1ff/` | Lumnus |
| 0002–0016 | wu1ff's 1.0.5 chain for vLLM 0.32 (target port, DFlash2, C1 verify/target graphs, astra500 MoE/HC/prefill paths, serving fixes), the `vllm/` hunks of his patch-chain taken unchanged, one commit per patch | wu1ff |
| 0017–0034 | our 0.30 layers ported: INT8 PLE from NVMe on wu1ff's host-prepared PLE (B70-L32-0013), thinking budget per effort, repetition stop, effort alias, dense-QSA indexer skip (main + MTP), MTP draft prefill graphs, uniform drafts guard, step trace, offload trace / junction / GDN backstep / empty-advance guard, prompt logprobs chunks, and their CPU tests | Lumnus |
| 0035–0037 | revert of our 0029b port, then vllm-project/vllm#63564 and #63566 | Lumnus |
| 0038–0049 | 0.32-specific: B70-V32-PR2, -01, -0053, -02/-02b (diagnostic), -03/-03b, -04, -07, -0054 (off), -08, -RC1 | Lumnus |

Off by default and not used in the served configuration: B70-V32-01 (`B70_C1_EAGER_ROUTE`), -02/-02b
(`B70_PHASE_TIMING`), -04 (`B70_HC_PARALLEL_MAX_ROWS`, superseded by -07), -0054 (`B70_MOE_DECODE_OP_RESIDENT`; the
astra 8-row path is faster at decode, and it needs a separate library built from Niluthon's `b70-moe-decode`).

[`patches/vllm-xpu-kernels-py/`](patches/vllm-xpu-kernels-py/): the Python side of vllm-xpu-kernels 0.1.15.4 —
wu1ff's `vllm_xpu_kernels/` hunks (0001–0007, astra500 wrappers and the fused-MoE interface) plus B70-V32-05
(`B70_ASTRA_MID`, rejected, off) and B70-V32-06 (`B70_ASTRA_OWNERSHIP=0`, the EP layout we serve).

## Software

| | version |
|---|---|
| GPU driver | Linux xe (kernel 6.18 on our host) |
| compute-runtime (Level Zero) | 26.35.39758.10 |
| oneAPI compiler (builds only) | 2026.1.1 (`icpx`) |
| Python | 3.12 |
| torch / triton | 2.14.0+xpu / triton-xpu 3.8.0 |
| Intel runtime wheels | intel-sycl-rt, dpcpp-cpp-rt, intel-opencl-rt 2026.1.1; onemkl-sycl 2026.1.0 |
| oneCCL | 2022.1.2 (pip `oneccl`) core and loader, legacy plugin rebuilt with oneCCL#229 |
| vllm-xpu-kernels | `release/0.1.15.4` + #658, #660 and an earlier pinned-only form of #662, oneDNN on |
| other pins | transformers 5.17.0, auto-round-lib 0.16.0 |
| vLLM | v0.32.0 + this series |

`/dev/shm`: no requirement any more (B70-V32-RC1). Without RC1, vLLM 0.32 maps the whole CPU KV tier into `/dev/shm`,
so a container needs `/dev/shm` ≥ 140 GiB for a 128 GiB tier.

## Host RAM

On our 256 GB host the engine takes ~164 GiB after start: the CPU KV tier is 128 GiB of pinned memory (32 GiB per
rank, which is why `AllowUnrestrictedSize=1` is needed), the PLE row cache 8 GiB, and the rest is the four workers.
MemAvailable stays at 35–39 GiB with ~20 GiB of other work on the host. Pinned memory does not count against a
container memory limit, so watch the host's MemAvailable, not the container's. With less RAM, lower
`--kv-offloading-size` (keep it a power of two times 4: 64 frees 64 GiB) or move the PLE table into RAM only if you
have the 49 GiB it then needs.

## Build and run

The commands we ran, with our paths replaced by placeholders. Not run end to end in this form.

1. **Base.** compute-runtime 26.35.39758.10 on the host, the xe driver, and oneAPI DPC++ 2026.1.1 for the builds.

2. **Python environment.**
   ```bash
   git clone -b v0.32.0-b70.1 https://github.com/Lumnus/vllm.git vllm-b70-0.32
   uv venv --python 3.12 /opt/b70/venv
   uv pip install --python /opt/b70/venv/bin/python --index-strategy unsafe-best-match \
       --extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://wheels.vllm.ai/xpu/ \
       -r vllm-b70-0.32/requirements/xpu.txt torch==2.14.0+xpu oneccl==2022.1.2 auto-round-lib==0.16.0 transformers==5.17.0
   VLLM_TARGET_DEVICE=xpu uv pip install --python /opt/b70/venv/bin/python --no-build-isolation -e vllm-b70-0.32
   # triton's XPU launcher needs CL/cl.h; intel-sycl-rt ships only include/sycl. Check:
   printf '#include <sycl/detail/cl.h>\nint main(){}\n' | g++ -std=c++17 -I/opt/b70/venv/include -x c++ -fsyntax-only -
   ```

3. **vllm-xpu-kernels.** Branch `release/0.1.15.4` of vllm-project/vllm-xpu-kernels (`ddf336d`) plus the three
   kernel PRs, taken from their upstream pull refs: #660 (`848aca4`), #658 (`ab17bc7`) and #662 (`dc16c9f`). What we
   serve was built with an earlier form of #662 (`3c8d392`, the pinned-memory half only); the posted head adds
   registered memory and its test, which this configuration does not reach because RC1's pool is pinned. Building with
   `dc16c9f` is what we would do today; we have not booted the engine on that build. Our #658 commit (`71b7c5a`)
   differs from the posted `ab17bc7` only in its message.
   ```bash
   git clone -b release/0.1.15.4 https://github.com/vllm-project/vllm-xpu-kernels.git && cd vllm-xpu-kernels
   for pr in 660 658 662; do git fetch -q origin pull/$pr/head; git cherry-pick FETCH_HEAD; done
   VLLM_XPU_ENABLE_ONEDNN=ON VLLM_XPU_ENABLE_XE3P=OFF FA2_KERNELS_ENABLED=OFF MQA_LOGITS_KERNELS_ENABLED=OFF \
       python -m build --wheel --no-isolation
   ```
   oneDNN is needed for the FP8 head (B70-V32-0053). We built with flash-attention and MQA-logits off to save build
   time and copied four unchanged libraries from the official 0.1.15.4 wheel into the installed package:
   `_vllm_fa2_C.abi3.so`, `libattn_kernels_xe_2.so`, `libattn_kernels_xe_3.so`, `libmqa_logits_kernels_xe_2.so`.
   A full build with those options on should be equivalent; we did not run one.

4. **Kernels, Python side.** Apply [`patches/vllm-xpu-kernels-py/`](patches/vllm-xpu-kernels-py/) to the installed
   `vllm_xpu_kernels/` Python files (they must equal 0.1.15.4's):
   `cat patches/vllm-xpu-kernels-py/*.patch | (cd /opt/b70/venv/lib/python3.12/site-packages && git apply -p1)`.
   Or `git am` them onto the kernels source before the build in step 3; they apply cleanly on `release/0.1.15.4` plus
   the three PRs.

5. **wu1ff's native libraries.** From [wu1ff/B70-LLM-Controller](https://github.com/wu1ff/B70-LLM-Controller) @
   `5e5cf43` (release `qwen3.8-flash-next-v1.0.5`, MIT), `model-packs/Qwen3.8-Flash-Next/patches/production/native`:
   build the libraries with the pack's own build scripts (icpx 2026.1.1, torch 2.14 headers from the venv) and install
   `libastra_*.so`, `libbeyond_prefill.so`, `libdominant_w4_layout.so`, `liblong_hc.so`, `liblong_prefill_v3.so`,
   `libserving_prefill.so` into `site-packages/vllm_xpu_kernels/`. Build `libgdn_index64.so` the same way after applying
   [`patches/libgdn-index64/`](patches/libgdn-index64/) to the pack's `dependencies/gdn-headers` (vLLM 0.32 runs the GDN
   op from this library). Put the Level Zero peer-residency shim at `/opt/b70/libl0_peer_residency_shim.so`. Our builds
   are not byte-identical to wu1ff's accepted hashes (different build paths); the shim is.

6. **oneCCL legacy plugin** (only for `serve.env`). One file, built against the venv's shipped oneCCL 2022.1.2:
   ```bash
   git clone -b master-v2 https://github.com/uxlfoundation/oneCCL.git && cd oneCCL
   git fetch -q origin pull/229/head && git checkout FETCH_HEAD   # caeb329, on master-v2 c5108e7
   source /opt/intel/oneapi/compiler/2026.1/env/vars.sh
   V=/opt/b70/venv; mkdir -p vinc /opt/b70/oneccl-legacy && ln -sfn $V/include/oneapi vinc/oneapi
   icpx -w -fsycl -O2 -fPIC -shared -std=c++17 -DNDEBUG -Ivinc -Isrc -Iinclude \
       plugins/legacy/ccl_legacy.cpp src/debug.cpp -o /opt/b70/oneccl-legacy/libccl_legacy.so \
       -L$V/lib -lccl -l:libccl.so.2 -ldl -Wl,--no-undefined -Wl,-soname,libccl_legacy.so
   ```
   `CCL_PLUGIN` in `serve.env` points the shipped loader at it; the venv stays untouched.

7. **Weights.** Intel AutoRound @ `4c67bf68` without the 102.4 GB PLE shard, devan's `ple_table_qwen4exp.pt`, the INT8
   table from `tools/build_int8_ple.py`, and the snapshot from `tools/intel_snapshot.py` (as for 0.30, see
   [engines/README.md](../engines/README.md)). Make a serve dir: symlink every snapshot file into it and copy
   [`engines/serve-config-intel-autoround.json`](../engines/serve-config-intel-autoround.json) over its `config.json`
   (it is byte-identical to the one we serve). The drafter: `incoai/Qwen3.8-Flash-Next-DFlash2` @ `a22a1744` into
   `/models/qwen3.8-flash-next/DFlash2`.

8. **Run.**
   ```bash
   set -a; . v0.32/engines/serve.env; set +a
   mapfile -t args < <(grep -v '^#' v0.32/engines/serve.args | sed '/^$/d' | sed 's/^\(--[^ ]*\) \(.*\)$/\1\n\2/')
   LD_LIBRARY_PATH=/opt/b70/venv/lib /opt/b70/venv/bin/vllm serve /models/qwen3.8-flash-next/serve-dir "${args[@]}"
   ```
   The venv's own oneAPI 2026.1.1 runtime (`$VENV/lib`) must come first on `LD_LIBRARY_PATH`; a mixed runtime
   segfaults at exit. First start compiles cold (minutes); our boots take ~4 min.

Chat template: the examples use the checkpoint's own template. We serve a lightly modified one (thinking switches off
on `enable_thinking=false` or a "none/off" effort); it is not included.

## Known issues (0.32)

1. **No multi-day run yet.** The served configuration has run since 2026-10-24 without an incident we know of; there is
   no endurance number for 0.32.
2. **One hang in development.** Before the copy-through-lock fix, one boot with FULL graphs hung in `sample_tokens`
   (unit stopped, cards rebound). It did not recur in later boots or in the drill; the cause is not known.
3. **KV is ~2 % below the 0.30 perf line.** Peak activation is 1.93 GiB per rank against 1.31 on 0.30, and it is not
   allocator fragmentation (expandable segments bought only ~0.1 GiB). Not yet attributed.
4. **GPU idles ~3 ms per step in the PLE sync at one stream.** A device→host probe takes 25 µs in isolation. Open.
5. **Only Intel AutoRound weights were measured on 0.32.** The AWQ build should work (vLLM 0.32 accepts it natively), but
   it was not booted on this line.
6. **AWQ / 0.30 items still apply** where the code is shared: card hangs after a GT reset ([docs/known-issues.md](../docs/known-issues.md)
   item 1), liveness must generate, and pinned memory escapes container limits.
7. **oneCCL `LL=twoshots`** corrupts output on this stack (burst gate 64/150 bad); every oneCCL change needs the token
   compare and the burst gate before serving.
8. **The kernels build is not attached.** Build it as in step 3 (~26 min on our build host with flash-attention and
   MQA off; much longer with them on).
