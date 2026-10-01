# Known issues

Each item: symptoms → cause → workaround. Status as of 2026-10-01.

## 1. A card keeps timing out driver jobs after a GT reset

- **Symptoms.** `dmesg` on one card's PCI address: an engine reset with `Kernel-submitted job timed out`, then after the
  GT reset repeated `Check job timeout: … guc_id=0, not started`, or `VM job timed out on non-killed execqueue`. The
  engine dies with `DEVICE_LOST`; new processes on that card hang or fail. A PCI function-level reset (FLR) does not
  help.
- **Cause.** A known xe driver bug: when a kernel migration-queue job times out, the driver bans the queue, and the GPU
  stays unusable until the driver is reloaded. The fix, `347ccc0453fc` ("drm/xe: fix job timeout recovery for unstarted
  jobs and kernel queues"), is in Linux 7.1-rc7; we found no 6.18.y backport. A likely trigger is CVE-2026-90047
  (`818bebeb63dd`, "drm/xe: Don't hand out the flat CCS storage as usable VRAM"): page tables placed in memory the
  compression hardware overwrites. Fixed in 6.18.51 and 7.2.5. The ban is driver state, which is why FLR (a hardware
  reset) leaves it in place.
- **Workaround.** Stop the engine, make sure nothing holds that card's `/dev/dri` nodes, then rebind the driver for that
  card only: `echo <BDF> > /sys/bus/pci/drivers/xe/unbind; echo <BDF> > /sys/bus/pci/drivers/xe/bind`. Check that GuC
  loads again, re-apply any fan table (it does not survive a rebind), and run a short GPU probe and a TP all-reduce
  smoke test before serving. We rebind all four cards whenever the engine is stopped. Reboot only if the unbind hangs,
  the bind fails to load GuC, the card still fails after a rebind, or several cards are wedged at once. After a burst of
  `DEVICE_LOST` errors, re-hash the weights and libraries (compute-runtime #966: host page-cache corruption that survives
  a rebind). Longer term: a kernel ≥ 6.18.51, better one with `347ccc0453fc`.

## 2. Boot fails with a CPU KV tier of ≥ ~31 GiB per rank

- **Symptoms.** `--kv-offloading-size 128` on TP4 (32 GiB per rank): on torch 2.13 every rank segfaults while filling
  the pool; torch 2.14 raises an allocation error instead.
- **Cause.** compute-runtime refuses one pinned host allocation of about 31 GiB or more (the limit is between 30 and
  31 GiB).
- **Workaround.** Patch 0018 (in the series) splits the pool into equal chunks when one allocation would be too
  large; nothing to set. Without 0018, `NEOReadDebugKeys=1 AllowUnrestrictedSize=1` in the engine environment lifts the
  driver limit.

## 3. Cold Triton cache: `fatal error: CL/cl.h` on every rank

- **Symptoms.** A fresh environment with intel-sycl-rt 2026.1.1 (the torch 2.14 stack) fails at boot on every rank
  when Triton builds its XPU launcher: `fatal error: CL/cl.h: No such file or directory`. A warm Triton cache hides it.
- **Cause.** intel-sycl-rt 2026.1.1 ships `include/sycl` but no `include/CL/*.h`.
- **Workaround.** Put the Khronos OpenCL headers into the environment's `include/CL/`. Check a new environment with
  `printf '#include <sycl/detail/cl.h>\nint main(){}\n' | g++ -std=c++17 -I$VENV/include -x c++ -fsyntax-only -`.

## 4. NaN "duct" loops with non-power-of-two cudagraph capture sizes

- **Symptoms.** In the minutes after a boot, some requests emit token 1023 ("duct") from the first token: finish
  `repetition`, or reasoning to the cap with empty content, or HTTP 400 "Out of range float values are not JSON
  compliant: nan" when logprobs are requested.
- **Cause.** Seen only with non-power-of-two capture sizes (e.g. 6, 12, 14 added to reach 12–14 slots); strong
  evidence, not a root cause.
- **Workaround.** Keep `cudagraph_capture_sizes` to powers of two: with `[1,2,4,8,256,512,1024]` the same items were
  60/60 clean; we serve 16 slots with `[1,2,4,8,16,256,512,1024]`. `B70_DEFAULT_REPETITION_DETECTION` (0010) bounds a
  loop if one happens anyway.

## 5. Edge line (vLLM main + torch 2.14): progressive decode corruption under concurrency

- **Symptoms.** After minutes to hours of concurrent use, answers turn into `!!!!` after the first token. With logprobs,
  every bad token has logprob −12.42 = −ln(vocab size): exactly uniform logits. Batched decode (≥2 requests) breaks
  first; later single requests and fresh prefills break too, and it does not recover without a restart. The warm-up
  passes, so a clean boot proves nothing.
- **Ruled out so far.** The CPU KV tier (breaks with offload off), main's pre-KV cudagraph profiling capture on XPU
  (breaks with it disabled), preemption (none happened), a full KV pool (breaks without one), the mixed
  prefill/decode path (it does not rescue decode).
- **Open.** The official kernels with a block cap, eager mode, piecewise-only graphs, and the prefix-cache retention
  tail. Upstream look-alikes: vllm-project/vllm#48327, intel/llm-scaler#698.
- **Workaround.** Use the stable line (`0.30.0-b70.N`), which has not shown it. The edge line is not recommended until
  this is found.

## 6. MTP speculative decoding: NaN with ≥ 3 concurrent requests

- **Symptoms.** With MTP (3 draft tokens) on the stable line, single-request decode is fast (97 vs 55 tok/s), but as
  soon as three or more requests share a step the logits go NaN and the output degenerates; the engine stays poisoned
  afterwards. A clean single-request benchmark proves nothing; test with simultaneous bursts.
- **Cause.** Under investigation. The threshold matches the first step that holds a chunked prefill next to new
  prefills or verification rows (a lead, not a finding). The GDN layers' inputs and outputs stay finite, so the search
  is in the last full-attention layer or later, and in the draft model's KV binding. A port of vllm-project/vllm#55506
  did not fix it.
- **Workaround.** Do not enable MTP for concurrent serving. The work-in-progress branch is listed in the README.
