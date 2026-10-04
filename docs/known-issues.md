# Known issues

Each item: symptoms → cause → workaround. Status as of 2026-10-03 (0.30.0-b70.2).

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
  smoke test before serving. We rebind the cards only after an unclean end (a crash, a stop timeout, GPU errors in the
  log, a wedge); a clean stop needs no rebind, and it costs ~2 min. Reboot only if the unbind hangs,
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
  60/60 clean; without MTP we serve 16 slots with `[1,2,4,8,16,256,512,1024]`. With MTP (item 6) we
  capture every multiple of 4 up to 64, which includes non-powers of two, and have seen no such loop (0/90 bursts,
  concurrency sweep 1–10, 1,000 quality answers): the padding-row bug in item 6 is a likely common cause, not a proven one. `B70_DEFAULT_REPETITION_DETECTION` (0010) bounds a
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

## 6. MTP speculative decoding: NaN unless the step is never padded and scheduling is synchronous

- **Symptoms.** With MTP (3 draft tokens) and default settings, single-request decode is fast (~90 vs ~55 tok/s), but
  when three or more requests share a step the logits go NaN and the output degenerates; the engine stays poisoned
  afterwards. A clean single-request benchmark proves nothing; test with simultaneous bursts.
- **Cause.** Two separate bugs, both reproduced on the GPU. (a) A FULL graph replay padded with a fake request lets the
  padding row write recurrent state into block 0, which for this hybrid model is one page shared by every GDN/PLE state
  and the attention null pages; masked attention then reads 0 × NaN. (b) Speculative decoding with asynchronous
  scheduling (vLLM's default) leaves residual NaN; the leading candidate is a staging-buffer reuse race.
- **Workaround (what we serve, 0/90 bad bursts).** `--no-async-scheduling`, and cudagraph capture sizes at every multiple
  of 1+k up to `max-num-seqs`×(1+k) (4, 8, …, 64 for k=3 and 16 slots), so a uniform spec-decode step is never padded
  (`engines/serve-s16-mtp3.args`). The cost: concurrent long-prompt TTFT ~35–40 % worse than without MTP. The proper
  fixes (padding rows that write nothing; the race) are work in progress.
- **Structured output.** Grammar-trimmed drafts give a row of fewer than 1+k tokens, which the stock 0.1.14 GDN kernel
  rejects (the engine crashes on the first `json_schema` request). Fixed by upstream vllm-xpu-kernels #600, which is in
  our 0.1.14.1+b70.2/b70.3 builds; 0026 (on by default) avoids it with stock kernels.
- **A quality note.** In hands-on use, games built through a multi-agent orchestrator came out slightly buggier with MTP
  on than off. Rejection sampling should leave the output distribution unchanged, so a real difference would point to
  numerics. A paired MTP on/off evaluation on one engine has not been run yet.

## 7. Intel AutoRound: two digit-run repetition stops in the coding set

- **Symptoms.** 2 of 150 coding samples (the same test-writing task) ended `finish=repetition`: the reasoning ran into
  `9999…` or `0000…` while enumerating edge cases, and the server's repetition stop (0010) ended it. AWQ had 0. It is
  ordinary text, not the NaN signature, and 1,000 of 1,000 MMLU/TruthfulQA answers ended normally.
- **Cause.** Unknown. Too few to call; the AWQ runs were without MTP, so weights and MTP are confounded.
- **Workaround.** Keep `B70_DEFAULT_REPETITION_DETECTION` on (it bounds the loop); a presence penalty would also help.

## 8. Intel AutoRound: group scales are F16, served as BF16

The checkpoint's int4 group scales are F16; vLLM creates the scale parameters in the model dtype (BF16) and the loader
casts them. On layer 0 expert 0 the rounding is at most 3.9e-3 relative. No quality difference to AWQ shows on any test
(docs/measurements/b70.2.md); whether Intel's own accuracy figures were measured with F16 scales is not known to us.

## 9. Prompt logprobs spill VRAM into host RAM (fixed by 0029b)

Full-vocab `prompt_logprobs` builds a [rows, vocab] fp32 tensor per step that vLLM's memory profile does not reserve.
At `--gpu-memory-utilization 0.88` with MTP only 0.3–0.8 GiB is free per card, and the driver spills ~3 GiB per rank
into host RAM, never returned. 0029b chunks it to 128 rows on Model Runner V2 (the runner this model uses; 0029 patches
the V1 runner and does not act here). Check for the one-time `B70-0029b` log line before requesting prompt logprobs.

## 10. A little host memory stays with the driver after each engine stop

Across an engine stop (and a card rebind), ~0.8 GiB of host memory stays allocated in the kernel with no owning
process; it was ~2 GiB per cycle before we turned the IOMMU off in the BIOS. The cause is not identified. It matters only
on hosts that stop and start engines often without rebooting.

## 11. Log noise: the GDN fallback warning every 2 s (fixed by 0032)

With 0027 and no `libgdn_index64.so`, a GDN mode that asks for an `index64` variant (through `B70_GDN_MODE` or 0023's
mode file) falls back to the `official` variant and, before 0032, logged that warning on every re-read (every 2 s).
0032 logs it once per requested mode. Harmless. Note: with **stock** 0.1.14.1 kernels and no `libgdn_index64.so`, the stock GDN op is
32-bit and an engine dies with `DEVICE_LOST` once a block id passes 5,042; keep wu1ff's library or use our kernels.

## 12. AWQ on this release: tested on the frozen stack, not on this exact tree

AWQ (`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`) has run on the frozen stack (2026-10-04): MTP k=3, 16 slots, 128 GiB CPU
tier, vllm-xpu-kernels 0.1.14.1+b70.3 (B70-K1), and 0029b as B70-0030. Measured there: the engine boots on the first
pass (first token 363 s after start); the K1 load line appears on all four workers; the sweep over 1–10 streams has 0 bad
outputs, the burst gate 0 of 90, the structured-output gate 36 ok + 18 plain ok with the engine alive; host memory is flat
under two six-session replays (xe host +0.32 GiB total over the run, ~480 GB and ~355 GB of offload loads); 0 tracebacks,
0 `DEVICE_LOST`. Speeds and quality: README and [measurements](measurements/b70.2.md).

What is still not measured: AWQ on the b70.2 series as released here (the AWQ tree ran 0030 on the b70.1-MTP branch,
without 0031; 0031 is redundant with b70.3 and stays off); AWQ with the kernels at 48K prompts; token-level identity
of the direct copy against staging on AWQ.
