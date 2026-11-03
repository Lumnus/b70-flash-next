# Lessons learned

What works on this stack, what does not, and how to tune and test it. The shipped presets already follow every item here;
this page is for changing them. Each model's numbers and gates are on its page in [`v0.32/models/`](../v0.32/README.md).

## 1. Keep these as shipped

1. **`B70_OFFLOAD_GDN_BACKSTEP=0`.** With the CPU KV tier and prefix caching on a hybrid GDN model, the backstep makes a
   prefix-cache hit restore a GDN state built through a different prefill chunk split than the cold path; the output
   diverges from the first token, and in an agent session the model re-issues tool calls it has already finished. Every
   preset ships 0 and the engine forces it off under `mamba_cache_mode=align` (`v0.32/engine/patches/vllm-models/` 0008).
   What it would buy, CPU-tier reuse one block below the replay boundary, is small; GPU prefix-cache hits do not depend on it.
2. **`int8_per_token_head` is the KV type of every Flash-Next preset, and it needs this release's engine and kernels.**
   FlashAttention serves it through the native INT8 paged decode (`B70_PTH_FA_DECODE=1`; without the switch it raises
   rather than answer), and the QSA kernels read and write it. Against BF16 KV it is at the repeat floor past 8K tokens and
   visible below that, mostly on prose ([model page](../v0.32/models/qwen38-flash-next/meta.md#numbers)). On earlier builds
   the type existed only on `TRITON_ATTN` and answered nonsense from the first greedy token on XPU. The Qwen3.8-27B 1-card
   preset runs fp8 (e4m3, scale 1.0) instead, which is what fits its 147K context on one card.
3. **Leave oneCCL's `LL` algorithm at its default.** `LL=twoshots` corrupts output on this stack (burst gate 64 of 150 bad).
4. **The GDN op comes from `libgdn_index64.so`.** The PyPI `_xpu_C` GDN op of vllm-xpu-kernels 0.1.16.2 has 32-bit
   conv-state offsets (upstream #658 is merged on main, in no release; the PyPI build fails `tests/gdn_attn/test_causal_conv1d.py
   -k large_state_offset`); an int32 wrap at block id ≥ 5,042 killed engines. Our kernels carry the 64-bit offsets, and the
   engine still calls `torch.ops._gdn_index64`.
5. **Plain `B70_QSA_TOPK=native` is not launch-repeatable** (equal scores ordered differently); `native_det` is the
   qualified native mode and the preset value.
6. **Do not capture the sparse attention path in graphs** (`B70_QSA_EAGER_BREAK=0`): beyond 2K tokens the output degrades.
7. **Expandable segments only with the patched oneCCL plugin.** With the stock plugin, oneCCL treats a virtual-memory
   device buffer as host memory and libfabric segfaults ([engine details](engine-details.md#what-the-engine-tunes-flash-next), item 8).
8. **Keep capture sizes at multiples of 1 + K** (8 for DFlash K = 7) up to slots × 8: a FULL graph that pads a speculative
   decode step writes recurrent state into the shared null block.

## 2. Choosing a configuration

1. **Flash-Next: sparse or dense attention.** The presets come in both (D for dense, S for sparse), on the same engine.
   Sparse (QSA) wins where the context is long: on the b70.4 build prefill was 1.06× dense at 131K and 1.31× at 260K,
   agent-session replays 204.7 s against 230–235 s, long-context tasks 80/78 % against 72/72 % at 128K
   ([measurements](measurements/v0.32-qsa.md)). Dense is faster on short single streams. On two cards of this release,
   D8-6 decodes 72 / 107 tok/s (prose / code, one stream) and S8-6 67 / 92, while S8-6 holds more GPU KV (503,866 against
   426,060 tokens) and prefills 128K slightly faster (2,505 against 2,412 tok/s). Composing the other one is in
   [the model page](../v0.32/models/qwen38-flash-next/meta.md#composing-other-modes).
2. **Flash-Next: 4 or 2 cards.** 4 cards for speed. 2 cards free the other pair for a second model: 145 experts per layer
   per rank live in host RAM and are read over PCIe, so one stream decodes at about 0.6× the 4-card preset (72 against 124 tok/s on prose with D8-6). Turn on the
   asynchronous swap (the 2-card presets do): it follows a change of traffic without stalling decode. The fused hot+cold
   decode op with bf16 scales differs from all-experts-in-VRAM at relative L2 ≤ 2.9e-3; teacher-forced fidelity of the
   2-card presets is at the TP4 floor.
3. **One card or two for the 27B and the 35B.** One card trades about a third of decode speed and half the prefill for a
   free card; the 27B needs fp8 KV and the W8A8 drafter to reach 147K context there. The 35B holds four full 262K requests
   in GPU KV on two cards.
4. **Sampling for agent work on the 35B:** the model card's precise-coding preset (temperature 0.6, presence 0) is the
   shipped default. Agent capability scores moved by 6 of 50 between sampling presets, which is inside the noise of a 50-task
   gate (Fisher p ≈ 0.23); MMLU and TruthfulQA did not move.

## 3. Host and cards

1. **A card that keeps timing out driver jobs after a GT reset needs an xe rebind, not an FLR.** The procedure, and when a
   reboot is needed instead, is item 1 of [the 0.30 page's list](../v0.30/docs/known-issues.md); it applies unchanged.
   Rebind only after an unclean end (a crash, a stop timeout, GPU errors in the log); a clean stop needs none.
2. **Liveness must generate.** `/health` and `/v1/models` stay 200 while an engine is wedged; probe with a 1-token chat
   completion, and give it ≥ 180 s under load (a saturated engine queues).
3. **Pinned memory escapes container limits.** The KV tier, the PLE row cache and the cold experts are pinned host memory;
   watch the host's MemAvailable, not the container's. Three engines on one 256 GB host leave 41–43 GiB
   ([engine details](engine-details.md#host-ram-with-three-engines); the engine page has the host RAM of the current presets).
4. **One pinned allocation ≥ ~31 GiB is refused by compute-runtime** unless the engine env carries `NEOReadDebugKeys=1
   AllowUnrestrictedSize=1`; the 4-card presets' 32 GiB per rank need it. Size the tier as a power of two times the card count.
5. **A cold Triton cache needs `CL/cl.h` in the venv** (build step 2 has the check); a warm cache hides the gap.
6. **Our kernels are built as a whole wheel,** with FlashAttention-2 and MQA-logits on and Xe3P off, and installed whole:
   a cold build took 5 h 27 min and peaked near 120 GiB of RAM at 16 jobs ([kernels](kernels.md)). Do not lay a reduced
   build over the PyPI libraries and expect the attention paths of this release.

## 4. Measuring and testing

1. **Gate with simultaneous bursts**, not staggered requests: the NaN classes on this stack need several requests landing
   in one step (`burst` 0 of 90–186 on every preset). Add a burst with prefix-cache hits and structured output with
   speculative decoding.
2. **Bit-level token compares do not work beyond a few hundred tokens.** Sparse and dense Flash-Next are run-to-run
   nondeterministic from about token 512; the hot/cold swap and one-card W4A8 shapes make two cold runs differ earlier.
   Compare teacher-forced KL against a same-engine floor, taken at equal concurrency on an otherwise idle engine.
3. **The backstep test** (cold against warm prefix cache with ≥ 1,600 generated tokens) is the gate for any hybrid-GDN
   configuration change; single-request bursts and structured-output gates do not see that class.
4. **A det gate with ignore-EOS at concurrency 8 runs into an invented turn** after the answer on the 27B (the INC w4a8
   path changes prefill numerics with batch shape at its 512-row threshold). Read divergence after the end of the answer
   as that, and check fidelity teacher-forced.
5. **Thinking caps shape quality numbers.** At medium effort and a 2,400-token cap about 10 % of the 35B's quality items
   hit the cap; use the same cap for every row you compare.
6. **Drafter acceptance depends on the text**: report code and prose decode separately (the 35B accepts about 2 of 7 draft
   tokens on prose, 5–6 on code and math).
7. **Timing events on the host's critical path cost ~3 ms per step themselves**; measure with them off.
