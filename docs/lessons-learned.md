# Lessons learned

What works on this stack, what does not, and how to tune and test it. The shipped presets already follow every item here;
this page is for changing them. Each model's numbers and gates are on its page in [`v0.32/models/`](../v0.32/README.md).

## 1. Keep these as shipped

1. **`B70_OFFLOAD_GDN_BACKSTEP=0`.** With the CPU KV tier and prefix caching on a hybrid GDN model, the backstep makes a
   prefix-cache hit restore a GDN state built through a different prefill chunk split than the cold path; the output
   diverges from the first token, and in an agent session the model re-issues tool calls it has already finished. Every
   preset ships 0 and the engine forces it off under `mamba_cache_mode=align` (`v0.32/engine/patches/vllm-models/` 0008).
   What it would buy, CPU-tier reuse one block below the replay boundary, is small; GPU prefix-cache hits do not depend on it.
2. **fp8 is the 8-bit KV type; `int8_per_token_head` is not.** fp8 (e4m3, scale 1.0) on the XPU FlashAttention backend
   serves correctly and is what fits the 27B's 147K context on one card. `int8_per_token_head` exists only on `TRITON_ATTN`
   and answered nonsense from the first greedy token on XPU.
3. **Leave oneCCL's `LL` algorithm at its default.** `LL=twoshots` corrupts output on this stack (burst gate 64 of 150 bad).
4. **The GDN op comes from `libgdn_index64.so`.** The stock `_xpu_C` GDN op of vllm-xpu-kernels 0.1.16.2 has 32-bit
   conv-state offsets (upstream #658 is on main, in no release; the PyPI build fails `tests/gdn_attn/test_causal_conv1d.py
   -k large_state_offset`); an int32 wrap at block id ≥ 5,042 killed engines. The engine calls `torch.ops._gdn_index64`.
5. **Plain `B70_QSA_TOPK=native` is not launch-repeatable** (equal scores ordered differently); `native_det` is the
   qualified native mode and the preset value.
6. **Do not capture the sparse attention path in graphs** (`B70_QSA_EAGER_BREAK=0`): beyond 2K tokens the output degrades.
7. **Expandable segments only with the patched oneCCL plugin.** With the stock plugin, oneCCL treats a virtual-memory
   device buffer as host memory and libfabric segfaults ([the engine page](../v0.32/README.md#what-the-032-engine-tunes-flash-next), item 8).
8. **Keep capture sizes at multiples of 1 + K** (8 for DFlash K = 7) up to slots × 8: a FULL graph that pads a speculative
   decode step writes recurrent state into the shared null block.

## 2. Choosing a configuration

1. **Flash-Next: sparse or dense attention.** Sparse (QSA, the default) wins where the context is long: prefill 1.06× dense
   at 131K and 1.31× at 260K, agent-session replays 204.7 s against 230–235 s, long-context tasks 80/78 % against 72/72 %
   at 128K. Dense is 3–11 % faster on short single streams and at 8 streams, and holds about 13 % more GPU KV (397,309
   against 344,706 tokens). Both are the same engine; the attention parameter is on the
   [Flash-Next page](../v0.32/models/qwen38-flash-next/meta.md#parameters).
2. **Flash-Next: 4 or 2 cards.** 4 cards for speed. 2 cards free the other pair for a second model: 145 experts per layer
   per rank live in host RAM and are read over PCIe, so one stream decodes at about 0.37× the 4-card preset. Turn on the
   asynchronous swap (the 2-card preset does): it follows a change of traffic without stalling decode. The fused hot+cold
   decode op with bf16 scales differs from all-experts-in-VRAM at relative L2 ≤ 2.9e-3; teacher-forced fidelity of the
   2-card preset is at the TP4 floor.
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
   ([the engine page](../v0.32/README.md#host-ram-with-three-engines)).
4. **One pinned allocation ≥ ~31 GiB is refused by compute-runtime** unless the engine env carries `NEOReadDebugKeys=1
   AllowUnrestrictedSize=1`; the 4-card preset's 32 GiB per rank needs it. Size the tier as a power of two times the card count.
5. **A cold Triton cache needs `CL/cl.h` in the venv** (build step 2 has the check); a warm cache hides the gap.
6. **Our kernel build has FlashAttention-2, MQA-logits and Xe3P off,** as in every build we serve: the attention libraries
   come from the PyPI wheel, and the fp8 MQA-logits ops are not registered. A model that needs them needs an MQA-on build,
   which needs more than 64 GiB of RAM on the builder.

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
