# The hybrid-model KV-offload miss, and patches 0014a–f

## Symptom

A long document is sent with question A, other traffic pushes it out of the GPU prefix cache, then the same document
is sent again with question B. The CPU offload tier (`--kv-offloading-size`, native backend) holds all of the
document's KV, yet the revisit gets **0** hit tokens and recomputes the whole prompt. Every later revisit misses the
same way. Other documents hit normally. It looks random, and early tests blamed concurrency or a leak. It is neither.

## Cause (vLLM v0.30.0; the same code is on main)

It follows from the prompt's **length**. For a hybrid model (full attention + Mamba/GDN layers, e.g.
Qwen3.8-Flash-Next) with the default `prefix_cache_retention_interval = 0`:

1. **Each request stores one Mamba/GDN state**, at `round_down(L − 1, BLOCK)` (BLOCK = 832 tokens here). The
   full-attention KV of every complete block is stored; the Mamba groups only through the aligned-boundary hand-off.
2. **On a revisit, all KV groups must agree.** The full-attention groups hit up to the block before the question
   diverges. The Mamba group needs a stored state at or below that point; the only one sits at the end of the
   original prompt, inside the part that changed. That group returns 0, and one group returning 0 makes the whole
   request return 0 (`scheduler.py`: `if num_hit_chunks == 0: return 0`), not a partial hit.
3. **It never heals.** The GPU prefix cache handles the same situation by pinning a "shared-prefix junction" at the
   longest single-group hit, so the recompute keeps a Mamba state there. The offload lookup never sets one. The
   recompute therefore stores its Mamba state at the end of the new prompt, again inside the varying suffix, and the
   next revisit misses again.

So a document is lost for good when its last full block reaches into the question: the question starts inside that
block. With a ~20-token question that is roughly `s_q / 832` of arbitrary lengths (~2–3 %), and 100 % of documents
whose length puts them in that band. Agent workloads with a fixed system prompt and a changing last user turn have
exactly this shape whenever the GPU tier has dropped the prefix. The GPU prefix cache shows the same placement effect
for an immediate re-send (a hit lands only on the previous request's last full block).

Byte accounting on the counters matched this exactly: a stored document = N × 40,894,464 B (the 12 full-attention
groups, 832 tokens × 49,152 B per chunk) + 116,195,328 B (one GDN state over the 36 linear-attention groups), and a
missed revisit re-stores exactly one new chunk plus one GDN state (157,089,792 B).

## Fix

| patch | switch | what it does |
|---|---|---|
| 0014a | `B70_OFFLOAD_TRACE` | instrumentation: per-group lookup results, hand-offs, junctions, the group that zeroed a request, a periodic state line. `1` is cheap (events only); `2` logs every lookup. |
| 0014b | `B70_OFFLOAD_JUNCTION=1` | **the heal.** When the first Mamba group cuts the hit below the boundary the full-attention groups confirmed, the connector sets `request.shared_prefix_boundary` there (never lowering an existing one). The core scheduler already honours it: the chunk splitter stops there, the Mamba manager retains that state, and the aligned hand-off stores it. The first revisit misses; later revisits hit. |
| 0014c | `B70_OFFLOAD_GDN_BACKSTEP=N` | **prevention.** Also retain the Mamba states at the N block boundaries below the replay boundary, so a divergence within the last N blocks hits on the first revisit. Cost: N × ~116 MB per request in the CPU tier. |
| 0014d | `B70_OFFLOAD_EMPTY_ADVANCE_GUARD=1` | guard for vllm#56795 (a stored-index advance past keys that are still write-pending under another request). Not the cause here; off in the candidate default. |
| 0014e | — | backport of vllm#51787 (request-scoped recency, tail-before-head eviction across all groups of a request, so a lone Mamba state is not evicted apart from its full-attention chunks). |
| 0014f | `B70_OFFLOAD_GROUP_EVICT=1` | gates 0014e: unset restores upstream v0.30.0 LRU/ARC byte for byte. Also makes TRACE=1 cheap. |

A config-only alternative exists: `--prefix-cache-retention-interval K` (a multiple of the block size) keeps periodic
Mamba checkpoints, at ~116 MB per K tokens per document, and affects both tiers. We have not measured it.

An upstream-style version of 0014b + 0014c for vLLM main (a connector-set junction and a
`--prefix-cache-retention-tail-blocks` option) is on the branch `offload-hybrid-junction` of github.com/Lumnus/vllm.

## Validation (4× Arc Pro B70, TP4 + EP, 64 GiB CPU KV tier, 832-token blocks)

Band test (`tools/repro_band.py`): 3 documents whose checkpoint block reaches into the question ("in band") and 3 that
do not, each stored with question A, flushed from the GPU cache with ~8 × 60K filler documents, then revisited with
question B and again with question C.

| engine | in band, revisit B | in band, revisit C | out of band, B / C | stored per document |
|---|---|---|---|---|
| unpatched | 0/3 (15.2 s TTFT) | 0/3 (15.1 s) | 3/3 · 3/3 | 3.06 GB |
| JUNCTION=1 | 0/3 (14.9 s) | **3/3** (0.70 s) | 3/3 · 3/3 | 3.06 GB |
| JUNCTION=1 + GDN_BACKSTEP=1 | **3/3** (0.96 s) | **3/3** (0.72 s) | 3/3 · 3/3 | 3.18 GB (+3.8 %) |

Concurrent revisits (5 trials, 30 revisits) with JUNCTION=1: 28 hits; the 2 misses were first revisits of in-band
documents and both hit on the next pass, so no document was lost permanently (the unpatched engine lost one in a
single trial). Hit TTFT 0.5–0.7 s, the same as unpatched hits.

A mixed-load run first looked ~50 % slower with the patch. The split showed the engine did the same prefill work at
the same speed (prefill time −0.4 %, RAM hit tokens −0.5 %, per-step time −15 %); that run's answers were 49 % longer
(temperature-0 batched MoE decode is not bit-deterministic, and the first turn already differed), which kept long
prefixes resident longer and added queue time. 0014f then gated the ungated eviction change and made tracing cheap.

CPU unit suites with every switch unset: the upstream suites fail exactly the same 34 tests as unpatched v0.30.0 (GPU-
or weights-bound tests on a CPU host); the b70 and group-evict suites pass.

Not yet covered: quality (MMLU/TQA) and long-context needles on the patched engine, a multi-hour soak, and a
decode-pinned mixed-load A/B.
