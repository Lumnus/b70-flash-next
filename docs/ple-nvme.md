# The INT8 PLE table served from NVMe (patches 0013, 0013b)

With 0008 the INT8 PLE table still pins 48.9 GiB of host RAM. 0013 leaves it on the NVMe and keeps only a small pinned
row cache, which frees that RAM for a larger CPU KV tier (or makes the model fit a 128 GB unified-memory box at all).

## Design

1. **A host hook in `execute_model`, before the graph replay, on real batches only** (not dummy or profiling runs). It
   hashes the batch's n-grams on the host (bit-identical to the device function; a boot self-test checks it), resolves
   the row ids against a per-rank pinned row cache, reads the misses from the existing INT8 `.safetensors` with
   `O_DIRECT` (a row is one 4 KiB read, 8 KiB for the ~4 % that cross a page), and launches the existing gather into
   the static prefetch buffer. **No host I/O runs inside a captured region.**
2. **Each TP rank serves only its own rows** (on 4 ranks: two ranks hold bigram heads, two trigram). Nothing is shared
   across processes.
3. **Its own row cache**, not the page cache: pinned slots, a row→slot map, CLOCK eviction that protects the current
   and the previous step. Slot offsets are int64.
4. **0013b** adds a native reader (a C thread pool, one call per batch, GIL released; compiled with `gcc` at first use)
   and an optional one-chunk prefill lookahead read by a background thread per rank.

Switches: `B70_PLE_INT8_NVME*` in docs/switches.md. It needs `B70_PLE_INT8=1` and a table built by
`tools/build_int8_ple.py`.

## Measured (4× Arc Pro B70, TP4 + EP, native reader, 2 GiB pinned row cache per rank, 64 GiB CPU KV tier)

| | INT8 table in RAM | INT8 table on NVMe |
|---|---|---|
| host MemAvailable at idle | 62 GiB | 101 GiB (**~39 GiB freed**) |
| cold prefill 8K / 32K / 64K | 1.74 / 7.42 / 16.47 s | 1.75 / 7.41 / 16.32 s |
| decode, aggregate at 1 / 4 / 8 streams | 57.6 / 182 / 285 tok/s | 55.5 / 177 / 279 tok/s (−2 … −4 %) |
| fidelity vs BF16 reference (KL / top-1) | 0.0163 / 96.35 % | 0.0171 / 96.06 % (at the INT8 noise floor) |

The decode delta is exactly the cost of the per-step host sync the hook needs: a diagnostic mode that keeps the table
in RAM but adds the sync (`B70_PLE_INT8_NVME_SYNC_ONLY=1`) measured the same −2 … −5 %, and 0 on prefill. The NVMe read
path itself adds nothing measurable. Under real agent traffic the row cache hit 97–98 %, with ~2–5 misses per step and
a host bubble of ~0.5 ms p50 / ~1 ms p99. Under a mixed load of heavy and light sessions, light-session inter-token
latency was equal (~30 ms p50); the heavy sessions' TTFT tail was longer (p90 12.0 vs 9.0 s).

Off the GPU: host hash equal to the device function on 13.7 M ids; rows read through the cache byte-equal to the file
on 8,177 rows over 4 ranks including page-crossing rows; per-step host path 0.33 / 0.41 / 0.51 ms at 1 / 4 / 8
sequences; a 1,024-token prefill chunk on a trigram rank reads in 63 ms (py), 31 ms (uring), 23 ms (native).

## Notes

1. Keep the NVMe to this table while measuring; a disk KV tier on the same device competes for the same I/O.
2. The native reader is compiled at first use into `B70_PLE_INT8_NVME_NATIVE_DIR` (default `$TMPDIR/b70-ple-nvme`);
   `TMPDIR` must be writable.
3. A 128 GB unified-memory machine could not hold the BF16 table at all and would spend ~38 % of its memory on the
   INT8 one; this path is what makes the model practical there. Untested on such hardware.
