# Example engines

Three example configurations, the ones behind the numbers in `docs/measurements/`. They are examples: adjust the
paths, the GPU memory fraction and the CPU KV size to your host. All three assume 4× Arc Pro B70 and ~256 GB host RAM.

| engine | PLE table | CPU KV tier | offload fix | host RAM note |
|---|---|---|---|---|
| `r8g-kv64` | INT8, pinned in RAM (48.9 GiB) | 64 GiB | off | ~62 GiB MemAvailable left at idle on our host |
| `r8g-kv64-0014jb` | INT8, pinned in RAM | 64 GiB | on (`JUNCTION=1`, `GDN_BACKSTEP=1`) | same |
| `nvme2-kv96-0014jb` | INT8 from NVMe, 8 GiB row cache | 96 GiB | on | NVMe frees ~39 GiB; 96 GiB KV spends most of it |

Files: `common.env` (shared environment), `<engine>.env` (the differences, plus `KV_OFFLOADING_SIZE` read by the
script), `serve.args` (`vllm serve` flags, one flag and its value per line), `run-example.sh` (docker).

```bash
MODELS=/srv/models CACHE=/srv/cache engines/run-example.sh r8g-kv64-0014jb b70-flash-next:0.30.0-b70.1
```

Notes:

1. **Chat template.** The examples use the checkpoint's own template. Our deployment used a lightly modified one
   (thinking switches off when a client sends `enable_thinking=false` *or* a "none/off" effort); it is not included.
2. **Boot host RAM.** Pinned memory does not count against a container memory limit. Watch the host's MemAvailable
   during the first boot; with the BF16 table and without 0006 the peak is ~191 GiB.
3. **Liveness.** Probe with a 1-token chat completion; `/health` and `/v1/models` stay 200 while the engine is wedged.
   Allow ≥180 s under load: a busy engine queues, it is not down.
4. **compute-runtime.** The numbers were measured on compute-runtime 26.35.39758.10; the base image carries 26.27.
   The image does not change the driver.
5. `--max-num-seqs 8` with power-of-two capture sizes; see docs/measurements/baseline.md item 4 before changing them.
