# Example engines

Example configurations for 4× Arc Pro B70 and ~256 GB host RAM. Adjust the paths, the GPU memory fraction and the CPU
KV size to your host.

| engine | weights | slots | PLE table | CPU KV tier | offload fix | note |
|---|---|---|---|---|---|---|
| **`s16-kv128-mtp3`** + `MODEL=awq` | AWQ (wtdcode @ `0939125`) | 16 | INT8, from NVMe | 128 GiB (0018 chunks) | on | **what we serve and recommend** (0.30.0-b70.2): MTP k=3, kernels 0.1.14.1+b70.3 |
| `s16-kv128-mtp3` + `MODEL=intel-autoround` | Intel AutoRound @ `4c67bf68` | 16 | INT8, from NVMe | 128 GiB (0018 chunks) | on | the alternative: same code and flags (patch 0028 acts here); verification on the release branch in progress |
| `awq-s16-kv128-chunked` | AWQ (wtdcode @ `0939125`) | 16 | INT8, from NVMe | 128 GiB (0018 chunks) | on | what we served on b70.1, without MTP |
| `r8g-kv64` | devan W4A16 @ `40b8f18d` | 8 | INT8, pinned in RAM | 64 GiB | off | behind the numbers in `docs/measurements/` |
| `r8g-kv64-0014jb` | devan W4A16 | 8 | INT8, pinned in RAM | 64 GiB | on (`JUNCTION=1`, `GDN_BACKSTEP=1`) | same |
| `nvme2-kv96-0014jb` | devan W4A16 | 8 | INT8, from NVMe | 96 GiB | on | NVMe frees ~39 GiB; 96 GiB KV spends most of it |

Files: `common.env` (shared environment), `<engine>.env` (the differences, plus the keys `run-example.sh` reads:
`KV_OFFLOADING_SIZE`, and optionally `MODEL_DIR`, `SERVE_ARGS`, `SERVE_CONFIG`), `models/<model>.env` (one checkpoint's
`MODEL_DIR` and `SERVE_CONFIG`, chosen with `MODEL=<model>`), `serve.args` / `serve-s16.args`
/ `serve-s16-mtp3.args` (`vllm serve` flags, one flag and its value per line), `serve-config-awq.json` and
`serve-config-intel-autoround.json` (serve configs), `run-example.sh` (docker; the b70.1 image).

```bash
MODELS=/srv/models CACHE=/srv/cache engines/run-example.sh awq-s16-kv128-chunked b70-flash-next:0.30.0-b70.1
MODELS=/srv/models CACHE=/srv/cache MODEL=awq engines/run-example.sh s16-kv128-mtp3 <your b70.2 image>
```

## The serving recipe (`s16-kv128-mtp3`, 0.30.0-b70.2; the model is a parameter)

We run it from a source tree, not the image (README, Quick start). One code tree and one set of flags; the weights are
chosen with `MODEL=awq` (recommended: snapshot from `tools/awq_snapshot.py`, `serve-config-awq.json`) or
`MODEL=intel-autoround` (`tools/intel_snapshot.py`, `serve-config-intel-autoround.json`). The configuration: MTP k=3,
16 slots, util 0.88, 262K context, 128 GiB CPU tier, INT8 PLE from NVMe, kernels b70.3. On AWQ it ran 3 days 16 h in one
engine without an error (README, Endurance). AWQ weights and snapshot: step 1 of "The b70.1 serving recipe" below. Intel
differs only in the weights step (Intel specifics, point 1); points 2 to 5 hold for both.

0. **Code.** Fork branch `b70/v0.30.0-stable` (tag `v0.30.0-b70.2`), exported to `patches/`. What was validated where:
   README, "Validation status" (AWQ ran its endurance on an earlier branch of the same code; the differences are listed
   in known issues item 12).

Intel specifics (point 1 only; points 2 to 5 are common):

1. **Weights.** `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68`, every file except
   `model-00016-of-00017.safetensors` (the 102.4 GB PLE shard); devan's `ple_table_qwen4exp.pt` as before. Then:
   ```bash
   tools/intel_snapshot.py snapshot /srv/models/qwen3.8-flash-next/W4A16-AutoRound \
       /srv/models/qwen3.8-flash-next/W4A16/ple_table_qwen4exp.pt /srv/models/qwen3.8-flash-next/W4A16-AutoRound-snapshot
   ```
   The serve config is `serve-config-intel-autoround.json` (it replaces the snapshot's `config.json` at start).
2. **MTP, 3 draft tokens** (`serve-s16-mtp3.args`): `--speculative-config {"method":"mtp","num_speculative_tokens":3}`,
   `--no-async-scheduling`, and capture sizes at every multiple of 4 up to 64 plus 256/512/1024, so a uniform
   spec-decode step is never padded; `B70_MTP_DRAFT_PREFILL_NO_PIECEWISE=1`. Both workarounds are needed
   ([known issues](../docs/known-issues.md) item 6).
3. **GPU memory 0.88** (`--gpu-memory-utilization`): 311,299 KV tokens; card 0 keeps ~0.7 GiB free. 0.90 fails vLLM's
   startup check on compute-runtime 26.35.
4. **Kernels 0.1.14.1+b70.3** with `B70_OFFLOAD_H2D_DIRECT=0`: the kernel copies CPU→GPU KV loads straight from the
   pinned pool, and a 128 GiB CPU KV tier holds host memory flat. With stock 0.1.14.1 kernels set
   `B70_OFFLOAD_H2D_DIRECT=1` (0031), keep wu1ff's `libgdn_index64.so`, and keep `B70_MTP_UNIFORM_DRAFTS` on.
5. **`--enable-prompt-tokens-details`**, so clients see cached prompt tokens in `usage`.

Run `vllm serve <snapshot> $(args from serve-s16-mtp3.args) --kv-offloading-size 128` with `common.env`,
`s16-kv128-mtp3.env` and the model file in the environment. Not exercised in this form: our launcher sets the same flags and variables, plus our modified
chat template (note 1) and a host-RAM guard.

## The b70.1 serving recipe (`awq-s16-kv128-chunked`)

1. **Weights.** Download `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` at revision `0939125`, plus `ple_table_qwen4exp.pt`
   from `devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d` (the BF16 PLE table; the INT8 table is built from it).
   Then make the snapshot the entrypoint expects, on the volume you mount at `/models`:
   ```bash
   tools/awq_snapshot.py snapshot /srv/models/qwen3.8-flash-next/AWQ-W4A16 \
       /srv/models/qwen3.8-flash-next/W4A16/ple_table_qwen4exp.pt /srv/models/qwen3.8-flash-next/AWQ-W4A16-snapshot
   ```
   Why, and how the AWQ serve config differs: [docs/weights.md](../docs/weights.md).
2. **16 slots, power-of-two graphs.** `serve-s16.args`: `--max-num-seqs 16`, capture sizes `[1,2,4,8,16,256,512,1024]`,
   `--gpu-memory-utilization 0.85`. Keep capture sizes to powers of two ([known issues](../docs/known-issues.md)).
3. **Sampling.** Server default temperature 0.7 (`--override-generation-config`), no server presence penalty
   (`B70_DEFAULT_PRESENCE_PENALTY=0`). A request's own values win.
4. **CPU KV tier.** `--kv-offloading-backend native`, 128 GiB = 32 GiB per rank. One pinned host allocation of ~31 GiB
   or more is refused by the driver; patch 0018 splits the pool into equal chunks, so no driver debug keys are
   needed. At 128 GiB we saw host MemAvailable drift down over hours of use (cause not identified), so on b70.1 we ran
   `KV_OFFLOADING_SIZE=64`. On b70.2, kernels b70.3 (or 0031) keep host memory flat under KV reload load (README, "The host-RAM KV tier"). Keep the size a power of two (total / 4).
5. **Offload fix.** The 0014jb arm: `B70_OFFLOAD_JUNCTION=1`, `B70_OFFLOAD_GDN_BACKSTEP=1` (docs/offload-fix.md).

## PLE table: RAM or NVMe

The INT8 PLE table is 48.9 GiB. Two ways to serve it, chosen with one switch:

| | `B70_PLE_INT8_NVME=1` (the AWQ and `nvme2-*` engines) | `B70_PLE_INT8_NVME` unset or `0` (the `r8g-*` engines) |
|---|---|---|
| host RAM for the table | 8 GiB pinned row cache in total (`B70_PLE_INT8_NVME_CACHE_GIB`) | 48.9 GiB pinned |
| cost | decode −2 … −4 %, prefill unchanged; reads the table file with `O_DIRECT` | none |
| needs | the table on a fast local NVMe (the native reader compiles with the image's `gcc`) | nothing extra |

Both need `B70_PLE_INT8=1` and `B70_PLE_INT8_PATH`. Use NVMe when the freed ~39 GiB buys you more CPU KV tier; keep it in
RAM when the host has the room. Details: [docs/ple-nvme.md](../docs/ple-nvme.md).

## Notes

1. **Chat template.** The examples use the checkpoint's own template. Our deployment uses a lightly modified one
   (thinking switches off when a client sends `enable_thinking=false` *or* a "none/off" effort); it is not included.
2. **Boot host RAM.** Pinned memory does not count against a container memory limit. Watch the host's MemAvailable
   during the first boot; with the BF16 table and without 0006 the peak is ~191 GiB.
3. **Liveness.** Probe with a 1-token chat completion; `/health` and `/v1/models` stay 200 while the engine is wedged.
   Allow ≥180 s under load: a busy engine queues, it is not down.
4. **compute-runtime.** Our numbers were measured on compute-runtime 26.35.39758.10; the base image carries 26.27.
   The image does not change the driver.
5. **GPU hangs.** If a card keeps timing out driver jobs after a GT reset, see
   [docs/known-issues.md](../docs/known-issues.md) (rebind the card, not an FLR).
