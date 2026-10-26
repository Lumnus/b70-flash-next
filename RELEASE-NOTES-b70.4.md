**0.32.0-b70.4**: one engine, one environment, two presets. Qwen3.8-Flash-Next on Intel Arc Pro B70, vLLM v0.32.0.

**Presets** (measured on 2026-10-26 with this engine and environment, Intel AutoRound, single runs, greedy):

| | `preset-4card` | `preset-2card` |
|---|---|---|
| cards | 4 (TP 4) | 2 (TP 2); the other pair stays free for any other model |
| attention, slots, context | sparse, 12, 262,144 | sparse, 4, 262,144 |
| expert offload | off | hot/cold, 145 of 256 experts per layer per rank in host RAM |
| KV tier (host RAM) | 128 GiB | 16 GiB |
| decode 1 / 4 / 8 streams (tok/s) | 156.4 / 287.4 / 580.9 | 58.2 / 69.4 / 109.2 |
| GPU KV cache (tokens) | 344,021 | 273,169 |
| host MemAvailable with it up (256 GB host) | 60.4 GiB | 154.9 GiB |

Attention (sparse default, dense), cards, drafter, expert offload, KV offload size, slots and GPU memory are parameters of the
same engine, each with its default and what it was measured at: `v0.32/README.md` §Parameters.

**Engine.** Fork tag `v0.32.0-b70.4`, 101 commits on vLLM v0.32.0, in three directories: `v0.32/patches/vllm/` (50, the dense
base; new 0050 backports vllm-project/vllm#54163, so DFlash keeps its trailing prefix-cache block), `v0.32/patches/vllm-qsa/`
(38, sparse attention; new: the KV split patches 0033 and 0036, the per-op timer 0034, the lifecycle driver 0035, C1 snapshots
in host RAM 0037, the DFlash draft page at TP2 0038) and `v0.32/patches/vllm-hotcold/` (13, new: Niluthon's hot/cold expert
offload from qwen38-flash-tp2 with his authorship, plus our cold-count and graph-capture fixes).

**Environment.** vllm-xpu-kernels 0.1.16.2 rebuilt from its source with our #660 and #662 (`v0.32/patches/vllm-xpu-kernels/`),
the attention libraries from the PyPI wheel, the astra Python modules (`v0.32/patches/vllm-xpu-kernels-py/`, now written
against 0.1.16.2), and our `libgdn_index64.so` for the 64-bit GDN offsets: #658 is merged upstream on main only, in no release.
torch 2.14.0+xpu, compute-runtime 26.35.39758.10, the oneCCL legacy plugin rebuilt with oneCCL #229.

**Sparse against dense**, identical engine code, measured on the 0.1.15.4 kernel build (the 0.1.16.2 environment passed the
gate with the token compare identical): agent-session replay 204.7 s against 230–235 s; one-stream decode 0.92×, 4 streams
level, 8 streams 0.93×, prefill up to 32K 0.89×; prefill 131K 1.06×, 260K 1.31×; long-context tasks equal up to 8K, 80/78 %
against 72/72 % at 128K. Details: `docs/measurements/v0.32-qsa.md` §0.

**Not measured:** a multi-day run; the agent replay and long prefill on the 0.1.16.2 environment; dense attention at 12 slots on
it; the stock oneCCL plugin with sparse attention; generation or reasoning quality of sparse against dense.

Build recipe: `v0.32/README.md` §Build and run. Changes in full: `CHANGELOG.md`.
