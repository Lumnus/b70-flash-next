# Measurements

All numbers: Qwen3.8-Flash-Next, `devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d`, on one host with
4× Intel Arc Pro B70 (32 GB each), TP4 + expert parallel, vLLM v0.30.0 XPU with this series, compute-runtime
26.35.39758.10, `--max-num-batched-tokens 1024`, cudagraph capture sizes `[1,2,4,8,256,512,1024]`, measured
2026-09-28 … 2026-09-30. Speeds are single runs on an otherwise idle engine unless noted; differences under ~10 %
from one run are not significant. Quality comparisons are paired tests on the same items.

| file | what |
|---|---|
| [baseline.md](baseline.md) | the engine against llama.cpp on the same cards; batch size and PLE table choices |
| [kv-offload.md](kv-offload.md) | what the CPU KV tier buys under growing agent sessions; the offload fix |
| [ple.md](ple.md) | PLE table formats (BF16 / FP8 / INT8) and PLE on NVMe |
