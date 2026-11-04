0.32.0-b70.7 of b70-flash-next: vLLM 0.32 serving for Intel Arc Pro B70, Qwen3.8-Flash-Next on 4 and 2 cards, INT8 execution and INT8 KV, six `-7` presets.

## What's new against b70.6

1. **Han mask** (`B70_CJK_MASK`, on in every preset): stray Chinese characters in non-Chinese answers are removed at sampling time, speculative verify rows included. It switches itself off when a system or user turn contains Chinese; `cjk_mask: false` per request is the explicit escape ([model page](v0.32/models/qwen38-flash-next/meta.md#chinese-characters-on-non-english-prompts)).
2. **Prefill scratch off** (`B70_PTH_FA_PREFILL`, default 0): the bf16 scratch halved prefill at 16 streams and cost KV; prefill now runs the native INT8 attention path.
3. **16 slots** on every 4-card preset (4 on 2 cards).
4. **`-7` presets:** D8-7, D16e-7, S8-7 and S16e-7 on 4 cards, D8-7 and S8-7 on 2 cards.
5. **Qwen3.8-27B and Qwen3.6-35B-A3B 1-card presets** measured on these kernels.

Engine `v0.32.0-b70.7` (53 commits in `vllm-v6/`); kernels unchanged, `b70/v0.1.16-b70.6`.

| preset | cards | decode, 1 / 4 / 12 / 16 streams | prefill 32K / 128K | GPU KV (tokens) |
|---|---|---|---|---|
| D8-7 | 4 | 120 / 333 / 667 / 769 | 6,772 / 5,004 | 678,259 |
| D16e-7 | 4 | 112 / 299 / 625 / 682 | 6,366 / 4,726 | 606,145 |
| S8-7 | 4 | 115 / 320 / 650 / 741 | 6,000 / 5,583 | 631,240 |
| S16e-7 | 4 | 105 / 284 / 600 / 739 | 5,395 / 5,050 | 566,177 |
| D8-7 | 2 | 69 / 136 / — / — | 2,860 / 2,418 | 474,109 |
| S8-7 | 2 | 69 / 133 / — / — | 2,834 / 2,892 | 559,191 |

Fidelity and quality validation ran on the 4-card dense engines. Every preset passed the gates on this tree; S16e-7's limits are on the model page.

[Changelog](CHANGELOG.md), [provenance](PROVENANCE.md), [switches](docs/switches.md).

Added on main after this tag: D16-7 and S16-7, the BF16 KV reference presets.
