# b70-flash-next

vLLM serving for **Intel Arc Pro B70** cards: Qwen3.8-Flash-Next on 4 and 2 cards, with dense or sparse attention, 8-bit or
BF16 execution and an 8-bit KV cache. Everything runs from one engine tree. Each mode is a set of environment switches, so
you can take the presets as they are or compose your own combination from the same tree.

## Presets

Flash-Next, Intel AutoRound W4A16 weights, DFlash2 speculative decoding, 262,144-token context. Every number here: thinking off,
temperature 0.7, total tok/s, best of two runs. 4 cards at 16 slots, 2 cards at 4 slots.

| preset | cards · slots | attention · execution · KV | decode, short prompt, 1 / 4 / 12 / 16 streams | decode, 2K prompt, 12 / 16 streams | prefill 32K / 128K | GPU KV (tokens) |
|---|---|---|---|---|---|---|
| **D8-7** | 4 · 16 | dense · INT8 · INT8 | 120 / 333 / 667 / 769 | 795 / **855** | **6,772** / 5,004 | 678,259 |
| **D16e-7** | 4 · 16 | dense · BF16 · INT8 | 112 / 299 / 625 / 682 | 747 / 770 | 6,366 / 4,726 | 606,145 |
| **D16-7** | 4 · 16 | dense · BF16 · BF16 | 114 / 306 / 628 / 698 | 748 / 775 | 6,358 / 4,718 | 386,060 |
| **S8-7** | 4 · 16 | sparse · INT8 · INT8 | 115 / 320 / 650 / 741 | 726 / 784 | 6,000 / **5,583** | 631,240 |
| **S16e-7** | 4 · 16 | sparse · BF16 · INT8 | 105 / 284 / 600 / 739 | 654 / 607 | 5,395 / 5,050 | 566,177 |
| **S16-7** | 4 · 16 | sparse · BF16 · BF16 | 101 / 292 / 610 / 676 | 701 / 651 | 5,664 / 5,051 | 340,469 |
| **D8-7** | 2 · 4 | dense · INT8 · INT8 | 69 / 136 / — / — | — | 2,860 / 2,418 | 474,109 |
| **S8-7** | 2 · 4 | sparse · INT8 · INT8 | 69 / 133 / — / — | — | 2,834 / 2,892 | 559,191 |

The names: D is dense and S sparse attention; 8 and 16 are the execution precision, INT8 or BF16; the `e` marks BF16 execution over an 8-bit KV cache. **D16-7 and S16-7 are the reference:** BF16 all the way, the straightforward choice if you don't want to experiment, with about 40 % less KV than their 8-bit-KV siblings.

Non-English prompts can draw stray Chinese characters from this model. Every preset masks them at serving time, unless a system or user turn contains Chinese ([details](v0.32/models/qwen38-flash-next/meta.md#chinese-characters-on-non-english-prompts)).

The 4-card presets keep the n-gram table in host RAM and use a 128 GiB KV tier. The 2-card presets read the table from NVMe and use a
32 GiB KV tier per engine, so two of them run side by side on four cards. Dense is faster on short prompts and sparse overtakes it as the prompt grows. Code-shaped output at temperature 0, and how
everything was measured: [model page](v0.32/models/qwen38-flash-next/meta.md).

## Run

Build the stack ([engine page](v0.32/README.md#build-and-run)), fetch the weights
([model page](v0.32/models/qwen38-flash-next/meta.md#weights)), then:

```bash
VENV=/opt/b70/venv v0.32/engine/serve.sh qwen38-flash-next preset-d8-7-4card
VENV=/opt/b70/venv v0.32/engine/serve.sh qwen38-flash-next preset-s8-7-2card
```

## More

1. [Engine page](v0.32/README.md): what the engine is, its patch series, build.
2. [Model page](v0.32/models/qwen38-flash-next/meta.md): presets, numbers, how to compose other modes.
3. [Qwen3.8-27B](v0.32/models/qwen38-27b/meta.md) and [Qwen3.6-35B-A3B](v0.32/models/qwen36-35b-a3b/meta.md) presets; the 1-card ones are measured on this release.
4. [Switches](docs/switches.md) · [lessons learned](docs/lessons-learned.md) · [measurements](docs/measurements/README.md) · [kernels](docs/kernels.md) · [weights](docs/weights.md).
5. [Changelog](CHANGELOG.md) · [provenance](PROVENANCE.md) · [upstream](docs/upstream.md) · [credits](docs/credits.md).
6. The engine code: [github.com/Lumnus/vllm](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.7), kernels:
   [github.com/Lumnus/vllm-xpu-kernels](https://github.com/Lumnus/vllm-xpu-kernels/tree/b70/v0.1.16-b70.6).

Built on [vLLM](https://github.com/vllm-project/vllm) and [wu1ff's B70 chain](https://github.com/wu1ff/B70-LLM-Controller), with
work from the people named in [credits](docs/credits.md).

## License

Apache-2.0 (LICENSE). wu1ff's files and binaries keep their MIT license; the vLLM and vllm-xpu-kernels patches modify Apache-2.0
code. Model weights are not included and keep their own licenses. See NOTICE.
