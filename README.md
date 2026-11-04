# b70-flash-next

vLLM serving for **Intel Arc Pro B70** cards: Qwen3.8-Flash-Next on 4 and 2 cards, with dense or sparse attention, 8-bit or
BF16 execution and an 8-bit KV cache. Everything runs from one engine tree. Each mode is a set of environment switches, so
you can take the presets as they are or compose your own combination from the same tree.

## Presets

Flash-Next, Intel AutoRound W4A16 weights, DFlash2 speculative decoding, 262,144-token context. Every number here: thinking off,
temperature 0.7, total tok/s, best of two runs. Measured at 16 slots on 4 cards, 4 slots on 2 cards.

| preset | cards · slots | attention · execution · KV | decode, short prompt, 1 / 4 / 12 / 16 streams | decode, 2K prompt, 12 / 16 streams | prefill 32K / 128K | GPU KV (tokens) |
|---|---|---|---|---|---|---|
| **D8-6** | 4 · 16 | dense · INT8 · INT8 | 124 / 330 / 678 / 777 | 778 / **880** | **6,671** / 5,016 | 645,113 |
| **D16e-6** | 4 · 16 | dense · BF16 · INT8 | 109 / 309 / 640 / 699 | 742 / 685 | 6,309 / 4,732 | 573,223 |
| **S8-6** | 4 · 16 | sparse · INT8 · INT8 | 117 / 309 / 644 / 738 | 757 / 715 | 5,947 / **5,586** | 600,045 |
| **S16e-6** | 4 · 16 | sparse · BF16 · INT8 | 102 / 275 / 606 / 801 | 678 / 607 | 5,308 / 5,057 | 535,094 |
| **D8-6** | 2 · 4 | dense · INT8 · INT8 | 72 / 132 / — / — | — | 2,865 / 2,412 | 426,060 |
| **S8-6** | 2 · 4 | sparse · INT8 · INT8 | 68 / 128 / — / — | — | 2,820 / 2,893 | 504,608 |

The names: D is dense and S sparse attention; 8 and 16 are the execution precision, INT8 or BF16; the `e` marks BF16 execution over an 8-bit KV cache. All these presets keep the KV cache in INT8.

The 8-bit presets bleed more Chinese tokens into answers to non-English prompts: on a Bulgarian-language replay, 80 samples each,
Chinese characters appeared in 21 % of D8-6 answers and 18 % of S8-6 answers, against 11 % on D16e-6.

The 4-card presets keep the n-gram table in host RAM and use a 128 GiB KV tier. The 2-card presets read the table from NVMe and use a
32 GiB KV tier per engine, so two of them run side by side on four cards. With prefill in the load, D8-6 keeps climbing to 16 streams and the others
peak at 12. Dense is faster on short prompts and sparse overtakes it as the prompt grows. Code-shaped output at temperature 0, and how
everything was measured: [model page](v0.32/models/qwen38-flash-next/meta.md).

## Run

Build the stack ([engine page](v0.32/README.md#build-and-run)), fetch the weights
([model page](v0.32/models/qwen38-flash-next/meta.md#weights)), then:

```bash
VENV=/opt/b70/venv v0.32/engine/serve.sh qwen38-flash-next preset-d8-6-4card
VENV=/opt/b70/venv v0.32/engine/serve.sh qwen38-flash-next preset-s8-6-2card
```

## More

1. [Engine page](v0.32/README.md): what the engine is, its patch series, build.
2. [Model page](v0.32/models/qwen38-flash-next/meta.md): presets, numbers, how to compose other modes.
3. [Qwen3.8-27B](v0.32/models/qwen38-27b/meta.md) and [Qwen3.6-35B-A3B](v0.32/models/qwen36-35b-a3b/meta.md) presets; the 1-card ones are booted on this release.
4. [Switches](docs/switches.md) · [lessons learned](docs/lessons-learned.md) · [measurements](docs/measurements/README.md) · [kernels](docs/kernels.md) · [weights](docs/weights.md).
5. [Changelog](CHANGELOG.md) · [provenance](PROVENANCE.md) · [upstream](docs/upstream.md) · [credits](docs/credits.md).
6. The engine code: [github.com/Lumnus/vllm](https://github.com/Lumnus/vllm/tree/v0.32.0-b70.6), kernels:
   [github.com/Lumnus/vllm-xpu-kernels](https://github.com/Lumnus/vllm-xpu-kernels/tree/b70/v0.1.16-b70.6).

Built on [vLLM](https://github.com/vllm-project/vllm) and [wu1ff's B70 chain](https://github.com/wu1ff/B70-LLM-Controller), with
work from the people named in [credits](docs/credits.md).

## License

Apache-2.0 (LICENSE). wu1ff's files and binaries keep their MIT license; the vLLM and vllm-xpu-kernels patches modify Apache-2.0
code. Model weights are not included and keep their own licenses. See NOTICE.
