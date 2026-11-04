# b70-flash-next

vLLM serving for **Intel Arc Pro B70** cards: Qwen3.8-Flash-Next on 4 and 2 cards, with dense or sparse attention, 8-bit or
BF16 execution and an 8-bit KV cache. Everything runs from one engine tree. Each mode is a set of environment switches, so
you can take the presets as they are or compose your own combination from the same tree.

## Presets

Flash-Next, Intel AutoRound W4A16 weights, DFlash2 speculative decoding, 262,144-token context. Decode is total tok/s, thinking off, prose at temperature 0.7, code at 0.

| preset | cards · slots | attention · execution | decode, 1 stream: prose / code | peak decode: prose / code (streams) | prefill 32K / 128K (tok/s) | GPU KV (tokens) |
|---|---|---|---|---|---|---|
| **D8-6** | 4 · 12 | dense · INT8 | 124 / 303 | 671 / 1,756 (12) | 6,758 / 4,972 | 672,772 |
| **D16e-6** | 4 · 16 | dense · BF16 | 109 / 258 | 699 / 1,662 (16) | 6,309 / 4,732 | 573,223 |
| **S8-6** | 4 · 12 | sparse · INT8 | 113 / 288 | 653 / 1,654 (12) | 5,944 / 5,586 | 616,088 |
| **S16e-6** | 4 · 16 | sparse · BF16 | 102 / 246 | 801 / 1,856 (16) | 5,308 / 5,057 | 535,094 |
| **D8-6** | 2 · 4 | dense · INT8 | 72 / 107 | 132 / 205 (4) | 2,865 / 2,412 | 426,060 |
| **S8-6** | 2 · 4 | sparse · INT8 | 68 / 92 | 128 / 195 (4) | 2,820 / 2,893 | 504,608 |

D8-6 also comes with 16 slots (`preset-d8-6s16-4card`), which carries more parallel clients for about 4 % less GPU KV. Peaks here are short prompts; [with a 2K prompt](v0.32/models/qwen38-flash-next/meta.md#numbers) D8-6 keeps climbing to 16 streams and the others peak at 12. The 4-card presets keep the n-gram table in host RAM and use a 128 GiB KV tier. The 2-card presets read the table from NVMe
and use a 32 GiB KV tier per engine, so two of them run side by side on four cards. Dense is faster on short prompts, and
sparse overtakes it as the prompt grows (128K row above). How each number was measured: [model page](v0.32/models/qwen38-flash-next/meta.md).

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
