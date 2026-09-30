# The PLE n-gram table in INT8 (patch 0008)

Qwen3.8-Flash-Next carries a per-layer-embedding (PLE) n-gram table: 320 M rows × 160 values, looked up 16 times per
token (orders 2 and 3 × 8 heads). In BF16 it is 95.4 GiB. wu1ff's patch 0002 keeps it in pinned host RAM, split per TP
rank into two slabs, and gathers the rows over UVA inside the captured graphs. 0006 fills those slabs without the
pageable intermediate copy; 0007 (FP8) and 0008 (INT8) shrink them.

## Format

`tools/build_int8_ple.py` streams the checkpoint's BF16 table (`ple_table_qwen4exp.pt`) and writes a `.safetensors`:

| tensor | dtype, shape | content |
|---|---|---|
| `table` | U8 `[320001536, 164]` | per row: 160 × int8 `q`, then the row's float32 scale `s` (little-endian). `s = absmax(row)/127`, `q = round_half_even(x/s)` clamped to ±127; `x ≈ q·s` |
| `ngram_heads_offsets`, `ngram_heads_vocab_sizes`, `layer_multipliers` | I64 | copied from the checkpoint; 0008 refuses a table whose hash layout differs from the model's |

The header is padded so the data starts at a 4 KiB boundary (0013 reads rows with `O_DIRECT` and relies on it).
`__metadata__.format` is `lumnus-ple-int8-rowscale/v1`; 0008 and 0013 check it by equality. The build is
deterministic: the same source bytes give the same file bytes.

```bash
BF16_PT=/models/qwen3.8-flash-next/W4A16/ple_table_qwen4exp.pt OUT_DIR=/models/qwen3.8-flash-next/int8-ple \
  nice -n 19 python3 tools/build_int8_ple.py build
BF16_PT=... OUT_DIR=... python3 tools/build_int8_ple.py verify
```

## Why INT8 per row

Measured on the full BF16 table, stratified sample of 262,041 rows:

| format | relative L2 error | pinned host RAM (4 ranks) |
|---|---|---|
| BF16 (as shipped) | 0 | 95.4 GiB |
| FP8 E4M3, one global scale (0007) | 2.65 % | 47.7 GiB |
| **INT8, one fp32 scale per row (0008)** | **0.66 %** | **48.9 GiB** |
| INT4, group 16/32 | 7–10 % | — |

On 4× B70 with INT8: no fault under full graph capture; MMLU 84.7 / TruthfulQA 88.5, not significantly different
from FP8 or BF16 (paired test); long-context needles at 18.7K/70K/98K pass; decode and prefill equal to BF16.
Against BF16 on the same prompts, the next-token KL divergence of INT8 sat 27 % above its own run-to-run noise floor,
with zero top-1 changes and zero answer changes (MMLU, TruthfulQA, needles, document QA). INT8 is the table we serve.

## Use

```
B70_PLE_DIRECT_PINNED=1
B70_PLE_INT8=1
B70_PLE_INT8_PATH=/models/qwen3.8-flash-next/int8-ple/ple_ngram_int8_rowscale.safetensors
PLE_TABLE_PATH=/models/qwen3.8-flash-next/W4A16/ple_table_qwen4exp.pt   # optional: boot cross-check
```

At boot 0008 refuses a wrong format tag, dtype, width or row count, a non-finite or negative scale in this rank's
rows, or (with `PLE_TABLE_PATH` set) a cross-check error above `B70_PLE_INT8_MAX_REL_ERR` (default 0.02; expected
~0.0066 on 4,096 random rows).
