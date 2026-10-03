# Weights: Intel AutoRound, AWQ or devan W4A16

Three 4-bit checkpoints of Qwen3.8-Flash-Next run on this series. Since 0.30.0-b70.2 we serve Intel's AutoRound build;
before it, the AWQ build (from 2026-09-30) and devan's W4A16 (from 2026-09-28). The README ranks them; the numbers are
in [measurements/b70.2.md](measurements/b70.2.md).

## Intel AutoRound (`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf68`)

| | |
|---|---|
| method | AutoRound tuning (signed-gradient optimisation of rounding and clipping, 200 iterations); int4 symmetric g128 |
| quantized | routed experts only; attention, GDN, shared experts, router, PLE, MTP, vision, embeddings and head stay BF16 |
| tensor format | `quant_method auto-round`, `packing_format auto_round:auto_gptq`: GPTQ-packed `qweight`, `qzeros` (all symmetric), **F16** `scales` |
| PLE n-gram table | inside shard 16, which holds nothing else (102.4 GB): not needed for serving |
| QSA indexer weights | shipped; unused on the dense-QSA config (0019/0020 skip them, the snapshot drops them) |
| MTP head | BF16, in `model_extra_tensors.safetensors` |
| license | qwen-community-1.0 (card) |
| what the series needs | patch 0028 (the PLE embedding accepts the INC config), a snapshot made with `tools/intel_snapshot.py`, `engines/serve-config-intel-autoround.json` |

**How it loads.** vLLM maps `auto-round` to its INC config. The 48 routed-expert layers go through the GPTQ MoE path to
the XPU WNA16 backend (`XPUExpertsWNA16`), the same int4 kernel and weight key as the AWQ build; the repack drops the
zero points. Everything else resolves to 16 bits through the checkpoint's `extra_config` patterns, the MTP experts
included. The registered but unused `qzeros` cost ~0.1–0.17 GiB of VRAM per card.

**Snapshot.** `tools/intel_snapshot.py snapshot <download> <devan ple_table_qwen4exp.pt> <snapshot>` symlinks the
download except `config.json`, the index and shard 16, writes a filtered index (224,280 → 224,113 tensors: −128 PLE
tensors, −39 indexer tensors) and links devan's PLE table, which `PLE_TABLE_PATH` uses for the INT8 table's boot
cross-check. `tools/intel_snapshot.py serve-config engines/serve-config-awq.json <download>/config.json out.json` writes
the serve config (the dense-QSA config with the checkpoint's auto-round block) and checks it byte for byte against the
file we serve.

**The PLE table.** We serve the same INT8 table as for the other two builds, built from devan's BF16 table. We have not
compared Intel's BF16 PLE rows in shard 16 with devan's table; the PLE is kept in BF16 by all three builds, and the
boot cross-check and every quality test pass, but the identity itself is unverified.

## AWQ and devan

| | `devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d` | `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125` |
|---|---|---|
| method | 4-bit, int4 symmetric g128 (most likely plain round-to-nearest) | AWQ (calibrated), smoothing folded in; int4 symmetric g128 |
| quantized | routed experts **and** the 12 full-attention layers' q/k/v/o | routed experts only; attention, GDN, shared experts, PLE, embeddings and head stay BF16 |
| tensor format | compressed-tensors `pack-quantized` | the same (no AWQ-specific kernel needed) |
| PLE n-gram table | separate `ple_table_qwen4exp.pt` | inside shard 1 (128 `…ngram_embedding.shard_<i>` tensors) |
| QSA indexer weights | none | shipped (12 layers + the MTP layer); unused on the dense-QSA config |
| MTP head | included | in a separate `model_mtp.safetensors` |
| what the image needs | nothing extra: wu1ff's entrypoint and `serve-config.json` | a snapshot made with `tools/awq_snapshot.py` and `engines/serve-config-awq.json` |

## What `tools/awq_snapshot.py` does

1. **Snapshot.** Symlinks the AWQ download into a new directory, writes a filtered `model.safetensors.index.json`
   without the 128 PLE shard tensors and the 39 `self_attn.indexer.*` tensors, and links `ple_table_qwen4exp.pt` to
   devan's table. The series loads the PLE table from `PLE_TABLE_PATH` / the INT8 table, not from the shards. The
   result equals the index we serve with (same 222,579 tensors). Patch 0019 would also skip the indexer tensors.
2. **Serve config.** `serve-config-awq.json` is the image's `serve-config.json` (devan's config minus the QSA indexer
   keys) with two `quantization_config.ignore` entries added:
   - `re:.*self_attn\..*`: the full-attention projections are BF16 in this checkpoint;
   - `re:^mtp.*`: the MTP tensors in `model_mtp.safetensors`.

   `tools/awq_snapshot.py serve-config image/files/opt/b70-flashnext/serve-config.json out.json` reproduces the file
   byte for byte (sha256 `c7a2b345…7342c8`). `engines/run-example.sh` mounts it over the image's serve config.

The INT8 PLE table is the same file for both checkpoints, built by `tools/build_int8_ple.py` from devan's
`ple_table_qwen4exp.pt`. Our deployment notes record the two checkpoints' BF16 PLE rows as identical; this repository
does not ship that check.

## The trade-off (our tests, 2026-09-30)

Same engine and sampling; only the weights differ. Paired tests on identical items.

| | AWQ | devan | significant? |
|---|---|---|---|
| MMLU 300, thinking off | 261 | 254 | no (McNemar p 0.12–0.21) |
| MMLU 300 / TruthfulQA 200, effort medium | 276 / 176 | 273 / 179 | no (p 0.55 each) |
| agentic coding, 10 tasks × 5, effort medium | **48/50** | 39/50 | **yes (Fisher p 0.015)** |
| agentic coding, effort none / high | 35 / 44 of 50 | 32 / 40 of 50 | no |
| 64K document questions, thinking off / medium | 12/15 / 15/15 | 7/15 / 15/15 | no (p 0.13) |
| needles at 32K / 64K | 8/8 | 8/8 | — |

1. **Code: better.** The gain sits in multi-step edits (a multi-file edit 1/5 → 5/5, a shell task 0/5 → 5/5). Hands-on
   use agreed: generated programs worked more often.
2. **Document QA (RAG-style): leans better**, not significant.
3. **Graphics: worse, by eye.** The visual design of generated games and pages looked better with devan's build, but
   its builds more often did not run. No instrument here measures design.
4. **Knowledge and truthfulness: the same.**
5. One behaviour change: with thinking off, AWQ tends to verify first (e.g. a shell call to check the date) before the
   requested tool call. Graders that require a specific first call count that as a failure.

Speed was not compared (the AWQ runs shared the engine with other traffic). The AWQ checkpoint keeps more weights in
BF16, so it uses somewhat more VRAM per card.
