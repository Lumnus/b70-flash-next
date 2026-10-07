#!/usr/bin/env python3
"""intel_snapshot.py — make a serve snapshot of Intel/Qwen3.8-Flash-Next-W4A16-AutoRound for this series.

The AutoRound checkpoint (revision 4c67bf68) needs the same treatment as the AWQ one (tools/awq_snapshot.py), adapted
to its layout (docs/weights.md):

  1. its PLE n-gram table (128 BF16 tensors `...ple.ple_embedding.ngram_embedding.shard_<i>.weight`) is the whole of
     shard model-00016-of-00017 (102.4 GB). The series serves the INT8 PLE table (B70_PLE_INT8, optionally from NVMe),
     so the filtered index drops those tensors and shard 16 is neither linked nor needed. The script refuses if shard
     16 holds anything else;
  2. it ships `self_attn.indexer.*` tensors that the dense-QSA serve config does not use (0019/0020 skip them too);
  3. its quantization block is auto-round (`packing_format auto_round:auto_gptq`): patch 0028 lets the PLE embedding
     accept it, and the routed experts load through vLLM's INC/GPTQ path onto the XPU WNA16 MoE backend.

    intel_snapshot.py snapshot <intel download dir> <devan ple_table_qwen4exp.pt> <new snapshot dir>
        symlink every file of the download into the new dir, except config.json (the entrypoint replaces it), the
        index (written filtered), shard 16 and *.bak; link ple_table_qwen4exp.pt to the given devan table (the boot
        cross-check reference that PLE_TABLE_PATH and the INT8 table are built from). Links are relative: keep the
        three paths on one volume. Shard 16 does not have to be downloaded at all.
    intel_snapshot.py serve-config <engines/serve-config-awq.json> <intel config.json> <out.json>
        the AWQ serve config (dense-QSA text_config) with quantization_config replaced by the checkpoint's auto-round
        block; refuses unless the two configs differ only by indexer_* keys and the quantization block. The output is
        byte-identical to engines/serve-config-intel-autoround.json (sha256 checked).

Nothing in the download is modified. Standard library only.
"""
import hashlib
import json
import os
import re
import sys

DROP = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$|\.self_attn\.indexer\.")
PLE_SHARD = "model-00016-of-00017.safetensors"
SKIP = {"config.json", "model.safetensors.index.json", "ple_table_qwen4exp.pt", PLE_SHARD}
INTEL_SERVE_CONFIG_SHA256 = "e6d529e6b791168a252978202596e794b101e1e679ad29494311fba90d5fbfb7"


def snapshot(src: str, ple: str, dst: str) -> int:
    src, ple = os.path.abspath(src), os.path.abspath(ple)
    if os.path.exists(dst):
        print(f"{dst} exists", file=sys.stderr)
        return 1
    if not os.path.isfile(ple):
        print(f"PLE table not found: {ple}", file=sys.stderr)
        return 1
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        index = json.load(f)
    wm = index["weight_map"]
    stray = [k for k, v in wm.items() if v == PLE_SHARD and not DROP.search(k)]
    if stray:
        print(f"{PLE_SHARD} holds non-PLE tensors, refusing: {stray[:5]}", file=sys.stderr)
        return 1
    dropped = sorted(k for k in wm if DROP.search(k))
    index["weight_map"] = {k: v for k, v in wm.items() if not DROP.search(k)}
    if PLE_SHARD in set(index["weight_map"].values()):
        print("filtered index still references the PLE shard", file=sys.stderr)
        return 1
    needed = sorted(set(index["weight_map"].values()))
    missing = [n for n in needed if not os.path.isfile(os.path.join(src, n))]
    if missing:
        print(f"shards referenced by the filtered index are missing: {missing}", file=sys.stderr)
        return 1
    os.makedirs(dst)
    for name in sorted(os.listdir(src)):
        if name in SKIP or name.startswith(".") or name.endswith(".bak"):
            continue
        os.symlink(os.path.relpath(os.path.join(src, name), dst), os.path.join(dst, name))
    os.symlink(os.path.relpath(ple, dst), os.path.join(dst, "ple_table_qwen4exp.pt"))
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    ple_n = sum(".ple." in k for k in dropped)
    print(f"{dst}: index {len(wm)} -> {len(index['weight_map'])} tensors "
          f"(dropped {ple_n} PLE shard tensors, {len(dropped) - ple_n} indexer tensors; {PLE_SHARD} not linked)")
    return 0


def serve_config(awq_cfg: str, intel_cfg: str, out: str) -> int:
    with open(awq_cfg) as f:
        a = json.load(f)
    with open(intel_cfg) as f:
        i = json.load(f)
    ia = {k: v for k, v in i["text_config"].items() if not k.startswith("indexer_")}
    rest_a = {k: v for k, v in a.items() if k not in ("text_config", "quantization_config")}
    rest_i = {k: v for k, v in i.items() if k not in ("text_config", "quantization_config")}
    if ia != a["text_config"] or rest_a != rest_i:
        print("configs differ beyond indexer_* + quantization_config; refusing", file=sys.stderr)
        return 1
    q = i["quantization_config"]
    if q.get("quant_method") != "auto-round" or q.get("packing_format") != "auto_round:auto_gptq":
        print(f"unexpected quantization_config {q.get('quant_method')}/{q.get('packing_format')}", file=sys.stderr)
        return 1
    a["quantization_config"] = q
    data = json.dumps(a, indent=1).encode()
    got = hashlib.sha256(data).hexdigest()
    with open(out, "wb") as f:
        f.write(data)
    note = "matches" if got == INTEL_SERVE_CONFIG_SHA256 else f"DIFFERS from {INTEL_SERVE_CONFIG_SHA256[:12]}…"
    print(f"{out}: dense-QSA text_config + auto-round quantization_config "
          f"({len(q.get('extra_config', {}))} extra_config entries); sha256 {got[:12]}… {note} the file we serve")
    return 0 if got == INTEL_SERVE_CONFIG_SHA256 else 1


def main() -> int:
    a = sys.argv[1:]
    if len(a) == 4 and a[0] == "snapshot":
        return snapshot(*a[1:])
    if len(a) == 4 and a[0] == "serve-config":
        return serve_config(*a[1:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
