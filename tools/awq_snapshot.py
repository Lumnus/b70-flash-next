#!/usr/bin/env python3
"""awq_snapshot.py — make a serve snapshot of wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 for this image.

The image's entrypoint (image/files/opt/b70-flashnext/prepare-serve.sh, wu1ff's) expects a snapshot directory with
`ple_table_qwen4exp.pt` and `model.safetensors.index.json`, and replaces `config.json` with its serve config. The AWQ
checkpoint differs from devan-carlin/Qwen3.8-Flash-Next-W4A16 in three ways that matter here (docs/weights.md):

  1. its PLE n-gram table is inside shard 1 (128 tensors `...ple.ple_embedding.ngram_embedding.shard_<i>.weight`);
     the series loads the PLE table from PLE_TABLE_PATH / the INT8 table instead, so the index must not list them;
  2. it ships `self_attn.indexer.*` tensors (12 layers + the MTP layer) that the dense-QSA config does not use
     (patch 0019 also skips them; the index drops them too, as in the snapshot we serve);
  3. its full-attention q/k/v/o projections stay BF16, so the serve config's `ignore` list needs `re:.*self_attn\\..*`
     (and `re:^mtp.*` for the separate model_mtp.safetensors).

    awq_snapshot.py snapshot <awq download dir> <devan ple_table_qwen4exp.pt> <new snapshot dir>
        symlink every file of the download into the new dir, except config.json (the entrypoint replaces it) and
        model.safetensors.index.json (written filtered); link ple_table_qwen4exp.pt to the given devan table.
        Links are relative: keep the three paths on one volume (e.g. under the directory mounted at /models).
    awq_snapshot.py serve-config <image serve-config.json> <out.json>
        write the AWQ serve config: the image's serve-config.json with the two ignore entries above added
        (byte-exact with the file we serve; sha256 checked).

Nothing in the download is modified. Standard library only.
"""
import hashlib
import json
import os
import re
import sys

DROP = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$|\.self_attn\.indexer\.")
AWQ_SERVE_CONFIG_SHA256 = "c7a2b345927976d911cfd57d1083b71d1a75fee245f61a17b8f126b6717342c8"


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
    dropped = sorted(k for k in wm if DROP.search(k))
    index["weight_map"] = {k: v for k, v in wm.items() if not DROP.search(k)}
    os.makedirs(dst)
    for name in sorted(os.listdir(src)):
        if name in ("config.json", "model.safetensors.index.json", "ple_table_qwen4exp.pt") or name.startswith("."):
            continue
        os.symlink(os.path.relpath(os.path.join(src, name), dst), os.path.join(dst, name))
    os.symlink(os.path.relpath(ple, dst), os.path.join(dst, "ple_table_qwen4exp.pt"))
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    ple_n = sum(".ple." in k for k in dropped)
    print(f"{dst}: index {len(wm)} -> {len(index['weight_map'])} tensors "
          f"(dropped {ple_n} PLE shards, {len(dropped) - ple_n} indexer tensors)")
    return 0


def serve_config(image_cfg: str, out: str) -> int:
    with open(image_cfg) as f:
        cfg = json.load(f)
    ig = cfg["quantization_config"]["ignore"]
    ig.insert(ig.index("re:.*mtp\\..*") + 1, "re:^mtp.*")
    ig.insert(ig.index("re:.*\\.linear_attn\\..*") + 1, "re:.*self_attn\\..*")
    data = json.dumps(cfg, indent=1).encode()
    got = hashlib.sha256(data).hexdigest()
    if got != AWQ_SERVE_CONFIG_SHA256:
        print(f"unexpected result sha256 {got} (is {image_cfg} the image's serve-config.json?)", file=sys.stderr)
        return 1
    with open(out, "wb") as f:
        f.write(data)
    print(f"{out}: sha256 {got}")
    return 0


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "snapshot":
        return snapshot(*sys.argv[2:])
    if len(sys.argv) == 4 and sys.argv[1] == "serve-config":
        return serve_config(*sys.argv[2:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
