#!/usr/bin/env python3
"""derive-serve-config.py — regenerate P8's files/opt/b70-flashnext/serve-config.json from the checkpoint.

P8 ("dense-QSA config surgery") is the only wu1ff change that is data, not code, so it is vendored as the file
itself. This script is its source form: it shows the file is exactly the HF config of
devan-carlin/Qwen3.8-Flash-Next-W4A16 @ 40b8f18df4d4a32cb6e687a51c78207e5e438522 (sha256 b9ef7d7d…) with the five
text_config.indexer_* keys removed, serialized as json.dumps(indent=2) + "\\n". The checkpoint ships no
indexer weights, and qwen4_exp/config.py turns QSA on whenever those keys are present.

    derive-serve-config.py <hf config.json> [--check files/opt/b70-flashnext/serve-config.json]

Without --check it prints the derived file to stdout. With --check it exits 1 unless the bytes match.
"""
import hashlib
import json
import sys

HF_CONFIG_SHA256 = "b9ef7d7d97a9c0bf046616470fc1db9e69d80c8b1998dc4a732ed6b0b4abf954"
SERVE_CONFIG_SHA256 = "91fa33ca705157739a56d2d56fd568fa20cf6b4e7928bcdc3410c8792408791f"
DROP = ("indexer_budget", "indexer_compress_ratio", "indexer_head_dim", "indexer_kv_heads", "indexer_n_heads")


def main() -> int:
    if len(sys.argv) not in (2, 4) or (len(sys.argv) == 4 and sys.argv[2] != "--check"):
        print(__doc__, file=sys.stderr)
        return 2
    raw = open(sys.argv[1], "rb").read()
    if hashlib.sha256(raw).hexdigest() != HF_CONFIG_SHA256:
        print("warning: input is not the HF config.json @40b8f18d this pack targets", file=sys.stderr)
    cfg = json.loads(raw)
    for key in DROP:
        cfg["text_config"].pop(key)  # KeyError on purpose: all five must be present
    out = (json.dumps(cfg, indent=2) + "\n").encode()
    if len(sys.argv) == 2:
        sys.stdout.buffer.write(out)
        return 0
    want = open(sys.argv[3], "rb").read()
    ok = out == want and hashlib.sha256(out).hexdigest() == SERVE_CONFIG_SHA256
    print(("OK" if ok else "MISMATCH") + f" derived sha256 {hashlib.sha256(out).hexdigest()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
