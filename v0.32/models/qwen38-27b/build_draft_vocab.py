"""Draft-vocab shortlist for the Qwen3.8-27B DFlash2 drafter (B70_DFLASH_DRAFT_VOCAB, patches/vllm-models 0006).

List = steveseguin/b70-optimization-lab @ b4f5f8c6, experiments/qwen38-27b-b70/docker/draft-shortlists/
shortlist-u-v1all-v2top65k.txt (The Unlicense, see draft-vocab.LICENSE) | every added/special token of the target
tokenizer, minus the DFlash2 mask token (dflash_config.mask_token_id, a draft input placeholder never worth proposing).
Optional probes (text files) only measure coverage; they do not change the list.

usage: build_draft_vocab.py TARGET_TOKENIZER_DIR SHORTLIST.txt DRAFTER_CONFIG.json OUT.txt [PROBE.txt ...]
The shipped draft-vocab-67263.txt is this script's output for Qwen3.8-27B (sha256 02ccd9cd...aa12e)."""
import collections, hashlib, json, sys
from transformers import AutoTokenizer
tok_dir, src, draft_cfg, out = sys.argv[1:5]
probes = sys.argv[5:]
tok = AutoTokenizer.from_pretrained(tok_dir)
mask_id = int(json.load(open(draft_cfg))["dflash_config"]["mask_token_id"])
base = {int(x) for x in open(src).read().split()}
added = set(tok.added_tokens_decoder)
ids = sorted((base | added) - {mask_id})
assert all(0 <= i < 248320 for i in ids)
open(out, "w").write("\n".join(map(str, ids)) + "\n")
h = hashlib.sha256(open(out, "rb").read()).hexdigest()
open(out + ".sha256", "w").write(f"{h}  {out.rsplit('/',1)[-1]}\n")
S = set(ids)
print(f"base {len(base)} | added {len(added)} (new {len(added - base)}) - mask {mask_id} -> {len(ids)} ids; sha256 {h}")
print(f"TP2 shard split (vocab 248320, shard 124160): low {sum(i < 124160 for i in ids)} high {sum(i >= 124160 for i in ids)}")
for p in probes:
    text = open(p, errors="ignore").read()
    c = collections.Counter()
    for i in range(0, len(text), 200000):
        c.update(tok(text[i:i+200000], add_special_tokens=False)["input_ids"])
    tot = sum(c.values()); cov = sum(v for t, v in c.items() if t in S)
    miss = [(tok.convert_ids_to_tokens(t), v) for t, v in c.most_common() if t not in S][:15]
    print(f"probe {p.rsplit('/',1)[-1]}: {tot} tokens, {len(c)} distinct, occurrence coverage {100*cov/tot:.2f}%, "
          f"distinct coverage {100*sum(t in S for t in c)/len(c):.2f}%; top misses {miss}")
