#!/usr/bin/env python3
"""minachist_repack.py — put Minachist's recalibrated routed experts into an Intel AutoRound serve snapshot.

Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound, branch `new-main` @ 411505aa, stores the routed experts as
compressed-tensors `pack-quantized` (int4, group 128, symmetric, F16 scales; stored nibble = q + 8) and quantizes
other layers to INT6/INT8, which the v0.30 XPU path does not load. Its experts use the same grid as Intel's, so only
the routed experts are taken: each `weight_packed` [out, in/8] / `weight_scale` [out, in/128] pair is repacked to
Intel's `auto_round:auto_gptq` layout (`qweight` [in/8, out] packed along the input dim in the same nibble order,
`scales` [in/128, out]) inside a copy of the Intel shard that holds it. Intel's `qzeros` are kept (uniform 0x77777777,
GPTQ v1, effective zero 8 = the compressed-tensors offset). Every other tensor (attention, GDN, shared expert, HC,
embed/head, MTP) stays Intel's, byte for byte; the serve config and the INT8 PLE table are unchanged.

    minachist_repack.py convert <intel snapshot> <minachist download> <new snapshot>
        <intel snapshot> is the directory tools/intel_snapshot.py made. Writes the expert shards (14 of them, each the
        same keys, dtypes, shapes and size as Intel's), symlinks every other entry of the Intel snapshot (absolute
        targets), and MANIFEST.json with the sha256 of each written shard. Resumable: shards already present are
        skipped. W_PAR=<n> sets the number of worker processes (default 3).
    minachist_repack.py verify <intel snapshot> <minachist download> <new snapshot> [layers] [expert step]
        dequantizes sampled experts both ways (our GPTQ unpack of the new snapshot; compressed-tensors'
        unpack_from_int32 of the download) and prints max |diff| (expected 0) and the rel-L2 distance between Intel's
        and Minachist's experts (we measured ~0.11 on layers 0, 1, 12, 24, 31, 47). Needs `compressed-tensors`.

The Minachist download does not need its BF16 n-gram table (the series serves the INT8 PLE table). Needs torch and
safetensors; CPU only.
"""
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

RX = re.compile(r"^(model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj)\.(qweight|qzeros|scales)$")
ARGS = {}


def _wrap32(x64):
    import torch
    x64 = x64 & 0xFFFFFFFF
    return torch.where(x64 >= 2**31, x64 - 2**32, x64).to(torch.int32)


def _ct_to_gptq(wp, ws, shape):
    import torch
    out_f, in_f = int(shape[0]), int(shape[1])
    assert wp.dtype == torch.int32 and tuple(wp.shape) == (out_f, in_f // 8), (wp.shape, shape)
    sh = torch.arange(8, dtype=torch.int32) * 4
    q = ((wp.unsqueeze(-1) >> sh) & 0xF).reshape(out_f, in_f)          # nibble i of word j = column 8j+i
    qt = q.t().contiguous().to(torch.int64).view(in_f // 8, 8, out_f)  # pack along the input dim, same nibble order
    qw = torch.zeros(in_f // 8, out_f, dtype=torch.int64)
    for i in range(8):
        qw |= qt[:, i, :] << (4 * i)
    return _wrap32(qw), ws.t().contiguous()


def _do_shard(fname):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    intel, mina, out_dir, wm = ARGS["intel"], ARGS["mina"], ARGS["out"], ARGS["wm"]
    handles = {}

    def mget(k):
        f = wm[k]
        if f not in handles:
            handles[f] = safe_open(os.path.join(mina, f), "pt")
        return handles[f].get_tensor(k)

    out, n = {}, 0
    with safe_open(os.path.join(intel, fname), "pt") as fi:
        meta, keys = fi.metadata(), list(fi.keys())
        for k in keys:
            m = RX.match(k)
            if not m:
                out[k] = fi.get_tensor(k)
                continue
            base, part = m.groups()
            if part == "scales":
                continue                                               # written together with qweight
            it = fi.get_tensor(k)
            if part == "qzeros":
                assert torch.unique(it).numel() == 1, k                # Intel's uniform zero point, kept
                out[k] = it
                continue
            qw, sc = _ct_to_gptq(mget(base + ".weight_packed"), mget(base + ".weight_scale"), mget(base + ".weight_shape"))
            isc = fi.get_slice(base + ".scales")
            assert qw.shape == it.shape and qw.dtype == it.dtype, (k, qw.shape, it.shape)
            assert list(sc.shape) == isc.get_shape() and sc.dtype == torch.float16 and isc.get_dtype() == "F16", k
            out[k], out[base + ".scales"] = qw, sc
            n += 1
    assert set(out) == set(keys)
    p = os.path.join(out_dir, fname)
    save_file(out, p + ".tmp", metadata=meta)
    os.rename(p + ".tmp", p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return fname, n, os.path.getsize(p), h.hexdigest()


def _init(intel, mina, out, wm):
    import torch
    torch.set_num_threads(4)
    ARGS.update(intel=intel, mina=mina, out=out, wm=wm)


def convert(intel, mina, out):
    intel, mina, out = (os.path.abspath(x) for x in (intel, mina, out))
    wi = json.load(open(os.path.join(intel, "model.safetensors.index.json")))["weight_map"]
    wm = json.load(open(os.path.join(mina, "model.safetensors.index.json")))["weight_map"]
    shards = sorted({f for k, f in wi.items() if RX.match(k)})
    os.makedirs(out, exist_ok=True)
    for name in sorted(os.listdir(intel)):                             # everything else: Intel's files
        if name in shards or name.endswith(".tmp"):
            continue
        dst = os.path.join(out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(os.path.join(intel, name)), dst)
    todo = [f for f in shards if not os.path.exists(os.path.join(out, f))]
    print(f"expert shards {len(shards)}, to write {len(todo)}", flush=True)
    res = []
    with ProcessPoolExecutor(int(os.environ.get("W_PAR", "3")), initializer=_init, initargs=(intel, mina, out, wm)) as ex:
        for r in ex.map(_do_shard, todo):
            print(r, flush=True)
            res.append(r)
    man = os.path.join(out, "MANIFEST.json")
    old = json.load(open(man)) if os.path.exists(man) else {}
    old.update({f: {"projections": n, "bytes": b, "sha256": h} for f, n, b, h in res})
    json.dump(old, open(man, "w"), indent=1, sort_keys=True)
    return 0


def verify(intel, mina, new, layers="0,1,12,24,31,47", step="16"):
    import torch
    from safetensors import safe_open
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    wi = json.load(open(os.path.join(intel, "model.safetensors.index.json")))["weight_map"]
    wm = json.load(open(os.path.join(mina, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(d, w, k):
        f = os.path.join(d, w[k])
        if f not in handles:
            handles[f] = safe_open(f, "pt")
        return handles[f].get_tensor(k)

    sh = torch.arange(8, dtype=torch.int32) * 4

    def gptq_deq(d, base, zoff):
        qw, qz, s = get(d, wi, base + ".qweight"), get(d, wi, base + ".qzeros"), get(d, wi, base + ".scales").float()
        q = ((qw.unsqueeze(1) >> sh.view(1, 8, 1)) & 0xF).reshape(-1, qw.shape[1]).float()
        z = ((qz.unsqueeze(-1) >> sh) & 0xF).reshape(qz.shape[0], -1).float() + zoff
        g = q.shape[0] // s.shape[0]
        return ((q - z.repeat_interleave(g, 0)) * s.repeat_interleave(g, 0)).t()

    def ct_deq(base):
        shape = get(mina, wm, base + ".weight_shape")
        q = unpack_from_int32(get(mina, wm, base + ".weight_packed"), 4, torch.Size(shape.tolist())).float()
        s = get(mina, wm, base + ".weight_scale").float()
        return q * s.repeat_interleave(q.shape[1] // s.shape[1], 1)

    zn = int(get(intel, wi, "model.language_model.layers.0.mlp.experts.0.up_proj.qzeros")[0, 0] & 0xF)
    zoff = 8 - zn
    print(f"Intel qzeros nibble {zn} -> zero offset +{zoff} (effective zero 8)")
    worst = 0.0
    for L in (int(x) for x in layers.split(",")):
        mx, num, den, n = 0.0, 0.0, 0.0, 0
        for e in range(0, 512, int(step)):
            for p in ("gate_proj", "up_proj", "down_proj"):
                b = f"model.language_model.layers.{L}.mlp.experts.{e}.{p}"
                r, c, i = ct_deq(b), gptq_deq(new, b, zoff), gptq_deq(intel, b, zoff)
                mx = max(mx, (c - r).abs().max().item())
                num += ((i - r) ** 2).sum().item()
                den += (r ** 2).sum().item()
            n += 1
        worst = max(worst, mx)
        print(f"layer {L:2d}: {n} experts x3  max|new - minachist| = {mx:.3g}  rel-L2(intel vs minachist) = {(num / den) ** 0.5:.4f}", flush=True)
    ks = [k for k in wi if "experts" not in k][::997][:40]
    same = all(torch.equal(get(new, wi, k), get(intel, wi, k)) for k in ks)
    print(f"non-expert sample identical to Intel: {same} ({len(ks)} tensors)")
    return 0 if worst == 0 and same else 1


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "convert":
        sys.exit(convert(*sys.argv[2:5]))
    if len(sys.argv) >= 5 and sys.argv[1] == "verify":
        sys.exit(verify(*sys.argv[2:7]))
    print(__doc__)
    sys.exit(2)
