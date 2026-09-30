#!/usr/bin/env python3
"""Build the INT8 symmetric per-row PLE n-gram table for vLLM patch 0008 from the checkpoint's BF16 table.

Source: $BF16_PT, default /models/qwen3.8-flash-next/W4A16-devan-40b8f18d/ple_table_qwen4exp.pt
        (devan-carlin/Qwen3.8-Flash-Next-W4A16@40b8f18d, LFS sha256 1d12b395...b3b3; torch zip, one stored
        BF16 entry [320001536, 160], located from the zip header, cross-checked against torch.load(mmap)).
Format (chosen after comparing FP8 E4M3, INT8 per-row and INT4 group formats on this table; docs/ple-int8.md):
  safetensors; header padded with spaces so data starts at a 4096-byte boundary; tensors in this order:
    table                    U8  [320001536, 164]  row r at file offset data_start + 164*r:
                                                   bytes 0..159   = int8 q (two's complement)
                                                   bytes 160..163 = float32 little-endian s
                                                   s = absmax(row)/127 (fp32, from the exact BF16 values)
                                                   q = round_half_even(x / s) clamped to [-127, 127]  (0 when s == 0)
                                                   dequant: x ~= q * s  (fp32, then one rounding to bf16)
    ngram_heads_offsets      I64 [16]  \
    ngram_heads_vocab_sizes  I64 [16]   > copied from the W4A16 checkpoint (model-00001.safetensors)
    layer_multipliers        I64 [3]   /
  __metadata__ names the format, source file + sha256, and this recipe.
Deterministic: same source bytes -> same file bytes (sorted-key header, fixed padding, fixed rounding).

Modes:
  build   stream the 95 GiB table in chunks (bounded RAM), write OUT.part, sha256 while writing,
          fdatasync, rename, write OUT.ok ("<sha256>  <name>"). Refuses to overwrite an existing OUT.
  verify  independent checks on the written file: header/layout, the SAME stratified sample as the format study
          (4096 strata x 64 rows, seed 20260928 -> 262,041 rows): rel-L2 mean/p50/p99/max vs BF16,
          byte-exact re-quantisation of every sampled row, scale finite and >= 0; prints JSON.
Run heavy: nice -n 19 ionice -c3; page cache released with posix_fadvise(DONTNEED) behind the stream.
"""
import hashlib, json, os, sys, time, zipfile
import numpy as np
import torch

SRC = os.environ.get("BF16_PT", "/models/qwen3.8-flash-next/W4A16-devan-40b8f18d/ple_table_qwen4exp.pt")
SRC_SHA = "1d12b3952c2ed42e50d2b22325556352ac10b41e558c446b9a8e51879c92b3b3"
CKPT = os.path.dirname(SRC)
OUT_DIR = os.environ.get("OUT_DIR", "/models/qwen3.8-flash-next/INT8-ple-rowscale")
NAME = "ple_ngram_int8_rowscale.safetensors"
ROWS, D, W = 320_001_536, 160, 164
USED_ROWS = 320_001_446
CHUNK = int(os.environ.get("CHUNK_ROWS", str(1 << 20)))
FORMAT = "lumnus-ple-int8-rowscale/v1"
LAYOUT_KEYS = ["ngram_heads_offsets", "ngram_heads_vocab_sizes", "layer_multipliers"]
PFX = "model.language_model.layers.1.ple.ple_embedding."
torch.set_num_threads(int(os.environ.get("THREADS", "12")))
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:8.1f}s]", *a, flush=True)


def bf16_offset():
    zf = zipfile.ZipFile(SRC)
    ents = [i for i in zf.infolist() if "/data/" in i.filename]
    assert len(ents) == 1 and ents[0].compress_type == 0, ents
    info = ents[0]
    assert info.file_size == ROWS * D * 2, info.file_size
    with open(SRC, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
    assert h[:4] == b"PK\x03\x04"
    off = info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little")
    t = torch.load(SRC, mmap=True, weights_only=True)
    tab = t["table"] if isinstance(t, dict) else t
    assert tuple(tab.shape) == (ROWS, D) and tab.dtype == torch.bfloat16 and tab.is_contiguous()
    bf = np.memmap(SRC, dtype=np.uint16, mode="r", offset=off, shape=(ROWS, D))
    chk = np.array([0, 1, 2500011, 2500012, 160000374, USED_ROWS - 1, ROWS - 1], dtype=np.int64)
    assert torch.equal(torch.from_numpy(bf[chk].astype(np.int16)).view(torch.bfloat16), tab[torch.from_numpy(chk)])
    return off


def layout_tensors():
    from safetensors import safe_open
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for k in LAYOUT_KEYS:
        f = idx[PFX + k]
        with safe_open(os.path.join(CKPT, f), framework="pt") as s:
            v = s.get_tensor(PFX + k)
        assert v.dtype == torch.int64, (k, v.dtype)
        out[k] = v.contiguous()
    assert out["ngram_heads_offsets"].numel() == 16 and out["layer_multipliers"].numel() == 3
    return out


def quantize(x_bf16_u16: np.ndarray) -> torch.Tensor:
    """[n,160] uint16 (BF16 bits) -> [n,164] uint8 packed rows (reference: int_sym(x, 8, 160) of the format study, fp32 scale)."""
    x = torch.from_numpy(x_bf16_u16.view(np.int16)).view(torch.bfloat16).to(torch.float32)
    s = x.abs().amax(1, keepdim=True) / 127.0
    q = (x / s.clamp_min(1e-30)).round().clamp_(-127, 127).to(torch.int8)
    out = torch.empty(x.shape[0], W, dtype=torch.uint8)
    out[:, :D] = q.view(torch.uint8)
    out[:, D:] = s.contiguous().view(torch.uint8).reshape(-1, 4)  # host little-endian
    return out


def header_bytes(layout):
    meta = {
        "format": FORMAT,
        "row_layout": "160 x int8 q | float32 LE s; s=absmax/127, q=round_half_even(x/s) clamp +-127; x~=q*s",
        "source_file": "devan-carlin/Qwen3.8-Flash-Next-W4A16@40b8f18d/ple_table_qwen4exp.pt",
        "source_sha256": SRC_SHA,
        "recipe": "github.com/Lumnus/b70-flash-next tools/build_int8_ple.py",
        "padding_rows": f"[{USED_ROWS}, {ROWS}) quantised like any row (unreachable by the hash)",
    }
    hdr = {"__metadata__": meta, "table": {"dtype": "U8", "shape": [ROWS, W], "data_offsets": [0, ROWS * W]}}
    pos = ROWS * W
    for k in LAYOUT_KEYS:
        n = layout[k].numel() * 8
        hdr[k] = {"dtype": "I64", "shape": list(layout[k].shape), "data_offsets": [pos, pos + n]}
        pos += n
    js = json.dumps(hdr, sort_keys=True, separators=(",", ":")).encode()
    total = 8 + len(js)
    js += b" " * ((-total) % 4096)
    return len(js).to_bytes(8, "little") + js, pos


def build():
    out_path = os.path.join(OUT_DIR, NAME)
    if os.path.exists(out_path):
        sys.exit(f"refuse: {out_path} exists")
    os.makedirs(OUT_DIR, exist_ok=True)
    off = bf16_offset()
    layout = layout_tensors()
    log("bf16 data offset", off, "layout", {k: v.tolist() for k, v in layout.items()})
    hdr, data_len = header_bytes(layout)
    log("header", len(hdr), "B; data", data_len, "B")
    h = hashlib.sha256()
    part = out_path + ".part"
    fin = os.open(SRC, os.O_RDONLY)
    fout = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fout, hdr); h.update(hdr)
    written = len(hdr); synced = 0
    smin, smax, nzero = float("inf"), 0.0, 0
    for r0 in range(0, ROWS, CHUNK):
        n = min(CHUNK, ROWS - r0)
        a = off + r0 * D * 2
        buf = os.pread(fin, n * D * 2, a)
        assert len(buf) == n * D * 2
        os.posix_fadvise(fin, a, n * D * 2, os.POSIX_FADV_DONTNEED)
        packed = quantize(np.frombuffer(buf, dtype=np.uint16).reshape(n, D))
        sc = packed[:, D:].contiguous().view(torch.float32).reshape(-1)
        if not bool(torch.isfinite(sc).all()) or bool((sc < 0).any()):
            sys.exit(f"bad scale in rows [{r0}, {r0+n})")
        nzero += int((sc == 0).sum()); pos = sc[sc > 0]
        if pos.numel():
            smin = min(smin, float(pos.min())); smax = max(smax, float(pos.max()))
        b = packed.numpy().tobytes()
        mv = memoryview(b)
        while mv:
            k = os.write(fout, mv); mv = mv[k:]
        h.update(b); written += len(b)
        if written - synced > (4 << 30):
            os.fdatasync(fout); os.posix_fadvise(fout, 0, written, os.POSIX_FADV_DONTNEED); synced = written
        if (r0 // CHUNK) % 32 == 0:
            log(f"rows {r0+n}/{ROWS} ({100*(r0+n)/ROWS:.1f}%)")
    for k in LAYOUT_KEYS:
        b = layout[k].numpy().astype("<i8").tobytes()
        os.write(fout, b); h.update(b); written += len(b)
    os.fdatasync(fout); os.posix_fadvise(fout, 0, written, os.POSIX_FADV_DONTNEED)
    os.close(fout); os.close(fin)
    assert written == len(hdr) + data_len, (written, len(hdr), data_len)
    sha = h.hexdigest()
    os.rename(part, out_path)
    with open(out_path + ".ok", "w") as f:
        f.write(f"{sha}  {NAME}\n")
    log(f"DONE {out_path} {written} B sha256 {sha}; scale range (nonzero) [{smin:.6g}, {smax:.6g}], zero-scale rows {nzero}")


def verify(path):
    from safetensors import safe_open
    with open(path, "rb") as f:
        hl = int.from_bytes(f.read(8), "little"); hdr = json.loads(f.read(hl))
    ds = 8 + hl
    assert ds % 4096 == 0, ds
    assert hdr["__metadata__"]["format"] == FORMAT
    t = hdr["table"]; assert t["dtype"] == "U8" and t["shape"] == [ROWS, W] and t["data_offsets"] == [0, ROWS * W]
    ref = layout_tensors()
    with safe_open(path, framework="pt") as s:
        for k in LAYOUT_KEYS:
            assert torch.equal(s.get_tensor(k), ref[k]), k
    off = bf16_offset()
    bf = np.memmap(SRC, dtype=np.uint16, mode="r", offset=off, shape=(ROWS, D))
    q8 = np.memmap(path, dtype=np.uint8, mode="r", offset=ds, shape=(ROWS, W))
    rng = np.random.default_rng(20260928)          # the format study's sample, for a like-for-like number
    edges = np.linspace(0, USED_ROWS, 4096 + 1).astype(np.int64)
    idx = np.unique(np.concatenate([rng.integers(edges[i], edges[i + 1], 64) for i in range(4096)]))
    idx = np.concatenate([idx, np.array([0, ROWS - 1])])
    log("sample rows", idx.size)
    src = np.ascontiguousarray(bf[idx]); got = torch.from_numpy(np.ascontiguousarray(q8[idx]))
    requant = quantize(src)
    mism = int((requant != got).any(1).sum())
    X = torch.from_numpy(src.view(np.int16)).view(torch.bfloat16).float()
    q = got[:, :D].contiguous().view(torch.int8).float(); s = got[:, D:].contiguous().view(torch.float32)
    assert bool(torch.isfinite(s).all()) and bool((s >= 0).all())
    R = q * s
    nrm = X.norm(dim=1); nz = nrm > 0
    rel = (R - X).norm(dim=1)[nz] / nrm[nz]
    cos = torch.nn.functional.cosine_similarity(R[nz], X[nz], dim=1)
    # negative control: row r+1 decoded against row r must be far off
    neg = ((R[1:] - X[:-1]).norm(dim=1) / X[:-1].norm(dim=1).clamp_min(1e-30)).median()
    res = {"file": path, "data_start": ds, "rows_sampled": int(idx.size), "requant_byte_mismatch_rows": mism,
           "rel_l2_mean": float(rel.mean()), "rel_l2_p50": float(rel.median()),
           "rel_l2_p99": float(torch.quantile(rel.double(), 0.99)), "rel_l2_max": float(rel.max()),
           "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
           "max_abs_err": float((R - X).abs().max()), "neg_control_shifted_rel_median": float(neg)}
    print(json.dumps(res, indent=1), flush=True)
    ok = mism == 0 and res["rel_l2_mean"] < 0.01 and res["neg_control_shifted_rel_median"] > 0.5
    log("VERIFY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "build":
        build()
    elif mode == "verify":
        verify(sys.argv[2])
    else:
        sys.exit("usage: build_int8_ple.py build | verify <file>")
