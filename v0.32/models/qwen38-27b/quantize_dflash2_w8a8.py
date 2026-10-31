"""RTN W8A8-int8 for the Qwen3.8-27B DFlash2 drafter with llm-compressor (data-free).

Scheme: compressed-tensors preset W8A8 = int8 symmetric per-channel weights (minmax, static) +
int8 symmetric per-token dynamic activations. Targets every Linear except
candidate_selector.hidden_projection (vLLM builds it with quant_config=None). Codebooks, norms,
conv base kernels stay bf16; the drafter has no embedding / lm_head (it shares the target's).

The drafter architecture (DFlash2DraftModel) has no transformers class, so the weights are put on
a structural holder (a PreTrainedModel whose nn.Linear modules carry the checkpoint names);
llm-compressor quantizes and saves it. Post-processing proves: every non-quantized tensor is
byte-identical to the source, and every int8 weight equals an independent RTN reference.
usage: quantize_dflash2_w8a8.py SRC_DIR OUT_DIR   (SRC_DIR = incoai/Qwen3.8-27B-DFlash2 @ 015e795; writes OUT_DIR and OUT_DIR.quant-report.json)

Tested with llm-compressor 0.15.0, compressed-tensors 0.20.1, torch 2.14.1 (CPU). Our output: model.safetensors
sha256 ff81cc3abb11647963aba4ff8a5ebfa23f806da2d1cf4c2a2464edcffdffd68c, config.json sha256
74076b26eaa959da756bea82a944a82af240150eee548eb23fd9536d9f0dcd37.
"""
import json, os, sys
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import PreTrainedModel, Qwen3Config
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

src, out = sys.argv[1], sys.argv[2]
tmp = out + ".lc-raw"
IGNORE = ["candidate_selector.hidden_projection"]
srct = load_file(os.path.join(src, "model.safetensors"))
cfg_src = json.load(open(os.path.join(src, "config.json")))


class Holder(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = ""

    def __init__(self, config, tensors):
        super().__init__(config)
        for name, t in tensors.items():
            parts = name.split(".")
            mod = self
            if parts[-1] == "weight" and t.dim() == 2 and "codebook" not in name:
                for p in parts[:-2]:
                    if not hasattr(mod, p):
                        mod.add_module(p, nn.Module())
                    mod = getattr(mod, p)
                lin = nn.Linear(t.shape[1], t.shape[0], bias=False, dtype=t.dtype, device="meta")
                lin.weight = nn.Parameter(t.clone(), requires_grad=False)
                mod.add_module(parts[-2], lin)
            else:
                for p in parts[:-1]:
                    if not hasattr(mod, p):
                        mod.add_module(p, nn.Module())
                    mod = getattr(mod, p)
                mod.register_parameter(parts[-1], nn.Parameter(t.clone(), requires_grad=False))

    def forward(self, *a, **k):
        raise NotImplementedError("structural holder")


config = Qwen3Config(**{k: v for k, v in cfg_src.items() if k not in ("architectures",)})
model = Holder(config, srct)
model.config._name_or_path = src  # llm-compressor validate_model re-reads the config from here
model.name_or_path = src
lin_names = sorted(n for n, m in model.named_modules() if isinstance(m, nn.Linear))
print("linears", len(lin_names))
oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme="W8A8", ignore=IGNORE))
model.save_pretrained(tmp, save_compressed=True)

lc = {}
for f in sorted(os.listdir(tmp)):
    if f.endswith(".safetensors"):
        lc.update(load_file(os.path.join(tmp, f)))
qcfg = json.load(open(os.path.join(tmp, "config.json")))["quantization_config"]

outt, report = {}, {"quantized": [], "kept": []}
for name, t in srct.items():
    base = name[: -len(".weight")] if name.endswith(".weight") else None
    if base and (base + ".weight_scale") in lc:
        q, s = lc[name], lc[base + ".weight_scale"]
        assert q.dtype == torch.int8 and q.shape == t.shape, (name, q.dtype, q.shape)
        w = t.float()
        ref_s = w.abs().amax(dim=1, keepdim=True) / 127.0
        ref_q = torch.clamp(torch.round(w / s.float()), -128, 127).to(torch.int8)
        zp = lc.get(base + ".weight_zero_point")
        assert zp is None or int(zp.abs().max()) == 0
        outt[name] = q.contiguous()
        outt[base + ".weight_scale"] = s.contiguous()
        report["quantized"].append(dict(
            name=name, shape=list(t.shape), scale_dtype=str(s.dtype), scale_shape=list(s.shape),
            q_vs_ref_mismatch=int((ref_q != q).sum()),
            scale_vs_absmax_maxrel=float(((s.float() - ref_s).abs() / ref_s.clamp_min(1e-12)).max()),
            weight_rel_l2_err=float((q.float() * s.float() - w).norm() / w.norm())))
    else:
        l = lc[name]
        assert l.dtype == t.dtype and torch.equal(l, t), name
        outt[name] = t
        report["kept"].append(name)
stray = set(lc) - set(srct) - {k for k in lc if k.endswith(("weight_scale", "weight_zero_point"))}
assert not stray, stray

os.makedirs(out, exist_ok=True)
save_file(outt, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
cfg = dict(cfg_src)
cfg["quantization_config"] = qcfg
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
json.dump(report, open(out.rstrip("/") + ".quant-report.json", "w"), indent=1)
q = report["quantized"]
print("quantized", len(q), "kept", len(report["kept"]),
      "| max q mismatch", max(r["q_vs_ref_mismatch"] for r in q),
      "| max scale rel", max(r["scale_vs_absmax_maxrel"] for r in q),
      "| mean w rel err", sum(r["weight_rel_l2_err"] for r in q) / len(q),
      "| scale dtypes", sorted({r["scale_dtype"] for r in q}))
print(json.dumps(qcfg, indent=1))
