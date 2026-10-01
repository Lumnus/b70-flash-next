# PROVENANCE — b70-flash-next 0.30.0-b70.1

Where every piece of the image comes from, and the hashes that prove it. Grades: **[M]** measured by us ·
**[W]** third-party claim, not reproduced · **UNVERIFIED**. The source of the vLLM patches is the fork branch
`b70/v0.30.0` of github.com/Lumnus/vllm (one commit per patch on v0.30.0 `ced6857afa`); `patches/` is exported from
it by `scripts/export-series.sh` and checked by `scripts/check-series.sh`.

## 1. Bases

| Piece | Identity | Grade |
|---|---|---|
| Base image | `docker.io/vllm/vllm-openai-xpu:v0.30.0` — index `sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d`, linux/amd64 `sha256:e4446310b1d30015e8fdc1a0a2ef1669ac6bef857cbe772487571ed5c1a926a9` (resolved with `crane digest` 2026-09-28). 18 layers, 4.31 GB compressed, largest layer `sha256:faa835aeb4928055b3a157d69dd439d19733099d2d44014dbbaddbe8237cdc63` = 3,271,409,304 B. Ubuntu 24.04, Python 3.12, venv `/opt/venv`, source checkout `/workspace/vllm` | M |
| vLLM source | tag `v0.30.0` = `ced6857afa0ea7b2e3f0846a62e1394e90f15607`. The 7 files the patches touch are byte-identical in the tag, in the image's site-packages (layer `sha256:8439dd1b…`) and in `/workspace/vllm` (layer `sha256:8fcb7387…`) | M |
| torch | 2.13.0 (XPU index), per `requirements/xpu.txt` @ v0.30.0 | M |
| triton-xpu | 3.7.2+xpu | M |
| vllm-xpu-kernels | wheel 0.1.14.1; source tag v0.1.14 = `6d92b1bfbf32767ecda8e819613eb151e70030ad` | M |
| compute-runtime (NEO) | 26.27.39122.11; IGC 2.38.2 (Dockerfile.xpu history) | M |
| level-zero loader | v1.32.0 | M |
| oneCCL | 2022.0.0 (arrives as a torch-xpu wheel dependency; sits in the 3.27 GB layer, not exported) | W (wu1ff image-inspect) |
| wu1ff pack image | `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` — index `sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122`, linux/amd64 `sha256:761f215485ab9da27816366c65222465ba6dbf8acb2000641774ed698ce589f8`. Its first 18 layers are the base's, in order; 26 small layers on top. Pack release 1.0.1 is pack-only (same image). Repo `github.com/wu1ff/B70-LLM-Controller` @ `5ce77ce`, MIT | M |

## 2. What this image changes on the base, and where each piece comes from

wu1ff built its image with whole-file `COPY` overlays. Each `.py` overlay is re-derived here as a diff of
wu1ff's file against v0.30.0. Applying the diffs to the real base-image files gives wu1ff's files byte-for-byte
(§4). The b70 rows (L1…L8f) are separately numbered patches applied on top in filename order
(`LC_ALL=C`); their full-stack result per file is recorded in §5 and asserted by `verify-overlay.sh final`. Test
changes for 0013/0014 live in `patches/vllm-tests/` (never applied to the image; the fork branch carries them). Layer digests are the wu1ff amd64 manifest's (site-packages copy / `/workspace/vllm` copy).

| # | Vendored as | Files | wu1ff layer(s) | Result sha256 (= wu1ff) | Depends on |
|---|---|---|---|---|---|
| P1 XPU gate narrowing | `patches/vllm/0001-qwen4exp-xpu-gate.patch` (+7 −2) · sha256 `86f9cc47ea34616c33bb95f38b5259ea39d01f7babeee0b94adeb60547e18744` | `vllm/models/qwen4_exp/__init__.py` | `717f200c…` / `f0346b24…` | `0677522662c9921dad228f8bf1d37975c13d30439f8d08bdf1514f3e81e57de5` | — |
| P2+P3 PLE pinned-host 2-slab UVA + FULL-capture stream | `0002-qwen4exp-ple-pinned-host-uva-and-capture-stream.patch` (+248 −24) · `cdf9b11ddd10e4b4a4875a1920877313d51d2251464b82f514e8e09d9ba87b56` | `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` (wu1ff's 2nd revision, 45,872 B) | `57a3b9ae…` / `6592f887…` | `b714a0dc88c3ea8bef3d113f3dbd66a9087353af60fe3cf5836bb643a1a8fa24` | P1 |
| P4 HC down-GEMM K-split | `0003-qwen4exp-hc-down-gemm-ksplit.patch` (+110 −4) · `1b3586f6778219aedad183e2f92feb4688fc041b863c2d1e6140cc22a8a31334` | `…/nvidia/hyperconnection.py`, `vllm/model_executor/layers/linear.py` | `259ace7d…`/`bfbfca9b…`, `2ccd4ca6…`/`d3fd6efc…` | `15eaf5683f62dca162eb8a1825b632598f60b230ceaf56b8611a338cb1315663`, `6c8ea995afb343d50d7a395eb50da14b5e2e5c33c246f8b93d8cc69673e2422c` | P1 |
| P5 GDN spec-metadata fusion (MTP perf only) | `0004-gdn-spec-metadata-fusion.patch` (+379 −1, creates 2 files) · `90e75faad31ba00b504b9d0769d454463141460ffc7d998c2d0a691397c1b6f7` | `vllm/v1/attention/backends/gdn_attn.py`, `vllm/v1/worker/gpu/attn_utils.py`, NEW `…/gdn_spec_metadata.py`, NEW `…/gdn_spec_metadata_batched.py` | `cafd7e9a…`/`a3e4f179…`, `5ef924d7…`/`005cc245…`, `7d9bfcfd…`/`c7b76fa1…`, `254ffbe8…`/`10f053ec…` | `e8afd4989233d1f971397cff49a5c760b99729fe9af539cfc19786f6139a627c`, `d5ef338178e824f0d2fb9447cdebda1e8f4e7ec9cd7791994083007ff1ad77c8`, `97f7ae107e03a64ccab4d73ca9bc705141d7ddcd64223d61dfbfb1a98721279c`, `88f1c49b152c5468838c3b448126275c04c34afb50526ffbe954b61b5e11ee9f` | — |
| P6 python hook | `0005-xpu-ops-gdn-index64-hook.patch` (+5 −1) · `77c6f2714b152a307eccc7dc6b2bca3fcdd72000e2d0bc33685a82e4d2dbbd06` | `vllm/_xpu_ops.py` | `69b3dc93…` / `28c5b22d…` | `1b4195c77f09f75f5f725b2d19a7aa176c8b90d6d90ab2e68ec86c480c8ca5ed` | the P6 binary (import fails without it) |
| L1 PLE direct-to-pinned load (**b70 series, default OFF**: `B70_PLE_DIRECT_PINNED=1`) | `0006-b70-ple-direct-pinned-load.patch` (+74 −3) · `780d73ab5ad9b4c9541818515982986ada85a3d88c1f21ab21397cf4e1a7cb1f` | `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | — (not in wu1ff) | `d28c99b19e077d549e581015569cf604aadcac32f0cef30a2c9c3e2b7ae0f116` (after 0006 alone) | P2; applied in a separate Dockerfile step AFTER the wu1ff identity gate. Why: docs/measurements/ple.md |
| L2 PLE table in FP8 (**b70 series, default OFF**: `B70_PLE_FP8=1` + `B70_PLE_FP8_PATH=<fp8 .safetensors>`) | `0007-b70-ple-fp8-pinned-table.patch` (+415 −6, rev 2) · `db306b7198888cf4deb8993fccd3fc47b4554f612981ae9a33185589c5061fdb` | `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | — (not in wu1ff) | `5251494c662fbc1d403d1641be5f3b12783518b49bde5fa1bdaa009b4cdaee4e` after 0006+0007 | 0006; same separate Dockerfile step, file order |
| L3 PLE table in INT8, per-row fp32 scale (**b70 series, default OFF**: `B70_PLE_INT8=1` + `B70_PLE_INT8_PATH=<int8 .safetensors>`; exclusive with `B70_PLE_FP8`) | `0008-b70-ple-int8-rowscale-pinned-table.patch` (+290 −8) · `8ab4107ec317f571e3be3f19f86e2c551ff21da5937b671eaf50eda64867609f` | `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | — (not in wu1ff) | `8dafd95b5657d4a0e267b49be2496b492e0190c916f134ddcf9f63f506276b20` after 0006+0007+0008 | 0007; same Dockerfile step, file order. Table recipe `tools/build_int8_ple.py`; docs/ple-int8.md |
| L4 thinking budget per requested effort (**b70 series, default OFF**: `B70_THINKING_BUDGET=<effort=tokens,…>`, `B70_DEFAULT_PRESENCE_PENALTY=<float>`) | `0009-b70-thinking-budget-per-requested-effort.patch` · `a84edfb11e8c532c3c3ad4ce230d1595f452ee8287d5a9e667e6dedad2ac39a7` | `vllm/entrypoints/openai/chat_completion/protocol.py` | — | see §6 | ungated part: `reasoning_effort` also accepts `"ultra"` (API widening only) |
| L5 default repetition stop (**default OFF**: `B70_DEFAULT_REPETITION_DETECTION="max=1,min=1,count=128"`) | `0010-b70-default-repetition-detection.patch` · `28fe9d04885ec320ef83bdcfb5c0e762b1616f07c9746d87d51d0db7d57d9740` | `…/chat_completion/protocol.py` | — | see §6 | 0009 |
| (0011 offload copy bounds check) | `0011-…patch.draft` — **not applied** | — | — | — | draft; not published |
| L6 `reasoning.effort` alias, rev 2 (**default OFF**: `B70_REASONING_EFFORT_ALIAS=1`) | `0012-b70-reasoning-effort-alias.patch` · `400361f89a2d7d6adb152ef4f74d6c2c7a6c58959509a44fcbfbaaf8c802254c` | `…/chat_completion/protocol.py` | — | see §6 | 0010. Rev 2 = rev 1 + the env gate |
| L7 INT8 PLE table served from NVMe (**default OFF**: `B70_PLE_INT8_NVME=1`, requires `B70_PLE_INT8=1`) | `0013-b70-ple-int8-nvme-table.patch` · `fb736800c4fce930802ad5fdb7c2410a08f1d8de3ccbd3f8729483dfe0e9adaf` (`vllm/` only) | `…/nvidia/ngram_embedding.py`, NEW `…/nvidia/ple_nvme.py`, `…/nvidia/model_state.py`, `vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/kv_offload/cpu/gpu_worker.py` | — | see §6 | 0008. Ungated part: the KV-offload host-tensor allocation log line moves DEBUG→INFO and prints `is_pinned()`. docs/ple-nvme.md |
| L7b NVMe PLE native reader + prefill lookahead (**default = 0013 v1 behaviour**: `B70_PLE_INT8_NVME_READER=py`, `…_LOOKAHEAD=0`) | `0013b-b70-ple-int8-nvme-lookahead.patch` · `bbb664fc44e17d42f45fe7fd7f8c979db214ffa3e8e1c3d3dd43e16d0fb39c7a` | same 3 qwen4_exp files | — | see §6 | 0013. The native/uring readers are C embedded in `ple_nvme.py`, compiled with the image's gcc (build-essential in the base) at first use into `$TMPDIR/b70-ple-nvme` |
| L8a KV-offload trace (**default OFF**: `B70_OFFLOAD_TRACE=1|2`, `…_TRACE_PERIOD_S`) | `0014a-b70-offload-trace.patch` · `c948154ba2fcd77c15c8512151b41c50d0adc9df0f06eb245c5626db2e0bdf7d` | `…/kv_connector/v1/offloading/scheduler.py`, NEW `…/offloading/b70_offload.py` | — | see §6 | — (independent of 0006–0013) |
| L8b hybrid junction heal (**default OFF**: `B70_OFFLOAD_JUNCTION=1`) | `0014b-b70-offload-junction.patch` · `2528b29f63eb92259ad6e938eb9929a1fe5d1a2c04640c287ab34077b289a297` | `…/offloading/scheduler.py` | — | see §6 | 0014a |
| L8c GDN backstep (**default OFF**: `B70_OFFLOAD_GDN_BACKSTEP=N`, 0 = off) | `0014c-b70-offload-gdn-backstep.patch` · `f454fa708ab091f4a01dba38d378ab92dc8c2a3a8eae0589fd1a376097f29a86` | `vllm/v1/core/kv_cache_coordinator.py`, `vllm/v1/core/sched/scheduler.py` | — | see §6 | 0014a |
| L8d #56795 empty-advance guard (**default OFF**: `B70_OFFLOAD_EMPTY_ADVANCE_GUARD=1`) | `0014d-b70-offload-empty-advance-guard.patch` · `8e928f2f421ceaaf59a846bd4c4b1c023e652768e26e51e13ff85dcf27e9579a` | `…/offloading/scheduler.py` | — | see §6 | 0014a |
| L8e vllm#51787 backport (group-coherent eviction) — ungated by itself, **gated by 0014f** | `0014e-b70-offload-group-evict-51787.patch` · `fd2d56c5339b7573e7bb678c7e907741bfbb69c82682940b01e228bf26e623aa` (backport of upstream `f12fe10c5c`) | `…/offloading/scheduler.py`, `vllm/v1/kv_offload/{base.py,cpu/manager.py,cpu/policies/{base,lru,arc}.py,tiering/manager.py}` | — | see §6 | 0014d; never ship without 0014f |
| L8f offload gates (**default OFF**: `B70_OFFLOAD_GROUP_EVICT=1` selects the #51787 policies; cheap `TRACE=1`) | `0014f-b70-offload-gates.patch` · `37af05ea603ab0563a089634b6c38ee3366f3d90072abd36d2b2efb6e7634f42` | `…/offloading/{scheduler,b70_offload}.py`, `…/cpu/manager.py`, `…/policies/{lru,arc}.py` (restored to v0.30.0), NEW `…/policies/{lru,arc}_group_evict.py` | — | see §6 | 0014e. Ungated part: `ReqContext` key-position bookkeeping (a dict write per key, never read with the flag off) and never-called policy hooks |
| L9 chunked pinned CPU KV pool (always on; one tensor whenever it fits, otherwise equal power-of-two row-aligned chunks; a null pinned allocation raises instead of being filled) | `0018-b70-offload-chunked-pinned-pool.patch` · `f97fb491372d0f420ba1eb099f92e06830577245aae240c028f79c09a2d83a7c` (tests: `patches/vllm-tests/0018-b70-offload-chunked-pinned-pool-tests.patch` · `479eb266e59bc161605230fd253e58ecb9c93b9258e58ae471d2561f3dcac7dc`) | `vllm/v1/kv_offload/cpu/gpu_worker.py` | — | see §5 | 0013 (same file) |
| L10 dense-QSA: skip `*.self_attn.indexer.*` checkpoint tensors (always on; affects only configs without `indexer_n_heads` whose checkpoint still ships indexer tensors) | `0019-b70-qwen4exp-dense-qsa-skip-indexer.patch` · `0188886fb52d25c11084bda06715cc4b69f2ae3ba8bb23786a63be70de264fd5` | `vllm/models/qwen4_exp/nvidia/model.py` | — | see §5 | — |
| P6 binary | **COPY --from the wu1ff image by digest**; no public source | `site-packages/vllm_xpu_kernels/libgdn_index64.so` (4,380,128 B; ELF, not stripped; embeds `csrc/xpu/gdn_attn/…` source paths; GCC 13.3.0 Ubuntu) | `426ebaaaf907281bbfaf34a3a54a7d69ed8a008e3810c1992f0b33b59536a92b` | `0fc700d337b71dfd6f2d4d08ca8ec588a26fc1bd669ed9034c6e4e468a804b75` (matches wu1ff's stated hash) | — |
| P6 kernel source | `patches/vllm-xpu-kernels/0001-gdn-causal-conv1d-int64-state-offset.patch` (7 sites) · `d7b1e3f7868c489e1ff0a35094c2d66dfc05ca31c3cc601f1567ceb17fc2d612` | `csrc/xpu/gdn_attn/causal_conv1d.hpp` (3), `csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_tiled_xe2.hpp` (2), `…/chunk_causal_conv1d_xe2.hpp` (2) | — | not used by this image (torch 2.13) | **built** into vllm-xpu-kernels `0.1.15.4+b70.1` (fork branch `b70/v0.1.15` @ `69b823f9` = upstream `release/0.1.15.4` + this commit, clean cherry-pick); device code compared with the official wheel by disassembly (§6) [M] |
| P7 L0 peer-residency shim | **COPY --from the wu1ff image by digest**; no public source (not in wu1ff's public repo) | `/opt/b70-residency-shim/libl0_peer_residency_shim.so` (17,760 B; source name `l0_peer_residency_shim.c`; zelTracer-based) | `512b136c0deacc7459daf8226f7ce8143a14a782070f4fa6312c0e7185e60401` | `ae2c82f549d97393268cbd7b90dba7cf38fd00dbccae54f090b1a545e5613b3c` (matches wu1ff's stated hash) | — |
| P8 dense-QSA serve config | file `image/files/opt/b70-flashnext/serve-config.json` (wu1ff, MIT); source form `image/derive-serve-config.py` (HF config `b9ef7d7d…` @`40b8f18d` minus 5 `indexer_*` keys, `json.dumps(indent=2)+"\n"` — checked OK) | `/opt/b70-flashnext/serve-config.json` | `c6306db9…` | `91fa33ca705157739a56d2d56fd568fa20cf6b4e7928bcdc3410c8792408791f` | — |
| P8 entrypoint | file `image/files/opt/b70-flashnext/prepare-serve.sh` (wu1ff, MIT), mode 0755 | `/opt/b70-flashnext/prepare-serve.sh` | `bcee61bc…` (+ chmod `fc11d9c5…`) | `3f246a046c51cda8e7e582524bcf806d34b2cca3425736bd53286621a4a170e4` | P8 config |
| P9 WORKDIR shadowing | Dockerfile: every patch applied to both `/opt/venv/lib/python3.12/site-packages` and `/workspace/vllm`; `WORKDIR /opt/venv` | — | — | — | — |

Binary trust: the two `.so` files are exactly as trustworthy as wu1ff's image, no more. Pinning by digest and
asserting the hash guarantees they are wu1ff's bytes, not that wu1ff's bytes are what they claim. The shim runs
in-process via `LD_PRELOAD`. Replacing them needs (a) a from-source kernel build using the vendored P6 patch and
(b) the shim's source, which is not public.

## 3. Differences from the wu1ff image

Functional: **none with every `B70_*` variable unset**, apart from the small ungated parts named in
the L4, L7 and L8f rows (an extra accepted `reasoning_effort` value, one log line at INFO, unread bookkeeping). Same base layers, same bytes in every file wu1ff adds (§4), same ENTRYPOINT
(`/opt/b70-flashnext/prepare-serve.sh`), CMD (none), WORKDIR (`/opt/venv`), ENV (the base's; nothing added).

Non-functional:

1. Layer structure: 5 content layers + 1 empty verification layer instead of wu1ff's 26 (one layer holds all
   patched `.py` in both roots; the binaries and pack files are separate small layers).
2. wu1ff's first overlay layer shipped stale `__pycache__/*.cpython-314.pyc` files for `__init__.py` and the
   first `ngram_embedding.py` revision, and its import check left `cpython-312.pyc` files. Python 3.12 ignores
   the 3.14 files; the 3.12 files are regenerated at runtime. Not reproduced.
3. File mtimes differ (patch vs COPY).
4. OCI labels added: title, description, source, revision (build arg `SOURCE_SHA`), version (build arg `GIT_SHA`;
   this overrides the inherited `org.opencontainers.image.version=24.04` from ubuntu), licenses,
   base.name, base.digest, `net.lumnus.vllm.source`, `net.lumnus.wu1ff.image`.

## 4. Validation of the wu1ff layer (2026-09-28, CPU host; no image built)

Base files were taken from the real base image, not only from GitHub: layer `sha256:8439dd1b…` (site-packages,
63.6 MB, fetched with `crane blob`) and the `/workspace/vllm` files streamed out of layer `sha256:8fcb7387…`.

| Check | Result |
|---|---|
| GitHub raw @ `ced6857a` == image site-packages == image `/workspace/vllm`, for all 7 touched files | identical |
| each vllm patch, `patch -p1 --fuzz=0 --dry-run` then apply, on the site-packages root | 5/5 clean, no `.orig`/`.rej` |
| same on the `/workspace/vllm` root | 5/5 clean |
| patched result vs wu1ff overlay, 9 files × 2 roots | 18/18 byte-identical |
| wu1ff's `/workspace/vllm` copies == its site-packages copies | identical (all 9) |
| `git apply --check` of the 5-patch series on the pristine tag files | OK |
| kernel patch on v0.1.14 `6d92b1bf` (`--fuzz=0`), result == intended edit; `git apply -R --check` | OK |
| kernel patch on v0.1.15 and main `68d82174` | applies with `--fuzz=0` (the bug is still there upstream) |
| `derive-serve-config.py` on HF config @`40b8f18d` | OK, `91fa33ca…` |
| `verify-overlay.sh` on an assembled rootfs (patched files + wu1ff binaries + pack files) | OK, 22 files; negative control (1 byte appended) → exit 1 |
| `docker buildx build --check .` | no warnings; both digest-pinned refs resolve |

## 5. b70 series, release 0.30.0-b70.1

Every switch the series adds carries the `B70_` prefix (docs/switches.md); an earlier internal build used `LUMNUS_`
names, which are no longer read. The INT8 table format tag `lumnus-ple-int8-rowscale/v1` is kept on purpose: it is a
data contract written into existing table files and checked by equality.

Validation (CPU host, no image built): a pristine `git archive ced6857afa`, the 19 patches (21 since 0018/0019 were added; re-checked for the 21: clean apply, result == the fork branch's `vllm/`, `py_compile` OK; `verify-overlay.sh` on an assembled rootfs not re-run) applied in filename order
with `patch --fuzz=0 --forward --dry-run` then for real: all clean, no `.orig`/`.rej`; the result equals the fork
branch's `vllm/` and, with the two test mboxes, its `tests/`. `py_compile` of all touched `.py`: OK. The 18
pre-existing touched files are byte-identical to the tag in the base image's site-packages and `/workspace/vllm`
layers. `image/verify-overlay.sh` on an assembled rootfs (wu1ff binaries extracted from the pack image): `wu1ff` OK
after 0001–0005 (22 files), `final` OK after the series (56 checks), negative control (1 byte appended) → exit 1.
Offload CPU suites with every switch unset: failure set identical to unpatched v0.30.0 (34 GPU- or weights-bound
tests); the b70 suites pass.

Full-stack result per file (both roots), as asserted by `verify-overlay.sh final`:

| sha256 | last patch | file |
|---|---|---|
| `9b7c10e15bc81586c92ef655710711d77a552efe0d99c78a73e67389ca91982d` | 0013b | `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` |
| `4940c2db71e9686cd157c7e61b415a66935f0b0d120efa695466d83449c869cd` | 0013b | `vllm/models/qwen4_exp/nvidia/model_state.py` |
| `6e3b952277dfb4beefd34491dfe19664b0d030d1cc98016b63b28859dd4fb7db` | 0013b | `vllm/models/qwen4_exp/nvidia/ple_nvme.py` (new) |
| `03dbbf27e4701feaff9bdbcbc702303ce9b0e53198a0cb68b1509e497f56fc36` | 0013 | `vllm/v1/worker/gpu/model_runner.py` |
| `8e0690668b9fa6c3ff4a66405fdf87d0c4a5e670440f12ea33c64f4c4799c4e5` | 0018 | `vllm/v1/kv_offload/cpu/gpu_worker.py` |
| `ff413ff88dd8830b5e147983b997f956ebfd446ddf57fdf0d10168fdfd75a5db` | 0019 | `vllm/models/qwen4_exp/nvidia/model.py` (v0.30.0: `2d9d9805…6dfb`; not yet checked against the base image layers) |
| `bd5b01d0a8b7c9a1b0dd2ace780aaf969c7c6ad4c458ad9f28568c2e7cf9d7a4` | 0012 | `vllm/entrypoints/openai/chat_completion/protocol.py` |
| `0923b00c12102558fbf2b633a24d337486d69acc5decdbab6f1c6d89bb40d6fe` | 0014f | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/b70_offload.py` (new) |
| `1099e97d0b9b842dbe03a665f100f4543811983193b2b7be336aab36b60cd1f9` | 0014f | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` |
| `62e4a41a1b638ec2b4382d768c5c0759d4d0cabedcfa2b128942779482b35140` | 0014c | `vllm/v1/core/kv_cache_coordinator.py` |
| `4b952ce24768725dd4da9094a94b4cfd1bb444b89ecadc4fc63ecd4dc6b26d41` | 0014c | `vllm/v1/core/sched/scheduler.py` |
| `6b3af4ef35901b7608f16686573efa0e33a226de6e510dd79fac83cd133a9747` | 0014e | `vllm/v1/kv_offload/base.py` |
| `b24e3b8eb1ddf56327375e909c8afa2dda58e3b3e724030ccd570d0cc87a1983` | 0014e | `vllm/v1/kv_offload/cpu/policies/base.py` |
| `1bb9af3d334fc322911aaffa6e21d2f15c19ae1b2bb6a5be425e31143172bcc6` | 0014e | `vllm/v1/kv_offload/tiering/manager.py` |
| `8ae30d43a7b1e527e00901be2ceaaeaabded088aff89cd83afb983fb04af739d` | 0014f | `vllm/v1/kv_offload/cpu/manager.py` |
| `cb5ba467dd0cc0d020289ddccc8fcbf1d62d8c938fd2c35d903f86f8ce247147` | 0014f | `vllm/v1/kv_offload/cpu/policies/arc_group_evict.py` (new) |
| `77acbb46b6af9eb5a39ece824bc7514a06c890ca98fefd0c687137692c250e87` | 0014f | `vllm/v1/kv_offload/cpu/policies/lru_group_evict.py` (new) |
| `e4350b3ee0a706c31a30338065506460d83b421416c86155ecbc49f18c06172a` | UPSTREAM | `vllm/v1/kv_offload/cpu/policies/arc.py` (0014e changed, 0014f restored) |
| `b037992a080ddba36111e05466066e4ab347ec7e556cbe7e387cea2967703bab` | UPSTREAM | `vllm/v1/kv_offload/cpu/policies/lru.py` (same) |

The other 8 wu1ff files and the 4 binaries/pack files keep their §2 hashes. The two closed binaries are unchanged
(COPY by digest, same sha256).

## 6. Kernel builds and the AWQ recipe (added 2026-10-01)

**vllm-xpu-kernels `0.1.15.4+b70.1`** (edge line, torch 2.14; docs/kernels.md) [M]:

| | |
|---|---|
| source | github.com/Lumnus/vllm-xpu-kernels `b70/v0.1.15` @ `69b823f9038671c3af7cea7adf615dd269d620ad` = upstream `release/0.1.15.4` (`ddf336d`) + the int64 conv-state offset commit (7 sites) |
| build | upstream builder image `pytorch/manylinux2_28-builder:xpu-v2.14.0-rc10` (oneAPI 2026.1), `VLLM_XPU_ENABLE_XE3P=OFF`, `VLLM_VERSION_OVERRIDE=0.1.15.4+b70.1`, `setup.py bdist_wheel --py-limited-api=cp38` |
| wheel | `vllm_xpu_kernels-0.1.15.4+b70.1-cp38-abi3-manylinux_2_28_x86_64.whl`, 370,656,165 B, sha256 `18ecc832240911eb5d07c4b8fbd8572f51c02b6cb504fde5a640c04f5005f37b` (not published) |
| API | 111 of 111 op schemas identical to the official 0.1.15.4 wheel; no op registration dropped by the B70-only build |
| fix in the binary | BMG device code disassembled (`ocloc disasm -device bmg`): 36 of 40 `causal_conv1d` variants gain the 64-bit product (`mach`), the 4 unchanged have no conv-state access; `update_states_kernel` and `chunk_update_states_kernel` likewise |
| not shown | correctness on a GPU past block id 5,042 by a targeted test |

**vllm-xpu-kernels `0.1.14.1+b70.1`** (stable line, torch 2.13): upstream tag `0.1.14.1` = `6d92b1bfbf32767ecda8e819613eb151e70030ad`
+ the same commit, oneAPI 2026.0. In progress; not published.

**AWQ serving recipe** (engines/awq-s16-kv128-chunked.env, docs/weights.md) [M]:

| | |
|---|---|
| weights | `wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125b929543a783ce700c90e36dd1a575c00c` (19 files, 180.77 GB) |
| index | `tools/awq_snapshot.py snapshot`: 222,746 → 222,579 tensors (−128 PLE shard tensors, −39 `self_attn.indexer.*`); equal, key for key and file for file, to the index of our serving snapshot |
| serve config | `engines/serve-config-awq.json`, sha256 `c7a2b345927976d911cfd57d1083b71d1a75fee245f61a17b8f126b6717342c8` = `image/files/opt/b70-flashnext/serve-config.json` + `ignore` entries `re:^mtp.*`, `re:.*self_attn\..*` (reproduced by `tools/awq_snapshot.py serve-config`); byte-identical to the file we serve |
| engine | derived from our serving definition: TP4 + EP, 16 slots, capture sizes `[1,2,4,8,16,256,512,1024]`, `--gpu-memory-utilization 0.85`, 128 GiB native CPU KV tier, INT8 PLE on NVMe, 0014jb, temperature 0.7, presence 0. Differences: the checkpoint's chat template instead of our modified one; docker instead of our launcher. Not run in this form |
