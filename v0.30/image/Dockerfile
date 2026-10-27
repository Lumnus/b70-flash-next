# b70-flash-next — vLLM v0.30.0 (XPU) for Qwen3.8-Flash-Next on 4x Intel Arc Pro B70.
#
# Build from the REPOSITORY ROOT (the patches live in ../patches):
#
#   docker buildx build -f image/Dockerfile -t b70-flash-next:0.30.0-b70.1 \
#     --build-arg GIT_SHA=0.30.0-b70.1 --build-arg SOURCE_SHA=$(git rev-parse HEAD) .
#
# = official vLLM v0.30.0 XPU image (pinned by digest)
# + wu1ff's Python changes re-derived as SOURCE patches (patches/vllm/0001–0005), applied with
#   `patch --fuzz=0` and then sha256-asserted byte-identical to wu1ff's files (verify-overlay.sh wu1ff)
# + the b70 series (patches/vllm/*-b70-*.patch: 0006–0010, 0012, 0013, 0013b, 0014a–f), every feature env-gated and
#   default OFF, applied after that gate and sha256-asserted per resulting file (verify-overlay.sh final)
# + two binaries with no public source (P6 libgdn_index64.so, P7 Level Zero peer-residency shim), copied from
#   wu1ff's public pack image BY DIGEST and sha256-asserted — never committed here
# + the pack's serve-config.json / prepare-serve.sh (image/files/opt/b70-flashnext/, MIT, wu1ff)
#
# Runtime contract is wu1ff's, unchanged: ENTRYPOINT /opt/b70-flashnext/prepare-serve.sh, WORKDIR /opt/venv,
# no ENV added, same paths. With no B70_* variable set the image behaves as wu1ff's pack image.
# Provenance, hashes and the per-patch table: PROVENANCE.md.
#
# Every image reference below is public (docker.io, ghcr.io). To pull through a private mirror, pass the SAME
# digest under the mirror's name: --build-arg BASE_IMAGE=<mirror>/vllm/vllm-openai-xpu@sha256:fc0e112a… and/or
# --build-arg WU1FF_IMAGE=<mirror>/wu1ff/qwen38-flashnext-b70@sha256:85512b52… — the digest is the identity,
# not the host. The base.digest label below assumes the default base.
#
# No `# syntax=` line on purpose: BuildKit's built-in frontend supports RUN --mount and COPY --chmod, and a
# syntax line would pull a frontend image on every build.

ARG BASE_IMAGE=docker.io/vllm/vllm-openai-xpu:v0.30.0@sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d
# Binary-only pieces come from this exact wu1ff pack image (1.0.0 == 1.0.1; 1.0.1 is pack-only).
# Its first 18 amd64 layers ARE the base's layers, so BuildKit fetches only its 26 small top layers extra.
ARG WU1FF_IMAGE=ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0@sha256:85512b52c09fa660a2e7fe441417129e7c47fac727fd85f66ccea6b65e0a9122

FROM ${WU1FF_IMAGE} AS wu1ff

FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG WU1FF_IMAGE
# GIT_SHA = the release version (e.g. 0.30.0-b70.1); SOURCE_SHA = the full commit of this repository.
ARG GIT_SHA=unset
ARG SOURCE_SHA=unset

# P1 P2 P3 P4 P5 P6(py) — source patches against v0.30.0 (ced6857a), applied to BOTH the installed package
# and the /workspace/vllm source checkout (P9: the checkout shadows site-packages for ad-hoc python run from
# /workspace/vllm; wu1ff mirrors every overlay there, so do we). --dry-run first so a failing patch never
# half-applies; --fuzz=0 so a moved base fails loudly instead of landing a patch in the wrong place.
# Patches are bind-mounted, not COPYed: they add no bytes to the image.
RUN --mount=type=bind,source=patches/vllm,target=/tmp/b70-patches \
    set -eu; export LC_ALL=C; \
    command -v patch >/dev/null; \
    for root in /opt/venv/lib/python3.12/site-packages /workspace/vllm; do \
      for p in /tmp/b70-patches/*.patch; do \
        case "$p" in *-b70-*) continue ;; esac; \
        patch -d "$root" -p1 --fuzz=0 --forward --no-backup-if-mismatch --dry-run -s < "$p" >/dev/null; \
        patch -d "$root" -p1 --fuzz=0 --forward --no-backup-if-mismatch -s < "$p"; \
        echo "applied $(basename "$p") -> $root"; \
      done; \
    done; \
    if find /opt/venv/lib/python3.12/site-packages/vllm /workspace/vllm/vllm \( -name '*.orig' -o -name '*.rej' \) | grep -q .; then \
      echo "patch left .orig/.rej files" >&2; exit 1; \
    fi; \
    mkdir -p /work

# P7 — Level Zero peer-residency shim (set LD_PRELOAD at run time). No public source. wu1ff layer
# sha256:512b136c0deacc7459daf8226f7ce8143a14a782070f4fa6312c0e7185e60401.
COPY --from=wu1ff --chmod=0644 /opt/b70-residency-shim/libl0_peer_residency_shim.so /opt/b70-residency-shim/libl0_peer_residency_shim.so
# P6 — prebuilt GDN op library with 64-bit conv-state offsets (a source form of the fix is in
# patches/vllm-xpu-kernels/, not used by this build). wu1ff layer
# sha256:426ebaaaf907281bbfaf34a3a54a7d69ed8a008e3810c1992f0b33b59536a92b.
COPY --from=wu1ff /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libgdn_index64.so /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libgdn_index64.so

# P8 — dense-QSA serve config (HF config.json @40b8f18d minus the 5 text_config.indexer_* keys) and the
# ENTRYPOINT that builds /work/flashnext-serve from symlinks + this config.
COPY --chmod=0644 image/files/opt/b70-flashnext/serve-config.json /opt/b70-flashnext/serve-config.json
COPY --chmod=0755 image/files/opt/b70-flashnext/prepare-serve.sh /opt/b70-flashnext/prepare-serve.sh

# Build-time gate 1 (wu1ff layer): every patched .py (both roots), both binaries and both pack files must be
# byte-identical to wu1ff's (same script an operator runs in the built container), and the private torch op
# library must load and register its schema (wu1ff's own build check). No GPU is needed for either.
# PYTHONDONTWRITEBYTECODE keeps this layer empty.
RUN --mount=type=bind,source=image/verify-overlay.sh,target=/tmp/verify-overlay.sh \
    set -eu; \
    bash /tmp/verify-overlay.sh wu1ff; \
    PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python -c 'import torch, vllm_xpu_kernels; from pathlib import Path; torch.ops.load_library(str(Path(vllm_xpu_kernels.__file__).with_name("libgdn_index64.so"))); print(torch.ops._gdn_index64.gdn_attention.default._schema)'

# b70 series (patches/vllm/*-b70-*.patch), applied only AFTER gate 1 has passed, so that gate keeps
# proving "wu1ff's overlay, unchanged". Filename order under LC_ALL=C: 0006 0007 0008 0009 0010 0012 0013 0013b
# 0014a..0014f. Every feature is default-OFF behind an env var (docs/switches.md), so the image behaves as
# wu1ff's unless the deployment opts in.
# Build-time gate 2 (verify-overlay.sh final): every file the series touches carries its recorded full-stack
# sha256 (B70-<last patch>), wu1ff files it does not touch are still wu1ff's, binaries unchanged. Then every
# touched .py must compile (in memory; no .pyc written).
RUN --mount=type=bind,source=patches/vllm,target=/tmp/b70-patches \
    --mount=type=bind,source=image/verify-overlay.sh,target=/tmp/verify-overlay.sh \
    set -eu; export LC_ALL=C; \
    for root in /opt/venv/lib/python3.12/site-packages /workspace/vllm; do \
      for p in /tmp/b70-patches/*-b70-*.patch; do \
        [ -e "$p" ] || continue; \
        patch -d "$root" -p1 --fuzz=0 --forward --no-backup-if-mismatch --dry-run -s < "$p" >/dev/null; \
        patch -d "$root" -p1 --fuzz=0 --forward --no-backup-if-mismatch -s < "$p"; \
        echo "applied $(basename "$p") -> $root"; \
      done; \
    done; \
    if find /opt/venv/lib/python3.12/site-packages/vllm /workspace/vllm/vllm \( -name '*.orig' -o -name '*.rej' \) | grep -q .; then \
      echo "patch left .orig/.rej files" >&2; exit 1; \
    fi; \
    bash /tmp/verify-overlay.sh final; \
    cd /opt/venv/lib/python3.12/site-packages; \
    PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python -c 'import sys; [compile(open(f, "rb").read(), f, "exec") for f in sys.argv[1:]]; print("compiled", len(sys.argv) - 1, "files")' \
      $(for p in /tmp/b70-patches/*-b70-*.patch; do sed -n 's#^+++ b/\(vllm/[^[:space:]]*\.py\).*#\1#p' "$p"; done | sort -u)

LABEL org.opencontainers.image.title="b70-flash-next" \
      org.opencontainers.image.description="vLLM v0.30.0 XPU + wu1ff Qwen3.8-Flash-Next B70 patch set (as source patches) + b70 series 0006-0014f (env-gated, default off) for 4x Arc Pro B70" \
      org.opencontainers.image.source="https://github.com/Lumnus/b70-flash-next" \
      org.opencontainers.image.revision="${SOURCE_SHA}" \
      org.opencontainers.image.version="${GIT_SHA}" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.base.digest="sha256:fc0e112afb64e3a06fe8daff34652435822a629412f38efce8f0f67a46636b8d" \
      net.lumnus.vllm.source="ced6857afa0ea7b2e3f0846a62e1394e90f15607" \
      net.lumnus.wu1ff.image="${WU1FF_IMAGE}"

WORKDIR /opt/venv
ENTRYPOINT ["/opt/b70-flashnext/prepare-serve.sh"]
