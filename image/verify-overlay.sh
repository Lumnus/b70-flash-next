#!/usr/bin/env bash
# verify-overlay.sh — prove a vllm-xpu-flash-next container carries exactly the patch set its build declares.
#
# Two stages, two modes (one script):
#
#   wu1ff  — after patches 0001–0005 only. Every file wu1ff's pack image (ghcr.io/wu1ff/qwen38-flashnext-b70@sha256:
#            85512b52…) adds on top of official vllm/vllm-openai-xpu:v0.30.0 must be byte-identical to wu1ff's copy
#            (PY_FILES + ABS_FILES). Equality here = no functional difference from wu1ff. Hashes taken from wu1ff's
#            extracted layers on 2026-09-28 (per-file layer digests in PROVENANCE.md).
#   final  — (default) after the b70 series 0006–0014f. Every file the b70 series touches must carry the sha256
#            recorded for the full stack (B70_FILES; tag = last patch that touches it, UPSTREAM = touched then
#            restored to v0.30.0). Every wu1ff file the b70 series does not touch must still be wu1ff's. The two
#            closed binaries and the pack files must be unchanged. Final hashes were computed on 2026-09-30 (release 0.30.0-b70.1) by
#            applying the whole series in filename order (LC_ALL=C, patch --fuzz=0) to v0.30.0 (ced6857a), and are
#            valid for both roots: the 18 pre-existing touched files are byte-identical to the tag in the image's
#            site-packages layer (sha256:8439dd1b…) and /workspace/vllm layer (sha256:8fcb7387…), checked that day.
#
# The hashes are specific to the patch series in patches/vllm/ at the same commit. An image built from another
# series (e.g. an earlier release) fails `final` by design; use `wu1ff` to check only the wu1ff layer.
#
# Runs in three places:
#   1. the Dockerfile (wu1ff after 0001–0005; final after the b70 series — build fails on any mismatch)
#   2. a built container:  docker run --rm --entrypoint bash <image> -s [wu1ff|final] < verify-overlay.sh
#   3. a running pod:      kubectl exec -i <pod> -c <container> -- bash -s [wu1ff|final] < verify-overlay.sh
# Exit 0 = all as recorded; exit 1 = at least one missing or different file (each is printed).
# Optional VERIFY_ROOT=<dir> checks an extracted rootfs instead of / (used for a local dry-run).
set -u

MODE=${1:-final}
case "$MODE" in wu1ff|final) ;; *) echo "usage: verify-overlay.sh [wu1ff|final]" >&2; exit 2 ;; esac

R=${VERIFY_ROOT:-}
SITE=$R/opt/venv/lib/python3.12/site-packages
WS=$R/workspace/vllm

# wu1ff 1.0.0 overlay. sha256  relative-path-under-vllm/  (each checked in BOTH $SITE and $WS — P9)
PY_FILES='
0677522662c9921dad228f8bf1d37975c13d30439f8d08bdf1514f3e81e57de5  vllm/models/qwen4_exp/__init__.py
b714a0dc88c3ea8bef3d113f3dbd66a9087353af60fe3cf5836bb643a1a8fa24  vllm/models/qwen4_exp/nvidia/ngram_embedding.py
15eaf5683f62dca162eb8a1825b632598f60b230ceaf56b8611a338cb1315663  vllm/models/qwen4_exp/nvidia/hyperconnection.py
6c8ea995afb343d50d7a395eb50da14b5e2e5c33c246f8b93d8cc69673e2422c  vllm/model_executor/layers/linear.py
e8afd4989233d1f971397cff49a5c760b99729fe9af539cfc19786f6139a627c  vllm/v1/attention/backends/gdn_attn.py
97f7ae107e03a64ccab4d73ca9bc705141d7ddcd64223d61dfbfb1a98721279c  vllm/v1/attention/backends/gdn_spec_metadata.py
88f1c49b152c5468838c3b448126275c04c34afb50526ffbe954b61b5e11ee9f  vllm/v1/attention/backends/gdn_spec_metadata_batched.py
d5ef338178e824f0d2fb9447cdebda1e8f4e7ec9cd7791994083007ff1ad77c8  vllm/v1/worker/gpu/attn_utils.py
1b4195c77f09f75f5f725b2d19a7aa176c8b90d6d90ab2e68ec86c480c8ca5ed  vllm/_xpu_ops.py
'

# b70 series 0006, 0007, 0008, 0009, 0010, 0012, 0013, 0013b, 0014a–f, applied in that (filename) order on top
# of 0001–0005. sha256  tag  relative-path-under-vllm/   (both roots)
B70_FILES='
9b7c10e15bc81586c92ef655710711d77a552efe0d99c78a73e67389ca91982d  B70-0013b     vllm/models/qwen4_exp/nvidia/ngram_embedding.py
4940c2db71e9686cd157c7e61b415a66935f0b0d120efa695466d83449c869cd  B70-0013b     vllm/models/qwen4_exp/nvidia/model_state.py
6e3b952277dfb4beefd34491dfe19664b0d030d1cc98016b63b28859dd4fb7db  B70-0013b     vllm/models/qwen4_exp/nvidia/ple_nvme.py
03dbbf27e4701feaff9bdbcbc702303ce9b0e53198a0cb68b1509e497f56fc36  B70-0013      vllm/v1/worker/gpu/model_runner.py
75d77e9b1c9511b2445a7b48e15a838e2df5ca5c064b066fe8235c49649027e9  B70-0013      vllm/v1/kv_offload/cpu/gpu_worker.py
bd5b01d0a8b7c9a1b0dd2ace780aaf969c7c6ad4c458ad9f28568c2e7cf9d7a4  B70-0012      vllm/entrypoints/openai/chat_completion/protocol.py
0923b00c12102558fbf2b633a24d337486d69acc5decdbab6f1c6d89bb40d6fe  B70-0014f     vllm/distributed/kv_transfer/kv_connector/v1/offloading/b70_offload.py
1099e97d0b9b842dbe03a665f100f4543811983193b2b7be336aab36b60cd1f9  B70-0014f     vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
62e4a41a1b638ec2b4382d768c5c0759d4d0cabedcfa2b128942779482b35140  B70-0014c     vllm/v1/core/kv_cache_coordinator.py
4b952ce24768725dd4da9094a94b4cfd1bb444b89ecadc4fc63ecd4dc6b26d41  B70-0014c     vllm/v1/core/sched/scheduler.py
6b3af4ef35901b7608f16686573efa0e33a226de6e510dd79fac83cd133a9747  B70-0014e     vllm/v1/kv_offload/base.py
b24e3b8eb1ddf56327375e909c8afa2dda58e3b3e724030ccd570d0cc87a1983  B70-0014e     vllm/v1/kv_offload/cpu/policies/base.py
1bb9af3d334fc322911aaffa6e21d2f15c19ae1b2bb6a5be425e31143172bcc6  B70-0014e     vllm/v1/kv_offload/tiering/manager.py
8ae30d43a7b1e527e00901be2ceaaeaabded088aff89cd83afb983fb04af739d  B70-0014f     vllm/v1/kv_offload/cpu/manager.py
cb5ba467dd0cc0d020289ddccc8fcbf1d62d8c938fd2c35d903f86f8ce247147  B70-0014f     vllm/v1/kv_offload/cpu/policies/arc_group_evict.py
77acbb46b6af9eb5a39ece824bc7514a06c890ca98fefd0c687137692c250e87  B70-0014f     vllm/v1/kv_offload/cpu/policies/lru_group_evict.py
e4350b3ee0a706c31a30338065506460d83b421416c86155ecbc49f18c06172a  UPSTREAM      vllm/v1/kv_offload/cpu/policies/arc.py
b037992a080ddba36111e05466066e4ab347ec7e556cbe7e387cea2967703bab  UPSTREAM      vllm/v1/kv_offload/cpu/policies/lru.py
'

# sha256  absolute-path   (binaries with no public source + the pack's two files) — unchanged in both modes
ABS_FILES='
0fc700d337b71dfd6f2d4d08ca8ec588a26fc1bd669ed9034c6e4e468a804b75  /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libgdn_index64.so
ae2c82f549d97393268cbd7b90dba7cf38fd00dbccae54f090b1a545e5613b3c  /opt/b70-residency-shim/libl0_peer_residency_shim.so
91fa33ca705157739a56d2d56fd568fa20cf6b4e7928bcdc3410c8792408791f  /opt/b70-flashnext/serve-config.json
3f246a046c51cda8e7e582524bcf806d34b2cca3425736bd53286621a4a170e4  /opt/b70-flashnext/prepare-serve.sh
'

fail=0
n=0
nl=0
check() { # expected-sha label path
  local want=$1 label=$2 path=$3 got
  n=$((n + 1))
  if [ ! -f "$path" ]; then
    echo "MISSING   $path"; fail=1; return
  fi
  got=$(sha256sum "$path" | cut -d' ' -f1)
  if [ "$got" = "$want" ]; then
    printf '%-13s %s\n' "$label" "$path"
  else
    echo "DIFFERENT $path  want $want ($label)  got $got"; fail=1
  fi
}
in_b70() { # rel -> 0 if the b70 series touches it
  local sha tag rel
  while read -r sha tag rel; do
    [ -n "${sha:-}" ] || continue
    [ "$rel" = "$1" ] && return 0
  done <<< "$B70_FILES"
  return 1
}

while read -r sha rel; do
  [ -n "${sha:-}" ] || continue
  if [ "$MODE" = final ] && in_b70 "$rel"; then continue; fi
  check "$sha" IDENTICAL "$SITE/$rel"
  check "$sha" IDENTICAL "$WS/$rel"
done <<< "$PY_FILES"

if [ "$MODE" = final ]; then
  while read -r sha tag rel; do
    [ -n "${sha:-}" ] || continue
    check "$sha" "$tag" "$SITE/$rel"; nl=$((nl + 1))
    check "$sha" "$tag" "$WS/$rel"; nl=$((nl + 1))
  done <<< "$B70_FILES"
fi

while read -r sha path; do
  [ -n "${sha:-}" ] || continue
  check "$sha" IDENTICAL "$R$path"
done <<< "$ABS_FILES"

if [ -x "$R/opt/b70-flashnext/prepare-serve.sh" ]; then
  echo "EXEC          /opt/b70-flashnext/prepare-serve.sh"
else
  echo "NOT-EXEC      /opt/b70-flashnext/prepare-serve.sh"; fail=1
fi

if [ "$fail" = 0 ]; then
  echo "verify-overlay[$MODE]: OK — $n files checked: $((n - nl)) byte-identical to wu1ff 1.0.0, $nl at the recorded b70-series result"
else
  echo "verify-overlay[$MODE]: FAILED" >&2
fi
exit "$fail"
