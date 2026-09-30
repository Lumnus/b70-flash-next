#!/usr/bin/env bash
# run-example.sh <engine> [image] — start one example engine with docker.
#
#   engines/run-example.sh r8g-kv64-0014jb b70-flash-next:0.30.0-b70.1
#
# Mounts: MODELS (default /models) read-only at /models, CACHE (default ./cache) at /cache. The model directory is
# /models/qwen3.8-flash-next/W4A16 (devan-carlin/Qwen3.8-Flash-Next-W4A16 @ 40b8f18d, with ple_table_qwen4exp.pt);
# the INT8 table comes from tools/build_int8_ple.py. Edit the paths in *.env if yours differ.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
eng=${1:?engine name (r8g-kv64 | r8g-kv64-0014jb | nvme2-kv96-0014jb)}
img=${2:-b70-flash-next:0.30.0-b70.1}
MODELS=${MODELS:-/models}; CACHE=${CACHE:-$PWD/cache}
mkdir -p "$CACHE"/{hf,tmp,triton,vllm,xdg,ple-nvme-native}
kv=$(sed -n 's/^KV_OFFLOADING_SIZE=//p' "$here/$eng.env")
mapfile -t args < <(grep -v '^#' "$here/serve.args" | sed '/^$/d' | sed 's/^\(--[^ ]*\) \(.*\)$/\1\n\2/')
exec docker run --rm --name "b70-$eng" \
  --device /dev/dri --group-add render --ipc host --network host --shm-size 16g \
  --ulimit memlock=-1 \
  --env-file "$here/common.env" --env-file "$here/$eng.env" \
  -v "$MODELS:/models:ro" -v "$CACHE:/cache" \
  "$img" vllm serve /models/qwen3.8-flash-next/W4A16 "${args[@]}" --kv-offloading-size "$kv"
