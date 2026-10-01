#!/usr/bin/env bash
# run-example.sh <engine> [image] — start one example engine with docker.
#
#   engines/run-example.sh awq-s16-kv128-chunked b70-flash-next:0.30.0-b70.1
#
# Mounts: MODELS (default /models) read-only at /models, CACHE (default ./cache) at /cache.
# Per engine, <engine>.env may set (all optional; the defaults are the devan-carlin W4A16 examples):
#   MODEL_DIR     model directory inside the container   (default /models/qwen3.8-flash-next/W4A16 =
#                 devan-carlin/Qwen3.8-Flash-Next-W4A16 @ 40b8f18d, with ple_table_qwen4exp.pt)
#   SERVE_ARGS    `vllm serve` flag file in engines/      (default serve.args)
#   SERVE_CONFIG  a serve config in engines/ mounted over the image's /opt/b70-flashnext/serve-config.json
#                 (the AWQ checkpoint needs serve-config-awq.json; docs/weights.md)
#   KV_OFFLOADING_SIZE  CPU KV tier in GiB, total over the 4 ranks
# The INT8 table comes from tools/build_int8_ple.py. Edit the paths in *.env if yours differ.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
eng=${1:?engine name (awq-s16-kv128-chunked | r8g-kv64 | r8g-kv64-0014jb | nvme2-kv96-0014jb)}
img=${2:-b70-flash-next:0.30.0-b70.1}
MODELS=${MODELS:-/models}; CACHE=${CACHE:-$PWD/cache}
envf=$here/$eng.env
[ -f "$envf" ] || { echo "no $envf" >&2; exit 2; }
val() { sed -n "s/^$1=//p" "$envf" | tail -n 1; }
kv=$(val KV_OFFLOADING_SIZE)
model=$(val MODEL_DIR); model=${model:-/models/qwen3.8-flash-next/W4A16}
argsf=$here/$(val SERVE_ARGS); [ "$argsf" != "$here/" ] || argsf=$here/serve.args
cfg=()
sc=$(val SERVE_CONFIG)
[ -z "$sc" ] || cfg=(-v "$here/$sc:/opt/b70-flashnext/serve-config.json:ro")
mkdir -p "$CACHE"/{hf,tmp,triton,vllm,xdg,ple-nvme-native}
mapfile -t args < <(grep -v '^#' "$argsf" | sed '/^$/d' | sed 's/^\(--[^ ]*\) \(.*\)$/\1\n\2/')
exec docker run --rm --name "b70-$eng" \
  --device /dev/dri --group-add render --ipc host --network host --shm-size 16g \
  --ulimit memlock=-1 \
  --env-file "$here/common.env" --env-file "$envf" \
  -v "$MODELS:/models:ro" -v "$CACHE:/cache" ${cfg[@]+"${cfg[@]}"} \
  "$img" vllm serve "$model" "${args[@]}" --kv-offloading-size "$kv"
