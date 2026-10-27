#!/usr/bin/env bash
# Qwen3.8-Flash-Next B70 pack: deterministic serve-directory preparator.
# Container ENTRYPOINT. Receives the Controller-resolved command
#   vllm serve <readonly snapshot mount> [flags...]
# Rebuilds the dense-QSA serve directory in the writable container
# filesystem and execs the real server against it. The Controller-mounted
# snapshot is never written.
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$1" != "vllm" ] || [ "$2" != "serve" ]; then
  echo "b70-flashnext: expected 'vllm serve <model-path> [flags...]'" >&2
  exit 2
fi
snapshot="$3"
serve_dir="/work/flashnext-serve"

[ -f "$snapshot/ple_table_qwen4exp.pt" ] || { echo "b70-flashnext: PLE table missing under $snapshot" >&2; exit 3; }
[ -f "$snapshot/model.safetensors.index.json" ] || { echo "b70-flashnext: weight index missing under $snapshot" >&2; exit 3; }

mkdir -p "$serve_dir"
for entry in "$snapshot"/*; do
  name="$(basename "$entry")"
  [ "$name" = "config.json" ] && continue
  ln -sfn "$entry" "$serve_dir/$name"
done
cp /opt/b70-flashnext/serve-config.json "$serve_dir/config.json"

exec /opt/venv/bin/vllm serve "$serve_dir" "${@:4}"
