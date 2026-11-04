#!/usr/bin/env bash
# serve.sh — start one preset of one model on the 0.32 engine.
#
#   v0.32/engine/serve.sh MODEL PRESET [extra vllm serve flags...]
#   v0.32/engine/serve.sh qwen38-flash-next preset-d8-7-4card
#   v0.32/engine/serve.sh qwen38-flash-next preset-s8-7-2card --port 8001
#
#   MODEL   a directory under v0.32/models/ (qwen38-flash-next, qwen38-27b, qwen36-35b-a3b)
#   PRESET  a preset of that model: the .args/.env pair of that name in the model's directory
#
# Environment:
#   VENV        the Python environment of "Build and run" steps 2-5 (default /opt/b70/venv)
#   MODEL_PATH  the weights (default: the model's path in its meta.md, under /models)
#   SERVE_DIR   Flash-Next only: where the snapshot is linked with the preset's config.json (default /cache/serve/<config name>)
#   DRY=1       print the command and exit
#
# The preset .env is sourced with `set -a` (no quotes, no expansion: docker --env-file format), the .args file gives one
# `--flag value` per line (comments and blank lines skipped), extra arguments are appended and win. The command runs from
# the kit root, so the chat-template paths in the .args files resolve. The venv's own oneAPI runtime comes first on
# LD_LIBRARY_PATH (a mixed runtime segfaults at exit).
set -euo pipefail
kit=$(cd "$(dirname "$0")/../.." && pwd)
[ $# -ge 2 ] || { sed -n 2,20p "$0"; exit 2; }
model=$1 preset=$2; shift 2
dir="$kit/v0.32/models/$model"
[ -f "$dir/$preset.args" ] && [ -f "$dir/$preset.env" ] || { echo "no preset $preset for $model (looked in $dir)" >&2; exit 2; }
case "$model" in
  qwen38-flash-next) def=/models/qwen3.8-flash-next/snapshot ;;
  qwen38-27b) def=/models/qwen3.8-27b/int4-autoround ;;
  qwen36-35b-a3b) def=/models/qwen3.6-35b-a3b/int4-mixed-autoround ;;
  *) def="" ;;
esac
path=${MODEL_PATH:-$def}
[ -n "$path" ] || { echo "set MODEL_PATH for $model" >&2; exit 2; }
venv=${VENV:-/opt/b70/venv}
args=()
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue ;; esac
  if [[ "$line" == --*' '* ]]; then args+=("${line%% *}" "${line#* }"); else args+=("$line"); fi
done < "$dir/$preset.args"
set -a; . "$dir/$preset.env"; set +a
export LD_LIBRARY_PATH="$venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$kit"
if [ -n "${SERVE_CONFIG:-}" ]; then  # the preset names the config.json: link the snapshot's files and copy the config over
  sd=${SERVE_DIR:-/cache/serve/${SERVE_CONFIG%.json}}
  if [ "${DRY:-0}" != 1 ]; then
    mkdir -p "$sd"; for f in "$path"/*; do [ "$(basename "$f")" = config.json ] || ln -sfn "$f" "$sd/$(basename "$f")"; done
    cp "$dir/$SERVE_CONFIG" "$sd/config.json"
  fi
  path=$sd
fi
[ "${DRY:-0}" = 1 ] || mkdir -p "${TMPDIR:-/tmp}" "${HF_HOME:-/tmp}" "${TRITON_CACHE_DIR:-/tmp}" "${VLLM_CACHE_ROOT:-/tmp}" "${XDG_CACHE_HOME:-/tmp}"
cmd=("$venv/bin/vllm" serve "$path" "${args[@]}" "$@")
if [ "${DRY:-0}" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; exit 0; fi
exec "${cmd[@]}"
