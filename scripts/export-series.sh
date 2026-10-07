#!/usr/bin/env bash
# export-series.sh — write patches/vllm/ and patches/vllm-tests/ from the fork branch.
#
#   scripts/export-series.sh [--git-dir DIR] [--ref REF] [--out DIR]
#
#   --git-dir  a vLLM clone that has REF and ced6857afa (default: ./vllm-src, cloned on demand from
#              https://github.com/Lumnus/vllm.git)
#   --ref      branch, tag or commit (default: b70/v0.30.0-stable, the one release branch from 0.30.0-b70.2 on; a
#              remote-tracking name like origin/b70/v0.30.0-stable works)
#   --out      where to write (default: the repo's patches/)
#
# Every commit on BASE..REF must have a key in patches/series.txt; an unknown commit is an error, so the
# series cannot silently grow or shrink. Output is byte-reproducible for a given commit graph.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
BASE=ced6857afa0ea7b2e3f0846a62e1394e90f15607
REPO_URL=https://github.com/Lumnus/vllm.git
gd="$here/vllm-src"; ref=b70/v0.30.0-stable; out="$here/patches"
while [ $# -gt 0 ]; do
  case "$1" in
    --git-dir) gd=$2; shift 2 ;;
    --ref) ref=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    -h|--help) sed -n 2,14p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ ! -e "$gd/.git" ] && [ ! -e "$gd/HEAD" ]; then
  git init -q "$gd"
  git -C "$gd" remote add origin "$REPO_URL"
fi
if ! git -C "$gd" rev-parse -q --verify "$ref^{commit}" >/dev/null; then
  git -C "$gd" fetch -q --depth 64 origin "$BASE" "+refs/heads/$ref:refs/heads/$ref"
fi
git -C "$gd" rev-parse -q --verify "$BASE^{commit}" >/dev/null || git -C "$gd" fetch -q --depth 1 origin "$BASE"
git -C "$gd" merge-base --is-ancestor "$BASE" "$ref" || { echo "$ref does not descend from $BASE" >&2; exit 1; }

series="$here/patches/series.txt"
mkdir -p "$out/vllm" "$out/vllm-tests"
rm -f "$out"/vllm/*.patch "$out"/vllm-tests/*.patch
export LC_ALL=C
n=0
for c in $(git -C "$gd" rev-list --reverse "$BASE..$ref"); do
  subj=$(git -C "$gd" log -1 --format=%s "$c")
  key=${subj%% *}; key=${key%:}
  row=$(awk -v k="$key" '$0 !~ /^#/ && $1 == k { print $2, $3; exit }' "$series")
  [ -n "$row" ] || { echo "commit $c ($subj): key '$key' not in patches/series.txt" >&2; exit 1; }
  set -- $row
  vf=$1; tf=$2
  hv=$(git -C "$gd" diff --name-only "$c^" "$c" -- vllm/ | head -1)
  ht=$(git -C "$gd" diff --name-only "$c^" "$c" -- tests/ | head -1)
  if [ -n "$hv" ]; then
    [ "$vf" != "-" ] || { echo "commit $c touches vllm/ but series.txt names no vllm file" >&2; exit 1; }
    git -C "$gd" -c core.abbrev=7 format-patch -1 --stdout --no-signature "$c" -- vllm/ > "$out/vllm/$vf"
  fi
  if [ -n "$ht" ]; then
    [ "$tf" != "-" ] || { echo "commit $c touches tests/ but series.txt names no tests file" >&2; exit 1; }
    git -C "$gd" -c core.abbrev=7 format-patch -1 --stdout --no-signature "$c" -- tests/ >> "$out/vllm-tests/$tf"
  fi
  other=$(git -C "$gd" diff --name-only "$c^" "$c" -- . ':!vllm/' ':!tests/')
  [ -z "$other" ] || { echo "commit $c touches files outside vllm/ and tests/: $other" >&2; exit 1; }
  n=$((n + 1))
done
echo "exported $n commits of $ref ($(git -C "$gd" rev-parse "$ref")) to $out"
