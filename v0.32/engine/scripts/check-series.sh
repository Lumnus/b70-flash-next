#!/usr/bin/env bash
# check-series.sh — fail unless a v0.32 patch directory equals a fresh export of the fork.
#
#   v0.32/engine/scripts/check-series.sh [--git-dir DIR]            dense:    patches/vllm         = v0.32.0..v0.32.0-b70.4-dense
#   v0.32/engine/scripts/check-series.sh --qsa [--git-dir DIR]      sparse:   patches/vllm-qsa     = v0.32.0-b70.4-dense..v0.32.0-b70.4-qsa
#   v0.32/engine/scripts/check-series.sh --hotcold [--git-dir DIR]  hot/cold: patches/vllm-hotcold = v0.32.0-b70.4-qsa..v0.32.0-b70.4
#   v0.32/engine/scripts/check-series.sh --models [--git-dir DIR]   models:   patches/vllm-models  = v0.32.0-b70.4..v0.32.0-b70.5
#   v0.32/engine/scripts/check-series.sh --all [--git-dir DIR]      all four
#
#   --git-dir  a vLLM clone (default: ./vllm-src-032, cloned on demand from https://github.com/Lumnus/vllm.git)
#
# The engine is fork tag v0.32.0-b70.5 (109 commits on v0.32.0); the three b70.4 tags mark where the dense base, the
# sparse layer and the hot/cold layer end. The export is `git format-patch` of the range, one file per commit, abbrev 10,
# no signature.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
BASE=5df63a7203cdb9328d824a27b682e77c7b9928be
REPO_URL=https://github.com/Lumnus/vllm.git
DENSE=v0.32.0-b70.4-dense; QSA=v0.32.0-b70.4-qsa; HOTCOLD=v0.32.0-b70.4; ENGINE=v0.32.0-b70.5
gd="$PWD/vllm-src-032"; which=(dense)
while [ $# -gt 0 ]; do
  case "$1" in
    --git-dir) gd=$2; shift 2 ;;
    --qsa) which=(qsa); shift ;;
    --hotcold) which=(hotcold); shift ;;
    --models) which=(models); shift ;;
    --all) which=(dense qsa hotcold models); shift ;;
    -h|--help) sed -n 2,15p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ ! -e "$gd/.git" ] && [ ! -e "$gd/HEAD" ]; then
  git clone -q --filter=blob:none --no-checkout "$REPO_URL" "$gd"
fi
for t in "$DENSE" "$QSA" "$HOTCOLD" "$ENGINE"; do
  git -C "$gd" rev-parse -q --verify "refs/tags/$t^{commit}" >/dev/null || git -C "$gd" fetch -q origin "refs/tags/$t:refs/tags/$t"
done
check() {  # dir from to
  local dir=$1 from=$2 to=$3 tmp rc=0
  git -C "$gd" merge-base --is-ancestor "$from^{commit}" "$to^{commit}" || { echo "$to does not descend from $from" >&2; return 1; }
  tmp=$(mktemp -d)
  git -C "$gd" -c core.abbrev=10 format-patch -q --no-signature -o "$tmp" "$from..$to"
  if diff -r "$here/patches/$dir" "$tmp" >"$tmp.diff"; then
    echo "OK — v0.32/engine/patches/$dir equals the export of $from..$to ($(git -C "$gd" rev-parse "$to^{commit}"), $(ls "$tmp" | wc -l) commits)"
  else
    echo "MISMATCH in v0.32/engine/patches/$dir:" >&2; sed -n 1,40p "$tmp.diff" >&2; rc=1
  fi
  rm -rf "$tmp" "$tmp.diff"; return $rc
}
rc=0
for w in "${which[@]}"; do
  case $w in
    dense) check vllm "$BASE" "$DENSE" || rc=1 ;;
    qsa) check vllm-qsa "$DENSE" "$QSA" || rc=1 ;;
    hotcold) check vllm-hotcold "$QSA" "$HOTCOLD" || rc=1 ;;
    models) check vllm-models "$HOTCOLD" "$ENGINE" || rc=1 ;;
  esac
done
exit $rc
