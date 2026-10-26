#!/usr/bin/env bash
# check-series.sh — fail unless v0.32/patches/vllm/ equals a fresh export of the fork tag.
#
#   v0.32/scripts/check-series.sh [--git-dir DIR] [--ref REF]
#   v0.32/scripts/check-series.sh --qsa [--git-dir DIR] [--qsa-ref REF]
#
#   --git-dir  a vLLM clone (default: ./vllm-src-032, cloned on demand from https://github.com/Lumnus/vllm.git)
#   --ref      default v0.32.0-b70.4-dense (tag: the dense series, 50 commits; v0.32.0-b70.1 is the same series without the last one)
#
# The export is `git format-patch` of v0.32.0..REF, one file per commit, abbrev 10, no signature.
#
#   --qsa      check the sparse add-on instead: v0.32/patches/vllm-qsa/ against the export of REF..QSA-REF
#   --qsa-ref  default v0.32.0-b70.4 (tag of the same fork: dense series plus the 36 sparse commits)
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
BASE=5df63a7203cdb9328d824a27b682e77c7b9928be
REPO_URL=https://github.com/Lumnus/vllm.git
gd="$PWD/vllm-src-032"; ref=v0.32.0-b70.4-dense; qsa=0; qref=v0.32.0-b70.4
while [ $# -gt 0 ]; do
  case "$1" in
    --git-dir) gd=$2; shift 2 ;;
    --ref) ref=$2; shift 2 ;;
    --qsa) qsa=1; shift ;;
    --qsa-ref) qref=$2; qsa=1; shift 2 ;;
    -h|--help) sed -n 2,13p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ ! -e "$gd/.git" ] && [ ! -e "$gd/HEAD" ]; then
  git clone -q --filter=blob:none --no-checkout "$REPO_URL" "$gd"
fi
git -C "$gd" rev-parse -q --verify "$ref^{commit}" >/dev/null || git -C "$gd" fetch -q --tags origin
if [ "$qsa" = 1 ]; then
  git -C "$gd" fetch -q origin "$qref" 2>/dev/null || git -C "$gd" fetch -q origin "refs/tags/$qref:refs/tags/$qref"
  qcommit=$(git -C "$gd" rev-parse "FETCH_HEAD^{commit}" 2>/dev/null || git -C "$gd" rev-parse "refs/tags/$qref^{commit}")
  git -C "$gd" merge-base --is-ancestor "$ref^{commit}" "$qcommit" || { echo "$qref does not descend from $ref" >&2; exit 1; }
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  git -C "$gd" -c core.abbrev=10 format-patch -q --no-signature -o "$tmp" "$ref..$qcommit"
  if diff -r "$here/patches/vllm-qsa" "$tmp" >"$tmp.diff"; then
    echo "OK — v0.32/patches/vllm-qsa equals the export of $ref..$qref ($qcommit, $(ls "$tmp" | wc -l) commits)"; exit 0
  else
    echo "MISMATCH:" >&2; sed -n 1,40p "$tmp.diff" >&2; exit 1
  fi
fi
git -C "$gd" merge-base --is-ancestor "$BASE" "$ref" || { echo "$ref does not descend from v0.32.0 ($BASE)" >&2; exit 1; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git -C "$gd" -c core.abbrev=10 format-patch -q --no-signature -o "$tmp" "$BASE..$ref"
if diff -r "$here/patches/vllm" "$tmp" >"$tmp.diff"; then
  echo "OK — v0.32/patches/vllm equals the export of $ref ($(git -C "$gd" rev-parse "$ref^{commit}"), $(ls "$tmp" | wc -l) commits)"
else
  echo "MISMATCH:" >&2; sed -n 1,40p "$tmp.diff" >&2; exit 1
fi
