#!/usr/bin/env bash
# check-series.sh — fail unless patches/vllm/ and patches/vllm-tests/ equal a fresh export of the fork branch.
#
#   scripts/check-series.sh [--git-dir DIR] [--ref REF]      (same defaults as export-series.sh)
#
# Run it before tagging a release and before building an image: the image is built from patches/, the fork
# branch is where the code lives, and the two must not drift.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
"$here/scripts/export-series.sh" "$@" --out "$tmp" >/dev/null
rc=0
for d in vllm vllm-tests; do
  if ! diff -r "$here/patches/$d" "$tmp/$d" >"$tmp/$d.diff"; then
    echo "MISMATCH in patches/$d:" >&2; sed -n 1,40p "$tmp/$d.diff" >&2; rc=1
  fi
done
[ $rc = 0 ] && echo "OK — patches/ equals the export of the fork branch ($(ls "$here"/patches/vllm/*.patch | wc -l) + $(ls "$here"/patches/vllm-tests/*.patch | wc -l) files)"
exit $rc
