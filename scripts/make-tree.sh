#!/usr/bin/env bash
# make-tree.sh — produce a vLLM source tree for a release, and verify it.
#
#   scripts/make-tree.sh <release|ref> <dir> [--from-patches]
#
#   <release|ref>   a release tag of this repo's series (e.g. 0.30.0-b70.1 -> fork branch b70/v0.30.0 or tag
#                   v0.30.0-b70.1 when it exists), or any fork branch/tag/commit
#   <dir>           target directory (must not exist)
#   --from-patches  do not clone the fork branch; apply this repo's patches/ onto a clean v0.30.0
#                   (ced6857afa) checkout instead
#
# Either way the result is checked the same way: every file the series touches must carry the sha256 recorded
# in image/verify-overlay.sh ("final" set). A clone is also checked against a --from-patches build.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
BASE=ced6857afa0ea7b2e3f0846a62e1394e90f15607
URL=https://github.com/Lumnus/vllm.git
[ $# -ge 2 ] || { sed -n 2,14p "$0"; exit 2; }
rel=$1; dir=$2; mode=${3:-clone}
[ ! -e "$dir" ] || { echo "$dir exists" >&2; exit 1; }
case "$rel" in
  0.30.0-b70.*) ref=b70/v0.30.0; tag="v$rel" ;;
  *) ref=$rel; tag="" ;;
esac

apply_series() {   # $1 = empty dir
  git init -q "$1"
  git -C "$1" remote add origin "$URL"
  git -C "$1" fetch -q --depth 1 origin "$BASE"
  git -C "$1" checkout -q FETCH_HEAD
  ( cd "$1"; export LC_ALL=C
    for p in "$here"/patches/vllm/*.patch; do
      patch -p1 --fuzz=0 --forward --no-backup-if-mismatch --dry-run -s < "$p" >/dev/null
      patch -p1 --fuzz=0 --forward --no-backup-if-mismatch -s < "$p"
    done
    ! find . -name '*.orig' -o -name '*.rej' | grep -q . )
}

if [ "$mode" = "--from-patches" ]; then
  apply_series "$dir"
  echo "applied $(ls "$here"/patches/vllm/*.patch | wc -l) patches onto $BASE in $dir"
else
  git init -q "$dir"
  git -C "$dir" remote add origin "$URL"
  if [ -n "$tag" ] && git -C "$dir" fetch -q --depth 1 origin "refs/tags/$tag:refs/tags/$tag" 2>/dev/null; then
    git -C "$dir" checkout -q "$tag"
  else
    git -C "$dir" fetch -q --depth 32 origin "+refs/heads/$ref:refs/remotes/origin/$ref" 2>/dev/null \
      || git -C "$dir" fetch -q --depth 32 origin "$ref"
    git -C "$dir" checkout -q "origin/$ref" 2>/dev/null || git -C "$dir" checkout -q FETCH_HEAD
  fi
  echo "checked out $(git -C "$dir" rev-parse HEAD) in $dir"
  ref_tree=$(mktemp -d); trap 'rm -rf "$ref_tree"' EXIT
  apply_series "$ref_tree/t"
  diff -rq --exclude=.git "$ref_tree/t/vllm" "$dir/vllm" \
    && echo "vllm/ equals this repo's patches/ applied to $BASE" \
    || { echo "vllm/ differs from patches/ applied to $BASE (run scripts/check-series.sh)" >&2; exit 1; }
fi

# hash check against image/verify-overlay.sh: every wu1ff-layer and b70-series file must carry its "final" sha256
python3 - "$dir" "$here/image/verify-overlay.sh" <<'PY'
import hashlib, re, sys, pathlib
root, vo = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text()
def block(name):
    body = re.search(name + r"='\n(.*?)'", vo, re.S).group(1)
    return [l.split() for l in body.splitlines() if l.strip()]
want = {rel: sha for sha, rel in block("PY_FILES")}                 # wu1ff layer
want.update({rel: sha for sha, _tag, rel in block("B70_FILES")})     # the series result overrides
bad = 0
for rel, sha in sorted(want.items()):
    got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    if got != sha:
        bad += 1
        print(f"DIFFERENT {rel}: want {sha} got {got}", file=sys.stderr)
print(f"hash check: {len(want) - bad} of {len(want)} files carry the verify-overlay.sh 'final' sha256")
sys.exit(1 if bad else 0)
PY
