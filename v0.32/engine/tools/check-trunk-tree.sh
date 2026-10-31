#!/usr/bin/env bash
# check-trunk-tree.sh — tree-level check that the kit's series equal a trunk.
#
#   v0.32/engine/tools/check-trunk-tree.sh --trunk-ref REF --rules FILE [--trunk-url URL] [--git-dir DIR] [--no-models]
#                             [--no-hotcold [--hotcold-count N] [--hotcold-base REF] [--hotcold-tip REF] [--hotcold-extra REF]...]
#                             [--no-qsa]
#
# 1. tree A = vanilla vLLM v0.32.0 + v0.32/engine/patches/vllm (dense) + vllm-qsa (sparse) + vllm-hotcold (hot/cold)
#    + vllm-models (the 27B/35B layer and the backstep guard), applied in order
# 2. tree B = the trunk at REF, with the scrub rules applied to file contents (text transforms, not history)
# 3. A against B: by default every hunk is DRIFT (the four series must equal the trunk exactly). With --no-hotcold
#    the hot/cold series is left out and the hot/cold commits of the trunk are allowed as differences: the last N
#    commits (REF~N..REF), or up to --hotcold-tip when later commits sit on top of them, plus single --hotcold-extra
#    commits. Anything else is DRIFT, printed per file with its hunks, exit 1. Hunks the rules cannot make
#    identical may be listed as "warn" entries in the rules file: printed as WARN, allowed.
#
#   --trunk-ref  the trunk commit or ref (required); with --trunk-url it is fetched from there first
#   --trunk-url  where to fetch the trunk from (optional; without it REF must already be in --git-dir)
#   --rules      JSON file: {"drop": [paths], "replace": [{"old": text, "new": text} | {"re": regex, "new": text}],
#                "warn": [{"path": p, "contains": text, "why": text}]}. These are the scrub rules that turned the
#                working branch into the published series. They name the working host's paths, so they are not in
#                this repository: the maintainer who holds the trunk keeps them.
#   --git-dir    a vLLM clone (default ./vllm-src-032, cloned on demand from https://github.com/Lumnus/vllm.git)
#   --no-models  leave v0.32/engine/patches/vllm-models out (for a trunk that ends at the hot/cold layer, e.g. b70.4's)
#   --no-hotcold leave v0.32/engine/patches/vllm-hotcold out and allow the trunk's hot/cold commits (count default 12)
#   --hotcold-extra  a single hot/cold commit that sits outside the range (e.g. one landed after later series commits); its own diff counts as hot/cold
#   --no-qsa     check the dense series only (trunk then has to be dense-only too)
#
# Nothing is checked out and no branch is moved: the trees are built with a temporary index. New objects (scrubbed
# blobs) are written to --git-dir as unreferenced objects. Exit 0 PASS, 1 DRIFT, 2 error.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
BASE=5df63a7203cdb9328d824a27b682e77c7b9928be
REPO_URL=https://github.com/Lumnus/vllm.git
gd="$PWD/vllm-src-032"; tref=""; turl=""; rules=""; n=12; hb=""; ht=""; qsa=1; hc=1; md=1; hx=()
while [ $# -gt 0 ]; do
  case "$1" in
    --trunk-ref) tref=$2; shift 2 ;;
    --trunk-url) turl=$2; shift 2 ;;
    --rules) rules=$2; shift 2 ;;
    --git-dir) gd=$2; shift 2 ;;
    --hotcold-count) n=$2; shift 2 ;;
    --hotcold-tip) ht=$2; shift 2 ;;
    --hotcold-base) hb=$2; shift 2 ;;
    --hotcold-extra) hx+=(--hotcold-extra "$2"); shift 2 ;;
    --no-qsa) qsa=0; hc=0; md=0; shift ;;
    --no-models) md=0; shift ;;
    --no-hotcold) hc=0; md=0; shift ;;
    -h|--help) sed -n 2,29p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$tref" ] && [ -f "$rules" ] || { echo "need --trunk-ref and --rules FILE (see --help)" >&2; exit 2; }
if [ ! -e "$gd/.git" ] && [ ! -e "$gd/HEAD" ]; then
  git clone -q --filter=blob:none --no-checkout "$REPO_URL" "$gd"
fi
git -C "$gd" cat-file -e "$BASE^{commit}" 2>/dev/null || git -C "$gd" fetch -q origin "$BASE"
if [ -n "$turl" ]; then
  git -C "$gd" fetch -q "$turl" "$tref"
  tref=$(git -C "$gd" rev-parse FETCH_HEAD)
fi
git -C "$gd" rev-parse -q --verify "$tref^{commit}" >/dev/null || { echo "trunk ref $tref not found in $gd" >&2; exit 2; }
series=(--series "$here/patches/vllm")
[ "$qsa" = 1 ] && series+=(--series "$here/patches/vllm-qsa")
[ "$hc" = 1 ] && { series+=(--series "$here/patches/vllm-hotcold"); n=0; hb=""; ht=""; hx=(); }
[ "$md" = 1 ] && series+=(--series "$here/patches/vllm-models")
gdir=$(git -C "$gd" rev-parse --absolute-git-dir)
args=(--git-dir "$gdir" --base "$BASE" --trunk "$tref" --hotcold-count "$n" --rules "$rules" "${series[@]}")
[ -n "$hb" ] && args+=(--hotcold-base "$hb")
[ -n "$ht" ] && args+=(--hotcold-tip "$ht")
[ ${#hx[@]} -gt 0 ] && args+=("${hx[@]}")
exec python3 -I "$here/tools/check_trunk_tree.py" "${args[@]}"
