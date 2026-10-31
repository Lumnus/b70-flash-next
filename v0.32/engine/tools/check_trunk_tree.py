#!/usr/bin/env python3
"""Tree-level check: kit series (dense + sparse + hot/cold + models) applied to vanilla vLLM == scrubbed trunk (with
--hotcold-count N > 0: except the trunk's hot/cold commits).  Called by v0.32/engine/tools/check-trunk-tree.sh; see its header for the arguments.

Everything runs on git objects with a temporary index: no checkout, no worktree, no branch is touched.
New blobs written by the scrub stay in the object database as unreferenced objects.
"""
import argparse, json, os, re, subprocess, sys, tempfile
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--git-dir", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--trunk", required=True)
ap.add_argument("--hotcold-count", type=int, default=12)
ap.add_argument("--hotcold-base", default=None)
ap.add_argument("--hotcold-extra", action="append", default=[], help="a hot/cold commit outside the range (its own diff against its parent counts as hot/cold); repeatable")
ap.add_argument("--hotcold-tip", default=None, help="last hot/cold commit when it is not the trunk tip (default: the trunk)")
ap.add_argument("--rules", required=True)
ap.add_argument("--series", action="append", required=True, help="patch directory, applied in the order given")
ap.add_argument("--show", type=int, default=60, help="max diff lines printed per drifting file")
a = ap.parse_args()
GD = a.git_dir


def git(*args, inp=None, env=None, raw=False, check=True):
    e = dict(os.environ, GIT_DIR=GD)
    if env: e.update(env)
    r = subprocess.run(["git", *args], input=inp, capture_output=True, env=e)
    if check and r.returncode:
        sys.exit(f"git {' '.join(args[:3])}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if raw else r.stdout.decode(errors="replace")


rules = json.load(open(a.rules))
DROP = set(rules.get("drop", []))
REPL = [("lit", r["old"], r["new"]) for r in rules.get("replace", []) if "old" in r]
REPL += [("re", re.compile(r["re"], re.M), r["new"]) for r in rules.get("replace", []) if "re" in r]
WARN = rules.get("warn", [])  # [{"path": ..., "contains": ..., "why": ...}]
used = Counter()


def scrubbed_tree(commit):
    """Tree of COMMIT with the scrub rules applied to file contents (text transforms) and DROP paths removed."""
    otree = git("rev-parse", commit + "^{tree}").strip()
    idx = tempfile.mktemp(); env = {"GIT_INDEX_FILE": idx}
    git("read-tree", otree, env=env)
    if DROP:
        git("update-index", "--index-info", env=env,
            inp="".join(f"0 {'0' * 40}\t{f}\n" for f in sorted(DROP)).encode())
    changed = git("diff", "--name-only", "-z", a.base, otree, raw=True).decode().split("\0")
    for f in filter(None, changed):
        if f in DROP: continue
        blob = git("cat-file", "blob", f"{otree}:{f}", raw=True, check=False)
        try: t = blob.decode()
        except UnicodeDecodeError: continue
        n = t
        for kind, old, new in REPL:
            if kind == "lit":
                if old in n: used[old] += n.count(old); n = n.replace(old, new)
            else:
                n2, c = old.subn(new, n)
                if c: used[old.pattern] += c; n = n2
        if n != t:
            h = git("hash-object", "-w", "--stdin", inp=n.encode()).strip()
            mode = git("ls-tree", otree, "--", f).split()[0]
            git("update-index", "--cacheinfo", f"{mode},{h},{f}", env=env)
    out = git("write-tree", env=env).strip(); os.unlink(idx)
    return out


def series_tree():
    idx = tempfile.mktemp(); env = {"GIT_INDEX_FILE": idx}
    git("read-tree", a.base, env=env)
    n = 0
    for d in a.series:
        for p in sorted(f for f in os.listdir(d) if f.endswith(".patch")):
            r = subprocess.run(["git", "apply", "--cached", os.path.join(d, p)], capture_output=True,
                               env=dict(os.environ, GIT_DIR=GD, GIT_INDEX_FILE=idx))
            if r.returncode:
                print(f"ERROR: {d}/{p} does not apply on the base: {r.stderr.decode().strip()}", file=sys.stderr); sys.exit(2)
            n += 1
    out = git("write-tree", env=env).strip(); os.unlink(idx)
    return out, n


def hunks(t1, t2):
    """{(path, header, body): count} of `git diff -U0 t1 t2`; body is the +/- lines of one hunk."""
    txt = git("diff", "-U0", "--no-color", "--no-renames", "--src-prefix=a/", "--dst-prefix=b/", t1, t2)
    res = Counter(); text = {}
    for blk in re.split(r"^diff --git ", txt, flags=re.M)[1:]:
        lines = blk.splitlines()
        path = lines[0].split(" b/", 1)[-1]
        i = 1; head = []
        while i < len(lines) and not lines[i].startswith("@@"):
            if not lines[i].startswith("index "): head.append(lines[i])
            i += 1
        head = "\n".join(head)
        cur = None
        for l in lines[i:]:
            if l.startswith("@@"):
                if cur is not None: k = (path, head, "\n".join(cur)); res[k] += 1; text[k] = None
                cur = []
            else: cur.append(l)
        if cur is not None: k = (path, head, "\n".join(cur)); res[k] += 1
        elif i >= len(lines): k = (path, head, ""); res[k] += 1
    return res


A, nfiles = series_tree()
tip = a.hotcold_tip or a.trunk
hb = a.hotcold_base or f"{tip}~{a.hotcold_count}"
B = scrubbed_tree(a.trunk)
Bb = scrubbed_tree(hb)
Bt = B if tip == a.trunk else scrubbed_tree(tip)
dAB = hunks(A, B)          # what separates the series from scrubbed trunk
dHC = hunks(Bb, Bt)        # what the hot/cold commits change (scrubbed)
if a.hotcold_extra:
    # the extra commits' own diffs are applied on top of the hot/cold tip tree, so a hot/cold file that a later
    # commit changed again is compared in its final form (not as two separate hunk sets)
    idx = tempfile.mktemp(); env = {"GIT_INDEX_FILE": idx}
    git("read-tree", f"{tip}^{{tree}}", env=env)
    for x in a.hotcold_extra:
        d = git("diff", "--binary", f"{x}~1", x, raw=True)
        r = subprocess.run(["git", "apply", "--cached", "-"], input=d, capture_output=True,
                           env=dict(os.environ, GIT_DIR=GD, GIT_INDEX_FILE=idx))
        if r.returncode: sys.exit(f"--hotcold-extra {x} does not apply on the hot/cold tip: {r.stderr.decode().strip()}")
    Bt = scrubbed_tree(git("write-tree", env=env).strip()); os.unlink(idx)
    dHC = hunks(Bb, Bt)
print(f"series: {nfiles} patches -> tree A {A[:12]}")
print(f"trunk {git('rev-parse', a.trunk).strip()[:12]} scrubbed -> tree B {B[:12]};  hot/cold base {git('rev-parse', hb).strip()[:12]} -> {sum(dHC.values())} hunks in {len({k[0] for k in dHC})} files")

drift, warn, allowed = [], [], 0
hc_left = Counter(dHC)
for k, n in dAB.items():
    for _ in range(n):
        if hc_left[k] > 0:
            hc_left[k] -= 1; allowed += 1; continue
        w = next((w for w in WARN if w["path"] == k[0] and w["contains"] in k[2]), None)
        (warn if w else drift).append((k, w))
absent = sum(hc_left.values())

for k, w in warn:
    print(f"WARN  {k[0]}: hunk the scrub rules cannot make identical ({w.get('why', 'listed in the rules file')}):")
    print("\n".join("      " + l for l in k[2].splitlines()[: a.show]))
if absent:
    print(f"note: {absent} hot/cold hunk(s) are not between series and trunk (already identical in the series): "
          + ", ".join(sorted({k[0] for k, n in hc_left.items() if n > 0})))
unused = [r for r in rules.get("replace", []) if (r.get("old") or r.get("re")) not in used]
if unused:
    print(f"note: {len(unused)} replace rule(s) matched nothing in trunk (stale, or only needed for the series' own history)")
print(f"allowed: {allowed} hot/cold hunk(s); warn: {len(warn)}; drift: {len(drift)}")
if drift:
    print("DRIFT: series tree A and scrubbed trunk B differ outside the hot/cold commits:")
    byfile = {}
    for k, _ in drift: byfile.setdefault(k[0], []).append(k)
    for f, ks in sorted(byfile.items()):
        print(f"  {f} ({len(ks)} hunk{'s' if len(ks) > 1 else ''}; '-' = series, '+' = trunk)")
        for k in ks:
            if k[1]: print("      [" + k[1].replace("\n", "; ") + "]")
            print("\n".join("      " + l for l in k[2].splitlines()[: a.show]))
    sys.exit(1)
print("PASS")
