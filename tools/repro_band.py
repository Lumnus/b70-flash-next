#!/usr/bin/env python3
"""repro_band.py — reproduce the hybrid-model KV-offload "same document, new question" miss (docs/offload-fix.md).

Standalone: Python 3.9+, standard library only (plus `transformers` if you count tokens locally).
Works against any OpenAI-compatible vLLM server that exposes /metrics and, for --tokenize-remote, /tokenize.

Hypothesis under test: with prefix_cache_retention_interval=0 each request stores ONE Mamba/GDN checkpoint at
B0 = round_down(L0-1, BLOCK) (BLOCK = the KV block size, 832 tokens for Qwen3.8-Flash-Next on vLLM v0.30.0). A revisit
(same document, different question) shares the prefix up to P (the token just before the question text diverges).
It hits from the CPU offload tier only if a stored checkpoint lies at or before P. So:
  in-band  doc: B0 >  P  (the checkpoint block reaches into the question)  -> predicted permanent MISS (unpatched)
  out-band doc: B0 <= P  (the checkpoint block is pure document)           -> predicted HIT
No concurrency: every request is serial, so every /metrics delta belongs to exactly one request.

Sequence: store pass (each doc + question A) -> FILLER x --filler distinct ~60K docs (flush the GPU prefix cache;
no cache reset) -> revisit pass B (each doc + question B) -> FILLER x --filler2 -> revisit pass C (each doc + question C).
(--filler2 0 skips the second flush; then C can be served from the GPU prefix cache, recorded as prefix_cache_hits.)

The band is COMPUTED from the template shape (exact token counts, exact prefix divergence), never hard-coded.
Docs are padded with " the" tokens so the shared prefix P lands exactly where the plan says.

Token counting: --tokenize-remote uses the server's POST /tokenize (chat form, same template kwargs as the request);
otherwise a local transformers AutoTokenizer from --tokenizer DIR (the served model's tokenizer + chat template).
With the patched engine (B70_OFFLOAD_JUNCTION=1) the in-band docs miss once (B) and hit on C; with
B70_OFFLOAD_GDN_BACKSTEP=1 as well they hit on B.

usage:
  repro_band.py --dry-run --tokenizer DIR                          # plan only, no server contact
  repro_band.py --out OUT.jsonl --tokenizer DIR --base http://HOST:8000 --model NAME [--k 3] [--filler 8]
  repro_band.py --out OUT.jsonl --tokenize-remote --base http://HOST:8000 --model NAME
A file named STOP next to OUT halts the run before the next request.
"""
import argparse, json, os, random, re, sys, time, urllib.error, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="repro_band.jsonl")
ap.add_argument("--base", default=os.environ.get("VLLM_BASE", "http://localhost:8000"), help="server base URL")
ap.add_argument("--model", default=os.environ.get("VLLM_MODEL"), help="served model name (default: the server's first model)")
ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "none"))
ap.add_argument("--tokenizer", default=None, help="dir with tokenizer.json + chat template (local counting)")
ap.add_argument("--tokenize-remote", action="store_true", help="count with the server's POST /tokenize")
ap.add_argument("--template-kwargs", default="{}", help="extra chat_template_kwargs as JSON (merged into every request)")
ap.add_argument("--k", type=int, default=3, help="docs IN the band and docs OUT of it (each)")
ap.add_argument("--block", type=int, default=832, help="KV block size in tokens")
ap.add_argument("--band-block", type=int, default=72, help="checkpoint block index m: B0 = m*BLOCK (72 -> 59,904)")
ap.add_argument("--margins", default="8,16,24", help="tokens between P and the block edge, cycled per doc (< s_A-1)")
ap.add_argument("--filler", type=int, default=8, help="~60K filler docs between store and revisit B")
ap.add_argument("--filler2", type=int, default=None, help="filler between B and C (default = --filler; 0 = none)")
ap.add_argument("--filler-tokens", type=int, default=60000)
ap.add_argument("--effort", default="none", help="'none' = enable_thinking false; else sent as reasoning_effort")
ap.add_argument("--max-tokens", type=int, default=48)
ap.add_argument("--seed", default=None, help="fix the salt (default random)")
ap.add_argument("--prefill-tps", type=float, default=2600.0, help="for the wall-time estimate")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
if args.filler2 is None: args.filler2 = args.filler
ROOT = re.sub(r"/v1/?$", "", args.base.rstrip("/"))
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
EXTRA_CTK = json.loads(args.template_kwargs)
BLK = args.block
MARGINS = [int(x) for x in args.margins.split(",")]

# ------------------------------------------------------------------ minimal client
def _post(path, body, timeout=120):
    req = urllib.request.Request(ROOT + path, data=json.dumps(body).encode(), headers=HDR)
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def _model():
    if args.model: return args.model
    if args.dry_run and not args.tokenize_remote: return "unset"
    req = urllib.request.Request(ROOT + "/v1/models", headers=HDR)
    return json.load(urllib.request.urlopen(req, timeout=30))["data"][0]["id"]
MODEL = _model()

def _kw(effort):
    kw = dict(EXTRA_CTK)
    if effort in (None, "none", "off"): kw["enable_thinking"] = False
    else: kw["reasoning_effort"] = effort
    return kw

def body(messages, max_tokens, effort):
    b = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
         "chat_template_kwargs": _kw(effort)}
    if effort not in (None, "none", "off"): b["reasoning_effort"] = effort
    return b

def new_salt():
    return uuid.uuid4().hex

def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")

# ------------------------------------------------------------------ streaming request (time to first token)
def chat_stream(b, wall_cap=1800):
    """Stream one chat completion. Times: t_first (first reasoning OR content token = end of prefill),
    t_first_content, t_last (last token). Returns a record dict."""
    b = dict(b, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(ROOT + "/v1/chat/completions", data=json.dumps(b).encode(), headers=HDR)
    t0 = time.time()
    rec = {"t_first": None, "t_first_content": None, "t_last": None, "reasoning": "", "content": "",
           "finish_reason": None, "usage": None, "error": None, "capped": False, "n_chunks": 0}
    try:
        with urllib.request.urlopen(req, timeout=min(wall_cap, 1800)) as r:
            for raw in r:
                now = time.time() - t0
                if now > wall_cap:
                    rec["capped"] = True
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    rec["usage"] = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    rs = d.get("reasoning_content") or d.get("reasoning") or ""
                    c = d.get("content") or ""
                    if rs or c:
                        rec["t_first"] = rec["t_first"] if rec["t_first"] is not None else now
                        rec["t_last"] = now
                        rec["n_chunks"] += 1
                    if rs:
                        rec["reasoning"] += rs
                    if c:
                        rec["t_first_content"] = rec["t_first_content"] if rec["t_first_content"] is not None else now
                        rec["content"] += c
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        rec["error"] = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["t_total"] = time.time() - t0
    if "<think>" in rec["content"]:  # no reasoning parser configured
        head, _, rest = rec["content"].partition("<think>")
        think, closed, tail = rest.partition("</think>")
        rec["reasoning"] += think
        rec["content"] = head + tail if closed else head
    return rec



# ------------------------------------------------------------------ synthetic documents (salted, deterministic)
_NAMES = ["Anja", "Borislav", "Céline", "Dario", "Elif", "Farid", "Greta", "Hamid", "Ines", "Joaquin", "Kaisa",
          "Lorenzo", "Maren", "Nikolai", "Oona", "Pavel", "Quinn", "Rosalind", "Soren", "Tamsin", "Ulrich",
          "Vesna", "Wendell", "Ximena", "Yusuf", "Zora"]
_PLACES = ["the upper valley", "the old harbour", "a hill town", "the salt flats", "the northern ridge",
           "the market square", "a fishing village", "the river delta", "the pine forest", "the quarry road",
           "a mountain pass", "the eastern plain", "the lighthouse point", "an orchard terrace", "the canal district"]
_THINGS = ["bread", "wool", "copper pots", "dried figs", "lamp oil", "rope", "honey", "slate tiles", "linen",
           "barley", "cedar planks", "glass beads", "salted fish", "leather boots", "clay jars"]
_WEATHER = ["a thin rain", "a dry wind from the south", "early frost", "a long heat", "low fog", "sudden hail",
            "clear cold mornings", "a week of storms", "mild grey skies", "heavy snow on the passes"]
_ADJ = ["patient", "stubborn", "careful", "restless", "generous", "quiet", "cheerful", "weary", "curious", "proud"]
_VERB = ["repaired", "painted", "measured", "carried", "sorted", "mended", "counted", "polished", "planted", "stacked"]
_OBJ = ["the fence by the well", "a cart with a split axle", "the chapel roof", "a row of beehives",
        "a stack of old ledgers", "the school benches", "a flock of goats", "the mill wheel",
        "a crate of apples", "the bridge railings", "the bakery ovens"]
_TPL = [
    "In {place}, {weather} arrived {when}, and most people stayed close to their stoves.",
    "{name} was a {adj} person who {verb} {obj} before anyone else was awake.",
    "Traders from {place} brought {thing} and left with {thing2}, grumbling about the prices.",
    "Nobody in {place} could remember a season like it; the elders compared it to the year of {weather}.",
    "The path from {place} to {place2} took most of a day on foot, longer if the ford was high.",
    "{name} and {name2} argued for an hour about whether {thing} kept better in a cellar or a loft.",
    "Children in {place} learned to count by stacking {thing} in the courtyard.",
    "When {weather} came, {name} {verb} {obj} and then sat down to write letters.",
    "There was a small museum in {place} that displayed {thing} from three centuries back.",
    "The inn at {place} served soup at noon and closed its shutters at dusk, whatever the season.",
    "{name} kept a notebook of every traveller who passed through {place}, with a line about each.",
    "A {adj} dog followed {name} everywhere, even to the edge of {place}.",
    "Some said the {thing} from {place} was the finest in the region; others disagreed loudly.",
    "By late afternoon the light over {place} turned the colour of {thing}, or so {name} claimed.",
    "Most weeks the carrier from {place2} was late, and {name} would wait on the steps with tea.",
    "{name} once walked from {place} to {place2} in the middle of {weather} just to deliver {thing}.",
    "The council in {place} met twice a month to discuss roads, water and the price of {thing}.",
    "In the evenings, {name2} played an old fiddle while {name} {verb} {obj}.",
    "It was said that {weather} always followed a good year for {thing} in {place}.",
    "Letters from {place} took nine days to reach {place2}, unless the mountain pass was closed.",
]
_WHEN = ["in the second week of spring", "without warning", "just after the harvest", "late one evening",
         "at the turn of the year", "on a market day", "before dawn", "in the middle of summer"]


def prose_paragraph(rng):
    out = []
    for _ in range(rng.randint(4, 8)):
        t = rng.choice(_TPL)
        out.append(t.format(place=rng.choice(_PLACES), place2=rng.choice(_PLACES), weather=rng.choice(_WEATHER),
                            when=rng.choice(_WHEN), name=rng.choice(_NAMES), name2=rng.choice(_NAMES),
                            adj=rng.choice(_ADJ), verb=rng.choice(_VERB), obj=rng.choice(_OBJ),
                            thing=rng.choice(_THINGS), thing2=rng.choice(_THINGS)))
    return " ".join(out)


def prose_paragraphs(n, salt, gen=prose_paragraph):
    rng = random.Random(salt)
    return [gen(rng) for _ in range(n)]


SYS = "You are a careful reader. Answer strictly from the document provided."

_COMP = ["label printer", "badge reader", "report scheduler", "font cache", "thumbnail renderer", "audit exporter",
         "locale loader", "map tiler", "invoice formatter", "sensor bridge", "calendar sync", "search indexer",
         "print spooler", "config watcher", "certificate rotator", "backup verifier"]
_ACT = ["refreshes its lookup table", "compacts its scratch files", "writes a summary row", "checks its clock",
        "rotates its log file", "validates its templates", "pings its upstream", "flushes its buffer",
        "reloads its settings", "recomputes its checksums"]
_STORE = ["the shared disk", "the reporting database", "the object store", "the local cache directory",
          "the status board", "the nightly bundle"]


def spec_paragraph(r):
    comp, comp2 = r.choice(_COMP), r.choice(_COMP)
    s = [f"Section {r.randint(2, 40)}.{r.randint(1, 20)} — the {comp}.",
         f"The {comp} {r.choice(_ACT)} every {r.randint(2, 90)} minutes and records the outcome in {r.choice(_STORE)}.",
         f"If the {comp} cannot reach {r.choice(_STORE)}, it retries up to {r.randint(2, 12)} times with a "
         f"{r.randint(5, 120)}-second pause between attempts.",
         f"Operators should note that the {comp} depends on the {comp2}, which {r.choice(_ACT)} at start-up.",
         f"Changes to the {comp} are reviewed at the {r.choice(['weekly', 'monthly', 'quarterly'])} design meeting "
         f"and noted in the change log with a short rationale.",
         f"Its default configuration allocates {r.randint(64, 4096)} MB of memory and "
         f"{r.randint(1, 16)} CPU threads, which is usually sufficient.",
         f"A known limitation is that the {comp} logs timestamps in local time rather than UTC."]
    r.shuffle(s)
    return " ".join(s[:r.randint(4, 7)])


_CODENAMES = ["Basalt", "Heliotrope", "Juniper", "Quartzline", "Marram", "Tessellate", "Obsidian", "Larchmont"]
_TEAMS = ["Heron", "Kestrel", "Lynx", "Marten", "Osprey", "Puffin", "Stoat", "Wren"]


def doc_facts(r):
    W, R, k = r.randint(4, 9), r.randint(5, 15), r.randint(3, 9)
    D, M = r.randint(10, 30), r.randint(2, 5)
    cn, team, ext = r.choice(_CODENAMES), r.choice(_TEAMS), r.randint(2000, 8999)
    Q = W * R * k
    facts = [
        f"The default processing pool of the Relay service runs {W} workers.",
        f"Each Relay worker handles {R} messages per second when the pool is healthy.",
        f"The Relay ingest queue can hold at most {Q} messages before it rejects new ones.",
        f"Messages in the Relay primary store are kept for {D} days; this is the primary retention window.",
        f"The long-term archive tier of the Relay service is code-named {cn}.",
        f"The {cn} tier keeps messages {M} times as long as the primary retention window.",
        f"Operational ownership of the {cn} tier belongs to Team {team}.",
        f"Team {team} can be reached on escalation extension {ext}.",
    ]
    qs = [
        ("How many messages per second can the Relay service's default processing pool handle in total when "
         "healthy? Answer with just the number.", W * R, "F1+F2"),
        ("Starting from a completely full Relay ingest queue with no new arrivals, how many seconds does the "
         "healthy default processing pool need to empty it? Answer with just the number.", k, "F1+F2+F3"),
        ("For how many days does the Relay long-term archive tier keep messages? Answer with just the number.",
         D * M, "F4+F5+F6"),
        ("Which escalation extension should be called about a problem with the Relay long-term archive tier? "
         "Answer with just the extension number.", ext, "F5+F7+F8"),
        ("If the Relay default processing pool ran three more workers than it does now, how many messages per "
         "second could it handle in total when healthy? Answer with just the number.", (W + 3) * R, "F1+F2"),
    ]
    return facts, qs


# ------------------------------------------------------------------ exact token ids
if args.tokenize_remote:
    def ids(msgs):
        b = {"model": MODEL, "messages": msgs, "chat_template_kwargs": _kw(args.effort), "add_generation_prompt": True}
        r = _post("/tokenize", b, timeout=300)
        return r["tokens"]
elif args.tokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("local counting needs `transformers` (pip install tokenizers transformers jinja2), or use --tokenize-remote")
    _tk = AutoTokenizer.from_pretrained(args.tokenizer)
    def ids(msgs):
        s = _tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **_kw(args.effort))
        return _tk(s, add_special_tokens=False)["input_ids"]
else:
    sys.exit("give --tokenizer DIR or --tokenize-remote")

def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return i

bnd = lambda L: ((L - 1) // BLK) * BLK          # where the single GDN checkpoint of a request of length L lands

# ------------------------------------------------------------------ doc builder (longctx doc shape + exact pad)
def doc_msgs(n, pad, salt, facts, question):
    r = random.Random(salt + "-place")
    paras = prose_paragraphs(n, salt, spec_paragraph)
    depths = [6, 18, 30, 42, 55, 67, 80, 94]
    order = list(range(len(facts))); r.shuffle(order)
    for d, fi in sorted(zip(depths, order), reverse=True):
        paras.insert(min(n, round(d / 100 * n)), facts[fi])
    doc = f"Relay service specification, revision {salt}\n\n" + "\n\n".join(paras) + (" the" * pad)
    user = (f"Below is the specification of the Relay service. Read it, then answer the question at the end.\n\n"
            f"{doc}\n\nQuestion: {question}")
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}]

# question suffix length s_q = tokens after the shared prefix, measured once on a small dummy doc
_f0, _qs0 = doc_facts(random.Random("band-dummy-facts"))
QUESTIONS = [q[0] for q in _qs0[:3]]      # A (store), B, C — first words differ, so the prefix diverges at the same token
WANT = [q[1] for q in _qs0[:3]]           # placeholder answers only valid per-doc; recomputed per doc below
_dd = [ids(doc_msgs(20, 0, "band-dummy", _f0, q)) for q in QUESTIONS]
P_DUMMY = min(lcp(_dd[0], x) for x in _dd[1:])
S_Q = [len(x) - P_DUMMY for x in _dd]

def build_doc(name, salt, target_P):
    """Return dict with per-question messages whose shared prefix ends EXACTLY at target_P tokens."""
    facts, qs = doc_facts(random.Random(salt + "-facts"))
    Q = [q[0] for q in qs[:3]]
    L_of = lambda n, pad: len(ids(doc_msgs(n, pad, salt, facts, Q[0]))) - S_Q[0]   # = P
    per = (L_of(200, 0) - L_of(100, 0)) / 100.0
    n = max(1, int((target_P - L_of(1, 0)) / per))
    while L_of(n, 0) > target_P: n -= 1
    while L_of(n + 8, 0) <= target_P: n += 8
    while L_of(n + 1, 0) <= target_P: n += 1
    pad = target_P - L_of(n, 0)
    for _ in range(6):                                                    # pad tokens are 1:1; iterate to exact
        P = L_of(n, pad)
        if P == target_P: break
        pad += target_P - P
        if pad < 0: n -= 1; pad = target_P - L_of(n, 0)
    msgs = [doc_msgs(n, pad, salt, facts, q) for q in Q]
    I = [ids(m) for m in msgs]
    P_AB, P_AC, P_BC = lcp(I[0], I[1]), lcp(I[0], I[2]), lcp(I[1], I[2])
    return dict(name=name, salt=salt, n=n, pad=pad, msgs=msgs, want=[qs[i][1] for i in range(3)], L=[len(x) for x in I],
                P=P_AB, P_AB=P_AB, P_AC=P_AC, P_BC=P_BC, B0=bnd(len(I[0])), B_B=bnd(len(I[1])),
                exact=(P_AB == target_P and P_AC == target_P and P_BC == target_P))

base = args.seed or new_salt()
m0 = args.band_block * BLK
docs = []
for i in range(args.k):
    docs.append(build_doc(f"in{i}", f"{base}-in-{i}", m0 - MARGINS[i % len(MARGINS)]))     # P below the block edge
    docs.append(build_doc(f"out{i}", f"{base}-out-{i}", m0 + MARGINS[i % len(MARGINS)]))   # P above it
for d in docs:
    d["in_band"] = d["B0"] > d["P"]
    d["pred_B"] = 0 if d["in_band"] else d["B0"]                       # predicted external-hit tokens on revisit B
    # C: ckpts = {B0 (orig)} + {B_B if B missed}; usable if <= P
    stored = [d["B0"]] + ([d["B_B"]] if d["pred_B"] == 0 else [])
    d["pred_C"] = max([c for c in stored if c <= d["P"]] or [0])

def plan():
    print(f"# base salt {base}  block {BLK}  band block m={args.band_block} (B0 target {m0})  s_q(A,B,C)={S_Q}  dummy P={P_DUMMY}")
    print(f"# band: in-band  <=> B0 > P, i.e. L0 in [{m0 + 1}, {m0 + S_Q[0] - 1}] for question A (s_A={S_Q[0]});"
          f" out-band <=> L0 >= {m0 + S_Q[0]}")
    print(f"{'doc':5s} {'band':4s} {'n':>4s} {'pad':>4s} {'L_A':>6s} {'L_B':>6s} {'L_C':>6s} {'P':>6s} {'B0':>6s} {'B_B':>6s} {'predB':>6s} {'predC':>6s} exactP")
    for d in docs:
        print(f"{d['name']:5s} {'IN' if d['in_band'] else 'OUT':4s} {d['n']:4d} {d['pad']:4d} {d['L'][0]:6d} {d['L'][1]:6d} {d['L'][2]:6d} "
              f"{d['P']:6d} {d['B0']:6d} {d['B_B']:6d} {d['pred_B']:6d} {d['pred_C']:6d} {d['exact']}")
    nin = sum(d["in_band"] for d in docs)
    assert nin == args.k and len(docs) - nin == args.k, f"band split wrong: {nin} in / {len(docs) - nin} out"
    bad = [d["name"] for d in docs if not d["exact"]]
    if bad: print("# WARNING: prefix not exactly on target for", bad, "(question starts share a token?)")
    ntok_doc = sum(d["L"][0] for d in docs)
    nfill = args.filler + args.filler2
    ncold = len(docs) + nfill + sum(d["in_band"] for d in docs) * 2                 # store + filler + in-band misses (B,C)
    tok = ntok_doc + nfill * args.filler_tokens + sum(d["L"][1] + d["L"][2] for d in docs if d["in_band"])
    nreq = len(docs) * 3 + nfill
    est = tok / args.prefill_tps + nreq * 1.5 + len(docs) * 2 * 1.0
    print(f"# requests {nreq} (docs {len(docs)}x3 + filler {nfill}); cold-prefilled tokens ~{tok:,} ; est wall @ {args.prefill_tps:.0f} tok/s"
          f" prefill: {est / 60:.1f} min (+ ~1 s per hit, +1.5 s metric settle per request included)")
    return est

est = plan()
if args.dry_run:
    print("# dry-run: engine not contacted"); sys.exit(0)

# ------------------------------------------------------------------ live run (serial, one request at a time)
KEYS = ("external_prefix_cache_hits_total", "external_prefix_cache_queries_total", "kv_offload_load_bytes_total",
        "kv_offload_store_bytes_total", "prefix_cache_hits_total", "prefix_cache_queries_total", "prompt_tokens_total",
        "num_requests_running", "num_requests_waiting")

def mx():
    txt = urllib.request.urlopen(urllib.request.Request(ROOT + "/metrics", headers=HDR), timeout=15).read().decode(); d = {}
    for line in txt.splitlines():
        if line.startswith("#"): continue
        n = line.split("{", 1)[0].split(" ", 1)[0].split(":")[-1]
        if n in KEYS:
            try: d[n] = d.get(n, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError: pass
    return d

STOPFILE = os.path.join(os.path.dirname(os.path.abspath(args.out)), "STOP")

def stop():
    if os.path.exists(STOPFILE): print("STOP seen"); sys.exit(0)

def run(kind, msgs, extra):
    stop(); m0_ = mx()
    if kind == "filler":
        b = body(msgs, 1, args.effort); t = time.time(); r = _post("/v1/chat/completions", b, timeout=900)
        row = {"ttft_s": round(time.time() - t, 2), "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens")}
    else:
        rec_ = chat_stream(body(msgs, args.max_tokens, args.effort), wall_cap=900)
        row = {"ttft_s": round(rec_["t_first"], 2) if rec_["t_first"] is not None else None,
               "prompt_tokens": (rec_.get("usage") or {}).get("prompt_tokens"), "content": rec_["content"][:400],
               "finish_reason": rec_["finish_reason"], "error": rec_["error"]}
    time.sleep(1.5); m1 = mx()
    d = {k: round(m1.get(k, 0) - m0_.get(k, 0), 1) for k in KEYS if not k.startswith("num_")}
    rec = {"t": time.strftime("%H:%M:%SZ", time.gmtime()), "kind": kind, **extra, **row,
           "ext_hit_tokens": d["external_prefix_cache_hits_total"], "ext_query_tokens": d["external_prefix_cache_queries_total"],
           "offload_loaded_bytes": d["kv_offload_load_bytes_total"], "offload_stored_bytes": d["kv_offload_store_bytes_total"],
           "gpu_prefix_hit_tokens": d["prefix_cache_hits_total"],
           "running_before": m0_.get("num_requests_running"), "waiting_before": m0_.get("num_requests_waiting")}
    append_jsonl(args.out, rec)
    tag = {k: extra[k] for k in extra if k in ("phase", "doc", "in_band")}
    print(f"{rec['t']} {kind:6s} {json.dumps(tag)} ttft={rec['ttft_s']} tok={rec['prompt_tokens']} ext {int(rec['ext_hit_tokens'])}/"
          f"{int(rec['ext_query_tokens'])} load={rec['offload_loaded_bytes']/1e9:.2f}GB store={rec['offload_stored_bytes']/1e9:.2f}GB "
          f"gpuhit={int(rec['gpu_prefix_hit_tokens'])} run/wait={rec['running_before']}/{rec['waiting_before']}", flush=True)
    return rec

# filler docs: distinct salted prose, length via local/remote count
def make_filler(tag, f):
    fs = f"{base}-fill-{tag}-{f}"
    build = lambda k: [{"role": "user", "content": f"Collection {fs}\n\n" + "\n\n".join(prose_paragraphs(k, fs)) + "\n\nReply OK."}]
    per = (len(ids(build(200))) - len(ids(build(100)))) / 100.0
    n = max(1, int(args.filler_tokens / per))
    while len(ids(build(n))) > args.filler_tokens and n > 1: n -= 1
    return build(n)

def flush(tag, count):
    for f in range(count): run("filler", make_filler(tag, f), {"phase": f"filler-{tag}", "filler": f})

print(f"# base {base}", flush=True)
recs = []
for d in docs:                                            # store pass: question A
    r = run("doc", d["msgs"][0], {"phase": "store", "doc": d["name"], "in_band": d["in_band"], "L_local": d["L"][0], "P": d["P"],
                                  "B0": d["B0"], "shared_prefix_end": d["P"]}); recs.append((d, "store", r))
flush("A", args.filler)
for d in docs:                                            # revisit B
    r = run("doc", d["msgs"][1], {"phase": "revisitB", "doc": d["name"], "in_band": d["in_band"], "L_local": d["L"][1], "P": d["P"],
                                  "B0": d["B0"], "shared_prefix_end": d["P"], "pred_hit_tokens": d["pred_B"]}); recs.append((d, "revisitB", r))
flush("B", args.filler2)
for d in docs:                                            # revisit C
    r = run("doc", d["msgs"][2], {"phase": "revisitC", "doc": d["name"], "in_band": d["in_band"], "L_local": d["L"][2], "P": d["P"],
                                  "B0": d["B0"], "shared_prefix_end": d["P"], "pred_hit_tokens": d["pred_C"]}); recs.append((d, "revisitC", r))

# ------------------------------------------------------------------ verdict
print("\n=== VERDICT (hit = external-hit tokens > 0 on the revisit) ===")
print(f"{'phase':9s} {'band':4s} hits/n   rate    mean ttft  predicted")
for ph in ("revisitB", "revisitC"):
    for band in (True, False):
        rs = [(d, r) for d, p, r in recs if p == ph and d["in_band"] == band]
        if not rs: continue
        h = sum(1 for d, r in rs if r["ext_hit_tokens"] > 0)
        pr = sum(1 for d, r in rs if (d["pred_B"] if ph == "revisitB" else d["pred_C"]) > 0)
        print(f"{ph:9s} {'IN' if band else 'OUT':4s} {h}/{len(rs)}    {100*h/len(rs):5.1f}%  {sum(r['ttft_s'] or 0 for d, r in rs)/len(rs):7.2f}s   predicted {pr}/{len(rs)}")
print("Band rule confirmed if IN ~0% and OUT ~100% on BOTH revisits (permanence: C still misses after B re-stored).")
print(f"wall estimate was {est/60:.1f} min; results in {args.out}")
