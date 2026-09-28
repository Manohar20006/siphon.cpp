#!/usr/bin/env python3
"""
vision-bench: benchmark qwen3.6's vision encoder at reading dense rendered text.

Measures the accuracy-vs-speed tradeoff of the "text-as-image prefill" idea
across a matrix of rendering configurations, against known ground truth.

Usage:
    python3 vision_bench.py --probe font=11 line=1.15 mode=gray size=48
    python3 vision_bench.py --configs "font=18,line=1.15,gray" "font=11,line=1.15,gray" "font=8,line=1.15,gray"
    python3 vision_bench.py --configs all
    python3 vision_bench.py --sizes 8 32 --configs all

Output: results/bench-<timestamp>.json + a printed comparison table.
"""

import argparse
import base64
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
RESULTS = HERE / "results"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SERVER = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.6"

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# corpus generation (deterministic)
# ---------------------------------------------------------------------------

def gen_doc(n_blocks: int, seed: int) -> dict:
    rng = random.Random(seed)
    lines = []
    truth = {"count": n_blocks, "blocks": {}}
    for i in range(n_blocks):
        checksum = f"{rng.randrange(16**64):064x}"
        max_retries = rng.randint(2, 9)
        ip_pool = [f"10.1.{i}.{j}" for j in range(1, 5)]
        token_rate = rng.randint(100, 9999)
        block = (
            f"server-{i} config (deploy id a7f3{i:04d}9c2)\n"
            f"api.endpoint=https://svc-{i}.internal.example.com/v2\n"
            f"max_retries={max_retries}\n"
            f"ip_pool={ip_pool}\n"
            f"checksum_sha256={checksum}\n"
            f"token_rate={token_rate}/min\n"
        )
        lines.append(block)
        truth["blocks"][str(i)] = {
            "checksum": checksum,
            "max_retries": max_retries,
            "ip_pool": ip_pool,
            "token_rate": token_rate,
        }
    return {"n_blocks": n_blocks, "text": "".join(lines), "truth": truth}


def load_corpus(sizes, variants=1):
    docs = []
    for n in sizes:
        for v in range(variants):
            p = CORPUS / f"doc_{n}_v{v}.json"
            if not p.exists():
                p.write_text(json.dumps(gen_doc(n, seed=1000 + n + v * 100), indent=1))
            docs.append(json.loads(p.read_text()))
    return docs


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_page(text, out, font_size, width=1400, max_h=720, line_mult=1.15,
                mode="gray", page_no=0):
    font = ImageFont.truetype(FONT_PATH, font_size)
    line_h = int(font_size * line_mult)
    col_w = font.getbbox("W")[2]
    cols = max(1, (width - 24) // col_w)
    lines = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        while len(raw) > cols:
            lines.append(raw[:cols])
            raw = raw[cols:]
        lines.append(raw)
    n_fit = max(1, (max_h - 24) // line_h)
    start = page_no * n_fit
    lines = lines[start:start + n_fit]
    h = 24 + len(lines) * line_h
    img = Image.new("RGB", (width, h), "white")
    d = ImageDraw.Draw(img)
    y = 12
    for ln in lines:
        d.text((12, y), ln, fill="black", font=font)
        y += line_h
    if mode == "bw":
        gray = np.asarray(img.convert("L"))
        img = Image.fromarray(np.where(gray > 127, 255, 0).astype("uint8")).convert("RGB")
    img.save(out)
    return {"w": width, "h": h, "cols": cols, "lines": len(lines),
            "chars": sum(len(l) for l in lines), "font": font_size,
            "line_mult": line_mult, "mode": mode, "page_no": page_no}


def render_pages(text, out_prefix, cfg, width=1400, max_h=720):
    """Render text into one or more capped pages. Returns list of page metas."""
    font = ImageFont.truetype(FONT_PATH, cfg["font_size"])
    line_h = int(cfg["font_size"] * cfg["line_mult"])
    col_w = font.getbbox("W")[2]
    cols = max(1, (width - 24) // col_w)
    lines = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        while len(raw) > cols:
            lines.append(raw[:cols])
            raw = raw[cols:]
        lines.append(raw)
    n_fit = max(1, (max_h - 24) // line_h)
    n_pages = max(1, (len(lines) + n_fit - 1) // n_fit)
    pages = []
    for p in range(n_pages):
        out = f"{out_prefix}_p{p}.png"
        meta = render_page(text, out, page_no=p, **cfg)
        pages.append({"path": out, "meta": meta})
    return pages


def parse_config(cfg: str) -> dict:
    d = {}
    for part in cfg.split(","):
        k, _, v = part.partition("=")
        if k == "font":
            d["font_size"] = int(v)
        elif k == "line":
            d["line_mult"] = float(v)
        elif k in ("gray", "bw"):
            d["mode"] = k
        else:
            d[k] = v
    d.setdefault("font_size", 11)
    d.setdefault("line_mult", 1.15)
    d.setdefault("mode", "gray")
    return d


def cfg_key(cfg: dict) -> str:
    return f"font{cfg['font_size']}_l{cfg['line_mult']}_{cfg['mode']}"


# ---------------------------------------------------------------------------
# querying the server
# ---------------------------------------------------------------------------

def build_question(doc):
    b = doc["truth"]["blocks"]
    probes = ["0", "3", "7", "13", "27"]
    probes = [p for p in probes if p in b][:3]
    spec = []
    spec.append('"count": <number of server blocks>')
    for p in probes:
        spec.append(f'"server_{p}_checksum": <the checksum_sha256 hex of server-{p}>')
        spec.append(f'"server_{p}_max_retries": <the max_retries of server-{p}>')
    q = ("Read this rendered document. It contains several server config blocks. "
         "Answer EXACTLY in JSON with keys: " + ", ".join(spec) + ". "
         "Do not guess; if you are unsure, put null. Output only the JSON.")
    return q, probes


def extract_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\})", text, re.S)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"no JSON in reply: {text[:120]!r}")


def query_image(img_paths, question, max_tokens=600):
    content = []
    for p in img_paths:
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": question})
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = urllib.request.Request(SERVER, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    usage = resp.get("usage", {})
    return {
        "text": resp["choices"][0]["message"]["content"],
        "time_s": round(dt, 1),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(doc, probes, reply, parsed):
    truth = doc["truth"]
    results = {}
    n_probes = 0
    n_ok = 0

    def norm(s):
        return re.sub(r"\s+", "", str(s)).lower()

    def check(name, expected, got):
        nonlocal n_probes, n_ok
        n_probes += 1
        ng = norm(got)
        ne = norm(expected)
        ok = ne == ng
        if ok:
            n_ok += 1
        lev = levenshtein(ne, ng)
        results[name] = {"expected": expected, "got": got, "ok": ok, "lev": lev}
        return ok

    check("count", str(truth["count"]), str(parsed.get("count")))
    for p in probes:
        b = truth["blocks"][p]
        check(f"server_{p}_checksum", b["checksum"], str(parsed.get(f"server_{p}_checksum", "")))
        check(f"server_{p}_max_retries", str(b["max_retries"]), str(parsed.get(f"server_{p}_max_retries", "")))
    return {"ok": n_ok, "total": n_probes, "lev_total": sum(r["lev"] for r in results.values()),
            "details": results}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[48])
    ap.add_argument("--variants", type=int, default=1,
                    help="number of distinct docs per size (different content, same size)")
    ap.add_argument("--configs", nargs="+", default=["font=11,line=1.15,gray"])
    ap.add_argument("--probe", default=None,
                    help="single config, prints full details")
    ap.add_argument("--reps", type=int, default=1,
                    help="repeat each config/doc N times and average (default 1)")
    ap.add_argument("--fresh", action="store_true",
                    help="append a per-rep nonce so the server cannot reuse cached KV; "
                         "timing then reflects a real (cold) prefill")
    ap.add_argument("--cold", action="store_true",
                    help="regenerate the corpus with a fresh random seed every rep, so "
                         "no KV is ever cached and timing is a true cold prefill")
    args = ap.parse_args()

    if args.probe:
        cfg = parse_config(args.probe)
        docs = load_corpus(args.sizes, args.variants)
        doc = docs[0]
        img = RESULTS / "probe.png"
        pages = render_pages(doc["text"], str(img).rsplit(".", 1)[0], cfg)
        print("render pages:", [p["meta"] for p in pages])
        q, probes = build_question(doc)
        out = query_image([p["path"] for p in pages], q)
        print(f"time={out['time_s']}s prompt_tokens={out['prompt_tokens']} "
              f"cached={out['cached_tokens']} completion={out['completion_tokens']}")
        print(out["text"])
        try:
            parsed = extract_json(out["text"])
            print("parsed:", json.dumps(parsed, indent=1))
            print("score:", score(doc, probes, out["text"], parsed))
        except ValueError as e:
            print("PARSE FAIL:", e)
        return

    cfgs = [parse_config(c) for c in args.configs]
    if any(c.get("all") for c in cfgs):
        cfgs = [{"font_size": f, "line_mult": l, "mode": m}
                for f in (18, 14, 11, 8)
                for l in (1.0, 1.3)
                for m in ("gray", "bw")]

    docs = load_corpus(args.sizes, args.variants)
    rows = []
    for cfg in cfgs:
        for d_idx, doc in enumerate(docs):
            key = cfg_key(cfg)
            img = RESULTS / f"{key}_n{doc['n_blocks']}"
            pages = render_pages(doc["text"], str(img), cfg)
            q, probes = build_question(doc)
            rep_results = []
            for rep in range(args.reps):
                cur_doc = doc
                cur_pages = pages
                cur_q, cur_probes = q, probes
                if args.cold:
                    cur_doc = gen_doc(doc["n_blocks"],
                                      seed=50000 + doc["n_blocks"] * 1000 + d_idx * 10 + rep)
                    cur_pages = render_pages(cur_doc["text"],
                                             f"{RESULTS / (key + '_cold' + str(d_idx) + '_' + str(rep))}", cfg)
                    cur_q, cur_probes = build_question(cur_doc)
                qq = cur_q
                if args.fresh:
                    qq = cur_q + f"\n\n(ignore the marker {time.time_ns():x})"
                out = query_image([p["path"] for p in cur_pages], qq)
                try:
                    parsed = extract_json(out["text"])
                    s = score(cur_doc, cur_probes, out["text"], parsed)
                except ValueError:
                    s = {"ok": 0, "total": 0, "parse_fail": True}
                rep_results.append({**out, "ok": s.get("ok"), "total": s.get("total"),
                                    "lev_total": s.get("lev_total"),
                                    "parse_fail": s.get("parse_fail", False)})
                print(f"[{key} n={doc['n_blocks']} rep{rep}] ok={s.get('ok')}/{s.get('total')} "
                      f"lev={s.get('lev_total')} time={out['time_s']}s toks={out['prompt_tokens']} "
                      f"{'PARSE_FAIL' if s.get('parse_fail') else ''}")
            reps = rep_results
            n_ok = sum(r["ok"] for r in reps if not r.get("parse_fail"))
            n_tot = sum(r["total"] for r in reps if not r.get("parse_fail"))
            levs = [r.get("lev_total", 0) for r in reps if not r.get("parse_fail")]
            mean_lev = round(sum(levs) / len(levs), 1) if levs else None
            rows.append({"config": key, "n_blocks": doc["n_blocks"],
                         "pages": [p["meta"] for p in pages], "reps": reps,
                         "ok_avg": round(n_ok / max(1, n_tot), 3),
                         "lev_avg": mean_lev,
                         "time_avg": round(sum(r["time_s"] for r in reps) / len(reps), 1),
                         "tokens_avg": round(sum(r["prompt_tokens"] or 0 for r in reps) / len(reps)),
                         "details": s.get("details"),
                         "reply": reps[-1]["text"]})

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS / f"bench-{ts}.json"
    out_path.write_text(json.dumps(rows, indent=1))
    print("\nWrote", out_path)
    print_table(rows)


def print_table(rows):
    print("\n=== SUMMARY (exact-match accuracy vs prefill speed) ===")
    hdr = f"{'config':26} {'n':>3} {'acc%':>5} {'lev':>5} {'time_s':>7} {'toks':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        accs = "PF" if all(x.get("parse_fail") for x in r["reps"]) else f"{100.0*r['ok_avg']:5.0f}"
        levs = "—" if r["lev_avg"] is None else f"{r['lev_avg']:5.1f}"
        print(f"{r['config']:26} {r['n_blocks']:>3} {accs:>5} {levs:>5} "
              f"{r['time_avg']:>7.1f} {r['tokens_avg']:>6}")


if __name__ == "__main__":
    main()