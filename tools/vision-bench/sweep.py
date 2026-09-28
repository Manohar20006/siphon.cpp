#!/usr/bin/env python3
"""
sweep.py — map the (prefill-speed, transcription-accuracy) frontier for
image-prefill across a grid of render configs, vs a text baseline.

For each corpus + config we measure:
  - PURE PREFILL TIME (max_tokens=1): image vs text  -> speedup
  - TRANSCRIPTION ACCURACY: render a ~1.8k-char sample, ask the model to
    transcribe verbatim, score line-exact% and mean per-line CER.

Output: a table per corpus. No hardcoded claims.
"""

import argparse
import base64
import io
import json
import random
import time
import urllib.request
from pathlib import Path

from PIL import Image

import prefill as P

SERVER = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.6"
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------

def gen_corpora():
    rng = random.Random(7)
    alpha = "abcdefghijklmnopqrstuvwxyz0123456789 "
    # dense: long lines ~180 chars
    dense = "\n".join("".join(rng.choice(alpha) for _ in range(180)) for _ in range(400))
    # sparse: code-like, short indented lines
    sparse = "\n".join("    " * (i % 5) + "x = " + str(i) + "  # " + rng.choice("abc") * 6
                       for i in range(1500))
    # mixed: terminal-ish (some long lines, some short, some blank)
    mixed_lines = []
    for i in range(600):
        if i % 7 == 0:
            mixed_lines.append("")
        elif i % 3 == 0:
            mixed_lines.append("ERROR: " + "".join(rng.choice(alpha) for _ in range(150)))
        else:
            mixed_lines.append("ok " + str(i))
    mixed = "\n".join(mixed_lines)
    (HERE / "corpus").mkdir(exist_ok=True)
    for name, txt in (("dense", dense), ("sparse", sparse), ("mixed", mixed)):
        (HERE / "corpus" / f"sweep_{name}.txt").write_text(txt)
    return {"dense": dense, "sparse": sparse, "mixed": mixed}


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------

def post(payload):
    req = urllib.request.Request(SERVER, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    u = resp.get("usage", {})
    return {"time_s": round(dt, 2), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "text": resp["choices"][0]["message"]["content"],
            "cached": u.get("prompt_tokens_details", {}).get("cached_tokens")}


def prefill_image(imgs, nonce):
    content = []
    for img in imgs:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": f"Say nothing. [{nonce}]"})
    return post({"model": MODEL, "messages": [{"role": "user", "content": content}],
                 "max_tokens": 1, "temperature": 0.0})


def prefill_text(text, nonce):
    return post({"model": MODEL,
                 "messages": [{"role": "user",
                               "content": text + f"\n[{nonce}]"}],
                 "max_tokens": 1, "temperature": 0.0})


def transcribe(imgs):
    content = []
    for img in imgs:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text":
                    "Transcribe the image EXACTLY, line by line, preserving every "
                    "character including spaces and blank lines. Output only the raw "
                    "text, no commentary, no markdown fences."})
    return post({"model": MODEL, "messages": [{"role": "user", "content": content}],
                 "max_tokens": 2000, "temperature": 0.0})


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def normalize(s):
    return s.replace("\r\n", "\n").strip("\n")


def score_transcription(gold_lines, got_text):
    got = normalize(got_text).split("\n")
    n = len(gold_lines)
    m = len(got)
    exact = 0
    cers = []
    for i in range(max(n, m)):
        g = gold_lines[i] if i < n else ""
        h = got[i] if i < m else ""
        if g == h:
            exact += 1
        cers.append(P.levenshtein(g, h) / max(1, len(g)))
    return {"line_exact": round(100 * exact / max(1, n), 1),
            "mean_cer": round(100 * sum(cers) / max(1, len(cers)), 1),
            "gold_lines": n, "got_lines": m}


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill-size", type=int, default=12000)
    ap.add_argument("--acc-size", type=int, default=1800)
    ap.add_argument("--fonts", type=int, nargs="+", default=[8, 10, 12, 14])
    ap.add_argument("--modes", nargs="+", default=["gray", "bw"])
    ap.add_argument("--page-w", type=int, default=1400)
    ap.add_argument("--page-h", type=int, default=720)
    ap.add_argument("--corpora", nargs="+", default=["dense", "sparse", "mixed"])
    args = ap.parse_args()

    corp = gen_corpora()

    for cname in args.corpora:
        text = corp[cname]
        print(f"\n########## CORPUS: {cname} ({len(text)} chars) ##########")
        # text baseline (prefill time)
        t_pre = prefill_text(text[:args.prefill_size], f"t{time.time_ns():x}")
        # text baseline (accuracy on small sample)
        acc_sample = text[:args.acc_size]
        t_imgs, _ = P.render_pages(acc_sample, font_size=12, page_w=args.page_w,
                                   page_h=args.page_h, mode="gray")
        t_acc = transcribe(t_imgs)
        gold0 = _gold_lines(acc_sample, args.page_w, args.page_h, 12)
        t_score = score_transcription(gold0, t_acc["text"])
        print(f"TEXT baseline: prefill={t_pre['time_s']}s tok={t_pre['prompt_tokens']} | "
              f"acc line_exact={t_score['line_exact']}% cer={t_score['mean_cer']}%")

        print(f"{'config':22} {'pg':>3} {'imgTok':>6} {'imgT':>6} {'x':>4} "
              f"{'lineOK%':>7} {'CER%':>5}")
        print("-" * 60)
        for font in args.fonts:
            for mode in args.modes:
                cfg = f"f{font}_{mode}"
                # prefill time (large)
                imgs, meta = P.render_pages(text[:args.prefill_size], font_size=font,
                                            page_w=args.page_w, page_h=args.page_h,
                                            mode=mode)
                pi = prefill_image(imgs, f"i{time.time_ns():x}")
                speed = t_pre["time_s"] / max(0.1, pi["time_s"])
                # accuracy (small)
                aimgs, ameta = P.render_pages(acc_sample, font_size=font,
                                              page_w=args.page_w, page_h=args.page_h,
                                              mode=mode)
                acc = transcribe(aimgs)
                gold = _gold_lines(acc_sample, args.page_w, args.page_h, font)
                sc = score_transcription(gold, acc["text"])
                print(f"{cfg:22} {meta['n_pages']:>3} {pi['prompt_tokens']:>6} "
                      f"{pi['time_s']:>6} {speed:>4.1f} {sc['line_exact']:>7} "
                      f"{sc['mean_cer']:>5}")


def _gold_lines(sample, pw, ph, font):
    _, meta = P.render_pages(sample, font_size=font, page_w=pw, page_h=ph, mode="gray")
    return meta["pages"][0]["lines_text"]


if __name__ == "__main__":
    main()
