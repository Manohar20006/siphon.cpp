#!/usr/bin/env python3
"""
bench_prefill.py — compare TEXT vs IMAGE pure-prefill time + tokens.

Isolates prefill by using max_tokens=1 (so reported time ~= prefill only).
Measures, for the same source text:
  - text path: send raw text as a chat message
  - image path: render to pages, send as images
Reports prompt_tokens and prefill seconds for each, cold (unique nonce to
avoid KV cache hits).
"""

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image

import prefill as P

SERVER = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.6"


def query_text(text, nonce):
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": text + f"\n[nonce {nonce}]"}],
               "max_tokens": 1, "temperature": 0.0}
    return _post(payload)


def query_image(imgs, nonce):
    content = []
    for img in imgs:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": f"Transcribe nothing. [nonce {nonce}]"})
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
               "max_tokens": 1, "temperature": 0.0}
    return _post(payload)


def _post(payload):
    req = urllib.request.Request(SERVFR if False else SERVER,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    u = resp.get("usage", {})
    return {"time_s": round(dt, 2), "prompt_tokens": u.get("prompt_tokens"),
            "cached": u.get("prompt_tokens_details", {}).get("cached_tokens")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--chars", type=int, default=0)
    ap.add_argument("--font", type=int, default=10)
    ap.add_argument("--page-w", type=int, default=1400)
    ap.add_argument("--page-h", type=int, default=720)
    ap.add_argument("--mode", choices=["gray", "bw"], default="gray")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    text = Path(args.file).read_text(errors="replace")
    if args.chars:
        text = text[:args.chars]

    imgs, meta = P.render_pages(text, page_w=args.page_w, page_h=args.page_h,
                               font_size=args.font, mode=args.mode)
    print(f"source: {meta['total_chars']} chars -> {meta['n_pages']} image pages "
          f"(font {args.font}, {args.page_w}x{args.page_h}, {args.mode})")

    def avg(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 2)

    t_rows, i_rows = [], []
    for r in range(args.reps):
        nonce = f"{time.time_ns():x}{r}"
        t_rows.append(query_text(text, nonce))
        i_rows.append(query_image(imgs, nonce))
        print(f"rep{r}: TEXT tok={t_rows[-1]['prompt_tokens']} t={t_rows[-1]['time_s']}s "
              f"| IMG tok={i_rows[-1]['prompt_tokens']} t={i_rows[-1]['time_s']}s")

    t_tok, t_t = avg(t_rows, "prompt_tokens"), avg(t_rows, "time_s")
    i_tok, i_t = avg(i_rows, "prompt_tokens"), avg(i_rows, "time_s")
    print("\n=== PURE PREFILL (max_tokens=1), averaged ===")
    print(f"TEXT : tokens={t_tok}  time={t_t}s")
    print(f"IMAGE: tokens={i_tok}  time={i_t}s")
    if t_t > 0:
        print(f"IMAGE is {t_t/i_t:.2f}x TEXT time  |  tokens {100*i_tok/max(1,t_tok):.0f}% of text")
    # text baseline chars/4
    print(f"text chars/4 baseline = {meta['total_chars']//4}")


if __name__ == "__main__":
    main()
