#!/usr/bin/env python3
"""
prefill.py — clean image-prefill renderer + honest probe.

Turns an arbitrary text blob into a set of images suitable for vision-prefill,
optimized for (a) minimal image tokens, (b) maximal readability, and
(c) pagination integrity (a logical line never splits across a page).

No accuracy claims are hardcoded here. Run the probe to measure.

Usage:
    python3 prefill.py probe --file PATH --chars 8000 --font 14 --page-w 2240 --page-h 1400
    python3 prefill.py probe --file PATH --font 18 --mode bw
"""

import argparse
import base64
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SERVER = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.6"


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def wrap_text(text, font, width, tab_width=4):
    """Split text into logical lines, each expanded for tabs and wrapped to
    `width` px into one or more visual lines. Returns list of blocks, where
    each block is a list of visual-line strings that must stay together."""
    col_w = font.getbbox("W")[2]
    cols = max(1, (width - 24) // col_w)
    blocks = []
    for raw in text.split("\n"):
        expanded = raw.replace("\t", " " * tab_width)
        if expanded == "":
            blocks.append([""])
            continue
        vis = []
        while len(expanded) > cols:
            vis.append(expanded[:cols])
            expanded = expanded[cols:]
        vis.append(expanded)
        blocks.append(vis)
    return blocks, cols, col_w


def render_pages(text, *, page_w=2240, page_h=1400, font_size=14,
                 line_mult=1.15, mode="gray", tab_width=4, compact=False):
    """Render `text` into one or more page images. A block (logical line +
    its continuations) is never split across a page. Returns (pages, meta)
    where pages is a list of PIL images.

    If `compact` is True, structural formatting is replaced with LLM-readable
    tokens (\\n for newline, \\t for tab) and the text is rendered as one
    continuous flow that fills each line to the full width — maximizing
    characters per page (and thus minimizing image tokens)."""
    if compact:
        text = text.replace("\t", "\\t").replace("\n", "\\n")
    font = ImageFont.truetype(FONT_PATH, font_size)
    line_h = int(font_size * line_mult)
    blocks, cols, col_w = wrap_text(text, font, page_w, tab_width)

    n_fit = max(1, (page_h - 24) // line_h)
    pages = []
    cur = []
    cur_lines = 0
    for block in blocks:
        if cur_lines + len(block) > n_fit:
            if cur:
                pages.append(cur)
                cur = []
                cur_lines = 0
            # a single block larger than a page must be split across pages
            if len(block) > n_fit:
                for i in range(0, len(block), n_fit):
                    pages.append(block[i:i + n_fit])
                cur = []
                cur_lines = 0
                continue
        cur.extend(block)
        cur_lines += len(block)
        if cur_lines >= n_fit:
            pages.append(cur)
            cur = []
            cur_lines = 0
    if cur:
        pages.append(cur)

    imgs = []
    meta = {"pages": [], "font": font_size, "line_mult": line_mult,
            "mode": mode, "cols": cols, "page_w": page_w, "page_h": page_h}
    for idx, lines in enumerate(pages):
        h = 24 + len(lines) * line_h
        img = Image.new("RGB", (page_w, h), "white")
        d = ImageDraw.Draw(img)
        y = 12
        for ln in lines:
            d.text((12, y), ln, fill="black", font=font)
            y += line_h
        if mode == "bw":
            import numpy as np
            g = np.asarray(img.convert("L"))
            img = Image.fromarray(np.where(g > 127, 255, 0).astype("uint8")).convert("RGB")
        imgs.append(img)
        meta["pages"].append({"page": idx, "lines": len(lines),
                              "chars": sum(len(l) for l in lines),
                              "lines_text": lines})
    meta["n_pages"] = len(imgs)
    meta["total_chars"] = len(text)
    return imgs, meta


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


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

def query_images(imgs, question, max_tokens=600):
    content = []
    for img in imgs:
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": question})
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(SERVER, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    u = resp.get("usage", {})
    return {"text": resp["choices"][0]["message"]["content"],
            "time_s": round(dt, 1),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "cached_tokens": u.get("prompt_tokens_details", {}).get("cached_tokens")}


# ---------------------------------------------------------------------------
# probe (honest measurement)
# ---------------------------------------------------------------------------

def extract_json(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\})", text, re.S)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"no JSON in reply: {text[:120]!r}")


def probe(args):
    if args.file:
        text = Path(args.file).read_text(errors="replace")
    else:
        # deterministic synthetic stand-in if no file given
        text = "".join(f"line {i:06d}: " + ("x" * 60) + "\n" for i in range(400))
    if args.chars:
        text = text[:args.chars]

    imgs, meta = render_pages(text, page_w=args.page_w, page_h=args.page_h,
                              font_size=args.font, line_mult=args.line_mult,
                              mode=args.mode)
    print("RENDER:", json.dumps({k: v for k, v in meta.items() if k != "pages"}, indent=1))
    print("PAGES:")
    for p in meta["pages"]:
        print("  ", p)

    # Honest readability check: ask for verbatim quotes of three specific
    # lines on page 1 (early / middle / late). Compare against ground truth.
    p1 = meta["pages"][0]
    n = p1["lines"]
    idxs = sorted(set([max(1, n // 10), max(1, n // 2), max(1, n - 2)]))
    q = ("This image shows rendered monospace text on a white background. "
         "Answer EXACTLY in JSON with keys "
         + ", ".join(f'"line_{i}": <verbatim text of 1-indexed line {i} on page 1, or null if unsure>'
                     for i in idxs)
         + ". Output only the JSON, no commentary.")
    out = query_images(imgs, q)
    print(f"\nQUERY: time={out['time_s']}s prompt_tokens={out['prompt_tokens']} "
          f"cached={out['cached_tokens']} completion={out['completion_tokens']}")
    print("REPLY:", out["text"])
    try:
        parsed = extract_json(out["text"])
        print("ACCURACY (char-exact match vs ground truth):")
        for i in idxs:
            truth = p1["lines_text"][i - 1]
            got = parsed.get(f"line_{i}")
            ok = (got is not None and got == truth)
            print(f"  line {i:3d}: {'OK ' if ok else 'BAD'} | truth={truth[:60]!r} got={str(got)[:60]!r}")
    except ValueError as e:
        print("PARSE FAIL:", e)

    # token/char economics (the whole point)
    pt = out["prompt_tokens"] or 0
    print(f"\nECONOMICS: {meta['total_chars']} chars -> {pt} img tokens "
          f"({pt/max(1,meta['total_chars']):.3f} tok/char). "
          f"Text baseline ~{meta['total_chars']//4} tokens "
          f"({100*pt/(meta['total_chars']//4):.0f}% of text cost).")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("probe")
    pp.add_argument("--file", default=None, help="text file to render")
    pp.add_argument("--chars", type=int, default=0, help="truncate source to N chars")
    pp.add_argument("--font", type=int, default=14)
    pp.add_argument("--line-mult", type=float, default=1.15)
    pp.add_argument("--page-w", type=int, default=2240)
    pp.add_argument("--page-h", type=int, default=1400)
    pp.add_argument("--mode", choices=["gray", "bw"], default="gray")
    pp.set_defaults(func=probe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
