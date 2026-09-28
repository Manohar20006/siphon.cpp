#!/usr/bin/env python3
"""
nl_test.py — does image-prefill work for REAL natural-language documents?

Uses siphon-deep-dive.md (prose). For each render config measures:
  - pure prefill time (max_tokens=1) vs TEXT baseline
  - transcription accuracy (line-exact %, mean CER) on a prose excerpt
"""

import time
import urllib.request
import json
import base64
import io
from pathlib import Path

from PIL import Image
import prefill as P

SERVER = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.6"
DOC = Path("/home/manohar/Desktop/siphon.cpp/siphon-deep-dive.md").read_text()


def post(payload):
    req = urllib.request.Request(SERVER, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    u = resp.get("usage", {})
    return {"time_s": round(dt, 2), "prompt_tokens": u.get("prompt_tokens"),
            "text": resp["choices"][0]["message"]["content"]}


def prefill_image(imgs, nonce):
    content = []
    for img in imgs:
        buf = io.BytesIO(); img.save(buf, format="PNG")
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," +
                                      base64.b64encode(buf.getvalue()).decode()}})
    content.append({"type": "text", "text": f"Say nothing. [{nonce}]"})
    return post({"model": MODEL, "messages": [{"role": "user", "content": content}],
                 "max_tokens": 1, "temperature": 0.0})


def prefill_text(text, nonce):
    return post({"model": MODEL, "messages": [{"role": "user",
               "content": text + f"\n[{nonce}]"}], "max_tokens": 1, "temperature": 0.0})


def transcribe(imgs):
    content = []
    for img in imgs:
        buf = io.BytesIO(); img.save(buf, format="PNG")
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," +
                                      base64.b64encode(buf.getvalue()).decode()}})
    content.append({"type": "text", "text":
                    "Transcribe the image EXACTLY, line by line, preserving every "
                    "character including spaces and blank lines. Output only the raw "
                    "text, no commentary, no markdown fences."})
    return post({"model": MODEL, "messages": [{"role": "user", "content": content}],
                 "max_tokens": 1000, "temperature": 0.0})


def score(gold_lines, got_text):
    got = got_text.replace("\r\n", "\n").strip("\n").split("\n")
    n = len(gold_lines); m = len(got)
    exact = sum(1 for i in range(max(n, m))
                if (gold_lines[i] if i < n else "") == (got[i] if i < m else ""))
    cers = [P.levenshtein(gold_lines[i] if i < n else "", got[i] if i < m else "")
            / max(1, len(gold_lines[i] if i < n else "")) for i in range(max(n, m))]
    return round(100 * exact / max(1, n), 1), round(100 * sum(cers) / max(1, len(cers)), 1), n, m


PRE = 20000   # chars for prefill-time test
ACC = 900     # chars for accuracy test
PW, PH = 1400, 720

print(f"DOC chars={len(DOC)}; prefill-test slice={PRE}; accuracy slice={ACC}", flush=True)

t_pre = prefill_text(DOC[:PRE], f"t{time.time_ns():x}")
print(f"\nTEXT baseline: prefill={t_pre['time_s']}s tok={t_pre['prompt_tokens']}", flush=True)

print(f"\n{'config':14} {'pg':>3} {'imgTok':>6} {'imgT':>6} {'x':>4} {'lineOK%':>7} {'CER%':>5}", flush=True)
print("-" * 52, flush=True)
for font in (12, 14):
    for mode in ("gray", "bw"):
        imgs, meta = P.render_pages(DOC[:PRE], font_size=font, page_w=PW, page_h=PH, mode=mode)
        pi = prefill_image(imgs, f"i{time.time_ns():x}")
        speed = t_pre["time_s"] / max(0.1, pi["time_s"])
        aimgs, ameta = P.render_pages(DOC[:ACC], font_size=font, page_w=PW, page_h=PH, mode=mode)
        acc = transcribe(aimgs)
        gold = ameta["pages"][0]["lines_text"]
        le, cer, n, m = score(gold, acc["text"])
        print(f"f{font}_{mode:4} {meta['n_pages']:>3} {pi['prompt_tokens']:>6} "
              f"{pi['time_s']:>6} {speed:>4.1f} {le:>7} {cer:>5}", flush=True)
