#!/usr/bin/env python3
"""compact_tune2.py — after pagination fix: measure compact tokens/time + accuracy."""
import time, json, base64, io, urllib.request
from pathlib import Path
import prefill as P

SERVER = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.6"
DOC = Path("/home/manohar/Desktop/siphon.cpp/siphon-deep-dive.md").read_text()


def post(payload):
    req = urllib.request.Request(SERVER, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return round(time.time() - t0, 2), resp.get("usage", {}).get("prompt_tokens"), \
        resp["choices"][0]["message"]["content"]


def ic(imgs):
    c = []
    for im in imgs:
        b = io.BytesIO(); im.save(b, format="PNG")
        c.append({"type": "image_url",
                  "image_url": {"url": "data:image/png;base64," +
                                base64.b64encode(b.getvalue()).decode()}})
    return c


def prefill(imgs):
    c = ic(imgs); c.append({"type": "text", "text": f"x{time.time_ns():x}"})
    return post({"model": MODEL, "messages": [{"role": "user", "content": c}],
                 "max_tokens": 1, "temperature": 0.0})


def trans(imgs):
    c = ic(imgs)
    c.append({"type": "text", "text":
              "Image encodes text where \\n means newline and \\t means tab. "
              "Transcribe EXACTLY as literal characters (backslash-n, not a real newline). "
              "Output only the text."})
    return post({"model": MODEL, "messages": [{"role": "user", "content": c}],
                 "max_tokens": 2500, "temperature": 0.0})


PRE = 20000
SAMP = 1500
text = DOC[:PRE]
_, ttok, _ = post({"model": MODEL, "messages": [{"role": "user",
              "content": text + f"\n{time.time_ns():x}"}], "max_tokens": 1, "temperature": 0.0})
print("TEXT baseline tok=", ttok, flush=True)
for name, fn in [
    ("compact f12", lambda t: P.render_pages(t, font_size=12, page_w=1400, page_h=720, mode="gray", compact=True)),
    ("compact f14", lambda t: P.render_pages(t, font_size=14, page_w=1400, page_h=720, mode="gray", compact=True)),
    ("compact f16", lambda t: P.render_pages(t, font_size=16, page_w=1400, page_h=720, mode="gray", compact=True)),
]:
    imgs, meta = fn(text)
    dt, tok, _ = prefill(imgs)
    simg, _ = fn(text[:SAMP])
    _, _, got = trans(simg)
    gold = text[:SAMP].replace("\t", "\\t").replace("\n", "\\n")
    g = got.replace("\r\n", "\n").strip("\n")
    cer = P.levenshtein(gold, g) / max(1, len(gold))
    print(f"{name:12} pages={meta['n_pages']} tok={tok} t={dt}s "
          f"tok%={100*tok//max(1,ttok)}% CER={100*cer:.0f}%", flush=True)
