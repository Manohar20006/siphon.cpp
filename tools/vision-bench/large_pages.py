#!/usr/bin/env python3
"""large_pages.py — font 16 at increasing page size: tokens/time/CER."""
import time, json, base64, io, urllib.request
from pathlib import Path
import prefill as P

SERVER = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.6"
DOC = Path("/home/manohar/Desktop/siphon.cpp/siphon-deep-dive.md").read_text()
PRE = 20000
SAMP = 1500


def post(p):
    r = urllib.request.Request(SERVER, data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    t = time.time(); resp = json.loads(urllib.request.urlopen(r, timeout=600).read())
    return round(time.time() - t, 2), resp.get("usage", {}).get("prompt_tokens"), \
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
              "Transcribe EXACTLY as literal characters (backslash-n). Output only text."})
    return post({"model": MODEL, "messages": [{"role": "user", "content": c}],
                 "max_tokens": 2500, "temperature": 0.0})


text = DOC[:PRE]
gold = DOC[:SAMP].replace("\t", "\\t").replace("\n", "\\n")
_, ttok, _ = post({"model": MODEL, "messages": [{"role": "user",
              "content": text + f"\n{time.time_ns():x}"}], "max_tokens": 1, "temperature": 0.0})
print(f"TEXT baseline tok={ttok}\n", flush=True)

for ph in (720, 1400, 2000):
    imgs, meta = P.render_pages(text, font_size=16, page_w=1400, page_h=ph,
                               mode="gray", compact=True)
    dt, tok, _ = prefill(imgs)
    simg, _ = P.render_pages(DOC[:SAMP], font_size=16, page_w=1400, page_h=ph,
                             mode="gray", compact=True)
    _, _, got = trans(simg)
    g = got.replace("\r\n", "\n").strip("\n")
    cer = P.levenshtein(gold, g) / max(1, len(gold))
    print(f"f16 page_h={ph}: pages={meta['n_pages']} tok={tok} "
          f"({100*tok//max(1,ttok)}% of text) time={dt}s CER={100*cer:.0f}%", flush=True)
