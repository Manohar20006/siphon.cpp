#!/usr/bin/env python3
"""compact_test.py — verify compact image-prefill: speed vs text + accuracy."""
import time, json, base64, io, urllib.request
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


def img_content(imgs):
    c = []
    for im in imgs:
        b = io.BytesIO(); im.save(b, format="PNG")
        c.append({"type": "image_url",
                  "image_url": {"url": "data:image/png;base64," +
                                base64.b64encode(b.getvalue()).decode()}})
    return c


def prefill(imgs, nonce):
    c = img_content(imgs)
    c.append({"type": "text", "text": f"Say nothing. [{nonce}]"})
    return post({"model": MODEL, "messages": [{"role": "user", "content": c}],
                 "max_tokens": 1, "temperature": 0.0})


# ---- 1. prefill speed: compact full doc vs text ----
PRE = 20000
text = DOC[:PRE]
imgs_c, meta_c = P.render_pages(text, font_size=12, page_w=1400, page_h=1400,
                               mode="gray", compact=True)
pc = prefill(imgs_c, f"c{time.time_ns():x}")
pt = post({"model": MODEL, "messages": [{"role": "user",
            "content": text + f"\n[{time.time_ns():x}]"}], "max_tokens": 1,
           "temperature": 0.0})
print(f"TEXT   : pages=text tok={pt['prompt_tokens']} time={pt['time_s']}s")
print(f"COMPACT: pages={meta_c['n_pages']} tok={pc['prompt_tokens']} time={pc['time_s']}s")
if pt["prompt_tokens"]:
    print(f"  compact tokens = {100*pc['prompt_tokens']/pt['prompt_tokens']:.0f}% of text; "
          f"speed = {pt['time_s']/max(0.1,pc['time_s']):.1f}x")

# ---- 2. accuracy: transcribe a compact sample, literal \n ----
SAMP = 1500
samp = DOC[:SAMP]
gold = samp.replace("\t", "\\t").replace("\n", "\\n")  # expected compact form
imgs_s, _ = P.render_pages(samp, font_size=12, page_w=1400, page_h=1400,
                           mode="gray", compact=True)
c = img_content(imgs_s)
c.append({"type": "text", "text":
          "This image shows text where the two characters \\n mean a newline and "
          "\\t means a tab. Transcribe it EXACTLY as the literal characters you see "
          "(so a line break in the source appears as backslash-n, not an actual new "
          "line). Output only the transcribed text, no commentary."})
acc = post({"model": MODEL, "messages": [{"role": "user", "content": c}],
            "max_tokens": 2500, "temperature": 0.0})
got = acc["text"].replace("\r\n", "\n").strip("\n")
cer = P.levenshtein(gold, got) / max(1, len(gold))
exact = 1 if gold == got else 0
print(f"\nACCURACY (compact sample {len(gold)} chars):")
print(f"  char-exact={exact}  CER={100*cer:.1f}%")
print(f"  gold[:90]={gold[:90]!r}")
print(f"  got [:90]={got[:90]!r}")
