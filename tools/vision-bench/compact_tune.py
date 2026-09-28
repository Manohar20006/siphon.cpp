#!/usr/bin/env python3
"""compact_tune.py — find a compact config that is (a) faster in time, (b) readable."""
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


def img_content(imgs):
    c = []
    for im in imgs:
        b = io.BytesIO(); im.save(b, format="PNG")
        c.append({"type": "image_url",
                  "image_url": {"url": "data:image/png;base64," +
                                base64.b64encode(b.getvalue()).decode()}})
    return c


def prefill(imgs):
    c = img_content(imgs); c.append({"type": "text", "text": f"x{time.time_ns():x}"})
    return post({"model": MODEL, "messages": [{"role": "user", "content": c}],
                 "max_tokens": 1, "temperature": 0.0})


def transcribe(imgs):
    c = img_content(imgs)
    c.append({"type": "text", "text":
              "This image encodes text where \\n means newline and \\t means tab. "
              "Transcribe EXACTLY as literal characters (backslash-n, not real newline). "
              "Output only the text."})
    return post({"model": MODEL, "messages": [{"role": "user", "content": c}],
                 "max_tokens": 2500, "temperature": 0.0})


def trimmed(text):
    # keep line structure but drop blank lines + strip indentation (saves space,
    # preserves the line-by-line layout the model reads well)
    out = []
    for ln in text.split("\n"):
        s = ln.strip()
        if s:
            out.append(s)
    return "\n".join(out)


SAMP = 1500
PRE = 20000
text = DOC[:PRE]
gold_compact = text[:SAMP].replace("\t", "\\t").replace("\n", "\\n")

# text baseline (cached likely) for token reference
_, ttok, _ = post({"model": MODEL, "messages": [{"role": "user",
                  "content": text + f"\n{time.time_ns():x}"}], "max_tokens": 1, "temperature": 0.0})
print(f"TEXT baseline tok={ttok}\n")

configs = [
    ("compact f12 1400x720",  lambda t: P.render_pages(t, font_size=12, page_w=1400, page_h=720, mode="gray", compact=True)),
    ("compact f16 1400x720",  lambda t: P.render_pages(t, font_size=16, page_w=1400, page_h=720, mode="gray", compact=True)),
    ("trimmed f12 1400x720",  lambda t: P.render_pages(trimmed(t), font_size=12, page_w=1400, page_h=720, mode="gray", compact=False)),
]
for name, fn in configs:
    imgs, meta = fn(text)
    dt, tok, _ = prefill(imgs)
    # accuracy on sample
    simgs, _ = fn(text[:SAMP])
    _, _, got = transcribe(simgs)
    gold = gold_compact if "compact" in name else trimmed(text[:SAMP])
    cer = P.levenshtein(gold, got.replace("\r\n", "\n").strip("\n")) / max(1, len(gold))
    exact = 1 if gold == got.replace("\r\n", "\n").strip("\n") else 0
    print(f"{name:22} pages={meta['n_pages']} tok={tok} t={dt}s "
          f"tok%={100*tok//max(1,ttok)}% CER={100*cer:.0f}% exact={exact}")
