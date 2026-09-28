#!/usr/bin/env python3
"""compact_iso.py — isolate accuracy cause: rendering density/format vs encoder."""
import time, json, base64, io, urllib.request
from pathlib import Path
import prefill as P

SERVER = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.6"
DOC = Path("/home/manohar/Desktop/siphon.cpp/siphon-deep-dive.md").read_text()
SAMP = 1500


def post(p):
    r = urllib.request.Request(SERVER, data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    t = time.time(); resp = json.loads(urllib.request.urlopen(r, timeout=600).read())
    return round(time.time() - t, 2), resp["choices"][0]["message"]["content"]


def ic(imgs):
    c = []
    for im in imgs:
        b = io.BytesIO(); im.save(b, format="PNG")
        c.append({"type": "image_url",
                  "image_url": {"url": "data:image/png;base64," +
                                base64.b64encode(b.getvalue()).decode()}})
    return c


def trans(imgs, marker=True):
    c = ic(imgs)
    if marker:
        c.append({"type": "text", "text":
                  "Image encodes text where \\n means newline and \\t means tab. "
                  "Transcribe EXACTLY as literal characters (backslash-n). Output only text."})
    else:
        c.append({"type": "text", "text":
                  "Transcribe the image EXACTLY, line by line, preserving every "
                  "character including spaces and blank lines. Output only the raw text."})
    return post({"model": MODEL, "messages": [{"role": "user", "content": c}],
                "max_tokens": 2500, "temperature": 0.0})


def trimmed(text):
    return "\n".join(ln.strip() for ln in text.split("\n") if ln.strip())


text = DOC[:SAMP]
gold_compact = text.replace("\t", "\\t").replace("\n", "\\n")
gold_trim = trimmed(text)

print("Isolating accuracy cause (same encoder, vary rendering):", flush=True)
# 1. compact f12 (dense marker flow)
imgs, _ = P.render_pages(text, font_size=12, page_w=1400, page_h=720, mode="gray", compact=True)
_, got = trans(imgs, True)
cer = P.levenshtein(gold_compact, got.replace("\r\n", "\n").strip("\n")) / max(1, len(gold_compact))
print(f"compact f12 (marker flow): CER={100*cer:.0f}%", flush=True)

# 2. compact f18 (bigger font, same dense flow)
imgs, _ = P.render_pages(text, font_size=18, page_w=1400, page_h=720, mode="gray", compact=True)
_, got = trans(imgs, True)
cer = P.levenshtein(gold_compact, got.replace("\r\n", "\n").strip("\n")) / max(1, len(gold_compact))
print(f"compact f18 (bigger font): CER={100*cer:.0f}%", flush=True)

# 3. trimmed (real newlines, no blank/indent) f12 — keeps line structure
imgs, _ = P.render_pages(trimmed(text), font_size=12, page_w=1400, page_h=720, mode="gray", compact=False)
_, got = trans(imgs, False)
cer = P.levenshtein(gold_trim, got.replace("\r\n", "\n").strip("\n")) / max(1, len(gold_trim))
print(f"trimmed real-newlines f12: CER={100*cer:.0f}%", flush=True)
