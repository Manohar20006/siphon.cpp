#!/usr/bin/env python3
"""compact_pages.py — show that image prefill time is driven by #encoder passes
(page count), not token count. Same 20k doc, compact, varying page height."""
import time, json, base64, io, urllib.request
from pathlib import Path
import prefill as P

SERVER = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.6"
DOC = Path("/home/manohar/Desktop/siphon.cpp/siphon-deep-dive.md").read_text()
PRE = 20000
text = DOC[:PRE]


def post(p):
    r = urllib.request.Request(SERVER, data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    t = time.time(); resp = json.loads(urllib.request.urlopen(r, timeout=600).read())
    return round(time.time() - t, 2), resp.get("usage", {}).get("prompt_tokens")


def prefill(imgs):
    c = []
    for im in imgs:
        b = io.BytesIO(); im.save(b, format="PNG")
        c.append({"type": "image_url",
                  "image_url": {"url": "data:image/png;base64," +
                                base64.b64encode(b.getvalue()).decode()}})
    c.append({"type": "text", "text": f"x{time.time_ns():x}"})
    return post({"model": MODEL, "messages": [{"role": "user", "content": c}],
                 "max_tokens": 1, "temperature": 0.0})


# text baseline (cold-ish: unique tail each call not used; note cache effect)
_, ttok = post({"model": MODEL, "messages": [{"role": "user",
             "content": text + f"\n{time.time_ns():x}"}], "max_tokens": 1, "temperature": 0.0})
print(f"TEXT tok={ttok}", flush=True)

for ph in (720, 1100, 1500):
    imgs, meta = P.render_pages(text, font_size=12, page_w=1400, page_h=ph,
                               mode="gray", compact=True)
    dt, tok = prefill(imgs)
    print(f"compact page_h={ph}: pages={meta['n_pages']} tok={tok} time={dt}s "
          f"| encoder_passes={meta['n_pages']}", flush=True)
