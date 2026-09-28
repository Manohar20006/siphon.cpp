#!/usr/bin/env python3
"""
Extended failure-pattern benchmark: tests qwen3.6 vision reading across
content types (json, xml, code, markdown, csv, messy) using real workspace
files where possible.

Each type has a snippet + ground-truth questions. Renders to image at
font 14 and 18, compares vs text baseline, scores exact/gist.
"""
import base64, json, re, time, urllib.request, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SERVER = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "qwen3.6"
RESULTS = HERE / "results"

import json as J

def render_pages(text, out_prefix, font_size=14, width=1400, max_h=720, bold=False):
    fp = FONT_BOLD if bold else FONT_REG
    font = ImageFont.truetype(fp, font_size)
    lh = int(font_size * 1.15)
    col_w = font.getbbox("W")[2]
    cols = max(1, (width - 24) // col_w)
    lines = []
    for raw in text.split("\n"):
        if not raw: lines.append(""); continue
        while len(raw) > cols: lines.append(raw[:cols]); raw = raw[cols:]
        lines.append(raw)
    n_fit = max(1, (max_h - 24) // lh)
    pages = []
    for p, start in enumerate(range(0, len(lines), n_fit)):
        chunk = lines[start:start+n_fit]
        h = 24 + len(chunk) * lh
        img = Image.new("RGB", (width, h), "white")
        d = ImageDraw.Draw(img)
        y = 12
        for ln in chunk:
            d.text((12, y), ln, fill="black", font=font); y += lh
        path = f"{out_prefix}_p{p}.png"
        img.save(path)
        pages.append(path)
    return pages

def query(img_path, question, max_tokens=400):
    # img_path may be a single path or list of paths (multi-page)
    paths = img_path if isinstance(img_path, list) else [img_path]
    content = []
    for p in paths:
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": question})
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(SERVER, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = J.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    return resp["choices"][0]["message"]["content"], dt, resp["usage"]["prompt_tokens"]

def query_text(text, question, max_tokens=400):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": question + "\n\n" + text}],
               "max_tokens": max_tokens, "temperature": 0.0}
    req = urllib.request.Request(SERVER, data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"})
    t0 = time.time()
    resp = J.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    return resp["choices"][0]["message"]["content"], dt, resp["usage"]["prompt_tokens"]

def lev(a,b):
    if len(a)<len(b): a,b=b,a
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(prev[j]+1,cur[j-1]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

# -- corpus: snippets + ground truth --

def snippet(path, n_chars=3500):
    t = Path(path).read_text(errors="ignore")[:n_chars]
    # trim to last newline to avoid mid-line cut
    cut = t.rfind("\n")
    return t[:cut] if cut > n_chars*0.8 else t

tests = []

# 1. JSON (synthetic, known truth)
import random as R
R.seed(42)
jdoc = {"servers": []}
for i in range(8):
    jdoc["servers"].append({"id": f"srv-{i:03d}", "checksum": f"{R.randrange(16**32):032x}", "port": 8000+i, "region": "us-east-1"})
jtext = J.dumps(jdoc, indent=2)
tests.append(("json", jtext,
    "Read this JSON. Answer in JSON: {\"count\": <number of servers>, \"srv_003_checksum\": \"<hex>\", \"srv_005_port\": <number>}",
    {"count": "8", "srv_003_checksum": jdoc["servers"][3]["checksum"], "srv_005_port": str(jdoc["servers"][5]["port"])}))

# 2. Markdown (real file)
md = snippet("/home/manohar/Desktop/siphon.cpp/siphon-deep-dive.md", 3500)
# ground truth: count H2 (## ) and first H2 text
h2s = [l[3:].strip() for l in md.split("\n") if l.startswith("## ")]
tests.append(("markdown", md,
    "Read this markdown. Answer in JSON: {\"h2_count\": <number of ## headings>, \"first_h2\": \"<text of first ## heading>\"}",
    {"h2_count": str(len(h2s)), "first_h2": h2s[0] if h2s else ""}))

# 3. Code (synthetic, known truth)
code = """int add(int a, int b) {
  return a + b;
}
void hello() {
  printf("hi\\n");
}
int main() {
  int x = add(2, 3);
  hello();
  return x;
}
"""
tests.append(("code_cpp", code,
    "Read this C++ code. Answer in JSON: {\"func_count\": <number of function definitions>, \"first_func\": \"<name of first function>\"}",
    {"func_count": "3", "first_func": "add"}))

# 4. XML (synthetic)
xml = "<catalog>\n"
for i in range(6):
    xml += f'  <item id="item-{i:03d}" sku="{R.randrange(16**16):016x}" price="{R.randint(10,999)}.{R.randint(0,99):02d}"/>\n'
xml += "</catalog>"
tests.append(("xml", xml,
    "Read this XML. Answer in JSON: {\"item_count\": <number of <item> elements>, \"item_002_sku\": \"<hex sku of item-002>\"}",
    {"item_count": "6", "item_002_sku": re.search(r'item-002" sku="([^"]+)"', xml).group(1)}))

# 5. CSV
csv = "id,name,region,balance\n"
for i in range(10):
    csv += f"{1000+i},user_{i:03d},{'us-east-1' if i%2==0 else 'eu-west-1'},{R.randint(100,9999)}.{R.randint(0,99):02d}\n"
tests.append(("csv", csv,
    "Read this CSV. Answer in JSON: {\"row_count\": <number of data rows>, \"row_5_name\": \"<name in row with id 1005>\"}",
    {"row_count": "10", "row_5_name": "user_005"}))

# 6. Messy log (mixed)
log = ""
for i in range(12):
    lvl = R.choice(["INFO","WARN","ERROR"])
    log += f"2026-08-20 10:{i:02d}:00 [{lvl}] server-{i%4} msg id {R.randrange(16**8):08x} latency {R.randint(5,500)}ms\n"
err_ids = re.findall(r"\[ERROR\].*?id (\w+)", log)
tests.append(("messy_log", log,
    "Read this log. Answer in JSON: {\"error_count\": <number of ERROR lines>, \"first_error_id\": \"<hex id of first ERROR>\"}",
    {"error_count": str(log.count("[ERROR]")), "first_error_id": err_ids[0] if err_ids else ""}))

# -- run --
for name, text, question, truth in tests:
    print(f"\n{'='*60}\n# {name} ({len(text)} chars, truth={truth})")
    for mode in ["text", "image_f14", "image_f18"]:
        if mode == "text":
            reply, dt, toks = query_text(text, question)
        else:
            fs = 14 if "f14" in mode else 18
            prefix = f"/tmp/opencode/types_{name}_{mode}"
            pages = render_pages(text, prefix, font_size=fs)
            reply, dt, toks = query(pages, question)
        # extract JSON
        m = re.search(r"\{.*\}", reply, re.S)
        try:
            parsed = J.loads(m.group(0)) if m else {}
        except: parsed = {}
        ok = 0; tot = len(truth); details=[]
        for k,v in truth.items():
            got = str(parsed.get(k,"")).strip().lower()
            exp = str(v).strip().lower()
            is_ok = got == exp
            ok += is_ok
            details.append(f"  {k}: {'OK' if is_ok else 'FAIL'} (exp {exp!r} got {got!r} lev {lev(exp,got)})")
        print(f"  {mode:14} {ok}/{tot}  {dt:.1f}s toks {toks}")
        for d in details: print(d)
        if ok < tot:
            print(f"  raw: {reply[:300]!r}")
