# Expert Suites

This folder contains small static MoE expert-selection files.

Default:

```text
moe-static-experts-suite-8192.txt
```

Prefill-ranked variant:

```text
moe-static-prefill-experts-suite-8192.txt
```

Each non-comment line uses:

```text
layer expert miss_count access_count bytes
```

These files are generated from routing traces with:

```bash
python3 scripts/select-static-moe-experts.py \
  moe-routing-suite-1000-combined.jsonl \
  expert-suites/moe-static-experts-suite-8192.txt \
  --budget-mib 8192
```

The full trace-to-suite flow is:

```bash
python3 scripts/build-moe-trace-prompt-suite.py

# Start a trace-recording server with LLAMA_DSTORAGE_ROUTING_TRACE set, then:
python3 scripts/run-moe-routing-trace.py \
  moe-trace-prompts-1000.jsonl \
  moe-routing-suite-1000.progress.jsonl \
  --url http://127.0.0.1:8091/completion \
  --n-predict 32

python3 scripts/select-static-moe-experts.py \
  moe-routing-suite-1000-combined.jsonl \
  expert-suites/moe-static-experts-suite-8192.txt \
  --budget-mib 8192
```
