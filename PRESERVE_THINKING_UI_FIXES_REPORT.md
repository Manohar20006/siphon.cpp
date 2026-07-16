# Fix Report: Chat Reprocessing and Web UI 404

This report documents the fixes made in the [siphon.cpp](file:///home/manohar/Desktop/inference/siphon.cpp) repository during this session to resolve KV cache reprocessing and Web UI loading issues.

---

## 1. Reprocessing Fix (preserve_thinking)

### Problem
During multi-turn chat sessions, follow-up queries failed to reuse context checkpoints, rolling back and re-prefilling the entire chat history from token 36. This led to massive latency on long context prompts.

### Root Cause
1. With reasoning disabled (`--reasoning off`), the assistant outputs empty `<think>\n\n</think>\n\n` blocks.
2. When the server re-rendered the chat history template to construct the prefix for the next turn, the Jinja engine stripped out these empty `<think>` tags.
3. This altered the rendered text and shifted the tokens, causing a sequence mismatch against the resident KV cache. As a result, the server discarded the cache and started pre-filling from scratch.

### Solution
We configured the server's Jinja template parser to keep the empty tags in the rendering history by passing `preserve_thinking: true`.
* **File Updated**: [configs/qwen3.6-moe-tq3.env](file:///home/manohar/Desktop/inference/siphon.cpp/configs/qwen3.6-moe-tq3.env)
* **Changes Committed**: Commit `3d3114b` ("Fix reprocessing (preserve_thinking)")
* **Line Added**:
  ```ini
  CHAT_TEMPLATE_KWARGS='{"preserve_thinking": true}'
  ```

---

## 2. Embedded Web UI 404 Fix

### Problem
Navigating to `http://127.0.0.1:8090` returned a `404 File Not Found` error.

### Root Cause
Because the environment is network-restricted, CMake failed to download the pre-compiled Web UI assets from Hugging Face / GitHub during the build of `siphon.cpp`. Consequently, `llama-server` compiled a minimal library without an embedded UI.

### Solution
1. We located the built UI asset files (`bundle.js`, `bundle.css`, `index.html`, `loading.html`) inside the build folder of the neighboring `qwen.cpp-public` clone.
2. Created the target asset directory [tools/ui/dist](file:///home/manohar/Desktop/inference/siphon.cpp/tools/ui/dist) in the new folder.
3. Copied the 4 asset files over:
   ```bash
   cp /home/manohar/Desktop/inference/qwen.cpp-public/build-gds-cuda/tools/ui/dist/* \
      /home/manohar/Desktop/inference/siphon.cpp/tools/ui/dist/
   ```
4. Recompiled the repository using `./scripts/build-gds-cuda.sh`. The CMake script successfully detected the local assets and embedded them into the compilation:
   ```text
   -- UI: using pre-built assets from /home/manohar/Desktop/inference/siphon.cpp/tools/ui/dist
   [  8%] Built target llama-ui-assets
   ```
This compiled a full `libllama-server-impl.so` library (size went from ~3.6 MB up to ~12.4 MB), restoring the built-in UI page.

---

## 3. Usage & On-Demand Reasoning

The server is currently running in the background listening on `http://127.0.0.1:8090`. 

Because reasoning is disabled by default for performance, you can toggle it on-demand for specific API requests by sending the following parameters in your request body:

```json
{
  "model": "qwen3.6-moe-tq3",
  "messages": [
    {"role": "user", "content": "Explain this complex multithreading race condition..."}
  ],
  "thinking_budget_tokens": 1024,
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```
`preserve_thinking: true` will ensure that any reasoning blocks generated during these on-demand requests are preserved in subsequent turns, maintaining KV cache prefix alignment.
