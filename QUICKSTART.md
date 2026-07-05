# Quickstart

1. Put the model in `models/`:

```text
models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

2. Build:

```bash
./scripts/build-gds-cuda.sh
```

3. Run:

```bash
./scripts/run-qwen36-moe.sh
```

4. Check the server:

```bash
curl http://127.0.0.1:8090/v1/models
```

Open the built-in web chat here:

```text
http://127.0.0.1:8090
```

The OpenAI-compatible base URL is:

```text
http://127.0.0.1:8090/v1
```
