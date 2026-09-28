# Siphon Studio — Headless Agent Guide

This guide is the machine-oriented entry point for agents and scripts that
need to discover, start, monitor, use, and stop Siphon Studio models without
opening the desktop app or terminal TUI.

## Scope

Siphon Studio exposes two headless surfaces:

1. The `siphon` CLI for lifecycle control and machine-readable status.
2. A local OpenAI-compatible HTTP API for inference requests.

The CLI controls the Siphon Studio runtime, including the `siphon.cpp`
inference engine. Use the CLI for process management and the HTTP API for
generation.

## Requirements

- Run commands from `siphon-studio/` in this repository.
- Use Python 3.
- The model file referenced by the selected profile must exist.
- The runtime must have been built successfully.
- NVIDIA/CUDA/GDS support must pass the environment checks for the Linux
  reference path.

The checkout-local entry point does not require installing a Python package:

```bash
cd /home/manohar/Desktop/siphon.cpp/siphon-studio
python3 bin/siphon --version
```

If the global `siphon` command is installed, it can be used instead of
`python3 bin/siphon`.

## Rules for agents

- Always run `list --json` before selecting a profile. Do not assume a profile
  name, model path, port, or model alias.
- Parse JSON from stdout. Diagnostic messages may be written to stderr.
- Use `--wait` or the `wait` command before sending an inference request.
- Check the model's actual health before treating it as ready.
- Use the profile's configured host and port. The common default is
  `127.0.0.1:8090`, but imported profiles may use another port.
- Do not start the desktop UI or TUI for an automated task.
- Stop the profile when the task is complete unless another process owns it.
- On a 6 GB GPU, only one model is normally allowed to run at a time.

## 1. Check the environment

Run the doctor command first. It checks the runtime prerequisites and storage
path used by the configured profiles.

```bash
cd /home/manohar/Desktop/siphon.cpp/siphon-studio
python3 bin/siphon doctor --json
```

The command exits non-zero if a required check fails. An agent should report
the failed check instead of attempting repeated starts.

## 2. Discover profiles and models

```bash
python3 bin/siphon list --json
```

The response contains:

- `profiles`: launchable Siphon Studio profiles and their environment;
- `models`: discovered GGUF files; and
- `suites`: available expert-cache suites.

Select a profile from the returned `profiles` array. The profile's `env`
contains the model path, API host, API port, and model alias. The alias is the
value to send in the HTTP request's `model` field.

Typical profiles in this checkout include:

- `qwen3.6-moe-tq3`
- `kat-coder-mtp`
- `qwen3-coder-next`
- `qwen3-coder-next-draft`
- imported model profiles created by the Studio model catalogue

These names are not a stable API. Agents must continue to use the names
returned by `list --json`.

## 3. Start a model and wait for readiness

Replace `<profile>` with a name returned by `list --json`.

```bash
python3 bin/siphon up <profile> --wait 300 --json
```

`up` is idempotent. It starts the profile if it is not already running and
waits up to 300 seconds for the server to become healthy.

If the agent wants separate lifecycle steps:

```bash
python3 bin/siphon up <profile> --json
python3 bin/siphon wait <profile> --timeout 300 --json
```

Do not send requests merely because the process exists. A model may still be
loading weights. The wait command must return `{"ok": true, ...}` before the
agent proceeds.

For a preview without starting anything:

```bash
python3 bin/siphon up <profile> --dry-run --json
```

## 4. Verify status and health

CLI status:

```bash
python3 bin/siphon status --json
```

The status response includes the profile name, running state, health state,
decode speed, prefill speed, and process ID.

HTTP health and model discovery, using the profile's configured port:

```bash
curl -fsS http://127.0.0.1:8090/health
curl -fsS http://127.0.0.1:8090/v1/models
```

If the profile uses another port, substitute that port. A failed health check
means the server is not ready or is no longer running.

## 5. Send a non-streaming request

Use the model alias returned in the selected profile's `ALIAS` environment
value.

```bash
curl -fsS http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-moe-tq3",
    "messages": [
      {"role": "user", "content": "Explain the purpose of Siphon Studio in three sentences."}
    ],
    "max_tokens": 256,
    "stream": false
  }'
```

The response follows the OpenAI Chat Completions shape. Agents should check
the HTTP status and parse the JSON response before using the generated text.

## 6. Send a streaming request

Use `-N` so curl does not buffer the streamed response.

```bash
curl -fsS -N http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-moe-tq3",
    "messages": [
      {"role": "user", "content": "Write a small Python function that reverses a string."}
    ],
    "max_tokens": 256,
    "stream": true
  }'
```

Streaming responses are Server-Sent Events. Each `data:` event contains a JSON
chunk, and the stream ends with `data: [DONE]`.

## 7. Read live metrics

Use the profile name, not necessarily the model alias:

```bash
python3 bin/siphon metrics <profile> --json
```

The result includes:

- `health`;
- `decode_tps`;
- `prefill_tps`;
- `n_decode`;
- `n_predicted`;
- `n_prompt`; and
- `requests_processing`.

For continuous monitoring, poll the command at a moderate interval, for
example once every 1–2 seconds. Do not use the dashboard as a data source for
automation.

## 8. Read logs when a start or request fails

```bash
python3 bin/siphon logs <profile> --tail 100 --json
```

The JSON response contains a `lines` array. Look for model-file errors,
CUDA/GDS initialization failures, port conflicts, out-of-memory messages, and
health transitions.

The runtime may take time to load a large model. A process that is running but
not healthy is not automatically a failure; use the configured wait timeout
and inspect logs before retrying.

## 9. Stop the model

```bash
python3 bin/siphon down <profile> --json
```

For emergency cleanup of a profile that did not exit normally:

```bash
python3 bin/siphon down <profile> --force --json
```

To stop every profile managed by the local Studio state:

```bash
python3 bin/siphon down --all --force --json
```

Only use `--all` when the agent owns all Siphon Studio processes on the host.

## Complete minimal runbook

The following is the shortest safe sequence for an agent using the default
Qwen profile. A production agent should discover the profile and API port with
`list --json` first.

```bash
cd /home/manohar/Desktop/siphon.cpp/siphon-studio

python3 bin/siphon doctor --json
python3 bin/siphon list --json
python3 bin/siphon up qwen3.6-moe-tq3 --wait 300 --json
python3 bin/siphon status --json

curl -fsS http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-moe-tq3",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 32,
    "stream": false
  }'

python3 bin/siphon metrics qwen3.6-moe-tq3 --json
python3 bin/siphon down qwen3.6-moe-tq3 --json
```

## Important limitations

- The Linux CUDA/GDS path is the primary validated path.
- Performance depends on model architecture, quantization, SSD, GPU, driver,
  context length, cache budgets, batching, and concurrency.
- Only one model is normally allowed on the 6 GB reference GPU at a time.
- A profile can fail if its model path or expert suite no longer exists.
- Do not infer that a successful process start means a healthy model. Always
  wait for health.
- Do not infer runtime metrics from hardcoded UI values; use `metrics --json`
  and the server API.

