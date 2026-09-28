# Universal Worker Bridge MCP

This MCP server supports both local stdio and authenticated Streamable HTTP
for the workflow you described:

* Codex Desktop and Astra remain the project authority and planner.
* Antigravity supplies Gemini and Claude implementation workers.
* A Kaggle-hosted Qwen 27B endpoint is another implementation worker.
* The bridge transports one complete Astra plan and its verification procedure.
* Workers return a structured report and unified diff.
* Codex can apply a reviewed worker diff to the canonical project workspace.
* The bridge executes Astra-selected verification commands and sends the real
  output back to the worker for interpretation.
* Finalization is blocked until those commands pass, the worker reviews their
  evidence, and Astra records its approval.

The server never calls Astra and never asks Astra to plan individual worker
steps. The caller supplies the complete implementation plan and the complete
verification instructions in one dispatch. This is the token-saving boundary.

## Worker transports

The Kaggle Qwen worker uses an OpenAI-compatible HTTP endpoint. Set
KAGGLE_QWEN_BASE_URL to the public deployment URL and KAGGLE_QWEN_API_KEY to
the key. The key is read only from the environment and is never returned by
list_workers or written to a file.

Antigravity can use either of these approaches:

1. If Antigravity exposes an OpenAI-compatible endpoint, set its transport to
   openai_compatible and configure its base URL and api_key_env.
2. If Antigravity exposes a CLI or local adapter, use transport command. The
   bridge sends one JSON payload on stdin and expects one JSON result on stdout.

The MCP protocol alone cannot control a private Antigravity UI. A callable API,
CLI, or adapter is required for the Antigravity side.

The command adapter receives the complete plan, context, and commands in one
payload. Normal MCP responses remain compact: source context, raw model output,
and full patches are returned only when explicitly requested.

~~~json
{
  "system_instructions": "...",
  "task": {
    "task_id": "...",
    "objective": "...",
    "implementation_plan": "...",
    "verification_instructions": "...",
    "verification_commands": [["python3", "-m", "unittest"]],
    "context_files": {}
  },
  "prompt": "..."
}
~~~

It must print a JSON object containing:

~~~json
{
  "status": "completed",
  "summary": "what changed",
  "patch": "unified diff",
  "tests_run": ["command"],
  "verification_report": "results",
  "blockers": []
}
~~~

## MCP tools

* readiness_check — validate project root, persistent storage, enabled workers,
  local worker executables, and the command allowlist before use.
* list_workers — show configured workers without exposing secrets.
* worker_health — check HTTP workers and configured command workers.
* delegate_workspace_operation — use Antigravity as the default hands for
  bounded inspection, command execution, or direct edits and return compact evidence.
* dispatch_worker_task — send Astra's complete plan and verification protocol.
  It runs asynchronously by default and returns a background job immediately.
* get_worker_job — poll only compact background-job state.
* get_task_events — poll compact checkpoint/question events by cursor.
* send_worker_message — asynchronously answer, redirect, or add requirements; messages
  arriving during an active worker turn are queued and delivered afterward.
* answer_worker_question — resolve a specific identified blocking or non-blocking question.
* dispatch_worker_review — start a clean-context independent reviewer conversation.
* resolve_worker_review — record Codex's disposition of reviewer findings; approval
  is blocked while an attached review remains unresolved.
* pause_worker_task / resume_worker_task / cancel_worker_task — lifecycle controls.
* continue_worker_task — send Astra feedback and request correction/re-verification.
* get_task_status — retrieve compact status; full patch, raw output, context,
  evidence output, and communication history are opt-in.
* apply_worker_patch — validate a reviewed diff with git and apply it locally.
* run_worker_verification — execute Astra's argv-form checks and ask the worker
  to interpret the captured evidence or return a corrective patch.
* approve_task — record Astra's acceptance after evidence-based verification.
* finalize_project — pass the finalization gate and submit the final version.

The required finalization sequence is:

1. dispatch_worker_task
2. review the worker report
3. continue_worker_task as many times as corrections require
4. apply_worker_patch with confirm=true when the worker returned a patch
5. run_worker_verification with confirm=true
6. If the worker returns a correction, apply it and repeat step 5
7. approve_task
8. finalize_project

The bridge does not automatically apply worker code or declare a final version.
Verification commands are executed directly as argument arrays, without a
shell, in the configured project root. They can still modify the project, so
Codex should prompt before this tool is allowed. Configure
`allowed_command_prefixes` to constrain both delegated commands and verification
commands. Prefixes are exact argv prefixes; avoid permissive entries such as
`["bash", "-c"]`. With `storage_path` configured, task and job snapshots are
durably stored in SQLite.

## Supervised worker loop

Supervision is turn-based rather than raw chain-of-thought streaming. Workers
return concise `checkpoint`, `question`, `risk`, `blocked`, or
`ready_for_review` events. Codex reads only new events using a sequence cursor
and can answer or redirect the worker with `send_worker_message`. Antigravity
conversation IDs are preserved so follow-ups resume the same session. Blocking
questions prevent approval and finalization until Codex responds.

Dispatch, follow-up messages, question answers, and independent reviews run as
background jobs by default. This keeps the MCP request responsive while the
worker is generating. A message sent during an active turn is stored in the
task inbox and delivered sequentially after that turn, avoiding concurrent use
of one model conversation. Reviewer tasks always start without the author's
conversation ID and retain a link to their parent implementation task.

The current event stream is checkpoint-based, not token streaming: a worker
cannot emit a semantic checkpoint until its current model turn returns.

Antigravity CLI prompts are bounded by `ANTIGRAVITY_MAX_PROMPT_BYTES` (120,000
by default) to avoid operating-system command-line limits. Split large audits or
implementations into bounded phases and pass only the files needed for each
phase. The bridge returns a structured blocker instead of crashing when the
limit is exceeded.

Task and job snapshots are persisted in SQLite when `storage_path` or
`UNIVERSAL_MCP_STORAGE_PATH` is configured. Completed and interrupted work,
questions, messages, patches, verification evidence, reviewer links, and
conversation IDs survive process restarts. Jobs that were active during a crash
are restored as `interrupted` rather than incorrectly reported as running.

Verification records a workspace fingerprint over the relevant changed paths.
Approval re-computes that fingerprint and rejects stale verification if files
changed afterward. Attached independent reviews must also be explicitly
resolved by Codex before approval. Status responses are compact by default;
set `compact=false` and request the desired evidence fields for deeper review.

See `TOKEN_RESULTS.md` for the measured Codex-context reduction and methodology.

## Default workspace worker

For substantive workspace work, Codex can call `delegate_workspace_operation`
with `inspect`, `command`, or `edit`. The bridge selects an Antigravity worker
by default and preserves its conversation ID for follow-up. To avoid broad
headless-agent permissions, the MCP reads bounded target files or executes the
exact supplied argv and gives the captured evidence to Antigravity. Edit work
returns a reviewable patch. Command and edit operations require `confirm=true`;
edits must name their allowed relative target paths.

This delegation is an efficiency policy, not a blind-trust policy. Codex should
directly inspect security-sensitive output, final diffs, and verification
evidence. Very small reads or commands may be cheaper to perform directly than
starting a model worker session.

## Run locally

1. Copy workers.json.example to workers.json.
2. Replace the Antigravity adapter commands.
3. Set the Kaggle URL and key in the parent environment.
4. Start the server:

~~~sh
UNIVERSAL_MCP_CONFIG=/absolute/path/to/universal-mcp/workers.json \
UNIVERSAL_MCP_PROJECT_ROOT=/absolute/path/to/project \
UNIVERSAL_MCP_STORAGE_PATH=/absolute/path/to/project/.universal-mcp-state.sqlite3 \
python3 /absolute/path/to/universal-mcp/server.py
~~~

For Codex Desktop, merge `codex-config.example.toml` into
`~/.codex/config.toml`. The example raises the MCP tool timeout for long worker
and verification calls and prompts before patch application or command
execution. Export the Kaggle URL and key before starting Codex; do not put the
real key in a committed file.

## Connect from Antigravity over HTTP

Generate a strong random bearer token, export it together with the worker
configuration, and start the HTTP entry point:

~~~sh
export UNIVERSAL_MCP_HTTP_TOKEN='replace-with-a-long-random-secret'
export UNIVERSAL_MCP_CONFIG='/absolute/path/to/universal-mcp/workers.json'
export UNIVERSAL_MCP_PROJECT_ROOT='/absolute/path/to/your/project'
python3 /absolute/path/to/universal-mcp/http_server.py --host 127.0.0.1 --port 8765
~~~

The MCP URL is `http://127.0.0.1:8765/mcp`. For another computer, place the
server behind HTTPS and set `--host 0.0.0.0`; never expose the plain HTTP port
directly to the internet. The server rejects unauthenticated requests. If a
client supplies an `Origin` header, list the exact allowed value in the
comma-separated `UNIVERSAL_MCP_ALLOWED_ORIGINS` environment variable.

Antigravity uses `serverUrl` for remote MCP servers. Copy
`antigravity-mcp-config.example.json` into the applicable `mcp_config.json`,
replace the URL and token, and keep that file out of version control. A local
configuration looks like this:

~~~json
{
  "mcpServers": {
    "universal-worker-bridge": {
      "serverUrl": "http://127.0.0.1:8765/mcp",
      "headers": {
        "Authorization": "Bearer replace-with-the-same-secret"
      }
    }
  }
}
~~~

`GET /healthz` is an unauthenticated liveness check. MCP traffic uses
`POST /mcp`; standalone SSE listening is intentionally not implemented because
this bridge does not send unsolicited server-to-client messages.

## Test

From the `universal-mcp` directory:

~~~sh
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
~~~
