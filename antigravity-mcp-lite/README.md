# Antigravity MCP Lite for Cline

A local stdio MCP server that lets Cline call models available through your
already-authenticated Antigravity `agy` CLI. It does not use a separate API
key or contact a hosted bridge. Your prompts are sent to the local CLI, which
uses the Antigravity account configured on that machine.

## Requirements

- Python 3.10 or newer
- The `agy` CLI installed and signed in on the same machine
- Cline for VS Code

The MCP server itself uses only Python's standard library. It has no API-key
provider or other remote-model backend.

## Install and configure

1. Unzip/copy this folder to a stable location on your machine.
2. Copy `config.example.json` to a private local config file and set
   `project_root` to the project directory Cline should work in. Use an
   absolute path. The model aliases can be changed to model IDs supported by
   your `agy` installation.
3. In Cline, open **MCP Servers** → **Configure** → **Configure MCP Servers**.
   Add the following entry under the existing `mcpServers` object, replacing
   both example paths with absolute paths on your machine:

   ```json
   {
     "mcpServers": {
       "antigravity-local": {
         "command": "python3",
         "args": ["/absolute/path/antigravity-mcp-lite/server.py"],
         "env": {
           "ANTIGRAVITY_MCP_LITE_CONFIG": "/absolute/path/antigravity-config.json"
         },
         "disabled": false,
         "autoApprove": []
       }
     }
   }
   ```

   A copy of this snippet is in `cline-mcp-settings.example.json`. On Windows,
   use the Python launcher (`py`) or the full path to `python.exe` if `python3`
   is not available.
4. Save the settings and restart/reconnect the server in Cline's MCP Servers
   panel. Confirm the Antigravity tools appear there.

If you prefer an installed command, run `python3 -m pip install .` from this
folder and use `"command": "antigravity-mcp-lite"` with no script path in
`args`. The direct `python3 server.py` configuration avoids needing to install
the package.

## Use from Cline

The server exposes model listing, health, one-shot prompting, and persistent
session tools. For repeated turns, call `create_antigravity_session` once, then
use the returned `session_id` with `send_session_message`; close it with
`close_antigravity_session` when finished. The first request still has model
startup/inference latency. A persistent process avoids restarting `agy` for
each turn, but does not guarantee sub-second responses.

This makes Antigravity models available to Cline as MCP tools. It does **not**
replace Cline's configured primary model: Cline still needs a model/provider
selected for its own planning and tool loop. Depending on that provider, Cline
may require its own account, credits, or API credentials. Only the Antigravity
worker calls made through this MCP use the local `agy` authentication.

## Sharing with someone else

Share the source folder or the zip bundle, not your Cline settings or local
config. The recipient must install/sign in to `agy` on their own machine, copy
the example config, set their own absolute project path, and add their own
absolute paths in Cline's MCP settings. Do not send account tokens or saved
authentication files.

The stdio server runs as a child process of Cline and is local to that user. It
does not expose a network listener. Keep `autoApprove` empty unless the user
understands and explicitly wants automatic approvals for specific tools.

## Troubleshooting

- If Cline cannot start the server, check that the Python executable and
  `server.py` paths are correct and that the config path exists.
- If `antigravity_health` reports that `agy` is missing, make sure `agy` is on
  the PATH inherited by VS Code, or set `agy_path` in the config to its full
  executable path.
- If the tools are visible but requests fail, sign in to Antigravity with the
  same OS account that launches VS Code and check that the configured model ID
  is accepted by `agy`.
