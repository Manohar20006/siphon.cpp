#!/usr/bin/env python3
"""Antigravity CLI worker adapter for the universal worker bridge.

This script reads a JSON payload on stdin conforming to the bridge command worker contract,
invokes the Antigravity CLI (`agy`), and emits the required structured JSON on stdout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys


def main() -> int:
    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input) if raw_input.strip() else {}
    except Exception as exc:
        output = {
            "status": "failed",
            "summary": f"Failed to parse stdin payload: {exc}",
            "patch": "",
            "tests_run": [],
            "verification_report": "",
            "blockers": [str(exc)],
        }
        print(json.dumps(output))
        return 0

    worker_id = os.environ.get("UNIVERSAL_MCP_WORKER_ID", "antigravity-gemini")
    task_data = payload.get("task", {}) if isinstance(payload.get("task"), dict) else {}
    conversation_id = payload.get("conversation_id") or task_data.get("conversation_id")
    prompt = payload.get("prompt") or payload.get("task", {}).get("objective", "")
    worker_tool_access = bool(task_data.get("worker_tool_access", False))
    operation_kind = str(task_data.get("operation_kind", "implementation"))
    operation_targets = task_data.get("operation_targets", [])
    if worker_tool_access:
        access_rule = {
            "inspect": "You may read files and run non-mutating inspection commands. Do not modify files.",
            "command": "You may use the terminal to execute only the requested command or directly necessary read-only diagnostics. Do not edit files.",
            "edit": "You may read and edit only the allowed target paths and run checks directly related to those edits.",
        }.get(operation_kind, "Use workspace tools only for the explicitly assigned operation.")
        prompt = (
            f"Workspace tools are enabled for this delegated {operation_kind} operation. "
            f"Allowed targets: {json.dumps(operation_targets)}. {access_rule} "
            "Return concise JSON evidence: what you inspected or executed, important output, "
            "paths changed, exit status when applicable, assumptions, and risks.\n\n"
            + prompt
        )
    else:
        # Normal implementation remains response-only: the bridge reviews and
        # applies the returned diff instead of granting direct workspace access.
        prompt = (
            "Do not use terminal, filesystem, browser, or other tools for this task. "
            "Return the required JSON result directly, including a unified diff for "
            "all requested changes. The MCP will review and apply the diff.\n\n"
            + prompt
        )

    max_prompt_bytes = int(os.environ.get("ANTIGRAVITY_MAX_PROMPT_BYTES", "120000"))
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes > max_prompt_bytes:
        output = {
            "status": "failed",
            "summary": "Worker prompt exceeds the safe Antigravity command-argument limit.",
            "patch": "",
            "tests_run": [],
            "verification_report": "",
            "blockers": [
                f"prompt_too_large: {prompt_bytes} bytes exceeds {max_prompt_bytes}; "
                "split the task/context into bounded phases"
            ],
            "error": "prompt_too_large",
        }
        print(json.dumps(output))
        return 0

    # Locate agy CLI
    agy_bin = shutil.which("agy") or os.path.expanduser("~/.local/bin/agy")
    if not (os.path.isfile(agy_bin) and os.access(agy_bin, os.X_OK)):
        output = {
            "status": "failed",
            "summary": f"Antigravity CLI ('agy') not found at {agy_bin}",
            "patch": "",
            "tests_run": [],
            "verification_report": "",
            "blockers": ["Antigravity CLI 'agy' is not available on PATH."],
        }
        print(json.dumps(output))
        return 0

    # Model and effort selection
    if "claude" in worker_id.lower():
        model_name = "claude-sonnet-4-6"
        cmd = [agy_bin, "--model", model_name, "--mode", "accept-edits", "--output-format", "json"]
    else:
        model_name = "gemini-3.8-flash"
        cmd = [agy_bin, "--model", model_name, "--effort", "medium", "--mode", "accept-edits", "--output-format", "json"]

    if conversation_id:
        cmd.extend(["--conversation", str(conversation_id)])
    cmd.append(f"--print={prompt}")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(os.environ.get("ANTIGRAVITY_TIMEOUT", "300")),
            check=False,
            cwd=(str(payload.get("project_root")) or None),
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if not stdout:
            detail = stderr or "Antigravity produced no structured output."
            output = {
                "status": "failed",
                "summary": detail[:500],
                "patch": "",
                "tests_run": [],
                "verification_report": "",
                "blockers": [detail[:1000]],
            }
            print(json.dumps(output))
            return 0

        usage = {}
        duration_seconds = None
        response_text = stdout
        try:
            envelope = json.loads(stdout)
            if isinstance(envelope, dict):
                if envelope.get("conversation_id"):
                    conversation_id = str(envelope["conversation_id"])
                if isinstance(envelope.get("response"), str):
                    response_text = envelope["response"].strip()
                usage = envelope.get("usage", {}) if isinstance(envelope.get("usage"), dict) else {}
                duration_seconds = envelope.get("duration_seconds")
        except json.JSONDecodeError:
            pass

        # Check if the model response already has structured worker JSON.
        # The bridge normalizer looks for JSON containing status, summary, patch, etc.
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "status" in parsed:
                denial_text = " ".join(
                    str(parsed.get(key, ""))
                    for key in ("summary", "verification_report", "error", "blockers")
                ).lower()
                if "permission" in denial_text and (
                    "auto-denied" in denial_text or "no output produced" in denial_text
                ):
                    parsed["status"] = "failed"
                    parsed["error"] = "worker_tool_permission_denied"
                    parsed["blockers"] = [
                        "Antigravity denied a required headless tool permission."
                    ]
                parsed["usage"] = usage
                parsed["duration_seconds"] = duration_seconds
                parsed.setdefault("conversation_id", conversation_id)
                print(json.dumps(parsed))
                return 0
        except json.JSONDecodeError:
            pass

        # Return structured format with model response
        status = "completed" if proc.returncode == 0 else "failed"
        blockers = [stderr] if proc.returncode != 0 and stderr else []
        if "permission" in response_text.lower() and "no output" in response_text.lower():
            status = "failed"
            blockers = [response_text[:1000]]
        output = {
            "status": status,
            "summary": response_text[:500] if response_text else (stderr[:500] if stderr else "No output from worker"),
            "patch": "",
            "tests_run": [],
            "verification_report": response_text if proc.returncode == 0 else stderr,
            "blockers": blockers,
            "usage": usage,
            "duration_seconds": duration_seconds,
            "conversation_id": conversation_id,
        }
        print(json.dumps(output))
        return 0

    except subprocess.TimeoutExpired:
        output = {
            "status": "failed",
            "summary": "Antigravity CLI execution timed out",
            "patch": "",
            "tests_run": [],
            "verification_report": "",
            "blockers": ["Execution timed out"],
        }
        print(json.dumps(output))
        return 0
    except Exception as exc:
        output = {
            "status": "failed",
            "summary": f"Error running Antigravity CLI: {exc}",
            "patch": "",
            "tests_run": [],
            "verification_report": "",
            "blockers": [str(exc)],
        }
        print(json.dumps(output))
        return 0


if __name__ == "__main__":
    sys.exit(main())
