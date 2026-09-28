"""Minimal stdio MCP server for the Astra worker bridge."""

from __future__ import annotations

import json
import sys
from typing import Any

from .config import load_config
from .orchestrator import Orchestrator, OrchestratorError
from .providers import ProviderError


SERVER_NAME = "universal-worker-bridge"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOL_VERSIONS = {
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _tool_result(value: Any, is_error: bool = False) -> dict[str, Any]:
    if isinstance(value, dict):
        visible = {
            key: value[key]
            for key in (
                "task_id",
                "worker_id",
                "status",
                "verification_passed",
                "finalized",
                "error",
                "next_action",
            )
            if key in value
        }
        text_value: Any = visible or {"message": "Structured result attached."}
    else:
        text_value = value
    result: dict[str, Any] = {
        "content": [{"type": "text", "text": _json_text(text_value)}],
        "isError": is_error,
    }
    if isinstance(value, dict):
        result["structuredContent"] = value
    return result


def _compact_status(value: dict[str, Any]) -> dict[str, Any]:
    task = value.get("task", {})
    result = value.get("result") or {}
    events = value.get("events", [])
    return {
        "task_id": task.get("task_id"),
        "worker_id": value.get("worker_id"),
        "status": value.get("status"),
        "stage": value.get("stage"),
        "summary": result.get("summary", ""),
        "event_type": result.get("event_type"),
        "next_step": result.get("next_step", ""),
        "pending_questions": [
            question for question in value.get("questions", [])
            if question.get("status") == "pending"
        ],
        "latest_event": events[-1] if events else None,
        "active_jobs": value.get("active_jobs", []),
        "verification_passed": value.get("verification_passed", False),
        "verification_workspace_hash": value.get("verification_workspace_hash"),
        "review_decisions": value.get("review_decisions", {}),
        "finalized": value.get("finalized", False),
    }


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "readiness_check",
            "description": "Run local configuration and dependency checks before production use.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
        },
        {
            "name": "list_workers",
            "description": (
                "List configured implementation workers. Workers can come from "
                "Antigravity or a hosted endpoint such as Kaggle Qwen 27B."
            ),
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "worker_health",
            "description": "Check whether one worker or all configured workers are reachable.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "worker_id": {
                        "type": "string",
                        "description": "Worker id, or all. Defaults to all.",
                    }
                },
                "additionalProperties": False,
            },
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": True,
            },
        },
        {
            "name": "delegate_workspace_operation",
            "description": (
                "Use Antigravity as Codex's default workspace worker. The MCP captures "
                "bounded file or command evidence for Antigravity to analyze; edits return "
                "a reviewable patch."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["operation", "instructions"],
                "properties": {
                    "task_id": {"type": "string"},
                    "conversation_id": {"type": "string"},
                    "worker_id": {"type": "string", "default": "auto"},
                    "operation": {"type": "string", "enum": ["inspect", "command", "edit"]},
                    "instructions": {"type": "string", "minLength": 1},
                    "targets": {"type": "array", "items": {"type": "string"}},
                    "command_argv": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 64,
                        "items": {"type": "string", "minLength": 1},
                        "description": "Exact argv array required for command operations.",
                    },
                    "timeout_seconds": {
                        "type": "integer", "minimum": 1, "maximum": 900, "default": 180
                    },
                    "confirm": {
                        "type": "boolean",
                        "default": False,
                    "description": "Required for command execution and edit generation.",
                    },
                },
                "additionalProperties": False,
            },
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": True,
                "idempotentHint": False,
                "openWorldHint": True,
            },
        },
        {
            "name": "dispatch_worker_task",
            "description": (
                "Send one complete Astra plan package to an implementation worker. "
                "Astra must provide the full implementation plan and the exact "
                "verification procedure in this call. The server does not call "
                "Astra for individual steps."
            ),
            "inputSchema": {
                "type": "object",
                "required": [
                    "objective",
                    "implementation_plan",
                    "verification_instructions",
                    "verification_commands",
                ],
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Optional stable id for retries or follow-up.",
                    },
                    "job_id": {
                        "type": "string",
                        "description": "Optional stable background job id.",
                    },
                    "async": {
                        "type": "boolean",
                        "default": True,
                        "description": "Return immediately and run the worker in background.",
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": (
                            "Optional existing worker conversation to resume. The bridge "
                            "stores a conversation id returned by the worker automatically."
                        ),
                    },
                    "title": {"type": "string"},
                    "worker_id": {
                        "type": "string",
                        "description": "Configured worker id, or auto.",
                    },
                    "preferred_source": {
                        "type": "string",
                        "description": "Optional source selector such as antigravity or kaggle.",
                    },
                    "objective": {
                        "type": "string",
                        "description": "The exact result the worker must deliver.",
                    },
                    "implementation_plan": {
                        "type": "string",
                        "description": "The complete plan already produced by Astra.",
                    },
                    "verification_instructions": {
                        "type": "string",
                        "description": (
                            "Astra-authored instructions describing exactly how "
                            "the worker must verify its implementation."
                        ),
                    },
                    "verification_commands": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 12,
                        "description": (
                            "Astra-selected commands that the MCP will execute as "
                            "argv arrays after the worker patch is applied."
                        ),
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 64,
                            "items": {"type": "string", "minLength": 1},
                        },
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "context_files": {
                        "type": "array",
                        "description": (
                            "Small text files needed by the worker. Supply content "
                            "directly, or omit content to let the local bridge read "
                            "the relative file from the Codex project."
                        ),
                        "items": {
                            "type": "object",
                            "required": ["path"],
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": False,
                "openWorldHint": True,
            },
        },
        {
            "name": "get_task_status",
            "description": (
                "Retrieve a task result, patch, worker report, verification report, "
                "communication history, and finalization gate state."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["task_id"],
                "properties": {
                    "task_id": {"type": "string"},
                    "include_patch": {"type": "boolean", "default": False},
                    "include_raw": {"type": "boolean", "default": False},
                    "include_context": {"type": "boolean", "default": False},
                    "include_communication": {"type": "boolean", "default": False},
                    "include_events": {"type": "boolean", "default": False},
                    "compact": {"type": "boolean", "default": True},
                    "include_verification_output": {
                        "type": "boolean",
                        "default": False,
                    },
                },
                "additionalProperties": False,
            },
            "annotations": {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "apply_worker_patch",
            "description": (
                "Apply a reviewed worker unified diff to the Codex project workspace. "
                "The patch is checked with git before it is applied."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["task_id", "confirm"],
                "properties": {
                    "task_id": {"type": "string"},
                    "confirm": {
                        "type": "boolean",
                        "description": "Set true only after reviewing the worker result.",
                    },
                },
                "additionalProperties": False,
            },
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": True,
                "idempotentHint": False,
                "openWorldHint": False,
            },
        },
        {
            "name": "continue_worker_task",
            "description": (
                "Send correction or follow-up instructions from Astra to the same "
                "worker after reviewing its report. The worker must implement the "
                "feedback and run the verification procedure again."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["task_id", "feedback"],
                "properties": {
                    "task_id": {"type": "string"},
                    "feedback": {
                        "type": "string",
                        "description": "Astra's correction or follow-up instructions.",
                    },
                    "verification_instructions": {
                        "type": "string",
                        "description": "Optional revised verification procedure.",
                    },
                    "verification_commands": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 12,
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 64,
                            "items": {"type": "string", "minLength": 1},
                        },
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "context_files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["path"],
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": False,
                "openWorldHint": True,
            },
        },
        {
            "name": "approve_task",
            "description": (
                "Record Astra's approval after reviewing the worker communication, "
                "verification report, blockers, and applied patch."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["task_id", "approval_note"],
                "properties": {
                    "task_id": {"type": "string"},
                    "approval_note": {
                        "type": "string",
                        "description": "Astra's concise acceptance decision.",
                    },
                },
                "additionalProperties": False,
            },
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": False,
                "openWorldHint": False,
            },
        },
        {
            "name": "finalize_project",
            "description": (
                "Submit the final project version only after worker communication, "
                "verification, patch application, and Astra approval have all passed."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["task_id"],
                "properties": {"task_id": {"type": "string"}},
                "additionalProperties": False,
            },
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": False,
            },
        },
        {
            "name": "run_worker_verification",
            "description": (
                "Execute Astra's structured verification commands in the Codex "
                "project, send the captured evidence to the worker, and require "
                "the worker to interpret the real results."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["task_id", "confirm"],
                "properties": {
                    "task_id": {"type": "string"},
                    "confirm": {
                        "type": "boolean",
                        "description": "Confirms execution of the planned commands.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 900,
                        "default": 180,
                    },
                },
                "additionalProperties": False,
            },
            "annotations": {
                "readOnlyHint": False,
                "destructiveHint": True,
                "idempotentHint": False,
                "openWorldHint": False,
            },
        },
        {
            "name": "get_task_events",
            "description": "Get compact supervision events after a sequence cursor.",
            "inputSchema": {
                "type": "object",
                "required": ["task_id"],
                "properties": {
                    "task_id": {"type": "string"},
                    "after_sequence": {"type": "integer", "minimum": 0, "default": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
                },
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True, "idempotentHint": True},
        },
        {
            "name": "get_worker_job",
            "description": "Poll a background worker call without loading task details.",
            "inputSchema": {
                "type": "object",
                "required": ["job_id"],
                "properties": {"job_id": {"type": "string"}},
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True, "idempotentHint": True},
        },
        {
            "name": "answer_worker_question",
            "description": (
                "Resolve one identified worker question and optionally continue its "
                "conversation with the answer."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["task_id", "question_id", "answer"],
                "properties": {
                    "task_id": {"type": "string"},
                    "question_id": {"type": "string"},
                    "answer": {"type": "string"},
                    "continue_worker": {"type": "boolean", "default": True},
                    "job_id": {"type": "string"},
                    "async": {"type": "boolean", "default": True},
                },
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": False, "idempotentHint": False, "openWorldHint": True},
        },
        {
            "name": "dispatch_worker_review",
            "description": (
                "Start an independent clean-context worker review of an implementation, "
                "its requirements, patch, and captured verification evidence."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["task_id", "review_instructions"],
                "properties": {
                    "task_id": {"type": "string"},
                    "review_task_id": {"type": "string"},
                    "reviewer_worker_id": {"type": "string", "default": "auto"},
                    "review_instructions": {"type": "string"},
                    "job_id": {"type": "string"},
                    "async": {"type": "boolean", "default": True},
                },
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": False, "idempotentHint": False, "openWorldHint": True},
        },
        {
            "name": "send_worker_message",
            "description": "Answer a worker question or steer its resumable conversation.",
            "inputSchema": {
                "type": "object",
                "required": ["task_id", "message"],
                "properties": {
                    "task_id": {"type": "string"},
                    "message": {"type": "string"},
                    "message_type": {"type": "string", "default": "steering"},
                    "job_id": {"type": "string"},
                    "async": {"type": "boolean", "default": True},
                },
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": False, "idempotentHint": False, "openWorldHint": True},
        },
        {
            "name": "resolve_worker_review",
            "description": "Record Codex's resolution of one independent review before approval.",
            "inputSchema": {
                "type": "object",
                "required": ["task_id", "review_task_id", "resolution_note"],
                "properties": {
                    "task_id": {"type": "string"},
                    "review_task_id": {"type": "string"},
                    "resolution_note": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": False, "idempotentHint": False},
        },
        *[
            {
                "name": tool_name,
                "description": description,
                "inputSchema": {
                    "type": "object",
                    "required": ["task_id"],
                    "properties": {
                        "task_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "annotations": {"readOnlyHint": False, "idempotentHint": True},
            }
            for tool_name, description in (
                ("pause_worker_task", "Pause worker continuation."),
                ("resume_worker_task", "Resume a paused worker task."),
                ("cancel_worker_task", "Cancel worker continuation."),
            )
        ],
    ]


def _call_tool(orchestrator: Orchestrator, name: str, arguments: Any) -> dict[str, Any]:
    args = arguments if isinstance(arguments, dict) else {}
    if name == "list_workers":
        return _tool_result(
            {
                "project_root": str(orchestrator.config.project_root),
                "workers": orchestrator.list_workers(),
                "planner": "Codex/Astra",
                "worker_sources": ["Antigravity", "Kaggle OpenAI-compatible endpoint"],
            }
        )
    if name == "readiness_check":
        return _tool_result(orchestrator.readiness())
    if name == "worker_health":
        return _tool_result(orchestrator.health(args.get("worker_id")))
    if name == "delegate_workspace_operation":
        return _tool_result(orchestrator.delegate_workspace_operation(args))
    if name == "dispatch_worker_task":
        if bool(args.get("async", True)):
            return _tool_result(orchestrator.dispatch_async(args))
        return _tool_result(orchestrator.dispatch(args))
    if name == "get_task_status":
        status = orchestrator.task_status(
                str(args.get("task_id", "")),
                include_patch=bool(args.get("include_patch", False)),
                include_raw=bool(args.get("include_raw", False)),
                include_context=bool(args.get("include_context", False)),
                include_communication=bool(
                    args.get("include_communication", False)
                ),
                include_verification_output=bool(
                    args.get("include_verification_output", False)
                ),
                include_events=(
                    bool(args.get("include_events", False))
                    or bool(args.get("compact", True))
                ),
            )
        return _tool_result(
            _compact_status(status) if bool(args.get("compact", True)) else status
        )
    if name == "continue_worker_task":
        return _tool_result(orchestrator.continue_task(args))
    if name == "approve_task":
        return _tool_result(
            orchestrator.approve_task(
                str(args.get("task_id", "")),
                str(args.get("approval_note", "")),
            )
        )
    if name == "finalize_project":
        return _tool_result(
            orchestrator.finalize_project(str(args.get("task_id", "")))
        )
    if name == "apply_worker_patch":
        return _tool_result(
            orchestrator.apply_worker_patch(
                str(args.get("task_id", "")), bool(args.get("confirm", False))
            )
        )
    if name == "run_worker_verification":
        return _tool_result(
            orchestrator.run_verification(
                str(args.get("task_id", "")),
                bool(args.get("confirm", False)),
                int(args.get("timeout_seconds", 180)),
            )
        )
    if name == "get_task_events":
        return _tool_result(orchestrator.get_task_events(
            str(args.get("task_id", "")),
            int(args.get("after_sequence", 0)),
            int(args.get("limit", 50)),
        ))
    if name == "get_worker_job":
        return _tool_result(orchestrator.get_job(str(args.get("job_id", ""))))
    if name == "answer_worker_question":
        if bool(args.get("async", True)) and bool(args.get("continue_worker", True)):
            return _tool_result(orchestrator.answer_question_async(
                str(args.get("task_id", "")),
                str(args.get("question_id", "")),
                str(args.get("answer", "")),
                str(args.get("job_id", "")) or None,
            ))
        return _tool_result(orchestrator.answer_question(
            str(args.get("task_id", "")),
            str(args.get("question_id", "")),
            str(args.get("answer", "")),
            continue_worker=bool(args.get("continue_worker", True)),
        ))
    if name == "dispatch_worker_review":
        if bool(args.get("async", True)):
            return _tool_result(orchestrator.dispatch_review_async(args))
        return _tool_result(orchestrator.dispatch_review(args))
    if name == "send_worker_message":
        if bool(args.get("async", True)):
            return _tool_result(orchestrator.send_worker_message_async(
                str(args.get("task_id", "")),
                str(args.get("message", "")),
                str(args.get("message_type", "steering")),
                str(args.get("job_id", "")) or None,
            ))
        return _tool_result(orchestrator.send_worker_message(
            str(args.get("task_id", "")),
            str(args.get("message", "")),
            str(args.get("message_type", "steering")),
        ))
    if name == "resolve_worker_review":
        return _tool_result(orchestrator.resolve_review(
            str(args.get("task_id", "")),
            str(args.get("review_task_id", "")),
            str(args.get("resolution_note", "")),
        ))
    if name == "pause_worker_task":
        return _tool_result(orchestrator.pause_task(
            str(args.get("task_id", "")), str(args.get("reason", ""))
        ))
    if name == "resume_worker_task":
        return _tool_result(orchestrator.resume_task(str(args.get("task_id", ""))))
    if name == "cancel_worker_task":
        return _tool_result(orchestrator.cancel_task(
            str(args.get("task_id", "")), str(args.get("reason", ""))
        ))
    raise OrchestratorError(f"unknown tool: {name}")


def _handle_request(orchestrator: Orchestrator, request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    request_id = request.get("id")

    if method in {"notifications/initialized", "notifications/cancelled"}:
        return None
    if method == "initialize":
        params = request.get("params") or {}
        requested = params.get("protocolVersion")
        version = (
            requested
            if requested in SUPPORTED_PROTOCOL_VERSIONS
            else "2024-11-05"
        )
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": (
                    "Use Antigravity as the default worker for substantive workspace "
                    "inspection, command execution, and edits through "
                    "delegate_workspace_operation. Codex still directly checks critical "
                    "evidence before approval. Codex/Astra plans once and supplies both implementation steps "
                    "and argv-form verification commands. Dispatch to a worker, "
                    "review its compact report, apply its patch, run real checks "
                    "with run_worker_verification, then approve and finalize. Use "
                    "continue_worker_task only for corrections. Do not request raw "
                    "output, full context, or full patches unless diagnosis requires it."
                ),
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": _tools()}}
    if method == "tools/call":
        params = request.get("params") or {}
        try:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": _call_tool(
                    orchestrator, str(params.get("name", "")), params.get("arguments")
                ),
            }
        except (OrchestratorError, ProviderError, ValueError, KeyError) as error:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": _tool_result({"error": str(error)}, is_error=True),
            }
    if request_id is None:
        return None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def _read_message(stream) -> dict[str, Any] | None:
    """Read newline-delimited JSON and tolerate legacy Content-Length framing."""

    while True:
        first = stream.readline()
        if not first:
            return None
        if first.strip():
            break

    if first.lower().startswith(b"content-length:"):
        headers = [first]
        while True:
            line = stream.readline()
            if not line or line in {b"\r\n", b"\n"}:
                break
            headers.append(line)
        try:
            length = int(first.split(b":", 1)[1].strip())
        except (IndexError, ValueError) as error:
            raise ValueError("invalid Content-Length header") from error
        body = stream.read(length)
    else:
        body = first

    parsed = json.loads(body.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("MCP message must be a JSON object")
    return parsed


def _write_message(stream, response: dict[str, Any]) -> None:
    encoded = (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
    stream.write(encoded)
    stream.flush()


def main() -> int:
    try:
        orchestrator = Orchestrator(load_config())
    except Exception as error:
        print(f"{SERVER_NAME}: configuration error: {error}", file=sys.stderr)
        return 1

    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    while True:
        try:
            request = _read_message(input_stream)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            print(f"{SERVER_NAME}: protocol error: {error}", file=sys.stderr)
            return 1
        if request is None:
            return 0
        try:
            response = _handle_request(orchestrator, request)
        except Exception as error:
            print(f"{SERVER_NAME}: internal error: {error}", file=sys.stderr)
            if request.get("id") is None:
                continue
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32603, "message": "internal server error"},
            }
        if response is not None:
            _write_message(output_stream, response)


if __name__ == "__main__":
    raise SystemExit(main())
