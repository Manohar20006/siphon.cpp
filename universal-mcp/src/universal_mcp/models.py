"""Data models shared by the MCP server and worker adapters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    """Return a stable, human-readable UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for one implementation worker."""

    worker_id: str
    source: str
    model: str
    transport: str = "openai_compatible"
    base_url: str | None = None
    api_key_env: str | None = None
    api_key_header: str = "Authorization"
    command: str | None = None
    timeout_seconds: int = 180
    enabled: bool = True
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WorkerConfig":
        worker_id = str(raw.get("id", "")).strip()
        source = str(raw.get("source", "")).strip()
        model = str(raw.get("model", "")).strip()
        transport = str(raw.get("transport", "openai_compatible")).strip()

        if not worker_id:
            raise ValueError("worker.id is required")
        if not source:
            raise ValueError(f"worker {worker_id!r} is missing source")
        if not model:
            raise ValueError(f"worker {worker_id!r} is missing model")
        if transport not in {"openai_compatible", "command"}:
            raise ValueError(
                f"worker {worker_id!r} has unsupported transport {transport!r}"
            )

        enabled = bool(raw.get("enabled", True))
        base_url = raw.get("base_url")
        command = raw.get("command")
        if enabled:
            if transport == "openai_compatible" and not base_url:
                raise ValueError(f"worker {worker_id!r} needs base_url")
            if transport == "command" and not command:
                raise ValueError(f"worker {worker_id!r} needs command")

        # Deliberately reject literal credentials in config files. Secrets belong
        # in the environment and are resolved only when a request is sent.
        if raw.get("api_key") is not None:
            raise ValueError(
                f"worker {worker_id!r} must use api_key_env instead of api_key"
            )

        timeout = int(raw.get("timeout_seconds", 180))
        if timeout < 1 or timeout > 3600:
            raise ValueError(f"worker {worker_id!r} timeout_seconds must be 1..3600")

        headers = raw.get("headers", {})
        if not isinstance(headers, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in headers.items()
        ):
            raise ValueError(f"worker {worker_id!r} headers must be a string map")
        sensitive_headers = {"authorization", "x-api-key", "api-key"}
        if any(key.lower() in sensitive_headers for key in headers):
            raise ValueError(
                f"worker {worker_id!r} must use api_key_env for credential headers"
            )
        api_key_header = str(raw.get("api_key_header", "Authorization")).strip()
        if not api_key_header:
            raise ValueError(f"worker {worker_id!r} api_key_header cannot be empty")

        return cls(
            worker_id=worker_id,
            source=source,
            model=model,
            transport=transport,
            base_url=str(base_url).strip() if base_url else None,
            api_key_env=str(raw["api_key_env"]).strip()
            if raw.get("api_key_env")
            else None,
            api_key_header=api_key_header,
            command=str(command) if command else None,
            timeout_seconds=timeout,
            enabled=bool(raw.get("enabled", True)),
            headers=dict(headers),
        )

    def public_dict(self) -> dict[str, Any]:
        """Return config safe to expose through MCP results."""

        return {
            "id": self.worker_id,
            "source": self.source,
            "model": self.model,
            "transport": self.transport,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_header": self.api_key_header,
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class TaskRequest:
    """The complete plan package sent by Astra to a worker."""

    task_id: str
    title: str
    objective: str
    implementation_plan: str
    verification_instructions: str
    verification_commands: list[list[str]] = field(default_factory=list)
    context_files: dict[str, str] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    follow_up_instructions: str = ""
    conversation_id: str | None = None
    operation_kind: str = "implementation"
    worker_tool_access: bool = False
    operation_targets: list[str] = field(default_factory=list)

    def to_dict(self, *, include_context: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_id": self.task_id,
            "title": self.title,
            "objective": self.objective,
            "implementation_plan": self.implementation_plan,
            "verification_instructions": self.verification_instructions,
            "verification_commands": self.verification_commands,
            "constraints": self.constraints,
            "follow_up_instructions": self.follow_up_instructions,
            "conversation_id": self.conversation_id,
            "operation_kind": self.operation_kind,
            "worker_tool_access": self.worker_tool_access,
            "operation_targets": self.operation_targets,
        }
        if include_context:
            result["context_files"] = self.context_files
        else:
            result["context_files"] = [
                {
                    "path": path,
                    "bytes": len(content.encode("utf-8")),
                }
                for path, content in self.context_files.items()
            ]
        return result


@dataclass
class WorkerResult:
    """Normalized result returned by any worker transport."""

    task_id: str
    worker_id: str
    status: str
    summary: str = ""
    patch: str = ""
    tests_run: list[str] = field(default_factory=list)
    verification_report: str = ""
    blockers: list[str] = field(default_factory=list)
    raw_output: str = ""
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    duration_seconds: float | None = None
    conversation_id: str | None = None
    event_type: str | None = None
    plan_step: str = ""
    next_step: str = ""
    question: str = ""
    blocking: bool = False
    assumptions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(
        self, *, include_patch: bool = False, include_raw: bool = False
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "status": self.status,
            "summary": self.summary,
            "patch_bytes": len(self.patch.encode("utf-8")),
            "patch_sha256": (
                hashlib.sha256(self.patch.encode("utf-8")).hexdigest()
                if self.patch
                else None
            ),
            "tests_run": self.tests_run,
            "verification_report": self.verification_report,
            "blockers": self.blockers,
            "error": self.error,
            "usage": self.usage,
            "duration_seconds": self.duration_seconds,
            "conversation_id": self.conversation_id,
            "event_type": self.event_type,
            "plan_step": self.plan_step,
            "next_step": self.next_step,
            "question": self.question,
            "blocking": self.blocking,
            "assumptions": self.assumptions,
            "risks": self.risks,
            "created_at": self.created_at,
        }
        if include_patch:
            result["patch"] = self.patch
        if include_raw:
            result["raw_output"] = self.raw_output
        return result


@dataclass
class VerificationRun:
    """Evidence from one Astra-specified command executed in the project."""

    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    created_at: str = field(default_factory=utc_now)

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self, *, include_output: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "argv": self.argv,
            "exit_code": self.exit_code,
            "passed": self.passed,
            "timed_out": self.timed_out,
            "duration_ms": self.duration_ms,
            "stdout_bytes": len(self.stdout.encode("utf-8")),
            "stderr_bytes": len(self.stderr.encode("utf-8")),
            "created_at": self.created_at,
        }
        if include_output:
            result["stdout"] = self.stdout
            result["stderr"] = self.stderr
        return result


@dataclass
class CommunicationEvent:
    """One explicit message in the Astra-worker review conversation."""

    sequence: int
    direction: str
    worker_id: str
    message: str
    kind: str
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "direction": self.direction,
            "worker_id": self.worker_id,
            "kind": self.kind,
            "message": self.message,
            "created_at": self.created_at,
        }


@dataclass
class TaskEvent:
    """One append-only structured supervision event."""

    sequence: int
    source: str
    type: str
    message: str
    plan_step: str = ""
    changed_paths: list[str] = field(default_factory=list)
    blocking: bool = False
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "source": self.source,
            "type": self.type,
            "message": self.message,
            "plan_step": self.plan_step,
            "changed_paths": self.changed_paths,
            "blocking": self.blocking,
            "created_at": self.created_at,
        }


@dataclass
class QuestionRecord:
    """A worker question with explicit blocking and resolution semantics."""

    question_id: str
    question: str
    blocking: bool
    status: str = "pending"
    answer: str | None = None
    created_at: str = field(default_factory=utc_now)
    answered_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "blocking": self.blocking,
            "status": self.status,
            "answer": self.answer,
            "created_at": self.created_at,
            "answered_at": self.answered_at,
        }


@dataclass
class TaskRecord:
    """In-memory task history for the lifetime of one MCP process."""

    request: TaskRequest
    worker_id: str
    status: str = "queued"
    result: WorkerResult | None = None
    communication: list[CommunicationEvent] = field(default_factory=list)
    events: list[TaskEvent] = field(default_factory=list)
    questions: list[QuestionRecord] = field(default_factory=list)
    pending_messages: list[dict[str, str]] = field(default_factory=list)
    parent_task_id: str | None = None
    role: str = "implementation"
    stage: str = "queued"
    tracked_paths: list[str] = field(default_factory=list)
    verification_workspace_hash: str | None = None
    reviewer_task_ids: list[str] = field(default_factory=list)
    review_decisions: dict[str, str] = field(default_factory=dict)
    conversation_id: str | None = None
    has_blocking_question: bool = False
    paused: bool = False
    cancelled: bool = False
    verification_runs: list[VerificationRun] = field(default_factory=list)
    verification_passed: bool = False
    patch_applied: bool = False
    astra_approved: bool = False
    finalized: bool = False
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(
        self,
        *,
        include_patch: bool = False,
        include_raw: bool = False,
        include_context: bool = False,
        include_communication: bool = True,
        include_verification_output: bool = False,
        include_events: bool = False,
    ) -> dict[str, Any]:
        return {
            "task": self.request.to_dict(include_context=include_context),
            "worker_id": self.worker_id,
            "status": self.status,
            "result": (
                self.result.to_dict(
                    include_patch=include_patch, include_raw=include_raw
                )
                if self.result
                else None
            ),
            "communication": (
                [event.to_dict() for event in self.communication]
                if include_communication
                else []
            ),
            "events": (
                [event.to_dict() for event in self.events]
                if include_events
                else []
            ),
            "questions": [question.to_dict() for question in self.questions],
            "pending_messages": list(self.pending_messages),
            "parent_task_id": self.parent_task_id,
            "role": self.role,
            "stage": self.stage,
            "tracked_paths": self.tracked_paths,
            "verification_workspace_hash": self.verification_workspace_hash,
            "reviewer_task_ids": self.reviewer_task_ids,
            "review_decisions": self.review_decisions,
            "verification_runs": [
                run.to_dict(include_output=include_verification_output)
                for run in self.verification_runs
            ],
            "verification_passed": self.verification_passed,
            "patch_applied": self.patch_applied,
            "astra_approved": self.astra_approved,
            "finalized": self.finalized,
            "conversation_id": self.conversation_id,
            "has_blocking_question": self.has_blocking_question,
            "paused": self.paused,
            "cancelled": self.cancelled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
