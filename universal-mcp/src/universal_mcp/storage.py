"""Durable SQLite snapshots for MCP task and job state."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import (
    CommunicationEvent,
    QuestionRecord,
    TaskEvent,
    TaskRecord,
    TaskRequest,
    VerificationRun,
    WorkerResult,
)


class SQLiteStateStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS tasks "
                "(task_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs "
                "(job_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def save_task(self, record: TaskRecord) -> None:
        payload = record.to_dict(
            include_patch=True,
            include_raw=True,
            include_context=True,
            include_communication=True,
            include_verification_output=True,
            include_events=True,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO tasks(task_id, payload, updated_at) VALUES(?, ?, ?) "
                "ON CONFLICT(task_id) DO UPDATE SET payload=excluded.payload, "
                "updated_at=excluded.updated_at",
                (record.request.task_id, json.dumps(payload), record.updated_at),
            )

    def save_job(self, job: dict[str, Any]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs(job_id, payload, updated_at) VALUES(?, ?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET payload=excluded.payload, "
                "updated_at=excluded.updated_at",
                (job["job_id"], json.dumps(job), job["updated_at"]),
            )

    def load_tasks(self) -> dict[str, TaskRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT task_id, payload FROM tasks").fetchall()
        return {task_id: _record_from_dict(json.loads(payload)) for task_id, payload in rows}

    def load_jobs(self) -> dict[str, dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT job_id, payload FROM jobs").fetchall()
        jobs = {job_id: json.loads(payload) for job_id, payload in rows}
        for job in jobs.values():
            if job.get("status") in {"queued", "running"}:
                job["status"] = "interrupted"
                job["error"] = "MCP process restarted while the job was active"
        return jobs


def _result_from_dict(raw: dict[str, Any] | None) -> WorkerResult | None:
    if not raw:
        return None
    allowed = WorkerResult.__dataclass_fields__
    values = {key: value for key, value in raw.items() if key in allowed}
    return WorkerResult(**values)


def _record_from_dict(raw: dict[str, Any]) -> TaskRecord:
    task_raw = dict(raw["task"])
    context = task_raw.get("context_files", {})
    task_raw["context_files"] = context if isinstance(context, dict) else {}
    request = TaskRequest(**{
        key: value for key, value in task_raw.items()
        if key in TaskRequest.__dataclass_fields__
    })
    runs = []
    for item in raw.get("verification_runs", []):
        values = {
            key: value for key, value in item.items()
            if key in VerificationRun.__dataclass_fields__
        }
        runs.append(VerificationRun(**values))
    return TaskRecord(
        request=request,
        worker_id=raw["worker_id"],
        status=raw.get("status", "queued"),
        result=_result_from_dict(raw.get("result")),
        communication=[CommunicationEvent(**item) for item in raw.get("communication", [])],
        events=[TaskEvent(**item) for item in raw.get("events", [])],
        questions=[QuestionRecord(**item) for item in raw.get("questions", [])],
        pending_messages=list(raw.get("pending_messages", [])),
        parent_task_id=raw.get("parent_task_id"),
        role=raw.get("role", "implementation"),
        stage=raw.get("stage", "queued"),
        tracked_paths=list(raw.get("tracked_paths", [])),
        verification_workspace_hash=raw.get("verification_workspace_hash"),
        reviewer_task_ids=list(raw.get("reviewer_task_ids", [])),
        review_decisions=dict(raw.get("review_decisions", {})),
        conversation_id=raw.get("conversation_id"),
        has_blocking_question=bool(raw.get("has_blocking_question", False)),
        paused=bool(raw.get("paused", False)),
        cancelled=bool(raw.get("cancelled", False)),
        verification_runs=runs,
        verification_passed=bool(raw.get("verification_passed", False)),
        patch_applied=bool(raw.get("patch_applied", False)),
        astra_approved=bool(raw.get("astra_approved", False)),
        finalized=bool(raw.get("finalized", False)),
        created_at=raw.get("created_at"),
        updated_at=raw.get("updated_at"),
    )
