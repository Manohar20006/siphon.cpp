"""Astra-to-worker orchestration and safe patch handling."""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
import hashlib
import json
import shutil
import tempfile
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from .config import AppConfig
from .models import (
    CommunicationEvent,
    QuestionRecord,
    TaskEvent,
    TaskRecord,
    TaskRequest,
    VerificationRun,
    WorkerResult,
    utc_now,
)
from .providers import HealthResult, cancel_worker_process, provider_for
from .storage import SQLiteStateStore


_MAX_CONTEXT_FILE_BYTES = 256 * 1024
_MAX_CONTEXT_BYTES = 1_500 * 1024
_MAX_PATCH_BYTES = 10 * 1024 * 1024
_MAX_VERIFICATION_COMMANDS = 12
_MAX_VERIFICATION_ARGS = 64
_MAX_VERIFICATION_OUTPUT_BYTES = 32 * 1024


class OrchestratorError(RuntimeError):
    """An expected, user-correctable orchestration error."""


def _safe_relative_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise OrchestratorError(f"path must stay inside project_root: {value!r}")
    resolved = (project_root / path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise OrchestratorError(f"path escapes project_root: {value!r}") from error
    return resolved


def _context_files(
    project_root: Path, entries: Any
) -> dict[str, str]:
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise OrchestratorError("context_files must be an array")

    result: dict[str, str] = {}
    total_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise OrchestratorError(
                "each context_files item must contain a relative path"
            )
        display_path = PurePosixPath(entry["path"]).as_posix()
        resolved = _safe_relative_path(project_root, display_path)

        if "content" in entry:
            content = entry["content"]
            if not isinstance(content, str):
                raise OrchestratorError(f"context content must be text: {display_path}")
        else:
            if not resolved.is_file():
                raise OrchestratorError(f"context file does not exist: {display_path}")
            try:
                content = resolved.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise OrchestratorError(
                    f"context file is not UTF-8 text: {display_path}"
                ) from error

        size = len(content.encode("utf-8"))
        if size > _MAX_CONTEXT_FILE_BYTES:
            raise OrchestratorError(
                f"context file is larger than 256 KiB: {display_path}"
            )
        total_bytes += size
        if total_bytes > _MAX_CONTEXT_BYTES:
            raise OrchestratorError("combined context_files exceeds 1.5 MiB")
        result[display_path] = content
    return result


def _patch_paths(patch: str) -> set[str]:
    """Extract and validate paths before passing a patch to git."""

    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                for token in parts[2:4]:
                    path = token[2:] if token.startswith(("a/", "b/")) else token
                    paths.add(path)
        elif line.startswith("--- ") or line.startswith("+++ "):
            path = line[4:].split("\t", 1)[0].strip()
            if path == "/dev/null":
                continue
            paths.add(path[2:] if path.startswith(("a/", "b/")) else path)

    for path in paths:
        if Path(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise OrchestratorError(f"worker patch contains unsafe path: {path}")
    return paths


def _verification_commands(value: Any) -> list[list[str]]:
    if not isinstance(value, list) or not value:
        raise OrchestratorError(
            "verification_commands must contain at least one argv array"
        )
    if len(value) > _MAX_VERIFICATION_COMMANDS:
        raise OrchestratorError(
            f"verification_commands cannot exceed {_MAX_VERIFICATION_COMMANDS} commands"
        )

    commands: list[list[str]] = []
    for index, command in enumerate(value):
        if not isinstance(command, list) or not command:
            raise OrchestratorError(
                f"verification command {index + 1} must be a non-empty argv array"
            )
        if len(command) > _MAX_VERIFICATION_ARGS:
            raise OrchestratorError(
                f"verification command {index + 1} has too many arguments"
            )
        argv: list[str] = []
        for argument in command:
            if not isinstance(argument, str) or not argument or "\0" in argument:
                raise OrchestratorError(
                    f"verification command {index + 1} contains an invalid argument"
                )
            argv.append(argument)
        commands.append(argv)
    return commands


def _trim_output(value: bytes) -> str:
    if len(value) <= _MAX_VERIFICATION_OUTPUT_BYTES:
        return value.decode("utf-8", errors="replace")
    marker = b"\n...[output truncated to final 32 KiB]...\n"
    tail_size = _MAX_VERIFICATION_OUTPUT_BYTES - len(marker)
    return (marker + value[-tail_size:]).decode("utf-8", errors="replace")


def _refresh_context_files(
    project_root: Path, current: dict[str, str]
) -> dict[str, str]:
    refreshed: dict[str, str] = {}
    for path, previous in current.items():
        resolved = _safe_relative_path(project_root, path)
        if path.startswith("__") and not resolved.exists():
            # Synthetic evidence (review requirements, command output, patches) is
            # intentionally not materialized in the user's workspace.
            content = previous
        elif resolved.is_file():
            try:
                content = resolved.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = previous
        else:
            content = f"(File is absent from the current workspace: {path})"
        if len(content.encode("utf-8")) > _MAX_CONTEXT_FILE_BYTES:
            content = previous
        refreshed[path] = content
    return refreshed


def _workspace_fingerprint(project_root: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    digest.update(head.stdout)
    if paths:
        for relative in sorted(set(paths)):
            digest.update(relative.encode("utf-8"))
            resolved = _safe_relative_path(project_root, relative)
            if resolved.is_file():
                digest.update(resolved.read_bytes())
            else:
                digest.update(b"<absent>")
    else:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=False,
        )
        digest.update(status.stdout)
    return digest.hexdigest()


def _preflight_patch(project_root: Path, patch: str, paths: set[str]) -> list[dict[str, Any]]:
    """Apply a patch in an isolated minimal tree and run deterministic syntax checks."""

    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="universal-mcp-preflight-") as directory:
        root = Path(directory)
        for relative in paths:
            source = _safe_relative_path(project_root, relative)
            destination = root / relative
            if source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        applied = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=root,
            input=patch.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        checks.append({
            "check": "isolated_patch_apply",
            "passed": applied.returncode == 0,
            "detail": applied.stderr.decode("utf-8", errors="replace").strip(),
        })
        if applied.returncode != 0:
            return checks

        for relative in sorted(paths):
            target = root / relative
            if not target.is_file():
                continue
            suffix = target.suffix.lower()
            command: list[str] | None = None
            if suffix == ".py":
                command = ["python3", "-m", "py_compile", str(target)]
            elif suffix in {".sh", ".bash"}:
                command = ["bash", "-n", str(target)]
            elif suffix in {".js", ".mjs", ".cjs"} and shutil.which("node"):
                command = ["node", "--check", str(target)]
            if command:
                completed = subprocess.run(
                    command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=30, check=False,
                )
                checks.append({
                    "check": f"syntax:{relative}",
                    "passed": completed.returncode == 0,
                    "detail": _trim_output(completed.stderr),
                })
            elif suffix == ".json":
                try:
                    json.loads(target.read_text(encoding="utf-8"))
                    error = ""
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    error = str(exc)
                checks.append({"check": f"json:{relative}", "passed": not error, "detail": error})
            elif suffix == ".toml":
                try:
                    tomllib.loads(target.read_text(encoding="utf-8"))
                    error = ""
                except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
                    error = str(exc)
                checks.append({"check": f"toml:{relative}", "passed": not error, "detail": error})
    return checks


class Orchestrator:
    """Route complete Astra task packages to configured workers."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._store = SQLiteStateStore(config.storage_path) if config.storage_path else None
        self.tasks: dict[str, TaskRecord] = self._store.load_tasks() if self._store else {}
        self.jobs: dict[str, dict[str, Any]] = self._store.load_jobs() if self._store else {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="universal-mcp")

    def _persist_record(self, record: TaskRecord) -> None:
        if self._store:
            self._store.save_task(record)

    def _persist_job(self, job: dict[str, Any]) -> None:
        if self._store:
            self._store.save_job(job)

    def list_workers(self) -> list[dict[str, Any]]:
        return [worker.public_dict() for worker in self.config.workers]

    def _assert_command_allowed(self, argv: list[str]) -> None:
        prefixes = self.config.allowed_command_prefixes
        if not prefixes:
            return
        if not any(tuple(argv[:len(prefix)]) == prefix for prefix in prefixes):
            rendered = " ".join(argv)
            raise OrchestratorError(
                f"command is not allowed by allowed_command_prefixes: {rendered}"
            )

    def readiness(self) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        checks.append({"name": "project_root", "passed": self.config.project_root.is_dir()})
        enabled = [worker for worker in self.config.workers if worker.enabled]
        checks.append({"name": "enabled_worker", "passed": bool(enabled), "count": len(enabled)})
        storage_ok = self.config.storage_path is not None
        storage_detail = "persistent SQLite storage is not configured"
        if self.config.storage_path:
            try:
                self.config.storage_path.parent.mkdir(parents=True, exist_ok=True)
                with self.config.storage_path.open("ab"):
                    pass
                storage_ok = True
                storage_detail = str(self.config.storage_path)
            except OSError as error:
                storage_ok = False
                storage_detail = str(error)
        checks.append({"name": "persistent_storage", "passed": storage_ok, "detail": storage_detail})
        checks.append({
            "name": "command_allowlist",
            "passed": bool(self.config.allowed_command_prefixes),
            "detail": "configured" if self.config.allowed_command_prefixes else "empty; all argv commands are accepted",
        })
        for worker in enabled:
            if worker.transport == "command":
                executable = (worker.command or "").split()[0]
                checks.append({
                    "name": f"worker_executable:{worker.worker_id}",
                    "passed": bool(executable and shutil.which(executable)),
                    "detail": executable,
                })
        return {
            "ready": all(check["passed"] for check in checks),
            "checks": checks,
            "project_root": str(self.config.project_root),
        }

    @staticmethod
    def _append_communication(
        record: TaskRecord, direction: str, kind: str, message: str
    ) -> None:
        record.communication.append(
            CommunicationEvent(
                sequence=len(record.communication) + 1,
                direction=direction,
                worker_id=record.worker_id,
                kind=kind,
                message=message,
            )
        )

    @staticmethod
    def _append_event(
        record: TaskRecord,
        source: str,
        event_type: str,
        message: str,
        *,
        plan_step: str = "",
        changed_paths: list[str] | None = None,
        blocking: bool = False,
    ) -> None:
        record.events.append(
            TaskEvent(
                sequence=len(record.events) + 1,
                source=source,
                type=event_type,
                message=message,
                plan_step=plan_step,
                changed_paths=changed_paths or [],
                blocking=blocking,
            )
        )

    def _store_worker_result(
        self,
        record: TaskRecord,
        result: WorkerResult,
        *,
        invalidate_verification: bool = True,
    ) -> None:
        record.result = result
        record.status = "cancelled" if record.cancelled else result.status
        record.stage = "cancelled" if record.cancelled else (
            "awaiting_input" if result.blocking and result.question
            else "ready_for_review" if result.status == "completed"
            else "blocked" if result.status == "failed"
            else "worker_checkpoint"
        )
        record.updated_at = result.created_at
        record.patch_applied = not bool(result.patch.strip())
        record.astra_approved = False
        if result.conversation_id:
            record.conversation_id = result.conversation_id
        if result.question:
            normalized_question = " ".join(result.question.lower().split())
            existing = next(
                (
                    question for question in record.questions
                    if question.status == "pending"
                    and " ".join(question.question.lower().split()) == normalized_question
                ),
                None,
            )
            if existing:
                existing.blocking = existing.blocking or result.blocking
            else:
                record.questions.append(
                    QuestionRecord(
                        question_id=f"question-{uuid.uuid4().hex[:10]}",
                        question=result.question,
                        blocking=result.blocking,
                    )
                )
        record.has_blocking_question = any(
            question.blocking and question.status == "pending"
            for question in record.questions
        )
        if invalidate_verification:
            record.verification_runs = []
            record.verification_passed = False
        verification = result.verification_report or "(no verification report)"
        blockers = ", ".join(result.blockers) or "none"
        self._append_communication(
            record,
            "worker_to_astra",
            "worker_report",
            (
                f"Status: {result.status}\n"
                f"Summary: {result.summary or '(no summary)'}\n"
                f"Verification: {verification}\n"
                f"Blockers: {blockers}"
            ),
        )
        event_type = result.event_type or (
            "question" if result.question else
            "blocked" if result.status == "failed" or result.blockers else
            "ready_for_review" if result.status == "completed" else "checkpoint"
        )
        changed_paths = sorted(_patch_paths(result.patch)) if result.patch.strip() else []
        record.tracked_paths = sorted(set(record.tracked_paths).union(changed_paths))
        self._append_event(
            record,
            "worker",
            event_type,
            result.question or result.summary or f"Worker status: {result.status}",
            plan_step=result.plan_step,
            changed_paths=changed_paths,
            blocking=result.blocking,
        )
        self._persist_record(record)

    def _record_for_task(self, task_id: str) -> TaskRecord:
        if task_id not in self.tasks:
            raise OrchestratorError(f"unknown task: {task_id}")
        return self.tasks[task_id]

    def health(self, worker_id: str | None = None) -> list[dict[str, Any]]:
        workers = (
            [self.config.worker(worker_id)]
            if worker_id and worker_id != "all"
            else list(self.config.workers)
        )
        results: list[dict[str, Any]] = []
        for worker in workers:
            if not worker.enabled:
                results.append(
                    HealthResult(worker.worker_id, False, "worker is disabled").to_dict()
                )
                continue
            results.append(provider_for(worker).health(worker).to_dict())
        return results

    def _select_worker(
        self, worker_id: str | None, preferred_source: str | None
    ):
        if worker_id and worker_id != "auto":
            try:
                worker = self.config.worker(worker_id)
            except KeyError as error:
                raise OrchestratorError(str(error)) from error
            if not worker.enabled:
                raise OrchestratorError(f"worker is disabled: {worker_id}")
            return worker

        candidates = [worker for worker in self.config.workers if worker.enabled]
        if preferred_source:
            preferred = [
                worker for worker in candidates if worker.source == preferred_source
            ]
            if preferred:
                candidates = preferred
        if not candidates:
            raise OrchestratorError(
                "no enabled workers are configured; add workers.json first"
            )
        return candidates[0]

    def dispatch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        objective = str(arguments.get("objective", "")).strip()
        implementation_plan = str(arguments.get("implementation_plan", "")).strip()
        verification = str(arguments.get("verification_instructions", "")).strip()
        if not objective:
            raise OrchestratorError("objective is required")
        if not implementation_plan:
            raise OrchestratorError(
                "implementation_plan is required; Astra must send the complete plan"
            )
        if not verification:
            raise OrchestratorError(
                "verification_instructions is required; Astra must define worker verification"
            )
        commands = _verification_commands(arguments.get("verification_commands"))
        for command in commands:
            self._assert_command_allowed(command)

        worker = self._select_worker(
            arguments.get("worker_id"),
            str(arguments["preferred_source"]).strip()
            if arguments.get("preferred_source")
            else None,
        )
        task_id = str(arguments.get("task_id") or f"task-{uuid.uuid4().hex[:12]}")
        if task_id in self.tasks:
            raise OrchestratorError(
                f"task already exists: {task_id}; use continue_worker_task for feedback"
            )
        task = TaskRequest(
            task_id=task_id,
            title=str(arguments.get("title") or objective[:80]),
            objective=objective,
            implementation_plan=implementation_plan,
            verification_instructions=verification,
            verification_commands=commands,
            context_files=_context_files(
                self.config.project_root, arguments.get("context_files")
            ),
            constraints=[
                str(item)
                for item in arguments.get("constraints", [])
                if str(item).strip()
            ],
            conversation_id=str(arguments.get("conversation_id") or "").strip() or None,
        )
        record = TaskRecord(
            request=task,
            worker_id=worker.worker_id,
            status="running",
            conversation_id=task.conversation_id,
            stage="implementing",
        )
        self.tasks[task_id] = record
        self._append_communication(
            record,
            "astra_to_worker",
            "initial_assignment",
            (
                f"Objective: {objective}\n"
                f"Implementation plan:\n{implementation_plan}\n"
                f"Verification instructions:\n{verification}"
            ),
        )
        self._append_event(record, "codex", "assignment", objective, plan_step="initial")
        self._persist_record(record)

        result = provider_for(worker).execute(worker, task)
        self._store_worker_result(record, result)
        self._drain_pending_messages(record)
        return {
            "task_id": task_id,
            "worker_id": worker.worker_id,
            "source": worker.source,
            "status": record.status,
            "result": result.to_dict(),
            "communication_rounds": len(record.communication),
            "next_action": (
                "Review the worker report. Use continue_worker_task for corrections, "
                "apply_worker_patch for an acceptable diff, run_worker_verification, "
                "then approve_task and finalize_project."
            ),
        }

    def delegate_workspace_operation(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Broker bounded workspace work through Antigravity with captured evidence."""

        operation = str(arguments.get("operation", "")).strip().lower()
        instructions = str(arguments.get("instructions", "")).strip()
        if operation not in {"inspect", "command", "edit"}:
            raise OrchestratorError("operation must be inspect, command, or edit")
        if not instructions:
            raise OrchestratorError("instructions are required")
        if operation in {"command", "edit"} and not bool(arguments.get("confirm", False)):
            raise OrchestratorError(
                "confirm must be true before delegating command execution or direct edits"
            )
        raw_targets = arguments.get("targets", [])
        if not isinstance(raw_targets, list):
            raise OrchestratorError("targets must be an array")
        targets: list[str] = []
        for value in raw_targets:
            target = str(value).strip()
            if not target:
                continue
            _safe_relative_path(self.config.project_root, target)
            targets.append(PurePosixPath(target).as_posix())
        if operation == "edit" and not targets:
            raise OrchestratorError("edit operations require at least one allowed target")

        context: dict[str, str] = {}
        execution_run: VerificationRun | None = None
        if operation in {"inspect", "edit"}:
            if not targets:
                raise OrchestratorError(
                    f"{operation} operations require at least one target"
                )
            context = _context_files(
                self.config.project_root, [{"path": target} for target in targets]
            )
        if operation == "command":
            command_value = arguments.get("command_argv")
            command = _verification_commands([command_value])[0]
            self._assert_command_allowed(command)
            timeout_seconds = int(arguments.get("timeout_seconds", 180))
            if timeout_seconds < 1 or timeout_seconds > 900:
                raise OrchestratorError("timeout_seconds must be between 1 and 900")
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.config.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_seconds,
                    check=False,
                )
                execution_run = VerificationRun(
                    argv=command,
                    exit_code=completed.returncode,
                    stdout=_trim_output(completed.stdout),
                    stderr=_trim_output(completed.stderr),
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            except subprocess.TimeoutExpired as error:
                execution_run = VerificationRun(
                    argv=command,
                    exit_code=124,
                    stdout=_trim_output(error.stdout if isinstance(error.stdout, bytes) else b""),
                    stderr=_trim_output(error.stderr if isinstance(error.stderr, bytes) else b""),
                    duration_ms=int((time.monotonic() - started) * 1000),
                    timed_out=True,
                )
            except OSError as error:
                execution_run = VerificationRun(
                    argv=command,
                    exit_code=127,
                    stdout="",
                    stderr=str(error),
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            context["__delegated_command_evidence__.txt"] = "\n".join([
                f"argv: {execution_run.argv!r}",
                f"exit_code: {execution_run.exit_code}",
                f"timed_out: {execution_run.timed_out}",
                f"stdout:\n{execution_run.stdout or '(empty)'}",
                f"stderr:\n{execution_run.stderr or '(empty)'}",
            ])

        requested_worker = str(arguments.get("worker_id") or "auto")
        worker = self._select_worker(
            requested_worker,
            None if requested_worker != "auto" else "antigravity",
        )
        if worker.source != "antigravity":
            raise OrchestratorError(
                "workspace operations require an Antigravity command worker"
            )
        task_id = str(arguments.get("task_id") or f"operation-{uuid.uuid4().hex[:12]}")
        if task_id in self.tasks:
            raise OrchestratorError(f"task already exists: {task_id}")
        task = TaskRequest(
            task_id=task_id,
            title=f"Delegated {operation}: {instructions[:60]}",
            objective=instructions,
            implementation_plan=(
                "Analyze the bounded evidence supplied by the MCP and return a concise "
                "factual answer. For edit operations, author a unified diff limited to "
                "the allowed targets; the MCP will review and apply it."
            ),
            verification_instructions=(
                "Report exact commands used, relevant output or findings, exit status, "
                "and any uncertainty. Never claim evidence you did not observe."
            ),
            verification_commands=(
                [execution_run.argv]
                if execution_run
                else [["python3", "-c", "print('delegation-recorded')"]]
            ),
            context_files=context,
            constraints=[
                "Stay inside the configured project root.",
                "Operate only on the allowed targets when targets are supplied.",
                "Return compact evidence rather than raw verbose output.",
            ],
            conversation_id=str(arguments.get("conversation_id") or "").strip() or None,
            operation_kind=operation,
            worker_tool_access=False,
            operation_targets=targets,
        )
        record = TaskRecord(
            request=task,
            worker_id=worker.worker_id,
            status="running",
            conversation_id=task.conversation_id,
            stage="implementing",
            tracked_paths=targets,
        )
        self.tasks[task_id] = record
        self._append_communication(
            record, "astra_to_worker", f"workspace_{operation}", instructions
        )
        self._append_event(
            record, "codex", f"workspace_{operation}", instructions,
            plan_step="delegated workspace operation",
        )
        self._persist_record(record)
        result = provider_for(worker).execute(worker, task)
        self._store_worker_result(record, result)
        if execution_run:
            record.verification_runs = [execution_run]
            record.verification_passed = execution_run.passed
        return {
            "task_id": task_id,
            "worker_id": worker.worker_id,
            "source": worker.source,
            "operation": operation,
            "status": record.status,
            "conversation_id": record.conversation_id,
            "result": result.to_dict(),
            "execution": execution_run.to_dict() if execution_run else None,
            "next_action": (
                "Codex should inspect critical evidence directly when the operation affects "
                "security, correctness, or final approval."
            ),
        }

    def continue_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Send a correction or follow-up from Astra to the same worker."""

        task_id = str(arguments.get("task_id", "")).strip()
        feedback = str(arguments.get("feedback", "")).strip()
        if not task_id:
            raise OrchestratorError("task_id is required")
        if not feedback:
            raise OrchestratorError("feedback is required")

        record = self._record_for_task(task_id)
        if record.finalized:
            raise OrchestratorError("task is already finalized")
        if record.paused:
            raise OrchestratorError("task is paused; resume it before continuing")
        if record.cancelled:
            raise OrchestratorError("task is cancelled")
        worker = self.config.worker(record.worker_id)
        verification = str(
            arguments.get("verification_instructions")
            or record.request.verification_instructions
        ).strip()
        commands = (
            _verification_commands(arguments["verification_commands"])
            if "verification_commands" in arguments
            else record.request.verification_commands
        )
        for command in commands:
            self._assert_command_allowed(command)
        context = (
            _context_files(self.config.project_root, arguments["context_files"])
            if "context_files" in arguments
            else _refresh_context_files(
                self.config.project_root, record.request.context_files
            )
        )
        constraints = (
            [
                str(item)
                for item in arguments.get("constraints", [])
                if str(item).strip()
            ]
            if "constraints" in arguments
            else record.request.constraints
        )
        task = TaskRequest(
            task_id=task_id,
            title=record.request.title,
            objective=record.request.objective,
            implementation_plan=record.request.implementation_plan,
            verification_instructions=verification,
            verification_commands=commands,
            context_files=context,
            constraints=constraints,
            follow_up_instructions=feedback,
            conversation_id=record.conversation_id,
            operation_kind=record.request.operation_kind,
            worker_tool_access=record.request.worker_tool_access,
            operation_targets=record.request.operation_targets,
        )
        record.request = task
        record.status = "running"
        record.astra_approved = False
        self._append_communication(record, "astra_to_worker", "correction", feedback)
        self._append_event(record, "codex", "steering", feedback)
        self._persist_record(record)

        result = provider_for(worker).execute(worker, task)
        self._store_worker_result(record, result)
        self._drain_pending_messages(record)
        return {
            "task_id": task_id,
            "worker_id": worker.worker_id,
            "status": record.status,
            "result": result.to_dict(),
            "communication_rounds": len(record.communication),
            "next_action": (
                "Review this worker response. Repeat continue_worker_task if "
                "needed, otherwise apply the patch and run verification."
            ),
        }

    def run_verification(
        self, task_id: str, confirm: bool, timeout_seconds: int
    ) -> dict[str, Any]:
        """Execute Astra's commands and ask the worker to interpret the evidence."""

        if not confirm:
            raise OrchestratorError(
                "confirm must be true before executing verification commands"
            )
        if timeout_seconds < 1 or timeout_seconds > 900:
            raise OrchestratorError("timeout_seconds must be between 1 and 900")
        record = self._record_for_task(task_id)
        if record.finalized:
            raise OrchestratorError("task is already finalized")
        if record.paused:
            raise OrchestratorError("task is paused; resume it before verification")
        if record.cancelled:
            raise OrchestratorError("task is cancelled")
        if not record.result:
            raise OrchestratorError("worker has not returned an implementation")
        if not record.patch_applied:
            raise OrchestratorError(
                "apply the current worker patch before running verification"
            )

        runs: list[VerificationRun] = []
        for argv in record.request.verification_commands:
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    argv,
                    cwd=self.config.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_seconds,
                    check=False,
                )
                run = VerificationRun(
                    argv=argv,
                    exit_code=completed.returncode,
                    stdout=_trim_output(completed.stdout),
                    stderr=_trim_output(completed.stderr),
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            except subprocess.TimeoutExpired as error:
                stdout = error.stdout if isinstance(error.stdout, bytes) else b""
                stderr = error.stderr if isinstance(error.stderr, bytes) else b""
                run = VerificationRun(
                    argv=argv,
                    exit_code=124,
                    stdout=_trim_output(stdout),
                    stderr=_trim_output(stderr),
                    duration_ms=int((time.monotonic() - started) * 1000),
                    timed_out=True,
                )
            except OSError as error:
                run = VerificationRun(
                    argv=argv,
                    exit_code=127,
                    stdout="",
                    stderr=str(error),
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            runs.append(run)

        record.verification_runs = runs
        record.stage = "verifying"
        evidence_sections: list[str] = []
        for index, run in enumerate(runs, start=1):
            evidence_sections.append(
                "\n".join(
                    [
                        f"Check {index}: {run.argv!r}",
                        f"Exit code: {run.exit_code}",
                        f"Timed out: {run.timed_out}",
                        f"STDOUT:\n{run.stdout or '(empty)'}",
                        f"STDERR:\n{run.stderr or '(empty)'}",
                    ]
                )
            )
        evidence = "\n\n".join(evidence_sections)
        self._append_communication(
            record,
            "system_to_worker",
            "verification_evidence",
            "\n".join(
                f"{run.argv!r}: exit={run.exit_code}, timed_out={run.timed_out}"
                for run in runs
            ),
        )

        worker = self.config.worker(record.worker_id)
        follow_up = TaskRequest(
            task_id=task_id,
            title=record.request.title,
            objective=record.request.objective,
            implementation_plan=record.request.implementation_plan,
            verification_instructions=record.request.verification_instructions,
            verification_commands=record.request.verification_commands,
            context_files=_refresh_context_files(
                self.config.project_root, record.request.context_files
            ),
            constraints=record.request.constraints,
            follow_up_instructions=(
                "The MCP executed Astra's verification commands in the project "
                "workspace. Analyze the evidence below. If any check failed because "
                "of the implementation, return a corrective patch. If all checks "
                "passed, return status completed with a precise verification report.\n\n"
                + evidence
            ),
            conversation_id=record.conversation_id,
            operation_kind=record.request.operation_kind,
            worker_tool_access=record.request.worker_tool_access,
            operation_targets=record.request.operation_targets,
        )
        record.request = follow_up
        result = provider_for(worker).execute(worker, follow_up)
        self._store_worker_result(
            record, result, invalidate_verification=False
        )
        commands_passed = all(run.passed for run in runs)
        record.verification_passed = (
            commands_passed
            and result.status == "completed"
            and not result.blockers
            and not result.patch.strip()
        )
        record.verification_workspace_hash = (
            _workspace_fingerprint(self.config.project_root, record.tracked_paths)
            if record.verification_passed else None
        )
        if record.verification_passed:
            record.stage = "verified"
        if result.patch.strip():
            record.patch_applied = False
        record.updated_at = utc_now()
        self._persist_record(record)
        return {
            "task_id": task_id,
            "worker_id": worker.worker_id,
            "commands_passed": commands_passed,
            "verification_passed": record.verification_passed,
            "verification_runs": [run.to_dict() for run in runs],
            "worker_result": result.to_dict(),
            "next_action": (
                "Approve the task if verification_passed is true. Otherwise review "
                "the worker correction, apply its patch, and run verification again."
            ),
        }

    def approve_task(self, task_id: str, approval_note: str) -> dict[str, Any]:
        """Record Astra's review decision before final submission."""

        record = self._record_for_task(task_id)
        if record.finalized:
            raise OrchestratorError("task is already finalized")
        if record.paused or record.cancelled:
            raise OrchestratorError("task is paused or cancelled")
        if record.has_blocking_question:
            raise OrchestratorError("answer the worker's blocking question first")
        if not approval_note.strip():
            raise OrchestratorError("approval_note is required")
        if not record.result:
            raise OrchestratorError("worker has not returned a result")
        if record.result.status != "completed":
            raise OrchestratorError(
                f"worker result is {record.result.status}; send correction instructions first"
            )
        if record.result.blockers:
            raise OrchestratorError(
                "worker reported blockers; resolve them before approval"
            )
        if not record.result.verification_report.strip():
            raise OrchestratorError(
                "worker must return a verification_report before approval"
            )
        if not record.patch_applied:
            raise OrchestratorError(
                "review and apply the worker patch before approval"
            )
        if not record.verification_passed:
            raise OrchestratorError(
                "Astra-specified verification commands must pass and be reviewed "
                "by the worker before approval"
            )
        current_fingerprint = _workspace_fingerprint(
            self.config.project_root, record.tracked_paths
        )
        if not record.verification_workspace_hash or (
            current_fingerprint != record.verification_workspace_hash
        ):
            record.verification_passed = False
            record.stage = "verification_stale"
            self._persist_record(record)
            raise OrchestratorError(
                "workspace changed after verification; run verification again"
            )
        unresolved_reviews = [
            review_id for review_id in record.reviewer_task_ids
            if record.review_decisions.get(review_id) != "resolved"
        ]
        if unresolved_reviews:
            raise OrchestratorError(
                "independent reviews require Codex resolution: "
                + ", ".join(unresolved_reviews)
            )

        record.astra_approved = True
        record.stage = "approved"
        record.updated_at = utc_now()
        self._append_communication(
            record, "astra_to_worker", "approval", approval_note.strip()
        )
        self._persist_record(record)
        return {
            "task_id": task_id,
            "astra_approved": True,
            "communication_rounds": len(record.communication),
            "next_action": "Call finalize_project to submit the final version.",
        }

    def finalize_project(self, task_id: str) -> dict[str, Any]:
        """Close the task only after worker communication and Astra approval."""

        record = self._record_for_task(task_id)
        if record.finalized:
            return {
                "task_id": task_id,
                "finalized": True,
                "message": "final version was already finalized",
            }
        if record.paused or record.cancelled:
            raise OrchestratorError("task is paused or cancelled")
        if record.has_blocking_question:
            raise OrchestratorError("answer the worker's blocking question first")
        if not record.result:
            raise OrchestratorError("no worker communication has occurred")
        if len(record.communication) < 3:
            raise OrchestratorError(
                "finalization requires assignment, worker report, and Astra review communication"
            )
        if not record.patch_applied:
            raise OrchestratorError("the worker patch has not been applied")
        if not record.astra_approved:
            raise OrchestratorError(
                "Astra must approve the verified worker result before finalization"
            )
        if record.result.status != "completed" or record.result.blockers:
            raise OrchestratorError("worker result is not in a final acceptable state")
        if not record.verification_passed:
            raise OrchestratorError("verification evidence has not passed")

        record.finalized = True
        record.status = "finalized"
        record.stage = "finalized"
        record.updated_at = utc_now()
        self._append_communication(
            record,
            "system",
            "finalized",
            "Final version is approved after the Astra-worker communication loop.",
        )
        self._persist_record(record)
        return {
            "task_id": task_id,
            "finalized": True,
            "status": record.status,
            "communication_rounds": len(record.communication),
            "message": "final version is ready for submission",
        }

    def task_status(
        self,
        task_id: str,
        *,
        include_patch: bool = False,
        include_raw: bool = False,
        include_context: bool = False,
        include_communication: bool = False,
        include_verification_output: bool = False,
        include_events: bool = False,
    ) -> dict[str, Any]:
        result = self._record_for_task(task_id).to_dict(
            include_patch=include_patch,
            include_raw=include_raw,
            include_context=include_context,
            include_communication=include_communication,
            include_verification_output=include_verification_output,
            include_events=include_events,
        )
        related_jobs = [job for job in self.jobs.values() if job.get("task_id") == task_id]
        result["jobs"] = related_jobs
        result["active_jobs"] = [
            job["job_id"] for job in related_jobs
            if job.get("status") in {"queued", "running"}
        ]
        return result

    def apply_worker_patch(
        self, task_id: str, confirm: bool
    ) -> dict[str, Any]:
        if not confirm:
            raise OrchestratorError(
                "confirm must be true after reviewing the worker patch"
            )
        if task_id not in self.tasks:
            raise OrchestratorError(f"unknown task: {task_id}")
        record = self.tasks[task_id]
        if record.paused:
            raise OrchestratorError("task is paused; resume it before applying a patch")
        if record.cancelled:
            raise OrchestratorError("task is cancelled")
        if not record.result:
            raise OrchestratorError("task has no worker result")
        patch = record.result.patch
        if not patch.strip():
            raise OrchestratorError("worker result contains no patch")
        if len(patch.encode("utf-8")) > _MAX_PATCH_BYTES:
            raise OrchestratorError("worker patch is larger than 10 MiB")
        paths = _patch_paths(patch)
        if not paths:
            raise OrchestratorError("worker result is not a unified diff")

        preflight = _preflight_patch(self.config.project_root, patch, paths)
        if not all(check["passed"] for check in preflight):
            return {
                "applied": False,
                "task_id": task_id,
                "paths": sorted(paths),
                "preflight": preflight,
                "error": "worker patch failed isolated preflight validation",
            }

        check = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", "-"],
            cwd=self.config.project_root,
            input=patch.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if check.returncode != 0:
            return {
                "applied": False,
                "task_id": task_id,
                "paths": sorted(paths),
                "error": check.stderr.decode("utf-8", errors="replace").strip(),
            }

        applied = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=self.config.project_root,
            input=patch.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        if applied.returncode != 0:
            return {
                "applied": False,
                "task_id": task_id,
                "paths": sorted(paths),
                "error": applied.stderr.decode("utf-8", errors="replace").strip(),
            }
        record.patch_applied = True
        record.verification_runs = []
        record.verification_passed = False
        record.verification_workspace_hash = None
        record.stage = "implementation_changed"
        record.astra_approved = False
        record.updated_at = utc_now()
        self._persist_record(record)
        return {
            "applied": True,
            "task_id": task_id,
            "paths": sorted(paths),
            "preflight": preflight,
            "message": "worker patch applied to the Codex project workspace",
        }

    def get_task_events(
        self, task_id: str, after_sequence: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        record = self._record_for_task(task_id)
        if after_sequence < 0 or not 1 <= limit <= 100:
            raise OrchestratorError("invalid event cursor or limit")
        remaining = [event for event in record.events if event.sequence > after_sequence]
        events = remaining[:limit]
        return {
            "task_id": task_id,
            "events": [event.to_dict() for event in events],
            "next_cursor": events[-1].sequence if events else after_sequence,
            "has_more": len(remaining) > len(events),
        }

    def _submit_job(self, job_id: str, task_id: str, operation) -> dict[str, Any]:
        with self._lock:
            if job_id in self.jobs:
                raise OrchestratorError(f"job already exists: {job_id}")
            self.jobs[job_id] = {
                "job_id": job_id,
                "task_id": task_id,
                "status": "queued",
                "error": None,
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            self._persist_job(self.jobs[job_id])

        def run() -> None:
            with self._lock:
                self.jobs[job_id]["status"] = "running"
                self.jobs[job_id]["updated_at"] = utc_now()
                self._persist_job(self.jobs[job_id])
            try:
                operation()
            except Exception as error:  # background errors must remain observable
                with self._lock:
                    self.jobs[job_id]["status"] = "failed"
                    self.jobs[job_id]["error"] = str(error)
                    self.jobs[job_id]["updated_at"] = utc_now()
                    self._persist_job(self.jobs[job_id])
            else:
                with self._lock:
                    task_record = self.tasks.get(task_id)
                    self.jobs[job_id]["status"] = (
                        "cancelled" if task_record and task_record.cancelled else "completed"
                    )
                    self.jobs[job_id]["updated_at"] = utc_now()
                    self._persist_job(self.jobs[job_id])

        self._executor.submit(run)
        return {
            "job_id": job_id,
            "task_id": task_id,
            "status": "queued",
            "next_action": "Poll get_worker_job until completed, then read task events.",
        }

    def dispatch_async(self, arguments: dict[str, Any]) -> dict[str, Any]:
        task_id = str(arguments.get("task_id") or f"task-{uuid.uuid4().hex[:12]}")
        job_id = str(arguments.get("job_id") or f"job-{uuid.uuid4().hex[:12]}")
        queued_arguments = dict(arguments)
        queued_arguments["task_id"] = task_id
        queued_arguments.pop("job_id", None)
        return self._submit_job(job_id, task_id, lambda: self.dispatch(queued_arguments))

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self.jobs:
                raise OrchestratorError(f"unknown job: {job_id}")
            return dict(self.jobs[job_id])

    def send_worker_message_async(
        self,
        task_id: str,
        message: str,
        message_type: str = "steering",
        job_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_job_id = job_id or f"job-{uuid.uuid4().hex[:12]}"
        return self._submit_job(
            resolved_job_id,
            task_id,
            lambda: self.send_worker_message(task_id, message, message_type),
        )

    def answer_question_async(
        self,
        task_id: str,
        question_id: str,
        answer: str,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        self.answer_question(task_id, question_id, answer, continue_worker=False)
        record = self._record_for_task(task_id)
        question = next(
            item for item in record.questions if item.question_id == question_id
        )
        message = f"Answer to {question_id} ({question.question}): {answer.strip()}"
        resolved_job_id = job_id or f"job-{uuid.uuid4().hex[:12]}"
        return self._submit_job(
            resolved_job_id,
            task_id,
            lambda: self.send_worker_message(task_id, message, "answer"),
        )

    def answer_question(
        self,
        task_id: str,
        question_id: str,
        answer: str,
        *,
        continue_worker: bool = True,
    ) -> dict[str, Any]:
        record = self._record_for_task(task_id)
        if not answer.strip():
            raise OrchestratorError("answer is required")
        question = next(
            (item for item in record.questions if item.question_id == question_id), None
        )
        if question is None:
            raise OrchestratorError(f"unknown question: {question_id}")
        if question.status != "pending":
            raise OrchestratorError("question is already resolved")
        question.status = "answered"
        question.answer = answer.strip()
        question.answered_at = utc_now()
        record.has_blocking_question = any(
            item.blocking and item.status == "pending" for item in record.questions
        )
        self._append_event(
            record, "codex", "answer", answer.strip(),
            plan_step=f"answer to {question_id}",
        )
        self._persist_record(record)
        if not continue_worker:
            return {
                "task_id": task_id,
                "question_id": question_id,
                "status": "answered",
                "worker_continued": False,
            }
        return self.send_worker_message(
            task_id,
            f"Answer to {question_id} ({question.question}): {answer.strip()}",
            "answer",
        )

    def dispatch_review(self, arguments: dict[str, Any]) -> dict[str, Any]:
        parent_id = str(arguments.get("task_id", "")).strip()
        instructions = str(arguments.get("review_instructions", "")).strip()
        if not parent_id or not instructions:
            raise OrchestratorError("task_id and review_instructions are required")
        parent = self._record_for_task(parent_id)
        if parent.finalized or parent.cancelled:
            raise OrchestratorError("finalized or cancelled task cannot be reviewed")
        if not parent.result:
            raise OrchestratorError("implementation worker has not returned a result")
        reviewer_id = str(arguments.get("reviewer_worker_id") or "auto")
        if reviewer_id == "auto":
            alternatives = [
                worker for worker in self.config.workers
                if worker.enabled and worker.worker_id != parent.worker_id
            ]
            reviewer_id = alternatives[0].worker_id if alternatives else parent.worker_id
        review_id = str(
            arguments.get("review_task_id")
            or f"review-{parent_id}-{uuid.uuid4().hex[:8]}"
        )
        reserved = bool(arguments.get("_review_reserved", False))
        if not reserved:
            if review_id in parent.reviewer_task_ids:
                raise OrchestratorError(f"review already exists: {review_id}")
            parent.reviewer_task_ids.append(review_id)
            parent.review_decisions[review_id] = "pending"
            self._persist_record(parent)
        evidence = "\n\n".join(
            f"Command: {run.argv!r}\nExit: {run.exit_code}\n"
            f"STDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}"
            for run in parent.verification_runs
        ) or "No MCP command evidence has been captured yet."
        try:
            dispatched = self.dispatch({
                "task_id": review_id,
                "title": f"Independent review of {parent_id}",
                "worker_id": reviewer_id,
                "objective": (
                    "Independently review the implementation against the original requirements. "
                    "Find defects, missing cases, unsafe assumptions, and verification gaps."
                ),
                "implementation_plan": (
                    "Review from a clean conversation. Do not assume the implementation worker "
                    "is correct and do not modify files."
                ),
                "verification_instructions": instructions,
                "verification_commands": parent.request.verification_commands,
                "constraints": [
                    "Remain independent from the implementation conversation.",
                    "Base findings on supplied requirements, patch, and objective evidence.",
                ],
                "context_files": [
                    {"path": "__review_original_requirements__.txt", "content": "\n".join([
                        parent.request.objective,
                        parent.request.implementation_plan,
                        parent.request.verification_instructions,
                    ])},
                    {"path": "__review_patch__.diff", "content": parent.result.patch},
                    {"path": "__review_verification_evidence__.txt", "content": evidence},
                ],
            })
        except Exception:
            parent.review_decisions[review_id] = "failed"
            self._persist_record(parent)
            raise
        review = self.tasks[review_id]
        review.parent_task_id = parent_id
        review.role = "reviewer"
        review.stage = "review_complete"
        parent.review_decisions[review_id] = "pending"
        self._persist_record(review)
        self._persist_record(parent)
        dispatched["parent_task_id"] = parent_id
        dispatched["role"] = "reviewer"
        dispatched["independent_conversation"] = (
            review.conversation_id != parent.conversation_id
        )
        return dispatched

    def resolve_review(
        self, task_id: str, review_task_id: str, resolution_note: str
    ) -> dict[str, Any]:
        parent = self._record_for_task(task_id)
        if review_task_id not in parent.reviewer_task_ids:
            raise OrchestratorError("review is not attached to this task")
        if not resolution_note.strip():
            raise OrchestratorError("resolution_note is required")
        review = self.tasks.get(review_task_id)
        if review is None:
            if parent.review_decisions.get(review_task_id) != "failed":
                raise OrchestratorError("review worker has not created a task")
        elif review.parent_task_id != task_id:
            raise OrchestratorError("review is not attached to this task")
        elif not review.result:
            raise OrchestratorError("review worker has not returned a result")
        parent.review_decisions[review_task_id] = "resolved"
        self._append_event(
            parent, "codex", "review_resolved", resolution_note.strip(),
            plan_step=review_task_id,
        )
        self._persist_record(parent)
        return {
            "task_id": task_id,
            "review_task_id": review_task_id,
            "status": "resolved",
        }

    def dispatch_review_async(self, arguments: dict[str, Any]) -> dict[str, Any]:
        parent_id = str(arguments.get("task_id", "")).strip()
        if not parent_id:
            raise OrchestratorError("task_id is required")
        parent = self._record_for_task(parent_id)
        if parent.finalized or parent.cancelled:
            raise OrchestratorError("finalized or cancelled task cannot be reviewed")
        review_id = str(
            arguments.get("review_task_id")
            or f"review-{arguments.get('task_id', 'task')}-{uuid.uuid4().hex[:8]}"
        )
        job_id = str(arguments.get("job_id") or f"job-{uuid.uuid4().hex[:12]}")
        queued_arguments = dict(arguments)
        queued_arguments["review_task_id"] = review_id
        queued_arguments["_review_reserved"] = True
        queued_arguments.pop("job_id", None)
        queued_arguments.pop("async", None)
        if review_id in parent.reviewer_task_ids:
            raise OrchestratorError(f"review already exists: {review_id}")
        # Reserve the gate before the background thread starts, eliminating the
        # window in which approval could race ahead of review registration.
        parent.reviewer_task_ids.append(review_id)
        parent.review_decisions[review_id] = "pending"
        self._persist_record(parent)
        try:
            return self._submit_job(
                job_id, review_id, lambda: self.dispatch_review(queued_arguments)
            )
        except Exception:
            parent.reviewer_task_ids.remove(review_id)
            parent.review_decisions.pop(review_id, None)
            self._persist_record(parent)
            raise

    def send_worker_message(
        self, task_id: str, message: str, message_type: str = "steering"
    ) -> dict[str, Any]:
        record = self._record_for_task(task_id)
        if not message.strip():
            raise OrchestratorError("message is required")
        if record.finalized or record.paused or record.cancelled:
            raise OrchestratorError("task cannot accept worker messages")
        if record.status == "running":
            message_id = f"message-{uuid.uuid4().hex[:10]}"
            record.pending_messages.append({
                "message_id": message_id,
                "message": message.strip(),
                "message_type": message_type,
            })
            self._append_event(
                record, "codex", "queued_message", message.strip(),
                plan_step="deliver after active worker turn",
            )
            self._persist_record(record)
            return {
                "task_id": task_id,
                "status": "queued",
                "message_id": message_id,
                "pending_messages": len(record.pending_messages),
            }
        self._append_communication(record, "astra_to_worker", message_type, message.strip())
        self._append_event(record, "codex", message_type, message.strip())
        self._persist_record(record)
        worker = self.config.worker(record.worker_id)
        task = TaskRequest(
            task_id=task_id,
            title=record.request.title,
            objective=record.request.objective,
            implementation_plan=record.request.implementation_plan,
            verification_instructions=record.request.verification_instructions,
            verification_commands=record.request.verification_commands,
            context_files=_refresh_context_files(
                self.config.project_root, record.request.context_files
            ),
            constraints=record.request.constraints,
            follow_up_instructions=message.strip(),
            conversation_id=record.conversation_id,
            operation_kind=record.request.operation_kind,
            worker_tool_access=record.request.worker_tool_access,
            operation_targets=record.request.operation_targets,
        )
        record.request = task
        record.status = "running"
        result = provider_for(worker).execute(worker, task)
        self._store_worker_result(record, result)
        self._drain_pending_messages(record)
        return {
            "task_id": task_id,
            "status": record.status,
            "conversation_id": record.conversation_id,
            "result": result.to_dict(),
            "events_count": len(record.events),
        }

    def _drain_pending_messages(self, record: TaskRecord) -> None:
        if not record.pending_messages or record.paused or record.cancelled:
            return
        pending = record.pending_messages.pop(0)
        self.send_worker_message(
            record.request.task_id,
            pending["message"],
            pending["message_type"],
        )

    def pause_task(self, task_id: str, reason: str = "") -> dict[str, Any]:
        record = self._record_for_task(task_id)
        if record.finalized or record.cancelled:
            raise OrchestratorError("task cannot be paused")
        record.paused = True
        self._append_event(record, "codex", "pause", reason or "Task paused")
        self._persist_record(record)
        return {"task_id": task_id, "paused": True}

    def resume_task(self, task_id: str) -> dict[str, Any]:
        record = self._record_for_task(task_id)
        if record.finalized or record.cancelled:
            raise OrchestratorError("task cannot be resumed")
        record.paused = False
        self._append_event(record, "codex", "resume", "Task resumed")
        self._persist_record(record)
        return {"task_id": task_id, "paused": False, "status": record.status}

    def cancel_task(self, task_id: str, reason: str = "") -> dict[str, Any]:
        record = self._record_for_task(task_id)
        if record.finalized:
            raise OrchestratorError("finalized task cannot be cancelled")
        record.cancelled = True
        record.status = "cancelled"
        record.stage = "cancelled"
        process_terminated = cancel_worker_process(task_id)
        self._append_event(record, "codex", "cancel", reason or "Task cancelled")
        self._persist_record(record)
        return {
            "task_id": task_id,
            "cancelled": True,
            "status": "cancelled",
            "process_terminated": process_terminated,
        }
