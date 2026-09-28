"""Worker transports for the universal MCP bridge."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .models import TaskRequest, WorkerConfig, WorkerResult


_ACTIVE_PROCESSES: dict[str, subprocess.Popen[bytes]] = {}
_ACTIVE_PROCESSES_LOCK = threading.RLock()


def cancel_worker_process(task_id: str) -> bool:
    with _ACTIVE_PROCESSES_LOCK:
        process = _ACTIVE_PROCESSES.get(task_id)
    if process is None or process.poll() is not None:
        return False
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
    return True


SYSTEM_INSTRUCTIONS = """You are an implementation and verification worker.
The coordinator has already planned the task. Follow the supplied objective,
implementation plan, constraints, and verification instructions exactly.
Do not replace the plan with a new architecture. Work only on the assigned
scope. Return one JSON object with keys: status, summary, patch, tests_run,
verification_report, and blockers. The patch must be a standard unified diff
relative to the supplied project files. If no code change is needed, return
an empty patch. Do not claim that checks passed unless you ran them yourself or
the coordinator supplied captured command evidence. When evidence is supplied,
interpret its exit codes and output rather than inventing results."""


class ProviderError(RuntimeError):
    """An expected worker transport failure."""


@dataclass(frozen=True)
class HealthResult:
    worker_id: str
    healthy: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "healthy": self.healthy,
            "detail": self.detail,
        }


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks)
    return str(value)


def _parse_worker_json(output: str) -> dict[str, Any] | None:
    candidates = [output.strip()]
    if "```" in output:
        for block in output.split("```")[1::2]:
            cleaned = block.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].lstrip()
            candidates.append(cleaned)

    start = output.find("{")
    end = output.rfind("}")
    if start >= 0 and end > start:
        candidates.append(output[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_worker_output(
    task_id: str, worker_id: str, output: str
) -> WorkerResult:
    """Normalize strict JSON output while remaining safe with plain text."""

    parsed = _parse_worker_json(output)
    if parsed is None:
        return WorkerResult(
            task_id=task_id,
            worker_id=worker_id,
            status="needs_review",
            summary="Worker did not return the required JSON result.",
            raw_output=output,
            error="invalid_worker_result",
        )

    tests_run = parsed.get("tests_run", [])
    if isinstance(tests_run, str):
        tests_run = [tests_run]
    if not isinstance(tests_run, list):
        tests_run = []

    blockers = parsed.get("blockers", [])
    if isinstance(blockers, str):
        blockers = [blockers]
    if not isinstance(blockers, list):
        blockers = []

    status = str(parsed.get("status", "needs_review")).strip().lower()
    if status not in {"completed", "failed", "needs_review"}:
        status = "needs_review"

    raw_usage = parsed.get("usage", {})
    usage = {
        str(key): int(value)
        for key, value in raw_usage.items()
        if isinstance(key, str) and isinstance(value, (int, float))
    } if isinstance(raw_usage, dict) else {}
    duration = parsed.get("duration_seconds")
    conversation_id = str(parsed.get("conversation_id") or "").strip() or None
    event_type = str(parsed.get("event_type") or "").strip().lower() or None
    allowed_events = {"checkpoint", "question", "risk", "blocked", "ready_for_review"}
    if event_type not in allowed_events:
        event_type = None
    assumptions = parsed.get("assumptions", [])
    risks = parsed.get("risks", [])
    if isinstance(assumptions, str):
        assumptions = [assumptions]
    if isinstance(risks, str):
        risks = [risks]

    return WorkerResult(
        task_id=task_id,
        worker_id=worker_id,
        status=status,
        summary=str(parsed.get("summary", "")),
        patch=str(parsed.get("patch", parsed.get("diff", "")) or ""),
        tests_run=[str(item) for item in tests_run],
        verification_report=str(parsed.get("verification_report", "")),
        blockers=[str(item) for item in blockers],
        raw_output=output,
        error=str(parsed["error"]) if parsed.get("error") else None,
        usage=usage,
        duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
        conversation_id=conversation_id,
        event_type=event_type,
        plan_step=str(parsed.get("plan_step") or ""),
        next_step=str(parsed.get("next_step") or ""),
        question=str(parsed.get("question") or ""),
        blocking=bool(parsed.get("blocking", False)),
        assumptions=[str(item) for item in assumptions] if isinstance(assumptions, list) else [],
        risks=[str(item) for item in risks] if isinstance(risks, list) else [],
    )


def _task_prompt(task: TaskRequest) -> str:
    context = "\n\n".join(
        f"===== FILE: {path} =====\n{content}"
        for path, content in task.context_files.items()
    )
    constraints = "\n".join(f"- {item}" for item in task.constraints) or "- None"
    verification_commands = "\n".join(
        f"- {json.dumps(command)}" for command in task.verification_commands
    ) or "- None supplied"
    return f"""TASK PACKAGE
Task ID: {task.task_id}
Title: {task.title}

OBJECTIVE
{task.objective}

IMPLEMENTATION PLAN
{task.implementation_plan}

VERIFICATION INSTRUCTIONS
{task.verification_instructions}

ASTRA-SPECIFIED VERIFICATION COMMANDS
{verification_commands}

CONSTRAINTS
{constraints}

FOLLOW-UP COMMUNICATION
{task.follow_up_instructions or "(Initial worker assignment; no earlier feedback.)"}

WORKSPACE OPERATION
Kind: {task.operation_kind}
Worker tool access: {task.worker_tool_access}
Allowed targets: {json.dumps(task.operation_targets)}

PROJECT CONTEXT
{context or "(No file context was supplied. Report a blocker if files are required.)"}

RESPONSE CONTRACT
Return only one JSON object. Never include raw chain-of-thought. Use concise
checkpoint fields so the coordinator can supervise the work:
{{
  "status": "completed|failed|needs_review",
  "summary": "what you changed",
  "patch": "unified diff, or empty string",
  "tests_run": ["commands or checks actually run"],
  "verification_report": "results against the coordinator instructions",
  "blockers": ["anything preventing completion"],
  "event_type": "checkpoint|question|risk|blocked|ready_for_review",
  "plan_step": "current plan step",
  "next_step": "next planned step",
  "question": "question for Codex, or empty string",
  "blocking": false,
  "assumptions": [],
  "risks": []
}}"""


class OpenAICompatibleWorker:
    """Call an OpenAI-compatible chat-completions endpoint."""

    def _base_v1_url(self, config: WorkerConfig) -> str:
        assert config.base_url is not None
        base = config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base[: -len("/chat/completions")]
        return base if base.endswith("/v1") else f"{base}/v1"

    def _url(self, config: WorkerConfig, suffix: str) -> str:
        base = self._base_v1_url(config)
        if config.base_url and config.base_url.rstrip("/").endswith(
            "/chat/completions"
        ):
            if suffix == "chat/completions":
                return config.base_url.rstrip("/")
        return f"{base}/{suffix.lstrip('/')}"

    def _headers(self, config: WorkerConfig) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                raise ProviderError(
                    f"environment variable {config.api_key_env!r} is not set"
                )
            if config.api_key_header.lower() == "authorization":
                headers[config.api_key_header] = f"Bearer {api_key}"
            else:
                headers[config.api_key_header] = api_key
        headers.update(config.headers)
        return headers

    def _request(
        self,
        config: WorkerConfig,
        method: str,
        url: str,
        body: bytes | None = None,
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(
            url=url,
            data=body,
            headers=self._headers(config),
            method=method,
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(
                    request, timeout=config.timeout_seconds
                ) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as error:
                if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                    raise ProviderError(
                        f"worker returned HTTP {error.code} for {config.worker_id}"
                    ) from error
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == 2:
                    detail = getattr(error, "reason", str(error))
                    raise ProviderError(
                        f"could not reach worker {config.worker_id}: {detail}"
                    ) from error
            time.sleep(0.25 * (2**attempt))
        raise ProviderError(f"could not reach worker {config.worker_id}")

    def health(self, config: WorkerConfig) -> HealthResult:
        try:
            status, body = self._request(config, "GET", self._url(config, "models"))
            if status < 200 or status >= 300:
                return HealthResult(config.worker_id, False, f"HTTP {status}")
            json.loads(body.decode("utf-8"))
            return HealthResult(config.worker_id, True, "models endpoint responded")
        except (ProviderError, UnicodeDecodeError, json.JSONDecodeError) as error:
            return HealthResult(config.worker_id, False, str(error))

    def execute(self, config: WorkerConfig, task: TaskRequest) -> WorkerResult:
        payload = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": _task_prompt(task)},
            ],
            "stream": False,
        }
        try:
            _, body = self._request(
                config,
                "POST",
                self._url(config, "chat/completions"),
                json.dumps(payload).encode("utf-8"),
            )
            response = json.loads(body.decode("utf-8"))
            choices = response.get("choices", [])
            if not choices:
                raise ProviderError("worker response did not contain choices")
            message = choices[0].get("message", {})
            output = _text_content(message.get("content", ""))
            result = normalize_worker_output(task.task_id, config.worker_id, output)
            raw_usage = response.get("usage", {})
            if isinstance(raw_usage, dict):
                result.usage = {
                    str(key): int(value)
                    for key, value in raw_usage.items()
                    if isinstance(key, str) and isinstance(value, (int, float))
                }
            return result
        except (ProviderError, UnicodeDecodeError, json.JSONDecodeError) as error:
            return WorkerResult(
                task_id=task.task_id,
                worker_id=config.worker_id,
                status="failed",
                error=str(error),
            )


class CommandWorker:
    """Run a local Antigravity adapter that accepts a JSON task on stdin."""

    def health(self, config: WorkerConfig) -> HealthResult:
        if not config.command:
            return HealthResult(config.worker_id, False, "command is not configured")
        command = shlex.split(config.command)
        if not command:
            return HealthResult(config.worker_id, False, "command is empty")
        executable = command[0]
        if "/" in executable:
            available = os.path.isfile(executable) and os.access(executable, os.X_OK)
        else:
            available = shutil.which(executable) is not None
        return HealthResult(
            config.worker_id,
            available,
            "command executable is available"
            if available
            else f"command executable was not found: {executable}",
        )

    def execute(self, config: WorkerConfig, task: TaskRequest) -> WorkerResult:
        if not config.command:
            return WorkerResult(
                task_id=task.task_id,
                worker_id=config.worker_id,
                status="failed",
                error="command is not configured",
            )

        command = shlex.split(config.command)
        if not command:
            return WorkerResult(
                task_id=task.task_id,
                worker_id=config.worker_id,
                status="failed",
                error="command is empty",
            )

        payload = {
            "system_instructions": SYSTEM_INSTRUCTIONS,
            "task": task.to_dict(include_context=True),
            "prompt": _task_prompt(task),
            "conversation_id": task.conversation_id,
            "project_root": os.environ.get("UNIVERSAL_MCP_PROJECT_ROOT", ""),
        }
        environment = os.environ.copy()
        environment["UNIVERSAL_MCP_WORKER_ID"] = config.worker_id
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            with _ACTIVE_PROCESSES_LOCK:
                _ACTIVE_PROCESSES[task.task_id] = process
            stdout_bytes, stderr_bytes = process.communicate(
                input=(json.dumps(payload) + "\n").encode("utf-8"),
                timeout=config.timeout_seconds,
            )
            completed_returncode = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return WorkerResult(
                task_id=task.task_id,
                worker_id=config.worker_id,
                status="failed",
                error=f"worker timed out after {config.timeout_seconds} seconds",
            )
        except OSError as error:
            return WorkerResult(
                task_id=task.task_id,
                worker_id=config.worker_id,
                status="failed",
                error=f"could not start worker command: {error}",
            )
        finally:
            with _ACTIVE_PROCESSES_LOCK:
                _ACTIVE_PROCESSES.pop(task.task_id, None)

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        if completed_returncode != 0:
            return WorkerResult(
                task_id=task.task_id,
                worker_id=config.worker_id,
                status="failed",
                raw_output=stdout,
                error=f"worker command exited with code {completed_returncode}",
            )
        result = normalize_worker_output(task.task_id, config.worker_id, stdout)
        if not result.conversation_id:
            result.conversation_id = task.conversation_id
        return result


def provider_for(config: WorkerConfig) -> OpenAICompatibleWorker | CommandWorker:
    if config.transport == "openai_compatible":
        return OpenAICompatibleWorker()
    if config.transport == "command":
        return CommandWorker()
    raise ProviderError(f"unsupported transport: {config.transport}")
