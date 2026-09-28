import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from universal_mcp.config import AppConfig
from universal_mcp.models import TaskRecord, TaskRequest, WorkerConfig, WorkerResult
from universal_mcp.orchestrator import (
    Orchestrator,
    OrchestratorError,
    _patch_paths,
    _refresh_context_files,
    _workspace_fingerprint,
)


class OrchestratorTests(unittest.TestCase):
    def test_refresh_preserves_virtual_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            refreshed = _refresh_context_files(
                Path(directory), {"__evidence__.txt": "captured output"}
            )
            self.assertEqual(refreshed["__evidence__.txt"], "captured output")

    def test_command_allowlist_blocks_unapproved_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                project_root=Path(directory),
                workers=(WorkerConfig(
                    worker_id="worker", source="antigravity", model="test",
                    transport="command", command="python3",
                ),),
                allowed_command_prefixes=(("python3",),),
            )
            orchestrator = Orchestrator(config)
            with self.assertRaisesRegex(OrchestratorError, "not allowed"):
                orchestrator.delegate_workspace_operation({
                    "operation": "command",
                    "instructions": "show files",
                    "command_argv": ["ls"],
                    "confirm": True,
                })

    def test_sqlite_state_survives_orchestrator_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                project_root=root,
                workers=(WorkerConfig(
                    worker_id="worker", source="test", model="test",
                    transport="command", command="unused",
                ),),
                storage_path=root / "state.sqlite3",
            )

            class FakeProvider:
                def execute(self, worker, task):
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        status="needs_review",
                        summary="waiting for durable answer",
                        question="Persist this question?",
                        blocking=True,
                        conversation_id="durable-conversation",
                    )

            with patch("universal_mcp.orchestrator.provider_for", return_value=FakeProvider()):
                first = Orchestrator(config)
                task_id = first.dispatch({
                    "task_id": "durable-task",
                    "objective": "persist state",
                    "implementation_plan": "ask a question",
                    "verification_instructions": "retain all state",
                    "verification_commands": [["python3", "-c", "print('ok')"]],
                })["task_id"]

            restored = Orchestrator(config)
            record = restored.tasks[task_id]
            self.assertEqual(record.conversation_id, "durable-conversation")
            self.assertTrue(record.has_blocking_question)
            self.assertEqual(record.questions[0].question, "Persist this question?")
            self.assertEqual(len(record.events), 2)

    def test_delegated_workspace_operation_defaults_to_antigravity_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("hello", encoding="utf-8")
            config = AppConfig(
                project_root=root,
                workers=(
                    WorkerConfig(
                        worker_id="other", source="kaggle", model="qwen",
                        transport="command", command="unused",
                    ),
                    WorkerConfig(
                        worker_id="antigravity", source="antigravity", model="gemini",
                        transport="command", command="unused",
                    ),
                ),
            )
            orchestrator = Orchestrator(config)
            captured = {}

            class FakeProvider:
                def execute(self, worker, task):
                    captured["worker"] = worker.worker_id
                    captured["task"] = task
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        status="completed",
                        summary="inspected notes",
                        verification_report="notes.txt contained hello",
                        conversation_id="workspace-conversation",
                    )

            with patch("universal_mcp.orchestrator.provider_for", return_value=FakeProvider()):
                result = orchestrator.delegate_workspace_operation({
                    "operation": "inspect",
                    "instructions": "Read notes.txt and summarize it.",
                    "targets": ["notes.txt"],
                })

            self.assertEqual(result["worker_id"], "antigravity")
            self.assertFalse(captured["task"].worker_tool_access)
            self.assertEqual(captured["task"].operation_kind, "inspect")
            self.assertEqual(captured["task"].operation_targets, ["notes.txt"])
            self.assertEqual(captured["task"].context_files["notes.txt"], "hello")
            with self.assertRaises(OrchestratorError):
                orchestrator.delegate_workspace_operation({
                    "operation": "command",
                    "instructions": "Run a command.",
                    "command_argv": ["python3", "-c", "print('ok')"],
                })
            with self.assertRaises(OrchestratorError):
                orchestrator.delegate_workspace_operation({
                    "operation": "edit",
                    "instructions": "Edit outside the project.",
                    "targets": ["../outside.txt"],
                    "confirm": True,
                })

    def test_patch_paths_are_extracted(self):
        patch = """diff --git a/src/example.py b/src/example.py
--- a/src/example.py
+++ b/src/example.py
@@ -1 +1 @@
-old
+new
"""
        self.assertEqual(_patch_paths(patch), {"src/example.py"})

    def test_python_syntax_preflight_blocks_workspace_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = AppConfig(
                project_root=root,
                workers=(WorkerConfig(
                    worker_id="worker", source="test", model="test",
                    transport="command", command="unused",
                ),),
            )
            orchestrator = Orchestrator(config)
            bad_patch = """diff --git a/bad.py b/bad.py
new file mode 100644
--- /dev/null
+++ b/bad.py
@@ -0,0 +1,2 @@
+if True:
+
"""

            class FakeProvider:
                def execute(self, worker, task):
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        status="completed",
                        summary="created file",
                        patch=bad_patch,
                    )

            with patch("universal_mcp.orchestrator.provider_for", return_value=FakeProvider()):
                task_id = orchestrator.dispatch({
                    "objective": "create bad file",
                    "implementation_plan": "return patch",
                    "verification_instructions": "compile it",
                    "verification_commands": [["python3", "-m", "py_compile", "bad.py"]],
                })["task_id"]
            result = orchestrator.apply_worker_patch(task_id, True)
            self.assertFalse(result["applied"])
            self.assertIn("preflight", result)
            self.assertFalse((root / "bad.py").exists())

    def test_unsafe_patch_path_is_rejected(self):
        with self.assertRaises(OrchestratorError):
            _patch_paths(
                """diff --git a/../secrets.txt b/../secrets.txt
--- a/../secrets.txt
+++ b/../secrets.txt
"""
            )

    def test_context_file_is_loaded_for_command_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ok.txt").write_text("hello", encoding="utf-8")
            config = AppConfig(
                project_root=root,
                workers=(
                    WorkerConfig(
                        worker_id="worker",
                        source="test",
                        model="test",
                        transport="command",
                        command=(
                            "python3 -c 'import sys; sys.stdin.read(); "
                            "print(\"{\\\"status\\\":\\\"completed\\\","
                            "\\\"summary\\\":\\\"received\\\"}\")'"
                        ),
                    ),
                ),
            )
            orchestrator = Orchestrator(config)
            result = orchestrator.dispatch(
                {
                    "objective": "inspect",
                    "implementation_plan": "Read the supplied file.",
                    "verification_instructions": "Report whether it was readable.",
                    "verification_commands": [["python3", "-c", "print('ok')"]],
                    "context_files": [{"path": "ok.txt"}],
                }
            )
            self.assertEqual(result["status"], "completed")
            self.assertNotIn("raw_output", result["result"])
            self.assertNotIn("patch", result["result"])

    def test_communication_loop_gates_finalization(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = AppConfig(
            project_root=Path(temporary.name),
            workers=(
                WorkerConfig(
                    worker_id="worker",
                    source="test",
                    model="test",
                    transport="command",
                    command="unused",
                ),
            ),
        )
        orchestrator = Orchestrator(config)

        class FakeProvider:
            def execute(self, worker, task):
                return WorkerResult(
                    task_id=task.task_id,
                    worker_id=worker.worker_id,
                    status="completed",
                    summary="implemented and checked",
                    verification_report="targeted checks passed",
                )

        with patch(
            "universal_mcp.orchestrator.provider_for",
            return_value=FakeProvider(),
        ):
            first = orchestrator.dispatch(
                {
                    "objective": "implement a change",
                    "implementation_plan": "Make the planned change.",
                    "verification_instructions": "Run the targeted checks.",
                    "verification_commands": [
                        ["python3", "-c", "print('verified')"]
                    ],
                }
            )
            task_id = first["task_id"]
            self.assertEqual(len(orchestrator.tasks[task_id].communication), 2)

            with self.assertRaises(OrchestratorError):
                orchestrator.finalize_project(task_id)

            verification = orchestrator.run_verification(task_id, True, 10)
            self.assertTrue(verification["verification_passed"])
            self.assertEqual(len(orchestrator.tasks[task_id].communication), 4)

            orchestrator.approve_task(
                task_id,
                approval_note="Astra reviewed the worker report and accepts it.",
            )
            final = orchestrator.finalize_project(task_id)
            self.assertTrue(final["finalized"])
            self.assertEqual(orchestrator.tasks[task_id].status, "finalized")

    def test_failed_real_command_cannot_be_approved(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                project_root=Path(directory),
                workers=(
                    WorkerConfig(
                        worker_id="worker",
                        source="test",
                        model="test",
                        transport="command",
                        command="unused",
                    ),
                ),
            )
            orchestrator = Orchestrator(config)

            class FakeProvider:
                def execute(self, worker, task):
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        status="completed",
                        summary="worker claims completion",
                        verification_report="worker reviewed the evidence",
                    )

            with patch(
                "universal_mcp.orchestrator.provider_for",
                return_value=FakeProvider(),
            ):
                result = orchestrator.dispatch(
                    {
                        "objective": "verify failure gating",
                        "implementation_plan": "No code change is required.",
                        "verification_instructions": "The command must exit zero.",
                        "verification_commands": [
                            ["python3", "-c", "raise SystemExit(7)"]
                        ],
                    }
                )
                task_id = result["task_id"]
                verification = orchestrator.run_verification(task_id, True, 10)
                self.assertFalse(verification["verification_passed"])
                with self.assertRaises(OrchestratorError):
                    orchestrator.approve_task(task_id, "approve anyway")

    def test_supervision_events_question_answer_and_conversation_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                project_root=Path(directory),
                workers=(WorkerConfig(
                    worker_id="worker", source="test", model="test",
                    transport="command", command="unused",
                ),),
            )
            orchestrator = Orchestrator(config)
            seen_conversations = []

            class FakeProvider:
                def execute(self, worker, task):
                    seen_conversations.append(task.conversation_id)
                    if not task.follow_up_instructions:
                        return WorkerResult(
                            task_id=task.task_id,
                            worker_id=worker.worker_id,
                            status="waiting",
                            summary="Need one decision",
                            conversation_id="conversation-123",
                            event_type="question",
                            question="Should the interface use a dark theme?",
                            blocking=True,
                        )
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        status="completed",
                        summary="Decision received",
                        conversation_id=task.conversation_id,
                        event_type="ready_for_review",
                        verification_report="No code change was required.",
                    )

            with patch("universal_mcp.orchestrator.provider_for", return_value=FakeProvider()):
                dispatched = orchestrator.dispatch({
                    "objective": "make a decision",
                    "implementation_plan": "Ask before choosing the theme.",
                    "verification_instructions": "Confirm the selected theme.",
                    "verification_commands": [["python3", "-c", "print('ok')"]],
                })
                task_id = dispatched["task_id"]
                first_page = orchestrator.get_task_events(task_id, limit=1)
                self.assertTrue(first_page["has_more"])
                second_page = orchestrator.get_task_events(
                    task_id, after_sequence=first_page["next_cursor"]
                )
                self.assertEqual(second_page["events"][0]["type"], "question")
                with self.assertRaises(OrchestratorError):
                    orchestrator.approve_task(task_id, "premature")

                question_id = orchestrator.tasks[task_id].questions[0].question_id
                response = orchestrator.answer_question(
                    task_id, question_id, "Use the dark theme."
                )
                self.assertEqual(response["status"], "completed")
                self.assertEqual(seen_conversations, [None, "conversation-123"])
                self.assertFalse(orchestrator.tasks[task_id].has_blocking_question)

    def test_nonblocking_question_does_not_block_approval_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                project_root=Path(directory),
                workers=(WorkerConfig(
                    worker_id="worker", source="test", model="test",
                    transport="command", command="unused",
                ),),
            )
            orchestrator = Orchestrator(config)

            class FakeProvider:
                def execute(self, worker, task):
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        status="completed",
                        summary="completed with an informational question",
                        verification_report="checked",
                        question="Would you like an optional cleanup later?",
                        blocking=False,
                    )

            with patch("universal_mcp.orchestrator.provider_for", return_value=FakeProvider()):
                task_id = orchestrator.dispatch({
                    "objective": "nonblocking question",
                    "implementation_plan": "Complete without changes.",
                    "verification_instructions": "Run check.",
                    "verification_commands": [["python3", "-c", "print('ok')"]],
                })["task_id"]
                record = orchestrator.tasks[task_id]
                self.assertEqual(len(record.questions), 1)
                self.assertFalse(record.has_blocking_question)
                self.assertFalse(record.events[-1].blocking)

    def test_repeated_pending_question_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                project_root=Path(directory),
                workers=(WorkerConfig(
                    worker_id="worker", source="test", model="test",
                    transport="command", command="unused",
                ),),
            )
            orchestrator = Orchestrator(config)

            class FakeProvider:
                def execute(self, worker, task):
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        status="needs_review",
                        summary="waiting",
                        question="Should the value be red or blue?",
                        blocking=True,
                    )

            with patch("universal_mcp.orchestrator.provider_for", return_value=FakeProvider()):
                task_id = orchestrator.dispatch({
                    "objective": "ask once",
                    "implementation_plan": "ask a question",
                    "verification_instructions": "do not duplicate it",
                    "verification_commands": [["python3", "-c", "print('ok')"]],
                })["task_id"]
                orchestrator.send_worker_message(task_id, "Provide a concise update.")

            self.assertEqual(len(orchestrator.tasks[task_id].questions), 1)
            self.assertTrue(orchestrator.tasks[task_id].has_blocking_question)

    def test_async_dispatch_returns_before_worker_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                project_root=Path(directory),
                workers=(WorkerConfig(
                    worker_id="worker", source="test", model="test",
                    transport="command", command="unused",
                ),),
            )
            orchestrator = Orchestrator(config)
            release = __import__("threading").Event()
            calls = []

            class FakeProvider:
                def execute(self, worker, task):
                    calls.append(task.follow_up_instructions)
                    release.wait(2)
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        status="completed",
                        summary="finished",
                    )

            with patch("universal_mcp.orchestrator.provider_for", return_value=FakeProvider()):
                queued = orchestrator.dispatch_async({
                    "task_id": "async-task",
                    "job_id": "async-job",
                    "objective": "run asynchronously",
                    "implementation_plan": "Wait and finish.",
                    "verification_instructions": "Report completion.",
                    "verification_commands": [["python3", "-c", "print('ok')"]],
                })
                self.assertEqual(queued["status"], "queued")
                self.assertIn(orchestrator.get_job("async-job")["status"], {"queued", "running"})
                for _ in range(100):
                    if "async-task" in orchestrator.tasks:
                        break
                    __import__("time").sleep(0.01)
                queued_message = orchestrator.send_worker_message(
                    "async-task", "Also report the final status.", "steering"
                )
                self.assertEqual(queued_message["status"], "queued")
                release.set()
                for _ in range(100):
                    if orchestrator.get_job("async-job")["status"] == "completed":
                        break
                    __import__("time").sleep(0.01)
                self.assertEqual(orchestrator.get_job("async-job")["status"], "completed")
                self.assertEqual(orchestrator.tasks["async-task"].status, "completed")
                self.assertEqual(calls, ["", "Also report the final status."])

    def test_independent_review_uses_clean_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                project_root=Path(directory),
                workers=(
                    WorkerConfig(
                        worker_id="author", source="antigravity", model="gemini",
                        transport="command", command="unused",
                    ),
                    WorkerConfig(
                        worker_id="reviewer", source="antigravity", model="claude",
                        transport="command", command="unused",
                    ),
                ),
            )
            orchestrator = Orchestrator(config)

            class FakeProvider:
                def execute(self, worker, task):
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        status="completed",
                        summary=f"response from {worker.worker_id}",
                        verification_report="reviewed",
                        conversation_id=f"conversation-{worker.worker_id}",
                    )

            with patch("universal_mcp.orchestrator.provider_for", return_value=FakeProvider()):
                parent_id = orchestrator.dispatch({
                    "task_id": "parent",
                    "worker_id": "author",
                    "objective": "implement",
                    "implementation_plan": "make the change",
                    "verification_instructions": "check behavior",
                    "verification_commands": [["python3", "-c", "print('ok')"]],
                })["task_id"]
                review = orchestrator.dispatch_review({
                    "task_id": parent_id,
                    "reviewer_worker_id": "reviewer",
                    "review_instructions": "Find mistakes independently.",
                })

            review_record = orchestrator.tasks[review["task_id"]]
            self.assertEqual(review_record.parent_task_id, parent_id)
            self.assertEqual(review_record.role, "reviewer")
            self.assertTrue(review["independent_conversation"])
            self.assertEqual(review_record.request.conversation_id, None)
            self.assertEqual(
                orchestrator.tasks[parent_id].review_decisions[review["task_id"]],
                "pending",
            )
            resolved = orchestrator.resolve_review(
                parent_id, review["task_id"], "Codex reviewed and accepted the findings."
            )
            self.assertEqual(resolved["status"], "resolved")

    def test_async_review_reserves_parent_gate_before_job_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                project_root=Path(directory),
                workers=(WorkerConfig(
                    worker_id="worker", source="antigravity", model="test",
                    transport="command", command="python3",
                ),),
            )
            orchestrator = Orchestrator(config)
            parent = orchestrator.tasks["parent"] = TaskRecord(
                request=TaskRequest(
                    task_id="parent", title="parent", objective="build",
                    implementation_plan="implement", verification_instructions="test",
                    verification_commands=[["python3", "-c", "print('ok')"]],
                ),
                worker_id="worker", status="completed",
                result=WorkerResult(
                    task_id="parent", worker_id="worker", status="completed",
                    verification_report="passed",
                ),
            )
            with patch.object(orchestrator, "_submit_job", return_value={"status": "queued"}):
                orchestrator.dispatch_review_async({
                    "task_id": "parent", "review_task_id": "review-parent",
                    "review_instructions": "audit it",
                })
            self.assertEqual(parent.review_decisions["review-parent"], "pending")
            with self.assertRaisesRegex(OrchestratorError, "independent reviews"):
                parent.patch_applied = True
                parent.verification_passed = True
                parent.verification_workspace_hash = _workspace_fingerprint(
                    config.project_root, []
                )
                orchestrator.approve_task("parent", "looks good")

    def test_workspace_change_invalidates_verified_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "tracked.txt"
            target.write_text("verified", encoding="utf-8")
            config = AppConfig(
                project_root=root,
                workers=(WorkerConfig(
                    worker_id="worker", source="test", model="test",
                    transport="command", command="unused",
                ),),
            )
            orchestrator = Orchestrator(config)

            class FakeProvider:
                def execute(self, worker, task):
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        status="completed",
                        summary="done",
                        verification_report="passed",
                    )

            with patch("universal_mcp.orchestrator.provider_for", return_value=FakeProvider()):
                task_id = orchestrator.dispatch({
                    "objective": "verify one file",
                    "implementation_plan": "do not change it",
                    "verification_instructions": "check it",
                    "verification_commands": [["python3", "-c", "print('ok')"]],
                })["task_id"]
            record = orchestrator.tasks[task_id]
            record.tracked_paths = ["tracked.txt"]
            record.verification_passed = True
            record.verification_workspace_hash = _workspace_fingerprint(root, record.tracked_paths)
            target.write_text("changed later", encoding="utf-8")
            with self.assertRaisesRegex(OrchestratorError, "workspace changed"):
                orchestrator.approve_task(task_id, "approve")
            self.assertFalse(record.verification_passed)
            self.assertEqual(record.stage, "verification_stale")

    def test_pause_resume_and_cancel_gate_mutating_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(
                project_root=Path(directory),
                workers=(WorkerConfig(
                    worker_id="worker", source="test", model="test",
                    transport="command", command="unused",
                ),),
            )
            orchestrator = Orchestrator(config)

            class FakeProvider:
                def execute(self, worker, task):
                    return WorkerResult(
                        task_id=task.task_id,
                        worker_id=worker.worker_id,
                        status="completed",
                        summary="done",
                        verification_report="checked",
                    )

            with patch("universal_mcp.orchestrator.provider_for", return_value=FakeProvider()):
                task_id = orchestrator.dispatch({
                    "objective": "lifecycle test",
                    "implementation_plan": "Return without a patch.",
                    "verification_instructions": "Run the check.",
                    "verification_commands": [["python3", "-c", "print('ok')"]],
                })["task_id"]
                orchestrator.pause_task(task_id, "reviewing")
                with self.assertRaises(OrchestratorError):
                    orchestrator.run_verification(task_id, True, 10)
                with self.assertRaises(OrchestratorError):
                    orchestrator.send_worker_message(task_id, "continue")
                orchestrator.resume_task(task_id)
                self.assertTrue(
                    orchestrator.run_verification(task_id, True, 10)["verification_passed"]
                )
                orchestrator.cancel_task(task_id, "no longer needed")
                with self.assertRaises(OrchestratorError):
                    orchestrator.run_verification(task_id, True, 10)


if __name__ == "__main__":
    unittest.main()
