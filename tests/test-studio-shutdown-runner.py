"""Bounded desktop runner shutdown tests using disposable process trees."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "siphon-studio"))

from siphon.core import runner
from siphon.core.profiles import Profile


class StudioShutdownRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.studio = self.root / "studio"
        self.env_patch = mock.patch.dict(os.environ, {
            "SIPHON_STUDIO_DIR": str(self.studio),
            "SIPHON_REPO_ROOT": str(self.root),
            "SIPHON_DRIVER": "direct",
        }, clear=False)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.old_driver = runner._DRIVER
        runner._DRIVER = "direct"
        self.addCleanup(setattr, runner, "_DRIVER", self.old_driver)

    def profile(self, name="shutdown-fixture", model=None):
        model = model or (self.root / "models" / "shared.gguf")
        model.parent.mkdir(parents=True, exist_ok=True)
        model.touch(exist_ok=True)
        return Profile(name=name, env={"MODEL": str(model), "PORT": "8090"})

    def stubborn_tree(self, *, env=None, direct=False, profile=None, config=None):
        """Start a parent and child in one disposable process group."""
        pids_file = self.root / f"pids-{time.monotonic_ns()}.json"
        script = self.root / f"tree-{time.monotonic_ns()}.py"
        script.write_text(
            "import os, signal, subprocess, sys, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "worker = 'import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)'\n"
            "child = subprocess.Popen([sys.executable, '-c', worker])\n"
            "with open(sys.argv[1], 'w', encoding='utf-8') as stream:\n"
            "    stream.write(str([os.getpid(), child.pid]))\n"
            "time.sleep(120)\n",
            encoding="utf-8",
        )
        if direct:
            ok, message = runner._direct_up(
                profile,
                [sys.executable, str(script), str(pids_file)],
                env or {},
            )
            self.assertTrue(ok, message)
            pid_file = self.studio / "state" / "run" / f"{profile.name}.pid"
            pid = int(pid_file.read_text(encoding="utf-8"))
        else:
            child_env = dict(os.environ)
            child_env.pop("SIPHON_PROFILE", None)
            child_env.pop("CONFIG", None)
            child_env.update(env or {})
            proc = subprocess.Popen(
                [sys.executable, str(script), str(pids_file)],
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            pid = proc.pid
            self.addCleanup(proc.wait)
        deadline = time.monotonic() + 5
        while not pids_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(pids_file.exists(), "fixture process tree did not start")
        pids = json.loads(pids_file.read_text(encoding="utf-8"))
        self.assertEqual(pids[0], pid)
        self.addCleanup(self._reap_runner_process, pid)
        self.addCleanup(self._kill_fixture_group, pid)
        return pid, pids[1]

    @staticmethod
    def _proc_active(pid):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except FileNotFoundError:
            return False
        fields = stat[stat.rfind(")") + 2:].split()
        return bool(fields) and fields[0] != "Z"

    @classmethod
    def _wait_inactive(cls, pids, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not any(cls._proc_active(pid) for pid in pids):
                return True
            time.sleep(0.02)
        return not any(cls._proc_active(pid) for pid in pids)

    @staticmethod
    def _kill_fixture_group(pid):
        try:
            if runner._active_group_members(pid):
                os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _stop_process(proc):
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    @staticmethod
    def _reap_runner_process(pid):
        proc = runner._DIRECT_PROCESSES.pop(pid, None)
        if proc is not None:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

    def test_direct_shutdown_kills_owned_tree_but_preserves_same_model_server(self):
        profile = self.profile()
        config = self.root / "studio" / "state" / "env" / "shutdown-fixture.env"
        tree_pid, child_pid = self.stubborn_tree(
            direct=True, profile=profile, env={"CONFIG": str(config)},
        )
        self.assertTrue(self._proc_active(tree_pid))
        self.assertTrue(self._proc_active(child_pid))
        environ = Path(f"/proc/{tree_pid}/environ").read_bytes().split(b"\0")
        self.assertIn(b"SIPHON_PROFILE=shutdown-fixture", environ)
        self.assertIn(f"CONFIG={config}".encode(), environ)

        outside_env = dict(os.environ)
        outside_env.pop("SIPHON_PROFILE", None)
        outside_env.pop("CONFIG", None)
        outside = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)",
             f"llama-server --model={profile.env['MODEL']}"],
            env=outside_env,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self._stop_process, outside)

        self.assertEqual(runner.shutdown([profile]), [])
        self.assertTrue(self._wait_inactive([tree_pid, child_pid]))
        self.assertIsNone(outside.poll(), "an unrelated same-model process was stopped")
        self.assertFalse((self.studio / "state" / "run" / f"{profile.name}.pid").exists())
        self.assertEqual(runner.shutdown([profile]), [])
        self.assertIsNone(outside.poll())

    def test_unowned_reused_legacy_pid_is_preserved_and_reported(self):
        profile = self.profile("reused-pid")
        pid, child = self.stubborn_tree()
        pid_file = self.studio / "state" / "run" / f"{profile.name}.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(pid), encoding="utf-8")

        errors = runner.shutdown([profile])

        self.assertTrue(errors, "shutdown must report an unowned legacy PID")
        self.assertTrue(self._proc_active(pid))
        self.assertTrue(self._proc_active(child))
        self.assertEqual(pid_file.read_text(encoding="utf-8"), str(pid))

    def test_direct_pid_with_reused_identity_is_never_signalled(self):
        profile = self.profile("reused-owned-pid")
        pid, child = self.stubborn_tree(direct=True, profile=profile)
        owner_path = self.studio / "state" / "run" / f"{profile.name}.owner.json"
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        owner["starttime"] += 1
        owner_path.write_text(json.dumps(owner), encoding="utf-8")

        errors = runner.shutdown([profile])

        self.assertTrue(errors)
        self.assertIn("identity changed", errors[0])
        self.assertTrue(self._proc_active(pid))
        self.assertTrue(self._proc_active(child))
        self.assertTrue(owner_path.exists())

    def test_legacy_pid_with_matching_profile_environment_is_stopped(self):
        profile = self.profile("legacy-owned")
        pid, child = self.stubborn_tree(env={"SIPHON_PROFILE": profile.name})
        pid_file = self.studio / "state" / "run" / f"{profile.name}.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(pid), encoding="utf-8")

        self.assertEqual(runner.shutdown([profile]), [])
        self.assertTrue(self._wait_inactive([pid, child]))
        self.assertFalse(pid_file.exists())

    def test_legacy_launcher_already_exited_still_stops_its_child(self):
        profile = self.profile("orphaned-legacy")
        pid, child = self.stubborn_tree(env={"SIPHON_PROFILE": profile.name})
        pid_file = self.studio / "state" / "run" / f"{profile.name}.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(pid), encoding="utf-8")
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        self.assertTrue(self._proc_active(child))
        self.assertEqual(runner.shutdown([profile]), [])
        self.assertFalse(self._proc_active(child), "orphan model outlived desktop shutdown")

    def test_tmux_legacy_script_and_untagged_logging_child_are_owned_by_pane_tree(self):
        profile = self.profile("legacy-tmux-tree")
        cmd_path = runner.paths.cmd_dir() / f"{profile.name}.sh"
        cmd_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen([
            sys.executable, "-c",
            "import signal,subprocess,sys,time; signal.signal(signal.SIGINT,signal.SIG_IGN); "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); time.sleep(120)",
            str(cmd_path),
        ], start_new_session=True, env={"PATH": os.environ["PATH"]})
        self.addCleanup(self._stop_process, proc)
        self.addCleanup(self._kill_fixture_group, proc.pid)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and len(runner._active_group_members(proc.pid)) < 2:
            time.sleep(0.01)
        pids = [pid for pid, _stat in runner._active_group_members(proc.pid)]
        self.assertEqual(len(pids), 2)
        running = True
        def tmux_run(command, **_kwargs):
            nonlocal running
            if command[1] == "has-session":
                return subprocess.CompletedProcess(command, 0 if running else 1, "", "" if running else "no server running")
            if command[1] == "list-panes":
                return subprocess.CompletedProcess(command, 0, str(proc.pid) + "\n", "")
            if command[1] == "kill-session":
                running = False
                return subprocess.CompletedProcess(command, 0, "", "")
            self.fail(command)
        with mock.patch.object(runner, "driver_name", return_value="tmux"), \
             mock.patch.object(runner.subprocess, "run", side_effect=tmux_run):
            errors = runner.shutdown([profile])
        self.assertEqual(errors, [])
        self.assertTrue(self._wait_inactive(pids))

    def test_tmux_shutdown_stops_verified_pane_group_and_checks_session_removal(self):
        profile = self.profile("tmux-owned")
        pid, child = self.stubborn_tree(env={"SIPHON_PROFILE": profile.name})
        session_running = True
        calls = []

        def tmux_run(command, **_kwargs):
            nonlocal session_running
            calls.append(command)
            if command[1] == "has-session":
                return subprocess.CompletedProcess(command, 0 if session_running else 1,
                                                    "", "" if session_running else "no server running")
            if command[1] == "list-panes":
                return subprocess.CompletedProcess(command, 0, str(pid) + "\n", "")
            if command[1] == "kill-session":
                session_running = False
                return subprocess.CompletedProcess(command, 0, "", "")
            self.fail(f"unexpected tmux command: {command}")

        with mock.patch.object(runner, "driver_name", return_value="tmux"), \
             mock.patch.object(runner.subprocess, "run", side_effect=tmux_run):
            errors = runner.shutdown([profile])

        self.assertEqual(errors, [])
        self.assertTrue(self._wait_inactive([pid, child]))
        self.assertTrue(any(command[1] == "kill-session" for command in calls))
        self.assertFalse(session_running)

    def test_systemd_stop_failure_is_returned_to_caller(self):
        profile = self.profile("unit-failure")
        failed = subprocess.CompletedProcess(
            ["systemctl"], 1, "", "unit stop failed",
        )
        with mock.patch.object(runner, "driver_name", return_value="systemd"), \
             mock.patch.object(runner.subprocess, "run", side_effect=[
                 subprocess.CompletedProcess([], 0, "LoadState=loaded\nActiveState=active\nMainPID=123\n", ""),
                 failed,
             ]) as run:
            errors = runner.shutdown([profile])

        self.assertTrue(errors)
        self.assertIn("unit stop failed", errors[0])
        self.assertEqual(run.call_args.args[0], [
            "systemctl", "--user", "stop", "siphon-unit-failure",
        ])

    def test_systemd_success_returns_no_errors(self):
        profile = self.profile("unit-stopped")
        with mock.patch.object(runner, "driver_name", return_value="systemd"), \
             mock.patch.object(
                 runner.subprocess, "run",
                 side_effect=[
                     subprocess.CompletedProcess([], 0, "LoadState=loaded\nActiveState=active\nMainPID=123\n", ""),
                     subprocess.CompletedProcess([], 0, "", ""),
                     subprocess.CompletedProcess([], 0, "LoadState=loaded\nActiveState=inactive\nMainPID=0\n", ""),
                 ],
             ):
            self.assertEqual(runner.shutdown([profile]), [])

    def test_systemd_inactive_or_missing_unit_needs_no_stop(self):
        for load in ("loaded", "not-found"):
            with self.subTest(load=load), \
                 mock.patch.object(runner, "driver_name", return_value="systemd"), \
                 mock.patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess(
                     [], 0, f"LoadState={load}\nActiveState=inactive\nMainPID=0\n", "",
                 )) as run:
                self.assertEqual(runner.shutdown([self.profile("unused-unit")]), [])
                self.assertEqual(run.call_count, 1)
                self.assertIn("timeout", run.call_args.kwargs)

    def test_systemd_stop_success_but_still_active_reports_failure(self):
        active = subprocess.CompletedProcess([], 0, "LoadState=loaded\nActiveState=active\nMainPID=123\n", "")
        with mock.patch.object(runner, "driver_name", return_value="systemd"), \
             mock.patch.object(runner.subprocess, "run", side_effect=[active,
                 subprocess.CompletedProcess([], 0, "", ""), active]):
            errors = runner.shutdown([self.profile("still-active-unit")])
        self.assertTrue(errors)
        self.assertIn("remained active", errors[0])

    def test_direct_records_are_cleaned_when_driver_has_changed(self):
        profile = self.profile("changed-driver")
        pid, child = self.stubborn_tree(direct=True, profile=profile)
        with mock.patch.object(runner, "driver_name", return_value="systemd"), \
             mock.patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess(
                 [], 0, "LoadState=not-found\nActiveState=inactive\nMainPID=0\n", "")):
            self.assertEqual(runner.shutdown([profile]), [])
        self.assertTrue(self._wait_inactive([pid, child]))

    def test_systemd_inactive_main_pid_does_not_hide_remaining_children(self):
        inactive = subprocess.CompletedProcess([], 0,
            "LoadState=loaded\nActiveState=inactive\nMainPID=0\nControlGroup=/user.slice/model.service\n", "")
        with mock.patch.object(runner, "driver_name", return_value="systemd"), \
             mock.patch.object(runner, "_systemd_group_active", return_value=True), \
             mock.patch.object(runner.subprocess, "run", side_effect=[inactive,
                 subprocess.CompletedProcess([], 0, "", ""), inactive]):
            errors = runner.shutdown([self.profile("orphaned-unit")])
        self.assertTrue(errors)
        self.assertIn("remained active", errors[0])

    def test_shutdown_never_signals_its_own_process_group(self):
        with mock.patch.object(runner.os, "killpg") as signal_group:
            error = runner._terminate_owned_group(os.getpgrp(), "any-profile", "")
        self.assertIn("non-dedicated", error)
        signal_group.assert_not_called()

    def test_timeout_does_not_prevent_other_profiles_being_cleaned(self):
        first, second = self.profile("timeout"), self.profile("next")
        with mock.patch.object(runner, "driver_name", return_value="tmux"), \
             mock.patch.object(runner, "_shutdown_direct", return_value=None), \
             mock.patch.object(runner, "_shutdown_tmux", side_effect=[
                 subprocess.TimeoutExpired(["tmux"], 5), None,
             ]) as shutdown_tmux:
            errors = runner.shutdown([first, second])
        self.assertEqual(len(errors), 1)
        self.assertIn("timeout", errors[0])
        self.assertEqual(shutdown_tmux.call_count, 2)

    def test_wait_healthy_returns_without_polling_when_cancelled(self):
        profile = self.profile("cancel-before-poll")
        event = threading.Event()
        event.set()
        with mock.patch.object(runner, "status", side_effect=AssertionError("polled after cancel")):
            self.assertFalse(runner.wait_healthy(profile, cancel_event=event))

    def test_wait_healthy_without_cancel_event_keeps_headless_call_shape(self):
        profile = self.profile("headless-wait")
        healthy = runner.ServerStatus(running=True)
        healthy.health.ok = True
        with mock.patch.object(runner, "status", return_value=healthy) as status:
            self.assertTrue(runner.wait_healthy(profile, timeout=1, interval=0))
        status.assert_called_once_with(profile)

    def test_wait_healthy_event_interrupts_readiness_interval(self):
        profile = self.profile("cancel-during-wait")
        cancelled = threading.Event()
        waiting = threading.Event()

        class ObservedEvent:
            def is_set(self):
                return cancelled.is_set()

            def wait(self, timeout=None):
                waiting.set()
                return cancelled.wait(timeout)

        calls = []
        result = []

        def unhealthy(_profile):
            calls.append(True)
            return runner.ServerStatus(running=True)

        with mock.patch.object(runner, "status", side_effect=unhealthy):
            worker = threading.Thread(target=lambda: result.append(
                runner.wait_healthy(profile, timeout=120, interval=120,
                                    cancel_event=ObservedEvent())
            ))
            worker.start()
            self.assertTrue(waiting.wait(2), "readiness loop did not use event.wait")
            cancelled.set()
            worker.join(2)

        self.assertFalse(worker.is_alive(), "cancel did not interrupt readiness wait")
        self.assertEqual(result, [False])
        self.assertEqual(len(calls), 1, "readiness loop polled again after cancel")


if __name__ == "__main__":
    unittest.main()
