"""Startup failure detection tests without importing the GTK desktop host."""

import ast
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "siphon-studio" / "siphon" / "desktop" / "app.py"


def load_startup_functions(namespace):
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    bridge = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DesktopBridge"
    )
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_file_size", "_startup_log_error",
        }
    ]
    nodes.append(next(
        node for node in bridge.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_model_action"
    ))
    module = ast.Module(body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0,
        ),
        *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(APP_PATH), "exec"), namespace)
    return namespace


class DesktopStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log_path = Path(self.temp.name) / "server.log"
        self.events = []
        self.process_running = False

        def launch(_profile):
            self.process_running = True
            return True, "started"

        def status(_profile, *, include_metrics=True):
            return SimpleNamespace(running=self.process_running)

        def stop(_profile, *, force=False):
            self.process_running = False
            return True, "stopped"

        self.runner = SimpleNamespace(
            up=mock.Mock(side_effect=launch),
            wait_healthy=mock.Mock(return_value=False),
            status=mock.Mock(side_effect=status),
            down=mock.Mock(side_effect=stop),
        )

        def immediate_thread(*, target, **_kwargs):
            return SimpleNamespace(start=target, is_alive=lambda: False)

        def idle_add(callback, *args):
            return callback(*args)

        self.functions = load_startup_functions({
            "Path": Path,
            "threading": SimpleNamespace(Thread=immediate_thread),
            "runner": self.runner,
            "GLib": SimpleNamespace(idle_add=idle_add),
            "_profile_config": lambda _profile: {},
            "_profile_memory_layout": lambda *_args, **_kwargs: None,
            "_profile_model_available": lambda _profile: False,
            "fetch_sys_stats": lambda: object(),
            "_validate_resource_settings": lambda *_args, **_kwargs: (True, ""),
            "_probe_inference": lambda _profile: None,
            "_record_runtime_memory": mock.Mock(),
        })
        self.profile = SimpleNamespace(name="fixture-model", log_file=self.log_path)
        self.bridge = SimpleNamespace(
            _catalogue_lock=threading.RLock(),
            _starting_models=set(),
            _stopping_models=set(),
            _log_offsets={self.profile.name: 0},
            _request_metrics={},
            _active_generations={},
            _dstorage_baselines={},
            _live_timing_samples={},
            _live_stream_metrics={},
            _dstorage_rate_samples={},
            _telemetry_epoch=0,
            _shutdown_event=threading.Event(),
            _launch_lock=threading.RLock(),
            _model_action_threads=[],
            _ensure_catalogue_state=lambda: None,
            _profile_for=lambda key: self.profile if key == "fixture" else None,
            _emit_event=self.events.append,
        )

        def finish(starting, ok, message, _profile_name):
            self.events.append({
                "type": "runtime_complete", "starting": starting,
                "ok": ok, "message": message,
            })
            return False

        self.bridge._finish_model_action = finish

    def run_startup(self):
        self.functions["_run_model_action"](self.bridge, True, "fixture")
        return self.events[-1]

    def test_loading_failures_return_the_complete_log_record(self):
        records = (
            "unknown model architecture: 'qwen4exp'",
            "error loading model: unable to initialize tensors",
            "failed to load model: invalid GGUF metadata",
            "exiting due to model loading error",
        )
        for record in records:
            with self.subTest(record=record):
                self.log_path.write_text(record + "\n", encoding="utf-8")
                self.assertEqual(
                    self.functions["_startup_log_error"](self.log_path, 0), record,
                )

    def test_startup_surfaces_new_qwen_architecture_error_without_stale_log(self):
        old_record = "unknown model architecture: 'old-model'"
        new_record = "error loading model: unknown model architecture: 'qwen4exp'"
        self.log_path.write_text(old_record + "\n", encoding="utf-8")

        def launch(_profile):
            self.process_running = True
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write(new_record + "\n")
            return True, "started"

        self.runner.up.side_effect = launch
        result = self.run_startup()

        self.assertFalse(result["ok"])
        self.assertIn(new_record, result["message"])
        self.assertNotIn(old_record, result["message"])
        self.runner.down.assert_called_once_with(self.profile, force=True)

    def test_error_before_start_offset_is_ignored(self):
        old_record = "failed to load model: previous launch"
        self.log_path.write_text(old_record + "\ncontinuing\n", encoding="utf-8")
        start_offset = self.log_path.stat().st_size
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write("warning: optional metadata unavailable; continuing\n")

        self.assertIsNone(
            self.functions["_startup_log_error"](self.log_path, start_offset)
        )

    def test_nonfatal_warning_and_generic_error_are_not_loading_failures(self):
        self.log_path.write_text(
            "warning: optional tokenizer metadata unavailable\n"
            "error: cache unavailable; continuing to load model\n",
            encoding="utf-8",
        )

        self.assertIsNone(self.functions["_startup_log_error"](self.log_path, 0))

    def test_exited_process_is_not_reported_as_a_ten_minute_timeout(self):
        def exit_before_ready(_profile, *, timeout, interval, cancel_event=None):
            self.process_running = False
            return False

        self.runner.wait_healthy.side_effect = exit_before_ready

        result = self.run_startup()

        self.assertFalse(result["ok"])
        self.assertIn("process exited before becoming healthy", result["message"])
        self.assertNotIn("10 minutes", result["message"])

    def test_running_process_after_readiness_window_is_reported_as_timeout(self):
        self.runner.wait_healthy.return_value = False

        result = self.run_startup()

        self.assertFalse(result["ok"])
        self.assertIn("within 10 minutes", result["message"])
        self.assertNotIn("process exited", result["message"])
        self.runner.wait_healthy.assert_called_once_with(
            self.profile, timeout=600.0, interval=1.0,
            cancel_event=self.bridge._shutdown_event,
        )

    def test_memory_calibration_is_saved_only_after_successful_readiness(self):
        self.runner.wait_healthy.return_value = True
        self.assertTrue(self.run_startup()["ok"])
        self.functions["_record_runtime_memory"].assert_called_once_with(self.profile, {}, 0)

    def test_failed_probe_does_not_record_a_successful_memory_calibration(self):
        self.runner.wait_healthy.return_value = True
        self.functions["_probe_inference"] = lambda _profile: "inference failed"
        self.assertFalse(self.run_startup()["ok"])
        self.functions["_record_runtime_memory"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
