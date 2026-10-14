"""Desktop close owns model cleanup, including startup/close races."""

import sys
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "siphon-studio"))
from siphon.desktop import app
from siphon.core.profiles import Profile


class DesktopShutdownTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.profile = Profile(name="closing-model", env={"MODEL": "/unused.gguf"})
        self.bridge = app.DesktopBridge.__new__(app.DesktopBridge)
        self.bridge.profiles = {self.profile.name: self.profile}
        self.bridge.profile_by_model_key = {"model": self.profile.name}
        self.bridge._closed = False
        self.bridge._chat_lock = threading.Lock()
        self.stop = threading.Event()
        self.bridge._chat_stop_events = {"active-chat": self.stop}
        self.bridge._telemetry_epoch = 0
        self.bridge._starting_models = set()
        self.bridge._stopping_models = set()
        for key in ("_log_offsets", "_request_metrics", "_active_generations",
                    "_dstorage_baselines", "_live_timing_samples", "_live_stream_metrics",
                    "_dstorage_rate_samples"):
            setattr(self.bridge, key, {})
        self.bridge._emit_event = lambda _event: None
        self.bridge._ensure_catalogue_state()

    def test_close_cancels_chat_and_keeps_completion_pending_until_all_models_stop(self):
        entered, release = threading.Event(), threading.Event()
        finished = []
        other = Profile(name="other-loaded-model", env={"MODEL": "/other.gguf"})
        self.bridge.profiles[other.name] = other

        def shutdown(profiles):
            self.assertEqual({p.name for p in profiles}, {"closing-model", "other-loaded-model"})
            entered.set()
            self.assertTrue(release.wait(2))
            return []

        with mock.patch.object(app.runner, "shutdown", side_effect=shutdown, create=True), \
             mock.patch.object(app, "discover", return_value={}), \
             mock.patch.object(app.GLib, "idle_add", side_effect=lambda cb, *args: cb(*args)):
            self.bridge.close(lambda errors: finished.append(errors))
            self.assertTrue(entered.wait(2))
            self.assertTrue(self.stop.is_set())
            self.assertEqual(finished, [])
            self.bridge.close()  # must not spawn a duplicate cleanup
            release.set()
            self.bridge.wait_closed()
        self.assertEqual(finished, [[]])

    def test_close_before_delayed_validation_finishes_never_launches_model(self):
        entered, release = threading.Event(), threading.Event()
        def validate(*_args, **_kwargs):
            entered.set()
            self.assertTrue(release.wait(2))
            return True, "safe"

        with mock.patch.object(app, "_profile_config", return_value={}), \
             mock.patch.object(app, "fetch_sys_stats"), \
             mock.patch.object(app, "_profile_memory_layout", return_value=None), \
             mock.patch.object(app, "_profile_model_available", return_value=False), \
             mock.patch.object(app, "_validate_resource_settings", side_effect=validate), \
             mock.patch.object(app.runner, "up") as up, \
             mock.patch.object(app.runner, "shutdown", return_value=[], create=True), \
             mock.patch.object(app, "discover", return_value={}):
            self.bridge._run_model_action(True, "model")
            self.assertTrue(entered.wait(2))
            self.bridge.close()
            self.bridge.wait_closed()
            release.set()
            for thread in self.bridge._model_action_threads:
                thread.join(2)
            up.assert_not_called()

    def test_close_serializes_with_spawn_and_stops_the_new_process(self):
        spawn_entered, release_spawn, cleanup_done = threading.Event(), threading.Event(), threading.Event()
        state = {"loaded": False}
        def spawn(_profile):
            spawn_entered.set()
            self.assertTrue(release_spawn.wait(2))
            state["loaded"] = True
            return True, "started"
        def shutdown(_profiles):
            self.assertTrue(state["loaded"])
            state["loaded"] = False
            cleanup_done.set()
            return []

        with mock.patch.object(app, "_profile_config", return_value={}), \
             mock.patch.object(app, "fetch_sys_stats"), \
             mock.patch.object(app, "_profile_memory_layout", return_value=None), \
             mock.patch.object(app, "_profile_model_available", return_value=False), \
             mock.patch.object(app, "_validate_resource_settings", return_value=(True, "safe")), \
             mock.patch.object(app.runner, "up", side_effect=spawn), \
             mock.patch.object(app.runner, "shutdown", side_effect=shutdown, create=True), \
             mock.patch.object(app, "discover", return_value={}):
            self.bridge._run_model_action(True, "model")
            self.assertTrue(spawn_entered.wait(2))
            self.bridge.close()
            self.assertFalse(cleanup_done.is_set())
            release_spawn.set()
            self.bridge.wait_closed()
        self.assertTrue(cleanup_done.is_set())
        self.assertFalse(state["loaded"])

    def test_shutdown_errors_are_reported_and_a_second_close_retries(self):
        finished = []
        with mock.patch.object(app.runner, "shutdown", side_effect=[["stop failed"], []], create=True), \
             mock.patch.object(app, "discover", return_value={}), \
             mock.patch.object(app.GLib, "idle_add", side_effect=lambda cb, *args: cb(*args)):
            self.bridge.close(lambda errors: finished.append(errors))
            self.assertEqual(self.bridge.wait_closed(), ["stop failed"])
            self.bridge.close(lambda errors: finished.append(errors))
            self.assertEqual(self.bridge.wait_closed(), [])
        self.assertEqual(finished, [["stop failed"], []])

    def test_completion_callback_can_request_another_close_without_losing_it(self):
        finished = []
        def first(errors):
            finished.append("first")
            self.bridge.close(lambda _errors: finished.append("second"))
        with mock.patch.object(app.runner, "shutdown", return_value=[], create=True), \
             mock.patch.object(app, "discover", return_value={}), \
             mock.patch.object(app.GLib, "idle_add", side_effect=lambda cb, *args: cb(*args)):
            self.bridge.close(first)
            self.bridge.wait_closed()
        self.assertEqual(finished, ["first", "second"])

    def test_native_close_waits_for_cleanup_before_destroying_window(self):
        callbacks = []
        window = SimpleNamespace(set_sensitive=mock.Mock(), set_title=mock.Mock(), destroy=mock.Mock())
        host = SimpleNamespace(bridge=SimpleNamespace(close=callbacks.append), window=window, _closing=False)
        host._finish_close = lambda errors: app.SiphonApplication._finish_close(host, errors)
        result = app.SiphonApplication._on_close_request(host, window)
        self.assertTrue(result)
        window.destroy.assert_not_called()
        callbacks[0]([])
        window.destroy.assert_called_once()

    def test_application_shutdown_also_waits_for_models(self):
        actions = []
        host = SimpleNamespace(bridge=SimpleNamespace(close=lambda: actions.append("close"),
                                                     wait_closed=lambda: actions.append("wait")))
        with mock.patch.object(app.Gtk.Application, "do_shutdown", side_effect=lambda _host: actions.append("gtk")):
            app.SiphonApplication.do_shutdown(host)
        self.assertEqual(actions, ["close", "wait", "gtk"])

    def test_native_failed_cleanup_reenables_close_for_retry(self):
        callbacks = []
        window = SimpleNamespace(set_sensitive=mock.Mock(), set_title=mock.Mock(), destroy=mock.Mock())
        host = SimpleNamespace(bridge=SimpleNamespace(close=callbacks.append), window=window, _closing=False)
        host._finish_close = lambda errors: app.SiphonApplication._finish_close(host, errors)
        app.SiphonApplication._on_close_request(host, window)
        app.SiphonApplication._on_close_request(host, window)
        self.assertEqual(len(callbacks), 1)
        callbacks[0](["cannot stop model"])
        window.destroy.assert_not_called()
        window.set_sensitive.assert_called_with(True)
        self.assertFalse(host._closing)
        app.SiphonApplication._on_close_request(host, window)
        self.assertEqual(len(callbacks), 2)
        callbacks[1]([])
        window.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
