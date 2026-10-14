"""Studio model removal keeps profiles and GGUF artifacts intact."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "siphon-studio"))

from siphon.core.profiles import Profile
from siphon.desktop import app


class DesktopModelRemovalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.studio_dir = self.root / "studio"
        env = mock.patch.dict(os.environ, {
            "SIPHON_STUDIO_DIR": str(self.studio_dir),
            "SIPHON_REPO_ROOT": str(ROOT),
        })
        env.start()
        self.addCleanup(env.stop)
        self.events = []

    def profile(self, name, model_path, *, bundled=False):
        profile = Profile(
            name=name,
            env={"MODEL": str(model_path), "PORT": "8090", "CTX": "4096",
                 "CACHE_RAM_MIB": "256", "SIPHON_TEMPERATURE": "0.4"},
            source="bundled" if bundled else "app",
            path=self.root / "configs" / f"{name}.env" if bundled else None,
        )
        if not bundled:
            profile.save()
        return profile

    def bridge(self, profiles):
        with mock.patch.object(app, "discover", return_value=profiles):
            bridge = app.DesktopBridge(mock.Mock(), mock.Mock())
        bridge._emit_async = self.events.append
        bridge._emit_event = self.events.append
        return bridge

    def send_remove(self, bridge, model_key, removal_id="remove-1"):
        javascript_result = mock.Mock()
        javascript_result.get_js_value.return_value.to_string.return_value = json.dumps({
            "action": "remove_model",
            "model": model_key,
            "removal_id": removal_id,
        })

        def immediate_thread(*, target, args=(), **_kwargs):
            return SimpleNamespace(start=lambda: target(*args))

        with mock.patch.object(app.threading, "Thread", side_effect=immediate_thread):
            bridge._message_received(mock.Mock(), javascript_result)

    def test_removal_reports_completion_then_empty_catalogue_without_deleting_sources(self):
        model = self.root / "model.gguf"
        model.write_bytes(b"fixture GGUF bytes")
        profile = self.profile("local-model", model)
        bridge = self.bridge({profile.name: profile})
        key = next(iter(bridge.profile_by_model_key))
        profile_bytes = profile.path.read_bytes()
        model_bytes = model.read_bytes()

        with mock.patch.object(app.runner, "status", return_value=SimpleNamespace(running=False)):
            self.send_remove(bridge, key, "confirm-42")

        self.assertEqual([event["type"] for event in self.events], ["model_removed", "catalogue"])
        self.assertEqual(self.events[0], {
            "type": "model_removed", "model_key": key, "removal_id": "confirm-42",
            "ok": True, "message": mock.ANY,
        })
        catalogue = self.events[1]
        self.assertEqual(catalogue["models"], [])
        self.assertEqual(catalogue["library_count"], 0)
        self.assertEqual(catalogue["library_storage_gib"], 0)
        self.assertIsNone(bridge._profile_for(key))
        self.assertEqual(profile.path.read_bytes(), profile_bytes)
        self.assertEqual(model.read_bytes(), model_bytes)

    def test_unavailable_registered_model_can_be_removed(self):
        missing_model = self.root / "missing.gguf"
        profile = self.profile("offline-model", missing_model)
        bridge = self.bridge({profile.name: profile})
        key = next(iter(bridge.profile_by_model_key))
        self.assertFalse(app._profile_model_available(profile))

        with mock.patch.object(app.runner, "status", return_value=SimpleNamespace(running=False)):
            self.send_remove(bridge, key)

        self.assertEqual([event["type"] for event in self.events], ["model_removed", "catalogue"])
        self.assertTrue(self.events[0]["ok"])
        self.assertEqual(self.events[1]["models"], [])

    def test_exclusion_survives_restart_and_blocks_bundled_fallback(self):
        override_model = self.root / "override.gguf"
        fallback_model = self.root / "fallback.gguf"
        override_model.touch()
        fallback_model.touch()
        override = self.profile("qwen3.6-moe-tq3", override_model)
        first_bridge = self.bridge({override.name: override})

        with mock.patch.object(app.runner, "status", return_value=SimpleNamespace(running=False)):
            self.send_remove(first_bridge, "qwen35")
        self.assertEqual([event["type"] for event in self.events], ["model_removed", "catalogue"])
        self.events.clear()

        fallback = self.profile("qwen3.6-moe-tq3", fallback_model, bundled=True)
        restarted = self.bridge({fallback.name: fallback})
        self.assertNotIn("qwen35", restarted.profile_by_model_key)
        self.assertIsNone(restarted._profile_for("qwen35"))
        restarted._schedule_catalogue()
        self.assertEqual(self.events[-1]["models"], [])

    def test_readding_excluded_profile_restores_registration_and_preserves_saved_settings(self):
        model = self.root / "readd.gguf"
        model.write_bytes(b"same fixture bytes")
        profile = self.profile("imported-readd", model)
        profile.model_files = [str(model)]
        profile.model_folder = str(self.root)
        profile.env["CACHE_RAM_MIB"] = "0"
        profile.env["SIPHON_TEMPERATURE"] = "0.25"
        profile.save()
        bridge = self.bridge({profile.name: profile})
        key = next(iter(bridge.profile_by_model_key))
        original_profile_bytes = profile.path.read_bytes()
        original_model_bytes = model.read_bytes()

        with mock.patch.object(app.runner, "status", return_value=SimpleNamespace(running=False)):
            self.send_remove(bridge, key)
        self.assertEqual([event["type"] for event in self.events], ["model_removed", "catalogue"])
        self.assertIsNone(bridge._profile_for(key))
        self.events.clear()

        bridge = self.bridge({profile.name: Profile.from_json(profile.path)})
        profile = bridge.profiles[profile.name]

        metadata = {"parts": [str(model)]}
        with mock.patch.object(app, "inspect_model_files", return_value=metadata):
            result = bridge._import_model(model)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["model_key"], key)
        self.assertIs(bridge._profile_for(key), profile)
        self.assertEqual(profile.path.read_bytes(), original_profile_bytes)
        self.assertEqual(Profile.from_json(profile.path).env["CACHE_RAM_MIB"], "0")
        self.assertEqual(Profile.from_json(profile.path).env["SIPHON_TEMPERATURE"], "0.25")
        self.assertEqual(model.read_bytes(), original_model_bytes)

    def test_model_info_totals_count_only_visible_registered_profiles(self):
        visible_model = self.root / "visible.gguf"
        hidden_model = self.root / "hidden.gguf"
        with visible_model.open("wb") as stream:
            stream.truncate(16 * 1024 * 1024)
        with hidden_model.open("wb") as stream:
            stream.truncate(32 * 1024 * 1024)
        visible = self.profile("visible-model", visible_model)
        hidden = self.profile("hidden-model", hidden_model)
        bridge = self.bridge({visible.name: visible, hidden.name: hidden})
        keys_by_profile = {name: key for key, name in bridge.profile_by_model_key.items()}

        with mock.patch.object(app.runner, "status", return_value=SimpleNamespace(running=False)):
            self.send_remove(bridge, keys_by_profile[hidden.name])
        self.events.clear()

        bridge._finish_model_info({
            "type": "model_info", "model_key": keys_by_profile[visible.name], "available": True,
        })
        info = self.events[-1]
        self.assertEqual(info["library_count"], 1)
        self.assertEqual(info["library_storage_gib"], 0.02)

    def test_running_starting_and_stopping_models_cannot_be_removed(self):
        model = self.root / "busy.gguf"
        model.touch()
        profile = self.profile("busy-model", model)
        bridge = self.bridge({profile.name: profile})
        key = next(iter(bridge.profile_by_model_key))

        for running, transitions in (
            (True, set()),
            (False, {"starting"}),
            (False, {"stopping"}),
        ):
            self.events.clear()
            bridge._starting_models = {profile.name} if "starting" in transitions else set()
            bridge._stopping_models = {profile.name} if "stopping" in transitions else set()
            with mock.patch.object(app.runner, "status", return_value=SimpleNamespace(running=running)):
                self.send_remove(bridge, key, f"busy-{running}-{transitions}")
            self.assertEqual([event["type"] for event in self.events], ["model_removed"])
            self.assertFalse(self.events[0]["ok"])
            self.assertEqual(len(self.events), 1)
            self.assertIs(bridge._profile_for(key), profile)

    def test_stop_action_marks_transition_before_background_work_starts(self):
        model = self.root / "stopping.gguf"
        model.touch()
        profile = self.profile("stopping-model", model)
        bridge = self.bridge({profile.name: profile})

        with mock.patch.object(app.threading, "Thread", return_value=SimpleNamespace(start=lambda: None)):
            bridge._run_model_action(False, next(iter(bridge.profile_by_model_key)))
        self.assertIn(profile.name, bridge._stopping_models)

        with mock.patch.object(app.runner, "status", return_value=SimpleNamespace(running=False)):
            self.send_remove(bridge, next(iter(bridge.profile_by_model_key)), "during-stop")
        self.assertFalse(self.events[0]["ok"])
        self.assertIn("finish stopping", self.events[0]["message"])

        bridge._finish_model_action(False, True, "stopped", profile.name)
        self.events.clear()
        with mock.patch.object(app.runner, "status", return_value=SimpleNamespace(running=False)):
            self.send_remove(bridge, next(iter(bridge.profile_by_model_key)), "after-stop")
        self.assertTrue(self.events[0]["ok"])

    def test_invalid_key_and_persistence_failure_leave_catalogue_visible(self):
        model = self.root / "write-failure.gguf"
        model.touch()
        profile = self.profile("write-failure", model)
        bridge = self.bridge({profile.name: profile})
        key = next(iter(bridge.profile_by_model_key))

        self.send_remove(bridge, "missing-key", "invalid-id")
        self.assertEqual([event["type"] for event in self.events], ["model_removed"])
        self.assertFalse(self.events[-1]["ok"])
        self.assertEqual(self.events[-1]["model_key"], "missing-key")
        self.assertEqual(len(self.events), 1)

        self.events.clear()
        with mock.patch.object(app, "save_excluded_model_keys", side_effect=OSError("read only")):
            self.send_remove(bridge, key, "write-error")
        self.assertFalse(self.events[0]["ok"])
        self.assertEqual(len(self.events), 1)
        self.assertIs(bridge._profile_for(key), profile)

    def test_metadata_and_poll_started_before_removal_cannot_emit_afterward(self):
        model = self.root / "late.gguf"
        model.touch()
        profile = self.profile("late-model", model)
        bridge = self.bridge({profile.name: profile})
        key = next(iter(bridge.profile_by_model_key))

        with mock.patch.object(app.threading, "Thread") as thread:
            bridge._schedule_model_info(key)
        metadata_work = thread.call_args.kwargs["target"]
        old_telemetry_epoch = bridge._telemetry_epoch
        self.events.clear()

        with mock.patch.object(app.runner, "status", return_value=SimpleNamespace(running=False)):
            self.send_remove(bridge, key)
        self.assertEqual([event["type"] for event in self.events], ["model_removed", "catalogue"])
        self.events.clear()

        system = SimpleNamespace(mem_free_gb=8, mem_total_gb=16,
                                 gpu_mib_free=4096, gpu_mib_total=8192)
        with mock.patch.object(app, "_cached_model_info", return_value={
            "type": "model_info", "model_key": key, "available": True,
        }), mock.patch.object(app, "fetch_sys_stats", return_value=system), \
             mock.patch.object(app, "_recommended_settings", return_value={}), \
             mock.patch.object(app.GLib, "idle_add", side_effect=lambda fn, *args: fn(*args)):
            metadata_work()
        bridge._finish_poll({"type": "telemetry", "running": True}, old_telemetry_epoch)

        self.assertEqual(self.events, [])

    def test_selecting_a_removed_key_does_not_replace_the_current_model(self):
        first = self.profile('first-model', self.root / 'first.gguf')
        second = self.profile('second-model', self.root / 'second.gguf')
        bridge = self.bridge({first.name: first, second.name: second})
        keys = {name: key for key, name in bridge.profile_by_model_key.items()}
        with mock.patch.object(app.runner, 'status', return_value=SimpleNamespace(running=False)):
            self.send_remove(bridge, keys[first.name])
        bridge.model_key = keys[second.name]
        result = mock.Mock()
        result.get_js_value.return_value.to_string.return_value = json.dumps({
            'action': 'select_model', 'model': keys[first.name],
        })
        bridge._message_received(mock.Mock(), result)
        self.assertEqual(bridge.model_key, keys[second.name])

    def test_metadata_queued_before_removal_is_rejected_after_readd(self):
        model = self.root / 'queued.gguf'
        model.write_bytes(b'fixture')
        profile = self.profile('queued-model', model)
        bridge = self.bridge({profile.name: profile})
        key = next(iter(bridge.profile_by_model_key))
        with mock.patch.object(app.threading, 'Thread') as thread:
            bridge._schedule_model_info(key)
        work = thread.call_args.kwargs['target']
        queued = []
        system = SimpleNamespace(mem_free_gb=8, mem_total_gb=16,
                                 gpu_mib_free=4096, gpu_mib_total=8192)
        with mock.patch.object(app, '_cached_model_info', return_value={
            'type': 'model_info', 'model_key': key, 'name': 'Old metadata', 'available': True,
        }), mock.patch.object(app, 'fetch_sys_stats', return_value=system), \
             mock.patch.object(app, '_recommended_settings', return_value={}), \
             mock.patch.object(app.GLib, 'idle_add', side_effect=lambda callback, *args: queued.append((callback, args))):
            work()
        with mock.patch.object(app.runner, 'status', return_value=SimpleNamespace(running=False)):
            self.send_remove(bridge, key)
        with mock.patch.object(app, 'inspect_model_files', return_value={'parts': [str(model)]}):
            self.assertTrue(bridge._import_model(model, emit=False)['ok'])
        self.events.clear()
        for callback, args in queued:
            callback(*args)
        self.assertEqual(self.events, [])
        bridge._finish_model_info({'type': 'model_info', 'model_key': key, 'name': 'Current metadata'})
        self.assertEqual(self.events[-1]['name'], 'Current metadata')


if __name__ == "__main__":
    unittest.main()
