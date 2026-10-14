"""Desktop folder import boundaries; all files/profiles are temporary."""

import os
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import runpy
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "siphon-studio"))
from siphon.core.profiles import Profile
from siphon.desktop import app

write_model = runpy.run_path(str(ROOT / "tests/test-studio-model-files.py"))["write_model"]


class FolderImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patch = mock.patch.dict(os.environ, {"SIPHON_STUDIO_DIR": str(self.root / "studio"),
                                             "SIPHON_REPO_ROOT": str(ROOT)})
        patch.start()
        self.addCleanup(patch.stop)
        self.bridge = app.DesktopBridge.__new__(app.DesktopBridge)
        self.bridge.profiles = {}
        self.bridge.profile_by_model_key = {}
        self.bridge._log_offsets = {}
        self.bridge._import_lock = threading.RLock()
        self.bridge._import_scan_id = "scan-a"
        self.bridge._import_candidates = {}
        self.bridge._closed = False
        self.bridge.window = mock.sentinel.window
        self.events = []
        self.bridge._emit_async = self.events.append
        self.bridge._emit_event = self.events.append
        self.bridge._schedule_model_info = lambda _key: None
        self.stats = SimpleNamespace(mem_total_gb=16, mem_free_gb=12,
                                     gpu_mib_total=6144, gpu_mib_free=6000)

    def candidate(self, name="one"):
        path = self.root / f"{name}.gguf"
        path.write_bytes(b"fixture")
        metadata = {"name": name, "architecture": "qwen3moe", "native_context": 32768,
                    "expert_count": 8, "expert_used_count": 2, "layers": 1,
                    "artifact_gib": 0.01, "dense_gib": 0.001, "expert_gib": 0.009,
                    "parts": [str(path)], "part_count": 1}
        value = {"id": name, "path": str(path), "parts": [str(path)],
                 "available": True, "metadata": metadata, "name": name}
        self.bridge._import_candidates[name] = value
        return value

    def test_profile_reload_keeps_ordered_parts_and_folder(self):
        value = self.candidate()
        profile = Profile(name="parts", env={"MODEL": value["path"]})
        profile.model_files = value["parts"]
        profile.model_folder = str(self.root)
        profile.save()
        saved = Profile.from_json(profile.path)
        self.assertEqual(saved.model_files, value["parts"])
        self.assertEqual(saved.model_folder, str(self.root))

    def test_stale_confirmation_creates_no_profiles(self):
        self.candidate()
        self.bridge._import_selected_models("old-scan", ["one"])
        self.assertEqual(self.bridge.profiles, {})
        self.assertFalse(self.events[-1]["results"][0]["ok"])

    def test_unknown_candidate_id_cannot_import_a_path(self):
        self.candidate()
        self.bridge._import_selected_models("scan-a", [str(self.root / "one.gguf")])
        self.assertEqual(self.bridge.profiles, {})
        self.assertFalse(self.events[-1]["results"][0]["ok"])

    def test_multiple_selected_models_persist_separate_profiles(self):
        one, two = self.candidate(), self.candidate("two")
        def inspect(path):
            return next(v["metadata"] for v in (one, two) if v["path"] == str(path))
        with mock.patch.object(app, "inspect_model_files", side_effect=inspect), \
             mock.patch.object(app, "fetch_sys_stats", return_value=self.stats):
            self.bridge._import_selected_models("scan-a", ["one", "two"])
        self.assertEqual(len(self.bridge.profiles), 2)
        profiles = list(self.bridge.profiles.values())
        self.assertEqual({p.model for p in profiles}, {one["path"], two["path"]})
        self.assertEqual(len({p.port for p in profiles}), 2)
        for profile in profiles:
            self.assertEqual(Profile.from_json(profile.path).model_files, [profile.model])
        self.assertTrue(all(r["ok"] for r in self.events[-1]["results"]))

    def test_repeated_confirmation_does_not_duplicate_or_reset_settings(self):
        value = self.candidate()
        with mock.patch.object(app, "inspect_model_files", return_value=value["metadata"]), \
             mock.patch.object(app, "fetch_sys_stats", return_value=self.stats):
            self.bridge._import_selected_models("scan-a", ["one"])
            profile = next(iter(self.bridge.profiles.values()))
            profile.env["CACHE_RAM_MIB"] = "0"
            profile.env["SIPHON_TEMPERATURE"] = "0.25"
            profile.save()
            self.bridge._import_selected_models("scan-a", ["one"])
        self.assertEqual(len(self.bridge.profiles), 1)
        self.assertEqual(profile.env["CACHE_RAM_MIB"], "0")
        self.assertEqual(Profile.from_json(profile.path).env["SIPHON_TEMPERATURE"], "0.25")

    def test_import_does_not_reuse_a_port_when_initial_range_is_full(self):
        value = self.candidate()
        for port in range(8090, 8191):
            profile = Profile(name=f"old-{port}", env={"PORT": str(port)})
            self.bridge.profiles[profile.name] = profile
        with mock.patch.object(app, "inspect_model_files", return_value=value["metadata"]), \
             mock.patch.object(app, "fetch_sys_stats", return_value=self.stats):
            self.bridge._import_selected_models("scan-a", ["one"])
        ports = [p.port for p in self.bridge.profiles.values()]
        self.assertEqual(len(ports), len(set(ports)))
        self.assertTrue(self.events[-1]["results"][0]["ok"])

    def test_cancelled_scan_cannot_publish_or_import(self):
        value = self.candidate()
        self.bridge._cancel_model_import("scan-a")
        self.bridge._finish_folder_scan("scan-a", str(self.root), [value])
        self.assertFalse(any(e.get("phase") == "ready" for e in self.events))
        self.bridge._import_selected_models("scan-a", ["one"])
        self.assertEqual(self.bridge.profiles, {})

    def test_scan_publishes_ready_candidates_with_file_snapshot(self):
        value = self.candidate()
        with mock.patch.object(app, "scan_model_folder", return_value=[value]):
            self.bridge._scan_model_folder(self.root, "scan-a")
        self.assertEqual([e["phase"] for e in self.events], ["scanning", "ready"])
        self.assertEqual(value["fingerprint"], app.model_fingerprint([Path(value["path"])]))

    def test_native_chooser_selects_a_folder_and_scans_off_ui_thread(self):
        dialog = mock.Mock()
        dialog.select_folder_finish.return_value.get_path.return_value = str(self.root)
        with mock.patch.object(app.Gtk.FileDialog, "new", return_value=dialog):
            self.bridge._choose_model_folder("native-scan")
        self.assertEqual(dialog.select_folder.call_args.args[0], self.bridge.window)
        callback = dialog.select_folder.call_args.args[2]
        with mock.patch.object(app.threading, "Thread") as thread:
            callback(dialog, mock.sentinel.result, None)
        self.assertEqual(thread.call_args.kwargs["target"], self.bridge._scan_model_folder)
        self.assertEqual(thread.call_args.kwargs["args"], (self.root, "native-scan"))
        thread.return_value.start.assert_called_once()

    def test_nonlocal_native_selection_reports_error_not_permanent_scanning(self):
        dialog = mock.Mock()
        dialog.select_folder_finish.return_value.get_path.return_value = None
        with mock.patch.object(app.Gtk.FileDialog, "new", return_value=dialog):
            self.bridge._choose_model_folder("native-scan")
        dialog.select_folder.call_args.args[2](dialog, mock.sentinel.result, None)
        self.assertEqual(self.events[-1]["phase"], "error")

    def test_native_dismissal_cancels_current_scan(self):
        dialog = mock.Mock()
        dialog.select_folder_finish.side_effect = app.GLib.Error("Dismissed by user")
        with mock.patch.object(app.Gtk.FileDialog, "new", return_value=dialog):
            self.bridge._choose_model_folder("native-scan")
        dialog.select_folder.call_args.args[2](dialog, mock.sentinel.result, None)
        self.assertEqual(self.events[-1]["phase"], "cancelled")
        self.assertEqual(self.bridge._import_scan_id, "")

    def test_partial_failure_does_not_hide_successful_neighbor(self):
        one, two = self.candidate(), self.candidate("two")
        def inspect(path):
            if str(path) == two["path"]:
                raise ValueError("Missing part two")
            return one["metadata"]
        with mock.patch.object(app, "inspect_model_files", side_effect=inspect), \
             mock.patch.object(app, "fetch_sys_stats", return_value=self.stats):
            self.bridge._import_selected_models("scan-a", ["two", "one"])
        self.assertEqual(len(self.bridge.profiles), 1)
        self.assertEqual([r["ok"] for r in self.events[-1]["results"]], [False, True])

    def test_nonfirst_part_change_invalidates_metadata_cache(self):
        value = self.candidate()
        second = self.root / "second.gguf"
        second.write_bytes(b"second")
        profile = Profile(name="parts", env={"MODEL": value["path"]})
        profile.model_files = value["parts"] + [str(second)]
        payload = {"path": profile.model, **value["metadata"], "parts": profile.model_files}
        app._write_metadata_cache(payload)
        self.assertIsNotNone(app._cached_model_info(profile, "test"))
        second.write_bytes(b"changed second")
        self.assertIsNone(app._cached_model_info(profile, "test"))

    def test_legacy_weight_cache_is_reinspected_without_changing_profile(self):
        value = self.candidate()
        profile = Profile(name="legacy", env={"MODEL": value["path"], "CACHE_RAM_MIB": "0"})
        profile.save()
        before = profile.path.read_bytes()
        app._write_metadata_cache({"path": profile.model, **value["metadata"]})
        cache_path = app._metadata_cache_path()
        cache = json.loads(cache_path.read_text())
        cache[str(Path(profile.model).resolve())]["version"] = 2
        cache_path.write_text(json.dumps(cache))

        self.assertIsNone(app._cached_model_info(profile, "test"))
        self.assertEqual(profile.path.read_bytes(), before)

    def test_weight_breakdown_survives_metadata_cache_round_trip(self):
        value = self.candidate()
        profile = Profile(name="breakdown", env={"MODEL": value["path"]})
        payload = {"path": profile.model, **value["metadata"],
                   "dense_gib": 5.0, "core_gib": 1.0, "lookup_gib": 4.0,
                   "expert_gib": 2.0, "tensor_gib": 7.0}
        app._write_metadata_cache(payload)

        info = app._cached_model_info(profile, "test")
        self.assertIsNotNone(info)
        self.assertEqual(info.get("core_gib"), 1.0)
        self.assertEqual(info.get("lookup_gib"), 4.0)
        self.assertEqual(info.get("tensor_gib"), 7.0)
        self.assertEqual(info["dense_gib"], 5.0)

    def test_missing_nonfirst_part_marks_catalogue_unavailable(self):
        value = self.candidate()
        profile = Profile(name="parts", env={"MODEL": value["path"]})
        profile.model_files = value["parts"] + [str(self.root / "missing.gguf")]
        self.bridge.profiles[profile.name] = profile
        self.bridge.profile_by_model_key["test"] = profile.name
        self.bridge._schedule_catalogue()
        self.assertFalse(self.events[-1]["models"][0]["available"])

    def test_changed_file_after_preview_requires_rescan(self):
        value = self.candidate()
        value["fingerprint"] = app.model_fingerprint([Path(value["path"])])
        Path(value["path"]).write_bytes(b"replacement different model")
        with mock.patch.object(app, "inspect_model_files", return_value=value["metadata"]), \
             mock.patch.object(app, "fetch_sys_stats", return_value=self.stats):
            self.bridge._import_selected_models("scan-a", ["one"])
        self.assertEqual(self.bridge.profiles, {})
        self.assertFalse(self.events[-1]["results"][0]["ok"])

    def test_cache_rejects_scan_if_files_changed_during_inspection(self):
        value = self.candidate()
        payload = {"path": value["path"], **value["metadata"],
                   "fingerprint": app.model_fingerprint([Path(value["path"])])}
        Path(value["path"]).write_bytes(b"replacement after reading")
        app._write_metadata_cache(payload)
        profile = Profile(name="changed", env={"MODEL": value["path"]})
        self.assertIsNone(app._cached_model_info(profile, "test"))

    def test_real_split_scan_import_reload_cache_and_missing_part_launch_guard(self):
        parts = write_model(self.root / "models", "split", part_count=5)
        self.bridge._scan_model_folder(parts[0].parent, "scan-a")
        candidate = self.events[-1]["models"][0]
        self.assertTrue(candidate["available"], candidate["error"])
        with mock.patch.object(app, "fetch_sys_stats", return_value=self.stats):
            self.bridge._import_selected_models("scan-a", [candidate["id"]])
        self.assertTrue(self.events[-1]["results"][0]["ok"])
        profile = Profile.from_json(next(iter(self.bridge.profiles.values())).path)
        self.assertEqual(profile.model, str(parts[0]))
        self.assertEqual(profile.model_files, [str(part) for part in parts])
        self.assertEqual(profile.model_folder, str(parts[0].parent))
        info = app._cached_model_info(profile, "test")
        self.assertIsNotNone(info)
        self.assertEqual(info["part_count"], 5)
        self.assertAlmostEqual(info["artifact_gib"], sum(p.stat().st_size for p in parts) / 1024 ** 3)
        self.assertAlmostEqual(info["expert_gib"], 5 * 32 / 1024 ** 3)
        self.assertEqual(profile.validate(), [])
        parts[-1].unlink()
        self.assertFalse(app._profile_model_available(profile))
        self.assertIsNone(app._cached_model_info(profile, "test"))
        self.assertTrue(any("Missing part 5 of 5" in issue for issue in profile.validate()))

    def test_library_counts_logical_models_and_each_physical_part_once(self):
        parts = write_model(self.root / "models", "split", part_count=8)
        single = write_model(self.root / "models", "single")
        for name, files in (("split", parts), ("duplicate", parts), ("single", single)):
            profile = Profile(name=name, env={"MODEL": str(files[0])},
                              model_files=[str(p) for p in files])
            self.bridge.profiles[name] = profile
            self.bridge.profile_by_model_key[name] = name
        # Exercise the production metadata worker deterministically, not the
        # import harness's no-op scheduling callback.
        def immediate_thread(*, target, **_kwargs):
            return SimpleNamespace(start=target)
        with mock.patch.object(app.threading, "Thread", side_effect=immediate_thread), \
             mock.patch.object(app.GLib, "idle_add", side_effect=lambda fn, *args: fn(*args)), \
             mock.patch.object(app, "fetch_sys_stats", return_value=self.stats):
            app.DesktopBridge._schedule_model_info(self.bridge, "split")
        info = self.events[-1]
        self.assertEqual(info["library_count"], 2)
        self.assertEqual(info["part_count"], 8)
        self.assertEqual(info["library_storage_gib"],
                         round(sum(p.stat().st_size for p in parts + single) / 1024 ** 3, 2))

    def test_unchanged_imported_metadata_reopen_avoids_tensor_inspection(self):
        parts = write_model(self.root / "models", "split", part_count=2)
        self.bridge._scan_model_folder(parts[0].parent, "scan-a")
        with mock.patch.object(app, "fetch_sys_stats", return_value=self.stats):
            self.bridge._import_selected_models("scan-a", list(self.bridge._import_candidates))
        profile = Profile.from_json(next(iter(self.bridge.profiles.values())).path)
        self.bridge.profiles[profile.name] = profile
        key = next(iter(self.bridge.profile_by_model_key))
        with mock.patch.object(app, "inspect_model_files", side_effect=AssertionError("Unexpected rescan")), \
             mock.patch.object(app.threading, "Thread", side_effect=lambda target, **_: SimpleNamespace(start=target)), \
             mock.patch.object(app.GLib, "idle_add", side_effect=lambda fn, *args: fn(*args)), \
             mock.patch.object(app, "fetch_sys_stats", return_value=self.stats):
            app.DesktopBridge._schedule_model_info(self.bridge, key)
        self.assertTrue(self.events[-1]["available"])
        self.assertEqual(self.events[-1]["part_count"], 2)


if __name__ == "__main__":
    unittest.main()
