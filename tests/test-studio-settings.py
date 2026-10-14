"""Regression tests for desktop settings saved against a temporary profile."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "siphon-studio"))

from siphon.core.profiles import Profile
from siphon.desktop import app


class DesktopSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.studio_dir = Path(self.temp.name) / "studio"
        self.env = mock.patch.dict(os.environ, {
            "SIPHON_STUDIO_DIR": str(self.studio_dir),
            "SIPHON_REPO_ROOT": str(REPO_ROOT),
        })
        self.env.start()
        self.addCleanup(self.env.stop)

        self.profile = Profile(
            name="settings-regression",
            env={
                "MODEL": "/tmp/test-model.gguf",
                "CTX": "4096",
                "MOE_GPU_CACHE_MIB": "512",
                "MOE_PINNED_CACHE_MIB": "4096",
                "CACHE_RAM_MIB": "1024",
                "GDS_READ_THREADS": "4",
                "SPEC_TYPE": "draft-mtp",
            },
            script="scripts/run-qwen36-moe.sh",
            source="app",
        )
        self.profile.save()
        self.profile.materialize_env()
        self.profile_path = self.profile.path
        self.env_path = self.studio_dir / "state" / "env" / "settings-regression.env"

        self.events = []
        self.bridge = app.DesktopBridge.__new__(app.DesktopBridge)
        self.bridge.profiles = {self.profile.name: self.profile}
        self.bridge.profile_by_model_key = {"test": self.profile.name}
        self.bridge._sampling = {}
        self.bridge._emit_async = self.events.append
        self.bridge._schedule_model_info = lambda _model_key: None

    def settings(self, **updates):
        values = {
            "vram_cache": 512,
            "pinned_cache": 4096,
            "cache_ram": 1024,
            "gds_threads": 4,
            "draft_model": "Native MTP / NextN",
            "temperature": 0.7,
        }
        values.update(updates)
        return values

    def system(self, *, ram_free=1.0, vram_free=128, ram_total=16, vram_total=8192):
        return SimpleNamespace(
            mem_free_gb=ram_free,
            mem_total_gb=ram_total,
            gpu_mib_free=vram_free,
            gpu_mib_total=vram_total,
        )

    def apply_with(self, values, *, running, system=None):
        status = SimpleNamespace(running=running)
        with mock.patch.object(app.runner, "status", return_value=status) as status_mock, \
             mock.patch.object(app, "fetch_sys_stats", return_value=system or self.system()):
            self.bridge._apply_settings("test", values)
        status_mock.assert_called_once_with(self.profile, include_metrics=False)
        return self.events[-1]

    def persisted_bytes(self):
        return self.profile_path.read_bytes(), self.env_path.read_bytes()

    def test_unchanged_running_allocations_can_save_nonmemory_settings_at_low_headroom(self):
        result = self.apply_with(
            {"temperature": 0.25}, running=True,
            system=self.system(ram_free=1, vram_free=128),
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(self.profile.env["MOE_GPU_CACHE_MIB"], "512")
        self.assertEqual(self.profile.env["MOE_PINNED_CACHE_MIB"], "4096")
        self.assertEqual(self.profile.env["CACHE_RAM_MIB"], "1024")
        self.assertEqual(self.profile.env["SIPHON_TEMPERATURE"], "0.25")

    def test_offline_partial_nonmemory_save_validates_existing_cache_allocations(self):
        before = self.persisted_bytes()
        result = self.apply_with(
            {"temperature": 0.25}, running=False,
            system=self.system(ram_free=3, vram_free=8192),
        )

        self.assertFalse(result["ok"])
        self.assertIn("2 GiB of free RAM", result["message"])
        self.assertEqual(self.persisted_bytes(), before)

    def test_running_memory_change_is_rejected_before_profile_or_env_persist(self):
        before = self.persisted_bytes()
        result = self.apply_with(
            self.settings(cache_ram=1536), running=True,
            system=self.system(ram_free=64, vram_free=16384),
        )

        self.assertFalse(result["ok"])
        self.assertIn("Stop the model before changing memory", result["message"])
        self.assertEqual(self.persisted_bytes(), before)
        self.assertEqual(self.profile.env["CACHE_RAM_MIB"], "1024")

    def test_running_malformed_memory_is_rejected_before_persist(self):
        before = self.persisted_bytes()
        result = self.apply_with(self.settings(pinned_cache="not-a-number"), running=True)

        self.assertFalse(result["ok"])
        self.assertIn("Memory allocations must be valid numbers", result["message"])
        self.assertEqual(self.persisted_bytes(), before)

    def test_running_unchanged_memory_still_validates_gds_workers(self):
        result = self.apply_with(self.settings(gds_threads=9999), running=True)

        self.assertFalse(result["ok"])
        self.assertIn("GDS workers", result["message"])

    def test_offline_profile_still_enforces_ram_headroom(self):
        before = self.persisted_bytes()
        result = self.apply_with(
            self.settings(), running=False,
            system=self.system(ram_free=3, vram_free=8192),
        )

        self.assertFalse(result["ok"])
        self.assertIn("2 GiB of free RAM", result["message"])
        self.assertEqual(self.persisted_bytes(), before)

    def test_offline_profile_still_enforces_vram_reserve(self):
        before = self.persisted_bytes()
        result = self.apply_with(
            self.settings(), running=False,
            system=self.system(ram_free=16, vram_free=512),
        )

        self.assertFalse(result["ok"])
        self.assertIn("VRAM cache", result["message"])
        self.assertEqual(self.persisted_bytes(), before)

    def test_model_reservations_limit_expert_pool_before_persist(self):
        layout = {
            "layer_weights_bytes": [2048 * 1024 * 1024],
            "output_weights_bytes": 0,
            "kv_bytes_by_type": {"tq3_0": [4480]},
            "draft_kv_bytes_by_type": {"f16": [0]},
            "recurrent_bytes_by_layer": [0], "compute_reserve_mib": 512,
        }
        before = self.persisted_bytes()
        with mock.patch.object(app, "_cached_model_info", return_value={"memory_layout": layout}):
            result = self.apply_with(
                self.settings(vram_cache=2048), running=False,
                system=self.system(ram_free=16, vram_free=4096),
            )
        self.assertFalse(result["ok"], result)
        self.assertIn("model", result["message"].lower())
        self.assertEqual(self.persisted_bytes(), before)

    def test_another_running_profile_does_not_exempt_selected_offline_profile(self):
        other = Profile(name="other-model", env={"MODEL": "/tmp/other.gguf"})
        self.bridge.profiles[other.name] = other
        status = mock.Mock(side_effect=lambda profile, include_metrics=False: SimpleNamespace(
            running=profile.name == other.name
        ))
        before = self.persisted_bytes()
        with mock.patch.object(app.runner, "status", status), \
             mock.patch.object(app, "fetch_sys_stats", return_value=self.system(ram_free=3)):
            self.bridge._apply_settings("test", self.settings())

        status.assert_called_once_with(self.profile, include_metrics=False)
        self.assertFalse(self.events[-1]["ok"])
        self.assertIn("2 GiB of free RAM", self.events[-1]["message"])
        self.assertEqual(self.persisted_bytes(), before)

    def test_missing_artifact_settings_round_trip_keeps_exact_saved_cache(self):
        result = self.apply_with(
            self.settings(pinned_cache=5120, cache_ram=0), running=False,
            system=self.system(ram_free=16, vram_free=8192),
        )
        self.assertTrue(result["ok"], result)
        saved = Profile(**json.loads(self.profile_path.read_text()))
        self.assertEqual(app._profile_config(saved)["l2_cache"], 5120)
        self.assertEqual(app._profile_config(saved)["cache_ram"], 0)
        self.assertIn("MOE_PINNED_CACHE_MIB=5120", self.env_path.read_text())

    def test_catalogue_reports_actual_artifact_availability(self):
        present_model = Path(self.temp.name) / "present.gguf"
        present_model.touch()
        self.bridge.profiles["present"] = Profile(name="present", env={"MODEL": str(present_model)})
        self.bridge.profile_by_model_key["present"] = "present"
        self.bridge._schedule_catalogue()
        availability = {entry["model_key"]: entry["available"] for entry in self.events[-1]["models"]}
        self.assertEqual(availability, {"test": False, "present": True})

    def test_cpu_config_does_not_inherit_gpu_cache_or_mtp_defaults(self):
        self.profile.env["SIPHON_EXECUTION_MODE"] = "cpu"
        config = app._profile_config(self.profile)
        self.assertEqual(config["execution_mode"], "cpu")
        self.assertEqual(config["cache_type"], "q8_0")
        self.assertEqual(config["spec_type"], "none")
        self.assertEqual(config["gpu_layers"], 0)
        self.assertEqual(config["l1_cache"], 0)
        self.assertEqual(config["l2_cache"], 0)

    def test_cpu_saves_separate_settings_and_ignores_gpu_memory_ceiling(self):
        self.profile.env["MODEL_DRAFT"] = "/tmp/preserved-gpu-draft.gguf"
        result = self.apply_with({"execution_mode": "cpu", "context": 4096,
            "cache_type": "q8_0", "cache_ram": 512, "cpu_threads": 2,
            "spec_type": "none", "draft_model": "Disabled"}, running=False,
            system=self.system(ram_free=16, vram_total=0, vram_free=0))
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.profile.env["CPU_CTX"], "4096")
        self.assertEqual(self.profile.env["CPU_THREADS"], "2")
        self.assertEqual(self.profile.env["MOE_GPU_CACHE_MIB"], "512")
        self.assertEqual(self.profile.env["MOE_PINNED_CACHE_MIB"], "4096")
        self.assertEqual(self.profile.env["CACHE_RAM_MIB"], "1024")
        self.assertEqual(self.profile.env["SPEC_TYPE"], "draft-mtp")
        self.assertEqual(self.profile.env["MODEL_DRAFT"], "/tmp/preserved-gpu-draft.gguf")
        saved = Profile(**json.loads(self.profile_path.read_text()))
        self.assertEqual(app._profile_config(saved)["cache_ram"], 512)

    def test_mode_switch_while_running_is_rejected_without_persisting(self):
        before = self.persisted_bytes()
        result = self.apply_with({"execution_mode": "cpu"}, running=True,
            system=self.system(ram_free=64, vram_free=16384))
        self.assertFalse(result["ok"], result)
        self.assertIn("Stop the model", result["message"])
        self.assertEqual(before, self.persisted_bytes())

    def test_cpu_ram_guard_includes_live_kv_and_not_unused_pinned_cache(self):
        layout = {"kv_bytes_by_type": {"q8_0": [16 * 1024]},
                  "draft_kv_bytes_by_type": {"f16": [0]},
                  "recurrent_bytes_by_layer": [0], "compute_reserve_mib": 512}
        config = {"execution_mode": "cpu", "context": 65536,
                  "cache_type": "q8_0", "cache_ram": 4096,
                  "pinned_cache": 99999, "vram_cache": 99999, "cpu_threads": 1}
        ok, message = app._validate_resource_settings(config,
            self.system(ram_free=6, vram_free=0, vram_total=0), memory_layout=layout)
        self.assertFalse(ok)
        self.assertIn("RAM", message)
        config["cache_ram"] = 512
        ok, message = app._validate_resource_settings(config,
            self.system(ram_free=6, vram_free=0, vram_total=0), memory_layout=layout)
        self.assertTrue(ok, message)


class DesktopStartupTests(unittest.TestCase):
    system = DesktopSettingsTests.system

    def setUp(self):
        DesktopSettingsTests.setUp(self)
        self.bridge.model_key = "test"
        self.bridge._closed = False
        self.bridge._poll_paused = False
        self.bridge._poll_busy = False
        self.bridge._telemetry_epoch = 0
        self.bridge._starting_models = set()
        self.bridge._request_metrics = {}
        self.bridge._active_generations = {}
        self.bridge._dstorage_baselines = {}
        self.bridge._live_timing_samples = {}
        self.bridge._live_stream_metrics = {}
        self.bridge._dstorage_rate_samples = {}
        self.bridge._log_offsets = {self.profile.name: 0}
        self.bridge._emit_event = self.events.append
        self.profile.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.system_stats = self.system(ram_free=16, vram_free=8192)
        self.system_stats.__dict__.update(
            cpu=6, cpu_temp=50, mem_used_gb=0, gpu_util=0,
            gpu_temp=45, gpu_mib_used=0,
        )
        self.status = SimpleNamespace(
            running=True, health=SimpleNamespace(status="ok"),
            metrics=SimpleNamespace(n_processing=0, n_prompt=0),
        )
        # Synchronous scheduling exercises the real bridge without launching
        # servers, touching user profiles, or requiring a GTK main loop.
        self.thread_patch = mock.patch.object(app.threading, "Thread", side_effect=lambda **kw:
            SimpleNamespace(start=kw["target"]))
        self.thread_patch.start()
        self.addCleanup(self.thread_patch.stop)
        for patch in (
            mock.patch.object(app.GLib, "idle_add", side_effect=lambda fn, *args: fn(*args)),
            mock.patch.object(app.runner, "status", return_value=self.status),
            mock.patch.object(app, "fetch_sys_stats", return_value=self.system_stats),
            mock.patch.object(app, "fetch_metrics", return_value=self.status.metrics),
            mock.patch.object(app, "fetch_slots", return_value=SimpleNamespace(busy=0, total_past=0)),
            mock.patch.object(app, "fetch_dstorage_metrics", return_value={}),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def test_startup_probe_is_not_a_user_request_and_real_request_still_counts(self):
        startup_payloads = []

        def probe(_profile):
            self.profile.log_file.write_text(
                "prompt eval time = 1000 ms / 8 tokens (125 ms per token, 8 tokens per second)\n"
                "eval time = 0.001 ms / 1 tokens (0.001 ms per token, 1000000 tokens per second)\n"
            )
            self.status.metrics.n_processing = 1
            self.bridge._poll()
            startup_payloads.append(self.events[-1])
            self.status.metrics.n_processing = 0
            return None

        with mock.patch.object(app.runner, "up", return_value=(True, "started")), \
             mock.patch.object(app.runner, "wait_healthy", return_value=True), \
             mock.patch.object(app, "_probe_inference", side_effect=probe):
            self.bridge._run_model_action(True, "test")

        self.assertEqual(startup_payloads[0]["request_state"], "idle")
        self.assertIsNone(startup_payloads[0]["decode_tps"])
        self.assertIsNone(startup_payloads[0]["prefill_tps"])
        self.bridge._poll()
        self.assertEqual(self.events[-1]["request_state"], "idle")
        self.assertIsNone(self.events[-1]["decode_tps"])
        self.assertIsNone(self.events[-1]["prefill_tps"])
        with self.profile.log_file.open("a") as log:
            log.write("eval time = 1000 ms / 20 tokens (50 ms per token, 20 tokens per second)\n")
        self.bridge._poll()
        self.assertEqual(self.events[-1]["request_state"], "completed")
        self.assertEqual(self.events[-1]["decode_tps"], 20)

    def test_startup_completion_invalidates_inflight_probe_poll(self):
        captured_epochs = []
        def probe(_profile):
            captured_epochs.append(self.bridge._telemetry_epoch)
            return None
        with mock.patch.object(app.runner, "up", return_value=(True, "started")), \
             mock.patch.object(app.runner, "wait_healthy", return_value=True), \
             mock.patch.object(app, "_probe_inference", side_effect=probe):
            self.bridge._run_model_action(True, "test")
        before = len(self.events)
        self.bridge._finish_poll({"type": "telemetry", "decode_tps": 1000000}, captured_epochs[0])
        self.assertEqual(len(self.events), before)

    def test_failed_resource_validation_releases_startup_guard(self):
        self.system_stats.mem_free_gb = 1
        with mock.patch.object(app.runner, "up") as launch:
            self.bridge._run_model_action(True, "test")
        launch.assert_not_called()
        self.assertFalse(self.events[-1]["ok"])
        self.assertNotIn(self.profile.name, self.bridge._starting_models)

    def test_launch_exception_releases_startup_guard_and_reports_failure(self):
        with mock.patch.object(app.runner, "up", side_effect=OSError("launch failed")), \
             mock.patch.object(app.runner, "down") as shutdown:
            self.bridge._run_model_action(True, "test")
        shutdown.assert_not_called()
        self.assertFalse(self.events[-1]["ok"])
        self.assertIn("launch failed", self.events[-1]["message"])
        self.assertNotIn(self.profile.name, self.bridge._starting_models)

    def test_probe_exception_cleans_up_launched_model_and_excludes_probe_log(self):
        def broken_probe(_profile):
            self.profile.log_file.write_text(
                "eval time = 1000 ms / 20 tokens (50 ms per token, 20 tokens per second)\n"
            )
            raise OSError("probe failed")
        with mock.patch.object(app.runner, "up", return_value=(True, "started")), \
             mock.patch.object(app.runner, "wait_healthy", return_value=True), \
             mock.patch.object(app, "_probe_inference", side_effect=broken_probe), \
             mock.patch.object(app.runner, "down", return_value=(True, "stopped")) as shutdown:
            self.bridge._run_model_action(True, "test")
        shutdown.assert_called_once_with(self.profile, force=True)
        self.assertFalse(self.events[-1]["ok"])
        self.assertNotIn(self.profile.name, self.bridge._starting_models)
        self.bridge._poll()
        self.assertEqual(self.events[-1]["request_state"], "idle")
        self.assertIsNone(self.events[-1]["decode_tps"])

    def test_rejected_readiness_probe_log_is_excluded_even_if_shutdown_fails(self):
        def rejected_probe(_profile):
            self.profile.log_file.write_text(
                "eval time = 0.001 ms / 1 tokens (0.001 ms per token, 1000000 tokens per second)\n"
            )
            return "probe rejected"
        with mock.patch.object(app.runner, "up", return_value=(True, "started")), \
             mock.patch.object(app.runner, "wait_healthy", return_value=True), \
             mock.patch.object(app, "_probe_inference", side_effect=rejected_probe), \
             mock.patch.object(app.runner, "down", return_value=(False, "still running")):
            self.bridge._run_model_action(True, "test")
        self.assertFalse(self.events[-1]["ok"])
        self.bridge._poll()
        self.assertEqual(self.events[-1]["request_state"], "idle")
        self.assertIsNone(self.events[-1]["decode_tps"])


if __name__ == "__main__":
    unittest.main()
