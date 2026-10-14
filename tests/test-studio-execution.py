"""CPU/GPU execution selection and launch contract tests."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "siphon-studio"))

from siphon import paths
from siphon.core import execution, runner
from siphon.core.profiles import Profile, preview_command


class StudioExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        self.studio = Path(self.temp.name) / "studio"
        self.env_patch = mock.patch.dict(os.environ, {
            "SIPHON_REPO_ROOT": str(self.root),
            "SIPHON_STUDIO_DIR": str(self.studio),
            "SIPHON_EXECUTION_MODE": "auto",
        }, clear=False)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        execution.gpu_available.cache_clear()
        self.addCleanup(execution.gpu_available.cache_clear)

    def model(self, name="model.gguf"):
        path = self.root / "models" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def cpu_server(self, root=None):
        path = (root or self.root) / "build-cpu" / "bin" / "llama-server"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        path.chmod(0o755)
        return path

    def profile(self, **values):
        script = values.pop("script", "missing-launcher.sh")
        env = {"MODEL": str(self.model()), "PORT": "8090", "CTX": "131072"}
        env.update(values)
        return Profile(name="execution-test", env=env, script=script)

    def test_gpu_probe_is_bounded_and_cached(self):
        result = SimpleNamespace(returncode=0, stdout="NVIDIA GPU\n", stderr="")
        with mock.patch.object(execution.subprocess, "run", return_value=result) as run:
            self.assertTrue(execution.gpu_available())
            self.assertTrue(execution.gpu_available())

        run.assert_called_once()
        # bounded, but long enough for a laptop GPU waking from runtime suspend (>2 s observed)
        self.assertEqual(run.call_args.kwargs["timeout"], 10)

    def test_auto_mode_defaults_to_cpu_when_nvidia_probe_fails(self):
        profile = self.profile()
        with mock.patch.object(execution, "gpu_available", return_value=False):
            self.assertEqual(execution.mode(profile), "cpu")

    def test_auto_mode_selects_gpu_when_nvidia_probe_succeeds(self):
        profile = self.profile()
        with mock.patch.object(execution, "gpu_available", return_value=True):
            self.assertEqual(execution.mode(profile), "gpu")

    def test_execution_mode_rejects_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "SIPHON_EXECUTION_MODE"):
            execution.mode(self.profile(SIPHON_EXECUTION_MODE="accelerated"))

    def test_explicit_cpu_server_override_fails_instead_of_falling_back(self):
        missing = Path(self.temp.name) / "missing-server"
        with mock.patch.dict(os.environ, {"SIPHON_CPU_SERVER_BIN": str(missing)}):
            with self.assertRaisesRegex(FileNotFoundError, "SIPHON_CPU_SERVER_BIN"):
                paths.cpu_server_bin()

    def test_cpu_binary_search_honors_cpu_root_before_repo_build(self):
        self.cpu_server()
        elsewhere = self.cpu_server(Path(self.temp.name) / "other")

        with mock.patch.dict(os.environ, {"SIPHON_CPU_ROOT": str(elsewhere.parents[2])}):
            self.assertEqual(paths.cpu_server_bin(), elsewhere)

    def test_effective_cpu_settings_are_separate_and_conservative(self):
        profile = self.profile(
            SIPHON_EXECUTION_MODE="cpu",
            CTX="131072", BATCH="2048", UBATCH="256", MOE_GPU_CACHE_MIB="900",
            CPU_BATCH="256", CPU_UBATCH="32", CPU_CACHE_TYPE="f16",
            CPU_CACHE_RAM_MIB="512", CPU_THREADS="2",
        )

        effective = execution.effective_env(profile)

        self.assertEqual(effective["CTX"], "65536")
        self.assertEqual(effective["BATCH"], "256")
        self.assertEqual(effective["UBATCH"], "32")
        self.assertEqual(effective["CACHE_TYPE"], "f16")
        self.assertEqual(effective["CACHE_RAM_MIB"], "512")
        self.assertEqual(effective["CPU_THREADS"], "2")
        self.assertEqual(effective["THREADS"], "2")
        self.assertEqual(effective["GPU_LAYERS"], "0")
        self.assertEqual(effective["MOE_GPU_CACHE_MIB"], "0")
        self.assertEqual(effective["MOE_PINNED_CACHE_MIB"], "0")
        self.assertEqual(effective["LLAMA_DSTORAGE_EVICT_TO_RAM"], "0")
        self.assertEqual(profile.env["CTX"], "131072")
        self.assertEqual(profile.env["MOE_GPU_CACHE_MIB"], "900")

    def test_effective_gpu_settings_are_returned_without_cpu_projection(self):
        profile = self.profile(
            SIPHON_EXECUTION_MODE="gpu", CTX="4096", THREADS="11",
            CPU_CTX="8192", CPU_THREADS="2", MODEL_DRAFT="gpu-draft.gguf",
        )

        self.assertEqual(execution.effective_env(profile), profile.env)

    def test_default_cpu_threads_is_capped_by_logical_cpu_count(self):
        with mock.patch.object(execution.os, "cpu_count", return_value=12):
            self.assertEqual(execution.default_cpu_threads(), 8)
        with mock.patch.object(execution.os, "cpu_count", return_value=4):
            self.assertEqual(execution.default_cpu_threads(), 4)

    def test_cpu_launch_uses_first_gguf_shard_and_cpu_flags(self):
        model = self.model("split-model-00002-of-00003.gguf")
        model.with_name("split-model-00001-of-00003.gguf").touch()
        model.with_name("split-model-00003-of-00003.gguf").touch()
        binary = self.cpu_server()
        profile = self.profile(SIPHON_EXECUTION_MODE="cpu", MODEL=str(model),
                               CPU_CTX="4096", CPU_BATCH="256",
                               CPU_UBATCH="32", CPU_THREADS="3")

        command, env = execution.cpu_launch(profile)

        self.assertEqual(command[0], str(binary))
        self.assertEqual(command[command.index("-m") + 1],
                         str(model.with_name("split-model-00001-of-00003.gguf")))
        self.assertEqual(command[command.index("-ngl") + 1], "0")
        self.assertEqual(command[command.index("-t") + 1], "3")
        self.assertEqual(command[command.index("-ctk") + 1], "q8_0")
        self.assertIn("--no-warmup", command)
        self.assertIn("--no-cont-batching", command)
        self.assertEqual(env["LLAMA_ARG_REPACK"], "0")
        self.assertEqual(env["LLAMA_ARG_FIT"], "off")
        self.assertEqual(env["LLAMA_MMAP_PREFETCH"], "0")

    def test_cpu_environment_removes_gpu_leaks_but_keeps_native_routing(self):
        self.cpu_server()
        profile = self.profile(
            SIPHON_EXECUTION_MODE="cpu",
            MOE_EXPERT_COUNT_MODE="native",
            LLAMA_MOE_EXPERT_USED_OVERRIDE="4",
            LLAMA_ARG_CTX_CHECKPOINTS="12",
        )
        inherited = {
            "CUDA_VISIBLE_DEVICES": "0",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "GGML_CUDA_DISABLE_GRAPHS": "1",
            "LLAMA_DSTORAGE_EVICT_TO_RAM": "1",
            "LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE": "/tmp/gpu-suite.txt",
            "LLAMA_GDS_READ_THREADS": "16",
            "LLAMA_ARG_N_GPU_LAYERS": "99",
            "LLAMA_ARG_FIT": "on",
            "LLAMA_ARG_CTX_CHECKPOINTS": "99",
            "LLAMA_ARG_CPU_MOE": "1",
            "GGML_CPU_BATCHED_EXPERTS": "1",
        }
        with mock.patch.dict(os.environ, inherited, clear=False):
            command, env = execution.cpu_launch(profile)

        self.assertEqual(env["LLAMA_MOE_EXPERT_USED_OVERRIDE"], "4")
        self.assertEqual(env["MOE_EXPERT_COUNT_MODE"], "native")
        for key in inherited:
            if not key.startswith("LLAMA_ARG_"):
                self.assertNotIn(key, env)
        self.assertNotIn("LLAMA_ARG_N_GPU_LAYERS", env)
        self.assertNotIn("LLAMA_ARG_CPU_MOE", env)
        self.assertEqual(env["LLAMA_ARG_FIT"], "off")
        self.assertEqual(env["LLAMA_ARG_CTX_CHECKPOINTS"], "12")
        self.assertEqual(command[command.index("-ngl") + 1], "0")

    def test_cpu_continuous_batching_flag_is_only_disabled_for_single_parallel_slot(self):
        self.cpu_server()
        single, _ = execution.cpu_launch(self.profile(
            SIPHON_EXECUTION_MODE="cpu", N_PARALLEL="1",
        ))
        multiple, _ = execution.cpu_launch(self.profile(
            SIPHON_EXECUTION_MODE="cpu", N_PARALLEL="2",
        ))

        self.assertIn("--no-warmup", single)
        self.assertIn("--no-cont-batching", single)
        self.assertIn("--no-warmup", multiple)
        self.assertNotIn("--no-cont-batching", multiple)

    def test_cpu_speculation_is_off_by_default_but_native_mtp_can_be_enabled(self):
        self.cpu_server()
        default_command, _ = execution.cpu_launch(self.profile(SIPHON_EXECUTION_MODE="cpu"))
        mtp_command, _ = execution.cpu_launch(self.profile(
            SIPHON_EXECUTION_MODE="cpu", CPU_SPEC_TYPE="draft-mtp",
        ))

        self.assertEqual(default_command[default_command.index("--spec-type") + 1], "none")
        self.assertEqual(default_command[default_command.index("-c") + 1], "65536")
        self.assertEqual(default_command[default_command.index("-b") + 1], "512")
        self.assertEqual(default_command[default_command.index("-ub") + 1], "64")
        self.assertEqual(default_command[default_command.index("--cache-ram") + 1], "0")
        self.assertEqual(default_command[default_command.index("--spec-draft-n-max") + 1], "3")
        self.assertEqual(default_command[default_command.index("--spec-draft-p-min") + 1], "0")
        self.assertEqual(mtp_command[mtp_command.index("--spec-type") + 1], "draft-mtp")

    def test_cpu_speculation_can_use_an_external_draft_model(self):
        self.cpu_server()
        gpu_draft = self.model("gpu-draft.gguf")
        cpu_draft = self.model("cpu-draft.gguf")
        command, _ = execution.cpu_launch(self.profile(
            SIPHON_EXECUTION_MODE="cpu", CPU_SPEC_TYPE="draft-simple",
            MODEL_DRAFT=str(gpu_draft), CPU_MODEL_DRAFT=str(cpu_draft),
        ))

        self.assertEqual(command[command.index("--model-draft") + 1], str(cpu_draft))

    def test_cpu_cache_types_are_restricted_to_verified_options(self):
        profile = self.profile(SIPHON_EXECUTION_MODE="cpu", CPU_CACHE_TYPE="tq3_0")

        issues = profile.validate()

        self.assertTrue(any("CPU_CACHE_TYPE" in issue for issue in issues), issues)

    def test_cpu_profile_validation_ignores_gpu_only_launcher_experts_and_budget(self):
        profile = self.profile(
            SIPHON_EXECUTION_MODE="cpu", EXPERTS="missing-suite.txt",
            MOE_GPU_CACHE_MIB="999999", MOE_PINNED_CACHE_MIB="999999",
        )
        self.assertEqual(profile.validate(), [])

    def test_cpu_profile_validation_reports_bad_cpu_types_and_cache(self):
        profile = self.profile(
            SIPHON_EXECUTION_MODE="cpu", CPU_CTX="wide", CPU_CACHE_TYPE="mystery",
        )

        issues = profile.validate()

        self.assertTrue(any("CPU_CTX" in issue for issue in issues), issues)
        self.assertTrue(any("CPU_CACHE_TYPE" in issue for issue in issues), issues)

    def test_cpu_settings_round_trip_without_replacing_gpu_profile_settings(self):
        profile = self.profile(
            SIPHON_EXECUTION_MODE="cpu", CTX="32768", MODEL_DRAFT="gpu-draft.gguf",
            CPU_CTX="8192", CPU_THREADS="4", CPU_MODEL_DRAFT="cpu-draft.gguf",
        )

        profile.save()
        loaded = Profile.from_json(profile.path)

        self.assertEqual(loaded.env["CTX"], "32768")
        self.assertEqual(loaded.env["MODEL_DRAFT"], "gpu-draft.gguf")
        self.assertEqual(loaded.env["CPU_CTX"], "8192")
        self.assertEqual(loaded.env["CPU_THREADS"], "4")
        self.assertEqual(loaded.env["CPU_MODEL_DRAFT"], "cpu-draft.gguf")

    def test_cpu_preview_contains_effective_env_and_command_without_materializing_file(self):
        self.cpu_server()
        profile = self.profile(
            SIPHON_EXECUTION_MODE="cpu", CPU_CTX="4096", GDS_READ_THREADS="16",
            LLAMA_ARG_FIT="on",
        )

        command = preview_command(profile)

        self.assertEqual(command[0], "env")
        self.assertIn("CTX=4096", command)
        self.assertIn("LLAMA_ARG_FIT=off", command)
        self.assertNotIn("GDS_READ_THREADS=16", command)
        self.assertIn("-ngl", command)
        self.assertFalse((self.studio / "state" / "env" / "execution-test.env").exists())

    def test_gpu_launch_path_is_unchanged(self):
        script = self.root / "scripts" / "run-qwen36-moe.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.touch()
        gpu_binary = self.root / "build-gds-cuda" / "bin" / "llama-server"
        gpu_binary.parent.mkdir(parents=True, exist_ok=True)
        gpu_binary.touch()
        gpu_binary.chmod(0o755)
        profile = self.profile(SIPHON_EXECUTION_MODE="gpu",
                               script="scripts/run-qwen36-moe.sh")
        captured = {}
        with mock.patch.object(runner, "status", return_value=runner.ServerStatus()), \
             mock.patch.object(runner, "running_profiles", return_value=[]), \
             mock.patch.object(runner, "driver_name", return_value="direct"), \
             mock.patch.object(execution, "gpu_available", return_value=True), \
             mock.patch.object(runner, "_direct_up", side_effect=lambda p, cmd, env: (
                 captured.update(profile=p, command=cmd, env=env) or (True, "started")
             )), \
             mock.patch.object(execution, "cpu_launch", side_effect=AssertionError("CPU path used")):
            ok, _message = runner.up(profile)

        self.assertTrue(ok)
        self.assertEqual(captured["command"][0], "bash")
        self.assertTrue(captured["env"]["CONFIG"].endswith("execution-test.env"))

    def test_explicit_gpu_validation_reports_missing_binary(self):
        profile = self.profile(SIPHON_EXECUTION_MODE="gpu")
        missing = Path(self.temp.name) / "missing-gpu-server"
        with mock.patch.dict(os.environ, {"SIPHON_SERVER_BIN": str(missing)}), \
             mock.patch.object(execution, "gpu_available", return_value=True):
            issues = profile.validate()

        self.assertTrue(any("GPU llama-server binary" in issue for issue in issues), issues)
        self.assertTrue(any(str(missing) in issue for issue in issues), issues)

    def test_systemd_cpu_launch_returns_an_unsupported_driver_error(self):
        self.cpu_server()
        profile = self.profile(SIPHON_EXECUTION_MODE="cpu")
        with mock.patch.object(runner, "status", return_value=runner.ServerStatus()), \
             mock.patch.object(runner, "running_profiles", return_value=[]), \
             mock.patch.object(runner, "driver_name", return_value="systemd"), \
             mock.patch.object(runner, "_systemd_up", side_effect=AssertionError("systemd used")):
            ok, message = runner.up(profile)

        self.assertFalse(ok)
        self.assertIn("systemd", message.lower())
        self.assertIn("CPU", message)

    def test_cpu_validation_stops_unsupported_cache_before_launch(self):
        profile = self.profile(SIPHON_EXECUTION_MODE="cpu", CPU_CACHE_TYPE="tq3_0")
        with mock.patch.object(runner, "status", return_value=runner.ServerStatus()), \
             mock.patch.object(runner, "running_profiles", return_value=[]), \
             mock.patch.object(runner, "driver_name", return_value="direct"), \
             mock.patch.object(execution, "cpu_launch", side_effect=AssertionError("launch used")):
            ok, message = runner.up(profile)

        self.assertFalse(ok)
        self.assertIn("CPU_CACHE_TYPE", message)

    def test_contention_message_is_neutral_about_cpu_and_gpu_resources(self):
        profile = self.profile(SIPHON_EXECUTION_MODE="gpu")
        with mock.patch.object(runner, "status", return_value=runner.ServerStatus()), \
             mock.patch.object(runner, "running_profiles", return_value=[Profile(name="other")]), \
             mock.patch.object(runner, "driver_name", return_value="direct"):
            ok, message = runner.up(profile)

        self.assertFalse(ok)
        self.assertIn("shared compute resources", message)
        self.assertNotIn("GPU", message)
        self.assertNotIn("6 GB", message)

    def test_auto_cpu_launch_reports_missing_binary_without_gpu_fallback(self):
        profile = self.profile()
        with mock.patch.object(execution, "gpu_available", return_value=False), \
             mock.patch.object(runner, "status", return_value=runner.ServerStatus()), \
             mock.patch.object(runner, "running_profiles", return_value=[]), \
             mock.patch.object(runner, "driver_name", return_value="direct"):
            ok, message = runner.up(profile)

        self.assertFalse(ok)
        self.assertIn("CPU", message)
        self.assertIn("llama-server", message)
        self.assertNotIn("CONFIG=", message)


if __name__ == "__main__":
    unittest.main()
