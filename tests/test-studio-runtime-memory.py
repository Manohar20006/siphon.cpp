"""Engine allocation records must be current, validated, and configuration-specific."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "siphon-studio"))


class RuntimeMemoryTests(unittest.TestCase):
    def test_parser_ignores_old_and_invalid_records(self):
        from siphon.core.runtime_memory import read_memory_record
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "server.log"
            record = {"version": 1, "gpu_name": "GPU", "device_count": 1,
                      "model_bytes": 100, "context_bytes": 200, "compute_bytes": 300}
            old = "SIPHON_GPU_MEMORY " + json.dumps(record) + "\n"
            path.write_text(old + 'SIPHON_GPU_MEMORY {"version":1}\n')
            self.assertIsNone(read_memory_record(path, len(old.encode())))
            with path.open("a") as stream:
                stream.write("SIPHON_GPU_MEMORY " + json.dumps(record) + "\n")
            self.assertEqual(read_memory_record(path, len(old.encode())), record)
            record["compute_bytes"] = -1
            path.write_text("SIPHON_GPU_MEMORY " + json.dumps(record) + "\n")
            self.assertIsNone(read_memory_record(path, 0))

    def test_observation_cache_is_invalidated_by_engine_model_or_launch_changes(self):
        from siphon.core.runtime_memory import save_observation, load_observation
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.json"
            identity = {"model": [{"mtime_ns": 123}], "runtime": [456], "env": {"CTX": "65536"}}
            record = {"gpu_name": "GPU", "model_bytes": 1, "context_bytes": 2, "compute_bytes": 3}
            save_observation(path, "model", identity, record)
            self.assertEqual(load_observation(path, "model", identity), record)
            for field in identity:
                with self.subTest(field=field):
                    self.assertIsNone(load_observation(path, "model", {**identity, field: "changed"}))
            path.write_text("broken json")
            self.assertIsNone(load_observation(path, "model", identity))


if __name__ == "__main__":
    unittest.main()
