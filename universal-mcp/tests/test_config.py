import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from universal_mcp.config import load_config


class ConfigTests(unittest.TestCase):
    def test_environment_overrides_root_and_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            configured = base / "configured"
            overridden = base / "overridden"
            configured.mkdir()
            overridden.mkdir()
            config_path = base / "workers.json"
            config_path.write_text(json.dumps({
                "project_root": str(configured),
                "storage_path": str(base / "configured.sqlite3"),
                "allowed_command_prefixes": [["python3"]],
                "workers": [],
            }), encoding="utf-8")
            with patch.dict(os.environ, {
                "UNIVERSAL_MCP_PROJECT_ROOT": str(overridden),
                "UNIVERSAL_MCP_STORAGE_PATH": str(base / "override.sqlite3"),
            }):
                config = load_config(config_path)
            self.assertEqual(config.project_root, overridden.resolve())
            self.assertEqual(config.storage_path, (base / "override.sqlite3").resolve())
            self.assertEqual(config.allowed_command_prefixes, (("python3",),))


if __name__ == "__main__":
    unittest.main()
