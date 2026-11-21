import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import project_config as config


class ConfigTests(unittest.TestCase):
    def tearDown(self):
        config.get_config.cache_clear()

    def test_relative_paths_follow_config_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            cfg = root / "config.toml"
            cfg.write_text('[inputs]\npatentsview_root="source"\n[paths]\nwork_dir="results"\n', encoding="utf-8")
            with patch.dict(os.environ, {"TFSC_CONFIG": str(cfg)}):
                config.get_config.cache_clear()
                self.assertEqual(config.input_path("patentsview_root"), root / "source")
                self.assertEqual(config.work_dir("analysis"), root / "results" / "analysis")
                self.assertIsNone(config.input_path("panel", required=False))
                with self.assertRaises(ValueError):
                    config.input_path("panel")

    def test_stage_cannot_escape_work_directory(self):
        with self.assertRaises(ValueError):
            config.work_dir("../outside")

    def test_source_directory_cannot_be_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / "config.toml"
            unsafe = (config.REPO_ROOT / "scripts" / "scratch").as_posix()
            cfg.write_text(f'[paths]\nwork_dir="{unsafe}"\n', encoding="utf-8")
            with patch.dict(os.environ, {"TFSC_CONFIG": str(cfg)}):
                config.get_config.cache_clear()
                with self.assertRaises(ValueError):
                    config.work_dir()


if __name__ == "__main__":
    unittest.main()
