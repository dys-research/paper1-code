from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(repo_root: Path) -> dict[str, Any]:
    config_path = Path(os.environ.get("PME_CONFIG", repo_root / "config.yaml"))
    if not config_path.exists():
        raise FileNotFoundError(
            "Configuration file not found. Copy config.example.yaml to config.yaml "
            "and set paths.patentsview_dir before running a script."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")
    config["_config_dir"] = config_path.parent.resolve()
    return config


def config_path(config: dict[str, Any], key: str) -> Path:
    paths = config.get("paths", {})
    value = paths.get(key)
    if not value or str(value).startswith("REPLACE_WITH_"):
        raise ValueError(f"Set paths.{key} in config.yaml before running this script.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()
