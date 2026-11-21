"""Portable configuration shared by command-line workflows.

Relative input and output paths are resolved against the configuration file's
directory. No user-specific paths or credentials are stored in source code.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
import tomllib

REPO_ROOT = Path(__file__).resolve().parent


def config_file() -> Path:
    value = os.environ.get("TFSC_CONFIG", "config.toml")
    path = Path(value).expanduser()
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


@lru_cache(maxsize=1)
def get_config() -> dict:
    path = config_file()
    if not path.is_file():
        return {}
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    for section, values in data.items():
        if not isinstance(values, dict):
            raise ValueError(f"Configuration section {section!r} must be a table")
    return data


def setting(section: str, key: str, default=None):
    return get_config().get(section, {}).get(key, default)


def resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (config_file().parent / path).resolve() if not path.is_absolute() else path.resolve()


def input_path(key: str, required: bool = True) -> Path | None:
    value = setting("inputs", key)
    if value is None or not str(value).strip():
        if required:
            raise ValueError(f"Set inputs.{key} in config.toml before running this step")
        return None
    path = resolve_path(value)
    if required and not path.exists():
        raise FileNotFoundError(f"Configured input inputs.{key} does not exist: {path}")
    return path


def work_dir(stage: str | None = None) -> Path:
    base = resolve_path(setting("paths", "work_dir", "work"))
    if stage is not None:
        if not stage or Path(stage).name != stage or stage in {".", ".."}:
            raise ValueError("Stage must be one directory name")
        base = base / stage
    # Source and input files are never implicitly used as output destinations.
    if base == REPO_ROOT or REPO_ROOT.is_relative_to(base):
        raise ValueError("The work directory must not be the repository root or its parent")
    if base.is_relative_to(REPO_ROOT / "scripts"):
        raise ValueError("The work directory must not contain the source directory")
    base.mkdir(parents=True, exist_ok=True)
    return base
