#!/usr/bin/env python3
"""Build a source-only ZIP with relative member names and a SHA-256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_ROOTS = {"data", "work", "models", "outputs", "dist"}
EXCLUDED_ANYWHERE = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints"}
SOURCE_SUFFIXES = {".py", ".md", ".toml", ".txt", ".json", ".yml", ".yaml", ".example"}
SOURCE_NAMES = {"LICENSE", ".gitignore", ".gitattributes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/tfsc-pme-github.zip")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "tools/check_repository.py")], cwd=ROOT, check=True)
    paths = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        parts = path.relative_to(ROOT).parts
        if path == ROOT / "SOURCE_MANIFEST.json":
            continue  # Always generate one fresh manifest instead of packaging an older copy too.
        if parts[0] in EXCLUDED_ROOTS or any(part in EXCLUDED_ANYWHERE for part in parts):
            continue
        if path.name == "config.toml" or (path.name.startswith(".env") and path.name != ".env.example"):
            continue
        if path.suffix in SOURCE_SUFFIXES or path.name in SOURCE_NAMES:
            paths.append(path)
    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            archive.writestr("tfsc-pme-code/" + relative, content)
            manifest.append({"path": relative, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        archive.writestr("tfsc-pme-code/SOURCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Created {target.name}: {len(paths)} source files plus a relative-path integrity manifest")


if __name__ == "__main__":
    main()
