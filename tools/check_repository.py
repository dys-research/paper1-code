#!/usr/bin/env python3
"""Offline source checks. Findings identify locations without echoing secrets."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED_ANYWHERE = {".git", "__pycache__", ".venv"}
IGNORED_ROOTS = {"work", "data", "models", "outputs", "dist"}
SUFFIXES = {".py", ".md", ".toml", ".txt", ".json", ".yml", ".yaml", ".example"}


def main() -> int:
    failures = []
    count = 0
    # Patterns are assembled to avoid flagging the scanner's own pattern source.
    windows_drive = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]" + r":[/\\]")
    private_unix = re.compile("/" + r"(?:Users|home|workspace|mnt|root)/[A-Za-z0-9_.-]+/")
    credential = re.compile(r"\b(?:sk|ghp|github_pat)" + r"[-_][A-Za-z0-9_\-]{20,}\b")
    key_assignment = re.compile(r"(?i)(?:api_key|access_token|secret_key)\s*=\s*['\"]([^'\"]{16,})['\"]")
    for path in sorted(ROOT.rglob("*")):
        parts = path.relative_to(ROOT).parts
        if not path.is_file() or any(part in IGNORED_ANYWHERE for part in parts) or parts[0] in IGNORED_ROOTS:
            continue
        if path.name in {"config.toml", ".env"} or path.suffix not in SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        count += 1
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=str(path.relative_to(ROOT)))
            except SyntaxError as exc:
                failures.append(f"{path.relative_to(ROOT)}:{exc.lineno}: syntax error")
        for number, line in enumerate(text.splitlines(), 1):
            for pattern, label in [(windows_drive, "absolute Windows path"), (private_unix, "personal filesystem path"), (credential, "possible credential"), (key_assignment, "literal credential assignment")]:
                if pattern.search(line):
                    failures.append(f"{path.relative_to(ROOT)}:{number}: {label}")
    plan = json.loads((ROOT / "pipeline.json").read_text(encoding="utf-8"))
    for stage, scripts in plan.items():
        if not scripts:
            failures.append(f"pipeline.json: empty stage {stage}")
        for name in scripts:
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                failures.append(f"pipeline.json: missing or unsafe step {name}")
    if failures:
        print("Repository checks failed:")
        print("\n".join(failures))
        return 1
    print(f"Passed: {count} source/config/document files; workflow paths exist; no detected personal paths or literal credentials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
