#!/usr/bin/env python3
"""Run documented workflow stages without changing the shell's directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def pipeline() -> dict:
    return json.loads((ROOT / "pipeline.json").read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["list", "check", *pipeline()])
    parser.add_argument("--step", help="Run one named script from the selected stage")
    parser.add_argument("--from-step", help="Start with this script, including it")
    parser.add_argument("--until-step", help="Stop after this script")
    parser.add_argument("--config", type=Path, help="Private TOML configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without reading data or making API calls")
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    if extra and not args.step:
        parser.error("Extra script arguments require --step")
    if args.step and (args.from_step or args.until_step):
        parser.error("Use either --step or a --from-step/--until-step range")
    stages = pipeline()
    if args.stage == "list":
        for stage, scripts in stages.items():
            print(stage)
            for script in scripts:
                print(f"  {Path(script).name}")
        return 0
    if args.stage == "check":
        if args.step or args.from_step or args.until_step or extra:
            parser.error("check does not accept workflow-step arguments")
        selected = ["tools/check_repository.py"]
    else:
        selected = stages[args.stage]
        names = [Path(script).stem for script in selected]

        def index(value):
            normalized = Path(value).stem
            if normalized not in names:
                parser.error(f"Unknown step {value!r} for {args.stage}; choose from {', '.join(names)}")
            return names.index(normalized)

        if args.step:
            selected = [selected[index(args.step)]]
        else:
            lo = index(args.from_step) if args.from_step else 0
            hi = index(args.until_step) + 1 if args.until_step else len(selected)
            if lo >= hi:
                parser.error("The requested step range is reversed")
            selected = selected[lo:hi]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    if args.config:
        cfg = args.config.expanduser().resolve()
        if not cfg.is_file():
            parser.error("--config must identify an existing TOML file")
        env["TFSC_CONFIG"] = str(cfg)
    for relative in selected:
        script = ROOT / relative
        if not script.is_file():
            parser.error(f"Workflow entry does not exist: {relative}")
        print(f"{'Would run' if args.dry_run else 'Running'} {relative}", flush=True)
        if not args.dry_run:
            result = subprocess.run([sys.executable, str(script), *extra], cwd=ROOT, env=env, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
