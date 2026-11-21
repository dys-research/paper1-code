"""Prepare text-only extraction inputs without making network requests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_config import work_dir
from core import iter_text_rows, make_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-base", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 prepares all eligible patents")
    parser.add_argument("--focal-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    text_base = args.text_base or work_dir("data") / "data/a61b_text_base_1995_2020.parquet"
    destination = args.output or work_dir("extraction") / "input.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.overwrite:
        parser.error("Output exists; choose a new --output or pass --overwrite")
    count = skipped = 0
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in iter_text_rows(text_base):
            if args.focal_only and not row["is_focal_candidate"]:
                continue
            record = make_input(row)
            if not all(record["input_sources"].values()):
                skipped += 1
                continue
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= args.limit:
                break
    temporary.replace(destination)
    print(json.dumps({"prepared": count, "incomplete_inputs_skipped": skipped, "network_requests": 0}))


if __name__ == "__main__":
    main()
