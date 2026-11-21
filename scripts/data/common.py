"""Shared paths and file helpers for normalized PatentsView preparation."""
from pathlib import Path
import json
import duckdb

from project_config import input_path, setting, work_dir

ROOT = work_dir("data")
NEW = input_path("patentsview_root")


def output(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Output path escapes the data work directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def qp(path):
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def source(relative):
    path = NEW / relative
    if not path.exists():
        raise FileNotFoundError(f"Required normalized PatentsView input is missing: {path}")
    return path


def connect():
    connection = duckdb.connect(":memory:")
    connection.execute("SET memory_limit=" + qp(setting("compute", "duckdb_memory", "4GB")))
    connection.execute("SET threads=" + str(int(setting("compute", "threads", 4))))
    connection.execute("SET temp_directory=" + qp(output("tmp/duckdb")))
    return connection


def parquet(connection, query, relative):
    connection.execute("COPY (" + query + ") TO " + qp(output(relative)) + " (FORMAT PARQUET, COMPRESSION ZSTD)")


def save(relative, data):
    output(relative).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def receipt(relative, paths, **details):
    save(relative, {"inputs": [{"name": Path(path).name, "bytes": Path(path).stat().st_size} for path in paths], **details})
