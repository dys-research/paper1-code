"""Shared paths and I/O for the measurement pipeline."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import time

# Support both the root runner and direct script execution.
REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from project_config import input_path, setting, work_dir
import duckdb
import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = work_dir('measurement')
STAGE1 = work_dir('data')
NEW = input_path('patentsview_root', required=False)
END = str(setting('study', 'observation_end', '2025-12-30'))


def out(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Measurement output escapes the configured work directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def qp(path):
    return "'" + str(path).replace('\\', '/').replace("'", "''") + "'"


def readp(path):
    return 'read_parquet(' + qp(path) + ')'


def dump(relative, value):
    out(relative).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def log(message):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(), message, flush=True)


def connect(persistent=False):
    connection = duckdb.connect(str(out('cache/measurement.duckdb')) if persistent else ':memory:')
    connection.execute('SET memory_limit=' + qp(setting('compute', 'duckdb_memory', '4GB')))
    connection.execute('SET threads=' + str(int(setting('compute', 'threads', 4))))
    connection.execute('SET temp_directory=' + qp(out('tmp/duckdb')))
    connection.execute('SET preserve_insertion_order=false')
    return connection


def parquet(connection, query, relative):
    connection.execute('COPY (' + query + ') TO ' + qp(out(relative)) + ' (FORMAT PARQUET, COMPRESSION ZSTD)')


def csv(connection, query, relative):
    connection.execute('COPY (' + query + ') TO ' + qp(out(relative)) + ' (FORMAT CSV, HEADER TRUE)')


def rows(connection, query):
    result = connection.execute(query)
    names = [column[0] for column in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def source_receipt(paths, name):
    result = []
    for path in dict.fromkeys(map(Path, paths)):
        stat = path.stat()
        record = {'path': str(path), 'size': stat.st_size, 'access': 'read_only'}
        if path.suffix == '.parquet':
            file = pq.ParquetFile(path)
            record.update(rows=file.metadata.num_rows, columns=file.schema_arrow.names)
        result.append(record)
    dump(name, result)
    return result


def finish(step, result):
    dump('checks/' + step + '.json', result)
    log(json.dumps(result, ensure_ascii=False, default=str))


def canonical(value):
    return '' if value is None else ' '.join(str(value).split())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
