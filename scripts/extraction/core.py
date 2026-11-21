"""Shared, source-grounded P/M/E input and output utilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROLES = ("P", "M", "E")
FIELDS = ("title", "abstract", "claim_1")


def canonical(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split())


def sources_for(row: dict) -> dict[str, str]:
    return {
        "title": canonical(row.get("patent_title")),
        "abstract": canonical(row.get("patent_abstract")),
        "claim_1": canonical(row.get("independent_claim_text")),
    }


def input_hash(sources: dict[str, str]) -> str:
    data = json.dumps(sources, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def make_input(row: dict) -> dict:
    sources = sources_for(row)
    fingerprint = input_hash(sources)
    patent_id = str(row["patent_id"])
    return {
        "request_id": f"pme_{patent_id}_{fingerprint[:12]}",
        "patent_id": patent_id,
        "grant_year": int(row["grant_year"]),
        "grant_date": str(row["grant_date"]),
        "fields": ["medical_device_a61b"],
        "input_sources": sources,
        "input_sources_sha256": fingerprint,
    }


def user_text(record: dict) -> str:
    """Preserve the original extraction's text-only user-message structure."""
    sources = record["input_sources"]
    return (
        f"PATENT_ID: {record['patent_id']}\n"
        f"FIELDS: {', '.join(record.get('fields', []))}\n"
        f"GRANT_YEAR: {record.get('grant_year')}\n\n"
        f"TITLE:\n{sources['title']}\n\n"
        f"ABSTRACT:\n{sources['abstract']}\n\n"
        f"CLAIM_1:\n{sources['claim_1']}\n\n"
        "Return JSON with exactly keys P, M, E."
    )


def parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != set(ROLES):
        raise ValueError("Response must have exactly P, M and E objects")
    for name in ROLES:
        element = value[name]
        if not isinstance(element, dict) or not isinstance(element.get("present"), bool):
            raise ValueError(f"{name}.present must be boolean")
        confidence = element.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ValueError(f"{name}.confidence must be in [0, 1]")
    return value


def span_check(element: dict, sources: dict[str, str]) -> tuple[bool, int | None, str]:
    if element.get("present") is False:
        return False, None, "MODEL_ABSENT"
    if element.get("present") is not True:
        return False, None, "INVALID_PRESENT"
    field, span = element.get("source_field"), element.get("evidence_span")
    if field not in FIELDS:
        return False, None, "INVALID_SOURCE"
    if not isinstance(span, str) or not span:
        return False, None, "INVALID_OR_NONVERBATIM_SPAN"
    position = sources[field].find(span)
    if position < 0:
        return False, None, "INVALID_OR_NONVERBATIM_SPAN"
    return True, position, "VALID_ORIGINAL_SPAN"


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at {path.name}:{number}")
                yield value


def iter_inputs(path: Path):
    seen = set()
    for record in iter_jsonl(path):
        for name in ("request_id", "patent_id", "input_sources_sha256"):
            if not isinstance(record.get(name), str) or not record[name]:
                raise ValueError(f"Input requires a nonempty string {name}")
        if record["request_id"] in seen:
            raise ValueError("Duplicate request_id in input batch")
        seen.add(record["request_id"])
        sources = record.get("input_sources")
        if not isinstance(sources, dict) or set(sources) != set(FIELDS):
            raise ValueError("Input requires exactly title, abstract, claim_1 source fields")
        if not all(isinstance(value, str) and value for value in sources.values()):
            raise ValueError("All three source fields must be nonempty strings")
        if input_hash(sources) != record["input_sources_sha256"]:
            raise ValueError("Input source fingerprint does not match its text")
        yield record


def iter_text_rows(path: Path, batch_size: int = 2000):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()
