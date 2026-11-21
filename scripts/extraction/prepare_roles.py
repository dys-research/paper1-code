"""Export primary role spans with literal source verification and missingness."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from project_config import input_path, work_dir
from core import ROLES, input_hash, iter_jsonl, iter_text_rows, sources_for, span_check


def get_elements(record):
    if "elements" in record:
        return record["elements"]
    if all(isinstance(record.get(role), dict) for role in ROLES):
        return {role: record[role] for role in ROLES}
    # Existing flat extraction outputs are accepted as an import format.
    return {
        role: {
            field: record.get(f"{role}_{field}")
            for field in ("present", "source_field", "evidence_span", "normalized_text", "confidence", "absence_reason")
        }
        for role in ROLES
    }


def convert(row, prediction, allow_missing_hash=False):
    sources = sources_for(row)
    current_hash = input_hash(sources)
    complete = all(sources.values())
    extracted_hash = prediction.get("input_sources_sha256") if prediction else None
    hash_matches = extracted_hash == current_hash
    hash_acceptable = hash_matches or (not extracted_hash and allow_missing_hash)
    elements = get_elements(prediction) if prediction else {}
    records = []
    for role in ROLES:
        element = elements.get(role) or {}
        valid, start, status = span_check(element, sources)
        if not complete:
            valid, status = False, "INCOMPLETE_THREE_FIELD_INPUT"
        elif prediction is None:
            valid, status = False, "NO_OUTPUT"
        elif not hash_acceptable:
            valid, status = False, "INPUT_CHANGED" if extracted_hash else "INPUT_HASH_MISSING"
        span = element.get("evidence_span")
        confidence = element.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence = None
        records.append({
            "patent_id": str(row["patent_id"]),
            "grant_year": int(row["grant_year"]),
            "is_focal_candidate": bool(row["is_focal_candidate"]),
            "full_input_available": complete,
            "role": role,
            "present": element.get("present") if isinstance(element.get("present"), bool) else None,
            "source_field": element.get("source_field"),
            "evidence_span": span if isinstance(span, str) else None,
            "normalized_text": element.get("normalized_text"),
            "confidence": confidence,
            "absence_reason": element.get("absence_reason"),
            "embedding_text": span if valid else None,
            "usable_for_primary_embedding": valid,
            "exact_locatable": start is not None,
            "role_status": status,
            "span_start": start,
            "span_end": start + len(span) if start is not None else None,
            "span_words": len(span.split()) if isinstance(span, str) and span else None,
            "all_input_text_matches": hash_matches,
            "input_hash_verified": hash_matches,
            "current_input_sources_sha256": current_hash,
            "input_sources_sha256": extracted_hash,
            "extraction_model": prediction.get("model", prediction.get("extraction_model")) if prediction else None,
            "extraction_origin": "imported_parsed_output" if prediction else None,
        })
    return records


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-base", type=Path)
    parser.add_argument("--parsed", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-missing-input-hash", action="store_true", help="Explicitly import legacy outputs lacking full-input fingerprints")
    args = parser.parse_args()
    text_base = args.text_base or work_dir("data") / "data/a61b_text_base_1995_2020.parquet"
    parsed = args.parsed or input_path("role_outputs", required=False) or work_dir("extraction") / "parsed.jsonl"
    destination = args.output or work_dir("extraction") / "role_inputs_primary.parquet"
    predictions = {}
    for record in iter_jsonl(parsed):
        patent_id = str(record["patent_id"])
        if patent_id in predictions:
            raise ValueError(f"Duplicate patent output {patent_id}; merge batches explicitly before exporting")
        predictions[patent_id] = record
    schema = pa.schema([
        ("patent_id", pa.string()), ("grant_year", pa.int32()),
        ("is_focal_candidate", pa.bool_()), ("full_input_available", pa.bool_()),
        ("role", pa.string()), ("present", pa.bool_()), ("source_field", pa.string()),
        ("evidence_span", pa.string()), ("normalized_text", pa.string()),
        ("confidence", pa.float64()), ("absence_reason", pa.string()),
        ("embedding_text", pa.string()), ("usable_for_primary_embedding", pa.bool_()),
        ("exact_locatable", pa.bool_()), ("role_status", pa.string()),
        ("span_start", pa.int64()), ("span_end", pa.int64()), ("span_words", pa.int64()),
        ("all_input_text_matches", pa.bool_()), ("input_hash_verified", pa.bool_()),
        ("current_input_sources_sha256", pa.string()), ("input_sources_sha256", pa.string()),
        ("extraction_model", pa.string()), ("extraction_origin", pa.string()),
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    counts = Counter()
    records, seen = [], set()
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for row in iter_text_rows(text_base):
            patent_id = str(row["patent_id"])
            if patent_id in seen:
                raise ValueError(f"Duplicate text-base patent {patent_id}")
            seen.add(patent_id)
            converted = convert(row, predictions.get(patent_id), args.allow_missing_input_hash)
            counts.update(record["role_status"] for record in converted)
            records.extend(converted)
            if len(records) >= 6000:
                writer.write_table(pa.Table.from_pylist(records, schema=schema))
                records.clear()
        if records:
            writer.write_table(pa.Table.from_pylist(records, schema=schema))
    temporary.replace(destination)
    summary = {
        "patents": len(seen), "roles": sum(counts.values()), "role_status": dict(counts),
        "allow_missing_input_hash": args.allow_missing_input_hash,
        "outputs_outside_text_base": len(set(predictions) - seen),
        "primary_embedding": "source-valid evidence_span only",
    }
    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
