from __future__ import annotations

import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
PARSED_PME = WORK / "llm_outputs" / "pme_outputs_parsed.jsonl"
LLM_INPUT = WORK / "sample" / "a61b_llm_input.jsonl"
OUT_DIR = WORK / "embedding_input"
EMBEDDING_INPUT = OUT_DIR / "a61b_embedding_texts.jsonl"
EMBED_SCRIPT = ROOT / "code" / "04_embed_patentsberta.py"
MANIFEST = OUT_DIR / "autodl_embedding_manifest.json"
ZIP_PATH = OUT_DIR / "embedding_package.zip"


EMBED_SCRIPT_TEXT = r'''from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


OUTPUT_NAMES = {
    "P": "a61b_patentsberta_P",
    "M": "a61b_patentsberta_M",
    "E": "a61b_patentsberta_E",
    "PM_config": "a61b_patentsberta_PM_config",
    "ME_config": "a61b_patentsberta_ME_config",
    "overall_ta": "a61b_patentsberta_overall_ta",
    "overall_tac": "a61b_patentsberta_overall_tac",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def embed_one_type(
    model: SentenceTransformer,
    input_path: Path,
    output_dir: Path,
    embedding_type: str,
    batch_size: int,
    model_name: str,
) -> dict:
    rows = [
        row
        for row in read_jsonl(input_path)
        if row.get("embedding_type") == embedding_type and row.get("embedding_text")
    ]
    if not rows:
        raise RuntimeError(f"No eligible rows for embedding_type={embedding_type}")

    texts = [str(row["embedding_text"]) for row in rows]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    stem = OUTPUT_NAMES[embedding_type]
    np.save(output_dir / f"{stem}.npy", vectors)
    metadata = []
    for index, row in enumerate(rows):
        metadata.append(
            {
                "row_index": index,
                "embedding_type": embedding_type,
                "patent_id": row.get("patent_id"),
                "request_id": row.get("request_id"),
                "grant_year": row.get("grant_year"),
                "grant_date": row.get("grant_date"),
                "fields": row.get("fields"),
                "sample_roles": row.get("sample_roles"),
                "source_field": row.get("source_field"),
                "confidence": row.get("confidence"),
                "span_valid": row.get("span_valid"),
                "text_chars": len(texts[index]),
                "source_input": str(input_path),
                "embedding_model": model_name,
            }
        )
    write_jsonl(output_dir / f"{stem}_metadata.jsonl", metadata)
    return {"embedding_type": embedding_type, "rows": len(rows), "shape": list(vectors.shape), "stem": stem}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--embedding-type",
        choices=["all", *sorted(OUTPUT_NAMES)],
        required=True,
        help="Use all to run P/M/E/PM_config/ME_config/overall_ta/overall_tac in one pass.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="AI-Growth-Lab/PatentSBERTa")
    parser.add_argument("--device", default=None, help="Default: cuda if available, otherwise cpu.")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_types = list(OUTPUT_NAMES) if args.embedding_type == "all" else [args.embedding_type]
    model = SentenceTransformer(args.model, device=device)
    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "embedding_model": args.model,
        "device": device,
        "batch_size": args.batch_size,
        "results": [],
    }
    for embedding_type in selected_types:
        print(f"\n=== Embedding {embedding_type} ===", flush=True)
        result = embed_one_type(
            model=model,
            input_path=args.input,
            output_dir=args.output_dir,
            embedding_type=embedding_type,
            batch_size=args.batch_size,
            model_name=args.model,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        summary["results"].append(result)

    summary_path = args.output_dir / "a61b_patentsberta_embedding_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def clean_text(value) -> str:
    return " ".join(str(value or "").split())


def make_embedding_rows():
    source_by_request = {row["request_id"]: row for row in read_jsonl(LLM_INPUT)}
    seen: set[str] = set()

    for row in read_jsonl(PARSED_PME):
        request_id = row.get("request_id")
        if request_id in seen:
            continue
        seen.add(request_id)
        patent_id = row.get("patent_id")
        source = source_by_request.get(request_id, {})
        input_sources = source.get("input_sources", {}) or {}
        base = {
            "request_id": request_id,
            "patent_id": patent_id,
            "grant_year": row.get("grant_year"),
            "grant_date": row.get("grant_date"),
            "fields": row.get("fields"),
            "sample_roles": row.get("sample_roles"),
            "llm_model": row.get("model"),
        }

        p_text = clean_text(row.get("P_text")) if row.get("P_present") else ""
        m_text = clean_text(row.get("M_text")) if row.get("M_present") else ""
        e_text = clean_text(row.get("E_text")) if row.get("E_present") else ""

        for element, text in [("P", p_text), ("M", m_text), ("E", e_text)]:
            if text:
                yield {
                    **base,
                    "embedding_type": element,
                    "embedding_text": text,
                    "source_field": row.get(f"{element}_source_field"),
                    "confidence": row.get(f"{element}_confidence"),
                    "span_valid": row.get(f"{element}_span_valid"),
                }

        if p_text and m_text:
            yield {
                **base,
                "embedding_type": "PM_config",
                "embedding_text": f"Problem: {p_text}\nMechanism: {m_text}",
                "source_field": "P_text+M_text",
                "confidence": min(float(row.get("P_confidence") or 0), float(row.get("M_confidence") or 0)),
                "span_valid": bool(row.get("P_span_valid")) and bool(row.get("M_span_valid")),
            }

        if m_text and e_text:
            yield {
                **base,
                "embedding_type": "ME_config",
                "embedding_text": f"Mechanism: {m_text}\nEffect: {e_text}",
                "source_field": "M_text+E_text",
                "confidence": min(float(row.get("M_confidence") or 0), float(row.get("E_confidence") or 0)),
                "span_valid": bool(row.get("M_span_valid")) and bool(row.get("E_span_valid")),
            }

        title = clean_text(input_sources.get("title"))
        abstract = clean_text(input_sources.get("abstract"))
        claim_1 = clean_text(input_sources.get("claim_1"))
        if title or abstract:
            yield {
                **base,
                "embedding_type": "overall_ta",
                "embedding_text": f"Title: {title}\nAbstract: {abstract}",
                "source_field": "title+abstract",
                "confidence": None,
                "span_valid": None,
            }
        if title or abstract or claim_1:
            yield {
                **base,
                "embedding_type": "overall_tac",
                "embedding_text": f"Title: {title}\nAbstract: {abstract}\nClaim 1: {claim_1}",
                "source_field": "title+abstract+claim_1",
                "confidence": None,
                "span_valid": None,
            }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_SCRIPT.write_text(EMBED_SCRIPT_TEXT, encoding="utf-8")

    counts: Counter[str] = Counter()
    rows_written = 0
    with EMBEDDING_INPUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in make_embedding_rows():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[row["embedding_type"]] += 1
            rows_written += 1

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "step": "patentsberta_embedding",
        "source_pme": str(PARSED_PME),
        "source_llm_input": str(LLM_INPUT),
        "embedding_method_consistent_with_v4": True,
        "embedding_model": "AI-Growth-Lab/PatentSBERTa",
        "normalize_embeddings": True,
        "embedding_types": ["P", "M", "E", "PM_config", "ME_config", "overall_ta", "overall_tac"],
        "upload_files": [EMBEDDING_INPUT.name, EMBED_SCRIPT.name, MANIFEST.name],
        "embedding_input": str(EMBEDDING_INPUT),
        "embed_script": str(EMBED_SCRIPT),
        "zip_path": str(ZIP_PATH),
        "rows_written": rows_written,
        "embedding_type_counts": dict(sorted(counts.items())),
        "recommended_autodl_command": (
            "python 05_embed_v6_a61b_patentsberta.py "
            "--input a61b_embedding_texts.jsonl "
            "--embedding-type all "
            "--output-dir outputs "
            "--batch-size 128"
        ),
        "expected_return_files": [
            "outputs/a61b_patentsberta_P.npy",
            "outputs/a61b_patentsberta_P_metadata.jsonl",
            "outputs/a61b_patentsberta_M.npy",
            "outputs/a61b_patentsberta_M_metadata.jsonl",
            "outputs/a61b_patentsberta_E.npy",
            "outputs/a61b_patentsberta_E_metadata.jsonl",
            "outputs/a61b_patentsberta_PM_config.npy",
            "outputs/a61b_patentsberta_PM_config_metadata.jsonl",
            "outputs/a61b_patentsberta_ME_config.npy",
            "outputs/a61b_patentsberta_ME_config_metadata.jsonl",
            "outputs/a61b_patentsberta_overall_ta.npy",
            "outputs/a61b_patentsberta_overall_ta_metadata.jsonl",
            "outputs/a61b_patentsberta_overall_tac.npy",
            "outputs/a61b_patentsberta_overall_tac_metadata.jsonl",
            "outputs/a61b_patentsberta_embedding_summary.json",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
        for path in [EMBEDDING_INPUT, EMBED_SCRIPT, MANIFEST]:
            zf.write(path, arcname=path.name)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
