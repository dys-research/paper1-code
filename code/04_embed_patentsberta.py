from __future__ import annotations

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
