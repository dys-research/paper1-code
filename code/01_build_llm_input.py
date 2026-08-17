from __future__ import annotations

import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
sys.path.insert(0, str(REPO_ROOT))
from project_config import config_path, load_config

PATENTSVIEW = config_path(load_config(REPO_ROOT), "patentsview_dir")

WORK = ROOT / "work"
SAMPLE_DIR = WORK / "sample"
LOG_DIR = WORK / "logs"
REPORT_DIR = WORK / "reports"

UNIVERSE_CSV = SAMPLE_DIR / "a61b_patent_universe_2005_2019.csv"
INPUT_JSONL = SAMPLE_DIR / "a61b_llm_input.jsonl"
SAMPLE_JSONL = LOG_DIR / "a61b_llm_input_samples.jsonl"
QC_JSON = LOG_DIR / "a61b_llm_input_qc.json"
QC_MD = LOG_DIR / "a61b_llm_input_qc.md"


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).split())


def patent_id_text(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def load_a61b_ids() -> pd.DataFrame:
    years = pd.read_parquet(PATENTSVIEW / "patent_year.parquet", columns=["patent_id", "grant_year"])
    years = years[(years["grant_year"] >= 2005) & (years["grant_year"] <= 2019)].copy()
    years["patent_id"] = years["patent_id"].astype(str)

    pf = pq.ParquetFile(PATENTSVIEW / "cpc.parquet")
    rows: list[pd.DataFrame] = []
    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg, columns=["patent_id", "cpc_subclass", "cpc_group"]).to_pandas()
        chunk = chunk[chunk["cpc_subclass"].eq("A61B")]
        if not chunk.empty:
            chunk["patent_id"] = chunk["patent_id"].astype(str)
            rows.append(chunk.drop_duplicates(["patent_id", "cpc_group"]))
    cpc = pd.concat(rows, ignore_index=True)
    cpc_groups = (
        cpc.groupby("patent_id")["cpc_group"]
        .apply(lambda s: "|".join(sorted(set(map(str, s.dropna())))))
        .reset_index(name="a61b_cpc_groups")
    )
    out = years.merge(cpc_groups, on="patent_id", how="inner").drop_duplicates("patent_id")
    return out.sort_values(["grant_year", "patent_id"]).reset_index(drop=True)


def load_titles_abstracts(patents: pd.DataFrame) -> pd.DataFrame:
    title = pd.read_parquet(PATENTSVIEW / "7Text" / "patents_title.parquet", columns=["patent_id", "patent_title"])
    abstract = pd.read_parquet(PATENTSVIEW / "7Text" / "patent_abstract.parquet", columns=["patent_id", "patent_abstract"])
    date = pd.read_parquet(PATENTSVIEW / "patent_date.parquet", columns=["patent_id", "grant_date"])
    for df in [title, abstract, date]:
        df["patent_id"] = df["patent_id"].astype(str)
    out = patents.merge(title, on="patent_id", how="left")
    out = out.merge(abstract, on="patent_id", how="left")
    out = out.merge(date, on="patent_id", how="left")
    return out


def load_claim1(patent_ids: set[str]) -> pd.DataFrame:
    claim_rows: list[pd.DataFrame] = []
    for year in range(2005, 2020):
        zpath = PATENTSVIEW / "11Claims" / "raw_zip" / f"g_claims_{year}.tsv.zip"
        with zipfile.ZipFile(zpath) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as handle:
                reader = pd.read_csv(
                    handle,
                    sep="\t",
                    dtype={
                        "patent_id": "string",
                        "claim_sequence": "Int64",
                        "claim_text": "string",
                        "claim_number": "string",
                    },
                    usecols=["patent_id", "claim_sequence", "claim_text", "claim_number"],
                    chunksize=400_000,
                    low_memory=False,
                )
                year_chunks = []
                for chunk in reader:
                    chunk = chunk[chunk["patent_id"].isin(patent_ids)]
                    if chunk.empty:
                        continue
                    is_first = chunk["claim_sequence"].eq(0) | chunk["claim_number"].astype(str).str.zfill(5).eq("00001")
                    chunk = chunk[is_first]
                    if not chunk.empty:
                        year_chunks.append(chunk)
                if year_chunks:
                    claim_rows.append(pd.concat(year_chunks, ignore_index=True))
    if not claim_rows:
        return pd.DataFrame(columns=["patent_id", "claim_1_text"])
    claims = pd.concat(claim_rows, ignore_index=True)
    claims["claim_sequence"] = pd.to_numeric(claims["claim_sequence"], errors="coerce").fillna(999999).astype(int)
    claims = claims.sort_values(["patent_id", "claim_sequence", "claim_number"])
    claims = claims.drop_duplicates("patent_id", keep="first")
    claims = claims.rename(columns={"claim_text": "claim_1_text"})
    return claims[["patent_id", "claim_1_text"]]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    started = time.time()
    for path in [SAMPLE_DIR, LOG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    patents = load_a61b_ids()
    patents = load_titles_abstracts(patents)
    claims = load_claim1(set(patents["patent_id"].astype(str)))
    patents = patents.merge(claims, on="patent_id", how="left")

    patents["patent_id"] = patents["patent_id"].map(patent_id_text)
    patents["patent_title"] = patents["patent_title"].map(clean_text)
    patents["patent_abstract"] = patents["patent_abstract"].map(clean_text)
    patents["claim_1_text"] = patents["claim_1_text"].map(clean_text)
    patents["grant_date"] = patents["grant_date"].astype(str).replace({"NaT": ""})
    patents = patents.sort_values(["grant_year", "patent_id"]).drop_duplicates("patent_id")

    records: list[dict] = []
    for row in patents.itertuples(index=False):
        patent_id = str(row.patent_id)
        records.append(
            {
                "request_id": f"a61b_pme_{patent_id}_v1",
                "patent_id": patent_id,
                "fields": ["medical_device_a61b"],
                "sample_roles": ["a61b_focus_2005_2019"],
                "grant_year": int(row.grant_year),
                "grant_date": row.grant_date,
                "input_sources": {
                    "title": row.patent_title,
                    "abstract": row.patent_abstract,
                    "claim_1": row.claim_1_text,
                },
            }
        )

    if len({r["request_id"] for r in records}) != len(records):
        raise RuntimeError("Duplicate request_id detected.")
    if len({r["patent_id"] for r in records}) != len(records):
        raise RuntimeError("Duplicate patent_id detected.")

    write_jsonl(INPUT_JSONL, records)
    patents.to_csv(UNIVERSE_CSV, index=False, encoding="utf-8-sig")

    rng = random.Random(20260801)
    sample = rng.sample(records, min(30, len(records)))
    write_jsonl(SAMPLE_JSONL, sample)

    parsed = 0
    with INPUT_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            json.loads(line)
            parsed += 1

    qc = {
        "field": "medical_device_a61b",
        "cpc_subclass": "A61B",
        "grant_year_min": int(patents["grant_year"].min()),
        "grant_year_max": int(patents["grant_year"].max()),
        "patents": int(len(patents)),
        "jsonl_rows": int(len(records)),
        "jsonl_parseable_rows": int(parsed),
        "unique_patent_id": len({r["patent_id"] for r in records}) == len(records),
        "unique_request_id": len({r["request_id"] for r in records}) == len(records),
        "title_missing_rate": float((patents["patent_title"].eq("")).mean()),
        "abstract_missing_rate": float((patents["patent_abstract"].eq("")).mean()),
        "claim_1_missing_rate": float((patents["claim_1_text"].eq("")).mean()),
        "mean_title_words": float(patents["patent_title"].str.split().map(len).mean()),
        "mean_abstract_words": float(patents["patent_abstract"].str.split().map(len).mean()),
        "mean_claim_1_words": float(patents["claim_1_text"].str.split().map(len).mean()),
        "outputs": {
            "universe_csv": str(UNIVERSE_CSV),
            "llm_input_jsonl": str(INPUT_JSONL),
            "sample_jsonl": str(SAMPLE_JSONL),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    QC_JSON.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    QC_MD.write_text(
        "\n".join(
            [
                "# A61B LLM Input QC",
                "",
                f"- Field: `{qc['field']}`",
                f"- CPC subclass: `{qc['cpc_subclass']}`",
                f"- Grant years: `{qc['grant_year_min']}-{qc['grant_year_max']}`",
                f"- Patents: `{qc['patents']}`",
                f"- JSONL rows: `{qc['jsonl_rows']}`",
                f"- JSONL parseable rows: `{qc['jsonl_parseable_rows']}`",
                f"- Unique patent_id: `{qc['unique_patent_id']}`",
                f"- Unique request_id: `{qc['unique_request_id']}`",
                f"- Title missing rate: `{qc['title_missing_rate']:.4f}`",
                f"- Abstract missing rate: `{qc['abstract_missing_rate']:.4f}`",
                f"- Claim 1 missing rate: `{qc['claim_1_missing_rate']:.4f}`",
                "",
                "## Outputs",
                "",
                f"- `{UNIVERSE_CSV}`",
                f"- `{INPUT_JSONL}`",
                f"- `{SAMPLE_JSONL}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
