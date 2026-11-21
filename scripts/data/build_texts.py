"""Choose the earliest non-cancelled independent claim from configured source files."""
from pathlib import Path
import re
import zipfile

import pandas as pd
import pyarrow.parquet as pq

from common import connect, input_path, output, parquet, qp, receipt, source

CLAIM_COLUMNS = ["patent_id", "claim_sequence", "claim_number", "claim_text", "dependent"]
EMPTY_DEPENDENCY = {"", "null", "none", "nan", "\\n", "<na>"}
CANCELLED = re.compile(r"^[\s\d.()\-]*(?:canceled|cancelled)\b", re.IGNORECASE)


def sequence_key(row):
    def number(value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 10**9
    return number(row["claim_sequence"]), number(row["claim_number"])


def claim_files(root):
    root = Path(root)
    files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    files = [path for path in files if path.suffix.lower() in {".parquet", ".tsv", ".zip", ".gz"}]
    if not files:
        raise FileNotFoundError("claims_root must contain claim Parquet, TSV, TSV.GZ, or ZIP files")
    return files


def iter_claim_chunks(path):
    if path.suffix.lower() == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=CLAIM_COLUMNS):
            yield batch.to_pandas()
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = sorted(name for name in archive.namelist() if name.lower().endswith(".tsv"))
            if not members:
                raise ValueError(f"Claim ZIP has no TSV member: {path.name}")
            for member in members:
                with archive.open(member) as stream:
                    yield from pd.read_csv(stream, sep="\t", dtype="string", keep_default_na=False,
                                           chunksize=50_000, usecols=CLAIM_COLUMNS)
    else:
        yield from pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False,
                               chunksize=50_000, usecols=CLAIM_COLUMNS)


def independent(row):
    dependency = row["dependent"]
    empty = pd.isna(dependency) or str(dependency).strip().lower() in EMPTY_DEPENDENCY
    text = row["claim_text"]
    return empty and isinstance(text, str) and bool(text.strip()) and not CANCELLED.match(text.strip())


def main():
    candidates = output("data/a61b_candidates_1995_2020.parquet")
    candidate_ids = set(pd.read_parquet(candidates, columns=["patent_id"]).patent_id.astype(str))
    paths = claim_files(input_path("claims_root"))
    selected = {}
    for path in paths:
        for chunk in iter_claim_chunks(path):
            chunk["patent_id"] = chunk.patent_id.astype(str)
            for row in chunk.loc[chunk.patent_id.isin(candidate_ids)].to_dict("records"):
                if independent(row):
                    patent_id = row["patent_id"]
                    key = sequence_key(row)
                    previous = selected.get(patent_id)
                    if previous is not None and key == sequence_key(previous) and row["claim_text"] != previous["claim_text"]:
                        raise ValueError(f"Conflicting claims at the same sequence/number for patent {patent_id}")
                    if previous is None or key < sequence_key(previous):
                        row.update(claim_sequence=key[0], claim_number=key[1],
                                   source_kind=path.name, independence_checked=True,
                                   selection_note="earliest_noncancelled_empty_dependency")
                        row["dependent"] = "" if pd.isna(row["dependent"]) else str(row["dependent"])
                        selected[patent_id] = row
        print(f"Read claims from {path.name}", flush=True)
    columns = CLAIM_COLUMNS + ["source_kind", "independence_checked", "selection_note"]
    claims = pd.DataFrame(selected.values(), columns=columns)
    claims["patent_id"] = claims.patent_id.astype("string")
    claims["claim_sequence"] = pd.to_numeric(claims.claim_sequence, errors="coerce").astype("Int64")
    claims["claim_number"] = pd.to_numeric(claims.claim_number, errors="coerce").astype("Int64")
    claims["claim_text"] = claims.claim_text.astype("string")
    claims["dependent"] = claims.dependent.astype("string")
    claims["source_kind"] = claims.source_kind.astype("string")
    claims["selection_note"] = claims.selection_note.astype("string")
    claims["independence_checked"] = claims.independence_checked.astype("boolean")
    selected_path = output("data/selected_independent_claims_1995_2020.parquet")
    claims.to_parquet(selected_path, index=False)
    text = source("02_core/patent_text.parquet")
    connection = connect()
    connection.execute("""CREATE TABLE text_base AS
      SELECT b.*, t.patent_title,t.patent_abstract,
        cl.claim_text independent_claim_text,cl.claim_sequence independent_claim_sequence,
        cl.claim_number independent_claim_number,cl.dependent selected_claim_dependency,
        cl.source_kind claim_source,cl.selection_note claim_selection_note,
        coalesce(cl.independence_checked,FALSE) independence_checked,
        length(trim(coalesce(t.patent_title,'')))>0 has_title,
        length(trim(coalesce(t.patent_abstract,'')))>0 has_abstract,
        length(trim(coalesce(cl.claim_text,'')))>0 AND coalesce(cl.independence_checked,FALSE) has_independent_claim
      FROM read_parquet(""" + qp(candidates) + """) b
      LEFT JOIN read_parquet(""" + qp(text) + """) t ON CAST(t.patent_id AS VARCHAR)=b.patent_id
      LEFT JOIN read_parquet(""" + qp(selected_path) + """) cl ON cl.patent_id=b.patent_id""")
    total, unique = connection.execute("SELECT count(*),count(DISTINCT patent_id) FROM text_base").fetchone()
    if total != unique:
        raise ValueError("Patent text input must contain one row per patent_id")
    parquet(connection, """SELECT *,has_title AND has_abstract AND has_independent_claim full_input_available,
            is_focal_candidate AND has_title AND has_abstract AND has_independent_claim annotation_candidate
            FROM text_base ORDER BY grant_date,patent_id""", "data/a61b_text_base_1995_2020.parquet")
    receipt("checks/texts.json", [candidates, text, *paths], patents=total, selected_claims=len(claims),
            selection_rule="Earliest non-cancelled claim with empty dependency; sequence then claim number.")
    print(f"Prepared {total:,} patent texts; {len(claims):,} have selected independent claims.")


if __name__ == "__main__":
    main()
