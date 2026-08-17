from __future__ import annotations

"""One-off cache for independent-claim counts from raw annual claim archives."""

from collections import Counter
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_config import config_path, load_config

PV = config_path(load_config(ROOT), "patentsview_dir")
WORK = ROOT / "work"
V6 = WORK / "features" / "a61b_modeling_panel_v6_a61b_only.parquet"
OUT = WORK / "prediction" / "data" / "cache_independent_claim_counts.parquet"

ids = set(pd.read_parquet(V6, columns=["patent_id"])["patent_id"].astype(str))
counts: Counter[str] = Counter()
for year in range(2005, 2020):
    path = PV / "11Claims" / "raw_zip" / f"g_claims_{year}.tsv.zip"
    for chunk in pd.read_csv(path, sep="\t", compression="zip", usecols=["patent_id", "dependent"], chunksize=250_000, dtype={"patent_id": str, "dependent": str}, low_memory=False):
        chunk = chunk.loc[chunk["patent_id"].isin(ids)]
        is_dependent = chunk["dependent"].fillna("").str.lower().isin({"1", "true", "t", "yes"})
        counts.update(chunk.loc[~is_dependent, "patent_id"].tolist())
    print(f"cached independent claims: {year}", flush=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"patent_id": list(counts), "independent_claim_count": list(counts.values())}).to_parquet(OUT, index=False)
print(f"wrote {OUT} ({len(counts)} patents)", flush=True)
