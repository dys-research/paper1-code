from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_config import config_path, load_config

WORK = ROOT / "work"
FEATURES = WORK / "features"
MODELS = WORK / "models"
REPORTS = WORK / "reports"
QA = WORK / "qa"

UNIVERSE = WORK / "sample" / "a61b_patent_universe_2005_2019.csv"
EMBED_DIR = WORK / "embedding_output"
PATENTSVIEW = config_path(load_config(ROOT), "patentsview_dir")

TOP_K = 10
NOVELTY_CHUNK = 512
ANCHOR_CHUNK = 100_000
SMALL_SUBFIELD_MIN_N = 1000

EMBEDDINGS = {
    "overall": {
        "stem": "a61b_patentsberta_overall_tac",
        "prefix": "overall_novelty",
    },
    "P": {
        "stem": "a61b_patentsberta_P",
        "prefix": "problem_novelty",
    },
    "M": {
        "stem": "a61b_patentsberta_M",
        "prefix": "mechanism_novelty",
    },
    "E": {
        "stem": "a61b_patentsberta_E",
        "prefix": "effect_novelty",
    },
    "PM_config": {
        "stem": "a61b_patentsberta_PM_config",
        "prefix": "pm_config_novelty",
    },
    "ME_config": {
        "stem": "a61b_patentsberta_ME_config",
        "prefix": "me_config_novelty",
    },
}

NOVELTY_PREFIXES = [
    "overall_novelty",
    "problem_novelty",
    "mechanism_novelty",
    "effect_novelty",
    "pm_config_novelty",
    "me_config_novelty",
]

CONTINUOUS_CONTROLS = [
    "log1p_backward_citation_count_z",
    "backward_citation_mean_age_z",
    "log1p_claim_count_z",
    "log1p_abstract_words_z",
    "log1p_claim_1_words_z",
    "log1p_cpc_count_z",
    "log1p_cpc_span_z",
    "log1p_inventor_count_z",
    "log1p_assignee_count_z",
]

MISSING_CONTROLS = ["P_missing", "E_missing"]
CLEAN_CONTROLS = CONTINUOUS_CONTROLS + MISSING_CONTROLS

CORE_TERMS = {
    "overall": "overall_novelty_fy_z_filled",
    "P": "problem_novelty_fy_z_filled",
    "M": "mechanism_novelty_fy_z_filled",
    "E": "effect_novelty_fy_z_filled",
    "PM": "pm_config_novelty_fy_z_filled",
    "ME": "me_config_novelty_fy_z_filled",
    "anchor": "anchoring_centered",
    "PM_anchor": "pm_config_x_anchoring",
    "ME_anchor": "me_config_x_anchoring",
}

LABELS = {
    "overall_novelty_fy_z_filled": "Overall novelty",
    "problem_novelty_fy_z_filled": "P novelty",
    "mechanism_novelty_fy_z_filled": "M novelty",
    "effect_novelty_fy_z_filled": "E novelty",
    "pm_config_novelty_fy_z_filled": "P-M configuration novelty",
    "me_config_novelty_fy_z_filled": "M-E configuration novelty",
    "anchoring_centered": "Anchoring",
    "anchoring_missing": "Anchoring missing",
    "pm_config_x_anchoring": "P-M configuration novelty x anchoring",
    "me_config_x_anchoring": "M-E configuration novelty x anchoring",
    "log1p_backward_citation_count_z": "ln(backward citations + 1)",
    "backward_citation_mean_age_z": "Backward citation mean age",
    "log1p_claim_count_z": "ln(claims + 1)",
    "log1p_abstract_words_z": "ln(abstract words + 1)",
    "log1p_claim_1_words_z": "ln(claim 1 words + 1)",
    "log1p_cpc_count_z": "ln(CPC count + 1)",
    "log1p_cpc_span_z": "ln(CPC span + 1)",
    "log1p_inventor_count_z": "ln(inventors + 1)",
    "log1p_assignee_count_z": "ln(assignees + 1)",
    "P_missing": "P missing",
    "E_missing": "E missing",
}

OUTCOMES = [
    ("forward_cites_5y", "poisson", "Five-year citations (PPML)"),
    ("ln_forward_cites_5y", "ols", "ln(five-year citations + 1) OLS"),
    ("top10_impact", "logit", "Top 10% impact Logit"),
    ("top1_impact", "logit", "Top 1% impact Logit"),
]


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (Path, datetime)):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def patent_id_as_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "+"
    return ""


def safe_name(value: Any) -> str:
    out = str(value)
    for old, new in [(" ", "_"), ("-", "_"), ("/", "_"), (".", "_"), ("'", "")]:
        out = out.replace(old, new)
    return out


def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - x.mean()) / sd


def word_count(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.findall(r"[A-Za-z0-9]+").str.len().astype(float)


def parse_a61b_groups(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    groups: list[str] = []
    for part in re.split(r"[;,\s|]+", str(value).upper()):
        m = re.match(r"^(A61B)(\d+)", part.strip())
        if not m:
            continue
        number = int(m.group(2).lstrip("0") or "0")
        if 2000 <= number <= 2099:
            number = number - 2000
        groups.append(f"A61B{number}")
    return groups


def choose_a61b_subfield(value: Any) -> str:
    groups = parse_a61b_groups(value)
    if not groups:
        return "A61B_other"
    for group in groups:
        number = int(group[4:])
        if number < 200:
            return group
    return groups[0]


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def prepare_base_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(UNIVERSE, dtype={"patent_id": str}, low_memory=False)
    df["patent_id"] = patent_id_as_str(df["patent_id"])
    df["grant_year"] = pd.to_numeric(df["grant_year"], errors="coerce").astype(int)
    df["grant_date"] = pd.to_datetime(df["grant_date"], errors="coerce")
    df["technical_subfield_raw"] = df["a61b_cpc_groups"].map(choose_a61b_subfield)

    counts = df["technical_subfield_raw"].value_counts()
    keep = set(counts[counts >= SMALL_SUBFIELD_MIN_N].index)
    df["technical_subfield"] = np.where(
        df["technical_subfield_raw"].isin(keep),
        df["technical_subfield_raw"],
        "A61B_other",
    )
    df["primary_field"] = "A61B"
    df["abstract_words"] = word_count(df.get("patent_abstract", pd.Series("", index=df.index)))
    df["claim_1_words"] = word_count(df.get("claim_1_text", pd.Series("", index=df.index)))
    df = df.sort_values(["grant_year", "grant_date", "patent_id"]).reset_index(drop=True)

    audit = {
        "universe": str(UNIVERSE),
        "rows": int(len(df)),
        "duplicate_patent_ids": int(df["patent_id"].duplicated().sum()),
        "grant_year_min": int(df["grant_year"].min()),
        "grant_year_max": int(df["grant_year"].max()),
        "small_subfield_min_n": SMALL_SUBFIELD_MIN_N,
        "technical_subfield_raw_counts": counts.to_dict(),
        "technical_subfield_counts": df["technical_subfield"].value_counts().to_dict(),
        "technical_subfield_year_cell_min": int(df.groupby(["technical_subfield", "grant_year"]).size().min()),
        "technical_subfield_year_cell_median": float(df.groupby(["technical_subfield", "grant_year"]).size().median()),
    }
    return df, audit


def load_embedding(kind: str) -> tuple[pd.DataFrame, np.ndarray]:
    spec = EMBEDDINGS[kind]
    meta = load_jsonl(EMBED_DIR / f"{spec['stem']}_metadata.jsonl")
    vectors = np.load(EMBED_DIR / f"{spec['stem']}.npy", mmap_mode="r")
    if vectors.shape[0] != len(meta):
        raise ValueError(f"{kind} metadata/vector mismatch: {len(meta)} vs {vectors.shape[0]}")
    meta = meta.copy()
    meta["patent_id"] = patent_id_as_str(meta["patent_id"])
    meta["vector_row"] = np.arange(len(meta), dtype=np.int64)
    return meta[["patent_id", "vector_row"]], vectors


def compute_one_novelty(kind: str, panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = EMBEDDINGS[kind]
    prefix = spec["prefix"]
    meta, vectors = load_embedding(kind)
    work = meta.merge(
        panel[["patent_id", "grant_year", "technical_subfield"]],
        on="patent_id",
        how="left",
        validate="one_to_one",
    )
    if work["technical_subfield"].isna().any():
        raise ValueError(f"{kind} embedding has rows outside panel")

    out_parts: list[pd.DataFrame] = []
    audit_groups: list[dict[str, Any]] = []
    for subfield, g in work.groupby("technical_subfield", sort=True):
        g = g.sort_values(["grant_year", "patent_id"]).reset_index(drop=True)
        raw = np.full(len(g), np.nan, dtype=np.float32)
        mean_sim = np.full(len(g), np.nan, dtype=np.float32)
        max_sim = np.full(len(g), np.nan, dtype=np.float32)
        hist_used = np.zeros(len(g), dtype=np.int32)
        rows = g["vector_row"].to_numpy(np.int64)
        years = g["grant_year"].to_numpy(np.int32)
        unique_years = sorted(np.unique(years).tolist())

        for year in unique_years:
            focal_pos = np.flatnonzero(years == year)
            hist_pos = np.flatnonzero(years < year)
            if hist_pos.size == 0 or focal_pos.size == 0:
                continue
            hist_vectors = np.asarray(vectors[rows[hist_pos]], dtype=np.float32)
            for start in range(0, focal_pos.size, NOVELTY_CHUNK):
                pos = focal_pos[start : start + NOVELTY_CHUNK]
                focal_vectors = np.asarray(vectors[rows[pos]], dtype=np.float32)
                sims = focal_vectors @ hist_vectors.T
                used = min(TOP_K, sims.shape[1])
                if sims.shape[1] > used:
                    top = np.partition(sims, sims.shape[1] - used, axis=1)[:, -used:]
                else:
                    top = sims
                mean_vals = top.mean(axis=1)
                max_vals = top.max(axis=1)
                mean_sim[pos] = mean_vals
                max_sim[pos] = max_vals
                raw[pos] = 1.0 - mean_vals
                hist_used[pos] = sims.shape[1]

        part = pd.DataFrame(
            {
                "patent_id": g["patent_id"].to_numpy(),
                f"{prefix}_raw": raw,
                f"{prefix}_topk_mean_similarity": mean_sim,
                f"{prefix}_topk_max_similarity": max_sim,
                f"{prefix}_history_count": hist_used,
            }
        )
        out_parts.append(part)
        audit_groups.append(
            {
                "embedding_type": kind,
                "technical_subfield": subfield,
                "rows": int(len(g)),
                "observed_novelty": int(np.isfinite(raw).sum()),
                "missing_novelty": int((~np.isfinite(raw)).sum()),
            }
        )

    novelty = pd.concat(out_parts, ignore_index=True)
    audit = {
        "embedding_type": kind,
        "prefix": prefix,
        "rows_with_embedding": int(len(work)),
        "observed_novelty": int(novelty[f"{prefix}_raw"].notna().sum()),
        "groups": audit_groups,
    }
    return novelty, audit


def add_grouped_standardization(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    groups = [df["grant_year"], df["technical_subfield"]]
    for prefix in NOVELTY_PREFIXES:
        raw_col = f"{prefix}_raw"
        mean = pd.to_numeric(df[raw_col], errors="coerce").groupby(groups).transform("mean")
        std = pd.to_numeric(df[raw_col], errors="coerce").groupby(groups).transform(lambda s: s.std(ddof=0))
        z = (pd.to_numeric(df[raw_col], errors="coerce") - mean) / std.replace(0, np.nan)
        pct = pd.to_numeric(df[raw_col], errors="coerce").groupby(groups).rank(pct=True)
        df[f"{prefix}_fy_z"] = z
        df[f"{prefix}_fy_pct"] = pct
        df[f"{prefix}_fy_z_filled"] = z.fillna(0.0)
    return df


def build_knowledge_increment(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = panel.copy()
    audits: list[dict[str, Any]] = []
    for kind in EMBEDDINGS:
        started = time.time()
        print(f"[knowledge] {kind} novelty started")
        novelty, audit = compute_one_novelty(kind, df)
        audit["seconds"] = round(time.time() - started, 2)
        audits.append(audit)
        df = df.merge(novelty, on="patent_id", how="left", validate="one_to_one")
        print(f"[knowledge] {kind} novelty done in {audit['seconds']}s")
    df = add_grouped_standardization(df)
    path = FEATURES / "a61b_knowledge_increment.parquet"
    df.to_parquet(path, index=False)
    audit_all = {
        "top_k": TOP_K,
        "standardization": "grant_year x technical_subfield",
        "output": str(path),
        "embeddings": audits,
    }
    write_json(QA / "a61b_knowledge_increment_qc.json", audit_all)
    return df, audit_all


def read_filtered_parquet(path: Path, columns: list[str], ids: set[str], id_col: str = "patent_id") -> pd.DataFrame:
    table = ds.dataset(path, format="parquet").to_table(
        columns=columns,
        filter=pc.field(id_col).isin(list(ids)),
    )
    df = table.to_pandas()
    if id_col in df.columns:
        df[id_col] = patent_id_as_str(df[id_col])
    return df


def build_outcomes(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = panel.copy()
    ids = set(df["patent_id"].astype(str))
    yearly = read_filtered_parquet(
        PATENTSVIEW / "1Citation" / "patent_YearlyCit.parquet",
        ["patent_id", "Year", "Citation"],
        ids,
    )
    yearly["Year"] = pd.to_numeric(yearly["Year"], errors="coerce")
    yearly["Citation"] = pd.to_numeric(yearly["Citation"], errors="coerce").fillna(0)
    y = df[["patent_id", "grant_year"]].merge(yearly, on="patent_id", how="left")
    in_window = (y["Year"] >= y["grant_year"]) & (y["Year"] <= y["grant_year"] + 5)
    cit5 = y.loc[in_window].groupby("patent_id")["Citation"].sum()
    df["forward_cites_5y"] = df["patent_id"].map(cit5).fillna(0.0).astype(float)
    df["ln_forward_cites_5y"] = np.log1p(df["forward_cites_5y"])
    cutoff10 = df["forward_cites_5y"].groupby([df["grant_year"], df["technical_subfield"]]).transform(
        lambda s: s.quantile(0.90)
    )
    cutoff1 = df["forward_cites_5y"].groupby([df["grant_year"], df["technical_subfield"]]).transform(
        lambda s: s.quantile(0.99)
    )
    df["top10_impact"] = df["forward_cites_5y"].ge(cutoff10).astype(int)
    df["top1_impact"] = df["forward_cites_5y"].ge(cutoff1).astype(int)
    audit = {
        "yearly_citation_table": str(PATENTSVIEW / "1Citation" / "patent_YearlyCit.parquet"),
        "yearly_rows_filtered": int(len(yearly)),
        "patents_with_yearly_rows": int(yearly["patent_id"].nunique()),
        "zero_filled_patents": int((df["forward_cites_5y"] == 0).sum()),
        "mean_forward_cites_5y": float(df["forward_cites_5y"].mean()),
        "top10_positive": int(df["top10_impact"].sum()),
        "top1_positive": int(df["top1_impact"].sum()),
        "outcome_rule": "Sum yearly citations from grant_year through grant_year+5 inclusive; absent yearly rows are treated as zero citations.",
    }
    return df, audit


def build_controls(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = panel.copy()
    ids = set(df["patent_id"].astype(str))

    nref = read_filtered_parquet(PATENTSVIEW / "1Citation" / "patent_nref.parquet", ["patent_id", "n_ref"], ids)
    nref["n_ref"] = pd.to_numeric(nref["n_ref"], errors="coerce").fillna(0)
    df["backward_citation_count"] = df["patent_id"].map(nref.set_index("patent_id")["n_ref"]).fillna(0.0)

    claims = read_filtered_parquet(PATENTSVIEW / "5Law" / "patents_nclaims.parquet", ["patent_id", "num_claims"], ids)
    claims["num_claims"] = pd.to_numeric(claims["num_claims"], errors="coerce").fillna(0)
    df["claim_count"] = df["patent_id"].map(claims.set_index("patent_id")["num_claims"]).fillna(0.0)

    cpc = read_filtered_parquet(
        PATENTSVIEW / "cpc.parquet",
        ["patent_id", "cpc_class", "cpc_subclass", "cpc_group"],
        ids,
    )
    cpc_agg = cpc.groupby("patent_id").agg(
        cpc_count=("cpc_group", "size"),
        cpc_span=("cpc_subclass", "nunique"),
    )
    df["cpc_count"] = df["patent_id"].map(cpc_agg["cpc_count"]).fillna(0.0)
    df["cpc_span"] = df["patent_id"].map(cpc_agg["cpc_span"]).fillna(0.0)

    inventors = read_filtered_parquet(PATENTSVIEW / "3Inventor" / "patent_inventor.parquet", ["patent_id", "inventor_id"], ids)
    inv_agg = inventors.groupby("patent_id")["inventor_id"].nunique()
    df["inventor_count"] = df["patent_id"].map(inv_agg).fillna(0.0)

    assignees = read_filtered_parquet(PATENTSVIEW / "6Firm" / "patent_assignee.parquet", ["patent_id", "assignee_id"], ids)
    assignees["assignee_id"] = assignees["assignee_id"].astype(str)
    asg_agg = assignees.groupby("patent_id")["assignee_id"].nunique()
    primary_asg = assignees.sort_values(["patent_id", "assignee_id"]).drop_duplicates("patent_id").set_index("patent_id")[
        "assignee_id"
    ]
    df["assignee_count"] = df["patent_id"].map(asg_agg).fillna(0.0)
    df["primary_assignee_id"] = df["patent_id"].map(primary_asg).fillna("missing_assignee")

    refs = read_filtered_parquet(
        PATENTSVIEW / "1Citation" / "patent_references.parquet",
        ["patent_id", "citation_patent_id"],
        ids,
    )
    refs["citation_patent_id"] = patent_id_as_str(refs["citation_patent_id"])
    cited_ids = set(refs["citation_patent_id"].dropna().astype(str))
    cited_year = read_filtered_parquet(
        PATENTSVIEW / "patent_year.parquet",
        ["patent_id", "grant_year"],
        cited_ids,
    )
    cited_year["grant_year"] = pd.to_numeric(cited_year["grant_year"], errors="coerce")
    age_work = refs.merge(cited_year.rename(columns={"patent_id": "citation_patent_id", "grant_year": "cited_grant_year"}), on="citation_patent_id", how="left")
    age_work = age_work.merge(df[["patent_id", "grant_year"]], on="patent_id", how="left")
    age_work["backward_age"] = age_work["grant_year"] - age_work["cited_grant_year"]
    age = age_work.groupby("patent_id")["backward_age"].mean()
    df["backward_citation_mean_age"] = df["patent_id"].map(age).fillna(0.0).clip(lower=0)

    meta_p, _ = load_embedding("P")
    meta_m, _ = load_embedding("M")
    meta_e, _ = load_embedding("E")
    p_ids = set(meta_p["patent_id"].astype(str))
    m_ids = set(meta_m["patent_id"].astype(str))
    e_ids = set(meta_e["patent_id"].astype(str))
    df["P_missing"] = (~df["patent_id"].isin(p_ids)).astype(int)
    df["M_missing"] = (~df["patent_id"].isin(m_ids)).astype(int)
    df["E_missing"] = (~df["patent_id"].isin(e_ids)).astype(int)

    raw_for_z = {
        "log1p_backward_citation_count_z": np.log1p(df["backward_citation_count"]),
        "backward_citation_mean_age_z": df["backward_citation_mean_age"],
        "log1p_claim_count_z": np.log1p(df["claim_count"]),
        "log1p_abstract_words_z": np.log1p(df["abstract_words"]),
        "log1p_claim_1_words_z": np.log1p(df["claim_1_words"]),
        "log1p_cpc_count_z": np.log1p(df["cpc_count"]),
        "log1p_cpc_span_z": np.log1p(df["cpc_span"]),
        "log1p_inventor_count_z": np.log1p(df["inventor_count"]),
        "log1p_assignee_count_z": np.log1p(df["assignee_count"]),
    }
    for col, values in raw_for_z.items():
        df[col] = zscore(pd.Series(values, index=df.index)).fillna(0.0)

    audit = {
        "control_tables": {
            "nref": str(PATENTSVIEW / "1Citation" / "patent_nref.parquet"),
            "references": str(PATENTSVIEW / "1Citation" / "patent_references.parquet"),
            "claims": str(PATENTSVIEW / "5Law" / "patents_nclaims.parquet"),
            "cpc": str(PATENTSVIEW / "cpc.parquet"),
            "inventors": str(PATENTSVIEW / "3Inventor" / "patent_inventor.parquet"),
            "assignees": str(PATENTSVIEW / "6Firm" / "patent_assignee.parquet"),
        },
        "P_missing": int(df["P_missing"].sum()),
        "M_missing": int(df["M_missing"].sum()),
        "E_missing": int(df["E_missing"].sum()),
        "assignee_missing_patents": int((df["primary_assignee_id"] == "missing_assignee").sum()),
        "reference_rows_filtered": int(len(refs)),
    }
    return df, audit


def build_anchoring(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = panel.copy()
    ids = set(df["patent_id"].astype(str))
    meta_m, vectors = load_embedding("M")
    m_ids = set(meta_m["patent_id"].astype(str))
    id_to_row = pd.Series(meta_m["vector_row"].to_numpy(), index=meta_m["patent_id"].astype(str)).to_dict()

    refs = read_filtered_parquet(
        PATENTSVIEW / "1Citation" / "patent_references.parquet",
        ["patent_id", "citation_patent_id"],
        ids,
    )
    refs["citation_patent_id"] = patent_id_as_str(refs["citation_patent_id"])
    refs = refs[refs["patent_id"].isin(m_ids) & refs["citation_patent_id"].isin(m_ids)].copy()
    refs = refs.merge(df[["patent_id", "grant_year"]].rename(columns={"grant_year": "focal_year"}), on="patent_id", how="left")
    refs = refs.merge(
        df[["patent_id", "grant_year"]].rename(columns={"patent_id": "citation_patent_id", "grant_year": "cited_year"}),
        on="citation_patent_id",
        how="left",
    )
    refs = refs[(refs["cited_year"].notna()) & (refs["cited_year"] < refs["focal_year"])].copy()

    if len(refs) == 0:
        df["anchoring_raw"] = np.nan
    else:
        focal_rows = refs["patent_id"].map(id_to_row).to_numpy(np.int64)
        cited_rows = refs["citation_patent_id"].map(id_to_row).to_numpy(np.int64)
        sims = np.empty(len(refs), dtype=np.float32)
        for start in range(0, len(refs), ANCHOR_CHUNK):
            end = min(start + ANCHOR_CHUNK, len(refs))
            focal_vec = np.asarray(vectors[focal_rows[start:end]], dtype=np.float32)
            cited_vec = np.asarray(vectors[cited_rows[start:end]], dtype=np.float32)
            sims[start:end] = np.sum(focal_vec * cited_vec, axis=1)
        refs["anchoring_similarity"] = sims
        anchor = refs.groupby("patent_id")["anchoring_similarity"].mean()
        df["anchoring_raw"] = df["patent_id"].map(anchor)

    df["anchoring_missing"] = df["anchoring_raw"].isna().astype(int)
    anchor_mean = pd.to_numeric(df["anchoring_raw"], errors="coerce").mean()
    df["anchoring_filled"] = pd.to_numeric(df["anchoring_raw"], errors="coerce").fillna(anchor_mean)
    group_mean = df["anchoring_filled"].groupby([df["grant_year"], df["technical_subfield"]]).transform("mean")
    df["anchoring_centered"] = df["anchoring_filled"] - group_mean
    df["pm_config_x_anchoring"] = df["pm_config_novelty_fy_z_filled"] * df["anchoring_centered"]
    df["me_config_x_anchoring"] = df["me_config_novelty_fy_z_filled"] * df["anchoring_centered"]

    audit = {
        "definition": "Mean cosine similarity between focal patent M and backward-cited A61B patents' M; missing values are mean-imputed and flagged by anchoring_missing.",
        "candidate_reference_rows": int(len(refs)),
        "anchoring_observed_patents": int((df["anchoring_missing"] == 0).sum()),
        "anchoring_missing_patents": int((df["anchoring_missing"] == 1).sum()),
        "anchoring_observed_rate": float((df["anchoring_missing"] == 0).mean()),
        "anchoring_mean_for_imputation": float(anchor_mean),
    }
    return df, audit


def build_model_panel(knowledge_panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, outcome_audit = build_outcomes(knowledge_panel)
    df, control_audit = build_controls(df)
    df, anchor_audit = build_anchoring(df)
    panel_path = FEATURES / "a61b_modeling_panel_v6_a61b_only.parquet"
    csv_path = FEATURES / "a61b_modeling_panel_v6_a61b_only.csv"
    df.to_parquet(panel_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    audit = {
        "panel_parquet": str(panel_path),
        "panel_csv": str(csv_path),
        "rows": int(len(df)),
        "outcomes": outcome_audit,
        "controls": control_audit,
        "anchoring": anchor_audit,
        "collinearity_prevention": [
            "primary_field FE is excluded because A61B-only makes it constant.",
            "observed complements such as *_observed are not included in regressions.",
            "M_missing is not included because M embeddings cover all patents.",
            "anchoring_missing is included only as the missing category required by the research design; anchoring_observed is not included.",
        ],
    }
    write_json(QA / "a61b_modeling_panel_qc.json", audit)
    return df, audit


def model_specs(interaction_cols: list[str]) -> dict[str, dict[str, list[str]]]:
    m2 = [CORE_TERMS["overall"]]
    m3 = m2 + [CORE_TERMS["P"], CORE_TERMS["M"], CORE_TERMS["E"]]
    m4 = m3 + [CORE_TERMS["PM"], CORE_TERMS["ME"]]
    m5 = m4 + [CORE_TERMS["anchor"], CORE_TERMS["PM_anchor"], CORE_TERMS["ME_anchor"]]
    return {
        "M0": {"terms": []},
        "M1": {"terms": CLEAN_CONTROLS},
        "M2": {"terms": CLEAN_CONTROLS + m2},
        "M3": {"terms": CLEAN_CONTROLS + m3},
        "M4": {"terms": CLEAN_CONTROLS + m4},
        "M5": {"terms": CLEAN_CONTROLS + ["anchoring_missing"] + m5},
        "M6": {"terms": CLEAN_CONTROLS + ["anchoring_missing"] + m5 + interaction_cols},
    }


def make_m6_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    out = df.copy()
    subfields = sorted(out["technical_subfield"].dropna().astype(str).unique().tolist())
    counts = out["technical_subfield"].value_counts()
    reference = "A61B5" if "A61B5" in subfields else str(counts.index[0])
    base_terms = [
        CORE_TERMS["overall"],
        CORE_TERMS["P"],
        CORE_TERMS["M"],
        CORE_TERMS["E"],
        CORE_TERMS["PM"],
        CORE_TERMS["ME"],
        CORE_TERMS["anchor"],
        CORE_TERMS["PM_anchor"],
        CORE_TERMS["ME_anchor"],
    ]
    interaction_cols: list[str] = []
    for subfield in subfields:
        if subfield == reference:
            continue
        dummy = out["technical_subfield"].eq(subfield).astype(float)
        suffix = safe_name(subfield)
        for term in base_terms:
            col = f"{term}_x_sub_{suffix}"
            out[col] = pd.to_numeric(out[term], errors="coerce").fillna(0.0) * dummy
            interaction_cols.append(col)
    return out, interaction_cols, reference


def drop_bad_terms(df: pd.DataFrame, terms: list[str]) -> tuple[list[str], list[str]]:
    kept, dropped = [], []
    for term in terms:
        if term not in df.columns:
            dropped.append(term)
            continue
        x = pd.to_numeric(df[term], errors="coerce")
        if x.notna().sum() == 0 or x.nunique(dropna=True) <= 1:
            dropped.append(term)
        else:
            kept.append(term)
    return kept, dropped


def build_x(df: pd.DataFrame, terms: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    pieces = []
    if terms:
        pieces.append(df[terms].apply(pd.to_numeric, errors="coerce").astype(float))
    pieces.append(pd.get_dummies(df["grant_year"].astype(str), prefix="year", drop_first=True, dtype=float))
    pieces.append(pd.get_dummies(df["technical_subfield"].astype(str), prefix="subfield", drop_first=True, dtype=float))
    xdf = pd.concat(pieces, axis=1)
    xdf = sm.add_constant(xdf, has_constant="add")
    return xdf, xdf.to_numpy(dtype=float)


def rank_info(x: np.ndarray) -> dict[str, Any]:
    singular = np.linalg.svd(x, compute_uv=False)
    tol = max(x.shape) * np.finfo(float).eps * singular[0]
    rank = int((singular > tol).sum())
    return {
        "columns": int(x.shape[1]),
        "rank": rank,
        "deficiency": int(x.shape[1] - rank),
        "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf,
    }


def fit_one(df: pd.DataFrame, outcome: str, family: str, model_name: str, terms: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    need = [outcome, "grant_year", "technical_subfield", "primary_assignee_id"] + terms
    d = df[need].replace([np.inf, -np.inf], np.nan).dropna().copy()
    kept_terms, dropped_terms = drop_bad_terms(d, terms)
    xdf, x = build_x(d, kept_terms)
    y = pd.to_numeric(d[outcome], errors="coerce").astype(float)
    groups = d["primary_assignee_id"].astype(str)
    rank = rank_info(x)

    cov_type = "cluster"
    try:
        if family == "poisson":
            mod = sm.GLM(y, xdf, family=sm.families.Poisson())
            res = mod.fit(maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
            pseudo_r2 = 1.0 - (res.deviance / res.null_deviance) if res.null_deviance else np.nan
        elif family == "logit":
            mod = sm.GLM(y, xdf, family=sm.families.Binomial())
            res = mod.fit(maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
            pseudo_r2 = 1.0 - (res.deviance / res.null_deviance) if res.null_deviance else np.nan
        elif family == "ols":
            mod = sm.OLS(y, xdf)
            res = mod.fit(cov_type="cluster", cov_kwds={"groups": groups})
            pseudo_r2 = float(res.rsquared)
        else:
            raise ValueError(f"Unsupported family: {family}")
    except Exception as exc:
        cov_type = "HC1"
        if family == "poisson":
            mod = sm.GLM(y, xdf, family=sm.families.Poisson())
            res = mod.fit(maxiter=200, cov_type="HC1")
            pseudo_r2 = 1.0 - (res.deviance / res.null_deviance) if res.null_deviance else np.nan
        elif family == "logit":
            mod = sm.GLM(y, xdf, family=sm.families.Binomial())
            res = mod.fit(maxiter=200, cov_type="HC1")
            pseudo_r2 = 1.0 - (res.deviance / res.null_deviance) if res.null_deviance else np.nan
        else:
            mod = sm.OLS(y, xdf)
            res = mod.fit(cov_type="HC1")
            pseudo_r2 = float(res.rsquared)
        dropped_terms.append(f"cluster_cov_failed:{type(exc).__name__}")

    rows = []
    for term in kept_terms:
        if term not in res.params.index:
            continue
        p = float(res.pvalues.get(term, np.nan))
        rows.append(
            {
                "outcome": outcome,
                "family": family,
                "model": model_name,
                "term": term,
                "label": LABELS.get(term, term),
                "coef": float(res.params[term]),
                "std_err": float(res.bse[term]),
                "z_or_t": float(res.tvalues[term]),
                "p_value": p,
                "stars": stars(p),
            }
        )

    meta = {
        "outcome": outcome,
        "family": family,
        "model": model_name,
        "nobs": int(res.nobs),
        "pseudo_r2_or_r2": float(pseudo_r2) if np.isfinite(pseudo_r2) else None,
        "aic": float(res.aic) if hasattr(res, "aic") and np.isfinite(res.aic) else None,
        "cov_type": cov_type,
        "cluster_variable": "primary_assignee_id" if cov_type == "cluster" else None,
        "requested_terms": terms,
        "kept_terms": kept_terms,
        "dropped_terms": dropped_terms,
        "rank": rank,
        "grant_year_fe": True,
        "technical_subfield_fe": True,
    }
    return pd.DataFrame(rows), meta


def run_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    df, interaction_cols, reference = make_m6_interactions(panel)
    specs = model_specs(interaction_cols)
    all_coef = []
    all_meta: list[dict[str, Any]] = []

    for outcome, family, label in OUTCOMES:
        for model_name, spec in specs.items():
            print(f"[model] {outcome} {model_name} started")
            coefs, meta = fit_one(df, outcome, family, model_name, spec["terms"])
            meta["outcome_label"] = label
            all_coef.append(coefs)
            all_meta.append(meta)
            print(f"[model] {outcome} {model_name} done")

    coef_df = pd.concat(all_coef, ignore_index=True)
    coef_path = MODELS / "a61b_m0_m6_coefficients.csv"
    meta_path = MODELS / "a61b_m0_m6_model_metadata.json"
    coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")
    write_json(meta_path, all_meta)
    audit = {
        "m6_reference_subfield": reference,
        "m6_interaction_terms": interaction_cols,
        "coefficients": str(coef_path),
        "metadata": str(meta_path),
    }
    write_json(QA / "a61b_model_run_qc.json", audit)
    return coef_df, all_meta, audit


def format_coef(coef: float, se: float, p: float) -> str:
    return f"{coef:.3f}{stars(p)}\n({se:.3f})"


def make_table(coef_df: pd.DataFrame, meta: list[dict[str, Any]], outcome: str, title: str) -> str:
    models = [f"M{i}" for i in range(7)]
    terms = [
        "overall_novelty_fy_z_filled",
        "problem_novelty_fy_z_filled",
        "mechanism_novelty_fy_z_filled",
        "effect_novelty_fy_z_filled",
        "pm_config_novelty_fy_z_filled",
        "me_config_novelty_fy_z_filled",
        "anchoring_centered",
        "pm_config_x_anchoring",
        "me_config_x_anchoring",
        "anchoring_missing",
        "log1p_backward_citation_count_z",
        "backward_citation_mean_age_z",
        "log1p_claim_count_z",
        "log1p_abstract_words_z",
        "log1p_claim_1_words_z",
        "log1p_cpc_count_z",
        "log1p_cpc_span_z",
        "log1p_inventor_count_z",
        "log1p_assignee_count_z",
        "P_missing",
        "E_missing",
    ]
    sub = coef_df[coef_df["outcome"].eq(outcome)].copy()
    lines = [f"### {title}", "", "| Variable | " + " | ".join(models) + " |", "|" + "---|" * (len(models) + 1)]
    for term in terms:
        cells = []
        for model in models:
            r = sub[(sub["model"].eq(model)) & (sub["term"].eq(term))]
            if len(r) == 0:
                cells.append("")
            else:
                row = r.iloc[0]
                cells.append(format_coef(row["coef"], row["std_err"], row["p_value"]).replace("\n", "<br>"))
        if any(cells):
            lines.append(f"| {LABELS.get(term, term)} | " + " | ".join(cells) + " |")
    meta_df = pd.DataFrame([m for m in meta if m["outcome"] == outcome])
    lines.append("| Year FE | " + " | ".join(["Yes"] * len(models)) + " |")
    lines.append("| Technical subfield FE | " + " | ".join(["Yes"] * len(models)) + " |")
    lines.append("| Subfield interactions | " + " | ".join(["No"] * 6 + ["Yes"]) + " |")
    lines.append("| Clustered SE | " + " | ".join(meta_df.sort_values("model")["cov_type"].map(lambda x: "Assignee" if x == "cluster" else "HC1").tolist()) + " |")
    lines.append("| Observations | " + " | ".join(meta_df.sort_values("model")["nobs"].map(lambda x: f"{int(x):,}").tolist()) + " |")
    lines.append("| Pseudo R2 / R2 | " + " | ".join(meta_df.sort_values("model")["pseudo_r2_or_r2"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}").tolist()) + " |")
    lines.append("")
    return "\n".join(lines)


def write_reports(coef_df: pd.DataFrame, meta: list[dict[str, Any]], qc: dict[str, Any]) -> None:
    sections = [
        "# V6 A61B-only M0-M6 regression results",
        "",
        "Notes: coefficients are reported with standard errors in parentheses. +, *, **, *** denote p < 0.10, 0.05, 0.01, 0.001. Continuous controls and novelty variables are standardized within the specified construction rules; year FE and coarse A61B technical subfield FE are included in all models.",
        "",
    ]
    for outcome, _family, label in OUTCOMES:
        sections.append(make_table(coef_df, meta, outcome, label))
    report = "\n".join(sections)
    (REPORTS / "a61b_m0_m6_regression_tables.md").write_text(report, encoding="utf-8")

    table_csv_rows = []
    for outcome, _family, label in OUTCOMES:
        for model in [f"M{i}" for i in range(7)]:
            m = next(x for x in meta if x["outcome"] == outcome and x["model"] == model)
            table_csv_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": label,
                    "model": model,
                    "nobs": m["nobs"],
                    "pseudo_r2_or_r2": m["pseudo_r2_or_r2"],
                    "cov_type": m["cov_type"],
                    "dropped_terms": ";".join(m["dropped_terms"]),
                    "rank_deficiency": m["rank"]["deficiency"],
                }
            )
    pd.DataFrame(table_csv_rows).to_csv(REPORTS / "a61b_m0_m6_model_summary.csv", index=False, encoding="utf-8-sig")
    write_json(QA / "a61b_full_run_qc.json", qc)


def main() -> None:
    for path in [FEATURES, MODELS, REPORTS, QA]:
        path.mkdir(parents=True, exist_ok=True)

    started = time.time()
    base, base_audit = prepare_base_panel()
    knowledge, knowledge_audit = build_knowledge_increment(base)
    panel, panel_audit = build_model_panel(knowledge)
    coef_df, meta, model_audit = run_models(panel)

    qc = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - started, 2),
        "base_panel": base_audit,
        "knowledge_increment": knowledge_audit,
        "modeling_panel": panel_audit,
        "models": model_audit,
    }
    write_reports(coef_df, meta, qc)
    print("[done] V6 A61B-only steps 4-6 completed")
    print(REPORTS / "a61b_m0_m6_regression_tables.md")


if __name__ == "__main__":
    main()
