from __future__ import annotations

"""Build one reproducible, grant-time prediction panel for the V7 A61B study.

The script deliberately excludes family size, NPL, and continuation/divisional
because no source table for those variables exists in the local PatentsView
release.  Every retained feature is either present at grant or is constructed
from information dated strictly before the focal patent's grant year.
"""

import json
import os
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_config import config_path, load_config

PV = config_path(load_config(ROOT), "patentsview_dir")
WORK = ROOT / "work"
V6 = WORK / "features" / "a61b_modeling_panel_v6_a61b_only.parquet"
OUT = WORK / "prediction"
DATA = OUT / "data"
QA = OUT / "qa"


def write_json(name: str, value: dict) -> None:
    (QA / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def scan_matching(relative: str, columns: list[str], ids: set[str], id_col: str = "patent_id") -> pd.DataFrame:
    """Scan a large parquet table and retain only focal patents."""
    rows: list[pd.DataFrame] = []
    dataset = ds.dataset(PV / relative, format="parquet")
    for batch in dataset.scanner(columns=columns, batch_size=262_144).to_batches():
        part = batch.to_pandas()
        part[id_col] = part[id_col].astype(str)
        part = part.loc[part[id_col].isin(ids)]
        if not part.empty:
            rows.append(part)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)


def add_cpc_features(base: pd.DataFrame) -> pd.DataFrame:
    ids = set(base["patent_id"])
    focal = scan_matching("cpc.parquet", ["patent_id", "cpc_section", "cpc_class", "cpc_subclass", "cpc_group"], ids)
    grouped = focal.groupby("patent_id", sort=False).agg(
        cpc_section_count=("cpc_section", "nunique"),
        cpc_class_count=("cpc_class", "nunique"),
        cpc_subclass_count=("cpc_subclass", "nunique"),
        cpc_group_count=("cpc_group", "nunique"),
    ).reset_index()
    base = base.merge(grouped, on="patent_id", how="left", validate="one_to_one")

    div = scan_matching("2Innovation/Diversity.parquet", ["patent_id", "Entropy", "RaoStirling", "Disparity", "Variety", "Balance"], ids)
    div = div.rename(columns={"Entropy": "cpc_entropy", "RaoStirling": "cpc_rao_stirling", "Disparity": "cpc_disparity", "Variety": "cpc_variety", "Balance": "cpc_balance"})
    base = base.merge(div, on="patent_id", how="left", validate="one_to_one")

    hhi = scan_matching("2Innovation/patent_hhi.parquet", ["patent_id", "HHI3_ref"], ids)
    hhi = hhi.rename(columns={"HHI3_ref": "backward_cpc_hhi"})
    base = base.merge(hhi, on="patent_id", how="left", validate="one_to_one")
    base["originality_cpc"] = 1.0 - pd.to_numeric(base["backward_cpc_hhi"], errors="coerce")
    return base


def add_dates_and_text(base: pd.DataFrame) -> pd.DataFrame:
    ids = set(base["patent_id"])
    dates = scan_matching("patent_date.parquet", ["patent_id", "grant_date", "filing_date"], ids)
    dates["grant_date_pv"] = pd.to_datetime(dates["grant_date"], errors="coerce")
    dates["filing_date"] = pd.to_datetime(dates["filing_date"], errors="coerce")
    dates["grant_lag_days"] = (dates["grant_date_pv"] - dates["filing_date"]).dt.days
    base = base.merge(dates[["patent_id", "grant_lag_days"]], on="patent_id", how="left", validate="one_to_one")
    base["title_words"] = base["patent_title"].fillna("").str.findall(r"\b\w+\b").str.len().astype(float)
    return base


def add_pair_atypicality(base: pd.DataFrame) -> pd.DataFrame:
    """Conservative A61B-history CPC-pair rarity: previous grant years only."""
    group_lists = {}
    for pid, value in zip(base["patent_id"], base["a61b_cpc_groups"]):
        groups = sorted(set(str(value).split("|"))) if pd.notna(value) else []
        group_lists[pid] = [g for g in groups if g and g != "nan"]
    score = pd.Series(np.nan, index=base.index, dtype=float)
    history: Counter[tuple[str, str]] = Counter()
    total = 0
    for year in sorted(base["grant_year"].unique()):
        ix = base.index[base["grant_year"].eq(year)]
        year_pairs: list[tuple[str, str]] = []
        for i in ix:
            groups = group_lists[base.at[i, "patent_id"]]
            pairs = [(groups[a], groups[b]) for a in range(len(groups)) for b in range(a + 1, len(groups))]
            if pairs:
                score.at[i] = float(np.mean([-np.log((history[p] + 1.0) / (total + 1.0)) for p in pairs]))
                year_pairs.extend(pairs)
            else:
                score.at[i] = 0.0
        history.update(year_pairs)
        total += len(year_pairs)
    base["cpc_pair_atypicality"] = score
    return base


def add_independent_claims(base: pd.DataFrame) -> pd.DataFrame:
    """Count claims marked non-dependent in raw USPTO claim files."""
    cache = DATA / "cache_independent_claim_counts.parquet"
    if cache.exists():
        cached = pd.read_parquet(cache)
        cached["patent_id"] = cached["patent_id"].astype(str)
        base = base.merge(cached, on="patent_id", how="left", validate="one_to_one")
        base["independent_claim_count"] = base["independent_claim_count"].fillna(0).astype(float)
        return base
    ids = set(base["patent_id"])
    counts: Counter[str] = Counter()
    root = PV / "11Claims" / "raw_zip"
    for year in range(2005, 2020):
        path = root / f"g_claims_{year}.tsv.zip"
        for chunk in pd.read_csv(path, sep="\t", compression="zip", usecols=["patent_id", "dependent"], chunksize=250_000, dtype={"patent_id": str, "dependent": str}, low_memory=False):
            chunk = chunk.loc[chunk["patent_id"].isin(ids)]
            if chunk.empty:
                continue
            is_dependent = chunk["dependent"].fillna("").str.lower().isin({"1", "true", "t", "yes"})
            counts.update(chunk.loc[~is_dependent, "patent_id"].tolist())
        print(f"independent-claims year={year}", flush=True)
    cached = pd.DataFrame({"patent_id": list(counts), "independent_claim_count": list(counts.values())})
    cached.to_parquet(cache, index=False)
    base["independent_claim_count"] = base["patent_id"].map(counts).fillna(0).astype(float)
    return base


def add_inventor_and_team_features(base: pd.DataFrame) -> pd.DataFrame:
    ids = set(base["patent_id"])
    links = scan_matching("3Inventor/patent_inventor.parquet", ["patent_id", "inventor_id"], ids)
    links["inventor_id"] = links["inventor_id"].astype(str)
    links = links.merge(base[["patent_id", "grant_year"]], on="patent_id", how="left", validate="many_to_one")
    inventor_ids = set(links["inventor_id"])
    dynamic = scan_matching("3Inventor/inventor_dynamic_info.parquet", ["inventor_id", "Year", "Career_age", "citation_count", "Productivity"], inventor_ids, id_col="inventor_id")
    dynamic["inventor_id"] = dynamic["inventor_id"].astype(str)
    dynamic = dynamic.rename(columns={"Year": "dynamic_year"})
    merged = links.merge(dynamic, on="inventor_id", how="left", validate="many_to_many")
    merged = merged.loc[pd.to_numeric(merged["dynamic_year"], errors="coerce") < merged["grant_year"]].copy()
    merged["dynamic_year"] = pd.to_numeric(merged["dynamic_year"], errors="coerce")
    merged = merged.sort_values(["patent_id", "inventor_id", "dynamic_year"]).drop_duplicates(["patent_id", "inventor_id"], keep="last")
    summary = merged.groupby("patent_id", sort=False).agg(
        inventor_prior_productivity=("Productivity", "mean"),
        inventor_prior_citations=("citation_count", "mean"),
        inventor_career_age=("Career_age", "mean"),
    ).reset_index()
    base = base.merge(summary, on="patent_id", how="left", validate="one_to_one")


    members = links.groupby("patent_id", sort=False)["inventor_id"].agg(lambda x: sorted(set(x))).to_dict()
    pair_score = pd.Series(np.nan, index=base.index, dtype=float)
    seen: Counter[tuple[str, str]] = Counter()
    for year in sorted(base["grant_year"].unique()):
        ix = base.index[base["grant_year"].eq(year)]
        update: list[tuple[str, str]] = []
        for i in ix:
            inv = members.get(base.at[i, "patent_id"], [])
            pairs = [(inv[a], inv[b]) for a in range(len(inv)) for b in range(a + 1, len(inv))]
            pair_score.at[i] = float(np.mean([1.0 / (1.0 + seen[p]) for p in pairs])) if pairs else 0.0
            update.extend(pairs)
        seen.update(update)
    base["team_collaboration_novelty"] = pair_score
    return base


def add_assignee_history(base: pd.DataFrame) -> pd.DataFrame:
    """Portfolio stock and pre-year citation stock for the focal primary assignee."""
    focal_assignees = set(base.loc[~base["primary_assignee_id"].eq("missing_assignee"), "primary_assignee_id"].astype(str))
    assignments: list[pd.DataFrame] = []
    dataset = ds.dataset(PV / "6Firm" / "patent_assignee.parquet", format="parquet")
    for batch in dataset.scanner(columns=["patent_id", "assignee_id"], batch_size=262_144).to_batches():
        part = batch.to_pandas(); part["assignee_id"] = part["assignee_id"].astype(str)
        part = part.loc[part["assignee_id"].isin(focal_assignees)]
        if not part.empty:
            assignments.append(part)
    portfolio = pd.concat(assignments, ignore_index=True).drop_duplicates()
    portfolio["patent_id"] = portfolio["patent_id"].astype(str)
    portfolio_ids = set(portfolio["patent_id"])
    dates = scan_matching("patent_date.parquet", ["patent_id", "grant_date"], portfolio_ids)
    dates["grant_year_portfolio"] = pd.to_datetime(dates["grant_date"], errors="coerce").dt.year
    portfolio = portfolio.merge(dates[["patent_id", "grant_year_portfolio"]], on="patent_id", how="inner", validate="many_to_one")
    annual = portfolio.groupby(["assignee_id", "grant_year_portfolio"], as_index=False)["patent_id"].nunique().rename(columns={"patent_id": "n"})
    annual["assignee_prior_patents"] = annual.groupby("assignee_id", sort=False)["n"].cumsum() - annual["n"]
    base = base.merge(annual.rename(columns={"assignee_id": "primary_assignee_id", "grant_year_portfolio": "grant_year"})[["primary_assignee_id", "grant_year", "assignee_prior_patents"]], on=["primary_assignee_id", "grant_year"], how="left", validate="many_to_one")


    portfolio_map = portfolio[["patent_id", "assignee_id"]].drop_duplicates()
    portfolio_set = set(portfolio_map["patent_id"])
    citation_year: Counter[tuple[str, int]] = Counter()
    yearly = ds.dataset(PV / "1Citation" / "patent_YearlyCit.parquet", format="parquet")
    for batch in yearly.scanner(columns=["patent_id", "Year", "Citation"], batch_size=524_288).to_batches():
        part = batch.to_pandas(); part["patent_id"] = part["patent_id"].astype(str)
        part = part.loc[part["patent_id"].isin(portfolio_set) & part["Year"].between(2000, 2018)]
        if part.empty:
            continue
        part = part.merge(portfolio_map, on="patent_id", how="left", validate="many_to_many")
        grouped = part.groupby(["assignee_id", "Year"], sort=False)["Citation"].sum()
        citation_year.update({(str(a), int(y)): float(v) for (a, y), v in grouped.items()})
    cit_rows = []
    for assignee in focal_assignees:
        cumulative = 0.0
        for year in range(2000, 2020):
            cit_rows.append((assignee, year, cumulative))
            cumulative += citation_year[(assignee, year)]
    cit = pd.DataFrame(cit_rows, columns=["primary_assignee_id", "grant_year", "assignee_prior_citations"])
    base = base.merge(cit, on=["primary_assignee_id", "grant_year"], how="left", validate="many_to_one")
    return base


def add_semantic_and_outcomes(base: pd.DataFrame) -> pd.DataFrame:
    ids = set(base["patent_id"])
    sim = scan_matching("7Text/PatentSimilarityImportanceBreakthrough_forPost2022.parquet", ["patent_num", "bsim5", "fsim01", "fsim25"], ids, id_col="patent_num")
    sim = sim.rename(columns={"patent_num": "patent_id", "bsim5": "kelly_bsim5", "fsim01": "fsim01", "fsim25": "fsim25"})
    sim["kelly_text_novelty"] = 1.0 - pd.to_numeric(sim["kelly_bsim5"], errors="coerce")
    sim["semantic_reuse_5y_log1p"] = np.log1p(pd.to_numeric(sim["fsim01"], errors="coerce") + pd.to_numeric(sim["fsim25"], errors="coerce"))
    base = base.merge(sim[["patent_id", "kelly_text_novelty", "semantic_reuse_5y_log1p"]], on="patent_id", how="left", validate="one_to_one")
    for q in (0.01, 0.05, 0.10):
        cutoff = base.groupby(["grant_year", "technical_subfield"])["forward_cites_5y"].transform(lambda x: x.quantile(1 - q))
        base[f"top{int(q * 100)}_impact"] = base["forward_cites_5y"].ge(cutoff).astype(int)
    return base


def add_strict_cross_field_outcome(base: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct five-year citations from patents with no A61B CPC code."""
    focal = base[["patent_id", "grant_date"]].copy()
    focal["pid_int"] = pd.to_numeric(focal["patent_id"], errors="raise").astype("int64")
    focal["grant_date"] = pd.to_datetime(focal["grant_date"])
    focal_ids = set(focal["pid_int"])
    forward_parts: list[pd.DataFrame] = []
    references = ds.dataset(PV / "1Citation" / "patent_references.parquet", format="parquet")
    for batch in references.scanner(columns=["patent_id", "citation_patent_id"], batch_size=524_288).to_batches():
        part = batch.to_pandas()
        part = part.loc[part["citation_patent_id"].isin(focal_ids)]
        if not part.empty:
            forward_parts.append(part)
    forward = pd.concat(forward_parts, ignore_index=True)
    citer_ids = set(forward["patent_id"].astype(int))
    citer_dates = scan_matching("patent_date.parquet", ["patent_id", "grant_date"], {str(x) for x in citer_ids})
    citer_dates["pid_int"] = pd.to_numeric(citer_dates["patent_id"], errors="coerce").astype("int64")
    citer_dates["citing_grant_date"] = pd.to_datetime(citer_dates["grant_date"], errors="coerce")


    a61b_citers: set[int] = set()
    non_a61b_citers: set[int] = set()
    cpc = ds.dataset(PV / "cpc.parquet", format="parquet")
    for batch in cpc.scanner(columns=["patent_id", "cpc_subclass"], batch_size=524_288).to_batches():
        part = batch.to_pandas()
        part = part.loc[part["patent_id"].isin(citer_ids)]
        if part.empty:
            continue
        is_a61b = part["cpc_subclass"].astype(str).str.startswith("A61B")
        a61b_citers.update(part.loc[is_a61b, "patent_id"].astype("int64").unique().tolist())
        non_a61b_citers.update(part.loc[~is_a61b, "patent_id"].astype("int64").unique().tolist())
    citer_dates["has_a61b"] = citer_dates["pid_int"].isin(a61b_citers)
    citer_dates["has_non_a61b"] = citer_dates["pid_int"].isin(non_a61b_citers)
    forward = forward.rename(columns={"patent_id": "citing_pid", "citation_patent_id": "pid_int"})
    forward["citing_pid"] = forward["citing_pid"].astype("int64")
    forward = forward.merge(focal[["pid_int", "grant_date"]], on="pid_int", how="left", validate="many_to_one")
    forward = forward.merge(citer_dates[["pid_int", "citing_grant_date", "has_a61b", "has_non_a61b"]].rename(columns={"pid_int": "citing_pid"}), on="citing_pid", how="left", validate="many_to_one")
    end = forward["grant_date"] + pd.DateOffset(years=5)
    keep = (forward["citing_grant_date"] > forward["grant_date"]) & (forward["citing_grant_date"] <= end) & (~forward["has_a61b"].fillna(False)) & forward["has_non_a61b"].fillna(False)
    counts = forward.loc[keep].groupby("pid_int").size()
    base["outside_field_cites_5y_strict"] = pd.to_numeric(base["patent_id"], errors="raise").astype("int64").map(counts).fillna(0).astype(float)
    return base


def main() -> None:
    for path in (DATA, QA):
        path.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(V6)
    if base["patent_id"].duplicated().any() or len(base) != 135_994:
        raise RuntimeError("Unexpected V6 sample grain.")
    base["patent_id"] = base["patent_id"].astype(str)
    base = add_dates_and_text(base)
    base = add_cpc_features(base)
    base = add_pair_atypicality(base)
    base = add_independent_claims(base)
    base = add_inventor_and_team_features(base)
    base = add_assignee_history(base)
    base = add_semantic_and_outcomes(base)
    base = add_strict_cross_field_outcome(base)

    keep = [
        "patent_id", "grant_year", "grant_date", "technical_subfield", "forward_cites_5y", "top1_impact", "top5_impact", "top10_impact", "outside_field_cites_5y_strict", "semantic_reuse_5y_log1p",
        "overall_novelty_raw", "problem_novelty_raw", "mechanism_novelty_raw", "effect_novelty_raw", "pm_config_novelty_raw", "me_config_novelty_raw", "anchoring_raw", "P_missing", "M_missing", "E_missing",
        "backward_citation_count", "backward_citation_mean_age", "claim_count", "independent_claim_count", "title_words", "abstract_words", "claim_1_words", "cpc_count", "cpc_span", "cpc_section_count", "cpc_class_count", "cpc_subclass_count", "cpc_group_count", "cpc_entropy", "cpc_rao_stirling", "cpc_disparity", "cpc_variety", "cpc_balance", "cpc_pair_atypicality", "grant_lag_days", "assignee_prior_patents", "assignee_prior_citations", "inventor_prior_productivity", "inventor_prior_citations", "inventor_career_age", "team_collaboration_novelty", "originality_cpc", "kelly_text_novelty",
    ]
    panel = base[keep].copy()
    panel.to_parquet(DATA / "a61b_oot_b0_b5_prediction_panel.parquet", index=False)
    panel.to_csv(DATA / "a61b_oot_b0_b5_prediction_panel.csv", index=False, encoding="utf-8-sig")
    excluded = {
        "family_size": "No local family/priority relation table.",
        "npl_citations": "No local non-patent-literature citation table.",
        "continuation_divisional": "No local application-relation table.",
        "disruption_generality_future_impact": "Outcome-dependent by construction; excluded from grant-time B1.",
        "global_historical_network_centrality": "Not included because a timestamped global graph was not precomputed; backward-reference count/age and originality are retained.",
    }
    audit = {
        "rows": int(len(panel)), "unique_patents": int(panel["patent_id"].nunique()), "grant_years": [int(panel.grant_year.min()), int(panel.grant_year.max())],
        "column_null_rate": {c: float(panel[c].isna().mean()) for c in panel.columns}, "excluded_or_deferred": excluded,
        "time_rule": "Portfolio, inventor, team, and CPC-pair history use prior grant years only; assignee citation stock is lagged one full calendar year.",
    }
    write_json("prediction_panel_audit.json", audit)
    print(json.dumps({"panel": str(DATA / "a61b_oot_b0_b5_prediction_panel.parquet"), "rows": len(panel)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
