from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_config import config_path, load_config

PATENTSVIEW = config_path(load_config(ROOT), "patentsview_dir")
WORK = ROOT / "work"
PANEL_PATH = WORK / "explanatory" / "data" / "a61b_v7_modeling_panel.parquet"
CITATION_PATH = PATENTSVIEW / "1Citation" / "patent_references.parquet"
YEAR_PATH = PATENTSVIEW / "patent_year.parquet"
CPC_PATH = PATENTSVIEW / "cpc.parquet"
OUTPUT_DIR = WORK / "diagnostics" / "cross_field_quick_check"


def load_v7_module():
    script = ROOT / "code" / "07_run_explanatory_models.py"
    spec = importlib.util.spec_from_file_location("v7_models", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scan_five_year_edges(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("[1/4] Loading patent grant years...", flush=True)
    years = pd.read_parquet(YEAR_PATH, columns=["patent_id", "grant_year"])
    years = years.dropna().drop_duplicates("patent_id", keep="last")
    year_map = pd.Series(
        years["grant_year"].astype(np.int16).to_numpy(),
        index=years["patent_id"].astype(np.int64).to_numpy(),
    )
    del years

    focal_year_map = pd.Series(
        panel["grant_year"].astype(np.int16).to_numpy(),
        index=panel["patent_id"].astype(np.int64).to_numpy(),
    )
    focal_ids = pa.array(panel["patent_id"].astype(np.int64).to_numpy())
    parquet = pq.ParquetFile(CITATION_PATH)
    kept: list[pd.DataFrame] = []
    matched_before_window = 0
    missing_citing_year = 0

    print(f"[2/4] Scanning {parquet.metadata.num_rows:,} citation edges...", flush=True)
    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(
            row_group, columns=["patent_id", "citation_patent_id"]
        )
        mask = pc.is_in(table["citation_patent_id"], value_set=focal_ids)
        selected = table.filter(mask).to_pandas()
        if selected.empty:
            continue
        matched_before_window += len(selected)
        selected = selected.rename(
            columns={"patent_id": "citing_patent_id", "citation_patent_id": "focal_patent_id"}
        )
        selected["focal_year"] = selected["focal_patent_id"].map(focal_year_map)
        selected["citing_year"] = selected["citing_patent_id"].map(year_map)
        missing_citing_year += int(selected["citing_year"].isna().sum())
        selected = selected.loc[
            selected["citing_year"].gt(selected["focal_year"])
            & selected["citing_year"].le(selected["focal_year"] + 5),
            ["focal_patent_id", "citing_patent_id"],
        ]
        if not selected.empty:
            kept.append(selected.astype(np.int64))
        if (row_group + 1) % 20 == 0:
            print(f"  citation row groups: {row_group + 1}/{parquet.metadata.num_row_groups}", flush=True)

    edges = pd.concat(kept, ignore_index=True).drop_duplicates()
    diagnostics = {
        "matched_edges_before_window": int(matched_before_window),
        "missing_citing_year_rows": int(missing_citing_year),
        "five_year_unique_edges": int(len(edges)),
        "unique_citing_patents": int(edges["citing_patent_id"].nunique()),
    }
    return edges, diagnostics


def classify_citing_patents(citing_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    citing_ids = np.sort(np.unique(citing_ids.astype(np.int64)))
    arrow_ids = pa.array(citing_ids)
    has_a61b = np.zeros(len(citing_ids), dtype=bool)
    has_non_a61b = np.zeros(len(citing_ids), dtype=bool)
    parquet = pq.ParquetFile(CPC_PATH)

    print(f"[3/4] Scanning {parquet.metadata.num_rows:,} CPC assignments...", flush=True)
    matched_rows = 0
    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(row_group, columns=["patent_id", "cpc_subclass"])
        mask = pc.is_in(table["patent_id"], value_set=arrow_ids)
        selected = table.filter(mask)
        if selected.num_rows == 0:
            continue
        matched_rows += selected.num_rows
        patent_ids = selected["patent_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        positions = np.searchsorted(citing_ids, patent_ids)
        subclass = pc.fill_null(selected["cpc_subclass"], "")
        is_a61b = np.asarray(pc.equal(subclass, "A61B")).astype(bool)
        is_nonempty = np.asarray(pc.not_equal(subclass, "")).astype(bool)
        np.logical_or.at(has_a61b, positions, is_a61b)
        np.logical_or.at(has_non_a61b, positions, is_nonempty & ~is_a61b)
        if (row_group + 1) % 10 == 0:
            print(f"  CPC row groups: {row_group + 1}/{parquet.metadata.num_row_groups}", flush=True)

    diagnostics = {
        "matched_cpc_rows": int(matched_rows),
        "citing_patents_with_any_cpc": int(np.sum(has_a61b | has_non_a61b)),
        "citing_patents_with_a61b": int(np.sum(has_a61b)),
        "citing_patents_with_non_a61b": int(np.sum(has_non_a61b)),
        "citing_patents_mixed_a61b_and_other": int(np.sum(has_a61b & has_non_a61b)),
        "citing_patents_strictly_outside_a61b": int(np.sum(~has_a61b & has_non_a61b)),
    }
    return has_a61b, has_non_a61b, diagnostics


def correlations(df: pd.DataFrame, left: str, right: str) -> dict:
    return {
        "pearson": float(df[left].corr(df[right], method="pearson")),
        "spearman": float(df[left].corr(df[right], method="spearman")),
    }


def main() -> None:
    started = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_PATH)
    panel["patent_id"] = pd.to_numeric(panel["patent_id"], errors="raise").astype(np.int64)
    if panel["patent_id"].duplicated().any():
        raise ValueError("V7 panel contains duplicate patent IDs")

    edges, edge_diag = scan_five_year_edges(panel)
    citing_ids = np.sort(edges["citing_patent_id"].unique().astype(np.int64))
    has_a61b, has_non_a61b, cpc_diag = classify_citing_patents(citing_ids)

    positions = np.searchsorted(citing_ids, edges["citing_patent_id"].to_numpy(np.int64))
    edges["has_a61b"] = has_a61b[positions]
    edges["has_non_a61b"] = has_non_a61b[positions]
    edges["outside_broad"] = edges["has_non_a61b"]
    edges["outside_strict"] = edges["has_non_a61b"] & ~edges["has_a61b"]

    counts = edges.groupby("focal_patent_id", sort=False).agg(
        reconstructed_cites_5y=("citing_patent_id", "size"),
        outside_field_cites_5y_broad=("outside_broad", "sum"),
        outside_field_cites_5y_strict=("outside_strict", "sum"),
    )
    model_df = panel.merge(counts, how="left", left_on="patent_id", right_index=True)
    count_cols = [
        "reconstructed_cites_5y",
        "outside_field_cites_5y_broad",
        "outside_field_cites_5y_strict",
    ]
    model_df[count_cols] = model_df[count_cols].fillna(0).astype(np.int64)
    positive = model_df["reconstructed_cites_5y"].gt(0)
    model_df["outside_share_broad"] = np.where(
        positive,
        model_df["outside_field_cites_5y_broad"] / model_df["reconstructed_cites_5y"],
        np.nan,
    )
    model_df["outside_share_strict"] = np.where(
        positive,
        model_df["outside_field_cites_5y_strict"] / model_df["reconstructed_cites_5y"],
        np.nan,
    )

    citation_comparison = {
        "exact_match_rate": float(
            model_df["reconstructed_cites_5y"].eq(model_df["forward_cites_5y"]).mean()
        ),
        "mean_absolute_difference": float(
            (model_df["reconstructed_cites_5y"] - model_df["forward_cites_5y"]).abs().mean()
        ),
        "reconstructed_mean": float(model_df["reconstructed_cites_5y"].mean()),
        "existing_mean": float(model_df["forward_cites_5y"].mean()),
        "correlation": correlations(model_df, "reconstructed_cites_5y", "forward_cites_5y"),
    }
    outcome_summary = {}
    for col in ["outside_field_cites_5y_broad", "outside_field_cites_5y_strict"]:
        outcome_summary[col] = {
            "mean": float(model_df[col].mean()),
            "zero_rate": float(model_df[col].eq(0).mean()),
            "maximum": int(model_df[col].max()),
            "correlation_with_existing_forward_cites": correlations(model_df, col, "forward_cites_5y"),
            "correlation_with_reconstructed_forward_cites": correlations(
                model_df, col, "reconstructed_cites_5y"
            ),
        }
    share_summary = {
        "broad_mean_among_cited": float(model_df.loc[positive, "outside_share_broad"].mean()),
        "broad_median_among_cited": float(model_df.loc[positive, "outside_share_broad"].median()),
        "strict_mean_among_cited": float(model_df.loc[positive, "outside_share_strict"].mean()),
        "strict_median_among_cited": float(model_df.loc[positive, "outside_share_strict"].median()),
    }

    print("[4/4] Fitting M3 and M5 PPML diagnostics...", flush=True)
    v7 = load_v7_module()
    specs = v7.model_specs([])
    coefficient_frames = []
    model_meta = []
    for outcome in ["outside_field_cites_5y_broad", "outside_field_cites_5y_strict"]:
        for model_name in ["M3", "M5"]:
            coef, meta, _, _ = v7.fit_one(
                model_df,
                outcome=outcome,
                family="poisson",
                model_name=model_name,
                terms=specs[model_name],
            )
            coefficient_frames.append(coef)
            model_meta.append(meta)

    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    core_terms = [
        v7.P_NOV,
        v7.M_NOV,
        v7.E_NOV,
        v7.PM_RESID,
        v7.ME_RESID,
        v7.ANCHOR,
        v7.PM_ANCHOR,
        v7.ME_ANCHOR,
    ]
    core_coefficients = coefficients.loc[coefficients["term"].isin(core_terms)].copy()

    diagnostic = {
        "elapsed_seconds": round(time.time() - started, 2),
        "definition_broad": "A citing patent has at least one non-A61B CPC subclass.",
        "definition_strict": "A citing patent has at least one non-A61B CPC subclass and no A61B subclass.",
        "time_window": "Citing grant year is focal grant year +1 through +5.",
        "rows": int(len(model_df)),
        "edge_scan": edge_diag,
        "cpc_scan": cpc_diag,
        "citation_comparison": citation_comparison,
        "outcomes": outcome_summary,
        "shares": share_summary,
        "models": model_meta,
    }
    model_df[["patent_id", "grant_year", "technical_subfield"] + count_cols + [
        "outside_share_broad", "outside_share_strict"
    ]].to_parquet(OUTPUT_DIR / "cross_field_counts_by_patent.parquet", index=False)
    coefficients.to_csv(OUTPUT_DIR / "quick_ppml_coefficients.csv", index=False, encoding="utf-8-sig")
    core_coefficients.to_csv(OUTPUT_DIR / "quick_ppml_core_coefficients.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "diagnostic_summary.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "elapsed_seconds": diagnostic["elapsed_seconds"],
        "citation_comparison": citation_comparison,
        "outcomes": outcome_summary,
        "shares": share_summary,
    }, ensure_ascii=False, indent=2), flush=True)
    print(core_coefficients[["outcome", "model", "label", "coef", "std_err", "p_value", "stars"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
