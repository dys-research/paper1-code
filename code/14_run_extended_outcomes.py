from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_config import config_path, load_config

WORK = ROOT / "work"
V7_SCRIPT = ROOT / "code" / "07_run_explanatory_models.py"
V7_PANEL = WORK / "explanatory" / "data" / "a61b_v7_modeling_panel.parquet"
CROSS_FIELD = WORK / "diagnostics" / "cross_field_quick_check" / "cross_field_counts_by_patent.parquet"
SEMANTIC_REUSE = config_path(load_config(ROOT), "patentsview_dir") / "7Text" / "PatentSimilarityImportanceBreakthrough_forPost2022.parquet"
EXP = WORK / "extended_outcomes"
DATA_DIR = EXP / "data"
MODEL_DIR = EXP / "models"
REPORT_DIR = EXP / "reports"
QA_DIR = EXP / "qa"

CROSS_OUTCOME = "outside_field_cites_5y_strict"
SEMANTIC_OUTCOME = "semantic_reuse_5y_log1p"
OUTCOMES = [
    (CROSS_OUTCOME, "poisson", "Table 4C: Strict five-year out-of-field citations, PPML"),
    (SEMANTIC_OUTCOME, "ols", "Table 4D: Five-year subsequent semantic reuse, OLS"),
]


def load_v7_module():
    spec = importlib.util.spec_from_file_location("v7_models_extended", V7_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {V7_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V7M = load_v7_module()


def load_panel() -> tuple[pd.DataFrame, dict]:
    panel = pd.read_parquet(V7_PANEL)
    panel["patent_id"] = panel["patent_id"].astype(str)
    cross = pd.read_parquet(CROSS_FIELD, columns=["patent_id", CROSS_OUTCOME])
    cross["patent_id"] = cross["patent_id"].astype(str)
    reuse = pd.read_parquet(SEMANTIC_REUSE, columns=["patent_num", "fsim01", "fsim25"])
    reuse["patent_id"] = reuse["patent_num"].astype(str)
    reuse["semantic_reuse_5y_raw"] = pd.to_numeric(reuse["fsim01"], errors="coerce") + pd.to_numeric(reuse["fsim25"], errors="coerce")
    reuse[SEMANTIC_OUTCOME] = np.log1p(reuse["semantic_reuse_5y_raw"])

    df = panel.merge(cross, on="patent_id", how="left", validate="one_to_one")
    df = df.merge(
        reuse[["patent_id", "semantic_reuse_5y_raw", SEMANTIC_OUTCOME]],
        on="patent_id", how="left", validate="one_to_one",
    )
    df[CROSS_OUTCOME] = df[CROSS_OUTCOME].fillna(0).astype(int)
    df.loc[df["grant_year"].gt(2017), SEMANTIC_OUTCOME] = np.nan
    df.loc[df["grant_year"].gt(2017), "semantic_reuse_5y_raw"] = np.nan

    semantic_sample = df.loc[df[SEMANTIC_OUTCOME].notna()]
    audit = {
        "panel_rows": int(len(df)),
        "unique_patents": int(df["patent_id"].nunique()),
        "cross_field": {
            "n": int(df[CROSS_OUTCOME].notna().sum()),
            "mean": float(df[CROSS_OUTCOME].mean()),
            "variance": float(df[CROSS_OUTCOME].var()),
            "zero_rate": float(df[CROSS_OUTCOME].eq(0).mean()),
            "maximum": int(df[CROSS_OUTCOME].max()),
        },
        "semantic_reuse": {
            "definition": "ln(fsim01 + fsim25 + 1)",
            "year_max": int(semantic_sample["grant_year"].max()),
            "n": int(len(semantic_sample)),
            "missing_2005_2017": int(df.loc[df["grant_year"].le(2017), SEMANTIC_OUTCOME].isna().sum()),
            "mean": float(semantic_sample[SEMANTIC_OUTCOME].mean()),
            "sd": float(semantic_sample[SEMANTIC_OUTCOME].std()),
            "minimum": float(semantic_sample[SEMANTIC_OUTCOME].min()),
            "maximum": float(semantic_sample[SEMANTIC_OUTCOME].max()),
        },
    }
    return df, audit


def make_table(
    coef_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    outcome: str,
    family: str,
    title: str,
    interaction_cols: list[str],
    interaction_labels: dict[str, str],
) -> str:
    models = [f"M{i}" for i in range(7)]
    terms = [
        V7M.OVERALL, V7M.P_NOV, V7M.M_NOV, V7M.E_NOV,
        V7M.PM_RESID, V7M.ME_RESID, V7M.ANCHOR,
        V7M.PM_ANCHOR, V7M.ME_ANCHOR, "anchoring_missing",
    ] + interaction_cols + V7M.CORE_CONTROLS
    lines = [f"### {title}", "", "| Variable | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for term in terms:
        label = interaction_labels.get(term, V7M.LABELS.get(term, term))
        cells = [V7M.format_cell(coef_df, outcome, model, term) for model in models]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    subset = meta_df.loc[meta_df["outcome"].eq(outcome)].set_index("model").reindex(models)
    lines.append("| Grant-year FE | " + " | ".join(["Yes"] * 7) + " |")
    lines.append("| Technical-subfield FE | " + " | ".join(["Yes"] * 7) + " |")
    lines.append("| Configuration × subfield interactions | " + " | ".join(["No"] * 6 + ["Yes"]) + " |")
    lines.append("| Assignee-clustered robust SE | " + " | ".join(["Yes"] * 7) + " |")
    lines.append("| Observations | " + " | ".join(f"{int(value):,}" for value in subset["nobs"]) + " |")
    fit_label = "Pseudo R²" if family == "poisson" else "R²"
    lines.append(f"| {fit_label} | " + " | ".join(f"{value:.4f}" for value in subset["fit_stat"]) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    started = time.time()
    for path in [DATA_DIR, MODEL_DIR, REPORT_DIR, QA_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    df, audit = load_panel()
    df, interaction_cols, reference, interaction_labels = V7M.make_subfield_interactions(df)
    V7M.LABELS.update(interaction_labels)
    specs = V7M.model_specs(interaction_cols)

    coefficient_parts = []
    metadata = []
    marginal_rows = []
    wald_rows = []
    for outcome, family, _title in OUTCOMES:
        for model_name, terms in specs.items():
            print(f"Fitting {outcome} {model_name}", flush=True)
            coefficients, meta, result, _ = V7M.fit_one(df, outcome, family, model_name, terms)
            coefficient_parts.append(coefficients)
            metadata.append(meta)
            if model_name == "M6":
                model_sample = df.loc[df[outcome].notna()]
                subfields = sorted(model_sample["technical_subfield"].astype(str).unique())
                marginal_rows.extend(V7M.subfield_marginal_effects(outcome, result, reference, subfields))
                wald_rows.extend(V7M.joint_wald_tests(outcome, result, interaction_cols))

    coef_df = pd.concat(coefficient_parts, ignore_index=True)
    meta_df = pd.DataFrame(metadata)
    marginal_df = pd.DataFrame(marginal_rows)
    wald_df = pd.DataFrame(wald_rows)
    coef_df.to_csv(MODEL_DIR / "extended_m0_m6_coefficients.csv", index=False, encoding="utf-8-sig")
    meta_df.to_csv(MODEL_DIR / "extended_m0_m6_model_summary.csv", index=False, encoding="utf-8-sig")
    marginal_df.to_csv(MODEL_DIR / "extended_m6_subfield_marginal_effects.csv", index=False, encoding="utf-8-sig")
    wald_df.to_csv(MODEL_DIR / "extended_m6_joint_wald_tests.csv", index=False, encoding="utf-8-sig")

    report_parts = [
        "# M0-M6 explanatory regressions for extended outcomes", "",
        "Note: Cells report coefficients with primary_assignee_id-clustered robust standard errors in parentheses. +, *, **, and *** denote p < 0.10, 0.05, 0.01, and 0.001, respectively. M4-M6 use residualized configuration measures only; M6 additionally includes interactions between the two residualized configuration measures and technical subfield. Strict out-of-field diffusion uses the full sample. Semantic reuse is defined as ln(fsim01 + fsim25 + 1), with the main sample ending in 2017 to retain a complete five-year window.", "",
    ]
    for outcome, family, title in OUTCOMES:
        report_parts.append(make_table(coef_df, meta_df, outcome, family, title, interaction_cols, interaction_labels))
    (REPORT_DIR / "extended_m0_m6_tables.md").write_text("\n".join(report_parts), encoding="utf-8-sig")

    vif_rows = []
    for outcome, _family, _title in OUTCOMES:
        sample = df.loc[df[outcome].notna()]
        block = [V7M.P_NOV, V7M.M_NOV, V7M.E_NOV, V7M.PM_RESID, V7M.ME_RESID, V7M.ANCHOR, V7M.PM_ANCHOR, V7M.ME_ANCHOR]
        vif_rows.extend([{**row, "outcome": outcome} for row in V7M.vif_table(sample, block, "M5 core and mechanism")])
    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(QA_DIR / "extended_vif.csv", index=False, encoding="utf-8-sig")

    qa = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 2),
        "outcome_audit": audit,
        "outcomes": {outcome: family for outcome, family, _ in OUTCOMES},
        "m6_reference_subfield": reference,
        "m6_interaction_count": len(interaction_cols),
        "all_models_converged": bool(meta_df["converged"].all()),
        "all_models_full_rank": bool(meta_df["rank"].map(lambda value: value["deficiency"] == 0).all()),
        "all_coefficients_finite": bool(np.isfinite(coef_df[["coef", "std_err", "p_value"]].to_numpy(float)).all()),
        "maximum_vif": float(vif_df["vif"].max()),
        "outputs": {
            "tables": str(REPORT_DIR / "extended_m0_m6_tables.md"),
            "coefficients": str(MODEL_DIR / "extended_m0_m6_coefficients.csv"),
            "marginal_effects": str(MODEL_DIR / "extended_m6_subfield_marginal_effects.csv"),
            "wald_tests": str(MODEL_DIR / "extended_m6_joint_wald_tests.csv"),
        },
    }
    V7M.write_json(QA_DIR / "extended_m0_m6_qc.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2, default=V7M.json_default), flush=True)


if __name__ == "__main__":
    main()
