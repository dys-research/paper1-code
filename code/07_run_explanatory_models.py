from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
SOURCE_PANEL = WORK / "features" / "a61b_modeling_panel_v6_a61b_only.parquet"
EXP = WORK / "explanatory"
DATA = EXP / "data"
MODELS = EXP / "models"
REPORTS = EXP / "reports"
QA = EXP / "qa"

CORE_CONTROLS = [
    "log1p_backward_citation_count_z",
    "backward_citation_mean_age_z",
    "log1p_claim_count_z",
    "log1p_inventor_count_z",
    "P_missing",
    "E_missing",
]

OVERALL = "overall_novelty_fy_z_filled"
P_NOV = "problem_novelty_fy_z_filled"
M_NOV = "mechanism_novelty_fy_z_filled"
E_NOV = "effect_novelty_fy_z_filled"
PM_ORIGINAL = "pm_config_novelty_fy_z_filled"
ME_ORIGINAL = "me_config_novelty_fy_z_filled"
PM_RESID = "pm_config_residual_z"
ME_RESID = "me_config_residual_z"
ANCHOR = "anchoring_centered"
PM_ANCHOR = "pm_residual_x_anchoring"
ME_ANCHOR = "me_residual_x_anchoring"

LABELS = {
    OVERALL: "Overall novelty",
    P_NOV: "P novelty",
    M_NOV: "M novelty",
    E_NOV: "E novelty",
    PM_RESID: "P-M residual configuration novelty",
    ME_RESID: "M-E residual configuration novelty",
    ANCHOR: "Anchoring",
    PM_ANCHOR: "P-M residual novelty x anchoring",
    ME_ANCHOR: "M-E residual novelty x anchoring",
    "anchoring_missing": "Anchoring missing",
    "log1p_backward_citation_count_z": "ln(backward citations + 1)",
    "backward_citation_mean_age_z": "Backward citation mean age",
    "log1p_claim_count_z": "ln(claims + 1)",
    "log1p_inventor_count_z": "ln(inventors + 1)",
    "P_missing": "P missing",
    "E_missing": "E missing",
}

OUTCOMES = [
    ("forward_cites_5y", "poisson", "Table 4A: Five-year forward citations, PPML"),
    ("cite5_percentile", "ols", "Table 4B: Continuous percentile of five-year citations, OLS"),
]


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def star(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.10:
        return "+"
    return ""


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def residualize(
    df: pd.DataFrame,
    target: str,
    predictors: list[str],
    observed_raw: str,
    output: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    fit_mask = out[observed_raw].notna()
    fit = out.loc[fit_mask, [target] + predictors].replace([np.inf, -np.inf], np.nan).dropna()
    x = sm.add_constant(fit[predictors].astype(float), has_constant="add")
    model = sm.OLS(fit[target].astype(float), x).fit()
    residual = pd.Series(np.nan, index=out.index, dtype=float)
    residual.loc[fit.index] = model.resid
    resid_mean = float(residual.loc[fit.index].mean())
    resid_std = float(residual.loc[fit.index].std(ddof=0))
    if not np.isfinite(resid_std) or resid_std <= 0:
        raise ValueError(f"Residual standard deviation is invalid for {target}: {resid_std}")
    out[output] = ((residual - resid_mean) / resid_std).fillna(0.0)

    diag = {
        "target": target,
        "predictors": predictors,
        "observed_raw": observed_raw,
        "fit_rows": int(len(fit)),
        "filled_zero_rows": int((~fit_mask).sum()),
        "r_squared": float(model.rsquared),
        "coefficients": {name: float(value) for name, value in model.params.items()},
        "residual_mean_before_z": resid_mean,
        "residual_sd_before_z": resid_std,
        "residual_correlations": {
            predictor: float(out.loc[fit.index, output].corr(out.loc[fit.index, predictor]))
            for predictor in predictors
        },
    }
    return out, diag


def prepare_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(SOURCE_PANEL)
    original_rows = len(df)
    if df["patent_id"].duplicated().any():
        raise ValueError("Source panel contains duplicate patent_id values")

    group = [df["grant_year"], df["technical_subfield"]]
    df["cite5_percentile"] = df["forward_cites_5y"].groupby(group).rank(method="average", pct=True)

    df, pm_diag = residualize(
        df,
        target=PM_ORIGINAL,
        predictors=[P_NOV, M_NOV],
        observed_raw="pm_config_novelty_raw",
        output=PM_RESID,
    )
    df, me_diag = residualize(
        df,
        target=ME_ORIGINAL,
        predictors=[M_NOV, E_NOV],
        observed_raw="me_config_novelty_raw",
        output=ME_RESID,
    )
    df[PM_ANCHOR] = df[PM_RESID] * df[ANCHOR]
    df[ME_ANCHOR] = df[ME_RESID] * df[ANCHOR]

    keep = [
        "patent_id",
        "grant_year",
        "technical_subfield",
        "primary_assignee_id",
        "forward_cites_5y",
        "cite5_percentile",
        OVERALL,
        P_NOV,
        M_NOV,
        E_NOV,
        PM_ORIGINAL,
        ME_ORIGINAL,
        PM_RESID,
        ME_RESID,
        ANCHOR,
        PM_ANCHOR,
        ME_ANCHOR,
        "anchoring_missing",
    ] + CORE_CONTROLS
    keep = list(dict.fromkeys(keep))
    model_df = df[keep].copy()
    model_df.to_parquet(DATA / "a61b_v7_modeling_panel.parquet", index=False)
    model_df.to_csv(DATA / "a61b_v7_modeling_panel.csv", index=False, encoding="utf-8-sig")

    group_sizes = model_df.groupby(["grant_year", "technical_subfield"]).size()
    audit = {
        "source_panel": str(SOURCE_PANEL),
        "source_rows": int(original_rows),
        "output_rows": int(len(model_df)),
        "unique_patents": int(model_df["patent_id"].nunique()),
        "year_min": int(model_df["grant_year"].min()),
        "year_max": int(model_df["grant_year"].max()),
        "technical_subfields": model_df["technical_subfield"].value_counts().sort_index().to_dict(),
        "citation": {
            "mean": float(model_df["forward_cites_5y"].mean()),
            "variance": float(model_df["forward_cites_5y"].var()),
            "zero_rate": float(model_df["forward_cites_5y"].eq(0).mean()),
            "maximum": float(model_df["forward_cites_5y"].max()),
        },
        "percentile": {
            "definition": "Midrank percentile of forward_cites_5y within grant_year x technical_subfield",
            "missing": int(model_df["cite5_percentile"].isna().sum()),
            "minimum": float(model_df["cite5_percentile"].min()),
            "maximum": float(model_df["cite5_percentile"].max()),
            "mean": float(model_df["cite5_percentile"].mean()),
            "sd": float(model_df["cite5_percentile"].std()),
            "groups": int(len(group_sizes)),
            "minimum_group_size": int(group_sizes.min()),
            "median_group_size": float(group_sizes.median()),
        },
        "residualization": {"P-M": pm_diag, "M-E": me_diag},
    }
    write_json(QA / "v7_panel_qc.json", audit)
    return model_df, audit


def make_subfield_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str, dict[str, str]]:
    out = df.copy()
    subfields = sorted(out["technical_subfield"].dropna().astype(str).unique())
    reference = "A61B5" if "A61B5" in subfields else str(out["technical_subfield"].value_counts().index[0])
    interaction_cols: list[str] = []
    interaction_labels: dict[str, str] = {}
    for subfield in subfields:
        if subfield == reference:
            continue
        dummy = out["technical_subfield"].astype(str).eq(subfield).astype(float)
        suffix = safe_name(subfield)
        for term, short_label in [(PM_RESID, "P-M residual"), (ME_RESID, "M-E residual")]:
            col = f"{term}_x_sub_{suffix}"
            out[col] = out[term].astype(float) * dummy
            interaction_cols.append(col)
            interaction_labels[col] = f"{short_label} x {subfield}"
    return out, interaction_cols, reference, interaction_labels


def model_specs(interaction_cols: list[str]) -> dict[str, list[str]]:
    m3 = CORE_CONTROLS + [P_NOV, M_NOV, E_NOV]
    m4 = m3 + [PM_RESID, ME_RESID]
    m5 = m4 + [ANCHOR, PM_ANCHOR, ME_ANCHOR, "anchoring_missing"]
    return {
        "M0": [],
        "M1": CORE_CONTROLS,
        "M2": CORE_CONTROLS + [OVERALL],
        "M3": m3,
        "M4": m4,
        "M5": m5,
        "M6": m5 + interaction_cols,
    }


def drop_bad_terms(df: pd.DataFrame, terms: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for term in terms:
        if term not in df.columns:
            dropped.append(f"{term}:missing")
            continue
        values = pd.to_numeric(df[term], errors="coerce")
        if values.notna().sum() == 0 or values.nunique(dropna=True) <= 1:
            dropped.append(f"{term}:constant")
        else:
            kept.append(term)
    return kept, dropped


def build_x(df: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if terms:
        pieces.append(df[terms].apply(pd.to_numeric, errors="coerce").astype(float))
    pieces.append(pd.get_dummies(df["grant_year"].astype(str), prefix="year", drop_first=True, dtype=float))
    pieces.append(pd.get_dummies(df["technical_subfield"].astype(str), prefix="subfield", drop_first=True, dtype=float))
    return sm.add_constant(pd.concat(pieces, axis=1), has_constant="add")


def rank_info(x: np.ndarray) -> dict[str, Any]:
    singular = np.linalg.svd(x, compute_uv=False)
    tolerance = max(x.shape) * np.finfo(float).eps * singular[0]
    rank = int((singular > tolerance).sum())
    return {
        "columns": int(x.shape[1]),
        "rank": rank,
        "deficiency": int(x.shape[1] - rank),
        "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf,
    }


def fit_one(
    df: pd.DataFrame,
    outcome: str,
    family: str,
    model_name: str,
    terms: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], Any, list[str]]:
    need = [outcome, "grant_year", "technical_subfield", "primary_assignee_id"] + terms
    d = df[need].replace([np.inf, -np.inf], np.nan).dropna().copy()
    kept, dropped = drop_bad_terms(d, terms)
    xdf = build_x(d, kept)
    y = pd.to_numeric(d[outcome], errors="coerce").astype(float)
    groups = d["primary_assignee_id"].fillna("__MISSING_ASSIGNEE__").astype(str)
    rank = rank_info(xdf.to_numpy(float))

    if family == "poisson":
        result = sm.GLM(y, xdf, family=sm.families.Poisson()).fit(
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": groups},
        )
        fit_stat = 1.0 - result.deviance / result.null_deviance if result.null_deviance else np.nan
        converged = bool(result.converged)
        dispersion = float(np.sum(result.resid_pearson**2) / result.df_resid)
    elif family == "ols":
        result = sm.OLS(y, xdf).fit(cov_type="cluster", cov_kwds={"groups": groups})
        fit_stat = float(result.rsquared)
        converged = True
        dispersion = None
    else:
        raise ValueError(f"Unsupported family: {family}")

    rows = []
    for term in kept:
        rows.append(
            {
                "outcome": outcome,
                "family": family,
                "model": model_name,
                "term": term,
                "label": LABELS.get(term, term),
                "coef": float(result.params[term]),
                "std_err": float(result.bse[term]),
                "z_or_t": float(result.tvalues[term]),
                "p_value": float(result.pvalues[term]),
                "stars": star(float(result.pvalues[term])),
            }
        )
    meta = {
        "outcome": outcome,
        "family": family,
        "model": model_name,
        "nobs": int(result.nobs),
        "fit_stat": float(fit_stat) if np.isfinite(fit_stat) else None,
        "aic": float(result.aic) if np.isfinite(result.aic) else None,
        "cov_type": "cluster",
        "cluster_variable": "primary_assignee_id",
        "cluster_count": int(groups.nunique()),
        "converged": converged,
        "pearson_dispersion": dispersion,
        "requested_terms": terms,
        "kept_terms": kept,
        "dropped_terms": dropped,
        "rank": rank,
    }
    return pd.DataFrame(rows), meta, result, kept


def vif_table(df: pd.DataFrame, terms: list[str], label: str) -> list[dict[str, Any]]:
    matrix = df[terms].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    corr = matrix.corr().to_numpy(float)
    inverse = np.linalg.pinv(corr)
    return [
        {"block": label, "term": term, "vif": float(inverse[i, i])}
        for i, term in enumerate(terms)
    ]


def subfield_marginal_effects(
    outcome: str,
    result: Any,
    reference: str,
    subfields: list[str],
) -> list[dict[str, Any]]:
    covariance = result.cov_params()
    rows: list[dict[str, Any]] = []
    for term, label in [(PM_RESID, "P-M residual configuration novelty"), (ME_RESID, "M-E residual configuration novelty")]:
        for subfield in subfields:
            estimate = float(result.params[term])
            variance = float(covariance.loc[term, term])
            interaction = None
            if subfield != reference:
                interaction = f"{term}_x_sub_{safe_name(subfield)}"
                estimate += float(result.params[interaction])
                variance += float(covariance.loc[interaction, interaction]) + 2.0 * float(covariance.loc[term, interaction])
            std_err = math.sqrt(max(variance, 0.0))
            statistic = estimate / std_err if std_err > 0 else np.nan
            p_value = float(2.0 * stats.norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan
            rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "label": label,
                    "technical_subfield": subfield,
                    "reference_subfield": reference,
                    "coef": estimate,
                    "std_err": std_err,
                    "z_value": statistic,
                    "p_value": p_value,
                    "stars": star(p_value),
                }
            )
    return rows


def joint_wald_tests(outcome: str, result: Any, interaction_cols: list[str]) -> list[dict[str, Any]]:
    groups = {
        "P-M residual x subfield": [term for term in interaction_cols if term.startswith(PM_RESID)],
        "M-E residual x subfield": [term for term in interaction_cols if term.startswith(ME_RESID)],
        "All residual configuration x subfield": interaction_cols,
    }
    names = list(result.params.index)
    rows: list[dict[str, Any]] = []
    for label, terms in groups.items():
        restriction = np.zeros((len(terms), len(names)), dtype=float)
        for row_index, term in enumerate(terms):
            restriction[row_index, names.index(term)] = 1.0
        test = result.wald_test(restriction, scalar=True)
        rows.append(
            {
                "outcome": outcome,
                "test": label,
                "restrictions": int(len(terms)),
                "statistic": float(np.asarray(test.statistic).squeeze()),
                "p_value": float(np.asarray(test.pvalue).squeeze()),
            }
        )
    return rows


def format_cell(coef_df: pd.DataFrame, outcome: str, model: str, term: str) -> str:
    row = coef_df.loc[
        coef_df["outcome"].eq(outcome) & coef_df["model"].eq(model) & coef_df["term"].eq(term)
    ]
    if row.empty:
        return ""
    item = row.iloc[0]
    return f"{item['coef']:.3f}{item['stars']}<br>({item['std_err']:.3f})"


def make_table(
    coef_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    outcome: str,
    title: str,
    interaction_cols: list[str],
    interaction_labels: dict[str, str],
) -> str:
    models = [f"M{i}" for i in range(7)]
    rows = [
        OVERALL,
        P_NOV,
        M_NOV,
        E_NOV,
        PM_RESID,
        ME_RESID,
        ANCHOR,
        PM_ANCHOR,
        ME_ANCHOR,
        "anchoring_missing",
    ] + interaction_cols + CORE_CONTROLS
    lines = [f"### {title}", "", "| Variable | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for term in rows:
        label = interaction_labels.get(term, LABELS.get(term, term))
        lines.append(f"| {label} | " + " | ".join(format_cell(coef_df, outcome, model, term) for model in models) + " |")

    subset = meta_df.loc[meta_df["outcome"].eq(outcome)].set_index("model").reindex(models)
    lines.append("| Grant-year FE | " + " | ".join(["Yes"] * len(models)) + " |")
    lines.append("| Technical-subfield FE | " + " | ".join(["Yes"] * len(models)) + " |")
    lines.append("| Configuration × subfield interactions | " + " | ".join(["No"] * 6 + ["Yes"]) + " |")
    lines.append("| Assignee-clustered robust SE | " + " | ".join(["Yes"] * len(models)) + " |")
    lines.append("| Observations | " + " | ".join(f"{int(value):,}" for value in subset["nobs"]) + " |")
    fit_label = "Pseudo R2" if outcome == "forward_cites_5y" else "R2"
    lines.append(f"| {fit_label} | " + " | ".join(f"{value:.4f}" for value in subset["fit_stat"]) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    for directory in [DATA, MODELS, REPORTS, QA]:
        directory.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("[V7] preparing panel")
    df, panel_audit = prepare_panel()
    df, interaction_cols, reference, interaction_labels = make_subfield_interactions(df)
    LABELS.update(interaction_labels)
    specs = model_specs(interaction_cols)

    coefficient_parts: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    marginal_rows: list[dict[str, Any]] = []
    wald_rows: list[dict[str, Any]] = []
    for outcome, family, _title in OUTCOMES:
        for model_name, terms in specs.items():
            print(f"[V7] fitting {outcome} {model_name}")
            coefs, meta, result, _kept = fit_one(df, outcome, family, model_name, terms)
            coefficient_parts.append(coefs)
            metadata.append(meta)
            if model_name == "M6":
                subfields = sorted(df["technical_subfield"].astype(str).unique())
                marginal_rows.extend(subfield_marginal_effects(outcome, result, reference, subfields))
                wald_rows.extend(joint_wald_tests(outcome, result, interaction_cols))

    coef_df = pd.concat(coefficient_parts, ignore_index=True)
    meta_df = pd.DataFrame(metadata)
    marginal_df = pd.DataFrame(marginal_rows)
    wald_df = pd.DataFrame(wald_rows)
    coef_df.to_csv(MODELS / "v7_m0_m6_coefficients.csv", index=False, encoding="utf-8-sig")
    meta_df.to_csv(MODELS / "v7_m0_m6_model_summary.csv", index=False, encoding="utf-8-sig")
    marginal_df.to_csv(MODELS / "v7_m6_subfield_marginal_effects.csv", index=False, encoding="utf-8-sig")
    wald_df.to_csv(MODELS / "v7_m6_joint_wald_tests.csv", index=False, encoding="utf-8-sig")
    write_json(MODELS / "v7_m0_m6_metadata.json", metadata)

    report_parts = [
        "# M0-M6 explanatory regression results",
        "",
        "Note: Cells report coefficients with primary_assignee_id-clustered robust standard errors in parentheses. +, *, **, and *** denote p < 0.10, 0.05, 0.01, and 0.001, respectively. All continuous novelty measures and controls are standardized as specified. M4-M6 use residualized configuration novelty only; M6 additionally includes interactions between the two residualized configuration measures and technical subfield.",
        "",
    ]
    for outcome, _family, title in OUTCOMES:
        report_parts.append(make_table(coef_df, meta_df, outcome, title, interaction_cols, interaction_labels))
    report = "\n".join(report_parts)
    (REPORTS / "v7_a61b_m0_m6_regression_tables.md").write_text(report, encoding="utf-8-sig")

    vif_rows = []
    vif_rows.extend(vif_table(df, [P_NOV, M_NOV, E_NOV, PM_RESID, ME_RESID], "M4 core novelty"))
    vif_rows.extend(vif_table(df, [P_NOV, M_NOV, E_NOV, PM_RESID, ME_RESID, ANCHOR, PM_ANCHOR, ME_ANCHOR], "M5 core and mechanism"))
    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(QA / "v7_vif.csv", index=False, encoding="utf-8-sig")

    all_finite = bool(np.isfinite(coef_df[["coef", "std_err", "p_value"]].to_numpy(float)).all())
    qa = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 2),
        "source_panel": str(SOURCE_PANEL),
        "rows": int(len(df)),
        "outcomes": {outcome: family for outcome, family, _ in OUTCOMES},
        "core_controls": CORE_CONTROLS,
        "m6_reference_subfield": reference,
        "m6_interaction_count": int(len(interaction_cols)),
        "all_models_converged": bool(meta_df["converged"].all()),
        "all_models_full_rank": bool(meta_df["rank"].map(lambda item: item["deficiency"] == 0).all()),
        "all_coefficients_finite": all_finite,
        "maximum_vif": float(vif_df["vif"].max()),
        "panel_audit": panel_audit,
        "outputs": {
            "panel_parquet": str(DATA / "a61b_v7_modeling_panel.parquet"),
            "panel_csv": str(DATA / "a61b_v7_modeling_panel.csv"),
            "coefficients": str(MODELS / "v7_m0_m6_coefficients.csv"),
            "model_summary": str(MODELS / "v7_m0_m6_model_summary.csv"),
            "m6_marginal_effects": str(MODELS / "v7_m6_subfield_marginal_effects.csv"),
            "m6_joint_wald_tests": str(MODELS / "v7_m6_joint_wald_tests.csv"),
            "report": str(REPORTS / "v7_a61b_m0_m6_regression_tables.md"),
        },
    }
    write_json(QA / "v7_run_qc.json", qa)
    print(json.dumps({
        "rows": qa["rows"],
        "all_models_converged": qa["all_models_converged"],
        "all_models_full_rank": qa["all_models_full_rank"],
        "maximum_vif": qa["maximum_vif"],
        "elapsed_seconds": qa["elapsed_seconds"],
        "report": qa["outputs"]["report"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
