from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
PANEL = WORK / "features" / "a61b_modeling_panel_v6_a61b_only.parquet"
OUT = WORK / "reduced_controls"
MODELS = OUT / "models"
REPORTS = OUT / "reports"
QA = OUT / "qa"

CORE_CONTROLS = [
    "log1p_backward_citation_count_z",
    "backward_citation_mean_age_z",
    "log1p_claim_count_z",
    "log1p_inventor_count_z",
    "P_missing",
    "E_missing",
]

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
    "pm_config_x_anchoring": "P-M novelty x anchoring",
    "me_config_x_anchoring": "M-E novelty x anchoring",
    "anchoring_missing": "Anchoring missing",
    "log1p_backward_citation_count_z": "ln(backward citations + 1)",
    "backward_citation_mean_age_z": "Backward citation mean age",
    "log1p_claim_count_z": "ln(claims + 1)",
    "log1p_inventor_count_z": "ln(inventors + 1)",
    "P_missing": "P missing",
    "E_missing": "E missing",
}

OUTCOMES = [
    ("forward_cites_5y", "poisson", "Table 4A Reduced: Five-year forward citations, PPML"),
    ("top10_impact", "logit", "Table 4B Reduced: Top 10% high-impact patents, Logit"),
]


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (Path, datetime)):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


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


def model_specs(interaction_cols: list[str]) -> dict[str, list[str]]:
    m2 = [CORE_TERMS["overall"]]
    m3 = m2 + [CORE_TERMS["P"], CORE_TERMS["M"], CORE_TERMS["E"]]
    m4 = m3 + [CORE_TERMS["PM"], CORE_TERMS["ME"]]
    m5 = m4 + [CORE_TERMS["anchor"], CORE_TERMS["PM_anchor"], CORE_TERMS["ME_anchor"]]
    return {
        "M0": [],
        "M1": CORE_CONTROLS,
        "M2": CORE_CONTROLS + m2,
        "M3": CORE_CONTROLS + m3,
        "M4": CORE_CONTROLS + m4,
        "M5": CORE_CONTROLS + ["anchoring_missing"] + m5,
        "M6": CORE_CONTROLS + ["anchoring_missing"] + m5 + interaction_cols,
    }


def make_m6_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    out = df.copy()
    subfields = sorted(out["technical_subfield"].dropna().astype(str).unique().tolist())
    reference = "A61B5" if "A61B5" in subfields else subfields[0]
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

    if family == "poisson":
        mod = sm.GLM(y, xdf, family=sm.families.Poisson())
    elif family == "logit":
        mod = sm.GLM(y, xdf, family=sm.families.Binomial())
    else:
        raise ValueError(f"Unsupported family: {family}")

    cov_type = "cluster"
    try:
        res = mod.fit(maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
    except Exception as exc:
        cov_type = "HC1"
        dropped_terms.append(f"cluster_cov_failed:{type(exc).__name__}")
        res = mod.fit(maxiter=200, cov_type="HC1")

    pseudo_r2 = 1.0 - (res.deviance / res.null_deviance) if res.null_deviance else np.nan
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
                "z_value": float(res.tvalues[term]),
                "p_value": p,
                "stars": stars(p),
            }
        )

    meta = {
        "outcome": outcome,
        "family": family,
        "model": model_name,
        "nobs": int(res.nobs),
        "pseudo_r2": float(pseudo_r2) if np.isfinite(pseudo_r2) else None,
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


def format_coef(coef: float, se: float, p: float) -> str:
    return f"{coef:.3f}{stars(p)}\n({se:.3f})"


def make_table(coef_df: pd.DataFrame, metas: list[dict[str, Any]], outcome: str, title: str) -> str:
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
        "log1p_inventor_count_z",
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

    meta_df = pd.DataFrame([m for m in metas if m["outcome"] == outcome]).sort_values("model")
    lines.append("| Grant-year FE | " + " | ".join(["Yes"] * len(models)) + " |")
    lines.append("| Technical-subfield FE | " + " | ".join(["Yes"] * len(models)) + " |")
    lines.append("| Subfield interactions | " + " | ".join(["No"] * 6 + ["Yes"]) + " |")
    lines.append("| Cluster-robust SE | " + " | ".join(meta_df["cov_type"].map(lambda x: "Assignee" if x == "cluster" else "HC1").tolist()) + " |")
    lines.append("| Observations | " + " | ".join(meta_df["nobs"].map(lambda x: f"{int(x):,}").tolist()) + " |")
    lines.append("| Pseudo R² | " + " | ".join(meta_df["pseudo_r2"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}").tolist()) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    for path in [OUT, MODELS, REPORTS, QA]:
        path.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PANEL)
    df, interaction_cols, reference = make_m6_interactions(df)
    specs = model_specs(interaction_cols)

    all_coef = []
    metas = []
    for outcome, family, _title in OUTCOMES:
        for model_name, terms in specs.items():
            print(f"[model] {outcome} {model_name} started")
            coefs, meta = fit_one(df, outcome, family, model_name, terms)
            all_coef.append(coefs)
            metas.append(meta)
            print(f"[model] {outcome} {model_name} done")

    coef_df = pd.concat(all_coef, ignore_index=True)
    coef_path = MODELS / "a61b_reduced_core_controls_coefficients.csv"
    meta_path = MODELS / "a61b_reduced_core_controls_metadata.json"
    coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")
    write_json(meta_path, metas)

    sections = [
        "# V6 A61B-only reduced core controls regression results",
        "",
        "Controls are limited to backward citation count, mean backward-citation age, claim count, inventor count, and P/E missingness; M5/M6 additionally retain anchoring_missing. All models include grant-year and technical-subfield fixed effects, with standard errors clustered by primary_assignee_id.",
        "",
    ]
    for outcome, _family, title in OUTCOMES:
        sections.append(make_table(coef_df, metas, outcome, title))
    report = "\n".join(sections)
    report_path = REPORTS / "a61b_reduced_core_controls_tables.md"
    report_path.write_text(report, encoding="utf-8")

    summary_rows = []
    for meta in metas:
        summary_rows.append(
            {
                "outcome": meta["outcome"],
                "model": meta["model"],
                "nobs": meta["nobs"],
                "pseudo_r2": meta["pseudo_r2"],
                "cov_type": meta["cov_type"],
                "rank_deficiency": meta["rank"]["deficiency"],
                "dropped_terms": ";".join(meta["dropped_terms"]),
            }
        )
    pd.DataFrame(summary_rows).to_csv(REPORTS / "a61b_reduced_core_controls_model_summary.csv", index=False, encoding="utf-8-sig")
    write_json(
        QA / "a61b_reduced_core_controls_qc.json",
        {
            "run_time": datetime.now().isoformat(timespec="seconds"),
            "source_panel": str(PANEL),
            "rows": int(len(df)),
            "core_controls": CORE_CONTROLS,
            "removed_controls": [
                "log1p_abstract_words_z",
                "log1p_claim_1_words_z",
                "log1p_cpc_count_z",
                "log1p_cpc_span_z",
                "log1p_assignee_count_z",
            ],
            "m6_reference_subfield": reference,
            "m6_interaction_terms": interaction_cols,
            "coefficients": str(coef_path),
            "metadata": str(meta_path),
            "report": str(report_path),
            "models": summary_rows,
        },
    )
    print("[done]", report_path)


if __name__ == "__main__":
    main()
