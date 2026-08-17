from __future__ import annotations

"""Rolling-origin, grant-time B0--B5 prediction experiment.

All transformations that learn from data (imputation, scaling, configuration
orthogonalization and penalty selection) are fit using a given training window
only.  Validation years choose a penalty; the held-out test year is untouched.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss,
    mean_absolute_error, mean_poisson_deviance, mean_squared_error, ndcg_score,
    r2_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "prediction"
PANEL = OUT / "data" / "a61b_oot_b0_b5_prediction_panel.parquet"
RESULTS = OUT / "results"
QA = OUT / "qa"

TEST_YEARS = list(range(2015, 2020))
SEMANTIC_TEST_YEARS = list(range(2015, 2018))
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

B1 = [
    "backward_citation_count", "backward_citation_mean_age", "claim_count", "independent_claim_count",
    "title_words", "abstract_words", "claim_1_words", "cpc_count", "cpc_span",
    "cpc_section_count", "cpc_class_count", "cpc_subclass_count", "cpc_group_count",
    "cpc_entropy", "cpc_rao_stirling", "cpc_disparity", "cpc_variety", "cpc_balance",
    "cpc_pair_atypicality", "grant_lag_days", "assignee_prior_patents", "assignee_prior_citations",
    "inventor_prior_productivity", "inventor_prior_citations", "inventor_career_age",
    "team_collaboration_novelty", "originality_cpc",
]




B2_ADD = ["overall_novelty_raw"]
B3_ADD = ["problem_novelty_raw", "mechanism_novelty_raw", "effect_novelty_raw", "P_missing", "M_missing", "E_missing"]


def feature_list(model: str) -> list[str]:
    if model == "B0":
        return ["grant_year", "technical_subfield"]
    fields = ["grant_year", "technical_subfield"] + B1
    if model in {"B2", "B3", "B4", "B5"}:
        fields += B2_ADD
    if model in {"B3", "B4", "B5"}:
        fields += B3_ADD
    if model in {"B4", "B5"}:
        fields += ["pm_config_orthogonal", "me_config_orthogonal"]
    if model == "B5":
        fields += ["anchoring_raw", "pm_orthogonal_x_anchoring"]
    return fields


def build_preprocessor(fields: list[str]) -> ColumnTransformer:
    cats = [x for x in fields if x == "technical_subfield"]
    nums = [x for x in fields if x not in cats]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), nums),
        ("subfield", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats),
    ], remainder="drop")


def orthogonalize(train: pd.DataFrame, *others: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Fit configuration projection in training data only, apply fixed weights elsewhere."""
    result = []
    specifications = [
        ("pm_config_novelty_raw", ["problem_novelty_raw", "mechanism_novelty_raw"], "pm_config_orthogonal"),
        ("me_config_novelty_raw", ["mechanism_novelty_raw", "effect_novelty_raw"], "me_config_orthogonal"),
    ]
    fitted = []
    for target, predictors, resid_name in specifications:
        imp = SimpleImputer(strategy="median")
        x = imp.fit_transform(train[predictors])
        y_imp = SimpleImputer(strategy="median")
        y = y_imp.fit_transform(train[[target]]).ravel()
        reg = LinearRegression().fit(x, y)
        fitted.append((predictors, resid_name, imp, y_imp, reg))
    anchor_fill = pd.to_numeric(train["anchoring_raw"], errors="coerce").median()
    for part in (train, *others):
        out = part.copy()
        for predictors, resid_name, x_imp, y_imp, reg in fitted:
            x = x_imp.transform(out[predictors])
            y = y_imp.transform(out[["pm_config_novelty_raw" if resid_name.startswith("pm") else "me_config_novelty_raw"]]).ravel()
            out[resid_name] = y - reg.predict(x)
        out["pm_orthogonal_x_anchoring"] = out["pm_config_orthogonal"] * pd.to_numeric(out["anchoring_raw"], errors="coerce").fillna(anchor_fill)
        result.append(out)
    return tuple(result)


def safe_spearman(y: np.ndarray, pred: np.ndarray) -> float:
    val = spearmanr(y, pred, nan_policy="omit").statistic
    return float(val) if np.isfinite(val) else np.nan


def metric(outcome_type: str, y: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(np.asarray(pred, dtype=float), 1e-9, None)
    if outcome_type == "count":
        return {"mae": float(mean_absolute_error(y, pred)), "poisson_deviance": float(mean_poisson_deviance(y, pred)), "spearman": safe_spearman(y, pred)}
    if outcome_type == "continuous":
        return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(mean_squared_error(y, pred) ** 0.5), "r2": float(r2_score(y, pred)), "spearman": safe_spearman(y, pred)}

    prevalence = max(1, int(np.sum(y)))
    top = np.argsort(-pred)[:prevalence]
    selected = np.zeros_like(y, dtype=bool); selected[top] = True
    ndcg = ndcg_score(y.reshape(1, -1), pred.reshape(1, -1), k=prevalence)
    return {"roc_auc": float(roc_auc_score(y, pred)), "pr_auc": float(average_precision_score(y, pred)), "brier": float(brier_score_loss(y, np.clip(pred, 1e-7, 1 - 1e-7))), "precision_at_k": float(y[selected].mean()), "recall_at_k": float(y[selected].sum() / max(1, y.sum())), "ndcg_at_k": float(ndcg), "prevalence": float(y.mean())}


def validation_score(outcome_type: str, y: np.ndarray, pred: np.ndarray) -> float:
    if outcome_type == "count":
        return -mean_poisson_deviance(y, np.clip(pred, 1e-9, None))
    if outcome_type == "continuous":
        return -mean_squared_error(y, pred)
    return average_precision_score(y, pred)


def fit_and_predict(outcome_type: str, alpha: float, fields: list[str], train: pd.DataFrame, target: str, target_part: pd.DataFrame) -> np.ndarray:
    pre = build_preprocessor(fields)
    x_train = pre.fit_transform(train[fields])
    x_part = pre.transform(target_part[fields])
    y = train[target].astype(float).to_numpy()
    if outcome_type == "count":
        model = PoissonRegressor(alpha=alpha, max_iter=1000, tol=1e-7)
    elif outcome_type == "continuous":
        model = Ridge(alpha=alpha, solver="lsqr")
    else:
        model = LogisticRegression(C=1.0 / alpha, penalty="l2", solver="lbfgs", max_iter=1000, n_jobs=1)
    model.fit(x_train, y)
    if outcome_type == "binary":
        return model.predict_proba(x_part)[:, 1]
    return model.predict(x_part)


def choose_alpha(outcome_type: str, fields: list[str], train: pd.DataFrame, valid: pd.DataFrame, target: str) -> tuple[float, list[dict]]:
    records = []
    yv = valid[target].astype(float).to_numpy()
    for alpha in ALPHAS:
        pred = fit_and_predict(outcome_type, alpha, fields, train, target, valid)
        records.append({"alpha": alpha, "validation_score": float(validation_score(outcome_type, yv, pred))})
    return max(records, key=lambda x: x["validation_score"])["alpha"], records


def correlation_diagnostic(panel: pd.DataFrame) -> None:
    new = ["problem_novelty_raw", "mechanism_novelty_raw", "effect_novelty_raw", "pm_config_novelty_raw", "me_config_novelty_raw", "anchoring_raw"]
    use = [x for x in (B1 + B2_ADD + new) if x in panel]
    corr = panel[use].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    corr.to_csv(RESULTS / "spearman_feature_correlation.csv", encoding="utf-8-sig")
    rows = []
    for n in new:
        for b in B1 + B2_ADD:
            value = corr.loc[n, b]
            rows.append({"new_feature": n, "benchmark_feature": b, "spearman_rho": value, "abs_rho": abs(value) if pd.notna(value) else np.nan})
    pd.DataFrame(rows).sort_values("abs_rho", ascending=False).to_csv(RESULTS / "new_vs_benchmark_spearman.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); QA.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL)
    correlation_diagnostic(panel)
    outcomes = [
        ("forward_cites_5y", "count", TEST_YEARS),
        ("outside_field_cites_5y_strict", "count", TEST_YEARS),
        ("semantic_reuse_5y_log1p", "continuous", SEMANTIC_TEST_YEARS),
        ("top1_impact", "binary", TEST_YEARS),
        ("top5_impact", "binary", TEST_YEARS),
        ("top10_impact", "binary", TEST_YEARS),
    ]
    records, predictions, tuning = [], [], []
    models = ["B0", "B1", "B2", "B3", "B4", "B5"]
    for target, kind, years in outcomes:
        for test_year in years:


            raw_train = panel.loc[panel.grant_year.le(test_year - 3)].dropna(subset=[target]).copy()
            raw_valid = panel.loc[panel.grant_year.between(test_year - 2, test_year - 1)].dropna(subset=[target]).copy()
            raw_test = panel.loc[panel.grant_year.eq(test_year)].dropna(subset=[target]).copy()
            train, valid, test = orthogonalize(raw_train, raw_valid, raw_test)
            for model_name in models:
                fields = feature_list(model_name)
                alpha, trials = choose_alpha(kind, fields, train, valid, target)
                final_train = pd.concat([train, valid], ignore_index=True)

                final_train, final_test = orthogonalize(final_train, raw_test)
                pred = fit_and_predict(kind, alpha, fields, final_train, target, final_test)
                y = final_test[target].astype(float).to_numpy()
                values = metric(kind, y, pred)
                records.append({"outcome": target, "outcome_type": kind, "test_year": test_year, "model": model_name, "n_train": len(train), "n_valid": len(valid), "n_test": len(test), "alpha": alpha, **values})
                predictions.append(pd.DataFrame({"patent_id": final_test.patent_id.values, "grant_year": test_year, "outcome": target, "model": model_name, "y_true": y, "prediction": pred}))
                tuning.extend({"outcome": target, "test_year": test_year, "model": model_name, **trial} for trial in trials)
            print(f"finished outcome={target} test_year={test_year}", flush=True)
    pd.DataFrame(records).to_csv(RESULTS / "rolling_oot_metrics_by_year.csv", index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_parquet(RESULTS / "rolling_oot_predictions.parquet", index=False)
    pd.DataFrame(tuning).to_csv(RESULTS / "penalty_tuning.csv", index=False, encoding="utf-8-sig")
    metrics = pd.DataFrame(records)
    numeric = [c for c in metrics.columns if c not in {"outcome", "outcome_type", "test_year", "model", "alpha"}]
    summary = metrics.groupby(["outcome", "outcome_type", "model"], as_index=False)[numeric].mean()
    summary.to_csv(RESULTS / "rolling_oot_metrics_summary.csv", index=False, encoding="utf-8-sig")
    (QA / "oot_protocol.json").write_text(json.dumps({"test_years": TEST_YEARS, "semantic_test_years": SEMANTIC_TEST_YEARS, "alpha_grid": ALPHAS, "models": {x: feature_list(x) for x in models}, "rule": "For test year t: train through t-3; validation t-2..t-1; final model refit on train+validation. Imputation, scaling and residualization are refit per window."}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
