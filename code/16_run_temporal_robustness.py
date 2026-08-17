from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_config import config_path, load_config

WORK = ROOT / "work"
V6_PANEL = WORK / "features" / "a61b_modeling_panel_v6_a61b_only.parquet"
CROSS_FIELD = WORK / "diagnostics" / "cross_field_quick_check" / "cross_field_counts_by_patent.parquet"
SEMANTIC_REUSE = config_path(load_config(ROOT), "patentsview_dir") / "7Text" / "PatentSimilarityImportanceBreakthrough_forPost2022.parquet"
BASE_SCRIPT = ROOT / "code" / "15_run_temporal_holdout.py"
EXP = WORK / "temporal_robustness"
DATA_DIR = EXP / "data"
MODEL_DIR = EXP / "models"
REPORT_DIR = EXP / "reports"
QA_DIR = EXP / "qa"

CV_YEARS = [2011, 2012, 2013, 2014]
TEST_YEARS = [2015, 2016, 2017, 2018, 2019]
SEMANTIC_MAIN_TEST_YEARS = [2015, 2016, 2017]
ALPHA_GRID = [0.0, 0.0001, 0.001, 0.01, 0.1]
BOOTSTRAP_REPS = 500
RANDOM_SEED = 20260802


def load_base_module():
    spec = importlib.util.spec_from_file_location("temporal_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()

RAW_NOVELTY = {
    "overall": "overall_novelty_raw",
    "P": "problem_novelty_raw",
    "M": "mechanism_novelty_raw",
    "E": "effect_novelty_raw",
    "PM": "pm_config_novelty_raw",
    "ME": "me_config_novelty_raw",
}

OUTCOMES = {
    "forward_cites_5y": {"family": "poisson", "label": "Five-year forward citations", "test_years": TEST_YEARS},
    "top10_impact": {"family": "binomial", "label": "Top 10% impact", "test_years": TEST_YEARS},
    "top1_impact": {"family": "binomial", "label": "Top 1% impact", "test_years": TEST_YEARS},
    "outside_field_cites_5y_strict": {"family": "poisson", "label": "Strict five-year out-of-field citations", "test_years": TEST_YEARS},
    "semantic_reuse_5y_log1p": {"family": "gaussian", "label": "Five-year subsequent semantic reuse (log1p)", "test_years": SEMANTIC_MAIN_TEST_YEARS},
}

MODEL_LABELS = {
    "P0": "Time trend + technical subfield",
    "P1": "P0 + conventional controls",
    "P2": "P1+Overall novelty",
    "P3": "P2+P/M/E novelty",
    "P4": "P3 + raw P-M/M-E configurations",
    "P5": "P4 + anchoring and interactions",
    "P6": "P5 + configuration × technical subfield",
}

CORE_CONTROLS = [
    "log_backward_cites", "backward_age", "log_claims", "log_inventors", "P_missing", "E_missing"
]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=BASE.json_default), encoding="utf-8")


class PredictiveFeatureBuilder:
    """Train-only imputation and scaling for the nested predictive feature blocks."""

    def __init__(self) -> None:
        self.impute: dict[str, float] = {}
        self.mean: dict[str, float] = {}
        self.sd: dict[str, float] = {}
        self.subfields: list[str] = []
        self.reference = "A61B5"
        self.dummy_cols: list[str] = []
        self.subfield_interactions: list[str] = []

    @staticmethod
    def _derived_raw(df: pd.DataFrame, impute: dict[str, float]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["year_trend"] = pd.to_numeric(df["grant_year"], errors="raise").astype(float)
        out["log_backward_cites"] = np.log1p(pd.to_numeric(df["backward_citation_count"], errors="coerce").fillna(0).clip(lower=0))
        out["backward_age"] = pd.to_numeric(df["backward_citation_mean_age"], errors="coerce").fillna(impute["backward_citation_mean_age"])
        out["log_claims"] = np.log1p(pd.to_numeric(df["claim_count"], errors="coerce").fillna(0).clip(lower=0))
        out["log_inventors"] = np.log1p(pd.to_numeric(df["inventor_count"], errors="coerce").fillna(0).clip(lower=0))
        for short, source in RAW_NOVELTY.items():
            out[f"novelty_{short}"] = pd.to_numeric(df[source], errors="coerce").fillna(impute[source]).astype(float)
        anchor = pd.to_numeric(df["anchoring_raw"], errors="coerce")
        out["anchoring"] = anchor.fillna(impute["anchoring_raw"]).astype(float)
        return out

    def fit(self, df: pd.DataFrame) -> "PredictiveFeatureBuilder":
        impute_cols = list(RAW_NOVELTY.values()) + ["backward_citation_mean_age", "anchoring_raw"]
        self.impute = {
            col: float(pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).mean())
            for col in impute_cols
        }
        self.subfields = sorted(df["technical_subfield"].astype(str).unique())
        if self.reference not in self.subfields:
            self.reference = str(df["technical_subfield"].astype(str).value_counts().index[0])
        raw = self._derived_raw(df, self.impute)
        for col in raw.columns:
            mean = float(raw[col].mean())
            sd = float(raw[col].std(ddof=0))
            self.mean[col] = mean if np.isfinite(mean) else 0.0
            self.sd[col] = sd if np.isfinite(sd) and sd > 1e-12 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = self._derived_raw(df, self.impute)
        out = pd.DataFrame(index=df.index)
        for col in raw.columns:
            out[col] = (raw[col] - self.mean[col]) / self.sd[col]
        out["P_missing"] = pd.to_numeric(df["P_missing"], errors="coerce").fillna(0).astype(float)
        out["E_missing"] = pd.to_numeric(df["E_missing"], errors="coerce").fillna(0).astype(float)
        out["anchoring_missing"] = pd.to_numeric(df["anchoring_raw"], errors="coerce").isna().astype(float)
        out["PM_x_anchoring"] = out["novelty_PM"] * out["anchoring"]
        out["ME_x_anchoring"] = out["novelty_ME"] * out["anchoring"]

        subfield = df["technical_subfield"].astype(str)
        self.dummy_cols = []
        self.subfield_interactions = []
        for category in self.subfields:
            if category == self.reference:
                continue
            suffix = safe_name(category)
            dummy_col = f"subfield_{suffix}"
            dummy = subfield.eq(category).astype(float)
            out[dummy_col] = dummy
            self.dummy_cols.append(dummy_col)
            for config in ["novelty_PM", "novelty_ME"]:
                col = f"{config}_x_sub_{suffix}"
                out[col] = out[config] * dummy
                self.subfield_interactions.append(col)
        return out.astype(float)

    def specs(self) -> dict[str, list[str]]:
        p0 = ["year_trend"] + self.dummy_cols
        p1 = p0 + CORE_CONTROLS
        p2 = p1 + ["novelty_overall"]
        p3 = p2 + ["novelty_P", "novelty_M", "novelty_E"]
        p4 = p3 + ["novelty_PM", "novelty_ME"]
        p5 = p4 + ["anchoring", "PM_x_anchoring", "ME_x_anchoring", "anchoring_missing"]
        return {
            "P0": p0, "P1": p1, "P2": p2, "P3": p3,
            "P4": p4, "P5": p5, "P6": p5 + self.subfield_interactions,
        }

    def audit(self) -> dict:
        return {
            "subfields": self.subfields,
            "reference_subfield": self.reference,
            "imputation_means": self.impute,
            "feature_count_by_model": {model: len(cols) for model, cols in self.specs().items()},
        }


def add_constant(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = x.to_numpy(float) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=float)
    return np.column_stack([np.ones(len(values)), values])


def fit_ridge_glm(
    x: pd.DataFrame,
    y: np.ndarray,
    family: str,
    alpha: float,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    matrix = add_constant(x)
    y = np.asarray(y, dtype=float)
    n, p = matrix.shape
    penalty = np.r_[0.0, np.ones(p - 1)]

    if family == "gaussian":
        gram = matrix.T @ matrix / n
        rhs = matrix.T @ y / n
        beta = np.linalg.solve(gram + alpha * np.diag(penalty) + 1e-10 * np.eye(p), rhs)
        return beta, {"converged": True, "iterations": 1}

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = matrix @ beta
        if family == "poisson":
            eta = np.clip(eta, -20, 20)
            mu = np.exp(eta)
            loss = float(np.mean(mu - y * eta) + 0.5 * alpha * np.sum((penalty * beta) ** 2))
            grad = matrix.T @ (mu - y) / n + alpha * penalty * beta
        elif family == "binomial":
            prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            loss = float(np.mean(np.logaddexp(0.0, eta) - y * eta) + 0.5 * alpha * np.sum((penalty * beta) ** 2))
            grad = matrix.T @ (prob - y) / n + alpha * penalty * beta
        else:
            raise ValueError(family)
        return loss, grad

    initial = np.zeros(p) if start is None or len(start) != p else start
    result = optimize.minimize(
        objective, initial, jac=True, method="L-BFGS-B",
        options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-6, "maxls": 50},
    )
    return result.x, {"converged": bool(result.success), "iterations": int(result.nit), "message": str(result.message)}


def predict_ridge(x: pd.DataFrame, beta: np.ndarray, family: str) -> np.ndarray:
    eta = add_constant(x) @ beta
    if family == "poisson":
        return np.exp(np.clip(eta, -20, 20))
    if family == "binomial":
        return np.clip(1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30))), 1e-8, 1 - 1e-8)
    return np.clip(eta, 0.0, None)


def evaluate(y: np.ndarray, pred: np.ndarray, family: str) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    if family == "poisson":
        return {
            "poisson_deviance": BASE.poisson_deviance(y, pred),
            "mae": float(np.mean(np.abs(y - pred))),
            "spearman": float(stats.spearmanr(y, pred).statistic),
        }
    if family == "binomial":
        slope, intercept = BASE.calibration_slope(y, pred)
        return {
            "pr_auc": BASE.weighted_average_precision(y, pred),
            "roc_auc": BASE.roc_auc(y, pred),
            "brier": float(np.mean((y - pred) ** 2)),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
        }
    residual = y - pred
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": float(stats.spearmanr(y, pred).statistic),
        "r2": float(1 - np.sum(residual**2) / denominator) if denominator > 0 else math.nan,
    }


def primary_metric(y: np.ndarray, pred: np.ndarray, family: str, weights: np.ndarray | None = None) -> float:
    if family == "poisson":
        return BASE.poisson_deviance(y, pred, weights)
    if family == "binomial":
        return BASE.weighted_average_precision(y, pred, weights)
    return float(np.average(np.abs(y - pred), weights=weights))


def better_is_higher(family: str) -> bool:
    return family == "binomial"


def load_panel() -> pd.DataFrame:
    columns = [
        "patent_id", "grant_year", "technical_subfield", "forward_cites_5y", "top10_impact", "top1_impact",
        "backward_citation_count", "backward_citation_mean_age", "claim_count", "inventor_count",
        "P_missing", "E_missing", "anchoring_raw",
    ] + list(RAW_NOVELTY.values())
    df = pd.read_parquet(V6_PANEL, columns=list(dict.fromkeys(columns)))
    df["patent_id"] = df["patent_id"].astype(str)
    cross = pd.read_parquet(CROSS_FIELD, columns=["patent_id", "outside_field_cites_5y_strict"])
    cross["patent_id"] = cross["patent_id"].astype(str)
    reuse = pd.read_parquet(SEMANTIC_REUSE, columns=["patent_num", "fsim01", "fsim25"])
    reuse["patent_id"] = reuse["patent_num"].astype(str)
    reuse["semantic_reuse_5y_raw"] = pd.to_numeric(reuse["fsim01"], errors="coerce") + pd.to_numeric(reuse["fsim25"], errors="coerce")
    reuse["semantic_reuse_5y_log1p"] = np.log1p(reuse["semantic_reuse_5y_raw"])
    df = df.merge(cross, on="patent_id", how="left", validate="one_to_one")
    df = df.merge(reuse[["patent_id", "semantic_reuse_5y_raw", "semantic_reuse_5y_log1p"]], on="patent_id", how="left", validate="one_to_one")
    df["outside_field_cites_5y_strict"] = df["outside_field_cites_5y_strict"].fillna(0).astype(int)
    return df


def run_cross_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    fit_records = []
    for val_year in CV_YEARS:
        fit = df.loc[df["grant_year"].between(2005, val_year - 1)].copy()
        valid = df.loc[df["grant_year"].eq(val_year)].copy()
        builder = PredictiveFeatureBuilder().fit(fit)
        x_fit = builder.transform(fit)
        x_valid = builder.transform(valid)
        specs = builder.specs()
        print(f"Rolling validation year {val_year}: train={len(fit):,}, validation={len(valid):,}", flush=True)
        for outcome, config in OUTCOMES.items():
            family = config["family"]
            fit_mask = fit[outcome].notna().to_numpy()
            val_mask = valid[outcome].notna().to_numpy()
            y_fit = fit.loc[fit_mask, outcome].to_numpy(float)
            y_valid = valid.loc[val_mask, outcome].to_numpy(float)
            for model, terms in specs.items():
                start = None
                for alpha in ALPHA_GRID:
                    beta, meta = fit_ridge_glm(x_fit.loc[fit_mask, terms], y_fit, family, alpha, start=start)
                    start = beta
                    pred = predict_ridge(x_valid.loc[val_mask, terms], beta, family)
                    metric = primary_metric(y_valid, pred, family)
                    records.append({
                        "validation_year": val_year, "outcome": outcome, "family": family,
                        "model": model, "alpha": alpha, "primary_metric": metric,
                        "n_train": int(np.sum(fit_mask)), "n_validation": int(np.sum(val_mask)),
                    })
                    fit_records.append({"stage": f"cv_{val_year}", "outcome": outcome, "model": model, "alpha": alpha, **meta})

    cv = pd.DataFrame(records)
    summaries = []
    for (outcome, family, model, alpha), group in cv.groupby(["outcome", "family", "model", "alpha"], sort=False):
        summaries.append({
            "outcome": outcome, "family": family, "model": model, "alpha": alpha,
            "mean_primary_metric": float(group["primary_metric"].mean()),
            "sd_primary_metric": float(group["primary_metric"].std()),
        })
    summary = pd.DataFrame(summaries)
    selected_rows = []
    for (outcome, family, model), group in summary.groupby(["outcome", "family", "model"], sort=False):
        ordered = group.sort_values(["mean_primary_metric", "alpha"], ascending=[not better_is_higher(family), True])
        selected_rows.append(ordered.iloc[0])
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    return cv, selected, pd.DataFrame(fit_records)


def fit_final_models(
    df: pd.DataFrame, selected: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]], pd.DataFrame, PredictiveFeatureBuilder]:
    train = df.loc[df["grant_year"].between(2005, 2014)].copy()
    builder = PredictiveFeatureBuilder().fit(train)
    x_train = builder.transform(train)
    metrics_rows = []
    yearly_rows = []
    fit_rows = []
    predictions: dict[str, dict[str, np.ndarray]] = {}

    for outcome, config in OUTCOMES.items():
        family = config["family"]
        test = df.loc[df["grant_year"].isin(config["test_years"])].copy()
        x_test = builder.transform(test)
        train_mask = train[outcome].notna().to_numpy()
        test_mask = test[outcome].notna().to_numpy()
        score_frame = test.loc[test_mask, ["patent_id", "grant_year", outcome]].copy()
        predictions[outcome] = {}
        for model, terms in builder.specs().items():
            alpha = float(selected.loc[selected["outcome"].eq(outcome) & selected["model"].eq(model), "alpha"].iloc[0])
            beta, meta = fit_ridge_glm(
                x_train.loc[train_mask, terms], train.loc[train_mask, outcome].to_numpy(float), family, alpha
            )
            pred = predict_ridge(x_test.loc[test_mask, terms], beta, family)
            predictions[outcome][model] = pred
            score_frame[model] = pred
            result = evaluate(test.loc[test_mask, outcome].to_numpy(float), pred, family)
            metrics_rows.append({
                "outcome": outcome, "outcome_label": config["label"], "family": family,
                "model": model, "model_label": MODEL_LABELS[model], "alpha": alpha,
                "n": int(np.sum(test_mask)), **result,
            })
            for year in sorted(test.loc[test_mask, "grant_year"].unique()):
                year_mask = test.loc[test_mask, "grant_year"].to_numpy(int) == year
                yearly_rows.append({
                    "year": int(year), "outcome": outcome, "model": model, "alpha": alpha,
                    "n": int(np.sum(year_mask)),
                    **evaluate(test.loc[test_mask, outcome].to_numpy(float)[year_mask], pred[year_mask], family),
                })
            fit_rows.append({"stage": "final", "outcome": outcome, "model": model, "alpha": alpha, **meta})
        score_frame.to_parquet(DATA_DIR / f"test_predictions_{outcome}.parquet", index=False)
        print(f"Final test predictions complete: {outcome}", flush=True)
    return pd.DataFrame(metrics_rows), pd.DataFrame(yearly_rows), predictions, pd.DataFrame(fit_rows), builder


def bootstrap_metrics(df: pd.DataFrame, predictions: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    sequence = list(MODEL_LABELS)
    rows = []
    for outcome, config in OUTCOMES.items():
        family = config["family"]
        test = df.loc[df["grant_year"].isin(config["test_years"]) & df[outcome].notna()].copy()
        y = test[outcome].to_numpy(float)
        years = test["grant_year"].to_numpy(int)
        unique_years = np.array(sorted(np.unique(years)))
        draw = rng.choice(unique_years, size=(BOOTSTRAP_REPS, len(unique_years)), replace=True)
        counts = np.zeros((BOOTSTRAP_REPS, len(unique_years)), dtype=float)
        for b, sampled in enumerate(draw):
            for year in sampled:
                counts[b, np.searchsorted(unique_years, year)] += 1
        boot = {model: np.zeros(BOOTSTRAP_REPS) for model in sequence}
        full = {}
        for model in sequence:
            pred = predictions[outcome][model]
            full[model] = primary_metric(y, pred, family)
            for b in range(BOOTSTRAP_REPS):
                weights = counts[b, np.searchsorted(unique_years, years)]
                boot[model][b] = primary_metric(y, pred, family, weights)
        for i, model in enumerate(sequence):
            row = {
                "outcome": outcome, "family": family, "model": model,
                "estimate": full[model],
                "ci_low": float(np.quantile(boot[model], 0.025)), "ci_high": float(np.quantile(boot[model], 0.975)),
            }
            for label, baseline in [
                ("previous", sequence[i - 1] if i else None),
                ("P1", "P1" if model != "P1" else None),
                ("P2", "P2" if model not in {"P0", "P1", "P2"} else None),
            ]:
                if baseline is None:
                    continue
                if better_is_higher(family):
                    delta = full[model] - full[baseline]
                    delta_boot = boot[model] - boot[baseline]
                else:
                    delta = full[baseline] - full[model]
                    delta_boot = boot[baseline] - boot[model]
                row[f"delta_vs_{label}"] = delta
                row[f"delta_vs_{label}_ci_low"] = float(np.quantile(delta_boot, 0.025))
                row[f"delta_vs_{label}_ci_high"] = float(np.quantile(delta_boot, 0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def build_report(metrics: pd.DataFrame, selected: pd.DataFrame, bootstrap: pd.DataFrame, coverage: pd.DataFrame) -> str:
    lines = [
        "# P0-P6 regularized temporal prediction results", "",
        "P0-P6 use truly nested feature blocks. Overall novelty is retained from P3 onward, P4 uses raw configuration measures, and the L2 penalty is selected through rolling validation over 2011–2014. The test period is 2015–2019 except for semantic reuse, whose main test period is 2015–2017 because 2018–2019 are right-censored.", "",
        "## Semantic-reuse coverage", "", BASE.markdown_table(coverage, 4), "",
        "## Alpha selected by rolling validation", "",
        BASE.markdown_table(selected[["outcome", "model", "alpha", "mean_primary_metric"]], 5),
    ]
    for outcome, config in OUTCOMES.items():
        family = config["family"]
        if family == "poisson":
            cols = ["model", "alpha", "poisson_deviance", "mae", "spearman"]
        elif family == "binomial":
            cols = ["model", "alpha", "pr_auc", "roc_auc", "brier", "calibration_slope"]
        else:
            cols = ["model", "alpha", "mae", "rmse", "spearman", "r2"]
        lines += ["", f"## {config['label']}", "", BASE.markdown_table(metrics.loc[metrics["outcome"].eq(outcome), cols], 5)]
        boot_cols = ["model", "estimate", "ci_low", "ci_high", "delta_vs_P1", "delta_vs_P1_ci_low", "delta_vs_P1_ci_high", "delta_vs_P2", "delta_vs_P2_ci_low", "delta_vs_P2_ci_high"]
        lines += ["", "Year-block bootstrap; positive gains indicate improvement over the corresponding baseline:", "", BASE.markdown_table(bootstrap.loc[bootstrap["outcome"].eq(outcome), boot_cols], 5)]
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.time()
    for path in [DATA_DIR, MODEL_DIR, REPORT_DIR, QA_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    coverage = df.groupby("grant_year").agg(
        n=("patent_id", "size"), semantic_observed=("semantic_reuse_5y_log1p", "count")
    ).reset_index()
    coverage["semantic_missing_rate"] = 1 - coverage["semantic_observed"] / coverage["n"]
    coverage = coverage.loc[coverage["grant_year"].between(2005, 2019)]
    coverage.to_csv(QA_DIR / "semantic_reuse_coverage.csv", index=False, encoding="utf-8-sig")

    print("Running rolling-year ridge selection...", flush=True)
    cv, selected, cv_fit = run_cross_validation(df)
    cv.to_csv(MODEL_DIR / "rolling_cv_metrics.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(MODEL_DIR / "selected_alpha.csv", index=False, encoding="utf-8-sig")

    print("Fitting frozen P0-P6 models on 2005-2014...", flush=True)
    metrics, yearly, predictions, final_fit, builder = fit_final_models(df, selected)
    print("Running year-block bootstrap...", flush=True)
    bootstrap = bootstrap_metrics(df, predictions)
    metrics.to_csv(MODEL_DIR / "p0_p6_test_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(MODEL_DIR / "p0_p6_yearly_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(MODEL_DIR / "p0_p6_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.concat([cv_fit, final_fit], ignore_index=True).to_csv(MODEL_DIR / "p0_p6_fit_meta.csv", index=False, encoding="utf-8-sig")

    qa = {
        "elapsed_seconds": round(time.time() - started, 2),
        "cv_years": CV_YEARS, "test_years": TEST_YEARS,
        "semantic_main_test_years": SEMANTIC_MAIN_TEST_YEARS,
        "alpha_grid": ALPHA_GRID, "bootstrap_reps": BOOTSTRAP_REPS,
        "rows": int(len(df)), "feature_audit": builder.audit(),
        "all_final_fits_converged": bool(final_fit["converged"].all()),
        "semantic_warning": "2018-2019 semantic reuse is right-censored; main semantic test uses 2015-2017.",
        "retrospective_warning": "Citation, Top10, Top1 and strict cross-field test outcomes were previously inspected. Semantic reuse outcome had not been modeled before this frozen run.",
    }
    write_json(QA_DIR / "p0_p6_qc.json", qa)
    report = build_report(metrics, selected, bootstrap, coverage)
    (REPORT_DIR / "p0_p6_results.md").write_text(report, encoding="utf-8")
    print(f"Completed in {qa['elapsed_seconds']} seconds", flush=True)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
