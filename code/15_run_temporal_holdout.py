from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
V6_PANEL = WORK / "features" / "a61b_modeling_panel_v6_a61b_only.parquet"
CROSS_FIELD = WORK / "diagnostics" / "cross_field_quick_check" / "cross_field_counts_by_patent.parquet"
EXP = WORK / "temporal_holdout"
DATA_DIR = EXP / "data"
MODEL_DIR = EXP / "models"
REPORT_DIR = EXP / "reports"
QA_DIR = EXP / "qa"

TRAIN_YEARS = range(2005, 2012)
VALID_YEARS = range(2012, 2015)
TEST_YEARS = range(2015, 2020)
BOOTSTRAP_REPS = 500
RANDOM_SEED = 20260802

RAW_NOVELTY = {
    "overall": "overall_novelty_raw",
    "P": "problem_novelty_raw",
    "M": "mechanism_novelty_raw",
    "E": "effect_novelty_raw",
    "PM": "pm_config_novelty_raw",
    "ME": "me_config_novelty_raw",
}

CORE_CONTROLS = [
    "log_backward_cites",
    "backward_age",
    "log_claims",
    "log_inventors",
    "P_missing",
    "E_missing",
]

OUTCOMES = {
    "forward_cites_5y": {"family": "poisson", "label": "Five-year forward citations"},
    "top10_impact": {"family": "binomial", "label": "Top 10% impact"},
    "top1_impact": {"family": "binomial", "label": "Top 1% impact"},
    "outside_field_cites_5y_strict": {"family": "poisson", "label": "Strict five-year out-of-field citations"},
}

MODEL_LABELS = {
    "M0": "Time trend + technical subfield",
    "M1": "M0 + conventional controls",
    "M2": "M1+Overall novelty",
    "M3": "M1+P/M/E novelty",
    "M4": "M3 + residualized configuration novelty",
    "M5": "M4 + anchoring and interactions",
    "M6": "M5 + configuration × technical subfield",
}


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def finite_mean(series: pd.Series, default: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = float(values.mean())
    return mean if np.isfinite(mean) else default


@dataclass
class ResidualRule:
    beta: np.ndarray
    mean: float
    sd: float
    target: str
    predictors: tuple[str, str]


class TemporalFeatureBuilder:
    """Fit all data transformations on a historical subset and apply them forward."""

    def __init__(self) -> None:
        self.impute: dict[str, float] = {}
        self.anchor_mean = 0.0
        self.pm_rule: ResidualRule | None = None
        self.me_rule: ResidualRule | None = None
        self.subfields: list[str] = []
        self.reference_subfield = "A61B5"
        self.scale_mean: dict[str, float] = {}
        self.scale_sd: dict[str, float] = {}
        self.scale_cols: list[str] = []
        self.subfield_dummy_cols: list[str] = []
        self.subfield_interaction_cols: list[str] = []

    @staticmethod
    def _fit_residual_rule(df: pd.DataFrame, target: str, p1: str, p2: str) -> ResidualRule:
        fit = df[[target, p1, p2]].apply(pd.to_numeric, errors="coerce").dropna()
        x = np.column_stack([np.ones(len(fit)), fit[p1].to_numpy(float), fit[p2].to_numpy(float)])
        beta = np.linalg.lstsq(x, fit[target].to_numpy(float), rcond=None)[0]
        residual = fit[target].to_numpy(float) - x @ beta
        mean = float(np.mean(residual))
        sd = float(np.std(residual, ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Invalid residual SD for {target}: {sd}")
        return ResidualRule(beta=beta, mean=mean, sd=sd, target=target, predictors=(p1, p2))

    def fit(self, df: pd.DataFrame) -> "TemporalFeatureBuilder":
        impute_cols = list(RAW_NOVELTY.values()) + ["backward_citation_mean_age", "anchoring_raw"]
        self.impute = {col: finite_mean(df[col]) for col in impute_cols}
        self.anchor_mean = self.impute["anchoring_raw"]
        self.pm_rule = self._fit_residual_rule(
            df, RAW_NOVELTY["PM"], RAW_NOVELTY["P"], RAW_NOVELTY["M"]
        )
        self.me_rule = self._fit_residual_rule(
            df, RAW_NOVELTY["ME"], RAW_NOVELTY["M"], RAW_NOVELTY["E"]
        )
        self.subfields = sorted(df["technical_subfield"].astype(str).unique())
        if self.reference_subfield not in self.subfields:
            self.reference_subfield = df["technical_subfield"].astype(str).value_counts().index[0]

        raw_features = self._transform_unscaled(df)
        binary_cols = {"P_missing", "E_missing", "anchoring_missing"} | set(self.subfield_dummy_cols)
        self.scale_cols = [col for col in raw_features.columns if col not in binary_cols]
        for col in self.scale_cols:
            mean = float(raw_features[col].mean())
            sd = float(raw_features[col].std(ddof=0))
            self.scale_mean[col] = mean if np.isfinite(mean) else 0.0
            self.scale_sd[col] = sd if np.isfinite(sd) and sd > 1e-12 else 1.0
        return self

    def _residual(self, df: pd.DataFrame, rule: ResidualRule) -> np.ndarray:
        p1, p2 = rule.predictors
        x1 = pd.to_numeric(df[p1], errors="coerce").fillna(self.impute[p1]).to_numpy(float)
        x2 = pd.to_numeric(df[p2], errors="coerce").fillna(self.impute[p2]).to_numpy(float)
        target = pd.to_numeric(df[rule.target], errors="coerce")
        fitted = rule.beta[0] + rule.beta[1] * x1 + rule.beta[2] * x2
        residual = (target.fillna(0.0).to_numpy(float) - fitted - rule.mean) / rule.sd
        residual[target.isna().to_numpy()] = 0.0
        return residual

    def _transform_unscaled(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pm_rule is None or self.me_rule is None:
            raise RuntimeError("Feature builder has not been fitted")
        out = pd.DataFrame(index=df.index)
        out["year_trend"] = pd.to_numeric(df["grant_year"], errors="raise").astype(float)
        out["log_backward_cites"] = np.log1p(pd.to_numeric(df["backward_citation_count"], errors="coerce").fillna(0).clip(lower=0))
        out["backward_age"] = pd.to_numeric(df["backward_citation_mean_age"], errors="coerce").fillna(self.impute["backward_citation_mean_age"])
        out["log_claims"] = np.log1p(pd.to_numeric(df["claim_count"], errors="coerce").fillna(0).clip(lower=0))
        out["log_inventors"] = np.log1p(pd.to_numeric(df["inventor_count"], errors="coerce").fillna(0).clip(lower=0))
        out["P_missing"] = pd.to_numeric(df["P_missing"], errors="coerce").fillna(0).astype(float)
        out["E_missing"] = pd.to_numeric(df["E_missing"], errors="coerce").fillna(0).astype(float)

        for short in ["overall", "P", "M", "E"]:
            source = RAW_NOVELTY[short]
            out[f"novelty_{short}"] = pd.to_numeric(df[source], errors="coerce").fillna(self.impute[source]).astype(float)

        out["pm_residual"] = self._residual(df, self.pm_rule)
        out["me_residual"] = self._residual(df, self.me_rule)
        anchor_raw = pd.to_numeric(df["anchoring_raw"], errors="coerce")
        out["anchoring"] = anchor_raw.fillna(self.anchor_mean).to_numpy(float) - self.anchor_mean
        out["anchoring_missing"] = anchor_raw.isna().astype(float)
        out["pm_x_anchoring"] = out["pm_residual"] * out["anchoring"]
        out["me_x_anchoring"] = out["me_residual"] * out["anchoring"]

        subfield = df["technical_subfield"].astype(str)
        self.subfield_dummy_cols = []
        self.subfield_interaction_cols = []
        for category in self.subfields:
            if category == self.reference_subfield:
                continue
            suffix = safe_name(category)
            dummy_col = f"subfield_{suffix}"
            dummy = subfield.eq(category).astype(float)
            out[dummy_col] = dummy
            self.subfield_dummy_cols.append(dummy_col)
            for residual in ["pm_residual", "me_residual"]:
                col = f"{residual}_x_sub_{suffix}"
                out[col] = out[residual] * dummy
                self.subfield_interaction_cols.append(col)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._transform_unscaled(df)
        for col in self.scale_cols:
            out[col] = (out[col] - self.scale_mean[col]) / self.scale_sd[col]
        return out.astype(float)

    def specs(self) -> dict[str, list[str]]:
        base = ["year_trend"] + self.subfield_dummy_cols
        m1 = base + CORE_CONTROLS
        m3 = m1 + ["novelty_P", "novelty_M", "novelty_E"]
        m4 = m3 + ["pm_residual", "me_residual"]
        m5 = m4 + ["anchoring", "pm_x_anchoring", "me_x_anchoring", "anchoring_missing"]
        return {
            "M0": base,
            "M1": m1,
            "M2": m1 + ["novelty_overall"],
            "M3": m3,
            "M4": m4,
            "M5": m5,
            "M6": m5 + self.subfield_interaction_cols,
        }

    def audit(self) -> dict:
        return {
            "subfields": self.subfields,
            "reference_subfield": self.reference_subfield,
            "imputation_means": self.impute,
            "pm_residual_beta": self.pm_rule.beta if self.pm_rule else None,
            "me_residual_beta": self.me_rule.beta if self.me_rule else None,
            "pm_residual_sd": self.pm_rule.sd if self.pm_rule else None,
            "me_residual_sd": self.me_rule.sd if self.me_rule else None,
            "feature_count_by_model": {key: len(value) for key, value in self.specs().items()},
        }


def poisson_deviance(y: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(pred, dtype=float), 1e-12, None)
    positive = y > 0
    terms = mu.copy()
    terms[positive] = y[positive] * np.log(y[positive] / mu[positive]) - y[positive] + mu[positive]
    return float(2.0 * np.average(terms, weights=weights))


def weighted_average_precision(y: np.ndarray, score: np.ndarray, weights: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y, score, w = y[order], score[order], w[order]
    tp = np.cumsum(w * y)
    total = np.cumsum(w)
    distinct_ends = np.r_[np.flatnonzero(score[1:] != score[:-1]), len(score) - 1]
    tp_at = tp[distinct_ends]
    total_at = total[distinct_ends]
    positives = float(np.sum(w * y))
    if positives <= 0:
        return math.nan
    recall = tp_at / positives
    precision = np.divide(tp_at, total_at, out=np.zeros_like(tp_at), where=total_at > 0)
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = stats.rankdata(score, method="average")
    return float((np.sum(ranks[y == 1]) - positives * (positives + 1) / 2) / (positives * negatives))


def calibration_slope(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(pred, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    x = sm.add_constant(logit, has_constant="add")
    try:
        result = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=100, disp=0)
        return float(result.params[1]), float(result.params[0])
    except Exception:
        return math.nan, math.nan


def evaluate(y: np.ndarray, pred: np.ndarray, family: str) -> dict[str, float]:
    if family == "poisson":
        return {
            "poisson_deviance": poisson_deviance(y, pred),
            "mae": float(np.mean(np.abs(y - pred))),
            "spearman": float(stats.spearmanr(y, pred).statistic),
        }
    slope, intercept = calibration_slope(y, pred)
    return {
        "pr_auc": weighted_average_precision(y, pred),
        "roc_auc": roc_auc(y, pred),
        "brier": float(np.mean((y - pred) ** 2)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def fit_predict(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_score: pd.DataFrame,
    family: str,
) -> tuple[np.ndarray, dict]:
    x1 = sm.add_constant(x_fit, has_constant="add").to_numpy(float)
    x2 = sm.add_constant(x_score, has_constant="add").to_numpy(float)
    glm_family = sm.families.Poisson() if family == "poisson" else sm.families.Binomial()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sm.GLM(y_fit, x1, family=glm_family).fit(maxiter=200, disp=0)
    pred = np.asarray(result.predict(x2), dtype=float)
    if family == "poisson":
        pred = np.clip(pred, 1e-12, None)
    else:
        pred = np.clip(pred, 1e-8, 1 - 1e-8)
    meta = {
        "converged": bool(result.converged),
        "iterations": int(result.fit_history.get("iteration", -1)),
        "nobs": int(result.nobs),
        "parameters": int(len(result.params)),
        "finite_predictions": bool(np.isfinite(pred).all()),
    }
    return pred, meta


def load_panel() -> pd.DataFrame:
    columns = [
        "patent_id", "grant_year", "technical_subfield", "forward_cites_5y",
        "top10_impact", "top1_impact", "backward_citation_count",
        "backward_citation_mean_age", "claim_count", "inventor_count",
        "P_missing", "E_missing", "anchoring_raw",
    ] + list(RAW_NOVELTY.values())
    df = pd.read_parquet(V6_PANEL, columns=list(dict.fromkeys(columns)))
    df["patent_id"] = df["patent_id"].astype(str)
    cross = pd.read_parquet(CROSS_FIELD, columns=["patent_id", "outside_field_cites_5y_strict"])
    cross["patent_id"] = cross["patent_id"].astype(str)
    df = df.merge(cross, on="patent_id", how="left", validate="one_to_one")
    df["outside_field_cites_5y_strict"] = df["outside_field_cites_5y_strict"].fillna(0).astype(int)
    for col in ["top10_impact", "top1_impact"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    if df["patent_id"].duplicated().any():
        raise ValueError("Duplicate patent IDs")
    return df


def metric_rows(
    split_name: str,
    frame: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
) -> list[dict]:
    rows = []
    for outcome, config in OUTCOMES.items():
        y = frame[outcome].to_numpy(float)
        for model, pred in predictions[outcome].items():
            metrics = evaluate(y, pred, config["family"])
            rows.append({
                "split": split_name,
                "outcome": outcome,
                "outcome_label": config["label"],
                "model": model,
                "model_label": MODEL_LABELS[model],
                "n": len(frame),
                **metrics,
            })
    return rows


def yearly_metric_rows(test: pd.DataFrame, predictions: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rows = []
    years = test["grant_year"].to_numpy(int)
    for year in sorted(np.unique(years)):
        mask = years == year
        for outcome, config in OUTCOMES.items():
            y = test[outcome].to_numpy(float)[mask]
            for model, pred in predictions[outcome].items():
                rows.append({
                    "year": int(year), "outcome": outcome, "model": model,
                    "n": int(np.sum(mask)), **evaluate(y, pred[mask], config["family"]),
                })
    return rows


def primary_metric(family: str) -> tuple[str, Callable]:
    if family == "poisson":
        return "poisson_deviance", poisson_deviance
    return "pr_auc", weighted_average_precision


def bootstrap_primary_metrics(
    test: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    years = test["grant_year"].to_numpy(int)
    unique_years = np.array(sorted(np.unique(years)))
    draws = rng.choice(unique_years, size=(BOOTSTRAP_REPS, len(unique_years)), replace=True)
    year_positions = {year: i for i, year in enumerate(unique_years)}
    count_matrix = np.zeros((BOOTSTRAP_REPS, len(unique_years)), dtype=float)
    for b, sample in enumerate(draws):
        for year in sample:
            count_matrix[b, year_positions[year]] += 1.0

    rows = []
    sequence = list(MODEL_LABELS)
    for outcome, config in OUTCOMES.items():
        family = config["family"]
        metric_name, metric_func = primary_metric(family)
        y = test[outcome].to_numpy(float)
        full = {}
        boot = {model: np.zeros(BOOTSTRAP_REPS) for model in sequence}
        for model in sequence:
            pred = predictions[outcome][model]
            full[model] = metric_func(y, pred)
            for b in range(BOOTSTRAP_REPS):
                weights = count_matrix[b, np.searchsorted(unique_years, years)]
                boot[model][b] = metric_func(y, pred, weights)

        for i, model in enumerate(sequence):
            values = boot[model]
            row = {
                "outcome": outcome,
                "metric": metric_name,
                "model": model,
                "estimate": full[model],
                "ci_low": float(np.nanquantile(values, 0.025)),
                "ci_high": float(np.nanquantile(values, 0.975)),
            }
            for baseline_name, baseline_model in [
                ("previous", sequence[i - 1] if i > 0 else None),
                ("M1", "M1" if model != "M1" else None),
                ("M2", "M2" if model not in {"M0", "M1", "M2"} else None),
            ]:
                if baseline_model is None:
                    continue
                if family == "poisson":
                    delta = full[baseline_model] - full[model]
                    delta_boot = boot[baseline_model] - values
                else:
                    delta = full[model] - full[baseline_model]
                    delta_boot = values - boot[baseline_model]
                row[f"delta_vs_{baseline_name}"] = delta
                row[f"delta_vs_{baseline_name}_ci_low"] = float(np.nanquantile(delta_boot, 0.025))
                row[f"delta_vs_{baseline_name}_ci_high"] = float(np.nanquantile(delta_boot, 0.975))
            rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    data = df.copy()
    for col in data.select_dtypes(include=[np.number]).columns:
        data[col] = data[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    header = "| " + " | ".join(map(str, data.columns)) + " |"
    rule = "|" + "|".join(["---"] * len(data.columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in data.itertuples(index=False, name=None)]
    return "\n".join([header, rule] + rows)


def build_report(metrics: pd.DataFrame, bootstrap: pd.DataFrame, yearly: pd.DataFrame, split_summary: dict) -> str:
    lines = [
        "# Temporal holdout prediction experiment",
        "",
        "## Design",
        "",
        "The training period is 2005–2011, the validation period is 2012–2014, and the fixed test period is 2015–2019. All imputation, standardization, and configuration residualization are fitted only within the corresponding training period. Final models are refitted on 2005–2014 before a single prediction pass over the test period.",
        "",
        "Grant-year fixed effects have no estimable coefficients for future years, so prediction models use a continuous grant-year trend while retaining technical-subfield fixed-effect encoding. Because 2015–2019 were previously included in full-sample explanatory regressions, this is a retrospective temporal holdout rather than a final untouched confirmation set.",
        "",
        "## Sample",
        "",
        markdown_table(pd.DataFrame(split_summary).T.reset_index().rename(columns={"index": "split"}), 4),
    ]
    test_metrics = metrics.loc[metrics["split"].eq("test")]
    for outcome, config in OUTCOMES.items():
        family = config["family"]
        cols = ["model", "model_label"] + (["poisson_deviance", "mae", "spearman"] if family == "poisson" else ["pr_auc", "roc_auc", "brier", "calibration_slope"])
        lines += ["", f"## {config['label']}", "", markdown_table(test_metrics.loc[test_metrics["outcome"].eq(outcome), cols], 4)]
        b = bootstrap.loc[bootstrap["outcome"].eq(outcome), ["model", "metric", "estimate", "ci_low", "ci_high", "delta_vs_M1", "delta_vs_M1_ci_low", "delta_vs_M1_ci_high", "delta_vs_M2", "delta_vs_M2_ci_low", "delta_vs_M2_ci_high"]]
        lines += ["", "Year-block bootstrap (500 draws; positive gains indicate improvement over the baseline):", "", markdown_table(b, 5)]

    key_yearly = yearly.loc[yearly["model"].isin(["M1", "M2", "M4", "M5", "M6"])].copy()
    key_yearly["primary_metric"] = np.where(
        key_yearly["outcome"].isin(["forward_cites_5y", "outside_field_cites_5y_strict"]),
        key_yearly["poisson_deviance"], key_yearly["pr_auc"],
    )
    lines += ["", "## Performance by year", "", markdown_table(key_yearly[["year", "outcome", "model", "n", "primary_metric"]], 5)]
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.time()
    for path in [DATA_DIR, MODEL_DIR, REPORT_DIR, QA_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    df = load_panel()
    train = df.loc[df["grant_year"].isin(TRAIN_YEARS)].copy()
    valid = df.loc[df["grant_year"].isin(VALID_YEARS)].copy()
    train_valid = df.loc[df["grant_year"].isin(list(TRAIN_YEARS) + list(VALID_YEARS))].copy()
    test = df.loc[df["grant_year"].isin(TEST_YEARS)].copy()

    split_summary = {}
    for name, part in [("train", train), ("validation", valid), ("train_validation", train_valid), ("test", test)]:
        split_summary[name] = {
            "n": int(len(part)), "year_min": int(part["grant_year"].min()), "year_max": int(part["grant_year"].max()),
            "mean_cites": float(part["forward_cites_5y"].mean()),
            "top10_rate": float(part["top10_impact"].mean()), "top1_rate": float(part["top1_impact"].mean()),
            "mean_strict_outside_cites": float(part["outside_field_cites_5y_strict"].mean()),
        }

    print("Fitting training-only feature transformations...", flush=True)
    train_builder = TemporalFeatureBuilder().fit(train)
    x_train = train_builder.transform(train)
    x_valid = train_builder.transform(valid)
    validation_predictions: dict[str, dict[str, np.ndarray]] = {outcome: {} for outcome in OUTCOMES}
    fit_meta = []
    for outcome, config in OUTCOMES.items():
        for model, terms in train_builder.specs().items():
            pred, meta = fit_predict(x_train[terms], train[outcome].to_numpy(float), x_valid[terms], config["family"])
            validation_predictions[outcome][model] = pred
            fit_meta.append({"stage": "validation", "outcome": outcome, "model": model, **meta})
        print(f"  validation predictions: {outcome}", flush=True)

    print("Refitting transformations and models on 2005-2014...", flush=True)
    final_builder = TemporalFeatureBuilder().fit(train_valid)
    x_train_valid = final_builder.transform(train_valid)
    x_test = final_builder.transform(test)
    test_predictions: dict[str, dict[str, np.ndarray]] = {outcome: {} for outcome in OUTCOMES}
    for outcome, config in OUTCOMES.items():
        prediction_frame = test[["patent_id", "grant_year", outcome]].copy()
        for model, terms in final_builder.specs().items():
            pred, meta = fit_predict(x_train_valid[terms], train_valid[outcome].to_numpy(float), x_test[terms], config["family"])
            test_predictions[outcome][model] = pred
            prediction_frame[model] = pred
            fit_meta.append({"stage": "test", "outcome": outcome, "model": model, **meta})
        prediction_frame.to_parquet(DATA_DIR / f"test_predictions_{outcome}.parquet", index=False)
        print(f"  test predictions: {outcome}", flush=True)

    metrics = pd.DataFrame(
        metric_rows("validation", valid, validation_predictions)
        + metric_rows("test", test, test_predictions)
    )
    yearly = pd.DataFrame(yearly_metric_rows(test, test_predictions))
    print("Running year-block bootstrap...", flush=True)
    bootstrap = bootstrap_primary_metrics(test, test_predictions)

    metrics.to_csv(MODEL_DIR / "temporal_holdout_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(MODEL_DIR / "temporal_holdout_yearly_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(MODEL_DIR / "temporal_holdout_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fit_meta).to_csv(MODEL_DIR / "temporal_holdout_fit_meta.csv", index=False, encoding="utf-8-sig")

    qa = {
        "run_time": pd.Timestamp.now().isoformat(), "elapsed_seconds": round(time.time() - started, 2),
        "protocol": {"train_years": list(TRAIN_YEARS), "validation_years": list(VALID_YEARS), "test_years": list(TEST_YEARS), "bootstrap_reps": BOOTSTRAP_REPS},
        "split_summary": split_summary,
        "training_feature_audit": train_builder.audit(), "final_feature_audit": final_builder.audit(),
        "all_models_converged": bool(pd.DataFrame(fit_meta)["converged"].all()),
        "all_predictions_finite": bool(pd.DataFrame(fit_meta)["finite_predictions"].all()),
        "retrospective_test_warning": "2015-2019 outcomes were previously used in full-sample V7 explanatory regressions.",
    }
    write_json(QA_DIR / "temporal_holdout_qc.json", qa)
    report = build_report(metrics, bootstrap, yearly, split_summary)
    (REPORT_DIR / "temporal_holdout_report.md").write_text(report, encoding="utf-8")

    print(f"Completed in {qa['elapsed_seconds']} seconds", flush=True)
    print(metrics.loc[metrics["split"].eq("test")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
