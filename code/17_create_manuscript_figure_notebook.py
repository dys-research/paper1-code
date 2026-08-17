from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "manuscript_figures"
NOTEBOOK = OUT / "a61b_manuscript_figures.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = [
        md("""# A61B manuscript figures\n\n"
            "This notebook reproduces four manuscript figures from the data and fitted-model outputs used in the current Word draft. "
            "It reads source files only and saves publication exports to `figures/`. Each plotting cell begins with a Chinese source-data comment."""),
        md("""## Data and modelling conventions\n\n"
            "- Explanatory figures use the V7 M5 specification that underlies the current regression tables: grant-year and technical-subfield fixed effects, "
            "assignee-clustered robust covariance, and the structured P/M/E, residual-configuration and anchoring terms.\n"
            "- Top-5% figures use the rebuilt B0--B5 rolling out-of-time predictions.\n"
            "- The spline figure is a robustness visualization: it replaces the linear P-novelty term in M5 with a centered restricted cubic spline while retaining the same outcome-specific data, controls and fixed effects."""),
        code(r'''
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from patsy import dmatrix, build_design_matrices
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = next((candidate for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (candidate / "project_config.py").exists()), None)
if ROOT is None:
    raise FileNotFoundError("Run the notebook from within the repository directory.")
sys.path.insert(0, str(ROOT))
from project_config import config_path, load_config

PATENTSVIEW = config_path(load_config(ROOT), "patentsview_dir")
WORK = ROOT / "work"
EXPLANATORY = WORK / "explanatory"
FIG_DIR = WORK / "manuscript_figures" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.right": False,
    "axes.spines.top": False, "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42,
})
COLORS = {"navy": "#274C77", "teal": "#4C8C7A", "orange": "#D28B50", "burgundy": "#9E3D4A", "grey": "#8E98A3"}

def save_pub(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")

def panel_label(ax, letter):
    ax.text(-0.18, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

script_path = ROOT / "code" / "07_run_explanatory_models.py"
spec = importlib.util.spec_from_file_location("v7_models_for_figures", script_path)
v7m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v7m
spec.loader.exec_module(v7m)

analysis_df = pd.read_parquet(EXPLANATORY / "data" / "a61b_v7_modeling_panel.parquet")
analysis_df["patent_id"] = analysis_df["patent_id"].astype(str)
cross = pd.read_parquet(WORK / "diagnostics" / "cross_field_quick_check" / "cross_field_counts_by_patent.parquet", columns=["patent_id", "outside_field_cites_5y_strict"])
cross["patent_id"] = cross["patent_id"].astype(str)
reuse = pd.read_parquet(PATENTSVIEW / "7Text" / "PatentSimilarityImportanceBreakthrough_forPost2022.parquet", columns=["patent_num", "fsim01", "fsim25"])
reuse["patent_id"] = reuse["patent_num"].astype(str)
reuse["semantic_reuse_5y_log1p"] = np.log1p(pd.to_numeric(reuse["fsim01"], errors="coerce") + pd.to_numeric(reuse["fsim25"], errors="coerce"))
analysis_df = analysis_df.merge(cross, on="patent_id", how="left", validate="one_to_one")
analysis_df = analysis_df.merge(reuse[["patent_id", "semantic_reuse_5y_log1p"]], on="patent_id", how="left", validate="one_to_one")
analysis_df["outside_field_cites_5y_strict"] = analysis_df["outside_field_cites_5y_strict"].fillna(0).astype(int)
analysis_df.loc[analysis_df["grant_year"].gt(2017), "semantic_reuse_5y_log1p"] = np.nan

coef_forward = pd.read_csv(EXPLANATORY / "models" / "v7_m0_m6_coefficients.csv")
coef_extended = pd.read_csv(WORK / "extended_outcomes" / "models" / "extended_m0_m6_coefficients.csv")
oot_results = WORK / "prediction" / "results"
oot_summary = pd.read_csv(oot_results / "rolling_oot_metrics_summary.csv")
oot_by_year = pd.read_csv(oot_results / "rolling_oot_metrics_by_year.csv")
oot_bootstrap = pd.read_csv(oot_results / "b5_vs_b1_year_block_bootstrap.csv")
oot_predictions = pd.read_parquet(oot_results / "rolling_oot_predictions.parquet")
print(f"Analysis panel: {len(analysis_df):,} patents; figure output: {FIG_DIR}")
'''),
        code(r'''
core_terms = [
    "problem_novelty_fy_z_filled", "mechanism_novelty_fy_z_filled", "effect_novelty_fy_z_filled",
    "pm_config_residual_z", "me_config_residual_z", "anchoring_centered", "pm_residual_x_anchoring",
]
term_labels = {
    "problem_novelty_fy_z_filled": "Problem novelty", "mechanism_novelty_fy_z_filled": "Mechanism novelty",
    "effect_novelty_fy_z_filled": "Effect novelty", "pm_config_residual_z": "P-M residual configuration",
    "me_config_residual_z": "M-E residual configuration", "anchoring_centered": "Anchoring",
    "pm_residual_x_anchoring": "P-M configuration × anchoring",
}
sources = [
    (coef_forward, "forward_cites_5y", "Five-year citations (PPML)"),
    (coef_extended, "semantic_reuse_5y_log1p", "Semantic reuse (OLS)"),
    (coef_extended, "outside_field_cites_5y_strict", "Outside-field citations (PPML)"),
]
fig, axes = plt.subplots(1, 3, figsize=(8.25, 3.65), sharey=True, constrained_layout=True)
for ax, (source, outcome, title), letter in zip(axes, sources, "abc"):
    d = source.loc[(source["outcome"].eq(outcome)) & (source["model"].eq("M5")) & (source["term"].isin(core_terms))].copy()
    d["term"] = pd.Categorical(d["term"], categories=core_terms, ordered=True)
    d = d.sort_values("term", ascending=False)
    outcome_sample = analysis_df.loc[analysis_df[outcome].notna(), core_terms]
    scale = outcome_sample.std(ddof=0).reindex(d["term"]).to_numpy()
    d["plot_coef"] = d["coef"].to_numpy() * scale
    d["plot_se"] = d["std_err"].to_numpy() * scale
    y = np.arange(len(d))
    lo, hi = d["plot_coef"] - 1.96*d["plot_se"], d["plot_coef"] + 1.96*d["plot_se"]
    colors = [COLORS["grey"] if p >= .05 else (COLORS["orange"] if c >= 0 else COLORS["navy"]) for c, p in zip(d["plot_coef"], d["p_value"])]
    ax.hlines(y, lo, hi, color=colors, lw=1.35, zorder=3)
    ax.scatter(d["plot_coef"], y, s=23, c=colors, edgecolor="white", linewidth=.65, zorder=4)
    ax.axvline(0, color="#C95E3D", lw=1.25, ls="--", zorder=2)
    for yy in y:
        ax.axhline(yy, color="#DCE3E8", lw=.7, ls=(0, (1.4, 1.8)), zorder=0)
    for xx, yy, stars in zip(d["plot_coef"], y, d["stars"]):
        if pd.notna(stars) and str(stars).strip():
            ax.text(xx, yy + .17, stars, ha="center", va="bottom", color="#C96C49", fontsize=7, fontweight="bold")
    lim = max(abs(lo.min()), abs(hi.max())) * 1.33
    ax.set_xlim(-lim, lim)
    ax.set_title(title, loc="left", pad=7)
    ax.set_yticks(y, [term_labels[t] for t in d["term"]])
    ax.set_xlabel("Coefficient per 1 SD (95% CI)")
    panel_label(ax, letter)
fig.text(.5, -.03, "M5 includes grant-year and technical-subfield fixed effects; standard errors clustered by assignee. Points and intervals are rescaled only for display to a one-standard-deviation change in each regressor.", ha="center", fontsize=6.1)
save_pub(fig, "fig1_core_coefficients_m5")
plt.show()
'''),
        code(r'''
m5_terms = v7m.model_specs([])["M5"]
outcome_specs = [
    ("forward_cites_5y", "poisson", "Five-year citations", "Percent change in expected citations"),
    ("semantic_reuse_5y_log1p", "ols", "Semantic reuse", "Change in ln semantic reuse"),
    ("outside_field_cites_5y_strict", "poisson", "Outside-field citations", "Percent change in expected citations"),
]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
for ax, (outcome, family, title, ylabel), letter in zip(axes, outcome_specs, "abc"):
    _, _, result, _ = v7m.fit_one(analysis_df, outcome, family, "M5", m5_terms)
    sample = analysis_df.loc[analysis_df[outcome].notna()]
    anchor_grid = np.linspace(sample["anchoring_centered"].quantile(.05), sample["anchoring_centered"].quantile(.95), 120)
    beta_pm = result.params["pm_config_residual_z"]
    beta_int = result.params["pm_residual_x_anchoring"]
    cov = result.cov_params().loc[["pm_config_residual_z", "pm_residual_x_anchoring"], ["pm_config_residual_z", "pm_residual_x_anchoring"]].to_numpy()
    design = np.column_stack([np.ones_like(anchor_grid), anchor_grid])
    eta = beta_pm + beta_int * anchor_grid
    se = np.sqrt(np.einsum("ij,jk,ik->i", design, cov, design))
    lo, hi = eta - 1.96*se, eta + 1.96*se
    if family == "poisson":
        effect, lower, upper = (np.exp(eta)-1)*100, (np.exp(lo)-1)*100, (np.exp(hi)-1)*100
    else:
        effect, lower, upper = eta, lo, hi
    ax.fill_between(anchor_grid, lower, upper, color=COLORS["teal"], alpha=.18, lw=0)
    ax.plot(anchor_grid, effect, color=COLORS["teal"], lw=1.7)
    ax.axhline(0, color="#555555", lw=.8)
    ax.axvline(0, color="#A5A5A5", lw=.7, ls="--")
    ax.grid(axis="y", color="#E6E9ED", lw=.65)
    ax.set_title(title, loc="left", pad=7)
    ax.set_xlabel("Anchoring (centered)")
    ax.set_ylabel(ylabel if ax is axes[0] else "")
    panel_label(ax, letter)
fig.text(.5, -.04, "Line: marginal effect of a one-SD increase in P-M residual configuration; band: 95% clustered CI.", ha="center", fontsize=6.5)
save_pub(fig, "fig2_pm_anchoring_marginal_effects")
plt.show()
'''),
        code(r'''
top5 = oot_summary.loc[oot_summary["outcome"].eq("top5_impact")].copy()
top5["model"] = pd.Categorical(top5["model"], categories=["B0", "B1", "B2", "B3", "B4", "B5"], ordered=True)
top5 = top5.sort_values("model")
yearly = oot_by_year.loc[oot_by_year["outcome"].eq("top5_impact")].copy()
b1_yearly = yearly.loc[yearly["model"].eq("B1"), ["test_year", "ndcg_at_k"]].rename(columns={"ndcg_at_k": "ndcg_b1"})
delta = yearly.loc[yearly["model"].isin(["B2", "B3", "B4", "B5"]), ["test_year", "model", "ndcg_at_k"]].merge(b1_yearly, on="test_year", validate="many_to_one")
delta["delta_ndcg"] = delta["ndcg_at_k"] - delta["ndcg_b1"]
delta["model"] = pd.Categorical(delta["model"], categories=["B2", "B3", "B4", "B5"], ordered=True)
b5_boot = oot_bootstrap.loc[(oot_bootstrap["outcome"].eq("top5_impact")) & (oot_bootstrap["metric"].eq("ndcg_at_k"))].iloc[0]

fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.55), constrained_layout=True)
ax = axes[0]
x = np.arange(len(top5))
ax.plot(x, top5["precision_at_k"], marker="o", ms=3.8, lw=1.25, color=COLORS["navy"], label="Precision@k (= Recall@k)")
ax.plot(x, top5["ndcg_at_k"], marker="o", ms=3.8, lw=1.25, color=COLORS["burgundy"], label="NDCG@k")
ax.set_xticks(x, top5["model"])
ax.set_ylim(0, .50)
ax.set_ylabel("Held-out ranking metric")
ax.set_title("Top 5% impact identification", loc="left", pad=7)
ax.grid(axis="y", color="#E6E9ED", lw=.65)
ax.legend(loc="upper left")
panel_label(ax, "a")

ax = axes[1]
models_delta = ["B2", "B3", "B4", "B5"]
palette = {"B2": "#AAB4BE", "B3": "#7D8FA1", "B4": "#5D7388", "B5": COLORS["burgundy"]}
for xpos, model in enumerate(models_delta):
    values = delta.loc[delta["model"].eq(model), "delta_ndcg"].to_numpy()
    jitter = np.linspace(-.085, .085, len(values))
    ax.scatter(np.full(len(values), xpos) + jitter, values, s=17, color=palette[model], alpha=.45, zorder=2)
    mean_value = values.mean()
    if model == "B5":
        low, high = float(b5_boot["ci95_lower"]), float(b5_boot["ci95_upper"])
        ax.errorbar(xpos, mean_value, yerr=[[mean_value-low], [high-mean_value]], fmt="o", ms=5.2, lw=1.35, capsize=2.5, color=palette[model], label="Mean; B5 95% block-bootstrap CI", zorder=4)
    else:
        ax.scatter(xpos, mean_value, s=38, color=palette[model], edgecolor="white", linewidth=.55, zorder=4)
ax.axhline(0, color="#555555", lw=.85)
ax.set_xticks(np.arange(len(models_delta)), models_delta)
ax.set_ylabel("Change in NDCG@k relative to B1")
ax.set_title("Incremental ranking gain", loc="left", pad=7)
ax.grid(axis="y", color="#E6E9ED", lw=.65)
ax.legend(loc="upper left", fontsize=5.6)
panel_label(ax, "b")

ax = axes[2]
for model, color in [("B1", COLORS["navy"]), ("B5", COLORS["burgundy"])]:
    d = oot_predictions.loc[(oot_predictions["outcome"].eq("top5_impact")) & (oot_predictions["model"].eq(model))].copy()
    d["bin"] = pd.qcut(d["prediction"].rank(method="first"), q=10, labels=False)
    calibration = d.groupby("bin", as_index=False).agg(predicted=("prediction", "mean"), observed=("y_true", "mean"))
    ax.plot(calibration["predicted"], calibration["observed"], marker="o", ms=3.2, lw=1.25, color=color, label=model)
limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, limit], [0, limit], ls="--", color="#909090", lw=.9, zorder=0)
ax.set_xlim(0, limit); ax.set_ylim(0, limit)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed Top-5% frequency")
ax.set_title("Calibration across held-out years", loc="left", pad=7)
ax.legend(loc="upper left")
panel_label(ax, "c")
fig.text(.5, -.04, "Panel b: light points are individual rolling test years; large points are mean gains. The B5 interval is a year-block bootstrap 95% CI.", ha="center", fontsize=6.2)
save_pub(fig, "fig3a_top5_delta_panel_and_calibration")
plt.show()
'''),
        code(r'''
top5 = oot_summary.loc[oot_summary["outcome"].eq("top5_impact")].copy()
top5["model"] = pd.Categorical(top5["model"], categories=["B0", "B1", "B2", "B3", "B4", "B5"], ordered=True)
top5 = top5.sort_values("model")
yearly = oot_by_year.loc[oot_by_year["outcome"].eq("top5_impact")].copy()
b1_yearly = yearly.loc[yearly["model"].eq("B1"), ["test_year", "ndcg_at_k"]].rename(columns={"ndcg_at_k": "ndcg_b1"})
delta = yearly.loc[yearly["model"].isin(["B2", "B3", "B4", "B5"]), ["test_year", "model", "ndcg_at_k"]].merge(b1_yearly, on="test_year", validate="many_to_one")
delta["delta_ndcg"] = delta["ndcg_at_k"] - delta["ndcg_b1"]
delta["model"] = pd.Categorical(delta["model"], categories=["B2", "B3", "B4", "B5"], ordered=True)
b5_boot = oot_bootstrap.loc[(oot_bootstrap["outcome"].eq("top5_impact")) & (oot_bootstrap["metric"].eq("ndcg_at_k"))].iloc[0]
models_delta = ["B2", "B3", "B4", "B5"]
palette = {"B2": "#AAB4BE", "B3": "#7D8FA1", "B4": "#5D7388", "B5": COLORS["burgundy"]}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), constrained_layout=True)
ax = axes[0]
x = np.arange(len(top5))
ax.plot(x, top5["precision_at_k"], marker="o", ms=3.8, lw=1.25, color=COLORS["navy"], label="Precision@k (= Recall@k)")
ax.plot(x, top5["ndcg_at_k"], marker="o", ms=3.8, lw=1.25, color=COLORS["burgundy"], label="NDCG@k")
ax.set_xticks(x, top5["model"])
ax.set_ylim(0, .50)
ax.set_ylabel("Held-out ranking metric")
ax.set_title("Top 5% impact identification", loc="left", pad=7)
ax.grid(axis="y", color="#E6E9ED", lw=.65)
ax.legend(loc="upper left")
panel_label(ax, "a")

inset = ax.inset_axes([.53, .10, .43, .40])
for xpos, model in enumerate(models_delta):
    values = delta.loc[delta["model"].eq(model), "delta_ndcg"].to_numpy()
    inset.scatter(np.full(len(values), xpos) + np.linspace(-.07, .07, len(values)), values, s=10, color=palette[model], alpha=.42, zorder=2)
    mean_value = values.mean()
    if model == "B5":
        low, high = float(b5_boot["ci95_lower"]), float(b5_boot["ci95_upper"])
        inset.errorbar(xpos, mean_value, yerr=[[mean_value-low], [high-mean_value]], fmt="o", ms=3.9, lw=1.0, capsize=1.7, color=palette[model], zorder=4)
    else:
        inset.scatter(xpos, mean_value, s=24, color=palette[model], edgecolor="white", linewidth=.4, zorder=4)
inset.axhline(0, color="#555555", lw=.6)
inset.set_xticks(np.arange(len(models_delta)), models_delta)
inset.set_title("ΔNDCG@k vs B1", fontsize=5.7, pad=2)
inset.tick_params(labelsize=5, length=2)
inset.grid(axis="y", color="#E6E9ED", lw=.45)

ax = axes[1]
for model, color in [("B1", COLORS["navy"]), ("B5", COLORS["burgundy"])]:
    d = oot_predictions.loc[(oot_predictions["outcome"].eq("top5_impact")) & (oot_predictions["model"].eq(model))].copy()
    d["bin"] = pd.qcut(d["prediction"].rank(method="first"), q=10, labels=False)
    calibration = d.groupby("bin", as_index=False).agg(predicted=("prediction", "mean"), observed=("y_true", "mean"))
    ax.plot(calibration["predicted"], calibration["observed"], marker="o", ms=3.2, lw=1.25, color=color, label=model)
limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, limit], [0, limit], ls="--", color="#909090", lw=.9, zorder=0)
ax.set_xlim(0, limit); ax.set_ylim(0, limit)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed Top-5% frequency")
ax.set_title("Calibration across held-out years", loc="left", pad=7)
ax.legend(loc="upper left")
panel_label(ax, "b")
fig.text(.5, -.04, "Precision@k equals Recall@k because k equals the number of actual Top-5% patents in each test set. Inset: light points are individual rolling test years; the B5 error bar is a year-block bootstrap 95% CI.", ha="center", fontsize=5.8)
save_pub(fig, "fig3b_top5_delta_inset_and_calibration")
plt.show()
'''),
        code(r'''
def fit_problem_spline(df, outcome, family):
    terms = [t for t in v7m.model_specs([])["M5"] if t != v7m.P_NOV]
    needed = [outcome, "grant_year", "technical_subfield", "primary_assignee_id", v7m.P_NOV] + terms
    n_before = len(df)
    d = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    n_after = len(d)
    excluded_count = n_before - n_after
    print(f"{outcome}: spline sample {n_after:,}/{n_before:,}; excluded {excluded_count:,} rows with missing required outcome/covariates.")
    spline = dmatrix("cr(x, df=4, constraints='center') - 1", {"x": d[v7m.P_NOV]}, return_type="dataframe")
    spline.columns = [f"p_spline_{i}" for i in range(spline.shape[1])]
    base_x = v7m.build_x(d, terms).reset_index(drop=True)
    x = pd.concat([base_x.reset_index(drop=True), spline.reset_index(drop=True)], axis=1)
    groups = d["primary_assignee_id"].fillna("__MISSING_ASSIGNEE__").astype(str)
    y = d[outcome].astype(float).reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    if family == "poisson":
        result = sm.GLM(y, x, family=sm.families.Poisson()).fit(maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
    else:
        result = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return d, spline.design_info, x.columns, result

spline_specs = [
    ("forward_cites_5y", "poisson", "Five-year citations", "Percent change relative to P novelty = 0"),
    ("semantic_reuse_5y_log1p", "ols", "Semantic reuse", "Change in ln semantic reuse"),
    ("outside_field_cites_5y_strict", "poisson", "Outside-field citations", "Percent change relative to P novelty = 0"),
]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
for ax, (outcome, family, title, ylabel), letter in zip(axes, spline_specs, "abc"):
    d, spline_info, xcols, result = fit_problem_spline(analysis_df, outcome, family)
    grid = np.linspace(d[v7m.P_NOV].quantile(.02), d[v7m.P_NOV].quantile(.98), 120)
    spline_grid = pd.DataFrame(build_design_matrices([spline_info], {"x": grid})[0], columns=[c for c in xcols if c.startswith("p_spline_")])
    spline_zero = pd.DataFrame(build_design_matrices([spline_info], {"x": [0.0]})[0], columns=spline_grid.columns)
    contrast = np.zeros((len(grid), len(xcols)))
    spline_positions = [list(xcols).index(c) for c in spline_grid.columns]
    contrast[:, spline_positions] = spline_grid.to_numpy() - spline_zero.to_numpy()
    beta = result.params.reindex(xcols).to_numpy()
    cov = result.cov_params().reindex(index=xcols, columns=xcols).to_numpy()
    eta = contrast @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", contrast, cov, contrast))
    lo, hi = eta - 1.96*se, eta + 1.96*se
    if family == "poisson":
        effect, lower, upper = (np.exp(eta)-1)*100, (np.exp(lo)-1)*100, (np.exp(hi)-1)*100
    else:
        effect, lower, upper = eta, lo, hi
    ax.fill_between(grid, lower, upper, color=COLORS["orange"], alpha=.18, lw=0)
    ax.plot(grid, effect, color=COLORS["orange"], lw=1.7)
    ax.axhline(0, color="#555555", lw=.8)
    ax.axvline(0, color="#A5A5A5", lw=.7, ls="--")
    ax.grid(axis="y", color="#E6E9ED", lw=.65)
    ax.set_title(title, loc="left", pad=7)
    ax.set_xlabel("Problem novelty (standardised)")
    ax.set_ylabel(ylabel if ax is axes[0] else "")
    panel_label(ax, letter)
fig.text(.5, -.04, "Centered restricted cubic spline (4 df); M5 controls, grant-year and technical-subfield fixed effects retained.", ha="center", fontsize=6.5)
save_pub(fig, "fig4_problem_novelty_restricted_cubic_spline")
plt.show()
'''),
        md("""## Takeaways\n\n"
            "The notebook separates explanatory and forecasting evidence. Figure 1 summarizes core M5 coefficients. Figure 2 makes the P-M configuration × anchoring mechanism visible. Figure 3 documents Top-5% screening and calibration using held-out predictions. Figure 4 provides a nonlinearity robustness check without altering the underlying sample or the Word draft's baseline data."""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
