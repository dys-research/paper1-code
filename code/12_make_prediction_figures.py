from __future__ import annotations

"""Publication-format figures for the rolling B0--B5 OOT experiment.

Figure contract
Core conclusion: Structured P/M/E information adds outcome-specific, rather
than uniformly positive, out-of-time predictive value beyond grant-time
baselines and overall novelty.
Archetype: quantitative grid.  Source data: rolling OOT result files.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "prediction"
RES = OUT / "results"
FIG = OUT / "figures"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.spines.right": False,
    "axes.spines.top": False, "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42,
})
COLORS = {"B0": "#a6a6a6", "B1": "#426A8C", "B2": "#719CBA", "B3": "#D08C60", "B4": "#C86B56", "B5": "#8A3B4A"}
MODELS = list(COLORS)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")


def label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.17, 1.08, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RES / "rolling_oot_metrics_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), constrained_layout=True)
    panels = [
        ("forward_cites_5y", "poisson_deviance", "Five-year citations\nPoisson deviance ↓"),
        ("outside_field_cites_5y_strict", "poisson_deviance", "Strict outside-field citations\nPoisson deviance ↓"),
        ("semantic_reuse_5y_log1p", "r2", "Semantic reuse\nOOT $R^2$ ↑"),
        ("top1_impact", "ndcg_at_k", "Top 1% impact\nNDCG@k ↑"),
        ("top5_impact", "ndcg_at_k", "Top 5% impact\nNDCG@k ↑"),
        ("top10_impact", "ndcg_at_k", "Top 10% impact\nNDCG@k ↑"),
    ]
    x = np.arange(len(MODELS))
    for ax, (outcome, metric, title), letter in zip(axes.flat, panels, list("abcdef")):
        d = summary.loc[summary.outcome.eq(outcome)].set_index("model").reindex(MODELS)
        ax.plot(x, d[metric], marker="o", lw=1.3, ms=3.5, color="#3c3c3c", zorder=2)
        ax.scatter(x, d[metric], s=22, c=[COLORS[m] for m in MODELS], zorder=3, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x, MODELS); ax.set_title(title, loc="left", pad=6); ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        label(ax, letter)
    fig.text(0.5, -0.02, "Nested grant-time information set", ha="center", fontsize=7)
    save(fig, "fig_oot_incremental_performance")
    plt.close(fig)

    pred = pd.read_parquet(RES / "rolling_oot_predictions.parquet")
    high = ["top1_impact", "top5_impact", "top10_impact"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.25), constrained_layout=True)
    for ax, outcome, letter in zip(axes, high, list("abc")):
        for model in ["B1", "B5"]:
            d = pred.loc[(pred.outcome.eq(outcome)) & (pred.model.eq(model))].copy()
            d["bin"] = pd.qcut(d.prediction.rank(method="first"), q=10, labels=False)
            cal = d.groupby("bin", as_index=False).agg(predicted=("prediction", "mean"), observed=("y_true", "mean"))
            ax.plot(cal.predicted, cal.observed, marker="o", ms=3, lw=1.1, label=model, color=COLORS[model])
        lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, "--", color="#888888", lw=0.8, zorder=0); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency" if outcome == "top1_impact" else "")
        ax.set_title(outcome.replace("_impact", "").replace("top", "Top ") + " calibration", loc="left", pad=6)
        ax.legend(loc="upper left"); label(ax, letter)
    save(fig, "fig_high_impact_calibration")
    plt.close(fig)


if __name__ == "__main__":
    main()
