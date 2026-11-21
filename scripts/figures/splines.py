"""Draw complete-observation problem-novelty spline associations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from project_config import work_dir


def main():
    SRC = work_dir('robustness') / 'results'
    OUT = work_dir('figures')
    curves = pd.read_csv(SRC / "observed_only_spline_curves.csv", encoding="utf-8-sig")
    definition = json.loads((SRC / "spline_definition.json").read_text(encoding="utf-8"))
    assert set(curves["target"]) == {"citations", "semantic", "newfield"}
    assert curves.groupby("target").size().eq(61).all()
    assert curves[["P_z", "linear_predictor_change_at_mean_anchor", "ci95_low", "ci95_high"]].notna().all().all()
    assert (curves["ci95_low"] <= curves["linear_predictor_change_at_mean_anchor"]).all()
    assert (curves["linear_predictor_change_at_mean_anchor"] <= curves["ci95_high"]).all()

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42,
    })

    panels = [
        ("citations", "Five-year citations", "Change in expected citations (%)"),
        ("semantic", "Five-year semantic reuse", "Change in ln(reuse + 1)"),
        ("newfield", "Five-year new-field citations", "Change in expected citations (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.80), constrained_layout=True)
    for ax, (target, title, ylabel), letter in zip(axes, panels, "abc"):
        d = curves.loc[curves.target.eq(target)].sort_values("P_z")
        assert d.P_z.is_monotonic_increasing and d.P_z.nunique() == 61
        x = d.P_z.to_numpy(float)
        estimate = d.linear_predictor_change_at_mean_anchor.to_numpy(float)
        low = d.ci95_low.to_numpy(float)
        high = d.ci95_high.to_numpy(float)
        # PPML differences are on the log mean scale; exponentiation gives a percent change.
        # The OLS outcome is already ln(semantic reuse + 1), so it stays on that scale.
        if target != "semantic":
            estimate, low, high = [100 * np.expm1(z) for z in (estimate, low, high)]
        ax.fill_between(x, low, high, color="#D28B50", alpha=0.18, linewidth=0)
        ax.plot(x, estimate, color="#B46B35", linewidth=1.9)
        ax.axhline(0, color="#52575C", linewidth=0.8)
        ax.axvline(0, color="#A4A9AD", linewidth=0.8, linestyle="--")
        ax.text(-0.16, 1.07, letter, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="baseline")
        ax.text(0.0, 1.07, title, transform=ax.transAxes,
                fontsize=10, va="baseline")
        ax.set_xlabel("Problem novelty (standardized)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E6E9ED", linewidth=0.65)
        ax.set_axisbelow(True)

    stem = OUT / "problem_novelty_spline"
    for extension in (".svg", ".pdf", ".png"):
        fig.savefig(stem.with_suffix(extension), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved spline figure for {definition['n']:,} complete-observation patents: {stem}")

if __name__ == "__main__":
    main()
