"""Draw annual errors, individual loss gains and role-availability comparisons."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from scipy.special import xlogy
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator, FuncFormatter
from project_config import work_dir
from scripts.analysis.common import readpanel
from scripts.analysis.summarize import paired_draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learner', choices=['glm', 'tree'], default='glm')
    parser.add_argument('--replicates', type=int, default=500)
    args = parser.parse_args()
    OUT = work_dir('figures')
    STEM = OUT / 'forecast_role_complementarity'
    data = pd.read_parquet(work_dir('analysis') / 'results/forecast_predictions.parquet')
    data = data.loc[data.learner.eq(args.learner)].copy()
    data['patent_id'] = data.patent_id.astype(str)
    meta = readpanel()
    if {'P_usable', 'M_usable', 'E_usable'} <= set(meta):
        meta['valid_role_n'] = meta[['P_usable', 'M_usable', 'E_usable']].sum(axis=1)
    else:
        meta['valid_role_n'] = meta[['problem_novelty_raw', 'mechanism_novelty_raw', 'effect_novelty_raw']].notna().sum(axis=1)
    YEARS = np.sort(data.grant_year.unique())
    annual_records, summary_records, ecdf_frames = [], [], []
    data_by_target, bootstrap_draws = {}, {}
    for target, kind in [('citations', 'count'), ('semantic', 'continuous')]:
        selected = data.loc[data.target.eq(target) & data.model.isin(['overall', 'overall_structure'])]
        frames = {}
        for model in ['overall', 'overall_structure']:
            frame = selected.loc[selected.model.eq(model)].sort_values(['grant_year', 'patent_id']).reset_index(drop=True)
            if frame.empty or frame.patent_id.duplicated().any():
                raise ValueError(f'Expected unique paired forecasts for {target} {model}')
            frames[model] = frame
        base, augmented = frames['overall'], frames['overall_structure']
        for key in ['patent_id', 'grant_year', 'y_true', 'assignee_cluster_key']:
            if not np.array_equal(base[key], augmented[key]):
                raise ValueError('Models are not paired on identical patents and labels')
        frame = base[['patent_id', 'grant_year', 'assignee_cluster_key', 'y_true']].copy()
        frame['prediction_text'] = base.prediction.to_numpy(float)
        frame['prediction_text_roles'] = augmented.prediction.to_numpy(float)
        frame = frame.merge(meta[['patent_id', 'valid_role_n']], on='patent_id', validate='one_to_one', how='left')
        if frame.valid_role_n.isna().any():
            raise ValueError('Forecast patents not found in measured panel')
        yy = frame.y_true.to_numpy(float)
        pred = frame[['prediction_text', 'prediction_text_roles']].to_numpy(float)
        losses = 2 * (xlogy(yy[:, None], yy[:, None] / pred) - yy[:, None] + pred) if kind == 'count' else (yy[:, None] - pred) ** 2
        frame['individual_loss_gain'] = losses[:, 0] - losses[:, 1]
        frame['target'] = target
        data_by_target[target] = frame
        gains = np.sort(frame.individual_loss_gain.to_numpy())
        ecdf_frames.append(pd.DataFrame({'target': target, 'individual_loss_gain': gains,
                                        'cumulative_fraction': np.arange(1, len(frame) + 1) / len(frame)}))
        for cohort in ['All patents', 'P/M/E available']:
            mask = np.ones(len(frame), bool) if cohort == 'All patents' else frame.valid_role_n.eq(3).to_numpy()
            sub = frame.loc[mask]
            if sub.empty or set(sub.grant_year) != set(YEARS):
                raise ValueError('Role-available subgroup must cover every plotted test year')
            annual_values = []
            for year in YEARS:
                indices = mask & frame.grant_year.eq(year).to_numpy()
                values = losses[indices].mean(axis=0)
                if kind == 'continuous':
                    values = np.sqrt(values)
                annual_values.append(values)
                annual_records.append({'target': target, 'cohort': cohort, 'year': int(year),
                                       'n': int(indices.sum()), 'text_error': values[0],
                                       'text_roles_error': values[1], 'gain': values[0] - values[1]})
            annual_values = np.asarray(annual_values)
            ids = set(sub.patent_id)
            draws, _, clusters = paired_draws(selected.loc[selected.patent_id.isin(ids)],
                                               ['overall', 'overall_structure'], kind,
                                               replicates=args.replicates, seed=90217)
            improvements = draws[:, 0] - draws[:, 1]
            low, high = np.quantile(improvements, [.025, .975])
            bootstrap_draws[f'{target}_{"all" if cohort == "All patents" else "pme"}'] = improvements
            summary_records.append({'target': target, 'cohort': cohort, 'n': len(sub),
                                    'assignee_clusters': clusters,
                                    'gain_equal_years': float((annual_values[:, 0] - annual_values[:, 1]).mean()),
                                    'ci95_low': low, 'ci95_high': high,
                                    'fraction_individual_patents_improved': float((losses[mask, 0] > losses[mask, 1]).mean())})
    annual = pd.DataFrame(annual_records)
    summary = pd.DataFrame(summary_records)
    ecdf = pd.concat(ecdf_frames, ignore_index=True)
    annual.to_csv(OUT / 'annual_performance_and_gains.csv', index=False)
    summary.to_csv(OUT / 'mean_gains_and_paired_intervals.csv', index=False)
    ecdf.to_csv(OUT / 'individual_gain_ecdf.csv', index=False)
    np.savez_compressed(OUT / 'paired_bootstrap_draws.npz', **bootstrap_draws)

    BLUE, TEAL, PURPLE, GREY = "#274C77", "#368875", "#795CA1", "#68727D"
    style = {
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": .8, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "axes.unicode_minus": True,
    }

    with mpl.rc_context(style):
        fig, axes = plt.subplots(4, 2, figsize=(9.0, 11.7), layout="constrained",
                                 gridspec_kw={"height_ratios": [1.0, 1.02, 1.0, .73]})
        fig.get_layout_engine().set(h_pad=.13, w_pad=.13, hspace=.08, wspace=.08)
        for ax in axes.flat:
            ax.grid(axis="y", color="#E7EBEE", linewidth=.65)
            ax.set_axisbelow(True)
            ax.tick_params(length=3.5, width=.8)
        def heading(ax, letter, text):
            ax.text(-.13, 1.075, letter, transform=ax.transAxes, fontsize=14,
                    fontweight="bold", va="baseline", ha="left")
            ax.text(0, 1.075, text, transform=ax.transAxes, fontsize=12,
                    va="baseline", ha="left")

        for col, target in enumerate(["citations", "semantic"]):
            all_years = annual.loc[(annual.target == target) & (annual.cohort == "All patents")].sort_values("year")
            complete = annual.loc[(annual.target == target) & (annual.cohort == "P/M/E available")].sort_values("year")
            sums = summary.loc[summary.target == target].set_index("cohort")
            ax = axes[0, col]
            ax.plot(YEARS, all_years.text_error, color=BLUE, marker="o", ms=5, lw=1.8, label="Overall-text embedding")
            ax.plot(YEARS, all_years.text_roles_error, color=TEAL, marker="s", ms=4.7, lw=1.8, label="+ structured role indicators")
            heading(ax, "ab"[col], "Five-year citations" if col == 0 else "Five-year semantic reuse")
            ax.set_ylabel("Poisson deviance" if col == 0 else "RMSE")
            ax.set_xlabel("Test year")
            ax.set_xticks(YEARS)
            ax.set_xlim(YEARS.min() - .3, YEARS.max() + .3)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            if col == 0:
                ax.legend(loc="lower left", frameon=False, handlelength=1.7, borderaxespad=.2)
            ax.text(.98, .98, "Lower is better", transform=ax.transAxes, ha="right", va="top", color=GREY, fontsize=10)

            ax = axes[1, col]
            d = ecdf.loc[ecdf.target == target]
            threshold = 1 if target == "citations" else .01
            extent = max(float(np.max(np.abs(d.individual_loss_gain))) * 1.06, threshold * 2)
            ax.set_xscale("symlog", linthresh=threshold, linscale=.65)
            ax.set_xlim(-extent, extent)
            ax.axvspan(0, extent, color=TEAL, alpha=.055, zorder=0)
            ax.step(d.individual_loss_gain, d.cumulative_fraction, where="post", color=TEAL, lw=1.7, rasterized=True)
            ax.axvline(0, color=GREY, lw=.8, ls="--")
            fraction = float(sums.loc["All patents", "fraction_individual_patents_improved"])
            fzero = float((data_by_target[target].individual_loss_gain <= 0).mean())
            ax.scatter([0], [fzero], color=TEAL, edgecolor="white", s=35, zorder=4)
            ax.text(.035, .96, f"{fraction:.1%} of patents\nhave lower error", transform=ax.transAxes,
                    ha="left", va="top", fontsize=11, color=TEAL)
            ax.set_ylim(0, 1.025)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set_ylabel("Cumulative share of patents")
            ticks = [-100, -1, 0, 1, 100] if target == "citations" else [-1, -.01, 0, .01, 1]
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
            ax.set_xlabel(("Per-patent deviance reduction" if col == 0 else "Per-patent squared-error reduction") + "\n(symmetric-log scale)")
            heading(ax, "cd"[col], "Individual prediction gains")

            ax = axes[2, col]
            scale = 1 if target == "citations" else 1000
            ax.axhline(0, color=GREY, lw=.85, ls="--")
            ax.plot(YEARS, all_years.gain * scale, color=GREY, marker="o", ms=5, lw=1.6, ls="--", label=f"All patents (n = {int(sums.loc['All patents', 'n']):,})")
            ax.plot(YEARS, complete.gain * scale, color=PURPLE, marker="s", ms=4.7, lw=1.9, label=f"P/M/E available (n = {int(sums.loc['P/M/E available', 'n']):,})")
            ax.set_xlim(YEARS.min() - .3, YEARS.max() + .3)
            ax.set_xticks(YEARS)
            ax.set_xlabel("Test year")
            ax.set_ylabel("Deviance reduction" if col == 0 else r"RMSE reduction ($\times10^{-3}$)")
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.margins(y=.2)
            heading(ax, "ef"[col], "Gains by role availability")
            if col == 0:
                ax.legend(loc="upper right", frameon=False, handlelength=1.7, fontsize=9.8, borderaxespad=.1)

            ax = axes[3, col]
            ax.grid(False)
            ax.grid(axis="x", color="#E7EBEE", linewidth=.65)
            ax.axvline(0, color=GREY, lw=.85, ls="--")
            for yloc, cohort, color, marker in [(1, "All patents", GREY, "o"), (0, "P/M/E available", PURPLE, "s")]:
                r = sums.loc[cohort]
                center, lo, hi = np.array([r.gain_equal_years, r.ci95_low, r.ci95_high]) * scale
                ax.errorbar(center, yloc, xerr=[[center-lo], [hi-center]], fmt=marker,
                            color=color, ecolor=color, ms=6, lw=1.7, capsize=4, capthick=1.3)
                ax.text(.98, yloc+.21, f"{center:.3f} [{lo:.3f}, {hi:.3f}]" if col == 0 else f"{center:.2f} [{lo:.2f}, {hi:.2f}]",
                        transform=ax.get_yaxis_transform(), ha="right", va="bottom", color=color, fontsize=10)
            ax.set_yticks([1, 0], ["All patents", "P/M/E available"])
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.set_ylim(-.48, 1.62)
            ax.margins(x=.25)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.set_xlabel("Deviance reduction" if col == 0 else r"RMSE reduction ($\times10^{-3}$)")
            heading(ax, "gh"[col], "Mean gains and 95% intervals")

        for extension in (".svg", ".pdf", ".png"):
            fig.savefig(STEM.with_suffix(extension), dpi=400, facecolor="white")
        plt.close(fig)
        print(f"Saved forecast figure: {STEM}")

if __name__ == "__main__":
    main()
