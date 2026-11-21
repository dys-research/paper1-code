"""Held-out calibration and fixed-budget screening from existing predictions."""
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from scripts.robustness.common import ANALYSIS, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=str, default=None)
    args = parser.parse_args()
    path = args.predictions or ANALYSIS / 'results/forecast_predictions.parquet'
    predictions = pd.read_parquet(path)
    if 'learner' not in predictions:
        predictions['learner'] = 'glm'
    calibrations, budgets, bins = [], [], []
    for (year, learner, model), group in predictions.loc[predictions.target.eq('top5')].groupby(
            ['grant_year', 'learner', 'model']):
        y, p = group.y_true.to_numpy(), group.prediction.to_numpy()
        z = logit(np.clip(p, 1e-8, 1-1e-8))
        fit = sm.GLM(y, sm.add_constant(z), family=sm.families.Binomial()).fit(
            cov_type='cluster', cov_kwds={'groups': group.assignee_cluster_key.to_numpy()})
        base = {'test_year': year, 'learner': learner, 'model': model}
        for index, term in enumerate(['intercept', 'slope']):
            calibrations.append({**base, 'term': term, 'estimate': fit.params[index],
                'std_err': fit.bse[index], 'ci95_low': fit.params[index]-1.95996398454*fit.bse[index],
                'ci95_high': fit.params[index]+1.95996398454*fit.bse[index],
                'n': len(group), 'positive_n': int(y.sum())})
        order = np.lexsort((group.patent_id.astype(str).to_numpy(), -p))
        for budget in [.01, .05, .10]:
            count = max(1, int(np.floor(budget * len(group))))
            hits = y[order[:count]].sum()
            budgets.append({**base, 'budget': budget, 'n': len(group), 'selected_k': count,
                'hits': hits, 'precision': hits/count, 'recall': hits/y.sum(),
                'hits_per_1000_screened': hits/count*1000, 'lift': (hits/count)/y.mean()})
        deciles = pd.qcut(p, 10, labels=False, duplicates='drop')
        for decile in np.unique(deciles):
            mask = deciles == decile
            bins.append({**base, 'bin': int(decile), 'n': int(mask.sum()),
                         'mean_prediction': p[mask].mean(), 'observed_rate': y[mask].mean()})
    for name, rows in [('calibration_by_year', calibrations), ('fixed_budget_by_year', budgets),
                       ('calibration_deciles', bins)]:
        pd.DataFrame(rows).to_csv(output('results/' + name + '.csv'), index=False)
    pd.DataFrame(budgets).groupby(['learner', 'model', 'budget'])[
        ['precision', 'recall', 'hits_per_1000_screened', 'lift']].mean().reset_index().to_csv(
            output('results/fixed_budget_summary.csv'), index=False)


if __name__ == '__main__':
    main()
