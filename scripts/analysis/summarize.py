"""Equal-year metrics and paired assignee-cluster intervals for held-out forecasts."""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import xlogy
from statsmodels.stats.multitest import multipletests

from scripts.analysis.common import OUTCOMES, log, output, save

METRICS = ['poisson_deviance', 'mae', 'rmse', 'spearman', 'r2', 'roc_auc', 'pr_auc',
           'brier', 'precision_at_k', 'recall_at_k', 'ndcg_at_k']
DEFAULT_PAIRS = [('overall_structure', 'overall'), ('structure', 'common'),
                 ('roles', 'common'), ('roles', 'text_fields'), ('overall', 'common')]


def centered_bootstrap_p(draws, point):
    """Retain the source analysis's two-sided, centered-null sensitivity p."""
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 1 or not len(draws) or not np.isfinite(draws).all() or not np.isfinite(point):
        raise ValueError('Finite paired differences and a finite point estimate are required')
    return float((1 + np.count_nonzero(np.abs(draws - point) >= abs(point))) / (len(draws) + 1))


def add_holm_four_tasks(intervals):
    """Adjust each learner/model pair across exactly the four study endpoints.

    A deliberately smaller forecast run still produces its original intervals
    and raw p-values, but is not labelled as a four-task Holm comparison.
    """
    result = intervals.copy()
    result['p_Holm_four_endpoints'] = np.nan
    result['holm_family_size'] = 4
    result['holm_observed_tasks'] = 0
    result['holm_status'] = 'incomplete_four_task_family'
    if result.empty:
        return result
    required = {'learner', 'candidate', 'baseline', 'target', 'centered_bootstrap_p'}
    if required - set(result):
        raise ValueError('Holm input is missing columns: ' + ', '.join(sorted(required - set(result))))
    for _, indices in result.groupby(['learner', 'candidate', 'baseline']).groups.items():
        group = result.loc[indices]
        if group.target.duplicated().any():
            raise ValueError('Repeated task within a Holm learner/model-pair family')
        result.loc[indices, 'holm_observed_tasks'] = len(group)
        if set(group.target) != set(OUTCOMES):
            continue
        values = group.centered_bootstrap_p.to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Holm correction requires finite p-values in [0, 1]')
        result.loc[indices, 'p_Holm_four_endpoints'] = multipletests(values, method='holm')[1]
        result.loc[indices, 'holm_status'] = 'complete_four_task_family'
    return result


def weighted_ap(y, weight, order, distinct):
    """Average precision with integer cluster multiplicities and tied scores."""
    yy, ww = y[order], weight[order]
    tp = np.cumsum(ww * yy)
    fp = np.cumsum(ww * (1 - yy))
    if tp[-1] <= 0:
        return np.nan
    denominator = tp[distinct] + fp[distinct]
    precision = np.divide(tp[distinct], denominator, out=np.zeros_like(denominator, dtype=float),
                          where=denominator > 0)
    recall = tp[distinct] / tp[-1]
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def paired_draws(frame, models, kind, replicates=500, seed=90217):
    base = frame.loc[frame.model.eq(models[0])].sort_values(['grant_year', 'patent_id']).reset_index(drop=True)
    if base.duplicated(['grant_year', 'patent_id']).any():
        raise ValueError('Repeated patent rows within a model')
    keys = base[['grant_year', 'patent_id']]
    columns = []
    for model in models:
        rows = frame.loc[frame.model.eq(model)].sort_values(['grant_year', 'patent_id']).reset_index(drop=True)
        if not rows[['grant_year', 'patent_id']].equals(keys) or not np.array_equal(rows.y_true, base.y_true):
            raise ValueError(f'Forecasts are not paired on identical patents and labels: {model}')
        if not np.array_equal(rows.assignee_cluster_key.astype(str), base.assignee_cluster_key.astype(str)):
            raise ValueError('Assignee cluster keys differ between models')
        columns.append(rows.prediction.to_numpy(float))
    predictions = np.column_stack(columns)
    y = base.y_true.to_numpy(float)
    cluster_ids = sorted(base.assignee_cluster_key.astype(str).unique())
    lookup = {value: i for i, value in enumerate(cluster_ids)}
    cluster = base.assignee_cluster_key.astype(str).map(lookup).to_numpy(int)
    groups = len(cluster_ids)
    annual = [np.flatnonzero(base.grant_year.eq(year)) for year in sorted(base.grant_year.unique())]
    if kind == 'count':
        predictions = np.clip(predictions, 1e-12, None)
        loss = 2 * (xlogy(y[:, None], y[:, None] / predictions) - y[:, None] + predictions)
        primary = 'poisson_deviance'
    elif kind == 'continuous':
        loss = (y[:, None] - predictions) ** 2
        primary = 'rmse'
    else:
        loss, primary = None, 'pr_auc'
        orders, distinct = [], []
        for indices in annual:
            oo, ee = [], []
            for j in range(len(models)):
                order = np.argsort(-predictions[indices, j], kind='stable')
                scores = predictions[indices, j][order]
                oo.append(order)
                ee.append(np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1])
            orders.append(oo)
            distinct.append(ee)
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, len(models)), float)
    for rep in range(replicates):
        multiplicity = np.bincount(rng.integers(0, groups, size=groups), minlength=groups)
        weights = multiplicity[cluster]
        values = []
        for yi, indices in enumerate(annual):
            w = weights[indices]
            if w.sum() == 0:
                raise ValueError('Bootstrap draw has no observations in a test year')
            if kind == 'count':
                value = np.sum(loss[indices] * w[:, None], axis=0) / w.sum()
            elif kind == 'continuous':
                value = np.sqrt(np.sum(loss[indices] * w[:, None], axis=0) / w.sum())
            else:
                value = np.array([weighted_ap(y[indices], w, orders[yi][j], distinct[yi][j])
                                  for j in range(len(models))])
            values.append(value)
        draws[rep] = np.mean(values, axis=0)
    if not np.isfinite(draws).all():
        raise ValueError('Undefined bootstrap metric; inspect class counts and cluster structure')
    return draws, primary, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=str)
    parser.add_argument('--metrics', type=str)
    parser.add_argument('--replicates', type=int, default=500)
    parser.add_argument('--seed', type=int, default=90217)
    parser.add_argument('--pairs', help='Comma-separated candidate:baseline pairs')
    args = parser.parse_args()
    started = time.time()
    if args.replicates < 2:
        raise ValueError('At least two bootstrap replicates are required')
    yearly = pd.read_csv(args.metrics or output('results/forecast_metrics_by_year.csv'))
    predictions = pd.read_parquet(args.predictions or output('results/forecast_predictions.parquet'))
    for frame in (yearly, predictions):
        if 'learner' not in frame:
            frame['learner'] = 'glm'
    available = [metric for metric in METRICS if metric in yearly]
    summary = yearly.groupby(['learner', 'target', 'model'], as_index=False)[available].mean()
    count_name = 'n_test' if 'n_test' in yearly else 'n'
    counts = yearly.groupby(['learner', 'target', 'model'], as_index=False)[count_name].sum().rename(columns={count_name: 'observations'})
    summary = summary.merge(counts, validate='one_to_one')
    summary.to_csv(output('results/forecast_metrics_summary.csv'), index=False)
    pairs = [tuple(pair.split(':')) for pair in args.pairs.split(',')] if args.pairs else DEFAULT_PAIRS
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError('Each pair must have candidate:baseline form')
    intervals, differences = [], []
    for (learner, target), frame in predictions.groupby(['learner', 'target']):
        kind = OUTCOMES[target][1]
        models = sorted(frame.model.unique())
        active_pairs = [pair for pair in pairs if set(pair) <= set(models)]
        if not active_pairs:
            continue
        draws, primary, groups = paired_draws(frame, models, kind, args.replicates, args.seed)
        np.save(output(f'results/bootstrap_{learner}_{target}_{primary}.npy'), draws)
        points = summary.loc[summary.learner.eq(learner) & summary.target.eq(target)].set_index('model')
        for candidate, baseline in active_pairs:
            delta = draws[:, models.index(candidate)] - draws[:, models.index(baseline)]
            low, high = np.quantile(delta, [.025, .975])
            value = float(points.loc[candidate, primary] - points.loc[baseline, primary])
            intervals.append({'learner': learner, 'target': target, 'metric': primary,
                              'candidate': candidate, 'baseline': baseline,
                              'delta_candidate_minus_baseline': value, 'ci95_low': low, 'ci95_high': high,
                              'relative_change_percent': 100 * value / points.loc[baseline, primary],
                              'better_direction': 'higher' if kind == 'binary' else 'lower',
                              'replicates': args.replicates, 'clusters': groups,
                              'centered_bootstrap_p': centered_bootstrap_p(delta, value),
                              'estimator': 'equal-year mean; paired assignee clusters across years; conditional on fitted models'})
            for metric in available:
                if pd.notna(points.loc[candidate, metric]) and pd.notna(points.loc[baseline, metric]):
                    differences.append({'learner': learner, 'target': target, 'metric': metric,
                                        'candidate': candidate, 'baseline': baseline,
                                        'delta_candidate_minus_baseline': points.loc[candidate, metric] - points.loc[baseline, metric]})
        log(f'Paired cluster intervals: {learner} {target}')
    inference = add_holm_four_tasks(pd.DataFrame(intervals))
    inference.to_csv(output('results/paired_cluster_intervals.csv'), index=False)
    inference.to_csv(output('results/paired_holm_four_tasks.csv'), index=False)
    pd.DataFrame(differences).to_csv(output('results/forecast_metric_differences.csv'), index=False)
    save('metadata/summary_complete.json', {'complete': True, 'replicates': args.replicates,
                                          'seed': args.seed, 'seconds': round(time.time() - started, 2),
                                          'p_value': 'Two-sided centered-null bootstrap sensitivity p with plus-one correction; finite resampling resolution.',
                                          'holm_family': 'Four study endpoints, separately for each learner and candidate/baseline pair; incomplete families remain unadjusted.',
                                          'interpretation': 'Intervals condition on fitted models and observed test years; no training refit or temporal extrapolation uncertainty.'})


if __name__ == '__main__':
    main()
