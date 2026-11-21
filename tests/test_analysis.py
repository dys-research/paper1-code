"""Small checks for forecasting information boundaries and paired inference."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from scripts.analysis import common, explanatory, forecast, prediction, summarize


def measured_fixture():
    rng = np.random.default_rng(19)
    years = np.repeat(np.arange(2005, 2016), 50)
    n = len(years)
    d = pd.DataFrame({'patent_id': [str(i) for i in range(n)], 'grant_year': years,
                      'is_focal_candidate': True, 'technical_subfield': np.where(np.arange(n) % 2, 'A61B5', 'A61B6'),
                      'assignee_cluster_key': ['owner_' + str(i % 25) for i in range(n)],
                      'assignee_missing': False})
    d['grant_date'] = pd.to_datetime(d.grant_year.astype(str) + '-01-02')
    d['label_window_end'] = d.grant_date + pd.DateOffset(years=5)
    d['label_available_date'] = d.label_window_end
    for col in common.CONTROLS.values():
        d[col] = rng.uniform(0, 2, n)
    for col in common.NOVELTY.values():
        d[col] = rng.uniform(.1, .9, n)
    for col in ['title_novelty_raw', 'abstract_novelty_raw', 'claim1_novelty_raw']:
        d[col] = rng.uniform(.1, .9, n)
    d['anchoring_raw'] = rng.uniform(.3, .8, n)
    d['M_coverage'] = rng.uniform(.1, .9, n)
    d['forward_cites_5y'] = rng.poisson(3, n)
    d['new_field_cites_5y'] = rng.poisson(.3, n)
    d['semantic_reuse_5y'] = np.log1p(rng.poisson(2, n))
    d['top5_impact_with_ties'] = (np.arange(n) % 20 == 0).astype(int)
    return d


class AnalysisChecks(unittest.TestCase):
    def test_weighted_ap_matches_sample_weight_with_ties(self):
        rng = np.random.default_rng(7)
        truth = rng.integers(0, 2, 300)
        scores = np.round(rng.random(300), 1)
        weight = rng.integers(0, 4, 300)
        order = np.argsort(-scores, kind='stable')
        distinct = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(scores) - 1]
        self.assertAlmostEqual(summarize.weighted_ap(truth, weight, order, distinct),
                               average_precision_score(truth, scores, sample_weight=weight))

    def test_fitting_transforms_do_not_learn_from_future_rows(self):
        raw = common.rawfeatures(measured_fixture())
        fit = raw.iloc[:150].copy()
        future = raw.iloc[150:].copy()
        fit.loc[0, 'P'] = np.nan
        pre = prediction.Preprocess().fit(fit)
        before = pre.mean.copy()
        future['log_claims'] = 1e9
        transformed = pre.transform(future)
        np.testing.assert_array_equal(pre.mean, before)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(pre.medians['P'], fit.P.median())
        self.assertEqual(fit.P.isna().sum(), 1)

    def test_clustered_explanatory_estimators_fit_same_sample(self):
        d = measured_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(common, 'ROOT', Path(temporary)):
                x, specs, fixed_effects, _ = explanatory.prepare(d, 'synthetic')
                rows = []
                for target, family in [('citations', 'count'), ('semantic', 'continuous')]:
                    rows.append(explanatory.fitmodel(d, x, fixed_effects, specs['M3'],
                                                    target, family, 'synthetic', 'M3'))
                self.assertEqual(rows[0]['nobs'], rows[1]['nobs'])
                self.assertEqual(rows[0]['clusters'], 25)
                self.assertTrue(all(row['converged'] for row in rows))

    def test_external_vectors_align_by_identifier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vectors = np.arange(3 * 768, dtype=np.float32).reshape(3, 768)
            np.save(root / 'vectors.npy', vectors)
            pd.DataFrame({'patent_id': ['b', 'c', 'a']}).to_csv(root / 'ids.csv', index=False)
            paths = {'overall_vectors': root / 'vectors.npy', 'overall_vector_ids': root / 'ids.csv'}
            with patch.object(forecast, 'input_path', side_effect=lambda key, required=True: paths.get(key)):
                loaded, rows, provenance = forecast.aligned_overall_vectors(pd.DataFrame({'patent_id': ['a', 'b']}))
                np.testing.assert_array_equal(rows, [2, 0])
                np.testing.assert_array_equal(loaded[rows], vectors[[2, 0]])
                loaded._mmap.close()  # Release the read-only handle before Windows removes the fixture.

    def test_small_forecast_then_cluster_intervals(self):
        d = measured_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            panel = root / 'panel.parquet'
            d.to_parquet(panel, index=False)
            rng = np.random.default_rng(5)
            np.save(root / 'vectors.npy', rng.normal(size=(len(d), 768)).astype(np.float32))
            d[['patent_id']].to_csv(root / 'ids.csv', index=False)
            args = SimpleNamespace(features=None, targets='semantic', learner='glm',
                                   first_year=2015, last_year=2015, overwrite=False)
            def configured_input(key, required=True):
                return {'panel': panel, 'overall_vectors': root / 'vectors.npy',
                        'overall_vector_ids': root / 'ids.csv'}.get(key)
            with patch.object(common, 'ROOT', root / 'analysis'), \
                 patch.object(forecast, 'input_path', side_effect=configured_input), \
                 patch.object(forecast, 'readpanel', return_value=d), \
                 patch.object(prediction, 'ALPHAS', [.1]):
                forecast.run(args)
                metrics = pd.read_csv(root / 'analysis/results/forecast_metrics_by_year.csv')
                self.assertEqual(set(metrics.model), set(forecast.DEFAULT_FEATURES))
                self.assertTrue((metrics.n_train == 150).all())
                self.assertTrue((metrics.n_valid == 100).all())
                self.assertTrue((metrics.n_refit == 250).all())
                estimates = pd.read_parquet(root / 'analysis/results/forecast_predictions.parquet')
                draws, metric, clusters = summarize.paired_draws(estimates, ['common', 'structure'],
                                                                'continuous', replicates=10)
                self.assertEqual(metric, 'rmse')
                self.assertEqual(clusters, 25)
                self.assertEqual(draws.shape, (10, 2))
                self.assertTrue(np.isfinite(draws).all())
                # Existing completed jobs can be loaded without fitting again.
                with patch.object(prediction, 'learn', side_effect=AssertionError('unexpected refit')):
                    forecast.run(args)


if __name__ == '__main__':
    unittest.main()
