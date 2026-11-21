"""Scientific boundaries and CLI handoffs for the added manuscript diagnostics."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.analysis import summarize
from scripts.robustness import conditional_sample, kelly, paired_text


class HolmTest(unittest.TestCase):
    def test_original_four_endpoint_example_and_separate_learners(self):
        raw = [340 / 501, 2 / 501, 1 / 501, 6 / 501]
        frame = pd.DataFrame({'learner': 'tree', 'candidate': 'structure', 'baseline': 'common',
                              'target': ['citations', 'semantic', 'newfield', 'top5'],
                              'centered_bootstrap_p': raw})
        other = frame.copy()
        other['learner'] = 'glm'
        other['centered_bootstrap_p'] = .9
        result = summarize.add_holm_four_tasks(pd.concat([frame, other], ignore_index=True))
        np.testing.assert_allclose(result.iloc[:4].p_Holm_four_endpoints, [340 / 501, 6 / 501, 4 / 501, 12 / 501])
        np.testing.assert_allclose(result.iloc[4:].p_Holm_four_endpoints, 1.)
        self.assertAlmostEqual(summarize.centered_bootstrap_p(np.array([.2, -.1, .3]), .2), .5)

    def test_incomplete_family_is_not_mislabelled_and_duplicates_fail(self):
        frame = pd.DataFrame([{'learner': 'tree', 'candidate': 'structure', 'baseline': 'common',
                               'target': 'top5', 'centered_bootstrap_p': .01}])
        result = summarize.add_holm_four_tasks(frame)
        self.assertTrue(result.p_Holm_four_endpoints.isna().all())
        self.assertEqual(result.holm_status.iloc[0], 'incomplete_four_task_family')
        with self.assertRaisesRegex(ValueError, 'Repeated task'):
            summarize.add_holm_four_tasks(pd.concat([frame, frame]))


class ConditionalSampleTest(unittest.TestCase):
    def fixture(self):
        data = pd.DataFrame({'patent_id': ['a', 'b', 'c'], 'overall_novelty_raw': [.2, .3, np.nan]})
        metrics = pd.DataFrame({'patent_id': ['c', 'b', 'a'], 'conditional_M_novelty': [.6, .5, .4]})
        variants = pd.DataFrame({'patent_id': ['b', 'a', 'c'],
            'strict10_overall_novelty_raw': [.3, .2, .1], 'window5_overall_novelty_raw': [np.nan, .1, .1]})
        return data, metrics, variants

    def test_original_rule_and_unchanged_available_default(self):
        data, metrics, variants = self.fixture()
        ids, metadata = conditional_sample.build_sample(data, metrics, variants)
        self.assertEqual(ids.patent_id.tolist(), ['a'])
        self.assertEqual(metadata['available_observation_n'], 2)
        self.assertEqual(metadata['excluded_from_available_n'], 1)
        available, _ = paired_text.select_conditional_sample(data, metrics)
        self.assertEqual(set(available.patent_id), {'a', 'b'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ids.csv'
            ids.to_csv(path, index=False)
            selected, _ = paired_text.select_conditional_sample(data, metrics, path, 'manuscript')
            self.assertEqual(selected.patent_id.tolist(), ['a'])
            pd.DataFrame({'patent_id': ['a', 'absent']}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'every ID'):
                paired_text.select_conditional_sample(data, metrics, path, 'manuscript')
        with self.assertRaisesRegex(ValueError, 'cover every focal'):
            conditional_sample.build_sample(data, metrics, variants.iloc[:1])
        with self.assertRaisesRegex(ValueError, 'unique'):
            conditional_sample.build_sample(data, pd.concat([metrics, metrics.iloc[:1]]), variants)


class KellyTest(unittest.TestCase):
    def test_windows_negative_cosines_and_means_vs_sums(self):
        # Leap-day grant: inclusive 2015-02-28 and 2025-02-28 bounds.
        dates = pd.to_datetime(['2015-02-28', '2015-02-27', '2020-02-29',
                                '2025-02-28', '2025-03-01', '2019-01-01']).to_numpy('datetime64[D]').astype(np.int64)
        query = np.array([[1., 0.]], np.float32)
        candidates = np.array([[1., 0.], [1., 0.], [1., 0.], [1., 0.], [1., 0.], [-1., 0.]], np.float32)
        day = lambda text: np.array([np.datetime64(text, 'D').astype(np.int64)])
        values = kelly.block_totals(query, day('2020-02-29'), day('2015-02-28'), day('2025-02-28'), candidates, dates)
        self.assertEqual(values['fn'][0], 1)
        self.assertEqual(values['bn'][0], 2)
        self.assertEqual(values['fsum'][0], 1)
        self.assertEqual(values['bsum'][0], 1)
        frame = pd.DataFrame({'patent_id': ['q'], 'grant_date': ['2020-02-29'],
            'label_window_end': ['2025-02-28'], 'complete_backward5y': [True], **values})
        scores = kelly.score_aggregates(frame, '2025-12-30')
        self.assertEqual(scores.positive_embedding_mean_ratio.iloc[0], 2.)
        self.assertAlmostEqual(scores.log_positive_embedding_mean_ratio.iloc[0], np.log(3))
        self.assertEqual(scores.positive_embedding_aggregate_ratio.iloc[0], 1.)
        frame.loc[0, 'complete_backward5y'] = False
        self.assertTrue(kelly.score_aggregates(frame, '2025-12-30').positive_embedding_mean_ratio.isna().all())
        frame.loc[0, 'complete_backward5y'] = True
        self.assertTrue(kelly.score_aggregates(frame, '2024-12-31').positive_embedding_mean_ratio.isna().all())
        frame.loc[0, 'bsum'] = 0
        self.assertTrue(kelly.score_aggregates(frame, '2025-12-30').positive_embedding_mean_ratio.isna().all())

    def test_vector_job_resume_and_existing_aggregates_have_identical_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / 'input'
            job.mkdir()
            pd.DataFrame({'patent_id': ['q'], 'grant_date': pd.to_datetime(['2020-02-29']),
                          'label_window_end': pd.to_datetime(['2025-02-28'])}).to_parquet(job / 'focal_metadata.parquet')
            pd.DataFrame({'patent_id': ['a', 'b', 'c'],
                          'grant_date': pd.to_datetime(['2015-02-28', '2019-01-01', '2025-02-28'])}).to_parquet(job / 'candidate_metadata.parquet')
            np.save(job / 'focal_vectors.npy', np.array([[1., 0.]], np.float32))
            np.save(job / 'candidate_vectors.npy', np.array([[1., 0.], [-1., 0.], [1., 0.]], np.float32))
            def target(name):
                path = root / 'output' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                return path
            with patch.object(kelly, 'output', side_effect=target):
                frame, _ = kelly.compute(job, query_rows=1, candidate_rows=1)
                with patch.object(kelly, 'block_totals', side_effect=AssertionError('unexpected recalculation')):
                    resumed, _ = kelly.compute(job, query_rows=1, candidate_rows=1)
                pd.testing.assert_frame_equal(frame, resumed)
                with self.assertRaisesRegex(ValueError, 'changed'):
                    kelly.compute(job, query_rows=1, candidate_rows=2)
                aggregate = root / 'aggregates.parquet'
                frame.to_parquet(aggregate, index=False)
                with patch.object(sys, 'argv', ['kelly', '--aggregates', str(aggregate)]), \
                     patch.object(kelly, 'save') as save:
                    kelly.main()
                self.assertEqual(save.call_args.args[1]['valid_mean_ratio_n'], 1)
                self.assertAlmostEqual(save.call_args.args[1]['median_log_one_plus_mean_ratio'], np.log(3))


if __name__ == '__main__':
    unittest.main()
