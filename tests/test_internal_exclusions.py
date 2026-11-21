"""Independent checks of internal-link masks and historical/future time rules."""
import unittest
import numpy as np
import pandas as pd

from scripts.robustness.internal_links import FamilyIndex, OwnerIndex
from scripts.robustness.internal_exclusions import history_topk, reuse_block


def family_fixture():
    # 10/20 join in 2019; 30 joins them in 2021. 40 and 50 remain separate.
    minimum = np.iinfo(np.int32).min
    day = lambda value: np.datetime64(value, 'D').astype(np.int32)
    parent = np.array([5, 5, 6, -1, -1, 6, -1], np.int32)
    upper = np.full(len(parent), -1, np.int32)
    ix = np.flatnonzero(parent >= 0)
    upper[ix] = parent[parent[ix]]
    return FamilyIndex({'ids': np.array([10, 20, 30, 40, 50]),
        'ancestor': np.stack([parent, upper]),
        'day': np.array([minimum] * 5 + [day('2019-06-01'), day('2021-06-01')], np.int32)})


def owners_fixture():
    return OwnerIndex(pd.DataFrame({'patent_id': ['10', '10', '20', '30', '40', '40'],
                                   'entity_id': [100, 200, 300, 400, 200, 500]}))


class InternalExclusionTest(unittest.TestCase):
    def test_any_owner_including_secondary_and_unknown(self):
        owners = owners_fixture()
        result = owners.overlap(owners.rows(['10', '50']), owners.rows(['40', '20', '50']))
        np.testing.assert_array_equal(result, [[True, False, False], [False, False, False]])

    def test_history_does_not_use_future_family_merges(self):
        family, owners = family_fixture(), owners_fixture()
        q = family.enrich(pd.DataFrame({'patent_id': ['10'], 'grant_date': pd.to_datetime(['2020-01-01'])}))
        h = family.enrich(pd.DataFrame({'patent_id': ['20', '30', '40', '50'],
            'grant_date': pd.to_datetime(['2011-01-01', '2019-01-01', '2018-01-01', '2017-01-01'])}))
        query = np.array([[1, 0]], np.float32)
        history = np.array([[1, 0], [.8, .6], [1, 0], [0, 1]], np.float32)
        results = history_topk(query, history, q, h, owners, k=2, query_rows=1, history_rows=1)
        # 20 is already in-family; 40 shares the query's second owner. Keep 30 and unknown-owner 50.
        self.assertEqual(results['strict10']['history_n'][0], 2)
        self.assertAlmostEqual(results['strict10']['score'][0], .6, places=6)
        self.assertEqual(results['window5']['history_n'][0], 3)
        self.assertAlmostEqual(results['window5']['score'][0], .1, places=6)
        insufficient = history_topk(query, history, q, h, owners, k=3, history_rows=2)
        self.assertTrue(np.isnan(insufficient['strict10']['score'][0]))

    def test_future_family_is_evaluated_at_candidate_grant(self):
        family, owners = family_fixture(), owners_fixture()
        query = family.enrich(pd.DataFrame({'patent_id': ['10'],
            'grant_date': pd.to_datetime(['2020-01-01']), 'label_window_end': pd.to_datetime(['2025-01-01'])}))
        candidates = family.enrich(pd.DataFrame({'patent_id': ['20', '30', '40', '50'],
            'grant_date': pd.to_datetime(['2020-07-01', '2021-07-01', '2022-01-01', '2025-01-01'])}))
        values = reuse_block(np.array([[1, 0]], np.float32), np.tile([[1., 0.]], (4, 1)).astype(np.float32),
                             query, candidates, owners)
        self.assertEqual(values['reuse80_all'][0], 4)
        self.assertEqual(values['reuse80_no_samefamily'][0], 2)
        self.assertEqual(values['reuse80_no_sameassignee'][0], 3)
        self.assertEqual(values['reuse80_no_either'][0], 1)
        self.assertEqual(values['reuse80_known_owner_pairs'][0], 3)
        # Before the 2021 family merge, patent 30 is still an external-family text.
        candidates.loc[1, 'grant_date'] = pd.Timestamp('2021-05-01')
        candidates = family.enrich(candidates)
        values = reuse_block(np.array([[1, 0]], np.float32), np.tile([[1., 0.]], (4, 1)).astype(np.float32),
                             query, candidates, owners)
        self.assertEqual(values['reuse80_no_samefamily'][0], 3)
        self.assertEqual(values['reuse80_no_either'][0], 2)


if __name__ == '__main__':
    unittest.main()
