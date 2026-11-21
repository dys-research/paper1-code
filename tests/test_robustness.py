"""Boundary check for dated lexical novelty and same-day first users."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.robustness import lexical


class LexicalChronologyTest(unittest.TestCase):
    def test_first_appearance_respects_earlier_dates_and_same_day_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            texts = pd.DataFrame({
                'patent_id': ['p1', 'p2', 'p3'],
                'patent_title': ['alpha beta', 'alpha gamma', 'alpha beta delta'],
                'patent_abstract': ['', '', ''],
                'independent_claim_text': ['', '', ''],
            })
            text_path = root / 'texts.parquet'
            texts.to_parquet(text_path, index=False)
            dates = pd.DataFrame({'patent_id': ['p1', 'p2', 'p3'],
                'grant_date': pd.to_datetime(['2000-01-01', '2000-01-01', '2001-01-01'])})

            def target(relative):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                return path

            with patch.object(lexical, 'panel', return_value=dates), \
                    patch.object(lexical, 'source', return_value=text_path), \
                    patch.object(lexical, 'output', side_effect=target), \
                    patch.object(lexical, 'save'):
                lexical.main()
            result = pd.read_parquet(root / 'data/local_lexical_features.parquet').set_index('patent_id')
            self.assertEqual(result.loc['p1', 'lexical_new_share_1'], 1.)
            self.assertEqual(result.loc['p2', 'lexical_new_share_1'], 1.)
            self.assertAlmostEqual(result.loc['p3', 'lexical_new_share_1'], 1 / 3)
            self.assertEqual(result.loc['p3', 'lexical_new_share_2'], .5)
            self.assertEqual(result.loc['p3', 'lexical_new_share_3'], 1.)


class PairedTextBoundaryTest(unittest.TestCase):
    def test_history_includes_lower_boundary_and_excludes_same_day_other_field(self):
        import numpy as np
        from scripts.robustness.paired_text import history_topk
        patents = pd.DataFrame({
            'grant_date': pd.to_datetime(['2010-01-01', '2009-12-31', '2020-01-01',
                                         '2019-01-01', '2020-01-01', '2015-01-01']),
            'technical_subfield': ['A', 'A', 'A', 'A', 'A', 'B'],
        })
        vectors = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [1, 0], [1, 0]], dtype=np.float32)
        queries = np.array([False, False, False, False, True, False])
        histories = ~queries
        scores, indices, counts = history_topk(vectors, np.arange(6), patents,
            queries, histories, k=2, qblock=1, hblock=1)
        self.assertEqual(counts[4], 2)
        self.assertEqual(indices[4].tolist(), [0, 3])
        np.testing.assert_allclose(scores[4], [1, 0])
        # The caller's valid-M mask must also restrict the P-neighbour pool.
        histories[0] = False
        scores, _, counts = history_topk(vectors, np.arange(6), patents,
            queries, histories, k=2, qblock=1, hblock=1)
        self.assertEqual(counts[4], 1)
        self.assertFalse(np.isfinite(scores[4]).all())

    def test_derangement_keeps_year_field_and_has_no_self_donor(self):
        from scripts.robustness.paired_text import derangement
        patents = pd.DataFrame({'patent_id': ['p1', 'p2', 'p3'],
            'grant_year': [2000, 2000, 2001],
            'grant_date': pd.to_datetime(['2000-01-01', '2000-12-01', '2001-01-01']),
            'technical_subfield': ['A', 'A', 'A']})
        roles = pd.DataFrame([{'patent_id': patent, 'role': role,
                              'embedding_text': patent + '_' + role}
                             for patent in patents.patent_id for role in ['P', 'M']])
        pairs, singletons = derangement(patents, roles)
        self.assertEqual(singletons, 1)
        self.assertEqual(set(zip(pairs.patent_id, pairs.partner_patent_id)),
                         {('p1', 'p2'), ('p2', 'p1')})
        self.assertTrue(pairs.text.str.startswith('Problem: ').all())
        self.assertTrue(pairs.text.str.contains('\nMechanism: ').all())


if __name__ == '__main__':
    unittest.main()
