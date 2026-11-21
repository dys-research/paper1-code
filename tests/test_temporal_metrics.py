from datetime import date
import unittest
import numpy as np

from scripts.measurement.temporal_metrics import (
    add_years, apply_joint_residual, fit_joint_residual,
    future_text_uptake, mechanism_anchoring, temporal_topk,
)


class TemporalMetricTests(unittest.TestCase):
    def test_exact_history_window_and_insufficient_history(self):
        query = [{"patent_id": "q", "grant_date": "2020-02-29", "technical_subfield": "A"}]
        history = [
            {"patent_id": "a", "grant_date": "2010-02-28", "technical_subfield": "A"},
            {"patent_id": "b", "grant_date": "2019-01-01", "technical_subfield": "A"},
            {"patent_id": "old", "grant_date": "2010-02-27", "technical_subfield": "A"},
            {"patent_id": "same", "grant_date": "2020-02-29", "technical_subfield": "A"},
            {"patent_id": "future", "grant_date": "2020-03-01", "technical_subfield": "A"},
            {"patent_id": "other", "grant_date": "2019-01-01", "technical_subfield": "B"},
            {"patent_id": "invalid", "grant_date": "2019-01-01", "technical_subfield": "A"},
        ]
        vectors = np.array([[1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [0, 0]], dtype=float)
        result = temporal_topk(np.array([[1., 0.]]), query, vectors, history, k=2, history_rows=2)
        self.assertEqual(result["historical_count"].tolist(), [2])
        np.testing.assert_allclose(result["novelty_raw"], [.5])
        sparse = temporal_topk(np.array([[1., 0.]]), query, vectors, history, k=3)
        self.assertTrue(np.isnan(sparse["novelty_raw"][0]))
        self.assertEqual(add_years(date(2020, 2, 29), 5), date(2025, 2, 28))

    def test_future_window_and_domain_counts(self):
        query = [{"patent_id": "q", "grant_date": "2020-02-29", "cpc_subclasses": ["A61B"]}]
        future = [
            {"patent_id": "same", "grant_date": "2020-02-29", "cpc_subclasses": ["A61B"]},
            {"patent_id": "shared", "grant_date": "2020-03-01", "cpc_subclasses": ["A61B"]},
            {"patent_id": "new", "grant_date": "2025-02-28", "cpc_subclasses": ["G06F"]},
            {"patent_id": "late", "grant_date": "2025-03-01", "cpc_subclasses": ["G06F"]},
            {"patent_id": "unknown", "grant_date": "2021-01-01", "cpc_subclasses": []},
        ]
        result = future_text_uptake(np.array([[1., 0.]]), query, [(np.tile([1., 0.], (5, 1)), future)], "2025-12-30", thresholds=(.8,))
        self.assertEqual(result["counts"][0, 0].tolist(), [3, 1, 1, 1])
        self.assertTrue(result["main_label_usable"][0])
        partial = future_text_uptake(np.array([[1., 0.]]), query, [], "2024-01-01", thresholds=(.8,))
        self.assertFalse(partial["main_label_usable"][0])

    def test_anchor_denominator_retains_unobserved_references(self):
        query = [{"patent_id": "q", "grant_date": "2020-01-01"}]
        historical = [{"patent_id": "h", "grant_date": "2019-01-01"}]
        result = mechanism_anchoring(np.array([[1., 0.]]), query, np.array([[1., 0.]]), historical, {"q": ["h", "h", "outside"]})[0]
        self.assertEqual(result["historical_reference_n"], 2)
        self.assertEqual(result["M_coverage"], .5)
        self.assertEqual(result["anchoring_mean"], 1.)

    def test_residual_model_only_uses_explicit_fitting_rows(self):
        x = np.arange(4, dtype=float)
        y = 2 + 3 * x
        fitted = fit_joint_residual(x, y)
        heldout = apply_joint_residual(np.array([100.]), np.array([305.]), fitted)
        self.assertAlmostEqual(heldout[0], 3., places=8)
        self.assertEqual(fitted["n_fit"], 4)


if __name__ == "__main__":
    unittest.main()
