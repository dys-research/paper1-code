"""Exercise source eligibility, CPC order, independent claims, and cluster IDs."""
from datetime import date
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]


class DataPipelineTest(unittest.TestCase):
    def test_normalized_inputs_to_texts_and_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "normalized"
            for folder in ("02_core", "01_stage_parquet", "03_metrics"):
                (source / folder).mkdir(parents=True)

            def write(relative, records):
                pd.DataFrame(records).to_parquet(source / relative, index=False)

            ids = [1, 2, 3, 4]
            write("02_core/patent_dates.parquet", [
                dict(patent_id=i, patent_date="1999-01-05" if i == 1 else "2005-01-04",
                     filing_date="1998-01-01" if i == 1 else "2003-01-01",
                     grant_year=1999 if i == 1 else 2005, filing_year=1998 if i == 1 else 2003)
                for i in ids])
            write("01_stage_parquet/patent.parquet", [
                dict(patent_id=i, patent_type="design" if i == 4 else "utility", withdrawn=int(i == 3))
                for i in ids])
            cpc = [dict(patent_id=i, cpc_sequence=0, cpc_group="A61B5/00", cpc_subclass="A61B",
                        cpc_class="A61", cpc_section="A") for i in ids]
            cpc.append(dict(patent_id=2, cpc_sequence=0, cpc_group="A61B3/00", cpc_subclass="A61B",
                            cpc_class="A61", cpc_section="A"))
            # A later, lower-numbered code must not override the first listed code.
            cpc.append(dict(patent_id=1, cpc_sequence=1, cpc_group="A61B1/00", cpc_subclass="A61B",
                            cpc_class="A61", cpc_section="A"))
            write("02_core/patent_cpc.parquet", cpc)
            write("02_core/patent_primary_field.parquet", [
                dict(patent_id=i, cpc_subclass="A61B", cpc_group="A61B5/00") for i in ids])
            write("02_core/patent_text.parquet", [
                dict(patent_id=i, patent_title=f"Task {i}", patent_abstract="Improves accuracy with a sensor.")
                for i in ids])
            write("02_core/patent_assignee.parquet", [dict(patent_id=1, assignee_id="a", assignee_sequence=0)])
            write("02_core/patent_application.parquet", [dict(patent_id=i, application_id=f"app{i}") for i in ids])
            write("03_metrics/abstract_npl_metrics.parquet", [
                dict(patent_id=i, non_patent_reference_count=0, doi_like_reference_count=0,
                     title_word_count=2, abstract_word_count=6) for i in ids])
            write("03_metrics/assignee_family_value_metrics.parquet", [
                dict(patent_id=i, grant_lag_years=2.0, num_claims=3, related_us_document_count=0,
                     foreign_priority_count=0, pct_document_count=0, assignee_prior_patents_mean=1.0)
                for i in ids])
            claims_root = base / "claims"
            claims_root.mkdir()
            rows = []
            for i in ids:
                for sequence, dependency, text in [
                    (0, "", "(canceled)"), (1, "1", "A dependent sensor."),
                    (2, "", "A sensor measures pressure.")]:
                    rows.append(dict(patent_id=i, claim_sequence=sequence, claim_number=sequence + 1,
                                     claim_text=text, dependent=dependency))
            claims = pd.DataFrame(rows)
            claims.loc[claims.patent_id.eq(1)].to_parquet(claims_root / "historical.parquet", index=False)
            with zipfile.ZipFile(claims_root / "additional.tsv.zip", "w") as archive:
                archive.writestr("claims.tsv", claims.loc[claims.patent_id.ne(1)].to_csv(sep="\t", index=False))
            config = base / "config.toml"
            config.write_text('[paths]\nwork_dir = "work"\n[inputs]\npatentsview_root = "normalized"\nclaims_root = "claims"\n')
            result = subprocess.run([sys.executable, str(REPOSITORY / "run.py"), "data", "--config", str(config)],
                                    cwd=base, env=os.environ.copy(), capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = base / "work/data/data"
            sample = pd.read_parquet(output / "a61b_candidates_1995_2020.parquet").set_index("patent_id")
            self.assertEqual(sample.index.tolist(), ["1", "2"])
            self.assertEqual(sample.loc["1", "first_a61b_group"], "A61B5/00")
            self.assertEqual(sample.loc["2", "first_a61b_group"], "A61B3/00")
            self.assertEqual(sample.loc["2", "label_window_end"], date(2010, 1, 4))
            self.assertTrue(sample.loc["1", "is_history_warmup"])
            self.assertTrue(sample.loc["2", "is_focal_candidate"])
            texts = pd.read_parquet(output / "a61b_text_base_1995_2020.parquet")
            self.assertTrue(texts.full_input_available.all())
            self.assertEqual(texts.independent_claim_sequence.tolist(), [2, 2])
            controls = pd.read_parquet(output / "candidate_control_availability.parquet").set_index("patent_id")
            self.assertEqual(controls.loc["1", "prospective_cluster_key"], "assignee:a")
            self.assertEqual(controls.loc["2", "prospective_cluster_key"], "patent:2")


if __name__ == "__main__":
    unittest.main()
