"""Collect source control fields; date-truncated histories are rebuilt downstream."""
from common import connect, output, parquet, qp, receipt, source


def main():
    connection = connect()
    candidates = output("data/a61b_candidates_1995_2020.parquet")
    npl = source("03_metrics/abstract_npl_metrics.parquet")
    value = source("03_metrics/assignee_family_value_metrics.parquet")
    assignees = source("02_core/patent_assignee.parquet")
    applications = source("02_core/patent_application.parquet")
    connection.execute("CREATE VIEW candidates AS SELECT * FROM read_parquet(" + qp(candidates) + ")")
    connection.execute("""CREATE TABLE first_assignee AS
      SELECT a.patent_id,a.assignee_id,a.assignee_sequence
      FROM read_parquet(""" + qp(assignees) + """) a
      JOIN candidates b ON a.patent_id=CAST(b.patent_id AS BIGINT)
      QUALIFY row_number() OVER(PARTITION BY a.patent_id ORDER BY a.assignee_sequence NULLS LAST,a.assignee_id)=1""")
    connection.execute("""CREATE TABLE controls AS
      SELECT b.patent_id,b.grant_year,b.is_focal_candidate,
        n.non_patent_reference_count,n.doi_like_reference_count,n.title_word_count,n.abstract_word_count,
        v.grant_lag_years,v.num_claims,v.related_us_document_count,v.foreign_priority_count,v.pct_document_count,
        v.assignee_prior_patents_mean,a.assignee_id first_assignee_id,a.assignee_sequence,
        CASE WHEN a.assignee_id IS NULL THEN 'patent:'||b.patent_id
          ELSE 'assignee:'||CAST(a.assignee_id AS VARCHAR) END prospective_cluster_key,
        pa.application_id
      FROM candidates b
      LEFT JOIN read_parquet(""" + qp(npl) + """) n ON n.patent_id=CAST(b.patent_id AS BIGINT)
      LEFT JOIN read_parquet(""" + qp(value) + """) v ON v.patent_id=CAST(b.patent_id AS BIGINT)
      LEFT JOIN first_assignee a ON a.patent_id=CAST(b.patent_id AS BIGINT)
      LEFT JOIN read_parquet(""" + qp(applications) + """) pa ON pa.patent_id=CAST(b.patent_id AS BIGINT)""")
    total, unique = connection.execute("SELECT count(*),count(DISTINCT patent_id) FROM controls").fetchone()
    if total != unique:
        raise ValueError("Normalized source metrics/applications must be unique per patent_id")
    parquet(connection, "SELECT * FROM controls ORDER BY grant_year,patent_id", "data/candidate_control_availability.parquet")
    receipt("checks/controls.json", [candidates, npl, value, assignees, applications], patents=total,
            historical_fields="Source history values are audit fields only. Recompute main histories using preceding events.")
    print(f"Prepared {total:,} control records.")


if __name__ == "__main__":
    main()
