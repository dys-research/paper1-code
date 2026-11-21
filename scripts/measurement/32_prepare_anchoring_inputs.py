"""Prepare all eligible backward-reference denominators and observed M pairs."""
from common import *
import argparse


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    connection = connect()
    connection.execute('CREATE VIEW p AS SELECT * FROM ' + readp(out('data/panel_covariates_and_labels_partial.parquet')))
    connection.execute('CREATE VIEW r AS SELECT * FROM ' + readp(out('data/role_inputs_primary.parquet')))
    connection.execute('''CREATE TABLE backward AS SELECT e.citing_id,e.cited_id,e.citing_date,e.cited_date
        FROM ''' + readp(out('data/study_citation_edges.parquet')) + ''' e
        JOIN p ON e.citing_id=CAST(p.patent_id AS BIGINT)
        WHERE p.is_focal_candidate AND e.cited_date<p.grant_date''')
    connection.execute('''CREATE TABLE coverage AS SELECT p.patent_id,p.grant_date,
        CAST(p.M_usable AS BOOLEAN) query_M_valid,count(e.cited_id) historical_reference_n,
        count(e.cited_id) FILTER(WHERE coalesce(cm.usable_for_primary_embedding,FALSE)) covered_M_n,
        count(e.cited_id) FILTER(WHERE cm.patent_id IS NULL) outside_base_reference_n
        FROM p LEFT JOIN backward e ON CAST(p.patent_id AS BIGINT)=e.citing_id
        LEFT JOIN r cm ON e.cited_id=CAST(cm.patent_id AS BIGINT) AND cm.role='M'
        WHERE p.is_focal_candidate GROUP BY p.patent_id,p.grant_date,p.M_usable''')
    parquet(connection, '''SELECT *,CASE WHEN historical_reference_n>0 THEN
        covered_M_n::DOUBLE/historical_reference_n END M_coverage,
        query_M_valid AND covered_M_n>0 anchoring_computable FROM coverage''',
        'data/anchoring_coverage_after_bulk.parquet')
    connection.execute('CREATE VIEW tasks AS SELECT * FROM ' + readp(out('encoding/full_frozen/task_metadata.parquet')))
    parquet(connection, '''SELECT CAST(e.citing_id AS VARCHAR) patent_id,CAST(e.cited_id AS VARCHAR) cited_id,
        q.embedding_row query_embedding_row,h.embedding_row cited_embedding_row
        FROM backward e JOIN tasks q ON e.citing_id=CAST(q.patent_id AS BIGINT) AND q.kind='M'
        JOIN tasks h ON e.cited_id=CAST(h.patent_id AS BIGINT) AND h.kind='M'
        ORDER BY e.citing_id,e.cited_id''', 'data/anchoring_valid_pairs_for_encoding.parquet')
    finish('32_anchoring_inputs', {'eligible_references': connection.execute('SELECT count(*) FROM backward').fetchone()[0],
                                   'valid_pairs': pq.ParquetFile(out('data/anchoring_valid_pairs_for_encoding.parquet')).metadata.num_rows})


if __name__ == '__main__':
    main()
