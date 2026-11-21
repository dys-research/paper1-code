"""Prepare date-eligible semantic-reuse metadata and exact CPC subclass bitsets."""
from common import *
from project_config import input_path
import argparse
import math
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    started = time.time()
    job = out('semantic_reuse/all_future_text')
    job.mkdir(parents=True, exist_ok=True)
    index = input_path('semantic_index')
    vectors = input_path('semantic_vectors')
    dates = NEW / '02_core/patent_dates.parquet'
    cpc = NEW / '02_core/patent_cpc.parquet'
    array = np.load(vectors, mmap_mode='r')
    if array.ndim != 2 or array.shape[1] != 768 or array.dtype not in (np.float16, np.float32, np.float64):
        raise ValueError('Semantic vectors must be a floating-point N x 768 NumPy array')
    c = connect()
    c.execute('CREATE TABLE idx AS SELECT CAST(patent_id AS BIGINT) patent_id, '
              'CAST(embedding_row AS BIGINT) embedding_row, CAST(patent_date AS DATE) grant_date, '
              'CAST(emb_valid AS BOOLEAN) emb_valid FROM ' + readp(index))
    check = rows(c, '''SELECT count(*) n,count(DISTINCT i.patent_id) unique_ids,
        count(DISTINCT embedding_row) unique_vector_rows,
        count(*) FILTER(WHERE i.patent_id IS NULL OR i.embedding_row IS NULL OR i.grant_date IS NULL) missing_key_n,
        min(embedding_row) min_row,max(embedding_row) max_row,
        count(*) FILTER(WHERE d.patent_id IS NULL OR CAST(d.patent_date AS DATE)<>i.grant_date) date_mismatch_n
        FROM idx i LEFT JOIN ''' + readp(dates) + ''' d USING(patent_id)''')[0]
    if not (check['n'] == check['unique_ids'] == check['unique_vector_rows'] and
            check['missing_key_n'] == 0 and check['date_mismatch_n'] == 0 and
            check['min_row'] is not None and check['min_row'] >= 0 and check['max_row'] < len(array)):
        raise ValueError('Semantic index IDs, vector rows or grant dates are invalid: ' + str(check))
    c.execute('CREATE TABLE q AS SELECT b.patent_id,b.grant_date,b.label_window_end,'
              'i.embedding_row,i.emb_valid FROM ' + readp(out('data/patent_panel_base.parquet')) +
              ''' b LEFT JOIN idx i ON CAST(b.patent_id AS BIGINT)=i.patent_id
              WHERE b.is_focal_candidate''')
    missing = c.execute('SELECT count(*) FROM q WHERE embedding_row IS NULL').fetchone()[0]
    if missing:
        raise ValueError(f'{missing} focal patents have no semantic vector index record')
    c.execute("CREATE TABLE candidates AS SELECT * FROM idx WHERE grant_date>(SELECT min(grant_date) FROM q) "
              "AND grant_date<=DATE " + qp(END))
    c.execute('CREATE TABLE used AS SELECT patent_id FROM candidates UNION SELECT CAST(patent_id AS BIGINT) FROM q')
    c.execute('''CREATE TABLE sets AS SELECT DISTINCT p.patent_id,p.cpc_subclass FROM ''' + readp(cpc) + ''' p
        SEMI JOIN used u USING(patent_id) WHERE p.cpc_subclass IS NOT NULL AND trim(p.cpc_subclass)<>'' ''')
    c.execute('CREATE TABLE vocabulary AS SELECT cpc_subclass,row_number() OVER(ORDER BY cpc_subclass)-1 bit_index '
              'FROM (SELECT DISTINCT cpc_subclass FROM sets)')
    vocabulary = rows(c, 'SELECT * FROM vocabulary ORDER BY bit_index')
    # Keep at least one word even if all candidate classifications are unknown.
    words = max(1, math.ceil(len(vocabulary) / 63))
    expressions = [f"CAST(bit_or(CASE WHEN v.bit_index//63={i} THEN "
                   f"(1::UBIGINT << (v.bit_index%63)) ELSE 0::UBIGINT END) AS BIGINT) cpc_bits_{i}"
                   for i in range(words)]
    c.execute('CREATE TABLE masks AS SELECT s.patent_id,' + ','.join(expressions) +
              ' FROM sets s JOIN vocabulary v USING(cpc_subclass) GROUP BY s.patent_id')
    # Cast after aggregation and after joining: never route 63-bit masks through floats.
    extra = ','.join(f'CAST(coalesce(m.cpc_bits_{i},0::BIGINT) AS BIGINT) cpc_bits_{i}' for i in range(words))
    for table, filename, ordering in [('q', 'focal_metadata.parquet', 'CAST(t.patent_id AS BIGINT)'),
                                       ('candidates', 'candidate_metadata.parquet', 't.grant_date,t.patent_id')]:
        parquet(c, 'SELECT t.*,m.patent_id IS NOT NULL cpc_known,' + extra + ' FROM ' + table +
                ' t LEFT JOIN masks m ON CAST(t.patent_id AS BIGINT)=m.patent_id ORDER BY ' + ordering,
                'semantic_reuse/all_future_text/' + filename)
    result = {
        'scope': 'ALL_ELIGIBLE_FUTURE_TEXTS_NOT_CITATION_ONLY',
        'focal_n': c.execute('SELECT count(*) FROM q').fetchone()[0],
        'candidate_n': c.execute('SELECT count(*) FROM candidates').fetchone()[0],
        'snapshot_end': END,
        'window': 'strictly after grant, through exact fifth anniversary inclusive',
        'thresholds': [.7, .8, .9], 'primary_threshold': .8,
        'primary_statistic': 'log1p(count of future texts with cosine>=0.8)',
        'auxiliary_statistic': 'sum max((cosine-tau)/(1-tau),0)',
        'domains': ['all', 'shared_subclass', 'new_field', 'classification_unknown'],
        'embedding_template': 'title + whitespace + abstract; same encoder and pooling for all rows',
        'source_vector_path': str(vectors), 'source_index_path': str(index),
        'source_index_sha256': sha(index), 'source_shape': list(array.shape), 'source_dtype': str(array.dtype),
        'source_vector_bytes': vectors.stat().st_size, 'source_vector_mtime_ns': vectors.stat().st_mtime_ns,
        'index_date_check': check, 'cpc_vocabulary': vocabulary, 'cpc_words': words,
        'cpc_bitset_version': 2, 'cpc_bitset_storage': 'int64 explicit cast after aggregate',
        'encoder_compatibility': 'Must be established by the supplier; metadata checks do not establish encoder equivalence or training cutoff.',
        'compute_complete': False, 'seconds': round(time.time() - started, 2)}
    dump('semantic_reuse/all_future_text/reuse_definition.json', result)
    finish('33_future_reuse_metadata', {k: v for k, v in result.items() if k != 'cpc_vocabulary'})


if __name__ == '__main__':
    main()
