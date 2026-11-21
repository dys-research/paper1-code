"""Create deduplicated source-text inputs for the fixed patent encoder."""
from common import *
import argparse
import collections

KINDS = ['P', 'M', 'E', 'PM_config', 'ME_config', 'overall_ta', 'overall_tac', 'title', 'abstract', 'claim_1']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, default=out('encoding/full_frozen'))
    args = parser.parse_args()
    job = args.job.resolve()
    if not job.is_relative_to(ROOT):
        raise ValueError('Encoding job must remain inside the configured measurement work directory')
    if (job / 'vectors.npy').exists():
        raise FileExistsError('Existing vectors found; use a separate work directory for different inputs')
    job.mkdir(parents=True, exist_ok=True)
    revision = str(setting('encoder', 'revision', '')).strip()
    if not revision or revision in ['main', 'master']:
        raise ValueError('Set encoder.revision to an immutable model commit before preparing inputs')
    connection = connect()
    connection.execute('CREATE VIEW roles AS SELECT * FROM ' + readp(out('data/role_inputs_primary.parquet')))
    connection.execute('CREATE TABLE wide AS SELECT patent_id,' + ','.join(
        "max(CASE WHEN role='" + role + "' AND usable_for_primary_embedding THEN embedding_text END) " + role +
        ",max(CASE WHEN role='" + role + "' THEN extraction_model END) " + role + '_model' +
        ",max(CASE WHEN role='" + role + "' THEN source_field END) " + role + '_source'
        for role in 'PME') + ' FROM roles GROUP BY patent_id')
    query = '''SELECT t.patent_id,t.patent_title title,t.patent_abstract abstract,t.independent_claim_text claim_1,
        b.grant_date,b.grant_year,b.technical_subfield,b.is_focal_candidate,b.is_history_warmup,
        w.* EXCLUDE(patent_id) FROM ''' + readp(STAGE1 / 'data/a61b_text_base_1995_2020.parquet') + ''' t
        JOIN ''' + readp(out('data/patent_panel_base.parquet')) + ''' b USING(patent_id)
        LEFT JOIN wide w USING(patent_id) WHERE t.full_input_available ORDER BY t.patent_id'''
    unique = {}
    counts = collections.Counter()
    task_count = 0
    with (job / 'unique_texts.jsonl').open('w', encoding='utf-8') as unique_stream, \
            (job / 'task_metadata.jsonl').open('w', encoding='utf-8') as task_stream:
        for batch in connection.execute(query).to_arrow_reader(batch_size=1000):
            for record in batch.to_pylist():
                pid = str(record['patent_id'])
                title, abstract, claim = [canonical(record[key]) for key in ['title', 'abstract', 'claim_1']]
                texts = {role: record.get(role) for role in 'PME'}
                if record['P'] and record['M']:
                    texts['PM_config'] = f"Problem: {record['P']}\nMechanism: {record['M']}"
                if record['M'] and record['E']:
                    texts['ME_config'] = f"Mechanism: {record['M']}\nEffect: {record['E']}"
                texts.update(overall_ta=f'Title: {title}\nAbstract: {abstract}',
                             overall_tac=f'Title: {title}\nAbstract: {abstract}\nClaim 1: {claim}',
                             title=title, abstract=abstract, claim_1=claim)
                for kind in KINDS:
                    text = texts.get(kind)
                    if not text:
                        continue
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if digest not in unique:
                        unique[digest] = len(unique)
                        unique_stream.write(json.dumps({'embedding_row': unique[digest], 'text_sha256': digest,
                                                        'text': text}, ensure_ascii=False) + '\n')
                    task = {'task_id': pid + ':' + kind, 'patent_id': pid, 'kind': kind,
                            'embedding_row': unique[digest], 'text_sha256': digest,
                            'grant_date': str(record['grant_date']), 'grant_year': record['grant_year'],
                            'technical_subfield': record['technical_subfield'],
                            'is_focal_candidate': record['is_focal_candidate'], 'is_history_warmup': record['is_history_warmup'],
                            'extraction_model': record.get(kind + '_model') if kind in ('P', 'M', 'E') else None,
                            'source_field': record.get(kind + '_source') if kind in ('P', 'M', 'E') else None}
                    task_count += 1
                    task_stream.write(json.dumps(task, ensure_ascii=False) + '\n')
                    counts[kind] += 1
    if not task_count:
        raise ValueError('No complete source texts are available for encoding')
    connection.execute('COPY (SELECT CAST(patent_id AS VARCHAR) patent_id,* EXCLUDE(patent_id) FROM read_json_auto(' + qp(job / 'task_metadata.jsonl') + ')) TO ' + qp(job / 'task_metadata.parquet') + ' (FORMAT PARQUET, COMPRESSION ZSTD)')
    local = input_path('encoder_model', required=False)
    hashes = {}
    if local and local.is_dir():
        for name in ['modules.json', 'config.json', 'sentence_bert_config.json', '1_Pooling/config.json']:
            if (local / name).is_file():
                hashes[name] = sha(local / name)
    configuration = {'model_id': setting('encoder', 'model', 'AI-Growth-Lab/PatentSBERTa'),
                     'checkpoint_revision': revision, 'configuration_hashes': hashes,
                     'local_model': str(local) if local else '', 'unique_texts': len(unique),
                     'inputs_sha256': sha(job / 'unique_texts.jsonl'),
                     'tasks_sha256': sha(job / 'task_metadata.jsonl'),
                     'task_metadata_parquet_sha256': sha(job / 'task_metadata.parquet'),
                     'scope': 'COMPLETE_FOCAL_HISTORY_INPUTS', 'representations': KINDS,
                     'max_tokens': 512, 'pooling': 'CLS', 'embedding_dimensions': 768, 'reuse_cache': []}
    (job / 'encoder_freeze.json').write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding='utf-8')
    finish('22_encoding_inputs', {'unique_texts': len(unique), 'tasks': task_count,
                                  'representations': dict(counts), 'paid_api_calls': 0})


if __name__ == '__main__':
    main()
