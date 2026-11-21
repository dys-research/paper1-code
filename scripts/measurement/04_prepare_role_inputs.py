"""Revalidate canonical role spans against the actual source fields.

Run extraction/prepare_roles.py first. This step deliberately does not ingest
parsed API responses, because their provenance must be checked by that exporter.
"""
from common import *
import argparse
import pyarrow as pa

BLOCKED = {'NO_OUTPUT', 'NO_EXTRACTION', 'INPUT_CHANGED', 'INPUT_HASH_MISSING', 'API_ERROR'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roles', type=Path, help='Canonical long-form role Parquet exported by extraction/prepare_roles.py')
    args = parser.parse_args()
    source = args.roles or input_path('role_inputs', required=False) or (work_dir('extraction') / 'role_inputs_primary.parquet')
    if source.suffix != '.parquet' or not source.is_file():
        raise FileNotFoundError('Run extraction/prepare_roles.py, then provide its canonical role Parquet: ' + str(source))
    available = set(pq.ParquetFile(source).schema_arrow.names)
    needed = {'patent_id', 'role', 'present', 'source_field', 'evidence_span', 'role_status',
              'current_input_sources_sha256'}
    if not needed.issubset(available):
        raise ValueError('Canonical role file is missing ' + repr(sorted(needed - available)) +
                         '; regenerate it with extraction/prepare_roles.py')
    columns = sorted(needed | ({'confidence', 'extraction_model', 'input_hash_verified'} & available))
    selected = {}
    for batch in pq.ParquetFile(source).iter_batches(batch_size=2000, columns=columns):
        for value in batch.to_pylist():
            key = (str(value['patent_id']), value['role'])
            if value['role'] not in ('P', 'M', 'E'):
                raise ValueError('Unknown role in canonical input: ' + repr(key))
            if key in selected:
                raise ValueError('Duplicate patent-role extraction: ' + repr(key))
            selected[key] = value
    text = STAGE1 / 'data/a61b_text_base_1995_2020.parquet'
    result = []
    for batch in pq.ParquetFile(text).iter_batches(batch_size=2000):
        for patent in batch.to_pylist():
            pid = str(patent['patent_id'])
            sources = {'title': canonical(patent.get('patent_title')),
                       'abstract': canonical(patent.get('patent_abstract')),
                       'claim_1': canonical(patent.get('independent_claim_text'))}
            fingerprint = hashlib.sha256(json.dumps(sources, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            for role in 'PME':
                value = selected.get((pid, role)) or {}
                field = value.get('source_field')
                span = value.get('evidence_span') or ''
                present = value.get('present')
                if not value:
                    status = 'NO_EXTRACTION'
                elif not patent['full_input_available']:
                    status = 'INCOMPLETE_THREE_FIELD_INPUT'
                elif value.get('role_status') in BLOCKED:
                    status = value['role_status']
                elif value.get('current_input_sources_sha256') != fingerprint:
                    status = 'INPUT_CHANGED'
                elif present is False:
                    status = 'MODEL_ABSENT'
                elif present is not True:
                    status = 'INVALID_PRESENT'
                elif field not in sources:
                    status = 'INVALID_SOURCE'
                elif not isinstance(span, str) or not span or span not in sources[field]:
                    status = 'INVALID_OR_NONVERBATIM_SPAN'
                else:
                    status = 'VALID_ORIGINAL_SPAN'
                usable = status == 'VALID_ORIGINAL_SPAN'
                start = sources[field].find(span) if usable else None
                result.append({'patent_id': pid, 'role': role, 'present': present,
                               'source_field': field, 'evidence_span': span or None,
                               'embedding_text': span if usable else None,
                               'span_start': start, 'span_end': start + len(span) if usable else None,
                               'span_words': len(span.split()) if usable else None,
                               'confidence': value.get('confidence'),
                               'extraction_model': value.get('extraction_model'),
                               'current_input_sources_sha256': fingerprint,
                               'input_hash_verified': value.get('input_hash_verified'),
                               'role_status': status, 'usable_for_primary_embedding': usable})
    output = out('data/role_inputs_primary.parquet')
    pq.write_table(pa.Table.from_pylist(result), output, compression='zstd')
    connection = connect()
    summary = rows(connection, 'SELECT role,role_status,count(*) n FROM ' + readp(output) +
                   ' GROUP BY role,role_status ORDER BY role,role_status')
    finish('04_role_inputs', {'source_sha256': sha(source), 'roles': summary,
                             'embedding_rule': 'Source-valid contiguous evidence_span only'})


if __name__ == '__main__':
    main()
