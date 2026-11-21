"""Build the original common sample for the similar-problem mechanism models."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.robustness.common import MEASUREMENT, input_path, output, panel, save, source


HISTORY_COLUMNS = ['strict10_overall_novelty_raw', 'window5_overall_novelty_raw']
SAMPLE_FILE = 'paired_text/input/conditional_manuscript_sample.csv'


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        result = pd.read_parquet(path)
    else:
        result = pd.read_csv(path, dtype={'patent_id': str})
    if 'patent_id' not in result or result.patent_id.isna().any():
        raise ValueError('Sample inputs require nonmissing patent_id')
    result['patent_id'] = result.patent_id.astype(str)
    if not result.patent_id.is_unique:
        raise ValueError('Sample input patent_id must be unique')
    return result


def build_sample(data, metrics, variants):
    """Select by availability, not by a stored target sample size or outcomes."""
    required = [(data, ['patent_id', 'overall_novelty_raw']),
                (metrics, ['patent_id', 'conditional_M_novelty']),
                (variants, ['patent_id', *HISTORY_COLUMNS])]
    for frame, columns in required:
        if set(columns) - set(frame):
            raise ValueError('Sample input is missing columns: ' + ', '.join(sorted(set(columns) - set(frame))))
        if frame.patent_id.isna().any() or not frame.patent_id.astype(str).is_unique:
            raise ValueError('Sample input patent_id must be nonmissing and unique')
    selected = data[['patent_id', 'overall_novelty_raw']].copy()
    selected['patent_id'] = selected.patent_id.astype(str)
    for frame, columns in [(metrics, ['conditional_M_novelty']), (variants, HISTORY_COLUMNS)]:
        values = frame[['patent_id', *columns]].copy()
        values['patent_id'] = values.patent_id.astype(str)
        if not set(selected.patent_id) <= set(values.patent_id):
            raise ValueError('Sample inputs must cover every focal patent; missing rows are not missing scores')
        selected = selected.merge(values, on='patent_id', how='left', validate='one_to_one')
    columns = ['overall_novelty_raw', 'conditional_M_novelty', *HISTORY_COLUMNS]
    numeric = selected[columns].apply(pd.to_numeric, errors='raise')
    if np.isinf(numeric.to_numpy(float)).any():
        raise ValueError('Novelty inputs cannot contain infinite values')
    available = numeric[['overall_novelty_raw', 'conditional_M_novelty']].notna().all(axis=1)
    complete = numeric.notna().all(axis=1)
    sample = selected.loc[complete, ['patent_id']].sort_values('patent_id').reset_index(drop=True)
    if sample.empty:
        raise ValueError('No patents satisfy the original conditional-model common-sample rule')
    return sample, {'focal_input_n': len(selected), 'available_observation_n': int(available.sum()),
        'manuscript_common_sample_n': len(sample),
        'excluded_from_available_n': int((available & ~complete).sum()),
        'required_nonmissing': columns,
        'rule': 'Nonmissing original, strict-ten-year and five-year overall novelty and conditional-M novelty; no outcome-based selection.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-variants', type=Path,
                        help='CSV/Parquet with patent_id and both historical overall-novelty variants')
    parser.add_argument('--metrics', type=Path, help='Existing paired_text_metrics.parquet or original novelty-wide table')
    args = parser.parse_args()
    variants_path = args.history_variants or input_path('history_novelty_variants', required=False)
    if variants_path is None:
        generated = output('data/history_novelty_variants.parquet')
        variants_path = generated if generated.is_file() else None
    if variants_path is None:
        parser.error('Run internal_exclusions history, or provide --history-variants / inputs.history_novelty_variants')
    metrics_path = args.metrics or output('data/paired_text_metrics.parquet')
    data = panel()
    sample, metadata = build_sample(data, read_table(metrics_path), read_table(variants_path))
    target = output(SAMPLE_FILE)
    sample.to_csv(target, index=False)
    metadata.update(sample_sha256=fingerprint(target), metrics_sha256=fingerprint(metrics_path),
                    history_variants_sha256=fingerprint(variants_path),
                    panel_sha256=fingerprint(source('panel', MEASUREMENT / 'data/panel_model_ready.parquet')),
                    focal_ids_sha256=hashlib.sha256('\n'.join(sorted(data.patent_id.astype(str))).encode()).hexdigest())
    save('paired_text/input/conditional_manuscript_sample.json', metadata)
    print(f'Conditional manuscript sample: {len(sample)} patents; excluded from available sample: {metadata["excluded_from_available_n"]}', flush=True)


if __name__ == '__main__':
    main()
