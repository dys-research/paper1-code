"""Five-year forward/backward positive-cosine diagnostic inspired by Kelly.

This is a retrospective outcome diagnostic, not a grant-time forecast feature
or a replication of Kelly et al.'s TF-IDF measure. Existing aggregate sums can
be summarized without rerunning the full vector search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.robustness.common import MEASUREMENT, output, save, setting


TOTALS = ['fsum', 'bsum', 'fn', 'bn']
DEFINITION = ('Positive cosine sums divided by all valid comparison-text counts in '
              '(grant, fifth anniversary] and [grant minus five years, grant); '
              'forward mean divided by backward mean; log1p; complete windows only.')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(norms) & (norms > 0)
    result = np.zeros_like(values)
    result[valid] = values[valid] / norms[valid, None]
    return result, valid


def block_totals(q, dates, lower, upper, h, history_dates, device='cpu'):
    """Use normalized valid vectors; keep negative cosines at zero in the sums.

    Counts include zero/negative similarities, as in the source computation.
    Same-day grants belong to neither window; both anniversary bounds are inclusive.
    """
    if device == 'cpu':
        similarity = np.maximum(np.clip(q @ h.T, -1, 1), 0).astype(np.float64)
        forward = (history_dates[None, :] > dates[:, None]) & (history_dates[None, :] <= upper[:, None])
        backward = (history_dates[None, :] < dates[:, None]) & (history_dates[None, :] >= lower[:, None])
        return {'fsum': (similarity * forward).sum(axis=1),
                'bsum': (similarity * backward).sum(axis=1),
                'fn': forward.sum(axis=1), 'bn': backward.sum(axis=1)}
    import torch
    def tensor(values):
        return torch.as_tensor(values, device=device)
    similarity = (tensor(q) @ tensor(h).T).clamp(0, 1).double()
    forward = (tensor(history_dates)[None, :] > tensor(dates)[:, None]) & (
        tensor(history_dates)[None, :] <= tensor(upper)[:, None])
    backward = (tensor(history_dates)[None, :] < tensor(dates)[:, None]) & (
        tensor(history_dates)[None, :] >= tensor(lower)[:, None])
    return {name: value.cpu().numpy() for name, value in {
        'fsum': (similarity * forward).sum(dim=1), 'bsum': (similarity * backward).sum(dim=1),
        'fn': forward.sum(dim=1), 'bn': backward.sum(dim=1)}.items()}


def score_aggregates(frame, observation_end):
    required = {'patent_id', 'grant_date', 'label_window_end', 'complete_backward5y', *TOTALS}
    if required - set(frame):
        raise ValueError('Kelly aggregates are missing columns: ' + ', '.join(sorted(required - set(frame))))
    result = frame.copy()
    if result.patent_id.isna().any() or not result.patent_id.astype(str).is_unique:
        raise ValueError('Kelly patent IDs must be nonmissing and unique')
    result['patent_id'] = result.patent_id.astype(str)
    for column in ['grant_date', 'label_window_end']:
        result[column] = pd.to_datetime(result[column], errors='raise')
        if result[column].isna().any():
            raise ValueError('Kelly observation dates must be nonmissing')
    if not result.label_window_end.equals(result.grant_date + pd.DateOffset(years=5)):
        raise ValueError('label_window_end must equal the exact fifth anniversary')
    values = result[TOTALS].apply(pd.to_numeric, errors='raise').to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Kelly sums/counts must be finite and nonnegative')
    for column in ['fn', 'bn']:
        if (result[column] != np.floor(result[column])).any():
            raise ValueError('Kelly comparison counts must be integers')
    if (result.fsum > result.fn + 1e-6).any() or (result.bsum > result.bn + 1e-6).any():
        raise ValueError('Positive cosine sums cannot exceed comparison-text counts')
    for flag in ['complete_backward5y', 'query_valid']:
        if flag in result:
            if result[flag].isna().any() or not result[flag].isin([True, False, 0, 1]).all():
                raise ValueError(flag + ' must be a nonmissing Boolean flag')
            result[flag] = result[flag].astype(bool)
    result['complete_forward5y'] = result.label_window_end.le(pd.Timestamp(observation_end))
    result['query_valid'] = result.get('query_valid', True)
    valid = (result.complete_backward5y & result.complete_forward5y & result.query_valid &
             result.bn.gt(0) & result.bsum.gt(1e-12))
    forward = result.fsum / np.maximum(result.fn, 1)
    backward = result.bsum / np.maximum(result.bn, 1)
    result['positive_embedding_mean_ratio'] = (forward / backward.where(backward > 0)).where(valid)
    result['positive_embedding_aggregate_ratio'] = (result.fsum / result.bsum.where(result.bsum > 0)).where(valid)
    for kind in ['mean', 'aggregate']:
        column = 'positive_embedding_' + kind + '_ratio'
        result['log_' + column] = np.log1p(result[column])
    return result


def summarize(scores):
    column = scores.log_positive_embedding_mean_ratio.dropna()
    return {'focal_n': len(scores), 'complete_backward5y_n': int(scores.complete_backward5y.sum()),
        'complete_both_windows_n': int((scores.complete_backward5y & scores.complete_forward5y).sum()),
        'valid_mean_ratio_n': len(column),
        'median_log_one_plus_mean_ratio': float(column.median()) if len(column) else None,
        'mean_log_one_plus_mean_ratio': float(column.mean()) if len(column) else None,
        'definition': DEFINITION,
        'interpretation': 'Retrospective embedding-based uptake diagnostic; future information; not a forecast input or original TF-IDF replication.'}


def compute(job, query_rows=256, candidate_rows=8192, device='cpu'):
    """Read the existing semantic-reuse arrays without changing their files."""
    job = Path(job)
    qmeta = pd.read_parquet(job / 'focal_metadata.parquet')
    hmeta = pd.read_parquet(job / 'candidate_metadata.parquet')
    qvectors = np.load(job / 'focal_vectors.npy', mmap_mode='r', allow_pickle=False)
    hvectors = np.load(job / 'candidate_vectors.npy', mmap_mode='r', allow_pickle=False)
    if (qvectors.ndim != 2 or hvectors.ndim != 2 or qvectors.shape[1] != hvectors.shape[1] or
            len(qmeta) != len(qvectors) or len(hmeta) != len(hvectors) or hmeta.empty):
        raise ValueError('Kelly metadata and vector shapes do not align')
    for frame in [qmeta, hmeta]:
        if frame.patent_id.isna().any() or not frame.patent_id.is_unique:
            raise ValueError('Semantic metadata IDs must be nonmissing and unique')
    dates = pd.to_datetime(qmeta.grant_date, errors='raise')
    history = pd.to_datetime(hmeta.grant_date, errors='raise')
    if dates.isna().any() or history.isna().any() or not history.is_monotonic_increasing:
        raise ValueError('Candidate dates must be nonmissing and chronologically sorted')
    days = dates.to_numpy('datetime64[D]').astype(np.int64)
    lower = (dates - pd.DateOffset(years=5)).to_numpy('datetime64[D]').astype(np.int64)
    upper = pd.to_datetime(qmeta.label_window_end).to_numpy('datetime64[D]').astype(np.int64)
    hd = history.to_numpy('datetime64[D]').astype(np.int64)
    q, valid = normalize(qvectors)
    if 'emb_valid' in qmeta:
        valid &= qmeta.emb_valid.fillna(False).to_numpy(bool)
    declared = hmeta.emb_valid.fillna(False).to_numpy(bool) if 'emb_valid' in hmeta else np.ones(len(hmeta), bool)
    identity = {'definition': DEFINITION, 'device': device, 'query_rows': query_rows, 'candidate_rows': candidate_rows,
                'source_job': str(job.resolve()), 'code_sha256': sha256(__file__)}
    for name in ['focal_metadata.parquet', 'candidate_metadata.parquet']:
        identity[name] = sha256(job / name)
    for name in ['focal_vectors.npy', 'candidate_vectors.npy']:
        stat = (job / name).stat()
        identity[name] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    checkpoint = output('kelly/checkpoint.npz')
    totals = {name: np.zeros(len(qmeta), np.float64 if name.endswith('sum') else np.int64) for name in TOTALS}
    begin, invalid_n = 0, 0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as previous:
            if str(previous['fingerprint']) != signature:
                raise ValueError('Kelly inputs/settings changed; use a new work directory for a new run')
            begin, invalid_n = int(previous['candidate_end']), int(previous['invalid_candidate_n'])
            totals = {name: previous[name].copy() for name in TOTALS}
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable; choose --device cpu explicitly')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
    for start in range(begin, len(hmeta), candidate_rows):
        end = min(start + candidate_rows, len(hmeta))
        h, hvalid = normalize(hvectors[start:end])
        hvalid &= declared[start:end]
        invalid_n += int((~hvalid).sum())
        h, hdays = h[hvalid], hd[start:end][hvalid]
        active = np.flatnonzero(valid & (lower <= hd[end-1]) & (upper >= hd[start]))
        for offset in range(0, len(active), query_rows):
            indices = active[offset:offset+query_rows]
            values = block_totals(q[indices], days[indices], lower[indices], upper[indices], h, hdays, device)
            for name in TOTALS:
                totals[name][indices] += values[name]
        if end == len(hmeta) or (start // candidate_rows) % 8 == 7:
            temporary = checkpoint.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **totals, fingerprint=signature, candidate_end=end,
                                    invalid_candidate_n=invalid_n)
            temporary.replace(checkpoint)
            print(f'Kelly candidates: {end}/{len(hmeta)}', flush=True)
    result = qmeta[['patent_id', 'grant_date', 'label_window_end']].copy()
    for name in TOTALS:
        result[name] = totals[name]
    result['complete_backward5y'] = lower >= hd.min()
    result['query_valid'] = valid
    return result, {'input_fingerprint': signature, 'inputs': identity,
        'candidate_n': len(hmeta), 'invalid_candidate_n': invalid_n,
        'invalid_query_n': int((~valid).sum()),
        'large_vector_full_hash_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregates', type=Path, help='Summarize existing fsum/bsum/fn/bn Parquet or CSV')
    parser.add_argument('--source-job', type=Path, default=MEASUREMENT / 'semantic_reuse/all_future_text')
    parser.add_argument('--observation-end', default=setting('study', 'observation_end', '2025-12-30'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--query-rows', type=int, default=256)
    parser.add_argument('--candidate-rows', type=int, default=8192)
    args = parser.parse_args()
    if min(args.query_rows, args.candidate_rows) < 1:
        parser.error('Block sizes must be positive')
    if args.aggregates:
        frame = (pd.read_parquet(args.aggregates) if args.aggregates.suffix.lower() == '.parquet'
                 else pd.read_csv(args.aggregates, dtype={'patent_id': str}))
        provenance = {'mode': 'existing_aggregates', 'input_sha256': sha256(args.aggregates)}
    else:
        frame, provenance = compute(args.source_job, args.query_rows, args.candidate_rows, args.device)
        provenance['mode'] = 'vector_computation'
    scores = score_aggregates(frame, args.observation_end)
    scores.to_parquet(output('kelly/kelly_patent_scores.parquet'), index=False)
    metadata = summarize(scores)
    metadata.update(provenance=provenance, observation_end=str(args.observation_end))
    save('kelly/kelly_summary.json', metadata)
    print(f'Kelly valid patents: {metadata["valid_mean_ratio_n"]}; median log1p ratio: {metadata["median_log_one_plus_mean_ratio"]}', flush=True)


if __name__ == '__main__':
    main()
