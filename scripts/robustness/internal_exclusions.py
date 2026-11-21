"""Historical and subsequent-text checks excluding known internal patent links.

The original main measures are unchanged. These separately named alternatives
use date-aware domestic families without CIP and any observed shared assignee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.robustness.common import MEASUREMENT, fit_variant, output, panel, save, setting
from scripts.robustness.conditional_sample import fingerprint, read_table
from scripts.robustness.internal_links import FamilyIndex, OwnerIndex, historical_family_mask, future_family_mask
from scripts.robustness.kelly import normalize


KINDS = {'overall_tac': 'overall_novelty_raw', 'P': 'problem_novelty_raw',
         'M': 'mechanism_novelty_raw', 'E': 'effect_novelty_raw',
         'PM_config': 'pm_config_novelty_raw', 'ME_config': 'me_config_novelty_raw'}
HISTORY_FILE = 'data/history_novelty_variants.parquet'
REUSE_FILE = 'data/internal_excluded_reuse.parquet'
COUNT_NAMES = ['reuse80_all', 'reuse80_no_samefamily', 'reuse80_no_sameassignee',
               'reuse80_no_either', 'reuse80_known_owner_pairs']


def cosine(q, h, device='cpu'):
    if device == 'cpu':
        return np.clip(q @ h.T, -1, 1)
    import torch
    return (torch.as_tensor(q, device=device) @ torch.as_tensor(h, device=device).T).clamp(-1, 1).cpu().numpy()


def history_topk(q, h, queries, histories, owners, k=10, query_rows=64, history_rows=2048, device='cpu'):
    """Exact strict ten-year and ordinary five-year alternatives in one subfield."""
    q, qvalid = normalize(q)
    h, hvalid = normalize(h)
    if not qvalid.all() or not hvalid.all():
        raise ValueError('History task vectors must be finite and nonzero')
    qdate = pd.to_datetime(queries.grant_date)
    qd = qdate.to_numpy('datetime64[D]').astype(np.int64)
    hd = pd.to_datetime(histories.grant_date).to_numpy('datetime64[D]').astype(np.int64)
    lower10 = (qdate - pd.DateOffset(years=10)).to_numpy('datetime64[D]').astype(np.int64)
    lower5 = (qdate - pd.DateOffset(years=5)).to_numpy('datetime64[D]').astype(np.int64)
    qr, hr = owners.rows(queries.patent_id), owners.rows(histories.patent_id)
    scores = {name: np.full((len(q), k), -np.inf, np.float32) for name in ['strict10', 'window5']}
    counts = {name: np.zeros(len(q), np.int64) for name in scores}
    for start in range(0, len(q), query_rows):
        end = min(start + query_rows, len(q))
        for begin in range(0, len(h), history_rows):
            stop = min(begin + history_rows, len(h))
            similarities = cosine(q[start:end], h[begin:stop], device)
            past = (hd[None, begin:stop] >= lower10[start:end, None]) & (hd[None, begin:stop] < qd[start:end, None])
            same_family = historical_family_mask(queries.iloc[start:end], histories.iloc[begin:stop])
            same_owner = owners.overlap(qr[start:end], hr[begin:stop])
            masks = {'strict10': past & ~same_family & ~same_owner,
                     'window5': past & (hd[None, begin:stop] >= lower5[start:end, None])}
            for name, mask in masks.items():
                counts[name][start:end] += mask.sum(axis=1)
                candidates = np.where(mask, similarities, -np.inf)
                order = np.argsort(-candidates, axis=1, kind='stable')[:, :k]
                candidates = np.take_along_axis(candidates, order, axis=1)
                joined = np.concatenate([scores[name][start:end], candidates], axis=1)
                best = np.argsort(-joined, axis=1, kind='stable')[:, :k]
                scores[name][start:end] = np.take_along_axis(joined, best, axis=1)
    return {name: {'score': np.where(np.isfinite(values).all(axis=1),
                    1 - values.mean(axis=1, dtype=np.float64), np.nan),
                  'history_n': counts[name]} for name, values in scores.items()}


def reuse_block(q, h, queries, histories, owners, device='cpu'):
    """Classify all qualifying future text matches, not citation edges."""
    qd = pd.to_datetime(queries.grant_date).to_numpy('datetime64[D]').astype(np.int64)
    end = pd.to_datetime(queries.label_window_end).to_numpy('datetime64[D]').astype(np.int64)
    hd = pd.to_datetime(histories.grant_date).to_numpy('datetime64[D]').astype(np.int64)
    hits = ((hd[None, :] > qd[:, None]) & (hd[None, :] <= end[:, None]) &
            (cosine(q, h, device) >= .8))
    hits &= queries.patent_id.astype(str).to_numpy()[:, None] != histories.patent_id.astype(str).to_numpy()[None, :]
    family = future_family_mask(queries, histories)
    qr, hr = owners.rows(queries.patent_id), owners.rows(histories.patent_id)
    same_owner = owners.overlap(qr, hr)
    known = (qr[:, None] >= 0) & (hr[None, :] >= 0)
    return dict(zip(COUNT_NAMES, [mask.sum(axis=1) for mask in [hits,
        hits & ~family, hits & ~same_owner, hits & ~(family | same_owner), hits & known]]))


def selected_focal(frame, ids):
    selected = frame.loc[frame.is_focal_candidate].copy() if 'is_focal_candidate' in frame else frame.copy()
    selected['patent_id'] = selected.patent_id.astype(str)
    if ids:
        if not set(ids) <= set(selected.patent_id):
            raise ValueError('Requested patent ID is not a focal query')
        selected = selected.loc[selected.patent_id.isin(ids)]
    if selected.empty or not selected.patent_id.is_unique:
        raise ValueError('Focal queries must be nonempty and unique')
    return selected.reset_index(drop=True)


def inputs(args):
    measurement = args.measurement_root or MEASUREMENT
    family_path = args.family_cache or measurement / 'cache/domestic_family_tree_no_cip.npz'
    owner_path = args.assignee_links or measurement / 'data/scope_assignee_portfolio.parquet'
    family = FamilyIndex.load(family_path)
    owners = OwnerIndex(pd.read_parquet(owner_path))
    return measurement, family, owners, {'family_cache_sha256': fingerprint(family_path),
        'assignee_links_sha256': fingerprint(owner_path),
        'family_rule': 'Date-aware domestic continuation/division component; CIP excluded.',
        'owner_rule': 'Any observed shared assignee; unknown ownership retained, not treated as a known match.'}


def history(args):
    measurement, family, owners, provenance = inputs(args)
    patents = panel(all_rows=True).sort_values('patent_id').reset_index(drop=True)
    patents = family.enrich(patents)
    tasks_path = measurement / 'encoding/full_frozen/task_metadata.parquet'
    vectors_path = measurement / 'encoding/full_frozen/vectors.npy'
    tasks = read_table_tasks(tasks_path)
    vectors = np.load(vectors_path, mmap_mode='r', allow_pickle=False)
    focal = selected_focal(patents, args.patent_id)
    wide = focal[['patent_id']].copy()
    kinds = args.kind or list(KINDS)
    identity = {**provenance, 'tasks_sha256': fingerprint(tasks_path),
                'vector_bytes': vectors_path.stat().st_size, 'vector_mtime_ns': vectors_path.stat().st_mtime_ns,
                'patents': fingerprint_panel(patents), 'query_ids': focal.patent_id.tolist(),
                'device': args.device, 'code_sha256': fingerprint(__file__),
                'links_code_sha256': fingerprint(Path(__file__).with_name('internal_links.py'))}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    for kind in kinds:
        lookup = tasks.loc[tasks.kind.eq(kind)].set_index('patent_id').embedding_row
        if not lookup.index.is_unique:
            raise ValueError('Duplicate patent/representation in encoding metadata')
        rows = patents.patent_id.map(lookup).fillna(-1).to_numpy(np.int64)
        qmask = patents.patent_id.isin(focal.patent_id).to_numpy() & (rows >= 0)
        parts = []
        for field in sorted(patents.loc[qmask].technical_subfield.astype(str).unique()):
            qix = np.flatnonzero(qmask & patents.technical_subfield.astype(str).eq(field).to_numpy())
            hix = np.flatnonzero((rows >= 0) & patents.technical_subfield.astype(str).eq(field).to_numpy())
            name = hashlib.sha256(field.encode()).hexdigest()[:12]
            cached = output(f'internal_links/history/{kind}/{name}.parquet')
            receipt = cached.with_suffix('.json')
            if cached.exists() and receipt.exists():
                if json.loads(receipt.read_text(encoding='utf-8'))['input_fingerprint'] != signature:
                    raise ValueError('History inputs changed; use a new work directory')
                part = pd.read_parquet(cached)
            else:
                result = history_topk(vectors[rows[qix]], vectors[rows[hix]], patents.iloc[qix], patents.iloc[hix],
                    owners, query_rows=args.query_rows, history_rows=args.candidate_rows, device=args.device)
                part = patents.iloc[qix][['patent_id']].copy()
                for branch, values in result.items():
                    part[branch + '_' + KINDS[kind]] = values['score']
                    part[branch + '_history_n'] = values['history_n']
                part.to_parquet(cached, index=False)
                receipt.write_text(json.dumps({'input_fingerprint': signature, 'field': field, 'kind': kind}), encoding='utf-8')
            parts.append(part)
            print(f'Internal-link history: {kind} {field} queries={len(qix)}', flush=True)
        if not parts:
            for branch in ['strict10', 'window5']:
                wide[branch + '_' + KINDS[kind]] = np.nan
        else:
            merged = pd.concat(parts).set_index('patent_id')
            for branch in ['strict10', 'window5']:
                column = branch + '_' + KINDS[kind]
                wide[column] = wide.patent_id.map(merged[column])
    wide.to_parquet(output(HISTORY_FILE), index=False)
    save('results/internal_history_definition.json', {'inputs': identity, 'rows': len(wide), 'kinds': kinds,
        'strict10': 'Same subfield, preceding ten calendar years, excluding known same-family and any-shared-owner histories at focal grant.',
        'window5': 'Same subfield, preceding five calendar years; no internal-link exclusion.',
        'primary_k': 10, 'insufficient_history': 'missing, never zero-filled'})


def read_table_tasks(path):
    tasks = pd.read_parquet(path, columns=['patent_id', 'kind', 'embedding_row'])
    tasks['patent_id'] = tasks.patent_id.astype(str)
    if tasks.duplicated(['patent_id', 'kind']).any():
        raise ValueError('Encoding metadata must have unique patent/representation pairs')
    return tasks


def fingerprint_panel(frame):
    columns = ['patent_id', 'grant_date', 'technical_subfield', 'is_focal_candidate']
    return hashlib.sha256(pd.util.hash_pandas_object(frame[columns], index=False).to_numpy().tobytes()).hexdigest()


def reuse(args):
    measurement, family, owners, provenance = inputs(args)
    job = args.source_job or measurement / 'semantic_reuse/all_future_text'
    qsource, hsource = job / 'focal_metadata.parquet', job / 'candidate_metadata.parquet'
    qmeta, hmeta = pd.read_parquet(qsource), pd.read_parquet(hsource)
    qmeta['patent_id'], hmeta['patent_id'] = qmeta.patent_id.astype(str), hmeta.patent_id.astype(str)
    selected = selected_focal(qmeta, args.patent_id)
    qrows = pd.Series(np.arange(len(qmeta)), index=qmeta.patent_id).loc[selected.patent_id].to_numpy()
    queries, histories = family.enrich(selected), family.enrich(hmeta)
    qpath, hpath = job / 'focal_vectors.npy', job / 'candidate_vectors.npy'
    qvectors = np.load(qpath, mmap_mode='r', allow_pickle=False)
    hvectors = np.load(hpath, mmap_mode='r', allow_pickle=False)
    if len(qvectors) != len(qmeta) or len(hvectors) != len(hmeta) or qvectors.shape[1] != hvectors.shape[1]:
        raise ValueError('Reuse vector arrays do not align with their metadata')
    q, qvalid = normalize(qvectors[qrows])
    if 'emb_valid' in queries:
        qvalid &= queries.emb_valid.fillna(False).to_numpy(bool)
    hd = pd.to_datetime(histories.grant_date)
    if hd.isna().any() or not hd.is_monotonic_increasing or not histories.patent_id.is_unique:
        raise ValueError('Future candidate IDs/dates must be unique, nonmissing and date-sorted')
    lower = pd.to_datetime(queries.grant_date)
    upper = pd.to_datetime(queries.label_window_end)
    if not upper.equals(lower + pd.DateOffset(years=5)):
        raise ValueError('Reuse windows must end at the exact fifth anniversary')
    identity = {**provenance, 'focal_metadata_sha256': fingerprint(qsource),
        'candidate_metadata_sha256': fingerprint(hsource), 'query_ids': queries.patent_id.tolist(),
        'vectors': {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in [qpath, hpath]},
        'device': args.device, 'query_rows': args.query_rows, 'candidate_rows': args.candidate_rows,
        'code_sha256': fingerprint(__file__), 'links_code_sha256': fingerprint(Path(__file__).with_name('internal_links.py'))}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    checkpoint = output('internal_links/reuse_checkpoint.npz')
    totals = {name: np.zeros(len(queries), np.int64) for name in COUNT_NAMES}
    begin = invalid_n = 0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as cached:
            if str(cached['fingerprint']) != signature:
                raise ValueError('Reuse inputs/settings changed; use a new work directory')
            begin, invalid_n = int(cached['candidate_end']), int(cached['invalid_candidate_n'])
            totals = {name: cached[name].copy() for name in COUNT_NAMES}
    for start in range(begin, len(histories), args.candidate_rows):
        end = min(start + args.candidate_rows, len(histories))
        h, valid = normalize(hvectors[start:end])
        if 'emb_valid' in histories:
            valid &= histories.iloc[start:end].emb_valid.fillna(False).to_numpy(bool)
        invalid_n += int((~valid).sum())
        active = np.flatnonzero(qvalid & lower.lt(hd.iloc[end-1]).to_numpy() & upper.ge(hd.iloc[start]).to_numpy())
        candidates = histories.iloc[start:end].iloc[np.flatnonzero(valid)]
        for offset in range(0, len(active), args.query_rows):
            ix = active[offset:offset + args.query_rows]
            values = reuse_block(q[ix], h[valid], queries.iloc[ix], candidates, owners, args.device)
            for name in COUNT_NAMES:
                totals[name][ix] += values[name]
        if end == len(histories) or (start // args.candidate_rows) % 8 == 7:
            temporary = checkpoint.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **totals, fingerprint=signature, candidate_end=end, invalid_candidate_n=invalid_n)
            temporary.replace(checkpoint)
            print(f'Internal-link future texts: {end}/{len(histories)}', flush=True)
    result = queries[['patent_id']].copy()
    usable = qvalid & upper.le(pd.Timestamp(setting('study', 'observation_end', '2025-12-30'))).to_numpy()
    for name in COUNT_NAMES:
        result[name] = np.where(usable, totals[name], np.nan)
    for branch in ['no_samefamily', 'no_sameassignee', 'no_either']:
        result['semantic_' + branch] = np.log1p(result['reuse80_' + branch])
    result.to_parquet(output(REUSE_FILE), index=False)
    save('results/internal_reuse_definition.json', {'inputs': identity, 'focal_n': len(result),
        'usable_n': int(usable.sum()), 'invalid_candidate_n': invalid_n,
        'threshold': .8, 'transformation': 'log1p qualifying text count',
        'family_date': 'Family relation evaluated at the subsequent candidate grant, without CIP.',
        'known_owner_pair_matches': int(totals['reuse80_known_owner_pairs'][usable].sum()),
        'unknown_owners': 'Retained; no presumed internal link.', 'citation_restriction': False})


def fit(args):
    data = panel()
    records = []
    if args.branch in ['all', 'history']:
        variants = read_table(args.history_metrics or output(HISTORY_FILE))
        columns = [prefix + '_' + name for prefix in ['strict10', 'window5'] for name in KINDS.values()]
        if set(columns) - set(variants):
            raise ValueError('All six historical representations are required for fitting the paper comparison')
        merged = data.merge(variants[['patent_id', *columns]], on='patent_id', validate='one_to_one')
        if len(merged) != len(data):
            raise ValueError('Historical variants must include a row for every focal patent')
        common = merged.loc[merged[['overall_novelty_raw', 'strict10_overall_novelty_raw',
                                   'window5_overall_novelty_raw']].notna().all(axis=1)].copy()
        for branch in ['original_common', 'strict10', 'window5']:
            selected = common.copy()
            if branch != 'original_common':
                for column in KINDS.values():
                    selected[column] = selected[branch + '_' + column]
            records += fit_variant(selected, 'internal_history_' + branch)
    if args.branch in ['all', 'reuse']:
        variants = read_table(args.reuse_results or output(REUSE_FILE))
        merged = data.merge(variants, on='patent_id', validate='one_to_one')
        if len(merged) != len(data):
            raise ValueError('Reuse alternatives must include a row for every focal patent')
        for branch in ['no_samefamily', 'no_sameassignee', 'no_either']:
            records += fit_variant(merged, 'internal_future_' + branch,
                                   targets={'semantic': ('semantic_' + branch, 'continuous')})
    save('results/internal_exclusion_models.json', {'models': records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['history', 'reuse', 'fit'])
    parser.add_argument('--measurement-root', type=Path)
    parser.add_argument('--family-cache', type=Path)
    parser.add_argument('--assignee-links', type=Path)
    parser.add_argument('--source-job', type=Path)
    parser.add_argument('--kind', action='append', choices=list(KINDS))
    parser.add_argument('--patent-id', action='append', help='Optional focal IDs for a separate verification run')
    parser.add_argument('--history-metrics', type=Path)
    parser.add_argument('--reuse-results', type=Path)
    parser.add_argument('--branch', choices=['all', 'history', 'reuse'], default='all')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--query-rows', type=int, default=64)
    parser.add_argument('--candidate-rows', type=int, default=2048)
    args = parser.parse_args()
    if min(args.query_rows, args.candidate_rows) <= 0:
        parser.error('Block sizes must be positive')
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA requested but unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
    {'history': history, 'reuse': reuse, 'fit': fit}[args.action](args)


if __name__ == '__main__':
    main()
