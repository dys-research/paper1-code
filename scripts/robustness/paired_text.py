"""Retrospective P–M pairing and similar-problem mechanism diagnostics.

The shuffled pairing is one fixed derangement, not a permutation test and not a
forecast-time predictor. Donors can have later grant dates within the same year.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.robustness.common import (
    MEASUREMENT, ROOT, analysis_outputs, ex, fit_variant, input_path, log,
    output, panel, save, source,
)
from scripts.robustness.conditional_sample import SAMPLE_FILE, read_table


LIMITATION = ('One fixed within-year/subfield derangement; some donor patents '
              'grant later in the same year. Retrospective diagnostic only, '
              'not a permutation p-value or an operational forecast feature.')


def normalized(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or (norms <= 0).any():
        raise ValueError('Vectors must be finite and nonzero')
    return values / norms[:, None]


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def derangement(patents, texts, seed=20261103):
    """Preserve the source script's sorted-group permutation then cyclic shift."""
    wide = texts.pivot(index='patent_id', columns='role', values='embedding_text')
    eligible = wide.dropna(subset=['P', 'M']).index
    pairs = patents.loc[patents.patent_id.isin(eligible),
                        ['patent_id', 'grant_year', 'grant_date', 'technical_subfield']]
    rng = np.random.default_rng(seed)
    records, singletons = [], 0
    for _, group in pairs.groupby(['grant_year', 'technical_subfield'], sort=True):
        if len(group) < 2:
            singletons += len(group)
            continue
        order = rng.permutation(group.patent_id.to_numpy())
        donors = np.roll(order, 1)
        for patent, donor in zip(order, donors):
            if patent == donor:
                raise AssertionError('Derangement contains a fixed point')
            records.append({'patent_id': str(patent), 'partner_patent_id': str(donor),
                'text': 'Problem: ' + wide.at[patent, 'P'] + '\nMechanism: ' + wide.at[donor, 'M'],
                'same_M_text': bool(wide.at[patent, 'M'] == wide.at[donor, 'M'])})
    return pd.DataFrame(records, columns=['patent_id', 'partner_patent_id', 'text', 'same_M_text']), singletons


def prepare(args):
    patents = panel(all_rows=True).sort_values('patent_id').reset_index(drop=True)
    role_path = source('role_inputs', MEASUREMENT / 'data/role_inputs_primary.parquet')
    roles = pd.read_parquet(role_path)
    roles['patent_id'] = roles.patent_id.astype(str)
    required = {'role', 'embedding_text', 'usable_for_primary_embedding'}
    if required - set(roles):
        raise ValueError('Role inputs require ' + ', '.join(sorted(required)))
    roles = roles.loc[roles.usable_for_primary_embedding & roles.role.isin(['P', 'M'])]
    if roles.duplicated(['patent_id', 'role']).any():
        raise ValueError('Duplicate patent-role records')
    pairs, singletons = derangement(patents, roles, seed=args.seed)
    if pairs.empty:
        raise ValueError('No group contains at least two usable P/M patents')
    grant_dates = patents.set_index('patent_id').grant_date
    mechanism = patents.set_index('patent_id').mechanism_novelty_raw
    pairs['donor_M_novelty_raw'] = pairs.partner_patent_id.map(mechanism)
    pairs['donor_after_recipient'] = pairs.partner_patent_id.map(grant_dates).gt(
        pairs.patent_id.map(grant_dates))
    path = output('paired_text/input/shuffled_PM_texts.jsonl')
    with path.open('w', encoding='utf-8') as stream:
        for row in pairs[['patent_id', 'partner_patent_id', 'text']].to_dict(orient='records'):
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    pairs.drop(columns='text').to_parquet(output('paired_text/input/shuffled_pair_mapping.parquet'), index=False)
    save('paired_text/input/preparation.json', {'seed': args.seed, 'pairs': len(pairs),
        'unshufflable_singletons': singletons, 'same_M_text': int(pairs.same_M_text.sum()),
        'donor_after_recipient_n': int(pairs.donor_after_recipient.sum()),
        'input_sha256': fingerprint(path), 'limitation': LIMITATION})
    log(f'Prepared {len(pairs)} shuffled texts; no model has been called')


def encode(args):
    """Encode prepared text with an explicit local copy of the main encoder."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    model_dir = Path(args.model_dir) if args.model_dir else input_path('encoder_model')
    if not model_dir.is_dir():
        raise ValueError('Use --model-dir or inputs.encoder_model for a local checkpoint')
    text_path = output('paired_text/input/shuffled_PM_texts.jsonl')
    records = [json.loads(line) for line in text_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError('Run paired_text prepare first')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(args.device).eval()
    path = output('paired_text/encoding/shuffled_vectors.npy')
    vectors = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(records), 768))
    for start in range(0, len(records), args.batch_size):
        batch = tokenizer([row['text'] for row in records[start:start+args.batch_size]],
            padding=True, truncation=True, max_length=512, return_tensors='pt').to(args.device)
        with torch.no_grad():
            values = torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0, :], dim=1).cpu().numpy()
        if values.shape[1] != 768 or not np.isfinite(values).all():
            raise ValueError('Encoder must produce finite 768-dimensional CLS vectors')
        vectors[start:start+len(values)] = values
    vectors.flush()
    pd.DataFrame({'patent_id': [row['patent_id'] for row in records]}).to_csv(
        output('paired_text/encoding/shuffled_vector_ids.csv'), index=False)
    save('paired_text/encoding/encoding.json', {'input_sha256': fingerprint(text_path),
        'vector_sha256': fingerprint(path), 'rows': len(records), 'dimensions': 768,
        'pooling': 'CLS', 'max_tokens': 512, 'normalization': 'L2',
        'model_config': model.config.to_dict()})


def indexed_patents():
    patents = panel(all_rows=True).sort_values('patent_id').reset_index(drop=True)
    task_path = MEASUREMENT / 'encoding/full_frozen/task_metadata.parquet'
    tasks = pd.read_parquet(task_path, columns=['patent_id', 'kind', 'embedding_row'])
    tasks['patent_id'] = tasks.patent_id.astype(str)
    for kind in ['P', 'M']:
        subset = tasks.loc[tasks.kind.eq(kind)].set_index('patent_id').embedding_row
        if not subset.index.is_unique:
            raise ValueError('Repeated task metadata for ' + kind)
        patents['row_' + kind] = patents.patent_id.map(subset).fillna(-1).astype(np.int64)
    return patents


def history_topk(vectors, embedding_rows, patents, query_mask, history_mask,
                 k=10, qblock=64, hblock=2048):
    """Exact stable top-k within subfield and [grant-10years, grant)."""
    days = patents.grant_date.to_numpy('datetime64[D]')
    lower = (patents.grant_date - pd.DateOffset(years=10)).to_numpy('datetime64[D]')
    fields = patents.technical_subfield.astype(str).to_numpy()
    query_ids = np.flatnonzero(query_mask)
    scores = np.full((len(patents), k), -np.inf, np.float32)
    indices = np.full((len(patents), k), -1, np.int64)
    counts = np.zeros(len(patents), np.int64)
    for field in sorted(set(fields[query_ids])):
        queries = query_ids[fields[query_ids] == field]
        histories = np.flatnonzero(history_mask & (fields == field))
        for start in range(0, len(queries), qblock):
            qids = queries[start:start+qblock]
            qvectors = normalized(vectors[embedding_rows[qids]])
            for begin in range(0, len(histories), hblock):
                hids = histories[begin:begin+hblock]
                hvectors = normalized(vectors[embedding_rows[hids]])
                similarity = np.clip(qvectors @ hvectors.T, -1, 1)
                allowed = (days[hids][None, :] < days[qids][:, None]) & (
                    days[hids][None, :] >= lower[qids][:, None])
                counts[qids] += allowed.sum(axis=1)
                similarity[~allowed] = -np.inf
                order = np.argsort(-similarity, axis=1, kind='stable')[:, :k]
                new_scores = np.take_along_axis(similarity, order, axis=1)
                new_indices = hids[order]
                joined_scores = np.concatenate([scores[qids], new_scores], axis=1)
                joined_indices = np.concatenate([indices[qids], new_indices], axis=1)
                best = np.argsort(-joined_scores, axis=1, kind='stable')[:, :k]
                scores[qids] = np.take_along_axis(joined_scores, best, axis=1)
                indices[qids] = np.take_along_axis(joined_indices, best, axis=1)
    return scores, indices, counts


def update_metrics(patents, values):
    path = output('data/paired_text_metrics.parquet')
    if path.exists():
        result = pd.read_parquet(path).set_index('patent_id').reindex(patents.patent_id)
        result.index.name = 'patent_id'
    else:
        result = pd.DataFrame(index=pd.Index(patents.patent_id, name='patent_id'))
    for name, value in values.items():
        result[name] = value
    result.reset_index().to_parquet(path, index=False)


def conditional(args):
    patents = indexed_patents()
    vectors = np.load(MEASUREMENT / 'encoding/full_frozen/vectors.npy', mmap_mode='r', allow_pickle=False)
    valid = patents.row_P.ge(0).to_numpy() & patents.row_M.ge(0).to_numpy()
    query = valid & patents.is_focal_candidate.to_numpy(bool)
    scores, neighbors, counts = history_topk(vectors, patents.row_P.to_numpy(), patents,
        query, valid, qblock=args.query_block, hblock=args.history_block)
    result = np.full(len(patents), np.nan)
    mechanism_rows = patents.row_M.to_numpy(np.int64)
    for index in np.flatnonzero(query & np.isfinite(scores).all(axis=1)):
        history = normalized(vectors[mechanism_rows[neighbors[index]]])
        focal = normalized(vectors[[mechanism_rows[index]]])[0]
        result[index] = 1 - (history @ focal).mean(dtype=np.float64)
    update_metrics(patents, {'conditional_M_novelty': result, 'conditional_P_history_n': counts})
    np.savez_compressed(output('paired_text/conditional_neighbors.npz'), scores=scores,
        history_patent_rows=neighbors, patent_ids=patents.patent_id.to_numpy(str))
    save('paired_text/conditional_definition.json', {'observed_n': int(np.isfinite(result).sum()),
        'definition': 'One minus mean focal-M cosine to M of ten most similar historical P passages; history requires valid P and M.',
        'history': 'Same subfield; inclusive grant minus ten calendar years, exclusive grant date.',
        'main_vectors_sha256': fingerprint(MEASUREMENT / 'encoding/full_frozen/vectors.npy')})


def shuffled_novelty(args):
    patents = indexed_patents()
    configured = input_path('shuffled_vectors', required=False)
    if configured is None:
        vector_path = output('paired_text/encoding/shuffled_vectors.npy')
        id_path = output('paired_text/encoding/shuffled_vector_ids.csv')
        receipt = json.loads(output('paired_text/encoding/encoding.json').read_text(encoding='utf-8'))
        if receipt['input_sha256'] != fingerprint(output('paired_text/input/shuffled_PM_texts.jsonl')):
            raise ValueError('Prepared shuffled text changed after encoding')
        if receipt['vector_sha256'] != fingerprint(vector_path):
            raise ValueError('Shuffled vector array changed after encoding')
    else:
        vector_path = configured
        id_path = input_path('shuffled_vector_ids')
    ids = pd.read_parquet(id_path).patent_id if id_path.suffix == '.parquet' else pd.read_csv(
        id_path, dtype={'patent_id': str}).patent_id
    ids = ids.astype(str)
    if not ids.is_unique:
        raise ValueError('Shuffled vector identifiers must be unique')
    vectors = np.load(vector_path, mmap_mode='r', allow_pickle=False)
    if vectors.shape != (len(ids), 768):
        raise ValueError('Shuffled vectors must have one 768-dimensional row per identifier')
    rows = patents.patent_id.map(pd.Series(np.arange(len(ids)), index=ids)).fillna(-1).to_numpy(np.int64)
    valid = rows >= 0
    query = valid & patents.is_focal_candidate.to_numpy(bool)
    scores, neighbors, counts = history_topk(vectors, rows, patents, query, valid,
        qblock=args.query_block, hblock=args.history_block)
    result = np.where(np.isfinite(scores).all(axis=1), 1 - scores.mean(axis=1, dtype=np.float64), np.nan)
    update_metrics(patents, {'shuffled_PM_novelty': result, 'shuffled_history_n': counts})
    save('paired_text/shuffled_novelty_definition.json', {'observed_n': int(np.isfinite(result).sum()),
        'vectors_sha256': fingerprint(vector_path), 'limitation': LIMITATION})


def fit_shuffled(data, metrics, pairs):
    selected = data.merge(metrics[['patent_id', 'shuffled_PM_novelty']], on='patent_id', validate='one_to_one')
    selected = selected.merge(pairs[['patent_id', 'donor_M_novelty_raw', 'donor_after_recipient']],
        on='patent_id', how='inner', validate='one_to_one')
    columns = ['problem_novelty_raw', 'mechanism_novelty_raw', 'pm_config_novelty_raw',
        'shuffled_PM_novelty', 'donor_M_novelty_raw']
    selected = selected.loc[selected[columns].notna().all(axis=1)].reset_index(drop=True)
    if len(selected) < 10:
        raise ValueError('Insufficient complete observed/shuffled paired sample')
    x, specs, fe, _ = ex.prepare(selected, 'paired_observed')
    changed = selected.copy()
    changed['mechanism_novelty_raw'] = changed.donor_M_novelty_raw
    changed['pm_config_novelty_raw'] = changed.shuffled_PM_novelty
    null, _, _, _ = ex.prepare(changed, 'paired_null')
    x['donor_M_z'] = null.M
    xnull = x.copy()
    xnull['PMres'], xnull['PM_missing'] = null.PMres, null.PM_missing
    xnull['PMxA'] = xnull.PMres * xnull.anchor
    terms = specs['M5'] + ['donor_M_z']
    records = []
    for target, (column, family) in {
        'citations': ('forward_cites_5y', 'count'),
        'semantic': ('semantic_reuse_5y', 'continuous'),
        'newfield': ('new_field_cites_5y', 'count')}.items():
        keep = selected[column].notna().to_numpy().copy()
        if family == 'count':
            while True:
                previous = keep.copy()
                for group in ['grant_year', 'technical_subfield']:
                    totals = selected.loc[keep, column].groupby(selected.loc[keep, group]).sum()
                    keep &= ~selected[group].isin(totals[totals == 0].index).to_numpy()
                if np.array_equal(previous, keep):
                    break
        for label, matrix in [('observed', x), ('one_fixed_derangement', xnull)]:
            records.append(ex.fitmodel(selected.loc[keep], matrix.loc[keep], fe, terms,
                target, family, 'paired_fixed_sample', label))
    return records, {'fixed_sample_n': len(selected),
        'donor_after_recipient_n': int(selected.donor_after_recipient.sum()), 'limitation': LIMITATION}


def select_conditional_sample(data, metrics, sample_path=None, rule='available'):
    """Keep the existing default; make the manuscript's fixed sample explicit."""
    selected = data.merge(metrics[['patent_id', 'conditional_M_novelty']],
                          on='patent_id', validate='one_to_one')
    selected = selected.loc[selected.conditional_M_novelty.notna() & selected.overall_novelty_raw.notna()]
    available_n = len(selected)
    if rule not in {'available', 'manuscript'}:
        raise ValueError('Unknown conditional sample rule')
    if rule == 'manuscript' and sample_path is None:
        sample_path = input_path('conditional_sample', required=False) or output(SAMPLE_FILE)
    description = 'Available conditional-M and original overall novelty'
    if sample_path is not None:
        sample_path = Path(sample_path)
        sample_ids = read_table(sample_path).patent_id
        if sample_ids.empty or not set(sample_ids) <= set(selected.patent_id):
            raise ValueError('Fixed sample must be nonempty and every ID must have available conditional-M and overall novelty')
        selected = selected.loc[selected.patent_id.isin(sample_ids)]
        description = 'Configured fixed sample: ' + fingerprint(sample_path)
    return selected, {'conditional_sample_n': len(selected), 'conditional_available_n': available_n,
        'conditional_sample_rule': description, 'conditional_rule_option': rule,
        'conditional_sample_boundary': (
            'The original common sample additionally requires strict-ten-year and five-year overall novelty. '
            'Use conditional_sample to generate its IDs, then fit --conditional-rule manuscript. '
            'The default available-observation diagnostic retains its previous behavior.')}


def fit(args):
    data = panel()
    metrics = pd.read_parquet(output('data/paired_text_metrics.parquet'))
    records, context = [], {}
    with analysis_outputs():
        if 'shuffled_PM_novelty' in metrics:
            pairs = pd.read_parquet(output('paired_text/input/shuffled_pair_mapping.parquet'))
            records, context = fit_shuffled(data, metrics, pairs)
        if 'conditional_M_novelty' in metrics:
            selected, selection = select_conditional_sample(data, metrics, args.conditional_sample,
                getattr(args, 'conditional_rule', 'available'))
            context.update(selection)
            records += fit_variant(selected, 'conditional_M_same_sample')
            records += fit_variant(selected, 'conditional_M_added', extra=['conditional_M_novelty'])
    save('results/paired_text_metadata.json', {'models': records, 'shuffled_context': context})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'encode', 'conditional', 'shuffled-novelty', 'fit'])
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--model-dir')
    parser.add_argument('--conditional-sample', help='Optional fixed-sample patent_id CSV/Parquet for conditional regressions')
    parser.add_argument('--conditional-rule', choices=['available', 'manuscript'], default='available',
                        help='Preserve available-observation default, or require the original generated common sample')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--query-block', type=int, default=64)
    parser.add_argument('--history-block', type=int, default=2048)
    args = parser.parse_args()
    if min(args.batch_size, args.query_block, args.history_block) < 1:
        raise ValueError('Block sizes must be positive')
    {'prepare': prepare, 'encode': encode, 'conditional': conditional,
     'shuffled-novelty': shuffled_novelty, 'fit': fit}[args.action](args)


if __name__ == '__main__':
    main()
