"""Compare manuscript feature sets using only mature fitting labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from project_config import input_path, setting, work_dir
from scripts.analysis import common, prediction as pred
from scripts.analysis.common import OUTCOMES, STAGE3, default, digest, log, output, rawfeatures, readpanel, save

FEATURES = {'common': 'B2', 'roles': 'B3', 'text_fields': 'B2',
            'structure': 'B5', 'overall': 'B2', 'overall_structure': 'B5', 'lexical': 'B2'}
DEFAULT_FEATURES = ['common', 'roles', 'text_fields', 'structure', 'overall', 'overall_structure']


def aligned_overall_vectors(d):
    """Locate vectors by patent identifier; never infer alignment by row count."""
    path = input_path('overall_vectors', required=False)
    if path is None:
        path = STAGE3 / 'encoding/full_frozen/vectors.npy'
        metadata = STAGE3 / 'encoding/full_frozen/task_metadata.parquet'
        if not path.is_file() or not metadata.is_file():
            raise FileNotFoundError('Run measurement encoding, or configure inputs.overall_vectors and inputs.overall_vector_ids.')
        tasks = pd.read_parquet(metadata, columns=['patent_id', 'kind', 'embedding_row'])
        tasks = tasks.loc[tasks.kind.eq('overall_tac')].copy()
        tasks['patent_id'] = tasks.patent_id.astype(str)
        if not tasks.patent_id.is_unique:
            raise ValueError('Multiple overall_tac tasks per patent')
        indices = d.patent_id.map(tasks.set_index('patent_id').embedding_row)
        if indices.isna().any():
            raise ValueError('Overall-text vectors are missing for panel patents')
        rows = indices.to_numpy(np.int64)
        provenance = {'vectors': str(path), 'metadata': str(metadata), 'metadata_sha256': digest(metadata)}
    else:
        ids_path = input_path('overall_vector_ids')
        if ids_path.suffix == '.parquet':
            ids = pd.read_parquet(ids_path)['patent_id'].astype(str)
        elif ids_path.suffix == '.csv':
            ids = pd.read_csv(ids_path, dtype={'patent_id': str})['patent_id']
        else:
            ids = pd.Series(ids_path.read_text(encoding='utf-8').splitlines(), dtype=str)
        if not ids.is_unique:
            raise ValueError('overall_vector_ids must contain one unique patent_id per vector row')
        indices = d.patent_id.map(pd.Series(np.arange(len(ids)), index=ids))
        if indices.isna().any():
            raise ValueError('overall_vector_ids does not cover the focal panel')
        rows = indices.to_numpy(np.int64)
        provenance = {'vectors': str(path), 'ids': str(ids_path), 'ids_sha256': digest(ids_path)}
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != 768:
        raise ValueError('Expected 768-dimensional overall embeddings')
    if len(rows) and (rows.min() < 0 or rows.max() >= len(values)):
        raise ValueError('Overall embedding row index is out of bounds')
    provenance.update(vectors_sha256=digest(path), dimensions=768)
    return values, rows, provenance


def scaled_extra(fit, apply):
    """Preserve the fitting-partition scaling used by the supplied benchmarks."""
    median = np.nanmedian(fit, axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    a = np.where(np.isfinite(fit), fit, median)
    b = np.where(np.isfinite(apply), apply, median)
    mean = a.mean(axis=0)
    scale = a.std(axis=0)
    scale[scale < 1e-12] = 1.
    return (a - mean) / scale, (b - mean) / scale, {'median': median, 'mean': mean, 'scale': scale}


def extra_features(d, fitix, applyix, vectors, vector_rows, need_fields, need_pca, need_lexical=False):
    result, transforms = {}, {}
    if need_fields:
        raw = d[['title_novelty_raw', 'abstract_novelty_raw', 'claim1_novelty_raw']].to_numpy(float)
        a, b, meta = scaled_extra(raw[fitix], raw[applyix])
        result['text_fields'] = (np.column_stack([a, ~np.isfinite(raw[fitix])]),
                                 np.column_stack([b, ~np.isfinite(raw[applyix])]))
        transforms['text_fields'] = meta
    if need_lexical:
        raw = d[['lexical_new_share_1', 'lexical_new_share_2', 'lexical_new_share_3']].to_numpy(float)
        a, b, meta = scaled_extra(raw[fitix], raw[applyix])
        result['lexical'] = (np.column_stack([a, ~np.isfinite(raw[fitix])]),
                             np.column_stack([b, ~np.isfinite(raw[applyix])]))
        transforms['lexical'] = meta
    if need_pca:
        dimension = int(setting('analysis', 'pca_components', 16))
        if dimension > min(len(fitix), vectors.shape[1]):
            raise ValueError('PCA dimension exceeds the fitting matrix dimensions')
        encoder = PCA(n_components=dimension, svd_solver='randomized',
                      random_state=int(setting('analysis', 'random_seed', 20261103)))
        a = encoder.fit_transform(vectors[vector_rows[fitix]])
        b = encoder.transform(vectors[vector_rows[applyix]])
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite overall-text embeddings or PCA scores')
        a, b, meta = scaled_extra(a, b)
        result['overall'] = (a, b)
        transforms['overall'] = meta
        transforms['pca_components'] = dimension
        transforms['pca_explained_variance_ratio'] = encoder.explained_variance_ratio_
        transforms['pca_model'] = encoder
    return result, transforms


def feature_matrix(base, pre, name, extras, side):
    indices = pre.indices(FEATURES[name])
    parts = [base[:, indices]]
    if name in ('text_fields', 'lexical'):
        parts.append(extras[name][side])
    if name in ('overall', 'overall_structure'):
        parts.append(extras['overall'][side])
    matrix = np.column_stack(parts)
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite forecasting design matrix')
    return matrix, indices


def fit_tree(kind, x, y, leaves):
    cls = HistGradientBoostingClassifier if kind == 'binary' else HistGradientBoostingRegressor
    loss = {} if kind == 'binary' else {'loss': 'poisson' if kind == 'count' else 'squared_error'}
    model = cls(max_iter=100, max_leaf_nodes=leaves, learning_rate=.1, l2_regularization=1.,
                early_stopping=False, random_state=int(setting('analysis', 'random_seed', 20261103)), **loss)
    model.fit(x, y)
    return model


def run(args):
    started = time.time()
    d = readpanel()
    raw = rawfeatures(d)
    names = args.features.split(',') if args.features else DEFAULT_FEATURES
    unknown = set(names) - set(FEATURES)
    if unknown:
        raise ValueError('Unknown feature sets: ' + ', '.join(sorted(unknown)))
    targets = args.targets.split(',') if args.targets else list(OUTCOMES)
    if set(targets) - set(OUTCOMES):
        raise ValueError('Unknown outcome target')
    need_pca = bool(set(names) & {'overall', 'overall_structure'})
    vectors = vector_rows = None
    provenance = {}
    if 'lexical' in names:
        lexical_path = input_path('lexical_features', required=False) or work_dir('robustness') / 'data/local_lexical_features.parquet'
        lexical = pd.read_parquet(lexical_path)
        lexical['patent_id'] = lexical.patent_id.astype(str)
        if not lexical.patent_id.is_unique:
            raise ValueError('Lexical features require unique patent identifiers')
        indexed = lexical.set_index('patent_id')
        for n in (1, 2, 3):
            d['lexical_new_share_' + str(n)] = d.patent_id.map(indexed['lexical_new_share_' + str(n)])
        provenance['lexical_sha256'] = digest(lexical_path)
    if need_pca:
        vectors, vector_rows, embedding_provenance = aligned_overall_vectors(d)
        provenance.update(embedding_provenance)
    panel_path = input_path('panel', required=False) or STAGE3 / 'data/panel_model_ready.parquet'
    design = {'panel_sha256': digest(panel_path), 'overall_embeddings': provenance,
              'code_sha256': {p.name: digest(p) for p in [Path(__file__), Path(pred.__file__), Path(common.__file__)]},
              'feature_sets': names, 'targets': targets, 'learner': args.learner,
              'first_test_year': args.first_year, 'last_test_year': args.last_year,
              'pca_components': int(setting('analysis', 'pca_components', 16)), 'alphas': pred.ALPHAS,
              'random_seed': int(setting('analysis', 'random_seed', 20261103)),
              'fitting_rule': 'initial <= t-8; validation t-7:t-6; refit <= t-6',
              'information_cutoff': '1 January of test year; labels must be fully available before fitting'}
    fingerprint = hashlib.sha256(json.dumps(design, sort_keys=True, default=default).encode()).hexdigest()
    design['fingerprint'] = fingerprint
    save('metadata/forecast_design.json', design)
    all_metrics, all_predictions = [], []
    for year in range(args.first_year, args.last_year + 1):
        ti = np.flatnonzero(d.grant_year <= year - 8)
        vi = np.flatnonzero(d.grant_year.between(year - 7, year - 6))
        ri = np.r_[ti, vi]
        si = np.flatnonzero(d.grant_year == year)
        if not len(ti) or not len(vi) or not len(si):
            raise ValueError(f'Empty fitting, validation or test partition for {year}')
        origin = pd.Timestamp(year=year, month=1, day=1)
        if not ((d.iloc[ri].label_window_end < origin).all() and
                (d.iloc[ri].label_available_date <= origin).all()):
            raise ValueError(f'Unmatured or unavailable fitting labels for test year {year}')
        pre = pred.Preprocess().fit(raw.iloc[ti])
        refit = pred.Preprocess().fit(raw.iloc[ri])
        a, b = pre.transform(raw.iloc[ti]), pre.transform(raw.iloc[vi])
        af, test = refit.transform(raw.iloc[ri]), refit.transform(raw.iloc[si])
        initial, initial_meta = extra_features(d, ti, vi, vectors, vector_rows,
                                              'text_fields' in names, need_pca, 'lexical' in names)
        final, final_meta = extra_features(d, ri, si, vectors, vector_rows,
                                          'text_fields' in names, need_pca, 'lexical' in names)
        joblib.dump({'initial': initial_meta, 'refit': final_meta,
                     'initial_preprocessor': pre, 'refit_preprocessor': refit,
                     'design_fingerprint': fingerprint}, output(f'preprocessing/forecast_{year}.joblib'), compress=3)
        split = {'test_year': year, 'n_train': len(ti), 'n_valid': len(vi),
                 'n_refit': len(ri), 'n_test': len(si), 'forecast_origin': str(origin.date())}
        for target in targets:
            col, kind = OUTCOMES[target]
            y, yv = d.iloc[ti][col].to_numpy(float), d.iloc[vi][col].to_numpy(float)
            yf, yt = d.iloc[ri][col].to_numpy(float), d.iloc[si][col].to_numpy(float)
            if not all(np.isfinite(z).all() for z in (y, yv, yf, yt)):
                raise ValueError(f'Missing or invalid {target} outcome labels')
            for name in names:
                rel = f'jobs/{args.learner}/{target}_{year}_{name}'
                done = output(rel + '/done.json')
                if done.is_file() and not args.overwrite:
                    record = json.loads(done.read_text(encoding='utf-8'))
                    if record.get('design_fingerprint') != fingerprint:
                        raise RuntimeError(f'Cached job has a different design: {rel}. Use --overwrite or a new work directory.')
                    forecast = pd.read_parquet(output(rel + '/predictions.parquet'))
                    all_metrics.append(record)
                    all_predictions.append(forecast)
                    continue
                tick = time.time()
                aa, indices = feature_matrix(a, pre, name, initial, 0)
                bb, _ = feature_matrix(b, pre, name, initial, 1)
                aa_f, final_indices = feature_matrix(af, refit, name, final, 0)
                tt, _ = feature_matrix(test, refit, name, final, 1)
                trials = []
                if args.learner == 'glm':
                    for alpha in pred.ALPHAS:
                        model, diagnostic = pred.learn(kind, alpha, aa, y)
                        estimate = pred.predict(model, kind, bb)
                        trials.append({'alpha': alpha, 'loss': pred.selection_loss(kind, yv, estimate), **diagnostic})
                    choice = min(trials, key=lambda r: (r['loss'], r['alpha']))['alpha']
                    model, diagnostic = pred.learn(kind, choice, aa_f, yf)
                else:
                    for leaves in [7, 15]:
                        model = fit_tree(kind, aa, y, leaves)
                        estimate = pred.predict(model, kind, bb)
                        trials.append({'leaves': leaves, 'loss': pred.selection_loss(kind, yv, estimate)})
                    choice = min(trials, key=lambda r: (r['loss'], r['leaves']))['leaves']
                    model = fit_tree(kind, aa_f, yf, choice)
                    diagnostic = {'learner': 'histogram_gradient_boosting', 'max_leaf_nodes': choice}
                estimate = pred.predict(model, kind, tt)
                forecast = d.iloc[si][['patent_id', 'grant_year', 'technical_subfield', 'assignee_cluster_key']].copy()
                forecast['target'], forecast['model'], forecast['learner'] = target, name, args.learner
                forecast['y_true'], forecast['prediction'] = yt, estimate
                forecast.to_parquet(output(rel + '/predictions.parquet'), index=False)
                joblib.dump({'model': model, 'preprocessor': refit, 'base_indices': final_indices,
                             'feature_set': name, 'extra_transforms': final_meta,
                             'design_fingerprint': fingerprint}, output(rel + '/model.joblib'), compress=3)
                record = {'target': target, 'kind': kind, 'model': name, 'learner': args.learner,
                          **split, **pred.metrics(kind, yt, estimate, forecast.patent_id),
                          'choice': choice, 'trials': trials, 'fit_diagnostics': diagnostic,
                          'design_fingerprint': fingerprint, 'seconds': round(time.time() - tick, 2)}
                save(rel + '/done.json', record)
                all_metrics.append(record)
                all_predictions.append(forecast)
                log(f'{args.learner} {target} {year} {name} complete')
    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ('trials', 'fit_diagnostics')} for r in all_metrics])
    table.to_csv(output('results/forecast_metrics_by_year.csv'), index=False)
    pd.concat(all_predictions, ignore_index=True).to_parquet(output('results/forecast_predictions.parquet'), index=False)
    save('metadata/forecast_complete.json', {'complete': True, 'design_fingerprint': fingerprint,
                                          'jobs': len(table), 'seconds': round(time.time() - started, 2)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learner', choices=['glm', 'tree'], default='glm')
    parser.add_argument('--features', help='Comma-separated: ' + ','.join(FEATURES))
    parser.add_argument('--targets', help='Comma-separated: ' + ','.join(OUTCOMES))
    parser.add_argument('--first-year', type=int, default=int(setting('study', 'test_start', 2015)))
    parser.add_argument('--last-year', type=int, default=int(setting('study', 'test_end', 2020)))
    parser.add_argument('--overwrite', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
