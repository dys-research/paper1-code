"""Prepare supplementary inputs from measured data and cached neighbours."""
from scripts.robustness.common import *
import duckdb


def add_confidence(data):
    """Use confidence from the extraction that supplied a role, when available."""
    path = source('role_inputs', MEASUREMENT / 'data/role_inputs_primary.parquet')
    if not path.exists():
        log('No role-input file found; confidence restrictions will be unavailable.')
        return data
    roles = pd.read_parquet(path)
    roles['patent_id'] = roles.patent_id.astype(str)
    if 'confidence' not in roles:
        return data
    for role in 'PME':
        selected = roles.loc[roles.role.eq(role)]
        if selected.patent_id.duplicated().any():
            raise ValueError(f'Duplicate {role} records in role input file')
        if role + '_confidence' not in data:
            data[role + '_confidence'] = data.patent_id.map(
                selected.set_index('patent_id').confidence)
    return data


def main():
    data = panel()
    skipped = []
    task_path = MEASUREMENT / 'encoding/full_frozen/task_metadata.parquet'
    vector_path = MEASUREMENT / 'encoding/full_frozen/vectors.npy'
    if task_path.exists() and vector_path.exists():
        tasks = pd.read_parquet(task_path, columns=['patent_id', 'kind', 'embedding_row'])
        tasks['patent_id'] = tasks.patent_id.astype(str)
        vectors = np.load(vector_path, mmap_mode='r')
        wide = tasks.loc[tasks.kind.isin(['P', 'M', 'E'])].pivot(
            index='patent_id', columns='kind', values='embedding_row')
        overlaps = []
        for left, right in [('P', 'E'), ('P', 'M'), ('M', 'E')]:
            if left not in wide or right not in wide:
                continue
            selected = wide.reindex(data.patent_id)[[left, right]].dropna()
            cosine = []
            for start in range(0, len(selected), 4096):
                rows = selected.iloc[start:start+4096]
                a = np.asarray(vectors[rows[left].to_numpy(np.int64)], float)
                b = np.asarray(vectors[rows[right].to_numpy(np.int64)], float)
                cosine.extend(np.clip(np.einsum('ij,ij->i', a, b), -1, 1))
            overlaps.append(pd.DataFrame({'patent_id': selected.index,
                'pair': left + '_' + right, 'cosine': cosine,
                'identical_text': selected[left].to_numpy() == selected[right].to_numpy()}))
        if overlaps:
            overlap = pd.concat(overlaps, ignore_index=True)
            overlap.to_parquet(output('data/within_patent_role_cosines.parquet'), index=False)
            summaries = []
            for name, group in overlap.groupby('pair'):
                summaries.append({'pair': name, 'n': len(group), 'mean': group.cosine.mean(),
                    'median': group.cosine.median(), 'p05': group.cosine.quantile(.05),
                    'p95': group.cosine.quantile(.95),
                    'cosine_ge_095_share': group.cosine.ge(.95).mean(),
                    'identical_text_n': int(group.identical_text.sum()),
                    'identical_text_share': group.identical_text.mean()})
            pd.DataFrame(summaries).to_csv(output('results/role_overlap_distribution.csv'), index=False)
    else:
        skipped.append('role-vector overlap: task metadata or vectors absent')
    kind_columns = {'overall_tac': 'overall_novelty_raw', 'P': 'problem_novelty_raw',
        'M': 'mechanism_novelty_raw', 'E': 'effect_novelty_raw',
        'PM_config': 'pm_config_novelty_raw', 'ME_config': 'me_config_novelty_raw'}
    for count in [5, 20]:
        result = data[['patent_id']].copy()
        available = True
        for kind, col in kind_columns.items():
            folder = MEASUREMENT / 'encoding/full_frozen/novelty' / kind
            if not (folder / 'metrics.parquet').exists() or not (folder / 'nearest_neighbors.npz').exists():
                available = False
                break
            metrics = pd.read_parquet(folder / 'metrics.parquet', columns=['patent_id'])
            with np.load(folder / 'nearest_neighbors.npz') as cached:
                similarities = cached['cosines'][:, :count]
            if similarities.shape != (len(metrics), count):
                raise ValueError(f'{kind}: cache has fewer than {count} neighbour columns')
            valid = np.isfinite(similarities).all(axis=1)
            values = np.where(valid, 1 - np.where(np.isfinite(similarities), similarities, 0).mean(
                axis=1, dtype=np.float64), np.nan)
            result[col] = result.patent_id.map(pd.Series(values, index=metrics.patent_id.astype(str)))
        if available:
            result.to_parquet(output(f'data/novelty_K{count}.parquet'), index=False)
        else:
            skipped.append(f'K={count}: one or more nearest-neighbour caches absent')
    data = add_confidence(data)
    reference_model = setting('robustness', 'reference_extractor_model', 'deepseek-v4-flash-260425')
    sources = []
    for role in 'PME':
        for name in [role + '_confidence', role + '_span_words']:
            if name not in data:
                data[name] = np.nan
        if role + '_source' not in data:
            data[role + '_source'] = None
        if role + '_extraction_model' not in data:
            data[role + '_extraction_model'] = None
        data['log_' + role + '_span_words'] = np.log1p(data[role + '_span_words'])
        for field in ['claim_1', 'title']:
            data[role + '_source_' + field] = data[role + '_source'].eq(field).astype(float)
        data[role + '_new_model'] = data[role + '_extraction_model'].ne(reference_model).astype(float)
        for (field, model), group in data.groupby([role + '_source', role + '_extraction_model'], dropna=False):
            sources.append({'role': role, 'source': field, 'model': model, 'n': len(group),
                'median_words': group[role + '_span_words'].median(),
                'p95_words': group[role + '_span_words'].quantile(.95)})
    pd.DataFrame(sources).to_csv(output('results/role_source_length_model_distribution.csv'), index=False)
    data['minimum_role_confidence'] = data[[r + '_confidence' for r in 'PME']].min(axis=1, skipna=False)
    scope = MEASUREMENT / 'data/citation_scope_cpc_sets.parquet'
    edges = MEASUREMENT / 'data/study_citation_edges.parquet'
    if scope.exists() and edges.exists():
        connection = duckdb.connect()
        connection.execute('SET threads=2')
        connection.register('focal', data[['patent_id', 'grant_date']])
        scope_q = str(scope).replace("'", "''")
        edge_q = str(edges).replace("'", "''")
        attrs = connection.execute(f'''SELECT CAST(s.patent_id AS VARCHAR) patent_id,
            len(s.subclass_set) cpc_subclasses_known,
            list_count(list_filter(s.subclass_set,lambda x: x<>'A61B'))::DOUBLE/nullif(len(s.subclass_set),0) focal_non_a61b_share
            FROM read_parquet('{scope_q}') s JOIN focal f ON CAST(s.patent_id AS VARCHAR)=f.patent_id''').df()
        backward = connection.execute(f'''SELECT f.patent_id,count(*) backward_edges_known_scope,
            avg((NOT s.contains_a61b)::INT) backward_non_a61b_share
            FROM focal f JOIN read_parquet('{edge_q}') e ON CAST(e.citing_id AS VARCHAR)=f.patent_id
            JOIN read_parquet('{scope_q}') s ON e.cited_id=s.patent_id
            WHERE e.cited_date<f.grant_date GROUP BY f.patent_id''').df()
        connection.close()
        data = data.merge(attrs, on='patent_id', how='left', validate='one_to_one').merge(
            backward, on='patent_id', how='left', validate='one_to_one')
    else:
        skipped.append('CPC composition: citation scope or edge file absent')
    data.to_parquet(output('data/local_panel.parquet'), index=False)
    save('results/preparation.json', {'n': len(data), 'skipped_optional': skipped,
        'reference_extractor_model': reference_model,
        'confidence_observed_n': int(data.minimum_role_confidence.notna().sum())})
    for message in skipped:
        log('Skipped optional diagnostic: ' + message)


if __name__ == '__main__':
    main()
