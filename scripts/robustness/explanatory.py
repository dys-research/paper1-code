"""Supplementary outcome, sample, evidence-source and anchoring specifications."""
from scripts.robustness.common import *


def main():
    data = pd.read_parquet(output('data/local_panel.parquet'))
    data['patent_id'] = data.patent_id.astype(str)
    data['grant_date'] = pd.to_datetime(data.grant_date)
    summaries, skipped = [], []

    def fit(selected, label, required=(), **kwargs):
        missing = [col for col in required if col not in selected]
        if missing or len(selected) < 10:
            skipped.append({'label': label, 'reason': 'missing columns ' + ','.join(missing)
                            if missing else 'fewer than ten records'})
            return
        summaries.extend(fit_variant(selected, label, **kwargs))

    for column, label in [
        ('cites_5y_excluding_known_same_assignee', 'no_assignee_selfcite'),
        ('cites_5y_excluding_same_family', 'no_family_selfcite'),
        ('cites_5y_excluding_known_same_assignee_or_family', 'no_internal_selfcite')]:
        fit(data, label, required=[column], targets={'citations': (column, 'count')})
    flags_path = MEASUREMENT / 'data/five_year_citation_flags.parquet'
    if flags_path.exists():
        flags = pd.read_parquet(flags_path)
        keep = ~flags.same_assignee.fillna(False).astype(bool) & ~flags.same_family_at_citing_date_no_cip.fillna(False).astype(bool)
        new = flags.loc[keep & ~flags.classification_unknown.fillna(True).astype(bool)
                        & ~flags.shares_any_cpc_subclass.fillna(False).astype(bool)].groupby('cited_id').size()
        data['newfield_no_internal_selfcite'] = data.patent_id.map(new.rename(index=str)).fillna(0)
        if not data.newfield_no_internal_selfcite.le(data.new_field_cites_5y).all():
            raise ValueError('Self-excluded new-field citations exceed original counts')
        fit(data, 'newfield_no_internal_selfcite', targets={
            'newfield': ('newfield_no_internal_selfcite', 'count')})
    else:
        skipped.append({'label': 'newfield_no_internal_selfcite', 'reason': 'citation flags absent'})
    fit(data.loc[~data.is_continuation & ~data.is_division], 'no_continuation_division')
    fit(data.loc[~data.is_continuation & ~data.is_division & ~data.is_cip], 'no_continuation_division_cip')
    extras = [f'log_{r}_span_words' for r in 'PME'] + [
        f'{r}_source_{f}' for r in 'PME' for f in ['claim_1', 'title']] + [f'{r}_new_model' for r in 'PME']
    fit(data, 'role_length_source_model', required=extras, extra=extras)
    if 'primary_sequence_is_a61b' in data:
        fit(data.loc[data.primary_sequence_is_a61b.fillna(False)], 'primary_CPC_A61B')
    fit(data.loc[data.P_source.eq('abstract') & data.E_source.eq('abstract')], 'PE_both_abstract')
    fit(data.loc[data.M_source.eq('abstract')], 'M_from_abstract')
    for threshold in [.7, .8, .9]:
        fit(data.loc[data.minimum_role_confidence.ge(threshold)], f'confidence_{int(threshold*100)}')
    for count in [5, 20]:
        path = output(f'data/novelty_K{count}.parquet')
        if not path.exists():
            skipped.append({'label': f'K{count}', 'reason': 'nearest-neighbour variant absent'})
            continue
        novelty = pd.read_parquet(path).set_index('patent_id')
        selected = data.copy()
        for col in novelty:
            selected[col] = selected.patent_id.map(novelty[col])
        fit(selected, f'K{count}')
    fit(data.loc[data.grant_year <= 2019], 'cohorts_through2019')
    if 'exact_window_cites_observed_through_2024' in data:
        selected = data.copy()
        selected['forward_cites_5y'] = selected.exact_window_cites_observed_through_2024
        fit(selected, 'observed_through2024_diagnostic', targets={'citations': ('forward_cites_5y', 'count')})
    for threshold in [70, 90]:
        column = f'reuse_count_{threshold}_all'
        if column in data:
            selected = data.copy()
            selected['semantic_reuse_5y'] = np.log1p(selected[column])
            fit(selected, f'reuse_threshold{threshold}', targets={'semantic': ('semantic_reuse_5y', 'continuous')})
    composition = ['focal_non_a61b_share', 'backward_non_a61b_share']
    fit(data, 'CPC_composition', required=composition, extra=composition)
    counts = data.assignee_cluster_key.value_counts()
    fit(data.loc[data.assignee_cluster_key.map(counts) >= 100], 'assignee_FE_repeated100',
        fe_assignee=True, models=('M5',))
    selected = data.loc[data.overall_novelty_raw.notna()].copy().reset_index(drop=True)
    with analysis_outputs():
        _, specs, _, _ = ex.prepare(selected, 'anchor_ablation')
    base = specs['M4']
    variants = {'H0': base,
        'Hmissing': base + ['anchor_missing', 'coverage', 'coverage_missing'],
        'Hvalue': base + ['anchor_missing', 'coverage', 'coverage_missing', 'anchor'],
        'H_PM': base + ['anchor_missing', 'coverage', 'coverage_missing', 'anchor', 'PMxA'],
        'Hfull': specs['M5']}
    fit(selected, 'anchor_ablation', models=tuple(variants), terms_override=variants)
    coefficients = [pd.read_csv(path) for path in (ROOT / 'explanatory').glob('*/coefficients.csv')]
    if coefficients:
        pd.concat(coefficients, ignore_index=True).to_csv(output('results/explanatory_coefficients.csv'), index=False)
    save('results/explanatory_metadata.json', {'models': summaries, 'skipped_optional': skipped})
    for item in skipped:
        log('Skipped ' + item['label'] + ': ' + item['reason'])


if __name__ == '__main__':
    main()
