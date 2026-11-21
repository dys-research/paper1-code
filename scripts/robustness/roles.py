"""Role ablations, P/E overlap restrictions and matched-domain uptake models."""
from scripts.robustness.common import *
import statsmodels.api as sm
from scipy import stats


def main():
    data = pd.read_parquet(output('data/local_panel.parquet'))
    data = data.loc[data.overall_novelty_raw.notna()].copy().reset_index(drop=True)
    records, skipped = [], []
    with analysis_outputs():
        x, specs, fe, _ = ex.prepare(data, 'role_ablation')
        x['PEmean'] = (x.P + x.E) / 2
        variants = {'DROP_P': [n for n in specs['M5'] if n != 'P'],
            'DROP_E': [n for n in specs['M5'] if n != 'E'],
            'PE_MEAN': [n for n in specs['M5'] if n not in ['P', 'E']] + ['PEmean']}
        for target in ['citations', 'semantic', 'newfield']:
            keep = ~data.technical_subfield.eq('A61B13') if target == 'newfield' else pd.Series(True, index=data.index)
            for model, terms in variants.items():
                records.append(ex.fitmodel(data.loc[keep], x.loc[keep], fe, terms,
                                           target, OUTCOMES[target][1], 'role_ablation', model))
    overlap_path = output('data/within_patent_role_cosines.parquet')
    if overlap_path.exists():
        overlap = pd.read_parquet(overlap_path)
        different = set(overlap.loc[overlap.pair.eq('P_E') & ~overlap.identical_text, 'patent_id'])
        records += fit_variant(data.loc[data.patent_id.isin(different)], 'exclude_identical_PE')
    else:
        skipped.append('P/E nonidentical restriction requires cached role overlap')
    reference = setting('robustness', 'reference_extractor_model', 'deepseek-v4-flash-260425')
    mask = np.logical_and.reduce([data[r + '_extraction_model'].eq(reference) for r in 'PME'])
    if mask.sum() >= 10:
        records += fit_variant(data.loc[mask], 'reference_extractor_only')
    for label, mask in [('below50', data.M_coverage.lt(.5)), ('atleast50', data.M_coverage.ge(.5))]:
        selected = data.loc[mask & data.anchoring_raw.notna()]
        if len(selected) >= 10:
            records += fit_variant(selected, 'anchor_coverage_' + label)
    binomial = []
    for prefix, inside, outside in [
        ('citation', 'within_shared_subclass_cites_5y', 'new_field_cites_5y'),
        ('reuse80', 'reuse_count_80_shared', 'reuse_count_80_newfield')]:
        if inside not in data or outside not in data:
            skipped.append(prefix + ': matched-domain outcomes absent')
            continue
        selected = data.loc[(data[inside] + data[outside]) > 0].copy().reset_index(drop=True)
        with analysis_outputs():
            xx, ss, ff, _ = ex.prepare(selected, 'binomial_' + prefix)
        keep = np.ones(len(selected), bool)
        for col in ['grant_year', 'technical_subfield']:
            success = selected[outside].groupby(selected[col]).sum()
            failure = selected[inside].groupby(selected[col]).sum()
            bad = success.index[(success == 0) | (failure == 0)]
            keep &= ~selected[col].isin(bad).to_numpy()
        for model in ['M2', 'M5']:
            design, names, _, _ = ex.design(xx.loc[keep], ss[model], ff)
            response = selected.loc[keep, [outside, inside]].to_numpy(float)
            result = sm.GLM(response, design, family=sm.families.Binomial()).fit(
                maxiter=250, tol=1e-8, cov_type='cluster',
                cov_kwds={'groups': selected.loc[keep].assignee_cluster_key.to_numpy()})
            if not result.converged:
                raise RuntimeError('Grouped-binomial model did not converge')
            for index, term in enumerate(names):
                binomial.append({'target': prefix, 'model': model, 'term': term,
                    'coef': result.params[index], 'std_err': result.bse[index],
                    'p_value': 2 * stats.norm.sf(abs(result.params[index] / result.bse[index])),
                    'n_patents': int(keep.sum())})
    pd.DataFrame(binomial).to_csv(output('results/matched_domain_binomial.csv'), index=False)
    save('results/role_coverage_metadata.json', {'models': records, 'skipped_optional': skipped})


if __name__ == '__main__':
    main()
