"""Assemble model-ready covariates, novelty, anchoring and mature outcomes."""
from common import *
import argparse
import csv as csvlib

NOVELTY_NAMES = {'overall_tac': 'overall_novelty_raw', 'P': 'problem_novelty_raw',
                 'M': 'mechanism_novelty_raw', 'E': 'effect_novelty_raw',
                 'PM_config': 'pm_config_novelty_raw', 'ME_config': 'me_config_novelty_raw',
                 'overall_ta': 'overall_ta_novelty_raw', 'title': 'title_novelty_raw',
                 'abstract': 'abstract_novelty_raw', 'claim_1': 'claim1_novelty_raw'}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    connection = connect()
    job = out('encoding/full_frozen')
    if json.loads((job / 'status.json').read_text(encoding='utf-8'))['status'] != 'COMPLETE':
        raise ValueError('Complete the source-text encoding job first')
    if json.loads(out('semantic_reuse/all_future_text/reuse_status.json').read_text(encoding='utf-8'))['status'] != 'COMPLETE':
        raise ValueError('Complete semantic reuse before publishing the final panel')
    connection.execute('CREATE VIEW p AS SELECT * FROM ' + readp(out('data/panel_covariates_and_labels_partial.parquet')))
    connection.execute('CREATE VIEW m AS SELECT * FROM ' + readp(job / 'novelty/metrics_long.parquet'))
    observed_kinds = {row['kind'] for row in rows(connection, 'SELECT DISTINCT kind FROM m')}
    if observed_kinds != set(NOVELTY_NAMES):
        raise ValueError('Final panel requires all ten declared representations; found: ' + repr(observed_kinds))
    connection.execute('CREATE TABLE novel AS SELECT patent_id,' + ','.join(
        f"max(CASE WHEN kind='{kind}' THEN novelty_raw END) {name}" for kind, name in NOVELTY_NAMES.items()) +
        ' FROM m GROUP BY patent_id')
    connection.execute('CREATE VIEW a AS SELECT * FROM ' + readp(out('data/anchoring_results.parquet')))
    connection.execute('CREATE VIEW s AS SELECT * FROM ' + readp(out('semantic_reuse/all_future_text/semantic_reuse_results.parquet')))
    replaced = list(NOVELTY_NAMES.values())[:6] + ['anchoring_raw', 'semantic_reuse_5y']
    parquet(connection, '''SELECT p.* EXCLUDE(''' + ','.join(replaced) + ''',panel_status),
        n.* EXCLUDE(patent_id),a.anchoring_raw,a.anchoring_max,a.historical_reference_n,a.covered_M_n,
        a.M_coverage,a.anchoring_computable,a.anchoring_observed_pair_n,s.* EXCLUDE(patent_id),
        CASE WHEN p.is_focal_candidate THEN 'MODEL_READY' ELSE 'HISTORY_REFERENCE_ONLY' END panel_status
        FROM p LEFT JOIN novel n USING(patent_id) LEFT JOIN a USING(patent_id) LEFT JOIN s USING(patent_id)''',
        'data/panel_model_ready.parquet')
    connection.execute('CREATE VIEW final AS SELECT * FROM ' + readp(out('data/panel_model_ready.parquet')))
    checks = rows(connection, '''SELECT count(*) n,count(DISTINCT patent_id) unique_ids,
        count(*) FILTER(WHERE is_focal_candidate AND (NOT reuse_window_mature OR semantic_reuse_5y IS NULL)) unusable_reuse_n,
        count(*) FILTER(WHERE is_focal_candidate AND label_window_end>local_snapshot_end) immature_n,
        count(*) FILTER(WHERE is_focal_candidate AND assignee_latest_prior_grant>=grant_date) future_assignee_n,
        count(*) FILTER(WHERE is_focal_candidate AND inventor_latest_prior_grant>=grant_date) future_inventor_n,
        count(*) FILTER(WHERE is_focal_candidate AND assignee_latest_pregrant_citation_date>=grant_date) future_history_citation_n,
        count(*) FILTER(WHERE is_focal_candidate AND reuse_count_80_all<>
        reuse_count_80_shared+reuse_count_80_newfield+reuse_count_80_cpc_unknown) domain_partition_error_n
        FROM final''')[0]
    if checks['n'] != checks['unique_ids'] or any(value for key, value in checks.items() if key.endswith('_n')):
        raise ValueError('Model-panel validation failed: ' + repr(checks))
    dictionary = {
        'patent_id': 'Unique patent identifier; string representation of the numeric utility grant ID',
        'grant_date': 'Patent grant date', 'filing_date': 'Application filing date',
        'is_focal_candidate': 'Outcome-analysis cohort (2005 through2020); historical references are false',
        'is_history_warmup': 'Historical-reference cohort (1995 through2004)',
        'technical_subfield': 'Mapped main group of first A61B code in recorded sequence',
        'label_window_end': 'Exact fifth anniversary of grant',
        'label_available_date': 'Exact five-year outcome window end; not a source ingestion timestamp',
        'forward_cites_5y': 'Utility citations with citing grant date in(grant, fifth anniversary]',
        'new_field_cites_5y': 'Five-year citations whose known complete CPC subclass sets share no subclass',
        'semantic_reuse_5y': 'log(1+reuse_count_80_all) for mature valid-vector queries only',
        'anchoring_raw': 'Mean focal/citedM cosine over strictly earlier cited patents with validM vectors',
        'M_coverage': 'Observed valid citedM count divided by all eligible earlier cited-patent count',
        'top5_impact_with_ties': 'Citation count at or above cohort95th percentile; ties retained',
        'assignee_cluster_key': 'Primary-assignee key; missing ownership receives patent-specific singleton key',
        'family_size_at_grant_no_cip': 'Date-truncated domestic continuity component size, excluding CIP links',
        'family_size_at_grant_with_cip': 'Date-truncated domestic continuity component size, including CIP links',
    }
    for kind, name in NOVELTY_NAMES.items():
        dictionary[name] = '1-mean(top10cosine); same mapped subfield and preceding ten calendar years; representation=' + kind + '; fewer than10valid historical neighbors remains missing'
    schema = pq.ParquetFile(out('data/panel_model_ready.parquet')).schema_arrow
    with out('data/variable_dictionary_final.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csvlib.writer(stream)
        writer.writerow(['variable', 'arrow_type', 'definition'])
        for field in schema:
            definition = dictionary.get(field.name)
            if definition is None and field.name.startswith('reuse_count_'):
                definition = 'Exact five-year similar-text count; threshold and CPC domain encoded in suffix'
            elif definition is None and field.name.startswith('reuse_excess_'):
                definition = 'Sum ofmax((cosine-threshold)/(1-threshold),0) over eligible five-year text comparisons'
            elif definition is None:
                definition = 'Prepared feature or diagnostic; see measurement scripts and analysis feature mapping'
            writer.writerow([field.name, str(field.type), definition])
    finish('37_final_panel', {'passed': True, 'checks': checks,
                              'panel_sha256': sha(out('data/panel_model_ready.parquet')),
                              'missingness': 'Missing raw role/novelty/anchoring scores remain missing'})


if __name__ == '__main__':
    main()
