"""build base panel."""
from common import *

import argparse

def main():
    if NEW is None:
        raise ValueError('Configure inputs.patentsview_root before preparing measurement inputs')
    t=time.time(); c=connect()
    src=[STAGE1/'data/a61b_text_base_1995_2020.parquet',STAGE1/'data/candidate_control_availability.parquet',NEW/'02_core/patent_cpc.parquet',NEW/'02_core/patent_assignee.parquet',NEW/'02_core/patent_inventor.parquet']
    source_receipt(src,'checks/01_sources_before.json')
    c.execute('CREATE VIEW text_base AS SELECT * FROM '+readp(src[0]))
    c.execute('CREATE VIEW prior_controls AS SELECT * FROM '+readp(src[1]))
    c.execute('CREATE TABLE ids AS SELECT CAST(patent_id AS BIGINT) pid FROM text_base')
    log('Read classification and entity IDs for the configured patent sample.')
    for entity in ['assignee','inventor']:
        c.execute('''CREATE TABLE '''+entity+'''_links AS SELECT DISTINCT p.patent_id,p.'''+entity+'''_id
        FROM '''+readp(NEW/'02_core'/('patent_'+entity+'.parquet'))+''' p JOIN ids ON p.patent_id=ids.pid
        WHERE p.'''+entity+'''_id IS NOT NULL''')
        parquet(c,'SELECT * FROM '+entity+'_links ORDER BY patent_id,'+entity+'_id','data/base_'+entity+'_links.parquet')
    c.execute('''CREATE TABLE cpc_counts AS SELECT p.patent_id,count(*) cpc_record_count,
     count(DISTINCT p.cpc_group) cpc_group_count,count(DISTINCT p.cpc_subclass) cpc_subclass_count,
     count(DISTINCT p.cpc_class) cpc_class_count,count(DISTINCT p.cpc_section) cpc_section_count
     FROM '''+readp(src[2])+''' p JOIN ids ON p.patent_id=ids.pid GROUP BY p.patent_id''')
    c.execute(r'''CREATE TABLE normalized AS SELECT t.*,
     trim(regexp_replace(coalesce(patent_title,''),'\s+',' ','g')) title_canonical,
     trim(regexp_replace(coalesce(patent_abstract,''),'\s+',' ','g')) abstract_canonical,
     trim(regexp_replace(coalesce(independent_claim_text,''),'\s+',' ','g')) claim_canonical,
     try_cast(regexp_extract(first_a61b_group,'^A61B([0-9]+)/',1) AS INTEGER) group_no
     FROM text_base t''')
    c.execute('''CREATE TABLE panel AS SELECT t.patent_id,t.grant_date,t.filing_date,t.grant_year,t.filing_year,
     t.first_a61b_group,t.first_a61b_sequence,t.current_primary_cpc_subclass,t.current_primary_cpc_group,
     t.primary_sequence_is_a61b,t.is_focal_candidate,t.is_history_warmup,t.full_input_available,
     t.label_window_end,t.mature_by_local_snapshot,t.local_snapshot_end,
     CASE WHEN group_no IN (1,3,5,6,7,8,9,10,13,16,17,18,34,42,46,50,90) THEN 'A61B'||group_no::VARCHAR
     WHEN group_no-2000 IN (1,3,5,6,7,8,9,10,13,16,17,18,34,42,46,50,90) THEN 'A61B'||(group_no-2000)::VARCHAR
     ELSE 'OTHER_INDEX' END technical_subfield,
     c.non_patent_reference_count,c.doi_like_reference_count,c.num_claims,
     c.related_us_document_count,c.foreign_priority_count,c.pct_document_count,c.application_id,
     c.first_assignee_id,c.prospective_cluster_key assignee_cluster_key,
     c.first_assignee_id IS NULL assignee_missing,
     date_diff('day',t.filing_date,t.grant_date) grant_lag_days,
     date_diff('day',t.filing_date,t.grant_date)/365.25 grant_lag_years,
     (t.filing_date IS NULL OR t.filing_date>t.grant_date) grant_lag_invalid,
     c.grant_lag_years producer_grant_lag_years,
     c.assignee_prior_patents_mean producer_assignee_history_not_main_feature,
     length(title_canonical) title_chars,length(abstract_canonical) abstract_chars,length(claim_canonical) independent_claim_chars,
     CASE WHEN title_canonical='' THEN 0 ELSE array_length(string_split(title_canonical,' ')) END title_words,
     CASE WHEN abstract_canonical='' THEN 0 ELSE array_length(string_split(abstract_canonical,' ')) END abstract_words,
     CASE WHEN claim_canonical='' THEN 0 ELSE array_length(string_split(claim_canonical,' ')) END independent_claim_words,
     coalesce(a.assignee_count,0) assignee_count,coalesce(i.inventor_count,0) inventor_count,
     i.inventor_count IS NULL inventor_links_missing,
     p.* EXCLUDE(patent_id),
     t.claim_source,t.independent_claim_sequence,t.independent_claim_number,t.independence_checked,
     'PENDING_ROLE_VECTORS_AND_OUTCOME_RECONSTRUCTION' panel_status
     FROM normalized t LEFT JOIN prior_controls c USING(patent_id)
     LEFT JOIN (SELECT patent_id,count(*) assignee_count FROM assignee_links GROUP BY patent_id) a ON CAST(t.patent_id AS BIGINT)=a.patent_id
     LEFT JOIN (SELECT patent_id,count(*) inventor_count FROM inventor_links GROUP BY patent_id) i ON CAST(t.patent_id AS BIGINT)=i.patent_id
     LEFT JOIN cpc_counts p ON CAST(t.patent_id AS BIGINT)=p.patent_id''')
    assert c.execute('SELECT count(*)=count(DISTINCT patent_id) FROM panel').fetchone()[0]
    assert c.execute("SELECT count(*) FROM panel WHERE is_focal_candidate AND NOT mature_by_local_snapshot").fetchone()[0]==0
    assert c.execute("SELECT count(*) FROM panel WHERE assignee_missing AND assignee_cluster_key<>'patent:'||patent_id").fetchone()[0]==0
    parquet(c,'SELECT * FROM panel ORDER BY grant_date,patent_id','data/patent_panel_base.parquet')
    csv(c,'''SELECT grant_year,count(*) patents,sum(full_input_available::INT) complete_text,
     sum(assignee_missing::INT) missing_assignee,sum(inventor_links_missing::INT) missing_inventor,
     sum(grant_lag_invalid::INT) invalid_lag,avg(non_patent_reference_count) mean_npl,
     avg(inventor_count) mean_team_size,avg(grant_lag_years) mean_grant_lag_years
     FROM panel GROUP BY grant_year ORDER BY grant_year''','quality/01_base_by_year.csv')
    summary=rows(c,'''SELECT count(*) n,sum(is_focal_candidate::INT) focal_n,sum(is_history_warmup::INT) history_n,
     sum(full_input_available::INT) text_complete_n,sum(assignee_missing::INT) missing_assignee_n,
     sum(inventor_links_missing::INT) missing_inventor_n,sum(grant_lag_invalid::INT) invalid_grant_lag_n,
     sum((non_patent_reference_count IS NULL)::INT) missing_npl_n,sum((num_claims IS NULL)::INT) missing_claim_count_n,
     sum((abs(grant_lag_years-producer_grant_lag_years)>1e-8)::INT) grant_lag_source_difference_n
     FROM panel''')[0]
    summary.update(seconds=round(time.time()-t,2),status='BASE_COVARIATES_READY_HISTORIES_FAMILIES_AND_OUTCOMES_PENDING')
    finish('01_base_panel',summary)

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
