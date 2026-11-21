"""assemble panel and quality."""
from common import *

import argparse

def main():
    t=time.time();c=connect()
    inputs={'b':'patent_panel_base','a':'assignee_history_controls','i':'inventor_history_controls',
     'h':'assignee_historical_impact_controls','f':'domestic_family_controls','y':'citation_outcomes',
     's':'self_citation_outcomes'}
    for k,v in inputs.items():c.execute('CREATE VIEW '+k+' AS SELECT * FROM '+readp(out('data/'+v+'.parquet')))
    c.execute('CREATE VIEW r AS SELECT * FROM '+readp(out('data/role_inputs_primary.parquet')))
    c.execute('''CREATE TABLE role_flags AS SELECT patent_id,
     '''+','.join("max(CASE WHEN role='"+role+"' THEN "+col+" ELSE NULL END) "+role+'_'+alias
     for role in ['P','M','E'] for col,alias in [('role_status','input_status'),('source_field','source'),('span_words','span_words'),
     ('extraction_model','extraction_model'),('usable_for_primary_embedding::INT','usable')])+'''
     FROM r GROUP BY patent_id''')
    c.execute('''CREATE TABLE panel AS SELECT b.* EXCLUDE(inventor_count,panel_status,producer_assignee_history_not_main_feature),
     CASE WHEN b.inventor_links_missing THEN NULL ELSE b.inventor_count END inventor_count,
     a.* EXCLUDE(patent_id),i.* EXCLUDE(patent_id),h.* EXCLUDE(patent_id),f.* EXCLUDE(patent_id),
     y.* EXCLUDE(patent_id),s.* EXCLUDE(patent_id),r.* EXCLUDE(patent_id),

     CASE WHEN f.parent_relation_rows>0 THEN 1.0-f.resolved_parent_rows::DOUBLE/f.parent_relation_rows ELSE NULL END unresolved_parent_relation_share,
     CAST(NULL AS DOUBLE) overall_novelty_raw,CAST(NULL AS DOUBLE) problem_novelty_raw,
     CAST(NULL AS DOUBLE) mechanism_novelty_raw,CAST(NULL AS DOUBLE) effect_novelty_raw,
     CAST(NULL AS DOUBLE) pm_config_novelty_raw,CAST(NULL AS DOUBLE) me_config_novelty_raw,
     CAST(NULL AS DOUBLE) anchoring_raw,CAST(NULL AS DOUBLE) semantic_reuse_5y,
     'COVARIATES_AND_CITATION_LABELS_READY_ROLE_VECTOR_METRICS_PENDING' panel_status
     FROM b LEFT JOIN a USING(patent_id) LEFT JOIN i USING(patent_id) LEFT JOIN h USING(patent_id)
     LEFT JOIN f USING(patent_id) LEFT JOIN y USING(patent_id) LEFT JOIN s USING(patent_id)
     LEFT JOIN role_flags r USING(patent_id)''')
    assert c.execute('SELECT count(*)=count(DISTINCT patent_id) FROM panel').fetchone()[0]
    assert c.execute('''SELECT count(*) FROM panel WHERE label_available_date<>label_window_end
     OR (assignee_latest_prior_grant>=grant_date) OR (inventor_latest_prior_grant>=grant_date)
     OR (assignee_latest_pregrant_citation_date>=grant_date)
     OR (is_focal_candidate AND forward_cites_5y IS NULL)
     OR (NOT is_focal_candidate AND forward_cites_5y IS NOT NULL)
     OR (cites_5y_excluding_known_same_assignee>forward_cites_5y)
     OR (family_size_at_grant_no_cip<1) OR (family_size_at_grant_with_cip<family_size_at_grant_no_cip)''').fetchone()[0]==0
    parquet(c,'SELECT * FROM panel ORDER BY grant_date,patent_id','data/panel_covariates_and_labels_partial.parquet')

    finish('11_covariates_and_labels', {'rows': c.execute('SELECT count(*) FROM panel').fetchone()[0]})

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
