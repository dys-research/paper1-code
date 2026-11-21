"""build entity histories."""
from common import *

import argparse

def main():
    if NEW is None:
        raise ValueError('Configure inputs.patentsview_root before preparing measurement inputs')
    t=time.time();c=connect()
    c.execute('CREATE VIEW dates AS SELECT patent_id,CAST(patent_date AS DATE) grant_date FROM '+readp(NEW/'02_core/patent_dates.parquet'))
    c.execute('CREATE VIEW base AS SELECT patent_id,grant_date FROM '+readp(out('data/patent_panel_base.parquet')))
    summary={}
    for ent in ['assignee','inventor']:
        log('Reconstruct strict pre-grant '+ent+' productivity using relevant entity IDs only.')
        c.execute('CREATE OR REPLACE TABLE base_links AS SELECT * FROM '+readp(out('data/base_'+ent+'_links.parquet')))
        c.execute('''CREATE OR REPLACE TABLE portfolio AS SELECT DISTINCT p.patent_id,p.'''+ent+'''_id entity_id,d.grant_date
          FROM '''+readp(NEW/'02_core'/('patent_'+ent+'.parquet'))+''' p
          JOIN dates d USING(patent_id) SEMI JOIN (SELECT DISTINCT '''+ent+'''_id FROM base_links) e USING('''+ent+'''_id)
          WHERE d.grant_date<=DATE '2020-12-31' ''')
        if ent=='assignee':
            parquet(c,'SELECT * FROM portfolio ORDER BY patent_id,entity_id','data/scope_assignee_portfolio.parquet')
        c.execute('''CREATE OR REPLACE TABLE daily AS SELECT entity_id,grant_date,count(*) granted_n
          FROM portfolio GROUP BY entity_id,grant_date''')
        c.execute('''CREATE OR REPLACE TABLE historical AS SELECT *,
          coalesce(sum(granted_n) OVER w,0) prior_patents,
          min(grant_date) OVER w first_prior_grant,max(grant_date) OVER w latest_prior_grant
          FROM daily WINDOW w AS (PARTITION BY entity_id ORDER BY grant_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)''')
        c.execute('''CREATE OR REPLACE TABLE base_entity AS SELECT l.patent_id,l.'''+ent+'''_id entity_id,b.grant_date,
          h.prior_patents,h.first_prior_grant,h.latest_prior_grant,
          CASE WHEN h.prior_patents>0 THEN date_diff('day',h.first_prior_grant,b.grant_date)/365.25 ELSE 0 END career_years
          FROM base_links l JOIN base b ON l.patent_id=CAST(b.patent_id AS BIGINT)
          LEFT JOIN historical h ON l.'''+ent+'''_id=h.entity_id AND b.grant_date=h.grant_date''')
        assert c.execute('SELECT count(*) FROM base_entity WHERE prior_patents IS NULL OR latest_prior_grant>=grant_date').fetchone()[0]==0
        parquet(c,'SELECT * FROM base_entity ORDER BY patent_id,entity_id','data/base_'+ent+'_history_links.parquet')
        parquet(c,'''SELECT CAST(patent_id AS VARCHAR) patent_id,
          avg(prior_patents) '''+ent+'''_prior_patents_mean,max(prior_patents) '''+ent+'''_prior_patents_max,
          avg(CASE WHEN prior_patents=0 THEN 1.0 ELSE 0.0 END) '''+ent+'''_new_entity_share,
          avg(career_years) '''+ent+'''_career_years_mean,
          max(latest_prior_grant) '''+ent+'''_latest_prior_grant
          FROM base_entity GROUP BY patent_id''','data/'+ent+'_history_controls.parquet')
        summary[ent]=rows(c,'''SELECT count(*) patent_entity_links,count(DISTINCT patent_id) base_patents,
          avg(prior_patents) mean_prior_patents,max(prior_patents) max_prior_patents
          FROM base_entity''')[0]
        summary[ent]['portfolio_links']=c.execute('SELECT count(*) FROM portfolio').fetchone()[0]
    summary.update(seconds=round(time.time()-t,2),date_policy='strictly earlier grant dates; all same-day grants excluded; national observed utility portfolio begins at source coverage',id_vintage='current disambiguated IDs, not historical ID snapshots')
    finish('02_entity_histories',summary)

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
