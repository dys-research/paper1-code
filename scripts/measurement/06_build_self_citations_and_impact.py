"""build self citations and impact."""
from common import *
import pyarrow as pa
import numpy as np
from importlib.util import spec_from_file_location,module_from_spec

import argparse

def main():
    if NEW is None:
        raise ValueError('Configure inputs.patentsview_root before preparing measurement inputs')
    t=time.time();c=connect()
    c.execute('CREATE VIEW base AS SELECT patent_id,grant_date,is_focal_candidate FROM '+readp(out('data/patent_panel_base.parquet')))
    c.execute('CREATE TABLE dates AS SELECT patent_id,CAST(patent_date AS DATE) grant_date FROM '+readp(NEW/'02_core/patent_dates.parquet'))
    c.execute('CREATE VIEW fwd AS SELECT * FROM '+readp(out('data/five_year_forward_edges.parquet')))
    c.execute('CREATE TABLE endpoints AS SELECT citing_id pid FROM fwd UNION SELECT cited_id FROM fwd')
    c.execute('''CREATE TABLE assignee_links AS SELECT DISTINCT a.patent_id,a.assignee_id
     FROM '''+readp(NEW/'02_core/patent_assignee.parquet')+''' a SEMI JOIN endpoints e ON a.patent_id=e.pid
     WHERE a.assignee_id IS NOT NULL''')
    log('Identify any-shared-assignee self citations; missing ownership stays explicitly unknown.')
    c.execute('''CREATE TABLE assignee_self AS SELECT DISTINCT f.citing_id,f.cited_id
     FROM fwd f JOIN assignee_links a ON f.citing_id=a.patent_id JOIN assignee_links b
     ON f.cited_id=b.patent_id AND a.assignee_id=b.assignee_id''')
    c.execute('''CREATE TABLE flags AS SELECT f.*,
     s.citing_id IS NOT NULL same_assignee,
     ca.patent_id IS NOT NULL AND cb.patent_id IS NOT NULL assignee_comparison_known,
     g.same_family_at_citing_date_no_cip,g.same_family_at_citing_date_with_cip
     FROM fwd f LEFT JOIN assignee_self s USING(citing_id,cited_id)
     LEFT JOIN (SELECT DISTINCT patent_id FROM assignee_links) ca ON f.citing_id=ca.patent_id
     LEFT JOIN (SELECT DISTINCT patent_id FROM assignee_links) cb ON f.cited_id=cb.patent_id
     LEFT JOIN '''+readp(out('data/five_year_family_citation_flags.parquet'))+''' g USING(citing_id,cited_id)''')
    assert c.execute('SELECT count(*) FROM flags WHERE same_family_at_citing_date_no_cip IS NULL').fetchone()[0]==0
    parquet(c,'SELECT * FROM flags ORDER BY citing_date,cited_id,citing_id','data/five_year_citation_flags.parquet')
    parquet(c,'''SELECT b.patent_id,
     CASE WHEN b.is_focal_candidate THEN coalesce(f.same_assignee_cites,0) ELSE NULL END same_assignee_cites_5y,
     CASE WHEN b.is_focal_candidate THEN coalesce(f.owner_unknown_cites,0) ELSE NULL END assignee_unknown_cites_5y,
     CASE WHEN b.is_focal_candidate THEN coalesce(f.samefamily_no_cip,0) ELSE NULL END same_family_cites_5y_no_cip,
     CASE WHEN b.is_focal_candidate THEN coalesce(f.samefamily_with_cip,0) ELSE NULL END same_family_cites_5y_with_cip,
     CASE WHEN b.is_focal_candidate THEN coalesce(f.external_assignee,0) ELSE NULL END cites_5y_excluding_known_same_assignee,
     CASE WHEN b.is_focal_candidate THEN coalesce(f.external_family,0) ELSE NULL END cites_5y_excluding_same_family,
     CASE WHEN b.is_focal_candidate THEN coalesce(f.external_both,0) ELSE NULL END cites_5y_excluding_known_same_assignee_or_family
     FROM base b LEFT JOIN (SELECT CAST(cited_id AS VARCHAR) patent_id,
     count(*) FILTER(WHERE same_assignee) same_assignee_cites,
     count(*) FILTER(WHERE NOT assignee_comparison_known) owner_unknown_cites,
     count(*) FILTER(WHERE same_family_at_citing_date_no_cip) samefamily_no_cip,
     count(*) FILTER(WHERE same_family_at_citing_date_with_cip) samefamily_with_cip,
     count(*) FILTER(WHERE NOT same_assignee) external_assignee,
     count(*) FILTER(WHERE NOT same_family_at_citing_date_no_cip) external_family,
     count(*) FILTER(WHERE NOT same_assignee AND NOT same_family_at_citing_date_no_cip) external_both
     FROM flags GROUP BY cited_id) f USING(patent_id)''','data/self_citation_outcomes.parquet')
    selfsummary=rows(c,'''SELECT count(*) n,sum(same_assignee::INT) same_assignee,
     sum((NOT assignee_comparison_known)::INT) owner_unknown,
     sum(same_family_at_citing_date_no_cip::INT) samefamily_no_cip,
     sum(same_family_at_citing_date_with_cip::INT) samefamily_with_cip
     FROM flags''')[0]
    finish('06a_self_citations',{'five_year_edges':selfsummary,'seconds':round(time.time()-t,2),
     'missing_owner_policy':'exclude only established shared-assignee citations; unknown is not asserted to be independent'})

    log('Reconstruct organizational historical impact: read citation ID columns for relevant assignee portfolios, count only citations published before each focal grant.')
    c.execute('CREATE VIEW portfolio AS SELECT * FROM '+readp(out('data/scope_assignee_portfolio.parquet')))
    c.execute('''CREATE TABLE impact_daily AS SELECT p.entity_id,cd.grant_date citation_date,count(*) cite_n
     FROM '''+readp(NEW/'02_core/citation.parquet')+''' e JOIN portfolio p ON e.citation_patent_id=p.patent_id
     JOIN dates cd ON e.patent_id=cd.patent_id
     WHERE e.patent_id<>e.citation_patent_id AND p.grant_date<=cd.grant_date AND cd.grant_date<=DATE '2020-12-31'
     GROUP BY p.entity_id,cd.grant_date''')
    c.execute('''CREATE TABLE cumulative AS SELECT *,sum(cite_n) OVER(PARTITION BY entity_id ORDER BY citation_date ROWS UNBOUNDED PRECEDING) received_cites
     FROM impact_daily''')
    parquet(c,'SELECT * FROM cumulative ORDER BY entity_id,citation_date','data/assignee_historical_citation_daily.parquet')
    c.execute('CREATE VIEW base_entity AS SELECT * FROM '+readp(out('data/base_assignee_history_links.parquet')))
    c.execute('''CREATE TABLE base_impact AS SELECT b.patent_id,b.entity_id,b.grant_date,
     coalesce(i.received_cites,0) prior_received_cites,i.citation_date latest_observed_citation_date
     FROM base_entity b ASOF LEFT JOIN cumulative i ON b.entity_id=i.entity_id AND b.grant_date>i.citation_date''')
    assert c.execute('SELECT count(*) FROM base_impact WHERE latest_observed_citation_date>=grant_date').fetchone()[0]==0
    parquet(c,'SELECT * FROM base_impact','data/base_assignee_historical_impact_links.parquet')
    parquet(c,'''SELECT CAST(patent_id AS VARCHAR) patent_id,
     avg(prior_received_cites) assignee_prior_received_cites_mean,max(prior_received_cites) assignee_prior_received_cites_max,
     max(latest_observed_citation_date) assignee_latest_pregrant_citation_date
     FROM base_impact GROUP BY patent_id''','data/assignee_historical_impact_controls.parquet')
    finish('06b_assignee_historical_impact',{'base_entity_links':c.execute('SELECT count(*) FROM base_impact').fetchone()[0],
     'daily_entity_rows':c.execute('SELECT count(*) FROM impact_daily').fetchone()[0],
     'historical_credited_citation_edges':c.execute('SELECT sum(cite_n) FROM impact_daily').fetchone()[0],
     'source_scope':'observed utility-to-utility core citation edges credited to each linked current assignee; no fractional allocation',
     'time_rule':'citation grant date strictly before focal grant; no final-portfolio future citation totals',
     'seconds':round(time.time()-t,2)})

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
