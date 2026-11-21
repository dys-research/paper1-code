"""build citation outcomes."""
from common import *

import argparse

def main():
    if NEW is None:
        raise ValueError('Configure inputs.patentsview_root before preparing measurement inputs')
    t=time.time(); c=connect()
    c.execute('CREATE VIEW base AS SELECT * FROM '+readp(out('data/patent_panel_base.parquet')))
    c.execute('CREATE TABLE ids AS SELECT CAST(patent_id AS BIGINT) pid FROM base')
    c.execute('CREATE TABLE dates AS SELECT patent_id,CAST(patent_date AS DATE) grant_date FROM '+readp(NEW/'02_core/patent_dates.parquet'))
    log('Retain citation edges touching the focal or historical sample.')
    c.execute('''CREATE TABLE selected_raw AS SELECT e.patent_id citing_id,e.citation_patent_id cited_id
     FROM '''+readp(NEW/'02_core/citation.parquet')+''' e
     LEFT JOIN ids a ON e.patent_id=a.pid LEFT JOIN ids b ON e.citation_patent_id=b.pid
     WHERE a.pid IS NOT NULL OR b.pid IS NOT NULL''')
    raw_n=c.execute('SELECT count(*) FROM selected_raw').fetchone()[0]
    c.execute('''CREATE TABLE selected_dated AS SELECT DISTINCT e.citing_id,e.cited_id,
     a.grant_date citing_date,b.grant_date cited_date
     FROM selected_raw e LEFT JOIN dates a ON e.citing_id=a.patent_id LEFT JOIN dates b ON e.cited_id=b.patent_id''')
    audit=rows(c,'''SELECT count(*) distinct_pairs,
     sum((citing_date IS NULL OR cited_date IS NULL)::INT) missing_endpoint_date,
     sum((citing_id=cited_id)::INT) self_edges,
     sum((cited_date>citing_date)::INT) future_backward_edges
     FROM selected_dated''')[0]
    c.execute('''CREATE TABLE edges AS SELECT * FROM selected_dated
     WHERE citing_id<>cited_id AND cited_date<=citing_date AND citing_date<=DATE '''+qp(END)+'''
     AND citing_date IS NOT NULL AND cited_date IS NOT NULL''')
    parquet(c,'SELECT * FROM edges ORDER BY citing_date,citing_id,cited_id','data/study_citation_edges.parquet')
    c.execute('DROP TABLE selected_raw');c.execute('DROP TABLE selected_dated')
    log('Build exact five-year outcomes and keep same-day citations out of forward outcomes.')
    c.execute('''CREATE TABLE fivey AS SELECT e.*,b.grant_year focal_grant_year,b.technical_subfield focal_subfield
     FROM edges e JOIN base b ON e.cited_id=CAST(b.patent_id AS BIGINT)
     WHERE b.is_focal_candidate AND b.mature_by_local_snapshot
     AND e.citing_date>b.grant_date AND e.citing_date<=b.label_window_end''')
    c.execute('''CREATE TABLE citing_ids AS SELECT DISTINCT citing_id pid FROM fivey
     UNION SELECT pid FROM ids''')
    c.execute('''CREATE TABLE cpc_sets AS SELECT p.patent_id,list(DISTINCT p.cpc_subclass ORDER BY p.cpc_subclass) subclass_set,
     bool_or(p.cpc_subclass='A61B') contains_a61b
     FROM '''+readp(NEW/'02_core/patent_cpc.parquet')+''' p SEMI JOIN citing_ids ON p.patent_id=citing_ids.pid
     WHERE p.cpc_subclass IS NOT NULL AND trim(p.cpc_subclass)<>'' GROUP BY p.patent_id''')
    parquet(c,'SELECT * FROM cpc_sets','data/citation_scope_cpc_sets.parquet')
    c.execute('''CREATE TABLE forward_domain AS SELECT f.*,
     a.subclass_set IS NULL OR b.subclass_set IS NULL classification_unknown,
     CASE WHEN a.subclass_set IS NOT NULL AND b.subclass_set IS NOT NULL
     THEN list_has_any(a.subclass_set,b.subclass_set) ELSE NULL END shares_any_cpc_subclass,
     a.contains_a61b citing_contains_a61b,
     a.subclass_set citing_cpc_subclasses,b.subclass_set focal_cpc_subclasses
     FROM fivey f LEFT JOIN cpc_sets a ON f.citing_id=a.patent_id LEFT JOIN cpc_sets b ON f.cited_id=b.patent_id''')
    parquet(c,'SELECT * EXCLUDE(citing_cpc_subclasses,focal_cpc_subclasses) FROM forward_domain ORDER BY citing_date,cited_id,citing_id','data/five_year_forward_edges.parquet')
    c.execute('''CREATE TABLE backward AS SELECT e.citing_id,count(*) backward_core_citation_count,
     avg(date_diff('day',e.cited_date,e.citing_date)/365.25) backward_citation_mean_age,
     sum((e.cited_date=e.citing_date)::INT) same_day_backward_count
     FROM edges e SEMI JOIN ids ON e.citing_id=ids.pid GROUP BY e.citing_id''')
    c.execute('''CREATE TABLE outcomes AS SELECT b.patent_id,
     CASE WHEN b.is_focal_candidate AND b.mature_by_local_snapshot THEN coalesce(f.n,0) ELSE NULL END forward_cites_5y,
     CASE WHEN b.is_focal_candidate AND b.mature_by_local_snapshot THEN coalesce(f.within_n,0) ELSE NULL END within_shared_subclass_cites_5y,
     CASE WHEN b.is_focal_candidate AND b.mature_by_local_snapshot THEN coalesce(f.newfield_n,0) ELSE NULL END new_field_cites_5y,
     CASE WHEN b.is_focal_candidate AND b.mature_by_local_snapshot THEN coalesce(f.unknown_n,0) ELSE NULL END cpc_unknown_cites_5y,
     CASE WHEN b.is_focal_candidate AND b.mature_by_local_snapshot THEN coalesce(f.legacy_out_n,0) ELSE NULL END outside_a61b_cites_5y,
     CASE WHEN b.is_focal_candidate AND b.mature_by_local_snapshot THEN coalesce(f.calendar_n,0) ELSE NULL END forward_cites_5_calendar_years,
     CASE WHEN b.is_focal_candidate AND b.mature_by_local_snapshot THEN coalesce(f.exact_through_2024,0) ELSE NULL END exact_window_cites_observed_through_2024,
     CASE WHEN b.is_focal_candidate AND b.mature_by_local_snapshot THEN coalesce(f.citing_2025,0) ELSE NULL END exact_window_citing_2025,
     b.label_window_end label_available_date,
     coalesce(k.backward_core_citation_count,0) backward_core_citation_count,
     k.backward_citation_mean_age,coalesce(k.same_day_backward_count,0) same_day_backward_count
     FROM base b LEFT JOIN (
     SELECT b0.patent_id,count(*) FILTER(WHERE e.citing_date>b0.grant_date AND e.citing_date<=b0.label_window_end) n,
     count(*) FILTER(WHERE e.citing_date>b0.grant_date AND e.citing_date<=b0.label_window_end AND d.shares_any_cpc_subclass) within_n,
     count(*) FILTER(WHERE e.citing_date>b0.grant_date AND e.citing_date<=b0.label_window_end AND NOT d.shares_any_cpc_subclass) newfield_n,
     count(*) FILTER(WHERE e.citing_date>b0.grant_date AND e.citing_date<=b0.label_window_end AND d.classification_unknown) unknown_n,
     count(*) FILTER(WHERE e.citing_date>b0.grant_date AND e.citing_date<=b0.label_window_end AND d.citing_contains_a61b=FALSE) legacy_out_n,
     count(*) FILTER(WHERE year(e.citing_date)>b0.grant_year AND year(e.citing_date)<=b0.grant_year+5) calendar_n,
     count(*) FILTER(WHERE e.citing_date>b0.grant_date AND e.citing_date<=b0.label_window_end AND e.citing_date<=DATE '2024-12-31') exact_through_2024,
     count(*) FILTER(WHERE e.citing_date>b0.grant_date AND e.citing_date<=b0.label_window_end AND year(e.citing_date)=2025) citing_2025
     FROM edges e JOIN base b0 ON e.cited_id=CAST(b0.patent_id AS BIGINT)
     LEFT JOIN forward_domain d USING(citing_id,cited_id,citing_date,cited_date)
     WHERE b0.is_focal_candidate AND b0.mature_by_local_snapshot GROUP BY b0.patent_id
     ) f USING(patent_id) LEFT JOIN backward k ON CAST(b.patent_id AS BIGINT)=k.citing_id''')
    c.execute('''CREATE TABLE labeled AS SELECT o.*,CASE WHEN forward_cites_5y>0 THEN
     new_field_cites_5y::DOUBLE/forward_cites_5y ELSE NULL END new_field_share_all_cites,
     CASE WHEN within_shared_subclass_cites_5y+new_field_cites_5y>0 THEN
     new_field_cites_5y::DOUBLE/(within_shared_subclass_cites_5y+new_field_cites_5y) ELSE NULL END new_field_share_known_cpc,
     ln(1+forward_cites_5y) ln_forward_cites_5y,
     CASE WHEN b.is_focal_candidate AND b.mature_by_local_snapshot THEN
     quantile_disc(forward_cites_5y,0.95) OVER(PARTITION BY b.grant_year,b.technical_subfield,b.is_focal_candidate) ELSE NULL END top5_count_cutoff,
     b.is_focal_candidate,b.grant_year,b.technical_subfield
     FROM outcomes o JOIN base b USING(patent_id)''')
    parquet(c,'''SELECT * EXCLUDE(is_focal_candidate,grant_year,technical_subfield),
     CASE WHEN forward_cites_5y IS NOT NULL THEN forward_cites_5y>=top5_count_cutoff ELSE NULL END top5_impact_with_ties
     FROM labeled ORDER BY patent_id''','data/citation_outcomes.parquet')
    assert c.execute('SELECT count(*) FROM labeled WHERE forward_cites_5y<>within_shared_subclass_cites_5y+new_field_cites_5y+cpc_unknown_cites_5y').fetchone()[0]==0
    csv(c,'''SELECT grant_year,count(*) patents,avg(forward_cites_5y) mean_cites,sum(forward_cites_5y) total_cites,
     sum(within_shared_subclass_cites_5y) within_cites,sum(new_field_cites_5y) newfield_cites,
     sum(cpc_unknown_cites_5y) unknown_cpc_cites,
     avg((forward_cites_5y=0)::INT) uncited_share,
     avg((forward_cites_5y>=top5_count_cutoff)::INT) top5_positive_share,
     sum(exact_window_citing_2025) terminal_2025_cites
     FROM labeled WHERE is_focal_candidate GROUP BY grant_year ORDER BY grant_year''','quality/03_citation_outcomes_by_year.csv')
    csv(c,'''SELECT grant_year,technical_subfield,count(*) n,min(top5_count_cutoff) cutoff,
     sum((forward_cites_5y>=top5_count_cutoff)::INT) positive_n
     FROM labeled WHERE is_focal_candidate GROUP BY ALL ORDER BY grant_year,technical_subfield''','quality/03_top5_cohorts_and_ties.csv')
    summary={'raw_touching_base_edges':raw_n,'pair_and_date_audit':audit,
     'valid_touching_base_edges':c.execute('SELECT count(*) FROM edges').fetchone()[0],
     'exact_five_year_edges':c.execute('SELECT count(*) FROM fivey').fetchone()[0],
     'domain_unknown_edges':c.execute('SELECT count(*) FROM forward_domain WHERE classification_unknown').fetchone()[0],
     'same_day_forward_excluded':c.execute('SELECT count(*) FROM edges e JOIN base b ON e.cited_id=CAST(b.patent_id AS BIGINT) WHERE b.is_focal_candidate AND e.citing_date=b.grant_date').fetchone()[0],
     'domain_definition':'primary: no shared current CPC subclass with focal entire subclass set; old no-A61B outcome separately retained',
     'top5_definition':'within grant year x mapped A61B subfield, empirical quantile_disc(0.95), >= cutoff with all ties retained; fixed screening budget belongs to stage4',
     'label_zero_policy':'zero only for mature complete observed core citation scope; histories have no focal outcome label',
     'seconds':round(time.time()-t,2)}
    finish('03_citation_outcomes',summary)

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
