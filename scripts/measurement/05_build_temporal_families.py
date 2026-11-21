"""Date-aware domestic granted-family reconstruction, with/without CIP."""
from common import *
import argparse
import numpy as np
import pyarrow as pa

def temporal_tree(ids, left, right, event_days):
    n=len(ids);parent=np.arange(n,dtype=np.int32);sz=np.ones(n,dtype=np.int32)
    top=np.arange(n,dtype=np.int32)
    tree_parent=np.full(2*n,-1,dtype=np.int32)
    tree_day=np.full(2*n,np.iinfo(np.int32).min,dtype=np.int32)
    tree_size=np.ones(2*n,dtype=np.int32)
    tree_min=np.zeros(2*n,dtype=np.int64);tree_min[:n]=ids
    count=n
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return int(x)
    for a,b,d in zip(left,right,event_days):
        a=find(int(a));b=find(int(b))
        if a==b:continue
        if sz[a]<sz[b]:a,b=b,a
        ta=int(top[a]);tb=int(top[b]);node=count;count+=1
        tree_parent[ta]=node;tree_parent[tb]=node
        tree_day[node]=d;tree_size[node]=int(sz[a])+int(sz[b])
        tree_min[node]=min(tree_min[ta],tree_min[tb])
        parent[b]=a;sz[a]+=sz[b];top[a]=node
    ancestor=[tree_parent[:count].copy()]
    while np.any(ancestor[-1]>=0):
        a=ancestor[-1];nxt=np.full(count,-1,dtype=np.int32)
        ok=a>=0;nxt[ok]=a[a[ok]]
        if not np.any(nxt>=0):break
        ancestor.append(nxt)
    return {'ids':ids,'ancestor':np.stack(ancestor),'day':tree_day[:count],
            'size':tree_size[:count],'min_id':tree_min[:count]}

def component_at(tree,patent_ids,days):
    pid=np.asarray(patent_ids,dtype=np.int64);days=np.asarray(days,dtype=np.int32)
    ids=tree['ids'];idx=np.searchsorted(ids,pid)
    valid=idx<len(ids);valid[valid]&=(ids[idx[valid]]==pid[valid])
    node=np.zeros(len(idx),dtype=np.int32);node[valid]=idx[valid].astype(np.int32)
    for up in tree['ancestor'][::-1]:
        p=up[node];ok=valid & (p>=0)
        select=np.flatnonzero(ok)
        select=select[tree['day'][p[select]]<=days[select]]
        node[select]=p[select]
    node[~valid]=-1
    return node

def test_temporal_tree():
    tr=temporal_tree(np.array([10,20,30]),np.array([0,1]),np.array([1,2]),np.array([200,500]))
    n=component_at(tr,[10,10,10],[100,300,600])
    assert list(tr['size'][n])==[1,2,3]
    a=component_at(tr,[10,10],[499,500]);b=component_at(tr,[30,30],[499,500])
    assert a[0]!=b[0] and a[1]==b[1]
    assert component_at(tr,[999],[900])[0]==-1

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if NEW is None:
        raise ValueError('Configure inputs.patentsview_root before preparing measurement inputs')
    test_temporal_tree();t=time.time();c=connect()
    paths=[NEW/'01_stage_parquet/us_rel_doc.parquet',NEW/'02_core/patent_application.parquet',NEW/'02_core/patent_dates.parquet']
    source_receipt(paths,'checks/05_sources_before.json')
    c.execute('CREATE VIEW dates AS SELECT patent_id,CAST(patent_date AS DATE) grant_date FROM '+readp(paths[2]))
    c.execute('''CREATE TABLE apps AS SELECT patent_id,application_id,try_cast(regexp_replace(application_id,'[^0-9]','','g') AS BIGINT) app_number
      FROM '''+readp(paths[1])+''' WHERE regexp_full_match(application_id,'[0-9 /,.-]+')''')
    c.execute('''CREATE TABLE app_map AS SELECT app_number,min(patent_id) patent_id,count(DISTINCT patent_id) mapped_patents
      FROM apps WHERE app_number IS NOT NULL GROUP BY app_number''')
    c.execute('''CREATE TABLE relations AS SELECT r.patent_id source_id,r.related_doc_type,r.related_doc_kind,r.related_doc_number,
      r.published_country,try_cast(regexp_replace(related_doc_number,'[^0-9]','','g') AS BIGINT) number,
      regexp_full_match(trim(related_doc_number),'[0-9 /,.-]+') numeric_document
      FROM '''+readp(paths[0])+''' r
      WHERE related_doc_type IN ('continuation','division','continuation-in-part')
      AND related_doc_kind IN ('parent-doc','parent-grant-document')''')
    c.execute('''CREATE TABLE resolved AS SELECT r.*,CASE
      WHEN numeric_document AND coalesce(published_country,'US')='US' AND related_doc_kind='parent-grant-document' THEN d.patent_id
      WHEN numeric_document AND coalesce(published_country,'US')='US' AND related_doc_kind='parent-doc' AND a.mapped_patents=1 THEN a.patent_id
      ELSE NULL END target_id,
      CASE WHEN related_doc_kind='parent-doc' THEN a.mapped_patents ELSE NULL END application_mapping_n
      FROM relations r LEFT JOIN app_map a ON r.number=a.app_number
      LEFT JOIN dates d ON r.number=d.patent_id''')
    audit=rows(c,'''SELECT related_doc_type,related_doc_kind,count(*) raw_rows,
      count(target_id) resolved_rows,sum((target_id=source_id)::INT) self_rows,
      sum((application_mapping_n>1)::INT) ambiguous_app_rows
      FROM resolved GROUP BY ALL ORDER BY ALL''')
    dump('quality/05_relation_resolution.json',audit)
    c.execute('''CREATE TABLE edges AS SELECT least(r.source_id,r.target_id) left_id,greatest(r.source_id,r.target_id) right_id,
      r.related_doc_type,min(greatest(a.grant_date,b.grant_date)) edge_available_date
      FROM resolved r JOIN dates a ON r.source_id=a.patent_id JOIN dates b ON r.target_id=b.patent_id
      WHERE r.target_id IS NOT NULL AND r.target_id<>r.source_id
      GROUP BY left_id,right_id,r.related_doc_type''')
    parquet(c,'SELECT * FROM edges ORDER BY edge_available_date,left_id,right_id','data/domestic_continuity_resolved_edges.parquet')
    # Only disclosed parent relationships identify whether the focal grant continues/divides an earlier application.
    c.execute('''CREATE TABLE direct_flags AS SELECT CAST(source_id AS VARCHAR) patent_id,
      bool_or(related_doc_type='continuation') is_continuation,
      bool_or(related_doc_type='division') is_division,
      bool_or(related_doc_type='continuation-in-part') is_cip,
      count(*) parent_relation_rows,count(target_id) resolved_parent_rows
      FROM resolved GROUP BY source_id''')
    parquet(c,'SELECT * FROM direct_flags','data/domestic_application_type_flags.parquet')
    base=pq.read_table(out('data/patent_panel_base.parquet'),columns=['patent_id','grant_date'])
    bp=np.array(base.column('patent_id').to_pylist(),dtype=np.int64)
    bd=np.array(base.column('grant_date').to_pylist(),dtype='datetime64[D]').astype(np.int32)
    fwd=pq.read_table(out('data/five_year_forward_edges.parquet'),columns=['citing_id','cited_id','citing_date'])
    cite=fwd.column('citing_id').to_numpy();cited=fwd.column('cited_id').to_numpy()
    fd=np.array(fwd.column('citing_date').to_pylist(),dtype='datetime64[D]').astype(np.int32)
    summaries={};base_result={'patent_id':[str(x) for x in bp]};edge_result={'citing_id':cite,'cited_id':cited}
    for label,where in [('no_cip',"related_doc_type IN ('continuation','division')"),('with_cip','true')]:
        log('Building temporal domestic-grant family '+label)
        e=c.execute('''SELECT left_id,right_id,min(edge_available_date) event_date FROM edges WHERE '''+where+''' GROUP BY left_id,right_id ORDER BY event_date,left_id,right_id''').to_arrow_table()
        left=e.column('left_id').to_numpy();right=e.column('right_id').to_numpy()
        days=np.array(e.column('event_date').to_pylist(),dtype='datetime64[D]').astype(np.int32)
        ids=np.unique(np.concatenate([bp,left,right]))
        tree=temporal_tree(ids,np.searchsorted(ids,left),np.searchsorted(ids,right),days)
        np.savez_compressed(out('cache/domestic_family_tree_'+label+'.npz'),**tree)
        nodes=component_at(tree,bp,bd)
        base_result['family_size_at_grant_'+label]=tree['size'][nodes]
        base_result['family_key_at_grant_'+label]=[label+':'+str(x) for x in tree['min_id'][nodes]]
        a=component_at(tree,cite,fd);b=component_at(tree,cited,fd)
        same=(a>=0)&(b>=0)&(a==b)
        edge_result['same_family_at_citing_date_'+label]=same
        summaries[label]={'unique_temporal_edges':len(left),'nodes':len(ids),'merge_nodes':len(tree['day'])-len(ids),
          'base_family_size_gt1':int(np.sum(tree['size'][nodes]>1)), 'max_base_size_at_grant':int(np.max(tree['size'][nodes])),
          'fivey_samefamily_edges':int(same.sum())}
        del tree
    c.register('family_base',pa.Table.from_pydict(base_result))
    parquet(c,'''SELECT f.*,coalesce(d.is_continuation,FALSE) is_continuation,
      coalesce(d.is_division,FALSE) is_division,coalesce(d.is_cip,FALSE) is_cip,
      coalesce(d.parent_relation_rows,0) parent_relation_rows,coalesce(d.resolved_parent_rows,0) resolved_parent_rows
      FROM family_base f LEFT JOIN direct_flags d USING(patent_id)''','data/domestic_family_controls.parquet')
    pq.write_table(pa.Table.from_pydict(edge_result),out('data/five_year_family_citation_flags.parquet'),compression='zstd')
    finish('05_domestic_families',{'networks':summaries,'resolution':audit,
      'date_rule':'edge at max(grant dates of source and resolved target); historical component queried at focal/citing grant; future merges excluded',
      'scope':'resolved US granted utility continuity; non-granted, unresolvable, international and related-publication documents excluded from graph and explicitly counted',
      'primary_network':'continuation/division without CIP; expanded network with CIP also retained',
      'application_flags':'parent-doc or parent-grant-document relation, including unresolved parent references; child-doc not interpreted as a separate descendant',
      'test':'synthetic temporal-merge assertions passed','seconds':round(time.time()-t,2)})

if __name__=='__main__':main()
