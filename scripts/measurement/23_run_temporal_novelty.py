"""Run exact blocked novelty on a frozen encoder job; with an explicit historical comparison window."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
from common import *
from temporal_metrics import temporal_topk
import argparse,numpy as np
import pyarrow as pa

class IndexedVectors:
    def __init__(self,array,indices):self.array=array;self.indices=np.asarray(indices,dtype=np.int64);self.shape=(len(self.indices),array.shape[1])
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):return self.array[self.indices[index]]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,default=out('encoding/full_frozen'))
    p.add_argument('--allow-pilot',action='store_true');p.add_argument('--engine',choices=['numpy','torch'],default='numpy')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--k',type=int,default=10)
    p.add_argument('--history-years',type=int,default=10);p.add_argument('--kind',action='append')
    p.add_argument('--cache-k',type=int,default=20)
    p.add_argument('--resume',action='store_true',help='Reuse only completed per-kind outputs with identical frozen inputs and algorithm')
    a=p.parse_args();job=a.job.resolve();assert job.is_relative_to(ROOT)
    cfg=json.loads((job/'encoder_freeze.json').read_text(encoding='utf-8'))
    scope=cfg['scope'];pilot=scope.startswith('PILOT')
    assert not pilot or a.allow_pilot,'Refusing pilot data as a main result'
    status=json.loads((job/'status.json').read_text(encoding='utf-8'));assert status['status']=='COMPLETE'
    assert status['input_sha256']==cfg['inputs_sha256']
    if 'task_metadata_parquet_sha256' in cfg:assert sha(job/'task_metadata.parquet')==cfg['task_metadata_parquet_sha256']
    vectors=np.load(job/'vectors.npy',mmap_mode='r');c=connect()
    c.execute('CREATE VIEW tasks AS SELECT * FROM '+readp(job/'task_metadata.parquet'))
    kinds=a.kind or [r['kind'] for r in rows(c,'SELECT DISTINCT kind FROM tasks ORDER BY kind')]
    result=[];allmetrics=[]
    for kind in kinds:
        folder=job/'novelty'/kind;folder.mkdir(parents=True,exist_ok=True)
        signature={'inputs_sha256':cfg['inputs_sha256'],'metadata_sha256':sha(job/'task_metadata.parquet'),
          'algorithm_sha256':sha(SCRIPT_DIR/'temporal_metrics.py'),'runner_sha256':sha(Path(__file__)),
          'k':a.k,'cache_k':max(a.k,a.cache_k),'history_years':a.history_years,'kind':kind}
        receipt=folder/'completed.json'
        if a.resume and receipt.exists():
            saved=json.loads(receipt.read_text(encoding='utf-8'))
            assert saved['signature']==signature,'Frozen inputs or algorithm changed; do not silently reuse'
            assert (folder/'nearest_neighbors.npz').exists() and (folder/'history_row_mapping.parquet').exists()
            records=pq.read_table(folder/'metrics.parquet').to_pylist()
            allmetrics.extend(records);result.append(saved['result']);continue
        meta=rows(c,"SELECT * FROM tasks WHERE kind="+qp(kind)+' ORDER BY patent_id')
        assert len({r['patent_id'] for r in meta})==len(meta)
        focal=[r for r in meta if r['is_focal_candidate']]
        hv=IndexedVectors(vectors,[r['embedding_row'] for r in meta]);qv=IndexedVectors(vectors,[r['embedding_row'] for r in focal])
        t=time.time();r=temporal_topk(qv,focal,hv,meta,k=max(a.k,a.cache_k),history_years=a.history_years,engine=a.engine,device=a.device)
        selected=r['neighbor_cosines'][:,:a.k];finite=np.isfinite(selected);actual_k=finite.sum(axis=1)
        mean=np.divide(np.where(finite,selected,0).sum(axis=1,dtype=np.float64),actual_k,
          out=np.full(len(focal),np.nan),where=actual_k>0)
        auxiliary=1-mean;primary=np.where(actual_k==a.k,auxiliary,np.nan)
        np.savez_compressed(folder/'nearest_neighbors.npz',history_rows=r['neighbor_rows'],cosines=r['neighbor_cosines'])
        hmap=pa.Table.from_pylist([{'history_row':i,'patent_id':x['patent_id'],'embedding_row':x['embedding_row'],'grant_date':str(x['grant_date'])} for i,x in enumerate(meta)])
        pq.write_table(hmap,folder/'history_row_mapping.parquet',compression='zstd')
        records=[]
        for i,x in enumerate(focal):
            record={'patent_id':x['patent_id'],'kind':kind,'historical_count':int(r['historical_count'][i]),
              'actual_k':int(actual_k[i]),'k':a.k,'history_years':a.history_years,
              'query_valid':bool(r['query_valid'][i]),'novelty_raw':float(primary[i]) if np.isfinite(primary[i]) else None,
              'novelty_min_k_auxiliary':float(auxiliary[i]) if np.isfinite(auxiliary[i]) else None,
              'scope':scope}
            records.append(record);allmetrics.append(record)
        pq.write_table(pa.Table.from_pylist(records),folder/'metrics.parquet',compression='zstd')
        kind_result={'kind':kind,'query_n':len(focal),'history_input_n':len(meta),
          'full_k_available_n':sum(x['novelty_raw'] is not None for x in records),
          'seconds':round(time.time()-t,2),'scope':scope}
        result.append(kind_result)
        receipt.write_text(json.dumps({'signature':signature,'result':kind_result},ensure_ascii=False,indent=2),encoding='utf-8')
    pq.write_table(pa.Table.from_pylist(allmetrics),job/'novelty/metrics_long.parquet',compression='zstd')
    summary={'results':result,'engine':a.engine,'device':a.device,'pilot_not_manuscript_result':pilot,
      'neighbor_cache_k':max(a.k,a.cache_k),'primary_k':a.k,
      'definition':'1-mean(topK cosine), ten-calendar-year same-subfield history, exclude same-day/future; insufficientK remains missing',
      'residuals':'not fitted here; fit_joint_residual must be called only inside the corresponding model fitting partition',
      'new_api_calls':0}
    (job/'novelty/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
