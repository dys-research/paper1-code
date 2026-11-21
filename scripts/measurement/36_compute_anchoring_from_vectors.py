"""Compute observed M-M anchoring, retaining the complete historical denominator."""
from common import *
import argparse
import numpy as np,pyarrow as pa

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    started=time.time();job=out('encoding/full_frozen');state=json.loads((job/'status.json').read_text(encoding='utf-8'))
    assert state['status']=='COMPLETE';cfg=json.loads((job/'encoder_freeze.json').read_text(encoding='utf-8'))
    assert state['input_sha256']==cfg['inputs_sha256']
    vectors=np.load(job/'vectors.npy',mmap_mode='r')
    cov=pq.read_table(out('data/anchoring_coverage_after_bulk.parquet')).to_pylist()
    order={r['patent_id']:i for i,r in enumerate(cov)}
    n=len(cov);counts=np.zeros(n,np.int64);sums=np.zeros(n,np.float64);maximum=np.full(n,-np.inf,np.float64)
    pairs=out('data/anchoring_valid_pairs_for_encoding.parquet')
    for batch in pq.ParquetFile(pairs).iter_batches(batch_size=32768):
        d=batch.to_pydict();qi=np.array([order[x] for x in d['patent_id']]);q=np.asarray(vectors[d['query_embedding_row']]);h=np.asarray(vectors[d['cited_embedding_row']])
        assert np.isfinite(q).all() and np.isfinite(h).all()
        sim=np.clip(np.sum(q*h,axis=1,dtype=np.float64),-1,1)
        counts+=np.bincount(qi,minlength=n);sums+=np.bincount(qi,weights=sim,minlength=n);np.maximum.at(maximum,qi,sim)
    for i,r in enumerate(cov):
        assert counts[i]==(r['covered_M_n'] if r['query_M_valid'] else 0)
        r.update(anchoring_raw=float(sums[i]/counts[i]) if counts[i] else None,
            anchoring_max=float(maximum[i]) if counts[i] else None,anchoring_observed_pair_n=int(counts[i]))
    result=out('data/anchoring_results.parquet');pq.write_table(pa.Table.from_pylist(cov),result,compression='zstd')
    finish('36_anchoring_results',{'focal_n':n,'computable_n':int((counts>0).sum()),'valid_pairs':int(counts.sum()),
        'new_model_calls':0,'seconds':round(time.time()-started,2),'definition':'M cosine among observed valid historical references; all eligible references retained in denominator'})

if __name__=='__main__':main()
