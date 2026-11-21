"""Exact all-future-text reuse, CPU/CUDA; no model API and block-level resume."""
from pathlib import Path
from datetime import date
import argparse,json,hashlib,time,os,sys
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
from temporal_metrics import unit_rows,day
REPOSITORY=Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:sys.path.insert(0,str(REPOSITORY))
from project_config import work_dir

ROOT=work_dir('measurement')

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def block_stats(q,qd,qe,qids,qbits,qknown,h,hd,hids,hbits,hknown,thresholds,device='cpu'):
    """q/h already normalized by the common CPU normalization for both engines."""
    if device=='cpu':
        sim=np.clip(q@h.T,-1,1)
        window=(hd[None,:]>qd[:,None]) & (hd[None,:]<=qe[:,None])
        window &= hids[None,:]!=qids[:,None]
        hvalid=np.isfinite(h).all(axis=1)&(np.linalg.norm(h,axis=1)>0)
        qvalid=np.isfinite(q).all(axis=1)&(np.linalg.norm(q,axis=1)>0)
        rr,cc=np.where(window & hvalid[None,:] & qvalid[:,None] & (sim>=min(thresholds)))
        counts=np.zeros((len(q),len(thresholds),4),np.int64)
        excess=np.zeros((len(q),len(thresholds)),np.float64)
        if len(rr):
            known=qknown[rr]&hknown[cc]
            shared=np.any(np.bitwise_and(qbits[rr],hbits[cc])!=0,axis=1)
            domains=np.where(~known,3,np.where(shared,1,2))
            scores=sim[rr,cc]
            for ti,tau in enumerate(thresholds):
                hit=scores>=tau;r=rr[hit];d=domains[hit]
                bins=np.bincount(r*4+d,minlength=len(q)*4).reshape(len(q),4)
                bins[:,0]=bins[:,1:].sum(axis=1);counts[:,ti]=bins
                values=np.maximum((scores[hit].astype(np.float64)-tau)/(1-tau),0)
                np.add.at(excess[:,ti],r,values)
        return counts,excess,window.sum(axis=1),(window&hvalid[None,:]).sum(axis=1)
    import torch
    conv=lambda x:torch.as_tensor(x,device=device)
    qs=conv(q);hs=conv(h);sim=(qs@hs.T).clamp(-1,1)
    window=(conv(hd)[None,:]>conv(qd)[:,None]) & (conv(hd)[None,:]<=conv(qe)[:,None])
    window &= conv(hids)[None,:]!=conv(qids)[:,None]
    hvalid=conv(np.isfinite(h).all(axis=1)&(np.linalg.norm(h,axis=1)>0))
    qvalid=conv(np.isfinite(q).all(axis=1)&(np.linalg.norm(q,axis=1)>0))
    usable=window & hvalid[None,:] & qvalid[:,None]
    # Only classify pairs that can hit any of the fixed thresholds.
    rr,cc=torch.where(usable & (sim>=min(thresholds)))
    counts=torch.zeros((len(q),len(thresholds),4),dtype=torch.int64,device=device)
    excess=torch.zeros((len(q),len(thresholds)),dtype=torch.float64,device=device)
    if len(rr):
        known=conv(qknown)[rr] & conv(hknown)[cc]
        shared=torch.any(torch.bitwise_and(conv(qbits)[rr],conv(hbits)[cc])!=0,dim=1)
        domains=torch.where(~known,3,torch.where(shared,1,2))
        scores=sim[rr,cc]
        for ti,tau in enumerate(thresholds):
            hit=scores>=tau;r=rr[hit];d=domains[hit]
            bins=torch.bincount(r*4+d,minlength=len(q)*4).reshape(len(q),4)
            bins[:,0]=bins[:,1:].sum(dim=1);counts[:,ti]=bins
            values=((scores[hit].double()-tau)/(1-tau)).clamp(min=0)
            excess[:,ti].index_add_(0,r,values)
    return (counts.cpu().numpy(),excess.cpu().numpy(),window.sum(dim=1).cpu().numpy(),
        (window & hvalid[None,:]).sum(dim=1).cpu().numpy())

def self_test(device):
    from temporal_metrics import future_text_uptake
    rng=np.random.default_rng(417);q,_=unit_rows(rng.normal(size=(5,6)));h,_=unit_rows(rng.normal(size=(9,6)))
    h[0]=q[0];h[1]=q[0];h[8]=0
    qm=[{'patent_id':str(i+1),'grant_date':'2020-02-29','cpc_subclasses':x} for i,x in enumerate([['A'],['B'],[],['A','B'],['C']])]
    hm=[{'patent_id':str(i+20),'grant_date':d,'cpc_subclasses':x} for i,(d,x) in enumerate([
        ('2021-01-01',['A']),('2025-02-28',['B']),('2020-02-29',['A']),('2025-03-01',['A']),
        ('2022-01-01',[]),('2023-01-01',['B']),('2024-01-01',['C']),('2024-02-01',['A','B']),('2021-01-01',['A'])])]
    masks=lambda mm:np.array([[sum(1<<{'A':0,'B':1,'C':2}[x] for x in r['cpc_subclasses'])] for r in mm],np.int64)
    thresholds=(.2,.7,.9)
    actual=block_stats(q,np.array([day(r['grant_date']) for r in qm]),np.full(5,day('2025-02-28')),
        np.arange(1,6),masks(qm),np.array([bool(r['cpc_subclasses']) for r in qm]),h,
        np.array([day(r['grant_date']) for r in hm]),np.arange(20,29),masks(hm),
        np.array([bool(r['cpc_subclasses']) for r in hm]),thresholds,device)
    reference=future_text_uptake(q,qm,[(h,hm)],'2025-12-30',thresholds)
    assert np.array_equal(actual[0],reference['counts'])
    assert np.allclose(actual[1],reference['excess_similarity_sum'],atol=1e-6)
    assert np.array_equal(actual[2],reference['window_candidate_n'])
    assert np.array_equal(actual[3],reference['valid_vector_candidate_n'])
    high_bit=block_stats(q[:1],np.array([day('2020-01-01')]),np.array([day('2025-01-01')]),
        np.array([1]),np.array([[(1<<62)+1]],dtype=np.int64),np.array([True]),q[:1],
        np.array([day('2021-01-01')]),np.array([2]),np.array([[1]],dtype=np.int64),
        np.array([True]),(.8,),device)
    assert high_bit[0][0,0,1]==1,'Low subclass bit must survive alongside bit62'
    return {'passed':True,'device':device,'CUDA_tested':device=='cuda','synthetic_not_study_results':True,
        'tests':['exact anniversary and same-day exclusion','three CPC partitions sum to total',
        'invalid vector denominator','three thresholds and excess sum vs independent CPU reference',
        'exact CPC bit62 and low-bit coexistence'],'api_calls':0}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,default=ROOT/'semantic_reuse/all_future_text')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--self-test',action='store_true')
    p.add_argument('--query-rows',type=int,default=512);p.add_argument('--candidate-rows',type=int,default=8192)
    a=p.parse_args()
    if a.query_rows<=0 or a.candidate_rows<=0:p.error('Block sizes must be positive')
    if a.device=='cuda':
        import torch
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        assert torch.cuda.is_available(),'No silent CPU fallback'
    if a.self_test:
        result=self_test(a.device);print(json.dumps(result))
        return
    assert a.job is not None
    import pyarrow.parquet as pq
    import pyarrow as pa
    job=a.job.resolve();assert job.is_relative_to(ROOT)
    cfg=load(job/'reuse_definition.json');assert cfg['scope']=='ALL_ELIGIBLE_FUTURE_TEXTS_NOT_CITATION_ONLY'
    assert load(job/'vectors_pack_status.json')['status']=='COMPLETE'
    qmeta=pq.read_table(job/'focal_metadata.parquet');hmeta=pq.read_table(job/'candidate_metadata.parquet')
    assert all(pa.types.is_int64(tab['cpc_bits_'+str(i)].type) for tab in [qmeta,hmeta] for i in range(cfg['cpc_words'])), 'CPC bitsets must be exact int64; rebuild legacy float metadata before computation'
    q=np.load(job/'focal_vectors.npy',mmap_mode='r');h=np.load(job/'candidate_vectors.npy',mmap_mode='r')
    assert len(q)==len(qmeta)==cfg['focal_n'] and len(h)==len(hmeta)==cfg['candidate_n']
    q,qvalid=unit_rows(q);thresholds=cfg['thresholds'];words=cfg['cpc_words']
    assert .8 in thresholds and all(0<t<1 for t in thresholds)
    col=lambda tab,name:tab[name].to_numpy()
    qd=np.array([day(x) for x in col(qmeta,'grant_date')]);qe=np.array([day(x) for x in col(qmeta,'label_window_end')])
    hd=np.array([day(x) for x in col(hmeta,'grant_date')]);qids=col(qmeta,'patent_id').astype(np.int64);hids=col(hmeta,'patent_id').astype(np.int64)
    qb=np.column_stack([col(qmeta,'cpc_bits_'+str(i)) for i in range(words)]).astype(np.int64)
    hb=np.column_stack([col(hmeta,'cpc_bits_'+str(i)) for i in range(words)]).astype(np.int64)
    qk=col(qmeta,'cpc_known');hk=col(hmeta,'cpc_known');mature=qe<=day(cfg['snapshot_end'])
    assert np.all(hd[1:]>=hd[:-1]),'Candidate metadata must be sorted by grant date'
    qvalid &= np.asarray(col(qmeta,'emb_valid'),dtype=bool)
    q[~qvalid]=0
    declared_hvalid=np.asarray(col(hmeta,'emb_valid'),dtype=bool)
    fingerprint={name:digest(job/name) for name in ['reuse_definition.json','focal_metadata.parquet','candidate_metadata.parquet']}
    fingerprint['runner_sha256']=digest(Path(__file__))
    fingerprint['kernel_sha256']=digest(Path(__file__).with_name('temporal_metrics.py'))
    for name in ['focal_vectors.npy','candidate_vectors.npy']:
        st=(job/name).stat();fingerprint[name]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    fp=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest();n=len(q)
    counts=np.zeros((n,len(thresholds),4),np.int64);excess=np.zeros((n,len(thresholds)),np.float64)
    window=np.zeros(n,np.int64);valid=np.zeros(n,np.int64);start=0
    checkpoint=job/'reuse_checkpoint.npz';statepath=job/'reuse_status.json'
    if statepath.exists():
        state=load(statepath);assert state['input_fingerprint']==fp
        if state['status']=='COMPLETE':print(json.dumps(state));return
        old=np.load(checkpoint);assert str(old['input_fingerprint'])==fp
        start=int(old['candidate_end']);assert start>=state['candidate_end']
        counts=old['counts'];excess=old['excess'];window=old['window'];valid=old['valid']
    state={'status':'IN_PROGRESS','candidate_end':start,'candidate_total':len(h),
           'input_fingerprint':fp,'device':a.device,'seconds_this_run':0,'api_calls':0}
    started=time.monotonic()
    for hs in range(start,len(h),a.candidate_rows):
        he=min(hs+a.candidate_rows,len(h));hh,_=unit_rows(h[hs:he])
        hh[~declared_hvalid[hs:he]]=0
        active=np.flatnonzero((qd<hd[he-1])&(qe>=hd[hs]))
        for qs in range(0,len(active),a.query_rows):
            qi=active[qs:qs+a.query_rows]
            c,e,w,v=block_stats(q[qi],qd[qi],qe[qi],qids[qi],qb[qi],qk[qi],hh,hd[hs:he],
                hids[hs:he],hb[hs:he],hk[hs:he],thresholds,a.device)
            counts[qi]+=c;excess[qi]+=e;window[qi]+=w;valid[qi]+=v
        if (hs//a.candidate_rows)%16==15 or he==len(h):
            tmp=checkpoint.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,counts=counts,excess=excess,window=window,valid=valid,candidate_end=he,input_fingerprint=fp)
            tmp.replace(checkpoint)
            state={'status':'IN_PROGRESS','candidate_end':he,'candidate_total':len(h),'input_fingerprint':fp,
                'device':a.device,'seconds_this_run':round(time.monotonic()-started,2),'api_calls':0}
            write(statepath,state);print(json.dumps(state),flush=True)
    assert np.array_equal(counts[:,:,0],counts[:,:,1:].sum(axis=2))
    records=[]
    for i,pid in enumerate(qids):
        ok=bool(mature[i] and qvalid[i]);r={'patent_id':str(pid),'reuse_window_candidates':int(window[i]),
            'reuse_valid_vector_candidates':int(valid[i]),'reuse_window_mature':bool(mature[i]),'reuse_query_valid':bool(qvalid[i])}
        for ti,tau in enumerate(thresholds):
            label=str(int(round(tau*100)))
            for di,domain in enumerate(['all','shared','newfield','cpc_unknown']):
                r[f'reuse_count_{label}_{domain}']=int(counts[i,ti,di]) if ok else None
            r['reuse_excess_'+label]=float(excess[i,ti]) if ok else None
        r['semantic_reuse_5y']=float(np.log1p(counts[i,thresholds.index(.8),0])) if ok else None
        records.append(r)
    pq.write_table(pa.Table.from_pylist(records),job/'semantic_reuse_results.parquet',compression='zstd')
    state.update(status='COMPLETE',focal_n=n,main_label_usable_n=int((mature&qvalid).sum()),
        primary_definition='log1p(all future text count cosine>=0.8), exact5y, not citation-restricted')
    write(statepath,state);print(json.dumps(state),flush=True)

if __name__=='__main__':main()
