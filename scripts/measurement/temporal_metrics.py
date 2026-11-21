"""Exact, blocked patent text metrics. No API access, no source-file writes."""
from datetime import date
import numpy as np

def as_date(value):
    return value if isinstance(value,date) and not hasattr(value,'hour') else date.fromisoformat(str(value)[:10])

def add_years(value,years):
    value=as_date(value)
    try:return value.replace(year=value.year+years)
    except ValueError:return value.replace(year=value.year+years,day=28)

def day(value):return as_date(value).toordinal()

def unit_rows(vectors):
    a=np.asarray(vectors,dtype=np.float32)
    finite=np.isfinite(a).all(axis=1)
    norms=np.linalg.norm(np.where(np.isfinite(a),a,0),axis=1)
    valid=finite & (norms>0)
    safe=np.zeros_like(a);safe[valid]=a[valid]/norms[valid,None]
    return safe,valid

def temporal_topk(qvectors,qmeta,hvectors,hmeta,k=10,history_years=10,
                  query_rows=256,history_rows=4096,engine='numpy',device='cpu'):
    """Same-subfield [grant-10 calendar years,grant) exact cosine neighbors.

    Main novelty requires k valid historical items. The min(k,n) alternative is
    separately returned, never zero-filled. Ties use lexicographic patent IDs.
    Histories are read in blocks; only study query vectors of one field reside
    together on the GPU. Neighbor row indices refer to the input hmeta order.
    """
    assert len(qvectors)==len(qmeta) and len(hvectors)==len(hmeta) and k>0
    assert qvectors.shape[1]==hvectors.shape[1] and query_rows>0 and history_rows>0
    qids=[str(r['patent_id']) for r in qmeta];hids=[str(r['patent_id']) for r in hmeta]
    assert len(set(qids))==len(qids) and len(set(hids))==len(hids)
    qlookup={pid:i for i,pid in enumerate(qids)}
    qdate=np.array([day(r['grant_date']) for r in qmeta],dtype=np.int64)
    qlow=np.array([day(add_years(r['grant_date'],-history_years)) for r in qmeta],dtype=np.int64)
    hdate=np.array([day(r['grant_date']) for r in hmeta],dtype=np.int64)
    qfield=[r.get('technical_subfield') for r in qmeta];hfield=[r.get('technical_subfield') for r in hmeta]
    scores=np.full((len(qmeta),k),-np.inf,np.float32);neighbors=np.full((len(qmeta),k),-1,np.int64)
    counts=np.zeros(len(qmeta),np.int64);qvalid=np.zeros(len(qmeta),bool)
    torch=None
    if engine=='torch':
        import torch
        if device=='cuda':
            assert torch.cuda.is_available()
            torch.backends.cuda.matmul.allow_tf32=False
        def tensor(a):return torch.as_tensor(a,device=device)
    else:assert engine=='numpy'
    for field in sorted({x for x in qfield if x is not None}):
        qi=np.array([i for i,f in enumerate(qfield) if f==field],dtype=np.int64)
        hi=np.array(sorted((i for i,f in enumerate(hfield) if f==field),key=lambda i:hids[i]),dtype=np.int64)
        q,valid=unit_rows(qvectors[qi]);valid &= np.array([qmeta[i].get('valid_embedding',True) for i in qi])
        qvalid[qi]=valid
        if not len(hi):continue
        if torch is not None:
            q=tensor(q);s=tensor(scores[qi]);idx=tensor(neighbors[qi]);ct=tensor(counts[qi])
            qd=tensor(qdate[qi]);ql=tensor(qlow[qi]);qok=tensor(valid);qix=tensor(qi)
        else:
            s=scores[qi].copy();idx=neighbors[qi].copy();ct=counts[qi].copy()
            qd=qdate[qi];ql=qlow[qi];qok=valid;qix=qi
        for hs in range(0,len(hi),history_rows):
            hb=hi[hs:hs+history_rows];h,hvalid=unit_rows(hvectors[hb])
            hvalid &= np.array([hmeta[i].get('valid_embedding',True) for i in hb])
            hd=hdate[hb];same_query=np.array([qlookup.get(hids[i],-1) for i in hb],dtype=np.int64)
            if torch is not None:h=tensor(h);hvalid=tensor(hvalid);hd=tensor(hd);same_query=tensor(same_query);hb_device=tensor(hb)
            for qs in range(0,len(qi),query_rows):
                sl=slice(qs,min(len(qi),qs+query_rows))
                allowed=(hd[None,:]>=ql[sl,None]) & (hd[None,:]<qd[sl,None])
                allowed &= hvalid[None,:] & (same_query[None,:]!=qix[sl,None])
                sim=q[sl]@h.T
                if torch is not None:
                    ct[sl]+=allowed.sum(dim=1)
                    sim=sim.clamp(-1,1).masked_fill(~(allowed & qok[sl,None]),-torch.inf)
                    order=torch.argsort(sim,dim=1,descending=True,stable=True)[:,:min(k,len(hb))]
                    best=torch.gather(sim,1,order);best_idx=hb_device[order]
                    merged=torch.cat([s[sl],best],dim=1);merged_idx=torch.cat([idx[sl],best_idx],dim=1)
                    order=torch.argsort(merged,dim=1,descending=True,stable=True)[:,:k]
                    s[sl]=torch.gather(merged,1,order);idx[sl]=torch.gather(merged_idx,1,order)
                else:
                    ct[sl]+=allowed.sum(axis=1)
                    sim=np.where(allowed & qok[sl,None],np.clip(sim,-1,1),-np.inf)
                    order=np.argsort(-sim,axis=1,kind='stable')[:,:min(k,len(hb))]
                    best=np.take_along_axis(sim,order,axis=1);best_idx=hb[order]
                    merged=np.concatenate([s[sl],best],axis=1);merged_idx=np.concatenate([idx[sl],best_idx],axis=1)
                    order=np.argsort(-merged,axis=1,kind='stable')[:,:k]
                    s[sl]=np.take_along_axis(merged,order,axis=1);idx[sl]=np.take_along_axis(merged_idx,order,axis=1)
        if torch is not None:s=s.cpu().numpy();idx=idx.cpu().numpy();ct=ct.cpu().numpy()
        scores[qi]=s;neighbors[qi]=idx;counts[qi]=ct
    finite=np.isfinite(scores);neighbors[~finite]=-1
    actual=finite.sum(axis=1)
    mean=np.divide(np.where(finite,scores,0).sum(axis=1,dtype=np.float64),actual,
      out=np.full(len(qmeta),np.nan),where=actual>0)
    min_k=1-mean;main=np.where(actual==k,min_k,np.nan)
    return {'novelty_raw':main,'novelty_min_k_auxiliary':min_k,'historical_count':counts,'actual_k':actual,
      'query_valid':qvalid,'neighbor_rows':neighbors,'neighbor_cosines':scores,
      'history_years':history_years,'k':k,'tie_rule':'exact cosine ties retain lexicographic patent ID order'}

def fit_joint_residual(predictors,joint,predictor_names=None):
    """Fit only supplied training rows. Main PM~P+M and ME~M+E.

    Overall novelty belongs in the main regression controls; additional
    residualization predictors must be explicitly named as a separate branch.
    """
    x=np.asarray(predictors,dtype=float);y=np.asarray(joint,dtype=float)
    if x.ndim==1:x=x[:,None]
    assert x.ndim==2 and len(x)==len(y)
    ok=np.isfinite(x).all(axis=1)&np.isfinite(y)
    if ok.sum()<3:raise ValueError('At least three observed fitting rows required')
    design=np.column_stack([np.ones(ok.sum()),x[ok]])
    coef,_,rank,_=np.linalg.lstsq(design,y[ok],rcond=None)
    names=predictor_names or ['predictor_'+str(i) for i in range(x.shape[1])]
    assert len(names)==x.shape[1]
    return {'intercept':float(coef[0]),'coefficients':coef[1:].astype(float).tolist(),
      'predictor_names':list(names),'n_fit':int(ok.sum()),'rank':int(rank),
      'scope':'coefficients fitted only on explicitly supplied fitting rows'}

def apply_joint_residual(predictors,joint,model):
    x=np.asarray(predictors,dtype=float)
    if x.ndim==1:x=x[:,None]
    assert x.shape[1]==len(model['coefficients'])
    return np.asarray(joint,dtype=float)-model['intercept']-x@np.asarray(model['coefficients'])

def mechanism_anchoring(qvectors,qmeta,cited_vectors,cited_meta,reference_ids):
    """Mean/max M cosine with all strictly earlier references in denominator."""
    q,qok=unit_rows(qvectors);h,hok=unit_rows(cited_vectors)
    lookup={str(r['patent_id']):i for i,r in enumerate(cited_meta)}
    result=[]
    for i,r in enumerate(qmeta):
        # Deduplicate edges; unknown/out-of-library cited IDs remain in denominator.
        references=list(dict.fromkeys(str(v) for v in reference_ids.get(str(r['patent_id']),[])))
        used=[]
        for pid in references:
            j=lookup.get(pid)
            if j is not None:
                if as_date(cited_meta[j]['grant_date'])>=as_date(r['grant_date']):raise ValueError('Reference IDs must be prefiltered to strictly earlier grants')
                if hok[j]:used.append(j)
        sims=h[used]@q[i] if used and qok[i] else np.array([])
        result.append({'patent_id':str(r['patent_id']),'historical_reference_n':len(references),
          'covered_M_n':len(used),'M_coverage':len(used)/len(references) if references else None,
          'anchoring_mean':float(sims.mean()) if len(sims) else None,
          'anchoring_max':float(sims.max()) if len(sims) else None,
          'query_M_valid':bool(qok[i])})
    return result

def future_text_uptake(qvectors,qmeta,candidate_batches,observation_end,thresholds=(.7,.8,.9),query_rows=256):
    """All future-text candidates, exact five-year dates, CPC-domain partitions.

    CPU reference implementation; batches stream (vectors,metadata). This is
    explicitly different from similarity-weighted citation counts and old fsim.
    A GPU implementation can be compared against this numerical reference.
    """
    q,qok=unit_rows(qvectors);n=len(qmeta);cut=day(observation_end)
    starts=np.array([day(r['grant_date']) for r in qmeta]);ends=np.array([day(add_years(r['grant_date'],5)) for r in qmeta])
    mature=ends<=cut;thresholds=tuple(float(x) for x in thresholds)
    assert all(-1<x<1 for x in thresholds)
    counts=np.zeros((n,len(thresholds),4),dtype=np.int64) # total,shared,new,unknown
    excess=np.zeros((n,len(thresholds)),dtype=np.float64)
    window_n=np.zeros(n,np.int64);valid_n=np.zeros(n,np.int64)
    qfields=[set(r.get('cpc_subclasses') or []) for r in qmeta]
    for hv,hm in candidate_batches:
        h,hok=unit_rows(hv);hd=np.array([day(r['grant_date']) for r in hm]);hf=[set(r.get('cpc_subclasses') or []) for r in hm]
        for qs in range(0,n,query_rows):
            qe=min(n,qs+query_rows);sim=np.clip(q[qs:qe]@h.T,-1,1)
            for local,i in enumerate(range(qs,qe)):
                date_ok=(hd>starts[i])&(hd<=ends[i])&(hd<=cut)
                date_ok &= np.array([str(r['patent_id'])!=str(qmeta[i]['patent_id']) for r in hm])
                window_n[i]+=int(date_ok.sum());valid_candidates=date_ok & hok
                valid_n[i]+=int(valid_candidates.sum());usable=valid_candidates & qok[i]
                domains=np.array([3 if not qfields[i] or not fields else (1 if qfields[i]&fields else 2) for fields in hf])
                for j,tau in enumerate(thresholds):
                    hit=usable & (sim[local]>=tau);counts[i,j,0]+=int(hit.sum())
                    for domain in [1,2,3]:counts[i,j,domain]+=int(np.sum(hit & (domains==domain)))
                    excess[i,j]+=float(np.maximum((sim[local][usable]-tau)/(1-tau),0).sum(dtype=np.float64))
    assert np.array_equal(counts[:,:,0],counts[:,:,1:].sum(axis=2))
    return {'counts':counts,'excess_similarity_sum':excess,'window_candidate_n':window_n,'valid_vector_candidate_n':valid_n,
      'mature':mature,'query_valid':qok,'main_label_usable':mature & qok,
      'thresholds':thresholds,'domains':['all','shared_subclass','new_field','classification_unknown'],
      'label_policy':'immature rows are observed partial counts only; main labels must be masked using main_label_usable'}
