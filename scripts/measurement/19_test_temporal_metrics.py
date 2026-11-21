"""Numerical and date-boundary tests; synthetic tests are not study results."""
from pathlib import Path
import json,sys
import numpy as np
from temporal_metrics import temporal_topk,add_years,as_date,fit_joint_residual,apply_joint_residual,mechanism_anchoring,future_text_uptake
from common import ROOT, input_path, setting
OUT=ROOT/'checks';OUT.mkdir(parents=True,exist_ok=True)

def brute(qv,qm,hv,hm,k,years):
    values=[];ids=[];counts=[]
    for v,q in zip(qv,qm):
        vv=[]
        for i,(u,h) in enumerate(zip(hv,hm)):
            if h['technical_subfield']!=q['technical_subfield'] or h['patent_id']==q['patent_id']:continue
            if not(add_years(q['grant_date'],-years)<=as_date(h['grant_date'])<as_date(q['grant_date'])):continue
            if not np.isfinite(u).all() or np.linalg.norm(u)==0:continue
            vv.append((float(np.clip(np.dot(v/np.linalg.norm(v),u/np.linalg.norm(u)),-1,1)),str(h['patent_id']),i))
        vv.sort(key=lambda r:(-r[0],r[1]));counts.append(len(vv));vv=vv[:k]
        values.append(1-np.mean([x[0] for x in vv]) if len(vv)==k else np.nan);ids.append([x[2] for x in vv])
    return values,ids,counts

qv=np.array([[1.,0,0],[1,0,0],[1,0,0]],dtype=np.float32)
qm=[{'patent_id':'q1','grant_date':'2020-02-29','technical_subfield':'A'},
 {'patent_id':'q2','grant_date':'2005-01-01','technical_subfield':'A'},
 {'patent_id':'q3','grant_date':'2020-02-29','technical_subfield':'NO_HISTORY'}]
hm=[{'patent_id':pid,'grant_date':d,'technical_subfield':g} for pid,d,g in [
 ('boundary','2010-02-28','A'),('too_old','2010-02-27','A'),('same_day','2020-02-29','A'),
 ('future','2021-01-01','A'),('other_field','2019-01-01','B'),
 ('tie_z','2019-01-02','A'),('tie_a','2019-01-03','A'),('invalid','2019-01-04','A')]]
hv=np.array([[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,0]],np.float32)
out=temporal_topk(qv,qm,hv,hm,k=2,query_rows=1,history_rows=2)
assert out['historical_count'].tolist()==[3,0,0]
assert [hm[i]['patent_id'] for i in out['neighbor_rows'][0]]==['boundary','tie_a']
assert np.isclose(out['novelty_raw'][0],.5) and np.isnan(out['novelty_raw'][1:]).all()
# Chunking, tie order and independent brute-force equivalence.
rng=np.random.default_rng(20261102)
h=rng.normal(size=(41,7)).astype(np.float32);q=rng.normal(size=(9,7)).astype(np.float32)
hm2=[{'patent_id':f'h{i:03d}','grant_date':f'{2000+i%21}-06-01','technical_subfield':'AB'[i%2]} for i in range(41)]
qm2=[{'patent_id':f'q{i:03d}','grant_date':f'{2010+i}-03-01','technical_subfield':'AB'[i%2]} for i in range(9)]
br,bi,bc=brute(q,qm2,h,hm2,3,10)
for qr,hr in [(1,1),(2,7),(20,100)]:
    res=temporal_topk(q,qm2,h,hm2,k=3,query_rows=qr,history_rows=hr)
    assert np.allclose(res['novelty_raw'],br,atol=1e-6,equal_nan=True)
    assert res['historical_count'].tolist()==bc
    assert all(res['neighbor_rows'][i,:len(ii)].tolist()==ii for i,ii in enumerate(bi))
model=fit_joint_residual([0,1,2],[3,5,7]);assert np.allclose(apply_joint_residual([100],[500],model),[297])
before=dict(model);apply_joint_residual([1e6],[0],model);assert model==before
two=fit_joint_residual([[0,0],[1,0],[0,1],[1,1]],[3,5,2,4],['P','M'])
assert np.allclose(two['coefficients'],[2,-1])
assert np.allclose(apply_joint_residual([[2,3]],[10],two),[6])
anchor=mechanism_anchoring(np.array([[1.,0.]]),[{'patent_id':'f','grant_date':'2020-01-01'}],
 np.array([[1.,0.],[0.,1.]]),[{'patent_id':'a','grant_date':'2010-01-01'},{'patent_id':'b','grant_date':'2011-01-01'}],{'f':['a','b','a','outside']})[0]
assert anchor['historical_reference_n']==3 and anchor['covered_M_n']==2 and np.isclose(anchor['M_coverage'],2/3)
assert np.isclose(anchor['anchoring_mean'],.5)
fv=np.array([[1.,0.],[1.,0.]])
fm=[{'patent_id':'f1','grant_date':'2020-02-29','cpc_subclasses':['A']},
 {'patent_id':'immature','grant_date':'2022-01-01','cpc_subclasses':['A']}]
cv=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,0]],float)
cm=[{'patent_id':str(i),'grant_date':d,'cpc_subclasses':fields} for i,(d,fields) in enumerate([
 ('2020-02-29',['A']),('2020-03-01',['A']),('2025-02-28',['B']),('2025-03-01',['A']),
 ('2021-01-01',[]),('2026-01-01',['B']),('2021-02-01',['A'])])]
reuse=future_text_uptake(fv,fm,[(cv[:3],cm[:3]),(cv[3:],cm[3:])],'2025-12-30',thresholds=(.8,))
assert reuse['counts'][0,0].tolist()==[3,1,1,1]
assert reuse['window_candidate_n'][0]==4 and reuse['valid_vector_candidate_n'][0]==3
assert reuse['main_label_usable'].tolist()==[True,False]
torch_status='not_available_in_this_python'
cuda_tested=False
try:
    import torch
    tr=temporal_topk(q,qm2,h,hm2,k=3,query_rows=2,history_rows=7,engine='torch',device='cpu')
    assert np.allclose(tr['novelty_raw'],br,atol=1e-6,equal_nan=True)
    assert all(tr['neighbor_rows'][i,:len(ii)].tolist()==ii for i,ii in enumerate(bi))
    torch_status='torch_CPU_equivalent_to_brute_and_numpy'
    if torch.cuda.is_available():
        gpu=temporal_topk(q,qm2,h,hm2,k=3,query_rows=2,history_rows=7,engine='torch',device='cuda')
        assert np.allclose(gpu['novelty_raw'],br,atol=1e-6,equal_nan=True)
        assert all(gpu['neighbor_rows'][i,:len(ii)].tolist()==ii for i,ii in enumerate(bi))
        cuda_tested=True
except ModuleNotFoundError:pass
result={'passed':True,'synthetic_not_study_results':True,
 'checks':['ten-calendar-year lower boundary inclusive','same-day/future exclusions','same-field restriction',
 'no-history and invalid vectors remain missing','ID tie order and chunk invariance','independent brute-force neighbors',
 'fit-only residual coefficients','anchoring distinct-edge coverage denominator','exact fifth anniversary and leap day',
 'same-metric domain partition and unknown CPC','immature future labels masked'],
 'torch_backend':torch_status,'CUDA_tested':cuda_tested,'new_paid_calls':0}
(OUT/('temporal_metrics_'+('torch_cpu' if torch_status.startswith('torch_') else 'numpy')+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
