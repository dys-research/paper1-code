from pathlib import Path
import os,sys,json,hashlib,time,datetime
from project_config import input_path, work_dir, setting
ROOT=work_dir('analysis')
STAGE3=work_dir('measurement')
import numpy as np,pandas as pd

def output(rel):
    p=(ROOT/rel).resolve();assert p.is_relative_to(ROOT.resolve())
    p.parent.mkdir(parents=True,exist_ok=True);return p
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def default(x):
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.bool_,)):return bool(x)
    if isinstance(x,np.ndarray):return x.tolist()
    return str(x)
def save(rel,obj):
    p=output(rel);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=default),encoding='utf-8');t.replace(p)
def log(msg):
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    print(stamp,msg,flush=True)
def readpanel(all_rows=False):
    """Read a measured panel without relying on private acceptance receipts."""
    configured=input_path('panel',required=False)
    path=configured if configured is not None else STAGE3/'data/panel_model_ready.parquet'
    if not path.is_file():
        raise FileNotFoundError(f"Measured panel not found: {path}. Run measurement or set inputs.panel.")
    d=pd.read_parquet(path)
    required={'patent_id','is_focal_candidate','grant_year','grant_date','label_window_end',
              'label_available_date','technical_subfield','assignee_cluster_key'}
    absent=required-set(d.columns)
    if absent:raise ValueError('Panel is missing columns: '+', '.join(sorted(absent)))
    if not d.patent_id.is_unique:raise ValueError('Panel patent_id must be unique')
    if not all_rows:d=d.loc[d.is_focal_candidate].copy()
    d=d.reset_index(drop=True)
    d['patent_id']=d.patent_id.astype(str)
    for col in ['grant_date','label_window_end','label_available_date']:d[col]=pd.to_datetime(d[col])
    return d

CONTROLS={
 'log_backward':'backward_core_citation_count','backward_age':'backward_citation_mean_age',
 'log_claims':'num_claims','log_inventors':'inventor_count','log_npl':'non_patent_reference_count',
 'log_cpc_groups':'cpc_group_count','log_cpc_subclasses':'cpc_subclass_count',
 'log_title_words':'title_words','log_abstract_words':'abstract_words','log_claim_words':'independent_claim_words',
 'grant_lag':'grant_lag_years','log_family':'family_size_at_grant_no_cip',
 'continuation':'is_continuation','division':'is_division','cip':'is_cip',
 'log_assignee_prior':'assignee_prior_patents_mean','log_assignee_cites':'assignee_prior_received_cites_mean',
 'log_inventor_prior':'inventor_prior_patents_mean','inventor_career':'inventor_career_years_mean',
 'assignee_new_share':'assignee_new_entity_share','inventor_new_share':'inventor_new_entity_share',
 'log_assignee_count':'assignee_count','parent_unresolved':'unresolved_parent_relation_share'}
NOVELTY={'overall':'overall_novelty_raw','P':'problem_novelty_raw','M':'mechanism_novelty_raw',
 'E':'effect_novelty_raw','PM':'pm_config_novelty_raw','ME':'me_config_novelty_raw'}
OUTCOMES={'citations':('forward_cites_5y','count'),'semantic':('semantic_reuse_5y','continuous'),
 'newfield':('new_field_cites_5y','count'),'top5':('top5_impact_with_ties','binary')}
LABELS={'overall':'Overall novelty','P':'Problem novelty','M':'Mechanism novelty','E':'Effect novelty',
 'PMres':'Residual P–M joint-text novelty','MEres':'Residual M–E joint-text novelty',
 'anchor':'Knowledge anchoring','PMxA':'Residual P–M joint-text novelty × knowledge anchoring',
 'MExA':'Residual M–E joint-text novelty × knowledge anchoring','anchor_missing':'Knowledge anchoring missingness indicator',
 'log_backward':'ln(number of backward citations + 1)','backward_age':'Mean age of backward citations',
 'log_claims':'ln(number of claims + 1)','log_inventors':'ln(number of inventors + 1)',
 'P_missing':'P missingness indicator','M_missing':'M missingness indicator','E_missing':'E missingness indicator',
 'PM_missing':'P–M joint-text missingness indicator','ME_missing':'M–E joint-text missingness indicator',
 'coverage':'Observed cited-M coverage','coverage_missing':'Cited-M coverage missingness indicator'}

def rawfeatures(d):
    x=pd.DataFrame(index=d.index)
    for name,source in CONTROLS.items():
        v=pd.to_numeric(d[source],errors='coerce').astype(float)
        if name=='parent_unresolved':v=v.fillna(0) # no recorded parent relation, not a role-imputation
        x[name]=np.log1p(v.clip(lower=0)) if name.startswith('log_') else v
    for name,source in NOVELTY.items():x[name]=pd.to_numeric(d[source],errors='coerce')
    for name in ['P','M','E','PM','ME']:x[name+'_missing']=x[name].isna().astype(float)
    x['anchor']=pd.to_numeric(d['anchoring_raw'],errors='coerce');x['anchor_missing']=x.anchor.isna().astype(float)
    x['coverage']=pd.to_numeric(d['M_coverage'],errors='coerce');x['coverage_missing']=x.coverage.isna().astype(float)
    x['assignee_missing']=d['assignee_missing'].astype(float)
    x['year_numeric']=d['grant_year'].astype(float)
    x['subfield']=d['technical_subfield'].astype(str)
    return x

def stars(p):return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''
