"""M1-M6 same-sample PPML/OLS; overall remains in M3-M6, assignee-cluster inference."""
from scripts.analysis.common import *
import statsmodels.api as sm
from scipy import linalg,stats
import warnings

def z_cell(series,year,field):
    grouped=series.groupby([year,field]);mean=grouped.transform('mean');sd=grouped.transform(lambda v:v.std(ddof=0))
    result=(series-mean)/sd.where(sd>1e-12)
    result=result.where(~(series.notna()&(sd<=1e-12)),0.)
    return result

def prepare(d,label):
    x=rawfeatures(d);meta={'imputation':{},'residuals':{}}
    controls=[]
    binary=['continuation','division','cip','assignee_missing']
    for name in [*CONTROLS,'assignee_missing']:
        v=x[name].astype(float);missing=v.isna();median=float(v.median());median=median if np.isfinite(median) else 0.
        v=v.fillna(median)
        if name not in binary:
            center=float(v.mean());scale=float(v.std(ddof=0));scale=scale if scale>1e-12 else 1.
            x[name]=(v-center)/scale
        else:center=0.;scale=1.;x[name]=v
        controls.append(name);meta['imputation'][name]={'median':median,'mean':center,'scale':scale,'missing_n':int(missing.sum())}
        if missing.any():x[name+'_ctrlmissing']=missing.astype(float);controls.append(name+'_ctrlmissing')
    for name in ['overall','P','M','E','PM','ME']:
        x[name+'_z_observed']=z_cell(x[name],d.grant_year,d.technical_subfield)
        x[name]=x[name+'_z_observed'].fillna(0.)
    for joint,components in [('PM',['P','M']),('ME',['M','E'])]:
        columns=[joint+'_z_observed',*[n+'_z_observed' for n in components]]
        fit=x[columns].notna().all(axis=1);a=x.loc[fit,[n+'_z_observed' for n in components]].to_numpy(float)
        b=x.loc[fit,joint+'_z_observed'].to_numpy(float);coef=np.linalg.lstsq(np.column_stack([np.ones(len(a)),a]),b,rcond=None)[0]
        residual=pd.Series(np.nan,index=x.index);residual.loc[fit]=b-coef[0]-a@coef[1:]
        x[joint+'res']=z_cell(residual,d.grant_year,d.technical_subfield).fillna(0.)
        x[joint+'_missing']=(~fit).astype(float)
        meta['residuals'][joint]={'intercept':coef[0],'component_slopes':coef[1:],'n_fit':int(fit.sum()),
            'fit_definition':'observed cohortstandardized joint against observed cohortstandardized two components, only fitting sample',
            'orthogonality_before_final_cell_rescale':[float(np.corrcoef(residual.loc[fit],a[:,i])[0,1]) for i in range(2)]}
    anchor=x.anchor.copy();meta['anchor_mean']=float(anchor.mean());x['anchor']=(anchor-anchor.mean()).fillna(0.)
    x['coverage']=(x.coverage-x.coverage.mean()).fillna(0.)
    x['PMxA']=x.PMres*x.anchor;x['MExA']=x.MEres*x.anchor
    counts=d.technical_subfield.value_counts();interact=d.technical_subfield.where(d.technical_subfield.map(counts)>=100,'OTHER_RARE')
    levels=sorted(interact.unique());interaction_cols=[]
    for field in levels:
        if field=='A61B5':continue
        for term in ['PMres','MEres']:
            name=term+'_sub_'+field;x[name]=x[term]*(interact==field).astype(float);interaction_cols.append(name)
    yearfe=pd.get_dummies(d.grant_year.astype(str),prefix='year',drop_first=True,dtype=float)
    fields=sorted(d.technical_subfield.unique());fields=['A61B5']+[f for f in fields if f!='A61B5']
    cat=pd.Categorical(d.technical_subfield,categories=fields)
    subfe=pd.get_dummies(cat,prefix='subfield',drop_first=True,dtype=float);subfe.index=d.index
    fe=pd.concat([yearfe,subfe],axis=1);x=pd.concat([x,fe],axis=1)
    base=controls+['P_missing','M_missing','E_missing']
    specs={'M0':[],'M1':base,'M2':base+['overall']}
    specs['M3']=specs['M2']+['P','M','E']
    specs['M4']=specs['M3']+['PMres','MEres','PM_missing','ME_missing']
    specs['M5']=specs['M4']+['anchor','anchor_missing','coverage','coverage_missing','PMxA','MExA']
    specs['M6']=specs['M5']+interaction_cols
    for name in ['P','M','E']:specs['S_'+name]=specs['M2']+[name]
    meta.update(controls=controls,interaction_cols=interaction_cols,interaction_levels=levels,reference='A61B5',
        observed_n=len(d),same_sample_sha256=hashlib.sha256('|'.join(d.patent_id).encode()).hexdigest())
    save(f'preprocessing/explanatory_{label}.json',meta)
    return x,specs,list(fe.columns),meta

def design(x,terms,fe):
    names=['Intercept'];parts=[np.ones(len(x))];dropped=[]
    for term in terms+fe:
        v=x[term].to_numpy(float);assert np.isfinite(v).all()
        if np.ptp(v)<1e-12:dropped.append(term+':constant');continue
        names.append(term);parts.append(v)
    a=np.column_stack(parts);gram=a.T@a
    # Pivoted QR identifies numerical redundancy. Retained columns keep their
    # original ordering; selection does not use outcome significance.
    q,r,pivot=linalg.qr(gram,pivoting=True,mode='economic');tol=np.max(np.abs(np.diag(r)))*1e-11
    rank=int((np.abs(np.diag(r))>tol).sum())
    if rank<len(names):
        keep=sorted(pivot[:rank]);dropped +=[names[i]+':rankredundant' for i in range(len(names)) if i not in keep]
        a=a[:,keep];names=[names[i] for i in keep]
    singular=np.linalg.svd(a.T@a,compute_uv=False)
    return a,names,dropped,{'columns':len(names),'rank':len(names),'condition_number':float(np.sqrt(singular[0]/singular[-1]))}

def fitmodel(d,x,fe,terms,target,family,label,model):
    rel=f'explanatory/{target}_{label}_{model}';done=output(rel+'/summary.json')
    started=time.time();a,names,dropped,rank=design(x,terms,fe);y=d[OUTCOMES[target][0]].to_numpy(float)
    groups=d.assignee_cluster_key.astype(str).to_numpy()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        if family=='count':
            result=sm.GLM(y,a,family=sm.families.Poisson()).fit(maxiter=250,tol=1e-8,cov_type='cluster',cov_kwds={'groups':groups})
            converged=bool(result.converged);fitstat=1-float(result.deviance/result.null_deviance)
            dispersion=float(np.sum(result.resid_pearson**2)/result.df_resid)
        else:
            result=sm.OLS(y,a).fit(cov_type='cluster',cov_kwds={'groups':groups});converged=True;fitstat=float(result.rsquared);dispersion=None
    assert converged,'PPML did not converge'
    coef=np.asarray(result.params);cov=np.asarray(result.cov_params());se=np.sqrt(np.diag(cov))
    assert np.isfinite(coef).all() and np.isfinite(se).all()
    pvals=2*stats.norm.sf(np.abs(coef/se));rows=[]
    for i,name in enumerate(names):
        rows.append({'target':target,'sample':label,'model':model,'term':name,'coef':coef[i],'std_err':se[i],
            'p_value':pvals[i],'ci95_low':coef[i]-1.95996398454*se[i],'ci95_high':coef[i]+1.95996398454*se[i],'stars':stars(pvals[i])})
    pd.DataFrame(rows).to_csv(output(rel+'/coefficients.csv'),index=False,encoding='utf-8-sig')
    np.savez_compressed(output(rel+'/parameters_covariance.npz'),names=np.array(names),coef=coef,covariance=cov)
    fitted=np.asarray(result.fittedvalues);score=(a.T@(y-fitted));score_norm=float(np.max(np.abs(score))/max(1.,float(y.sum())))
    summary={'target':target,'model':model,'sample':label,'nobs':len(d),'clusters':len(np.unique(groups)),
        'family':'PPML' if family=='count' else 'OLS','converged':converged,'fit_stat':fitstat,'aic':float(result.aic),
        'pearson_dispersion':dispersion,'score_max_relative':score_norm,'rank':rank,'dropped_terms':dropped,
        'max_fitted':float(fitted.max()),'warnings':[str(w.message) for w in seen],
        'sample_sha256':hashlib.sha256('|'.join(d.patent_id.astype(str)).encode()).hexdigest(),
        'seconds':round(time.time()-started,2)}
    if model=='M6':
        wald=[]
        for part in ['PMres_sub_','MEres_sub_','ALL']:
            ix=[i for i,name in enumerate(names) if ('_sub_' in name if part=='ALL' else name.startswith(part))]
            b=coef[ix];v=cov[np.ix_(ix,ix)];dof=int(np.linalg.matrix_rank(v));statistic=float(b@np.linalg.pinv(v)@b)
            wald.append({'term':part,'chi2':statistic,'df':dof,'p_value':float(stats.chi2.sf(statistic,dof))})
        summary['joint_wald']=wald
    save(rel+'/summary.json',summary);log(f'Explanation {target} {label} {model} complete n={len(d)} {summary["seconds"]}s')
    return summary

def main():
    started=time.time();base=readpanel();d=base.loc[base.overall_novelty_raw.notna()].copy().reset_index(drop=True)
    x,specs,fe,meta=prepare(d,'main');records=[]
    for target in ['citations','semantic','newfield']:
        # Zero-outcome FE categories create PPML separation and carry no slope
        # information. Remove them identically for every model of that outcome.
        keep=np.ones(len(d),bool);removed=[]
        if OUTCOMES[target][1]=='count':
            outcome=d[OUTCOMES[target][0]]
            for col in ['grant_year','technical_subfield']:
                sums=outcome.groupby(d[col]).sum();levels=sums[sums==0].index.tolist()
                if levels:keep &=~d[col].isin(levels).to_numpy();removed.append({'column':col,'levels':levels})
        save(f'checks/separation_{target}.json',{'removed_zerooutcome_FE':removed,'removed_n':int((~keep).sum())})
        dd=d.loc[keep];xx=x.loc[keep]
        for name in ['M0','M1','M2','M3','M4','M5','M6','S_P','S_M','S_E']:
            records.append(fitmodel(dd,xx,fe,specs[name],target,OUTCOMES[target][1],'main',name))
    complete=d[['overall_novelty_raw','problem_novelty_raw','mechanism_novelty_raw','effect_novelty_raw',
        'pm_config_novelty_raw','me_config_novelty_raw','anchoring_raw']].notna().all(axis=1)
    cc=d.loc[complete].copy().reset_index(drop=True);xc,specc,fec,mc=prepare(cc,'completecase')
    for target in ['citations','semantic','newfield']:
        for name in ['M2','M5']:records.append(fitmodel(cc,xc,fec,specc[name],target,OUTCOMES[target][1],'completecase',name))
    core=['overall','P','M','E','PMres','MEres','anchor','PMxA','MExA']
    corr=x[core].corr();corr.to_csv(output('quality/explanatory_core_correlations.csv'),encoding='utf-8-sig')
    inv=np.linalg.pinv(corr.to_numpy());pd.DataFrame({'term':core,'VIF_correlation_inverse':np.diag(inv)}).to_csv(output('quality/explanatory_core_vif.csv'),index=False,encoding='utf-8-sig')
    parts=[pd.read_csv(path) for path in output('explanatory').glob('*/coefficients.csv')]
    pd.concat(parts,ignore_index=True).to_csv(output('results/explanatory_coefficients.csv'),index=False,encoding='utf-8-sig')
    save('results/explanatory_metadata.json',records)
    x[['overall','P','M','E','PMres','MEres','anchor','PMxA','MExA']].assign(patent_id=d.patent_id).to_parquet(output('data/explanatory_transformed_core.parquet'),index=False)
    save('checks/01_explanatory_complete.json',{'complete':True,'models':len(records),'main_n':len(d),'completecase_n':len(cc),
        'all_converged':all(r['converged'] for r in records),'max_score':max(r['score_max_relative'] for r in records if r['family']=='PPML'),
        'seconds':round(time.time()-started,2),'source_panel':'configured measured panel'})

if __name__=='__main__':main()
