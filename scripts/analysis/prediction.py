"""Training-only transforms and learners shared by forecasting tasks."""
from scripts.analysis.common import *
import warnings,joblib
from scipy.stats import spearmanr
from sklearn.linear_model import PoissonRegressor,Ridge,LogisticRegression
from sklearn.metrics import mean_poisson_deviance,mean_absolute_error,mean_squared_error,r2_score,average_precision_score,roc_auc_score,brier_score_loss
from sklearn.exceptions import ConvergenceWarning

ALPHAS=setting('analysis','alphas',[.0001,.001,.01,.1,1.,10.])

class Preprocess:
    def fit(self,d):
        self.base=list(CONTROLS)+['assignee_missing']
        self.control_missing=[x for x in CONTROLS if d[x].isna().any()]
        self.medians=d[self.base+['overall','P','M','E','coverage']].median().fillna(0)
        self.anchor_mean=float(d.anchor.mean());self.residuals={}
        for joint,components in [('PM',['P','M']),('ME',['M','E'])]:
            ok=d[[joint,*components]].notna().all(axis=1);a=d.loc[ok,components].to_numpy(float)
            b=d.loc[ok,joint].to_numpy(float);design=np.column_stack([np.ones(len(a)),a])
            coef=np.linalg.lstsq(design,b,rcond=None)[0];assert len(b)>100
            self.residuals[joint]=(components,coef,int(ok.sum()))
        self.rare={v for v,n in d.subfield.value_counts().items() if n>=100}
        self.categories=sorted({v if v in self.rare else 'OTHER' for v in d.subfield})
        if 'OTHER' not in self.categories:self.categories.append('OTHER')
        self.names=self.numeric_names()+['sub_'+v for v in self.categories]
        a=self.unscaled(d);self.mean=a.mean(axis=0);self.scale=a.std(axis=0)
        self.scale[self.scale<1e-12]=1
        return self
    def numeric_names(self):
        return ['year_numeric',*self.base,*[x+'_ctrlmissing' for x in self.control_missing],
            'overall','overall_missing','P','M','E','P_missing','M_missing','E_missing',
            'PMres','MEres','PM_missing','ME_missing','anchor','anchor_missing','coverage','coverage_missing','PMxA']
    def unscaled(self,d):
        z=pd.DataFrame(index=d.index);z['year_numeric']=d.year_numeric
        for name in self.base:z[name]=d[name].fillna(self.medians[name])
        for name in self.control_missing:z[name+'_ctrlmissing']=d[name].isna().astype(float)
        for name in ['overall','P','M','E']:
            z[name]=d[name].fillna(self.medians[name]);z[name+'_missing']=d[name].isna().astype(float)
        for joint,(components,coef,n) in self.residuals.items():
            ok=d[[joint,*components]].notna().all(axis=1)
            residual=d[joint]-coef[0]-d[components].to_numpy(float)@coef[1:]
            z[joint+'res']=residual.where(ok,0.);z[joint+'_missing']=(~ok).astype(float)
        z['anchor']=(d.anchor-self.anchor_mean).fillna(0);z['anchor_missing']=d.anchor.isna().astype(float)
        z['coverage']=d.coverage.fillna(self.medians['coverage']);z['coverage_missing']=d.coverage.isna().astype(float)
        z['PMxA']=z.PMres*z.anchor
        sub=d.subfield.where(d.subfield.isin(self.rare),'OTHER')
        numeric=z.reindex(columns=self.numeric_names()).to_numpy(float)
        cats=np.column_stack([(sub==v).to_numpy(float) for v in self.categories])
        a=np.column_stack([numeric,cats]);assert np.isfinite(a).all();return a
    def transform(self,d):return (self.unscaled(d)-self.mean)/self.scale
    def indices(self,name):
        keep=['year_numeric']+['sub_'+v for v in self.categories]
        if name!='B0':keep+=self.base+[x+'_ctrlmissing' for x in self.control_missing]
        if name in ['B2','B3','B4','B5']:keep+=['overall','overall_missing']
        if name in ['B3','B4','B5']:keep+=['P','M','E','P_missing','M_missing','E_missing']
        if name in ['B4','B5']:keep+=['PMres','MEres','PM_missing','ME_missing']
        if name=='B5':keep+=['anchor','anchor_missing','coverage','coverage_missing','PMxA']
        return [self.names.index(x) for x in keep]

def learn(kind,alpha,x,y):
    if kind=='count':m=PoissonRegressor(alpha=alpha,max_iter=1000,tol=1e-7)
    elif kind=='continuous':m=Ridge(alpha=alpha,solver='cholesky')
    else:m=LogisticRegression(C=1/alpha,solver='lbfgs',max_iter=1000,tol=1e-7)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always');m.fit(x,y)
    issues=[str(w.message) for w in seen];niter=getattr(m,'n_iter_',None);iterations=int(np.max(niter)) if niter is not None else 0
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in seen)
    if not converged:
        m.set_params(max_iter=4000)
        with warnings.catch_warnings(record=True) as retry:
            warnings.simplefilter('always');m.fit(x,y)
        issues +=['retry4000: '+str(w.message) for w in retry]
        converged=not any(issubclass(w.category,ConvergenceWarning) for w in retry)
        iterations=int(np.max(getattr(m,'n_iter_',0)))
    if not converged:raise RuntimeError('Model failed convergence: '+str(issues))
    return m,{'converged':converged,'iterations':iterations,'warnings':issues}

def predict(model,kind,x):
    v=model.predict_proba(x)[:,1] if kind=='binary' else model.predict(x)
    assert np.isfinite(v).all()
    return np.clip(v,1e-12,None) if kind=='count' else v

def selection_loss(kind,y,pred):
    if kind=='count':return float(mean_poisson_deviance(y,pred))
    if kind=='continuous':return float(mean_squared_error(y,pred))
    return -float(average_precision_score(y,pred))

def metrics(kind,y,pred,ids):
    rho=float(spearmanr(y,pred).statistic)
    if kind=='count':return {'poisson_deviance':float(mean_poisson_deviance(y,pred)),'mae':float(mean_absolute_error(y,pred)),'spearman':rho}
    if kind=='continuous':return {'mae':float(mean_absolute_error(y,pred)),'rmse':float(np.sqrt(mean_squared_error(y,pred))),
        'spearman':rho,'r2':float(r2_score(y,pred))}
    k=max(1,int(np.floor(.05*len(y))));order=np.lexsort((np.asarray(ids).astype(str),-pred));hits=y[order[:k]]
    dcg=np.sum(hits/np.log2(np.arange(2,k+2)));ideal=np.sum(1/np.log2(np.arange(2,min(k,int(y.sum()))+2)))
    return {'roc_auc':float(roc_auc_score(y,pred)),'pr_auc':float(average_precision_score(y,pred)),
        'brier':float(brier_score_loss(y,pred)),'precision_at_k':float(hits.mean()),
        'recall_at_k':float(hits.sum()/y.sum()),'ndcg_at_k':float(dcg/ideal) if ideal else None,
        'budget_k':k,'positive_n':int(y.sum()),'prevalence':float(y.mean())}
