"""Joint tests, observed-support anchoring associations and complete-case splines."""
from scripts.robustness.common import *
from scipy import stats
from statsmodels.stats.multitest import multipletests

def restricted_spline(v,knots):
    v=np.asarray(v,float);k=np.asarray(knots,float);cols=[v]
    for t in k[:-2]:
        h=np.maximum(v-t,0)**3-(k[-1]-t)/(k[-1]-k[-2])*np.maximum(v-k[-2],0)**3+(k[-2]-t)/(k[-1]-k[-2])*np.maximum(v-k[-1],0)**3
        cols.append(h/(k[-1]-k[0])**2)
    return np.column_stack(cols)

def run():
    # Joint role/joint-text/anchoring blocks, using stored robust covariance.
    wald=[];subfields=[]
    for target in ['citations','semantic','newfield']:
        for model,terms in [('M3',['P','M','E']),('M4',['PMres','MEres']),('M5',['anchor','anchor_missing','coverage','coverage_missing','PMxA','MExA'])]:
            z=np.load(ANALYSIS/f'explanatory/{target}_main_{model}/parameters_covariance.npz');names=z['names'].tolist()
            ix=[names.index(t) for t in terms if t in names];coef=z['coef'][ix];cov=z['covariance'][np.ix_(ix,ix)]
            dof=int(np.linalg.matrix_rank(cov));value=float(coef@np.linalg.pinv(cov)@coef)
            wald.append({'target':target,'model':model,'block':'+'.join(terms),'chi2':value,'df':dof,'p_value':stats.chi2.sf(value,dof)})
        z=np.load(ANALYSIS/f'explanatory/{target}_main_M6/parameters_covariance.npz');names=z['names'].tolist();beta=z['coef'];cov=z['covariance']
        for role in ['PMres','MEres']:
            levels=['A61B5']+[n[len(role+'_sub_'):] for n in names if n.startswith(role+'_sub_')]
            rr=[]
            for level in levels:
                a=np.zeros(len(beta));a[names.index(role)]=1
                if level!='A61B5':a[names.index(role+'_sub_'+level)]=1
                estimate=float(a@beta);se=float(np.sqrt(a@cov@a))
                rr.append({'target':target,'role':role,'subfield':level,'coef':estimate,'std_err':se,
                    'p_value':2*stats.norm.sf(abs(estimate/se)),'ci95_low':estimate-1.95996398454*se,'ci95_high':estimate+1.95996398454*se})
            corrected=multipletests([x['p_value'] for x in rr],method='fdr_bh')[1]
            for row,padj in zip(rr,corrected):row['p_BH_within_target_role']=padj;subfields.append(row)
    pd.DataFrame(wald).to_csv(output('results/joint_block_Wald.csv'),index=False,encoding='utf-8-sig')
    pd.DataFrame(subfields).to_csv(output('results/subfield_marginals_BH.csv'),index=False,encoding='utf-8-sig')
    # Response-scale average marginal associations, varying anchor only within observed support.
    d=panel();d=d.loc[d.overall_novelty_raw.notna()].copy().reset_index(drop=True)
    x,spec,fe,meta=ex.prepare(d,'main_reconstruction');effects=[]
    for target in ['citations','semantic','newfield']:
        keep=np.ones(len(d),bool)
        if target=='newfield':keep&=~d.technical_subfield.eq('A61B13').to_numpy()
        a,names,_,_=ex.design(x.loc[keep],spec['M5'],fe)
        stored=np.load(ANALYSIS/f'explanatory/{target}_main_M5/parameters_covariance.npz')
        assert names==stored['names'].tolist();beta=stored['coef'];cov=stored['covariance']
        for quantile in [.05,.10,.25,.5,.75,.9,.95]:
            anchor=float(d.anchoring_raw.quantile(quantile));centered=anchor-meta['anchor_mean'];aa=a.copy()
            aa[:,names.index('anchor')]=centered
            for term,role in [('PMxA','PMres'),('MExA','MEres')]:aa[:,names.index(term)]=aa[:,names.index(role)]*centered
            mu=aa@beta if target=='semantic' else np.exp(aa@beta)
            for role,interaction in [('PMres','PMxA'),('MEres','MExA')]:
                l=np.zeros(len(beta));l[names.index(role)]=1;l[names.index(interaction)]=centered
                slope=l@beta
                if target=='semantic':value=float(slope);gradient=l
                else:value=float(mu.mean()*slope);gradient=(mu[:,None]*aa).mean(axis=0)*slope+mu.mean()*l
                se=float(np.sqrt(gradient@cov@gradient))
                effects.append({'target':target,'role':role,'anchor_quantile':quantile,'anchor_raw':anchor,
                   'average_marginal_association':value,'std_err':se,'ci95_low':value-1.95996398454*se,'ci95_high':value+1.95996398454*se})
    pd.DataFrame(effects).to_csv(output('results/response_scale_marginal_effects.csv'),index=False,encoding='utf-8-sig')
    # Complete observed records: no missing-role observations in spline evidence.
    cols=['overall_novelty_raw','problem_novelty_raw','mechanism_novelty_raw','effect_novelty_raw','pm_config_novelty_raw','me_config_novelty_raw','anchoring_raw']
    cc=d.loc[d[cols].notna().all(axis=1)].copy().reset_index(drop=True)
    xx,ss,ff,mm=ex.prepare(cc,'observed_spline');knots=np.quantile(xx.P,[.05,.35,.65,.95]);basis=restricted_spline(xx.P,knots)
    xx['P_nonlinear1']=basis[:,1];xx['P_nonlinear2']=basis[:,2]
    terms=ss['M5']+['P_nonlinear1','P_nonlinear2'];curves=[];spline_summaries=[]
    for target in ['citations','semantic','newfield']:
        summary=ex.fitmodel(cc,xx,ff,terms,target,OUTCOMES[target][1],'observed_spline','SPLINE');spline_summaries.append(summary)
        z=np.load(ROOT/f'explanatory/{target}_observed_spline_SPLINE/parameters_covariance.npz');names=z['names'].tolist();beta=z['coef'];cov=z['covariance']
        grid=np.linspace(*np.quantile(xx.P,[.01,.99]),61);ref=restricted_spline([0.],knots)[0]
        for value,b in zip(grid,restricted_spline(grid,knots)):
            v=np.zeros(len(beta))
            for term,diff in zip(['P','P_nonlinear1','P_nonlinear2'],b-ref):v[names.index(term)]=diff
            delta=float(v@beta);se=float(np.sqrt(v@cov@v))
            curves.append({'target':target,'P_z':value,'linear_predictor_change_at_mean_anchor':delta,'std_err':se,
               'ci95_low':delta-1.95996398454*se,'ci95_high':delta+1.95996398454*se,
               'scale':'log conditional mean ratio' if target!='semantic' else 'log1p reuse difference'})
    pd.DataFrame(curves).to_csv(output('results/observed_only_spline_curves.csv'),index=False,encoding='utf-8-sig')
    save('results/spline_definition.json',{'n':len(cc),'knots':knots,'summaries':spline_summaries,
         'plot_scale':'linear predictor difference at mean anchoring, interactions retained; not causal'})


def main():
    with analysis_outputs():
        run()


if __name__ == "__main__":
    main()
