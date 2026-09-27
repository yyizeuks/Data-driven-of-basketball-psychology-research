import numpy as np, pandas as pd, json
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.multitest import multipletests
df=pd.read_pickle('merged3_topics.pkl'); dt=np.load('doc_topic8.npy'); K=8
LB=['Perceptual-Cognitive & Motor Performance','Physiological Load & Recovery','Training, Practice & Conditioning','Injury Risk & Adapted Sport','Concussion, Anxiety & Symptomatology','Coaching & Youth Development','Rehabilitation & Sport Technology','Education, Motivation & Psych. Needs']
yrs=df['year'].values; years=[y for y in range(1991,2025) if (yrs==y).sum()>=5]
n_by=np.array([(yrs==y).sum() for y in years]); Y=np.array(years,float)
def mann_kendall(x):
    n=len(x); s=sum(np.sign(x[j]-x[i]) for i in range(n-1) for j in range(i+1,n))
    var=n*(n-1)*(2*n+5)/18; z=(s-np.sign(s))/np.sqrt(var) if s!=0 else 0.0
    p=2*(1-stats.norm.cdf(abs(z))); sen=np.median([(x[j]-x[i])/(j-i) for i in range(n-1) for j in range(i+1,n)])
    return z,p,sen
rows=[]; pvals=[]
for t in range(K):
    m=np.array([dt[yrs==y,t].mean() for y in years])*100   # % scale
    X=sm.add_constant(Y); ols=sm.OLS(m,X).fit(); hac=sm.OLS(m,X).fit(cov_type='HAC',cov_kwds={'maxlags':2})
    wls=sm.WLS(m,X,weights=n_by).fit(); ci=ols.conf_int()[1]
    bp_p=het_breuschpagan(ols.resid,X)[1]; dw=durbin_watson(ols.resid); z,mkp,sen=mann_kendall(m)
    rows.append(dict(topic=f'T{t+1}',label=LB[t],slope_pp_decade=round(ols.params[1]*10,2),ci_low=round(ci[0]*10,2),ci_high=round(ci[1]*10,2),
        p_ols=round(ols.pvalues[1],4),p_hac=round(hac.pvalues[1],4),slope_wls=round(wls.params[1]*10,2),p_wls=round(wls.pvalues[1],4),
        dw=round(dw,2),bp_p=round(bp_p,3),mk_z=round(z,2),p_mk=round(mkp,4),sen_pp_decade=round(sen*10,2),r2=round(ols.rsquared,3)))
    pvals.append(ols.pvalues[1])
rej,q,_,_=multipletests(pvals,method='fdr_bh')
for r,qq,rj in zip(rows,q,rej): r['q_fdr']=round(qq,4); r['sig_fdr05']=bool(rj)
out=pd.DataFrame(rows); out.to_csv('trend_stats.csv',index=False); json.dump(dict(years=years,n_by_year=n_by.tolist()),open('trend_years.json','w'))
print('years used:',years[0],'-',years[-1],'n=',len(years),'| min docs/yr',n_by.min())
print(out[['topic','slope_pp_decade','ci_low','ci_high','p_ols','p_hac','q_fdr','sig_fdr05','p_wls','p_mk','sen_pp_decade','dw','bp_p']].to_string(index=False))
# ---- construct hit distribution (S6) ----
from collections import Counter
hits=Counter(); sole=Counter()
for h in df['psy_hits']:
    for c in h: hits[c]+=1
    if len(h)==1: sole[h[0]]+=1
tab=pd.DataFrame([(c,hits[c],round(100*hits[c]/len(df),1),sole[c]) for c in hits],columns=['construct_stem','docs_matched','pct_of_corpus','sole_trigger_docs']).sort_values('docs_matched',ascending=False)
tab.to_csv('construct_hits.csv',index=False); print(tab.head(14).to_string(index=False))
print('docs with a single triggering construct:',sum(sole.values()),'| median constructs/doc:',int(df['psy_hits'].map(len).median()))
