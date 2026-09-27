import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
LB=['T1 Perceptual-Cognitive & Motor Performance','T2 Physiological Load & Recovery','T3 Training, Practice & Conditioning','T4 Injury Risk & Adapted Sport','T5 Concussion, Anxiety & Symptoms','T6 Coaching & Youth Development','T7 Rehabilitation & Sport Technology','T8 Education, Motivation & Psych. Needs']
SH=[l.split(' ',1)[0] for l in LB]; COL=plt.cm.tab10(np.linspace(0,1,10))[:8]
S=json.load(open('val.json')); ts=pd.read_csv('trend_stats.csv'); df=pd.read_pickle('merged3_topics.pkl')
# ---- Figure 3 ----
pt=np.array([r['jac20'] for r in S['seeds']]).mean(0)
fig,axs=plt.subplots(2,2,figsize=(8.6,6.2))
a=axs[0,0]; a.bar(range(1,9),pt,color=COL); a.axhline(pt.mean(),ls='--',c='gray',lw=1); a.set_xticks(range(1,9),SH); a.set_ylabel('Jaccard@20 vs reference'); a.set_title('(a) Seed stability by topic (5 seeds)',fontsize=9.5)
b=axs[0,1]; b.boxplot([[r['jac20_mean'] for r in S['seeds']],[r['jac20_mean'] for r in S['split']]],tick_labels=['reseeded (5)','split-half (5)']); b.set_ylabel('Mean Jaccard@20'); b.set_title('(b) Stability distributions',fontsize=9.5)
c=axs[1,0]; ref=S['ref_slopes']
for r in S['seeds']: c.scatter(ref,r['slopes'],alpha=0.55,s=22)
lim=[min(min(ref),min(min(r['slopes']) for r in S['seeds']))-0.2,max(max(ref),max(max(r['slopes']) for r in S['seeds']))+0.2]; c.plot(lim,lim,'k--',lw=0.8)
c.set_xlabel('Reference slope (pp/decade)'); c.set_ylabel('Reseeded slope (pp/decade)'); c.set_title('(c) Trend-slope replication (r̄ = %.2f)'%np.mean([r['r'] for r in S['seeds']]),fontsize=9.5)
d=axs[1,1]; E=S['criterion']['enrichments']
pairs=[(k.replace('->',' → ').replace('Sport Sciences','Sport Sci.').replace('Neurosciences','Neurosci.'),v[0],v[1]) for k,v in E.items() if k in ('T8->Education','T2->Sport Sciences','T1->Neurosciences')]
x=np.arange(len(pairs)); w=0.36
d.bar(x-w/2,[p[1] for p in pairs],w,label='within topic',color='#2b5f9e'); d.bar(x+w/2,[p[2] for p in pairs],w,label='corpus base',color='#b9c6d8')
d.set_xticks(x,[p[0] for p in pairs],fontsize=8); d.set_ylabel('% of documents'); d.set_title('(d) External criterion enrichment (χ²(56)=190.1, V=.152)',fontsize=9.5); d.legend(fontsize=8)
plt.tight_layout(); plt.savefig('fig3_validation.png',dpi=300,bbox_inches='tight'); plt.close()
# ---- Figure 9 ----
vol=df.groupby('year').size()
fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,3.9))
a1.bar(vol.index,vol.values,color='#4d7db4'); a1.set_title('(a) Annual publication volume',fontsize=10); a1.set_xlabel('Year'); a1.set_ylabel('Number of papers')
order=np.argsort(ts['slope_pp_decade'].values); v=ts.iloc[order]
cols=['#2e7d43' if s>0 else '#b64040' for s in v['slope_pp_decade']]
err=np.vstack([v['slope_pp_decade']-v['ci_low'],v['ci_high']-v['slope_pp_decade']])
a2.barh(range(8),v['slope_pp_decade'],xerr=err,color=cols,error_kw=dict(ecolor='#333333',lw=1.1,capsize=3))
a2.set_yticks(range(8),[f"{t} {LB[int(t[1])-1].split(' ',1)[1]}" for t in v['topic']],fontsize=7.2)
for i,(s,hi,lo,q,p) in enumerate(zip(v['slope_pp_decade'],v['ci_high'],v['ci_low'],v['q_fdr'],v['p_ols'])):
    lab=('**' if q<.01 else ('*' if q<.05 else ('†' if p<.05 else '')))
    if lab: a2.text((hi+0.06) if s>=0 else (lo-0.06),i,lab,va='center',ha='left' if s>=0 else 'right',fontsize=9)
a2.axvline(0,color='k',lw=0.8); a2.set_title('(b) Topic trend slope with 95% CI (pp per decade)',fontsize=10); a2.set_xlabel('Change in mean proportion per decade (pp)')
plt.tight_layout(); plt.savefig('fig5_volume_slope.png',dpi=300,bbox_inches='tight'); plt.close()
from PIL import Image
import os; os.makedirs('/mnt/user-data/outputs/tiff_rev',exist_ok=True)
for f,o in [('fig3_validation.png','Figure3'),('fig5_volume_slope.png','Figure9')]:
    im=Image.open(f).convert('RGB'); im.save(f'/mnt/user-data/outputs/tiff_rev/{o}.tif',format='TIFF',compression='tiff_lzw',dpi=(300,300)); print(o,im.size)
