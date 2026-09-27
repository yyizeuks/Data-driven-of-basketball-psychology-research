"""Validation battery (seed stability, split-half, trend replication, criterion validity). Resumable: python3 validate.py [budget_s]"""
import pickle, json, os, sys, time, numpy as np, pandas as pd
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress, pearsonr, chi2_contingency
BUDGET=int(sys.argv[1]) if len(sys.argv)>1 else 540; T0=time.time()
corpus=pickle.load(open('bow3.pkl','rb')); dic=Dictionary.load('dict3.gensim'); df=pd.read_pickle('merged3_topics.pkl')
Pref=np.load('phi8.npy'); DTref=np.load('doc_topic8.npy'); K=8; yrs=df['year'].values
SPEC=dict(passes=25,iterations=600,alpha='auto',eta='auto',chunksize=200)
years=[y for y in range(1991,2025) if (yrs==y).sum()>=5]
S=json.load(open('val.json')) if os.path.exists('val.json') else {'seeds':[],'split':[],'criterion':None}
def save(): json.dump(S,open('val.json','w'),indent=1)
def theta(m,cps):
    dt=np.zeros((len(cps),K))
    for i,b in enumerate(cps):
        for t,p in m.get_document_topics(b,minimum_probability=0): dt[i,t]=p
    return dt
def match(P,Q):
    D=np.array([[jensenshannon(P[i],Q[j]) for j in range(K)] for i in range(K)]); ri,ci=linear_sum_assignment(D)
    A=[set(np.argsort(-P[t])[:20]) for t in range(K)]; B=[set(np.argsort(-Q[t])[:20]) for t in range(K)]
    return ri,ci,[len(A[i]&B[j])/len(A[i]|B[j]) for i,j in zip(ri,ci)]
def slopes(DT,order,yv):
    return [linregress(years,[DT[yv==y,order[r]].mean() for y in years]).slope*1000 for r in range(K)]
ref_sl=slopes(DTref,list(range(K)),yrs); S['ref_slopes']=[round(x,2) for x in ref_sl]
done={r['seed'] for r in S['seeds']}
for seed in [7,21,77,123,2024]:
    if seed in done: continue
    if time.time()-T0>BUDGET: save(); print('PAUSE seeds'); sys.exit()
    m=LdaModel(corpus=corpus,id2word=dic,num_topics=K,random_state=seed,**SPEC); Q=m.get_topics(); DT=theta(m,corpus)
    ri,ci,jac=match(Pref,Q); order=[None]*K
    for r,c in zip(ri,ci): order[r]=int(c)
    agree=float(np.mean([order[d]==q for d,q in zip(DTref.argmax(1),DT.argmax(1))])); sl=slopes(DT,order,yrs)
    S['seeds'].append(dict(seed=seed,jac20=[round(j,3) for j in jac],jac20_mean=round(float(np.mean(jac)),3),doc_agree=round(agree,3),
        slopes=[round(x,2) for x in sl],sign=[int(np.sign(a)==np.sign(b)) for a,b in zip(sl,ref_sl)],r=round(float(pearsonr(sl,ref_sl)[0]),3)))
    save(); print(f'seed {seed}: jac={np.mean(jac):.3f} agree={agree:.3f} r={S["seeds"][-1]["r"]}')
rng=np.random.RandomState(42); done={r['rep'] for r in S['split']}
for rep in range(5):
    idx=rng.permutation(len(corpus))[:len(corpus)//2]
    if rep in done: continue
    if time.time()-T0>BUDGET: save(); print('PAUSE split'); sys.exit()
    m=LdaModel(corpus=[corpus[i] for i in idx],id2word=dic,num_topics=K,random_state=42,**SPEC); ri,ci,jac=match(Pref,m.get_topics())
    S['split'].append(dict(rep=rep,jac20=[round(j,3) for j in jac],jac20_mean=round(float(np.mean(jac)),3))); save(); print(f'split {rep}: jac={np.mean(jac):.3f}')
if S['criterion'] is None:
    d=df[df['research_areas'].notna()].copy()
    def cat(a):
        a=str(a).lower()
        if 'psycholog' in a: return 'Psychology'
        if 'sport science' in a: return 'Sport Sciences'
        if 'educat' in a: return 'Education'
        if 'neuro' in a: return 'Neurosciences'
        if 'rehabil' in a: return 'Rehabilitation'
        if 'orthoped' in a: return 'Orthopedics'
        if 'physiol' in a: return 'Physiology'
        if 'public' in a and 'health' in a: return 'Public Health'
        if 'engineer' in a or 'computer' in a: return 'Eng/CS'
        if 'business' in a or 'econom' in a or 'management' in a: return 'Business'
        return 'Other'
    d['cat']=d['research_areas'].apply(cat); ct=pd.crosstab(d['dom'],d['cat']); chi2,p,dof,_=chi2_contingency(ct)
    n=int(ct.values.sum()); V=float(np.sqrt(chi2/(n*(min(ct.shape)-1)))); base=d['cat'].value_counts(normalize=True)
    enr={}
    for t in range(K):
        sub=d[d.dom==t]
        for c in ct.columns:
            if base[c]>0.02 and len(sub)>=50 and (sub['cat']==c).mean()/base[c]>1.4: enr[f'T{t+1}->{c}']=[round(float((sub['cat']==c).mean()*100),1),round(float(base[c]*100),1),int(len(sub))]
    S['criterion']=dict(n=n,chi2=round(float(chi2),1),dof=int(dof),p=float(p),V=round(V,3),enrichments=enr,table={str(k):{c:int(v) for c,v in row.items()} for k,row in ct.iterrows()}); save()
    print('criterion: chi2=%.1f dof=%d p=%.2g V=%.3f n=%d'%(chi2,dof,p,V,n))
sd=S['seeds']; sig=[0,2,6,7]
print('SUMMARY seeds Jac M=%.2f SD=%.2f | agree %.2f-%.2f | per-topic %s'%(np.mean([r['jac20_mean'] for r in sd]),np.std([r['jac20_mean'] for r in sd]),min(r['doc_agree'] for r in sd),max(r['doc_agree'] for r in sd),[round(float(x),2) for x in np.mean([r['jac20'] for r in sd],0)]))
print('split Jac M=%.2f | sig-trend replication %d/20 | rbar=%.2f range %.2f-%.2f'%(np.mean([r['jac20_mean'] for r in S['split']]),sum(r['sign'][i] for r in sd for i in sig),np.mean([r['r'] for r in sd]),min(r['r'] for r in sd),max(r['r'] for r in sd)))
print('VAL DONE')
