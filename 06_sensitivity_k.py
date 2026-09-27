import pickle, json, os, sys, time, numpy as np, pandas as pd
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import linregress
BUDGET=int(sys.argv[1]) if len(sys.argv)>1 else 500; T0=time.time()
docs=pickle.load(open('docs3.pkl','rb')); corpus=pickle.load(open('bow3.pkl','rb')); dic=Dictionary.load('dict3.gensim'); df=pd.read_pickle('merged3_topics.pkl')
Pref=np.load('phi8.npy'); DTref=np.load('doc_topic8.npy'); yrs=df['year'].values
years=[y for y in range(1991,2025) if (yrs==y).sum()>=5]
SPEC=dict(passes=25,iterations=600,alpha='auto',eta='auto',chunksize=200)
CORE=[0,2,5,7]; SIG=[0,2,6,7]
def top20(P): return [set(np.argsort(-P[t])[:20]) for t in range(len(P))]
def match(P,Q):
    D=np.array([[jensenshannon(P[i],Q[j]) for j in range(len(Q))] for i in range(len(P))])
    ri,ci=linear_sum_assignment(D); A=top20(P); B=top20(Q)
    jac=[len(A[i]&B[j])/len(A[i]|B[j]) for i,j in zip(ri,ci)]
    return ri,ci,D,jac
def slopes(DT,cols,yv):
    out=[]
    for c in cols:
        m=[DT[yv==y,c].mean() for y in years]; out.append(linregress(years,m).slope*1000)
    return out
def core_together(Q,ci_map):
    # ci_map: ref topic -> Q topic (only for matched); check core topics matched land in one cluster (4-cluster cut)
    D=np.array([[jensenshannon(Q[i],Q[j]) for j in range(len(Q))] for i in range(len(Q))])
    Z=linkage(D[np.triu_indices(len(Q),1)],'average'); lab=fcluster(Z,4,'maxclust')
    qs=[ci_map[r] for r in CORE if r in ci_map]; return len(set(lab[q] for q in qs))==1, len(qs)
ref_sl=slopes(DTref,range(8),yrs)
R=json.load(open('sens_results.json')) if os.path.exists('sens_results.json') else {}
def theta(m,cps):
    dt=np.zeros((len(cps),m.num_topics))
    for i,b in enumerate(cps):
        for t,p in m.get_document_topics(b,minimum_probability=0): dt[i,t]=p
    return dt
# ---- A. k sensitivity ----
for k in [6,7,9,10]:
    key=f'k{k}'
    if key in R: continue
    if time.time()-T0>BUDGET: print('PAUSE'); sys.exit()
    m=LdaModel(corpus=corpus,id2word=dic,num_topics=k,random_state=42,**SPEC); Q=m.get_topics(); DT=theta(m,corpus)
    ri,ci,D,jac=match(Pref,Q); cmap={int(r):int(c) for r,c in zip(ri,ci)}
    sl=slopes(DT,range(k),yrs); sign=[int(np.sign(sl[cmap[r]])==np.sign(ref_sl[r])) for r in SIG if r in cmap]
    together,nc=core_together(Q,cmap)
    unmatched_ref=[r for r in range(8) if r not in cmap]; absorbed={}
    for r in unmatched_ref: absorbed[f'T{r+1}']=f'k{k}-topic{int(np.argmin(D[r]))+1} (JS={D[r].min():.3f})'
    extra=[c for c in range(k) if c not in cmap.values()]; split={}
    for c in extra: split[f'k{k}-topic{c+1}']=f'nearest ref T{int(np.argmin(D[:,c]))+1} (JS={D[:,c].min():.3f})'
    cv=CoherenceModel(model=m,texts=docs,dictionary=dic,coherence='c_v').get_coherence()
    R[key]=dict(k=k,c_v=round(cv,4),matched=len(cmap),jac_mean=round(float(np.mean(jac)),3),jac_per_ref={f'T{int(r)+1}':round(j,2) for r,j in zip(ri,jac)},
               sig_trend_sign_agree=f'{sum(sign)}/{len(sign)}',core_in_one_cluster=together,core_matched=nc,absorbed=absorbed,extra_topics=split,
               top8={f'topic{t+1}':[w for w,_ in m.show_topic(t,8)] for t in range(k)})
    json.dump(R,open('sens_results.json','w'),indent=1); print(key,'done',R[key]['jac_mean'],R[key]['sig_trend_sign_agree'],'core:',together)
print('K-SENS DONE')
