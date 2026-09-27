import pickle, json, os, sys, time, numpy as np, pandas as pd
from gensim.models import LdaModel, Phrases, TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress
from sklearn.decomposition import NMF
from nlp_defs import get_nouns
docs=pickle.load(open('docs3.pkl','rb')); corpus=pickle.load(open('bow3.pkl','rb')); dic=Dictionary.load('dict3.gensim'); df=pd.read_pickle('merged3_topics.pkl')
Pref=np.load('phi8.npy'); DTref=np.load('doc_topic8.npy'); yrs=df['year'].values; V=len(dic)
SPEC=dict(passes=25,iterations=600,alpha='auto',eta='auto',chunksize=200); SIG=[0,2,6,7]; ROB=[0,7]
def top20(P): return [set(np.argsort(-P[t])[:20]) for t in range(len(P))]
def jacmat(P,Q):
    A=top20(P);B=top20(Q); return np.array([[len(a&b)/len(a|b) for b in B] for a in A])
def slopes(DT,yv):
    years=[y for y in range(1991,2025) if (yv==y).sum()>=5]
    return [linregress(years,[DT[yv==y,c].mean() for y in years]).slope*1000 for c in range(DT.shape[1])]
def theta(m,cps):
    dt=np.zeros((len(cps),m.num_topics))
    for i,b in enumerate(cps):
        for t,p in m.get_document_topics(b,minimum_probability=0): dt[i,t]=p
    return dt
ref_sl=slopes(DTref,yrs); R=json.load(open('sens2_results.json')) if os.path.exists('sens2_results.json') else {}
def save(): json.dump(R,open('sens2_results.json','w'),indent=1)
# ---- NMF ----
if 'nmf' not in R:
    tfidf=TfidfModel(corpus); X=corpus2csc(tfidf[corpus],num_terms=V).T.tocsr()
    nmf=NMF(n_components=8,init='nndsvda',random_state=42,max_iter=600); W=nmf.fit_transform(X); H=nmf.components_
    Hn=H/H.sum(1,keepdims=True); Wn=W/np.clip(W.sum(1,keepdims=True),1e-12,None)
    J=jacmat(Pref,Hn); ri,ci=linear_sum_assignment(-J); cmap={int(r):int(c) for r,c in zip(ri,ci)}
    sl=slopes(Wn,yrs); sign4=[int(np.sign(sl[cmap[r]])==np.sign(ref_sl[r])) for r in SIG]; sign2=[int(np.sign(sl[cmap[r]])==np.sign(ref_sl[r])) for r in ROB]
    dom_agree=float(np.mean([cmap[d]==w for d,w in zip(DTref.argmax(1),Wn.argmax(1))]))
    R['nmf']=dict(jac_mean=round(float(J[ri,ci].mean()),3),jac_per_ref={f'T{r+1}':round(float(J[r,c]),2) for r,c in zip(ri,ci)},
        sig4_sign=f'{sum(sign4)}/4',robust2_sign=f'{sum(sign2)}/2',dominant_agree=round(dom_agree,3),
        nmf_top10={f'T{r+1}->nmf{c+1}':[dic[i] for i in np.argsort(-Hn[c])[:10]] for r,c in zip(ri,ci)},
        nmf_slopes_matched={f'T{r+1}':round(sl[c],2) for r,c in zip(ri,ci)})
    save(); print('NMF:',R['nmf']['jac_mean'],R['nmf']['sig4_sign'],R['nmf']['robust2_sign'],'domAgree',R['nmf']['dominant_agree'])
# ---- strict-construct sensitivity ----
if 'strict' not in R:
    def strict_ok(h):
        gen=lambda s: s.startswith('attention') or s.startswith('stress') or s in ('decision','decisions') or s.startswith('confiden') or s.startswith('percept')
        return any(not gen(s) for s in h)
    keep=[i for i,h in enumerate(df['psy_hits']) if strict_ok(h)]
    cps=[corpus[i] for i in keep]; yv=yrs[keep]
    m=LdaModel(corpus=cps,id2word=dic,num_topics=8,random_state=42,**SPEC); Q=m.get_topics(); DT=theta(m,cps)
    D=np.array([[jensenshannon(Pref[i],Q[j]) for j in range(8)] for i in range(8)]); ri,ci=linear_sum_assignment(D); cmap={int(r):int(c) for r,c in zip(ri,ci)}
    J=jacmat(Pref,Q); sl=slopes(DT,yv)
    R['strict']=dict(n_docs=len(keep),removed=len(corpus)-len(keep),jac_mean=round(float(np.mean([J[r,c] for r,c in zip(ri,ci)])),3),
        jac_per_ref={f'T{r+1}':round(float(J[r,c]),2) for r,c in zip(ri,ci)},
        sig4_sign=f"{sum(int(np.sign(sl[cmap[r]])==np.sign(ref_sl[r])) for r in SIG)}/4",robust2_sign=f"{sum(int(np.sign(sl[cmap[r]])==np.sign(ref_sl[r])) for r in ROB)}/2")
    save(); print('STRICT:',R['strict'])
# ---- OpenAlex expansion sensitivity (fixed vocabulary) ----
if 'openalex' not in R:
    oa=pd.read_pickle('oa_unique_articles.pkl')  # produced by 07a from openalex_export.csv; od=[get_nouns(a) for a in oa['abstract'].tolist()]
    bigram=Phrases(docs,min_count=15,threshold=12); od=[bigram[d] for d in od]
    ob=[dic.doc2bow(d) for d in od]; ok=[i for i,b in enumerate(ob) if len(b)>=3]
    cps=corpus+[ob[i] for i in ok]; yv=np.concatenate([yrs,oa['year'].values[ok]])
    m=LdaModel(corpus=cps,id2word=dic,num_topics=8,random_state=42,**SPEC); Q=m.get_topics(); DT=theta(m,cps)
    D=np.array([[jensenshannon(Pref[i],Q[j]) for j in range(8)] for i in range(8)]); ri,ci=linear_sum_assignment(D); cmap={int(r):int(c) for r,c in zip(ri,ci)}
    J=jacmat(Pref,Q); sl=slopes(DT,yv); sl_orig=slopes(DT[:len(corpus)],yrs)
    R['openalex']=dict(added_docs=len(ok),total=len(cps),jac_mean=round(float(np.mean([J[r,c] for r,c in zip(ri,ci)])),3),
        jac_per_ref={f'T{r+1}':round(float(J[r,c]),2) for r,c in zip(ri,ci)},
        sig4_sign_expanded=f"{sum(int(np.sign(sl[cmap[r]])==np.sign(ref_sl[r])) for r in SIG)}/4",
        robust2_sign_expanded=f"{sum(int(np.sign(sl[cmap[r]])==np.sign(ref_sl[r])) for r in ROB)}/2",
        sig4_sign_on_original_docs=f"{sum(int(np.sign(sl_orig[cmap[r]])==np.sign(ref_sl[r])) for r in SIG)}/4",
        avg_tokens_oa=round(float(np.mean([len(d) for d in od])),1))
    save(); print('OPENALEX:',R['openalex'])
print('SENS2 DONE')
