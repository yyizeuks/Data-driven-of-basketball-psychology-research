"""Dual-metric k sweep (k=4..18): C_v, UMass, held-out perplexity (90/10 split, base-2). Resumable."""
import json, os, pickle, sys, time, numpy as np
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
BUDGET=int(sys.argv[1]) if len(sys.argv)>1 else 540; T0=time.time()
docs=pickle.load(open('docs3.pkl','rb')); corpus=pickle.load(open('bow3.pkl','rb')); dic=Dictionary.load('dict3.gensim')
rng=np.random.RandomState(42); idx=rng.permutation(len(corpus)); ncut=int(len(corpus)*0.9)
tr=[corpus[i] for i in idx[:ncut]]; te=[corpus[i] for i in idx[ncut:]]
res=json.load(open('sweep.json')) if os.path.exists('sweep.json') else {}
for k in range(4,19):
    if str(k) in res: continue
    if time.time()-T0>BUDGET: print('PAUSE'); sys.exit()
    lda=LdaModel(corpus=corpus,id2word=dic,num_topics=k,random_state=42,passes=15,iterations=400,alpha='auto',eta='auto',chunksize=200)
    cv=CoherenceModel(model=lda,texts=docs,dictionary=dic,coherence='c_v').get_coherence(); um=CoherenceModel(model=lda,corpus=corpus,coherence='u_mass').get_coherence()
    ldah=LdaModel(corpus=tr,id2word=dic,num_topics=k,random_state=42,passes=15,iterations=400,alpha='auto',eta='auto',chunksize=200)
    res[str(k)]={'c_v':round(float(cv),4),'u_mass':round(float(um),4),'holdout_perp':round(float(np.exp2(-ldah.log_perplexity(te))),1)}
    json.dump(res,open('sweep.json','w'),indent=1)
print('SWEEP DONE'); print({k:(v['c_v'],v['u_mass'],v['holdout_perp']) for k,v in res.items() if k in ('4','5','6','7','8','10','12','15')})
