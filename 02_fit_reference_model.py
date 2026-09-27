import pickle, json, numpy as np, pandas as pd
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
docs=pickle.load(open('docs3.pkl','rb')); corpus=pickle.load(open('bow3.pkl','rb')); dic=Dictionary.load('dict3.gensim'); df=pd.read_pickle('merged3_final.pkl')
K=8
lda=LdaModel(corpus=corpus,id2word=dic,num_topics=K,random_state=42,passes=25,iterations=600,alpha='auto',eta='auto',chunksize=200)
lda.save('lda8.gensim')
cv=CoherenceModel(model=lda,texts=docs,dictionary=dic,coherence='c_v').get_coherence()
phi=lda.get_topics(); np.save('phi8.npy',phi)
dt=np.zeros((len(corpus),K))
for i,b in enumerate(corpus):
    for t,p in lda.get_document_topics(b,minimum_probability=0): dt[i,t]=p
np.save('doc_topic8.npy',dt); df['dom']=dt.argmax(1); df.to_pickle('merged3_topics.pkl')
tt={str(t):[[w,round(float(p),4)] for w,p in lda.show_topic(t,20)] for t in range(K)}; json.dump(tt,open('topic_terms8.json','w'))
print('C_v = %.4f'%cv)
for t in range(K): print(f'T{t+1} {dt[:,t].mean()*100:5.1f}% :', ', '.join(w for w,_ in lda.show_topic(t,8)))
