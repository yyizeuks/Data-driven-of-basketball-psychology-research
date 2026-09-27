# Pooled screening (construct-anchored filter), deduplication, preprocessing -> docs3.pkl, bow3.pkl, dict3.gensim, merged3_final.pkl
import pandas as pd, re, json, pickle, numpy as np
from nlp_defs import get_nouns, stop
from gensim.models import Phrases
from gensim.corpora import Dictionary
import sys
UP=sys.argv[1] if len(sys.argv)>1 else 'basketball_psychology_papers.xlsx'
wos=pd.read_pickle('corpus_final.pkl')[['title','year','abstract','research_areas']].copy(); wos['db_source']='wos'
up=pd.read_excel(UP,sheet_name='basketball_psychology'); up.columns=['title','year','abstract']; up['db_source']='pm_cr'; up['research_areas']=None
up=up.dropna(subset=['title','abstract']); up['abstract']=up['abstract'].astype(str); up['title']=up['title'].astype(str)
up=up[up['abstract'].str.len()>100]; up['year']=pd.to_numeric(up['year'],errors='coerce'); up=up.dropna(subset=['year']); up['year']=up['year'].astype(int); up=up[(up.year>=1991)&(up.year<=2024)]
def eng(s):
    L=sum(c.isalpha() for c in s); A=sum(c.isalpha() and ord(c)<128 for c in s); return L>50 and A/max(L,1)>0.9
up=up[up['abstract'].map(eng)]
allr=pd.concat([wos,up],ignore_index=True)
bball=['basketball','wheelchair basketball',' nba',' wnba',' fiba','euroleague','free throw','free-throw','jump shot','jump-shot','three-point','3-point']
txt=(' '+allr['title'].str.lower()+' '+allr['abstract'].str.lower()); allr=allr[txt.apply(lambda t: any(x in t for x in bball))].copy(); n_b=len(allr)
PSY=[r'psycholog\w*',r'anxiet\w*',r'motivat\w*',r'cognit\w*',r'\bmental\b',r'emotion\w*',r'self[- ]efficacy',r'\battention\w*',r'perceptual',r'\bperception\b',r'\bmood\b',r'\bcoping\b',r'\barousal\b',r'decision[- ]making',r'\bdecisions?\b',r'attitudes?\b',r'personalit\w*',r'\bimagery\b',r'self[- ]talk',r'resilien\w*',r'burnout',r'well[- ]?being',r'leadership',r'\bcohesion\b',r'\bexpertise\b',r'\bidentity\b',r'\bengagement\b',r'\bsatisfaction\b',r'motor learning',r'\bstress(?!\s+fractur)\w*',r'\bconfidence(?!\s+interval)\b',r'mental fatigue',r'choking',r'\bfear\b',r'depress\w*',r'\bmindfulness\b']
pat=re.compile('|'.join(PSY)); txt=(allr['title'].str.lower()+' '+allr['abstract'].str.lower())
allr['psy_hits']=txt.apply(lambda t: sorted(set(m.group(0).split()[0] for m in pat.finditer(t))))
allr=allr[allr['psy_hits'].map(len)>0].copy(); n_p=len(allr)
norm=lambda s: re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',s.lower())).strip()
allr['tnorm']=allr['title'].map(norm); allr['pri']=(allr['db_source']!='wos').astype(int)
allr=allr.sort_values('pri').drop_duplicates('tnorm').drop(columns='pri').reset_index(drop=True); n_d=len(allr)
print(f'basketball={n_b} refined_psych={n_p} dedup={n_d} (wos={int((allr.db_source=="wos").sum())}, pm_cr={int((allr.db_source=="pm_cr").sum())})')
docs=[get_nouns(a) for a in allr['abstract'].tolist()]
print('avg tokens/doc', round(np.mean([len(d) for d in docs]),1))
bigram=Phrases(docs,min_count=15,threshold=12); docs=[bigram[d] for d in docs]
dic=Dictionary(docs); v0=len(dic); dic.filter_extremes(no_below=8,no_above=0.45)
corpus=[dic.doc2bow(d) for d in docs]; keep=[i for i,c in enumerate(corpus) if len(c)>=3]
corpus=[corpus[i] for i in keep]; docs=[docs[i] for i in keep]; df=allr.iloc[keep].reset_index(drop=True)
pickle.dump(docs,open('docs3.pkl','wb')); pickle.dump(corpus,open('bow3.pkl','wb')); dic.save('dict3.gensim'); df.to_pickle('merged3_final.pkl')
print(f'FINAL docs={len(df)} vocab={len(dic)} (before {v0})  avg_nouns={np.mean([len(d) for d in docs]):.1f}')
