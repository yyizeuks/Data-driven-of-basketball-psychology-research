"""Single-pass corpus construction from raw exports (WoS savedrecs + PubMed/Crossref sheet). Writes prisma_counts.json."""
import pandas as pd, re, json, pickle, numpy as np, sys
from nlp_defs import get_nouns
from gensim.models import Phrases
from gensim.corpora import Dictionary
WOS=sys.argv[1] if len(sys.argv)>1 else 'raw/Source_Data_2.xlsx'
UP=sys.argv[2] if len(sys.argv)>2 else 'raw/basketball_psychology_papers.xlsx'
C={}
w=pd.concat([pd.read_excel(WOS,sheet_name='savedrecs'),pd.read_excel(WOS,sheet_name='savedrecs (2)')],ignore_index=True); C['wos_identified']=len(w)
w=w[['Article Title','Abstract','Publication Year','Research Areas']].copy(); w.columns=['title','abstract','year','research_areas']
w=w.dropna(subset=['abstract','title']); w['title']=w['title'].astype(str); w['abstract']=w['abstract'].astype(str)
w['year']=pd.to_numeric(w['year'],errors='coerce'); w=w.dropna(subset=['year']); w['year']=w['year'].astype(int); C['wos_no_abstract_or_year']=C['wos_identified']-len(w)
w['title_key']=w['title'].str.lower().str.replace(r'\s+',' ',regex=True).str.strip(); n=len(w); w=w.drop_duplicates('title_key'); C['wos_within_duplicates']=n-len(w)
n=len(w); w=w[(w.year>=1991)&(w.year<=2024)]; C['wos_out_of_window']=n-len(w); w['db_source']='wos'
u=pd.read_excel(UP,sheet_name='basketball_psychology'); C['pmcr_identified']=len(u); u.columns=['title','year','abstract']
u=u.dropna(subset=['title','abstract']); u['abstract']=u['abstract'].astype(str); u['title']=u['title'].astype(str); u=u[u['abstract'].str.len()>100]
u['year']=pd.to_numeric(u['year'],errors='coerce'); u=u.dropna(subset=['year']); u['year']=u['year'].astype(int); C['pmcr_no_abstract_or_year']=C['pmcr_identified']-len(u)
n=len(u); u=u[(u.year>=1991)&(u.year<=2024)]; C['pmcr_out_of_window']=n-len(u)
def eng(s):
    L=sum(c.isalpha() for c in s); A=sum(c.isalpha() and ord(c)<128 for c in s); return L>50 and A/max(L,1)>0.9
n=len(u); u=u[u['abstract'].map(eng)]; C['pmcr_non_english']=n-len(u); u['db_source']='pm_cr'; u['research_areas']=None
allr=pd.concat([w.drop(columns='title_key'),u],ignore_index=True); C['screened']=len(allr)
bball=['basketball','wheelchair basketball',' nba',' wnba',' fiba','euroleague','free throw','free-throw','jump shot','jump-shot','three-point','3-point']
txt=(' '+allr['title'].str.lower()+' '+allr['abstract'].str.lower()); allr=allr[txt.apply(lambda t: any(x in t for x in bball))].copy(); C['after_basketball']=len(allr); C['excluded_no_basketball']=C['screened']-len(allr)
PSY=[l.strip() for l in open('data_constructs.txt') if l.strip() and not l.startswith('#')]
pat=re.compile('|'.join(PSY)); txt=(allr['title'].str.lower()+' '+allr['abstract'].str.lower())
allr['psy_hits']=txt.apply(lambda t: sorted(set(m.group(0).split()[0] for m in pat.finditer(t))))
allr=allr[allr['psy_hits'].map(len)>0].copy(); C['after_construct']=len(allr); C['excluded_no_construct']=C['after_basketball']-len(allr)
norm=lambda s: re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',s.lower())).strip()
allr['tnorm']=allr['title'].map(norm); allr['pri']=(allr['db_source']!='wos').astype(int)
n=len(allr); allr=allr.sort_values('pri').drop_duplicates('tnorm').drop(columns='pri').reset_index(drop=True); C['cross_source_duplicates']=n-len(allr); C['after_dedup']=len(allr)
C['after_dedup_wos']=int((allr.db_source=='wos').sum()); C['after_dedup_pmcr']=int((allr.db_source=='pm_cr').sum())
docs=[get_nouns(a) for a in allr['abstract'].tolist()]
bigram=Phrases(docs,min_count=15,threshold=12); docs=[bigram[d] for d in docs]
dic=Dictionary(docs); C['vocab_before']=len(dic); dic.filter_extremes(no_below=8,no_above=0.45)
corpus=[dic.doc2bow(d) for d in docs]; keep=[i for i,c in enumerate(corpus) if len(c)>=3]
C['empty_after_preprocessing']=len(corpus)-len(keep); corpus=[corpus[i] for i in keep]; docs=[docs[i] for i in keep]; df=allr.iloc[keep].reset_index(drop=True)
C['final']=len(df); C['final_wos']=int((df.db_source=='wos').sum()); C['final_pmcr']=int((df.db_source=='pm_cr').sum()); C['vocab']=len(dic); C['avg_nouns']=round(float(np.mean([len(d) for d in docs])),1)
out=sys.argv[3] if len(sys.argv)>3 else 'clean'
pickle.dump(docs,open(f'docs_{out}.pkl','wb')); pickle.dump(corpus,open(f'bow_{out}.pkl','wb')); dic.save(f'dict_{out}.gensim'); df.to_pickle(f'corpus_{out}.pkl')
json.dump(C,open(f'prisma_counts_{out}.json','w'),indent=1); print(json.dumps(C))
