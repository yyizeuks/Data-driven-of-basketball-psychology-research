"""Screen an OpenAlex export (raw/openalex_export.csv; columns Title, Publication Year, Abstract, Type, DOI, OpenAlex ID)
with the same filters as the main corpus and identify journal articles not already in the corpus -> oa_unique_articles.pkl"""
import pandas as pd, re, json, sys
SRC=sys.argv[1] if len(sys.argv)>1 else 'raw/openalex_export.csv'
sc=pd.read_csv(SRC,low_memory=False).rename(columns={'Title':'title','Publication Year':'year','Abstract':'abstract'})
n0=len(sc); sc=sc.dropna(subset=['title','abstract']); sc['abstract']=sc['abstract'].astype(str); sc['title']=sc['title'].astype(str)
sc=sc[sc['abstract'].str.len()>100]; n_abs=len(sc)
sc['year']=pd.to_numeric(sc['year'],errors='coerce'); sc=sc.dropna(subset=['year']); sc['year']=sc['year'].astype(int); sc=sc[(sc.year>=1991)&(sc.year<=2024)]; n_yr=len(sc)
def eng(s):
    L=sum(c.isalpha() for c in s); A=sum(c.isalpha() and ord(c)<128 for c in s); return L>50 and A/max(L,1)>0.9
sc=sc[sc['abstract'].map(eng)]; n_eng=len(sc)
bball=[l.rstrip('\n') for l in open('data/basketball_terms.txt') if l.strip()]
txt=(' '+sc['title'].str.lower()+' '+sc['abstract'].str.lower()); sc=sc[txt.apply(lambda t: any(x in t for x in bball))]; n_b=len(sc)
PSY=[l.strip() for l in open('data/construct_patterns.txt') if l.strip() and not l.startswith('#')]
pat=re.compile('|'.join(PSY)); txt=(sc['title'].str.lower()+' '+sc['abstract'].str.lower()); sc=sc[txt.apply(lambda t: bool(pat.search(t)))]; n_p=len(sc)
norm=lambda s: re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',s.lower())).strip()
sc['tnorm']=sc['title'].map(norm); sc=sc.drop_duplicates('tnorm'); n_d=len(sc)
ref=pd.read_pickle('merged3_final.pkl'); ref['tnorm']=ref['title'].map(norm)
inref=sc['tnorm'].isin(set(ref['tnorm'])); p50=set(t[:50] for t in ref['tnorm']); fuzzy=sc['tnorm'].str[:50].isin(p50)
uniq=sc[~inref & ~fuzzy]; art=uniq[uniq['Type']=='article'] if 'Type' in uniq.columns else uniq
cnt=dict(openalex_raw=n0,with_abstract=n_abs,year_1991_2024=n_yr,english=n_eng,basketball=n_b,refined_psych=n_p,dedup=n_d,overlap_with_corpus=int((inref|fuzzy).sum()),
         unique_all_types=int(len(uniq)),unique_journal_articles=int(len(art)),corpus_covered_pct=round(100*ref['tnorm'].isin(set(sc['tnorm'])).mean(),1))
json.dump(cnt,open('openalex_counts.json','w'),indent=1); art.to_pickle('oa_unique_articles.pkl'); print(json.dumps(cnt,indent=1))
