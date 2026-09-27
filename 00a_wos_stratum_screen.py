# Original single-database (WoS) screening pipeline, executed first; produces corpus_filtered.pkl
import pandas as pd, numpy as np, re, json
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

# ---------- Load & combine ----------
df1 = pd.read_excel('raw/Source_Data_2.xlsx', sheet_name='savedrecs')
df2 = pd.read_excel('raw/Source_Data_2.xlsx', sheet_name='savedrecs (2)')
df = pd.concat([df1, df2], ignore_index=True)
df = df[['Authors','Article Title','Source Title','Abstract','Publication Year','Research Areas']].copy()
df.columns = ['authors','title','source','abstract','year','research_areas']
df = df.dropna(subset=['abstract','title'])
df['title']=df['title'].astype(str); df['abstract']=df['abstract'].astype(str)
df['year']=pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year']); df['year']=df['year'].astype(int)
n_raw=len(df)

# Dedup
df['title_key']=df['title'].str.lower().str.replace(r'\s+',' ',regex=True).str.strip()
df=df.drop_duplicates(subset='title_key')
n_dedup=len(df)

# ---------- Basketball + psychology filter ----------
text=(df['title']+' '+df['abstract']).str.lower()
bball=['basketball','wheelchair basketball',' nba',' wnba',' fiba','euroleague',
       'free throw','free-throw','jump shot','jump-shot','three-point','3-point']
bmask=text.apply(lambda t: any(x in t for x in bball))
df_b=df[bmask].copy()
n_bball=len(df_b)

psy=['psycholog','anxiety','motivat','cognit','mental','stress','emotion','confidence',
     'self-efficacy','efficacy','attention','perception','perceptual','behav','mood','coping',
     'arousal','flow','decision','attitude','personality','imagery','self-talk','resilien',
     'burnout','wellbeing','well-being','leadership','cohesion','attribution','expertise',
     'learning','focus','concentration','social','identity','satisfaction','engagement']
ptext=(df_b['title']+' '+df_b['abstract']).str.lower()
pmask=ptext.apply(lambda t: any(x in t for x in psy))
df_f=df_b[pmask].copy().reset_index(drop=True)
n_final=len(df_f)

print(f'raw={n_raw}  dedup={n_dedup}  basketball={n_bball}  final(basketball+psych)={n_final}')
print('year range:', df_f.year.min(),'-',df_f.year.max())

# Restrict to years with stable volume (>=1995) but report full
df_f.to_pickle('corpus_filtered.pkl')
with open('corpus_counts.json','w') as f:
    json.dump(dict(raw=n_raw,dedup=n_dedup,bball=n_bball,final=n_final,
                   ymin=int(df_f.year.min()),ymax=int(df_f.year.max())), f)

# Top sources
print('\nTop journals:')
print(df_f['source'].value_counts().head(12).to_string())
