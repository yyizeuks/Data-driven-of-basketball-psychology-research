# Original single-database (WoS) NLP step; produces corpus_final.pkl (WoS stratum input to 01)
import pandas as pd, numpy as np, re, json, pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from gensim.models import Phrases
from gensim.corpora import Dictionary

df = pd.read_pickle('corpus_filtered.pkl')
lem = WordNetLemmatizer()
tok = RegexpTokenizer(r'[a-zA-Z][a-zA-Z\-]+')

base_stop = set(stopwords.words('english'))
domain_stop = set('''study studies result results research method methods analysis aim aims
purpose using used use paper article finding findings show showed shown suggest suggests
conclusion conclusions however therefore thus group groups participant participants subject
subjects data sample samples test tests measure measures effect effects difference differences
significant significance level levels score scores variable variables factor factors model
models present current investigate investigated examine examined evaluate evaluated assess
assessed determine determined compare compared associate associated relationship relationships
association correlation between among within across number total mean average standard deviation
year years age aged old male female men women boy boys girl girls player players athlete
athletes team teams game games sport sports basketball player season match competition
one two three four five first second high higher low lower large small good better best
increase increased decrease decreased change changed difference role aspect aspects type
context approach process processes condition conditions response responses outcome outcomes
ability abilities task tasks performance time times day days week weeks month months hour
question questionnaire survey item items scale subscale dimension version reliability validity
n p value values ratio rate percent group ci sd anova regression p< respectively
moreover furthermore additionally finally overall well also may might could would
example instance particular specific general different various several many much more most
less least new finding objective field area domain aim importance important'''.split())
stop = base_stop | domain_stop

def get_nouns(text):
    text = re.sub(r'\(c\)\s*\d{4}.*$','',text)         # remove copyright
    text = re.sub(r'\b\d+[\d\.,%]*\b',' ',text)         # numbers
    toks=[w.lower() for w in tok.tokenize(text)]
    toks=[w for w in toks if len(w)>2]
    tagged=pos_tag(toks)
    nouns=[lem.lemmatize(w) for w,t in tagged if t in ('NN','NNS','NNP','NNPS')]
    nouns=[w for w in nouns if w not in stop and len(w)>2 and not w.startswith('-') and not w.endswith('-')]
    return nouns

print('Extracting nouns...')
docs=[get_nouns(a) for a in df['abstract'].tolist()]
print('done. avg tokens/doc:', np.mean([len(d) for d in docs]).round(1))

# Bigrams (collocations) – capture e.g. self_efficacy, free_throw, jump_shot
bigram=Phrases(docs, min_count=15, threshold=12)
docs=[bigram[d] for d in docs]

# Build dictionary, filter extremes
dic=Dictionary(docs)
print('vocab before filter:', len(dic))
dic.filter_extremes(no_below=8, no_above=0.45)
print('vocab after filter:', len(dic))
corpus=[dic.doc2bow(d) for d in docs]

# Drop empty docs
keep=[i for i,c in enumerate(corpus) if len(c)>=3]
corpus=[corpus[i] for i in keep]
docs=[docs[i] for i in keep]
df=df.iloc[keep].reset_index(drop=True)
print('docs after empty removal:', len(corpus))

pickle.dump(docs, open('docs.pkl','wb'))
pickle.dump(corpus, open('corpus_bow.pkl','wb'))
dic.save('dict.gensim')
df.to_pickle('corpus_final.pkl')
print('Saved NLP artifacts.')
