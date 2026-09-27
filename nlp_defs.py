import pandas as pd, numpy as np, re, json, pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from gensim.models import Phrases
from gensim.corpora import Dictionary


lem = WordNetLemmatizer()
tok = RegexpTokenizer(r'[a-zA-Z][a-zA-Z\-]+')
base_stop=set(stopwords.words('english'))
domain_stop=set('''study studies result results research method methods analysis aim aims
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
football soccer volleyball handball baseball hockey rugby tennis golf swimming cricket badminton
background objective objectives setting context interval intervals'''.split())
stop=base_stop|domain_stop
def get_nouns(text):
    text=re.sub(r'\(c\)\s*\d{4}.*$','',str(text))
    text=re.sub(r'\b(BACKGROUND|OBJECTIVES?|METHODS?|RESULTS?|CONCLUSIONS?|PURPOSE|DESIGN|SETTING|PARTICIPANTS|INTERVENTIONS?|CONTEXT|AIMS?|HYPOTHESIS|SIGNIFICANCE|MAIN OUTCOME MEASURES?)\s*:?','',text,flags=re.I)
    text=re.sub(r'\b\d+[\d\.,%]*\b',' ',text)
    toks=[w.lower() for w in tok.tokenize(text)]
    toks=[w for w in toks if len(w)>2]
    tagged=pos_tag(toks)
    nouns=[lem.lemmatize(w) for w,t in tagged if t in ('NN','NNS','NNP','NNPS')]
    return [w for w in nouns if w not in stop and len(w)>2 and not w.startswith('-') and not w.endswith('-')]
