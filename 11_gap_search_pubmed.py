"""Reproducible gap search (Supplementary S0): PubMed E-utilities, title/abstract fields. Prints count and records."""
import json, urllib.request, urllib.parse
TERM='basketball[tiab] AND (bibliometric*[tiab] OR scientometric*[tiab] OR "topic model*"[tiab] OR "text mining"[tiab])'
B='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
r=json.load(urllib.request.urlopen(B+'esearch.fcgi?db=pubmed&retmax=200&retmode=json&term='+urllib.parse.quote(TERM)))['esearchresult']
print('count:',r['count']); ids=','.join(r['idlist'])
if ids:
    s=json.load(urllib.request.urlopen(B+'esummary.fcgi?db=pubmed&retmode=json&id='+ids))['result']
    for i in s['uids']:
        e=s[i]; doi=[x['value'] for x in e.get('articleids',[]) if x['idtype']=='doi']
        print(e.get('pubdate','')[:4],'|',e['title'],'|',e.get('source',''),'|',doi[0] if doi else '')
