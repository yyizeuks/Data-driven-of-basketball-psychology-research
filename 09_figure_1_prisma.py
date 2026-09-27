import json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig,ax=plt.subplots(figsize=(8.2,9.2)); ax.axis('off')
def box(x,y,w,h,text,fc='#dbe7f6',fs=8.4):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.006",fc=fc,ec='#41597a',lw=1.3,transform=ax.transAxes))
    lines=text.split('\n')
    ax.text(x+w/2,y+h-0.020,lines[0],ha='center',va='center',fontsize=fs+0.6,fontweight='bold',transform=ax.transAxes)
    rest='\n'.join(lines[1:])
    if rest: ax.text(x+w/2,y+(h-0.028)/2,rest,ha='center',va='center',fontsize=fs,transform=ax.transAxes,linespacing=1.25)
def arrow(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),transform=ax.transAxes,arrowstyle='-|>',mutation_scale=15,color='#41597a',lw=1.5))
def phase(y,label): ax.text(0.015,y,label,fontsize=9.5,fontweight='bold',color='#41597a',transform=ax.transAxes,va='top')
MX,MW=0.05,0.44; SX,SW=0.545,0.445; CX=MX+MW/2
phase(0.995,'IDENTIFICATION')
box(MX,0.880,MW,0.095,'Records identified from databases\n(n = 8,928; searches executed February 2026)\nWeb of Science Core Collection: 5,249\nPubMed/MEDLINE: 2,140 · Crossref: 1,539')
box(SX,0.700,SW,0.150,'Records removed before screening\n(n = 3,638)\nNo abstract/year: WoS 193; PubMed/Crossref 562\nPublished outside 1991–2024: 825\nNon-English abstract: 6\nDuplicate records within WoS export: 2,052',fc='#f2f2f2',fs=7.9)
phase(0.660,'SCREENING')
box(MX,0.565,MW,0.070,'Records screened\n(n = 5,290)')
box(SX,0.385,SW,0.140,'Records excluded (n = 2,376)\nNo basketball-related term: 491\nNo psychological construct: 1,885\n– WoS preliminary substring screen: 840\n– construct-anchored filter: 1,045\n(WoS 500; PubMed/Crossref 545)',fc='#f2f2f2')
box(MX,0.295,MW,0.075,'Records meeting both topical criteria\n(n = 2,914)')
box(SX,0.130,SW,0.110,'Removed before analysis (n = 556)\nCross-source duplicate records: 555\nEmpty token set after preprocessing: 1',fc='#f2f2f2')
phase(0.110,'INCLUDED')
box(MX,0.005,MW,0.085,'Articles included in the synthesis\n(n = 2,358)\nWoS stratum: 1,178 · PubMed/Crossref: 1,181\n(1991–2024)',fc='#e2f0dc')
arrow(CX,0.880,CX,0.637); arrow(CX,0.565,CX,0.372); arrow(CX,0.295,CX,0.092)
arrow(CX,0.775,SX,0.775); arrow(CX,0.455,SX,0.455); arrow(CX,0.185,SX,0.185)
plt.savefig('fig1_flow.png',dpi=300,bbox_inches='tight'); plt.close()
from PIL import Image
im=Image.open('fig1_flow.png').convert('RGB'); im.save('/mnt/user-data/outputs/tiff_rev/Figure1.tif',format='TIFF',compression='tiff_lzw',dpi=(300,300)); print('Figure1',im.size)
