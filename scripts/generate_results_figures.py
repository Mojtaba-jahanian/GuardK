from pathlib import Path
import re,json,io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parents[1]; e=p/'evidence'; out=p/'figures/results_g13';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.titleweight':'bold'})
colors=['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#777777']
manifest=[]
def save(fig,name,sources):
 buf=io.BytesIO();fig.savefig(buf,format='pdf',bbox_inches='tight');(out/(name+'.pdf')).write_bytes(buf.getvalue());buf=io.BytesIO();fig.savefig(buf,format='png',dpi=300,bbox_inches='tight');(out/(name+'.png')).write_bytes(buf.getvalue());plt.close(fig);manifest.append({'figure':name,'sources':sources,'note':'Existing summaries only; no new fitting or invented run-level uncertainty.'})
a=pd.read_csv(e/'results__g9_confirmatory__G9_confirmatory_aggregate.csv');ci=pd.read_csv(e/'results__g9_confirmatory__G9_confirmatory_bootstrap_CI.csv')
methods=['guardk_v04','guardk_v02','kmeans_silhouette','agglomerative_silhouette','gmm_bic','hdbscan'];labels=['GuardK v0.4','GuardK v0.2','K-means + silhouette','Ward + silhouette','GMM + BIC','HDBSCAN'];a=a.set_index('method').loc[methods]
fig,axs=plt.subplots(2,2,figsize=(10,6.6),layout='constrained')
for ax,col,cimet,title in zip(axs.flat,['coverage','exact_k_rate_all','exact_k_rate_accepted','mean_ARI_accepted'],['coverage','exact_k_rate_all','exact_k_rate_accepted','ARI_accepted'],['(a) Coverage','(b) Exact K: all cases','(c) Exact K: accepted outputs','(d) ARI: accepted outputs']):
 v=a[col].to_numpy();c=ci[ci.metric==cimet].set_index('method').loc[methods];ax.barh(labels[::-1],v[::-1],color=colors[::-1]);ax.errorbar(v[::-1],np.arange(6),xerr=np.array([v-c.ci95_low,c.ci95_high-v])[:,::-1],fmt='none',ecolor='black',capsize=2,lw=.8);ax.set(xlim=(0,1.06),title=title);ax.grid(axis='x',alpha=.18)
save(fig,'fig_confirmatory',['G9_confirmatory_aggregate.csv','G9_confirmatory_bootstrap_CI.csv'])
fig,axs=plt.subplots(2,3,figsize=(10,6),layout='constrained')
for row,(file,group,levels) in enumerate([('results__g9_confirmatory__G9_by_overlap.csv','overlap',['low','medium','high']),('results__g9_confirmatory__G9_by_ratio.csv','ratio',[1,5,20,50])]):
 d=pd.read_csv(e/file)
 if group not in d.columns:group=next(c for c in d.columns if 'ratio' in c)
 for ax,col,title in zip(axs[row],['coverage','exact_k_rate_accepted','mean_ARI_accepted'],['Coverage','Exact K: accepted','ARI: accepted']):
  for m,l,c in zip(methods[:4],labels[:4],colors[:4]):
   s=d[d.method==m].set_index(group).reindex(levels);ax.plot(range(len(levels)),s[col],'-o',label=l,color=c,ms=4)
  ax.set(xticks=range(len(levels)),xticklabels=levels,ylim=(-.03,1.04),title=title,xlabel='Overlap' if row==0 else 'Imbalance ratio');ax.grid(alpha=.15)
axs[0,0].legend(fontsize=7);save(fig,'fig_regimes',['G9_by_overlap.csv','G9_by_ratio.csv'])
d=pd.read_csv(e/'frozen_modern_comparators__G9F_FINAL_MODERN_COMMON_DATA.csv'); classical=pd.read_csv(e/'results__g7_standard_real__G7_standard_real_aggregate.csv');print('real columns',classical.columns.tolist())
fig,axs=plt.subplots(1,3,figsize=(11,3.5),layout='constrained');ds=['breast_cancer_wisconsin','iris','wine','digits'];ms=['GuardK_v04','DeepDPM','UNSEEN_DEC']
for ax,col,title in zip(axs,['coverage','exact_k_rate_all','mean_ARI_accepted'],['(a) Coverage','(b) Exact K: all runs','(c) ARI: accepted runs']):
 vals=d.pivot(index='dataset',columns='method',values=col).reindex(index=ds,columns=ms).to_numpy();im=ax.imshow(np.ma.masked_invalid(vals),vmin=0,vmax=1,cmap='Blues',aspect='auto')
 for i in range(4):
  for j in range(3):ax.text(j,i,'Abstain' if np.isnan(vals[i,j]) else f'{vals[i,j]:.2f}',ha='center',va='center',color='white' if vals[i,j]>.65 else 'black',fontsize=8)
 ax.set(xticks=range(3),xticklabels=['GuardK','DeepDPM','UNSEEN'],yticks=range(4),yticklabels=['Breast Cancer','Iris','Wine','Digits'],title=title)
save(fig,'fig_real',['G9F_FINAL_MODERN_COMMON_DATA.csv'])
# Extract G11 means from the manuscript table; source precision is retained.
s=(p/'tables/table_g11_scalability.tex').read_text();rows=[]
for line in s.splitlines():
 if line.startswith(('GuardK','K-means')):
  bits=line.replace('\\\\','').split('&');rows.append([bits[0].strip(),int(bits[1].replace(',','')),*[float(x) for x in bits[2:]]])
g=pd.DataFrame(rows,columns=['method','n','coverage','exact','ari','runtime','rss']);g.to_csv(out/'g11_scalability_transcribed.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
for ax,col,title in zip(axs,['runtime','rss'],['(a) Wall-clock runtime (s)','(b) Peak resident memory (MB)']):
 for (m,v),c in zip(g.groupby('method',sort=False),colors):ax.plot(v.n,v[col],'-o',label=m,color=c)
 ax.set(xscale='log',xlabel='Observations',title=title);ax.grid(alpha=.2)
axs[0].set_yscale('log');axs[1].legend(fontsize=7);save(fig,'fig_scalability',['table_g11_scalability.tex; rounded three-seed means'])
s=(p/'appendices/appendix_g_g11_stress.tex').read_text();rows=[]
for l in s.splitlines():
 if re.match(r'^8 & (low|medium)',l):
  b=l.replace('\\\\','').split('&');rows.append([b[1].strip(),int(b[2]),float(b[3]),float(b[4])])
fig,axs=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
for ov,c in zip(['low','medium'],colors):
 r=[x for x in rows if x[0]==ov];axs[0].plot([x[1] for x in r],[x[2] for x in r],'-o',color=c,label=ov);axs[0].plot([x[1] for x in r],[x[3] for x in r],'--x',color=c)
axs[0].set(title='(a) K*=8 candidate-range stress',xlabel='Maximum candidate K',ylabel='Rate',ylim=(-.04,1.03),xticks=[8,12,20]);axs[0].legend(title='Solid: coverage\nDashed: exact K (all)',fontsize=7)
axs[1].bar(['K=3','K=4','K=6','K=7','K=8'],[9,2,6,1,0],color=colors[:5]);axs[1].set(title='(b) Base proposals in 18 cases',ylabel='Cases')
axs[2].bar(['Abstained','Accepted\nwrong K','Accepted\ncorrect K'],[10,8,0],color=['#777777','#D55E00','#009E73']);axs[2].set(title='(c) Outer-guard decisions',ylabel='Cases')
for ax in axs:ax.grid(axis='y',alpha=.15)
save(fig,'fig_failure',['appendix_g_g11_stress.tex','sections/06_results.tex; G11B diagnostic counts'])
(out/'figure_provenance.json').write_text(json.dumps(manifest,indent=2))
