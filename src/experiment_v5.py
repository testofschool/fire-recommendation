#!/usr/bin/env python3
"""
FIRE v5 — Final production. All audit issues resolved.
  [v4.1] Full-catalog evaluation (no candidate sampling — 3883 items is feasible)
  [v4.2] Eval users: random sample, not first-N
  [v4.3] Figure 3 FIRE label: offset left to avoid clipping
  [v4.4] Figure 4 caption: states seed-specific
  [v4.5] LLTM metric: squared correlation, not R²
  [v4.6] README auto-generated from results (single source of truth)
  [v4.7] Table 1 AUC caption: no "shared candidates"
  [v4.8] "optimal" → "standard local" for Fisher Info
  [v4.9] GitHub URL placeholder until repo is public
  [v4.10] Correct script name in README
"""
import argparse, os, json, time, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import sem
from sklearn.metrics import roc_auc_score
from collections import defaultdict
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size':9,'font.family':'serif','figure.dpi':150,
    'axes.grid':True,'grid.alpha':0.3,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42})

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir',default='data/ml-1m')
    p.add_argument('--out_dir',default='figures')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,123,789])
    p.add_argument('--N0',type=int,default=20); p.add_argument('--M0',type=int,default=10)
    p.add_argument('--n_eval_users',type=int,default=0,help='0=all eligible')
    return p.parse_args()

def load_movielens(d):
    r=pd.read_csv(f'{d}/ratings.dat',sep='::',engine='python',names=['uid','iid','rating','ts'],encoding='latin-1')
    m=pd.read_csv(f'{d}/movies.dat',sep='::',engine='python',names=['iid','title','genres'],encoding='latin-1')
    nu,ni=int(r['uid'].max()),int(r['iid'].max())
    valid_items=set(int(x)-1 for x in m['iid'].values)
    ag=sorted(set(g for gs in m['genres'] for g in gs.split('|')))
    gm=np.zeros((ni,len(ag))); yrs=np.full(ni,0.5)
    for _,row in m.iterrows():
        idx=int(row['iid'])-1
        for g in row['genres'].split('|'): gm[idx,ag.index(g)]=1
        try: yrs[idx]=(int(row['title'].strip()[-5:-1])-1920)/80
        except: pass
    r['u']=r['uid']-1; r['i']=r['iid']-1; r['b']=(r['rating']>=4).astype(float)
    return r,m,gm,ag,yrs,nu,ni,valid_items

def build_features(train,gm,yrs,ni,ag):
    ist=train.groupby('i').agg(ar=('rating','mean'),nr=('rating','count'),sr=('rating','std')).reindex(range(ni),fill_value=0)
    ist['sr']=ist['sr'].fillna(0)
    return np.hstack([gm,np.column_stack([np.log1p(ist['nr'].values)/8,ist['ar'].values/5,ist['sr'].values/2,yrs])]),ag+['log_pop','avg_rating','rating_std','year']

def temporal_split(r):
    r=r.sort_values(['uid','ts']); tp,ep=[],[]
    for _,g in r.groupby('uid'): s=int(len(g)*.8); tp.append(g.iloc[:s]); ep.append(g.iloc[s:])
    return pd.concat(tp),pd.concat(ep)

class IRT1D:
    def __init__(s,nu,ni): s.th=np.random.normal(0,.3,nu);s.b=np.random.normal(0,.3,ni);s.a=np.ones(ni)
    def fit(s,u,i,r,ep=20):
        for _ in range(ep):
            for x in np.random.permutation(len(u))[:100000]:
                uu,ii,rr=u[x],i[x],r[x];p=np.clip(expit(s.a[ii]*(s.th[uu]-s.b[ii])),1e-7,1-1e-7);e=rr-p
                s.th[uu]+=.01*e*s.a[ii];s.b[ii]-=.01*e*s.a[ii];s.a[ii]=np.clip(s.a[ii]+.01*e*(s.th[uu]-s.b[ii]),.1,5)
    def predict(s,u,i): return expit(s.a[i]*(s.th[u]-s.b[i]))
    def recommend(s,u,c,k=10): c=np.array(c);return c[np.argsort(s.predict(np.full(len(c),u),c))[::-1][:k]].tolist()

class SVD50:
    def __init__(s,nu,ni,k=50): s.U=np.random.normal(0,.1,(nu,k));s.V=np.random.normal(0,.1,(ni,k));s.ub=np.zeros(nu);s.ib=np.zeros(ni);s.mu=0
    def fit(s,u,i,r,ep=15):
        s.mu=float(np.mean(r))
        for _ in range(ep):
            for x in np.random.permutation(len(u))[:100000]:
                uu,ii,rr=u[x],i[x],r[x];e=rr-(s.mu+s.ub[uu]+s.ib[ii]+s.U[uu]@s.V[ii])
                s.U[uu]+=.005*(e*s.V[ii]-.02*s.U[uu]);s.V[ii]+=.005*(e*s.U[uu]-.02*s.V[ii])
                s.ub[uu]+=.005*(e-.02*s.ub[uu]);s.ib[ii]+=.005*(e-.02*s.ib[ii])
    def predict(s,u,i): return s.mu+s.ub[u]+s.ib[i]+np.sum(s.U[u]*s.V[i],axis=-1)
    def recommend(s,u,c,k=10): c=np.array(c);return c[np.argsort(s.predict(np.full(len(c),u),c))[::-1][:k]].tolist()

class FIRE:
    def __init__(s,nu,ni,Q,N0=20,M0=10):
        s.nu,s.ni=nu,ni;s.th=np.random.normal(0,.3,nu);s.b=np.random.normal(0,.3,ni);s.a=np.ones(ni)
        s.Q=Q.astype(float);s.eta=np.zeros(Q.shape[1]);s.N0,s.M0=N0,M0;s.uc=np.zeros(nu);s.ic=np.zeros(ni);s.Imax=1.0
    def fit_irt(s,u,i,r,ep=20):
        for uu in u: s.uc[uu]+=1
        for ii in i: s.ic[ii]+=1
        for _ in range(ep):
            for x in np.random.permutation(len(u))[:100000]:
                uu,ii,rr=u[x],i[x],r[x];p=np.clip(expit(s.a[ii]*(s.th[uu]-s.b[ii])),1e-7,1-1e-7);e=rr-p
                s.th[uu]+=.01*e*s.a[ii];s.b[ii]-=.01*e*s.a[ii];s.a[ii]=np.clip(s.a[ii]+.01*e*(s.th[uu]-s.b[ii]),.1,5)
        s.Imax=max(1.0,float((s.a**2).max()*.25))
    def fit_lltm(s):
        warm=(s.ic>=s.M0)[:s.Q.shape[0]];nw=int(warm.sum())
        if nw<50: return {'rho':0,'rho_sq':0,'n_warm':nw}
        Qw,bw=s.Q[warm],s.b[:s.Q.shape[0]][warm];v=~np.isnan(bw)&~np.isinf(bw);Qv,bv=Qw[v],bw[v]
        if len(bv)<50: return {'rho':0,'rho_sq':0,'n_warm':nw}
        s.eta=np.linalg.solve(Qv.T@Qv+.01*np.eye(Qv.shape[1]),Qv.T@bv)
        bp=Qv@s.eta;rho=float(np.corrcoef(bv,bp)[0,1])
        return {'rho':rho,'rho_sq':rho**2,'n_warm':nw}  # [v4.5] rho_sq not R²
    def score(s,u,i):
        au=expit(np.log(np.maximum(s.uc[u],1)/s.N0));li=np.clip(1-s.ic[i]/s.M0,0,1)
        p_irt=expit(s.a[i]*(s.th[u]-s.b[i]));p_lltm=np.full(len(i),.5)
        ok=i<s.Q.shape[0]
        if ok.any(): p_lltm[ok]=expit(-(s.Q[i[ok]]@s.eta))
        pe=(1-li)*p_irt+li*p_lltm;fi=(s.a[i]**2)*pe*(1-pe)/s.Imax
        return au*pe+(1-au)*fi
    def recommend(s,u,c,k=10): c=np.array(c);return c[np.argsort(s.score(np.full(len(c),u),c))[::-1][:k]].tolist()

def recall_k(r,rel,k=10): return len(set(r[:k])&set(rel))/max(len(rel),1)
def ndcg_k(r,rel,k=10):
    rs=set(rel);d=sum(1/np.log2(j+2) for j,x in enumerate(r[:k]) if x in rs)
    g=sum(1/np.log2(j+2) for j in range(min(len(rs),k)));return d/g if g>0 else 0

def run_seed(seed,rat,mov,gm,ag,yrs,nu,ni,valid_items,args):
    rng=np.random.RandomState(seed); np.random.seed(seed)
    tr,te=temporal_split(rat); feats,fnames=build_features(tr,gm,yrs,ni,ag)
    tu,ti,tb=tr['u'].values,tr['i'].values,tr['b'].values

    fire=FIRE(nu,ni,feats,args.N0,args.M0);fire.fit_irt(tu,ti,tb);lr=fire.fit_lltm()
    irt=IRT1D(nu,ni);irt.fit(tu,ti,tb)
    svd=SVD50(nu,ni);svd.fit(tu,ti,tr['rating'].values)
    ip=defaultdict(lambda:[0,0])
    for u,i,r in zip(tu,ti,tb): ip[i][0]+=r;ip[i][1]+=1

    teu,tei,teb=te['u'].values,te['i'].values,te['b'].values
    auc={'FIRE':roc_auc_score(teb,fire.score(teu,tei)),'IRT':roc_auc_score(teb,irt.predict(teu,tei)),
         'SVD':roc_auc_score(teb,svd.predict(teu,tei)),'Pop':roc_auc_score(teb,np.array([ip[i][0] for i in tei]))}

    # Low-count items
    itc=tr.groupby('i').size();te_low=te[te['i'].map(lambda x:itc.get(x,0)<args.M0)]
    low_auc={}
    if len(te_low)>100 and len(set(te_low['b']))>1:
        lu,li,lb=te_low['u'].values,te_low['i'].values,te_low['b'].values
        low_auc={'FIRE':roc_auc_score(lb,fire.score(lu,li)),'IRT':roc_auc_score(lb,irt.predict(lu,li)),
                 'Pop':roc_auc_score(lb,np.array([ip[i][0] for i in li]))}

    # Top-K: [v4.1] FULL CATALOG, [v4.2] RANDOM eval users
    trel=defaultdict(list)
    for _,r in te[te['b']==1].iterrows(): trel[int(r['u'])].append(int(r['i']))
    titems=defaultdict(set)
    for _,r in tr.iterrows(): titems[int(r['u'])].add(int(r['i']))
    eligible=np.array([u for u in trel if len(trel[u])>=3])
    eu=eligible if args.n_eval_users<=0 else rng.choice(eligible,size=min(args.n_eval_users,len(eligible)),replace=False)  # [v4.2]

    # [v4.1] Full catalog — no sampling
    candidate_sets={}
    for u in eu: candidate_sets[u]=list(valid_items-titems[u])

    def pop_rec(u,c,k=10):
        c=np.array(c);scores=np.array([ip.get(i,[0,0])[0] for i in c])
        return c[np.argsort(scores)[::-1][:k]].tolist()
    models={'FIRE':fire.recommend,'IRT':irt.recommend,'SVD':svd.recommend,'Pop':pop_rec}
    rec={}
    for mn,mf in models.items():
        ra,rc,nd,hr=[],[],[],[]
        for u in eu:
            recs=mf(u,candidate_sets[u],10);ra.append(recs)
            rc.append(recall_k(recs,trel[u]));nd.append(ndcg_k(recs,trel[u]))
            hr.append(1.0 if set(recs[:10])&set(trel[u]) else 0.0)
        cv=len(set(x for r in ra for x in r))/len(valid_items)
        rec[mn]={'R@10':float(np.mean(rc)),'NDCG@10':float(np.mean(nd)),'HR@10':float(np.mean(hr)),'Cov':float(cv)}

    return {'seed':seed,'auc':auc,'low_auc':low_auc,'rec':rec,'lltm':lr,
            'fnames':fnames,'eta':list(fire.eta)}

def main():
    args=parse_args();os.makedirs(args.out_dir,exist_ok=True)
    print('='*70);print('FIRE v5: Final Production');print('='*70)
    rat,mov,gm,ag,yrs,nu,ni,vi=load_movielens(args.data_dir)
    n_seeds=len(args.seeds)
    print(f'  {nu} users, {len(vi)} valid items (max ID {ni}), {len(rat)} ratings')

    runs=[]
    for seed in args.seeds:
        t0=time.time();r=run_seed(seed,rat,mov,gm,ag,yrs,nu,ni,vi,args);r['time']=round(time.time()-t0,1);runs.append(r)
        a=r['auc'];rc=r['rec']
        print(f'\n  Seed {seed} ({r["time"]:.0f}s):')
        print(f'    AUC: FIRE={a["FIRE"]:.4f} IRT={a["IRT"]:.4f} SVD={a["SVD"]:.4f} Pop={a["Pop"]:.4f}')
        for mn in ['FIRE','IRT','SVD','Pop']:
            x=rc[mn];print(f'    {mn:5s}: R={x["R@10"]:.4f} N={x["NDCG@10"]:.4f} HR={x["HR@10"]:.4f} C={x["Cov"]:.4f}')

    # Aggregate
    methods=['FIRE','IRT','SVD','Pop']
    agg={'auc':{},'rec':{},'low_auc':{}}
    for m in methods:
        v=[r['auc'][m] for r in runs];agg['auc'][m]={'mean':round(np.mean(v),4),'sem':round(sem(v),4)}
    for m in methods:
        agg['rec'][m]={}
        for met in ['R@10','NDCG@10','HR@10','Cov']:
            v=[r['rec'][m][met] for r in runs];agg['rec'][m][met]={'mean':round(np.mean(v),4),'sem':round(sem(v),4)}
    for m in ['FIRE','IRT','Pop']:
        v=[r['low_auc'].get(m,0) for r in runs if r['low_auc']]
        if v: agg['low_auc'][m]={'mean':round(np.mean(v),4),'sem':round(sem(v),4) if len(v)>1 else 0}
    lrhos=[r['lltm']['rho'] for r in runs];lrsq=[r['lltm']['rho_sq'] for r in runs]
    agg['lltm']={'rho_mean':round(np.mean(lrhos),4),'rho_sem':round(sem(lrhos),4),
                 'rho_sq_mean':round(np.mean(lrsq),4),'rho_sq_sem':round(sem(lrsq),4)}

    print(f'\n{"="*70}');print(f'AGGREGATED ({n_seeds} seeds, full-catalog eval, random users)')
    print(f'{"="*70}')
    print('\n  AUC (temporal held-out):')  # [v4.7]
    for m in methods: print(f'    {m:5s}: {agg["auc"][m]["mean"]:.4f} ± {agg["auc"][m]["sem"]:.4f}')
    print('\n  Top-10 (full catalog, no candidate sampling):')
    for m in methods:
        r=agg['rec'][m]
        print(f'    {m:5s}: R={r["R@10"]["mean"]:.4f}±{r["R@10"]["sem"]:.4f} '
              f'N={r["NDCG@10"]["mean"]:.4f}±{r["NDCG@10"]["sem"]:.4f} '
              f'HR={r["HR@10"]["mean"]:.4f}±{r["HR@10"]["sem"]:.4f} '
              f'C={r["Cov"]["mean"]:.4f}±{r["Cov"]["sem"]:.4f}')
    print(f'\n  LLTM (warm items, in-sample squared correlation): '  # [v4.5]
          f'ρ²={agg["lltm"]["rho_sq_mean"]:.4f}±{agg["lltm"]["rho_sq_sem"]:.4f}')

    # Figures
    print('\n  Generating figures...')
    fig,ax=plt.subplots(1,1,figsize=(5.5,2.8))
    th=np.linspace(-3,3,200);P=expit(1.5*th);I_n=1.5**2*P*(1-P)/(1.5**2*.25)
    ax.plot(th,P,'-',color='#2979ff',lw=2,label=r'$P(\theta)$  [exploit]')
    ax.plot(th,I_n,'--',color='#d32f2f',lw=2,label=r'$\bar{I}(\theta)$  [explore]')
    ax.axvline(0,color='gray',ls=':',alpha=.5);ax.fill_between(th,0,I_n,alpha=.06,color='#d32f2f')
    ax.set_xlabel(r'$\theta$');ax.set_ylabel('Score');ax.set_title('FIRE: Exploitation vs Exploration',fontweight='bold')
    ax.legend(fontsize=7.5,loc='upper left');ax.set_xlim(-3,3);ax.set_ylim(0,1.08)
    ax.annotate(r'$\theta=b$'+'\n(frontier)',xy=(0,1),xytext=(.8,.85),fontsize=7,arrowprops=dict(arrowstyle='->',color='gray'),color='gray')
    plt.tight_layout();fig.savefig(f'{args.out_dir}/fig1_concept.pdf',bbox_inches='tight');plt.close()

    fig,ax=plt.subplots(1,1,figsize=(4.5,2.8))
    means=[agg['auc'][m]['mean'] for m in methods];sems_v=[agg['auc'][m]['sem'] for m in methods]
    colors=['#2979ff','#546e7a','#78909c','#90a4ae']
    bars=ax.bar(methods,means,yerr=sems_v,color=colors,width=.5,edgecolor='white',lw=.5,capsize=3)
    for bar,v,s in zip(bars,means,sems_v):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+s+.003,f'{v:.4f}',ha='center',fontsize=7.5,fontweight='bold')
    ax.set_ylabel('AUC');ax.set_title(f'Engagement Prediction ({n_seeds} seeds, mean±SEM)',fontweight='bold')
    ax.set_ylim(min(means)-.03,max(means)+.02)
    plt.tight_layout();fig.savefig(f'{args.out_dir}/fig2_auc.pdf',bbox_inches='tight');plt.close()

    # [v4.3] Figure 3 with FIRE label offset to avoid clipping
    fig,ax=plt.subplots(1,1,figsize=(4.5,3.2))
    for m,c in zip(methods,colors):
        hr=agg['rec'][m]['HR@10'];cv=agg['rec'][m]['Cov']
        ax.errorbar(cv['mean'],hr['mean'],xerr=cv['sem'],yerr=hr['sem'],fmt='o',color=c,
            markersize=8 if m=='FIRE' else 6,capsize=3,zorder=5)
        if m=='FIRE':
            ax.annotate(m,xy=(cv['mean'],hr['mean']),xytext=(cv['mean']-.004,hr['mean']-.025),
                arrowprops=dict(arrowstyle='->',lw=.5,color=c),fontsize=8,fontweight='bold',color=c)
        elif m=='Pop':
            ax.annotate(m,xy=(cv['mean'],hr['mean']),xytext=(cv['mean']-.003,hr['mean']-.028),
                arrowprops=dict(arrowstyle='->',lw=.5,color=c),fontsize=8,fontweight='bold',color=c)
        elif m=='IRT':
            ax.annotate(m,xy=(cv['mean'],hr['mean']),xytext=(cv['mean']-.005,hr['mean']+.020),
                arrowprops=dict(arrowstyle='->',lw=.5,color=c),fontsize=8,fontweight='bold',color=c)
        else:
            ax.annotate(m,xy=(cv['mean'],hr['mean']),xytext=(cv['mean']+.003,hr['mean']+.020),
                arrowprops=dict(arrowstyle='->',lw=.5,color=c),fontsize=8,fontweight='bold',color=c)
    ax.set_xlabel('Coverage');ax.set_ylabel('HitRate@10')
    ax.set_title('Accuracy–Diversity Tradeoff (full catalog)',fontweight='bold')
    # Expand x-axis to ensure labels fit
    all_covs=[agg['rec'][m]['Cov']['mean'] for m in methods]
    ax.set_xlim(min(all_covs)-.008,max(all_covs)+.008)
    plt.tight_layout();fig.savefig(f'{args.out_dir}/fig3_tradeoff.pdf',bbox_inches='tight');plt.close()

    # [v4.4] Figure 4: note seed in title
    eta_sorted=sorted(zip(runs[0]['fnames'],runs[0]['eta']),key=lambda x:abs(x[1]),reverse=True)[:10]
    fig,ax=plt.subplots(1,1,figsize=(5,3))
    names=[x[0] for x in eta_sorted];vals=[x[1] for x in eta_sorted]
    ax.barh(range(len(names)),vals,color=['#00c853' if v<0 else '#d32f2f' for v in vals],height=.6)
    ax.set_yticks(range(len(names)));ax.set_yticklabels(names,fontsize=8)
    ax.set_xlabel(r'$\eta$');ax.axvline(0,color='gray',lw=.5);ax.invert_yaxis()
    ax.set_title(f'LLTM Weights (warm items, in-sample, seed {args.seeds[0]})',fontweight='bold',fontsize=9)
    plt.tight_layout();fig.savefig(f'{args.out_dir}/fig4_lltm.pdf',bbox_inches='tight');plt.close()
    print('  4 figures saved')

    # Save results
    output={'_note':'Single source of truth. Paper tables must match these values.',
        'n_seeds':n_seeds,'seeds':args.seeds,'eval_protocol':'full_catalog_no_sampling',
        'eval_users':'random_sample','n_eval_users':args.n_eval_users,
        'dataset':{'users':nu,'items_valid':len(vi),'items_max_id':ni,'ratings':len(rat)},
        'config':{'N0':args.N0,'M0':args.M0},
        'auc':agg['auc'],'rec':agg['rec'],'low_count_auc':agg['low_auc'],'lltm':agg['lltm'],
        'per_run':[{k:v for k,v in r.items() if k not in ('fnames','eta')} for r in runs]}
    out_path=os.path.join(os.path.dirname(args.out_dir) or '.','results_v5.json')
    with open(out_path,'w') as f:
        json.dump(output,f,indent=2,default=lambda x:float(x) if hasattr(x,'item') else str(x))

    # Auto-generate LaTeX tables
    print(f'\n  === LaTeX Table 1 (AUC, temporal held-out) ===')
    for m in methods: a=agg['auc'][m];print(f'  {m} & ${a["mean"]:.4f} \\pm {a["sem"]:.4f}$ \\\\')
    print(f'\n  === LaTeX Table 2 (Top-10, full catalog) ===')
    for m in methods:
        r=agg['rec'][m]
        print(f'  {m} & ${r["R@10"]["mean"]:.4f}\\pm{r["R@10"]["sem"]:.3f}$ '
              f'& ${r["NDCG@10"]["mean"]:.4f}\\pm{r["NDCG@10"]["sem"]:.3f}$ '
              f'& ${r["HR@10"]["mean"]:.3f}\\pm{r["HR@10"]["sem"]:.3f}$ '
              f'& ${r["Cov"]["mean"]:.4f}\\pm{r["Cov"]["sem"]:.3f}$ \\\\')

    # [v4.6] Auto-generate README
    readme=f"""# FIRE: Fisher Information Recommendation Engine

## Results (MovieLens-1M, {n_seeds} seeds, full-catalog eval, mean ± SEM)

| Method | AUC | Recall@10 | NDCG@10 | HitRate@10 | Coverage |
|---|---:|---:|---:|---:|---:|
"""
    for m in methods:
        a=agg['auc'][m];r=agg['rec'][m]
        readme+=f"| {m} | {a['mean']:.4f}±{a['sem']:.4f} | {r['R@10']['mean']:.4f}±{r['R@10']['sem']:.4f} | {r['NDCG@10']['mean']:.4f}±{r['NDCG@10']['sem']:.4f} | {r['HR@10']['mean']:.3f}±{r['HR@10']['sem']:.3f} | {r['Cov']['mean']:.4f}±{r['Cov']['sem']:.4f} |\n"
    readme+=f"""
- FIRE's main advantage over IRT/SVD: catalog coverage with explicit per-component score decompositions.
- Popularity dominates Top-K accuracy; SVD-50 leads AUC.
- LLTM squared correlation (in-sample, warm items): ρ²={agg['lltm']['rho_sq_mean']:.4f}

## Reproduce

```bash
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d data/
pip install -r requirements.txt  # Python >= 3.11
python src/experiment_v5.py --data_dir data/ml-1m --out_dir figures --seeds 42 123 789 --n_eval_users {args.n_eval_users}
```

## Author

Jung Min Kang · Independent Researcher, Seoul · ORCID: 0009-0007-9599-2792
"""
    with open('README.md','w') as f: f.write(readme)
    print(f'\n  README.md auto-generated from results')
    print(f'\n{"="*70}');print('DONE');print(f'{"="*70}')

if __name__=='__main__': main()
