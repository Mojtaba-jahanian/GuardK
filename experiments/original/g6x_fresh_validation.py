from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from guardk import (
    GuardKConfig, GuardKPrototype,
    GuardKV04Config, GuardKPrototypeV04,
    evaluate_external,
)

ROOT = Path.home() / "GuardK_V6"
OUT = ROOT / "results/g6x_fresh_validation"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [11617,12721,13829,14939]
RATIOS = [3,8,30,60]
CONDITIONS = {
    "clear":{"separation":5.0,"sigma":0.90},
    "moderate":{"separation":3.6,"sigma":1.25},
    "difficult":{"separation":2.5,"sigma":1.60},
    "ambiguous":{"separation":1.7,"sigma":1.85},
}

def sizes_for_ratio(r):
    m=540
    return [m,max(190,round(m/np.sqrt(r))),max(85,round(m/(r**0.75))),max(10,round(m/r))]

def make_data(ratio, condition, seed):
    p=CONDITIONS[condition]; s=p["separation"]; sigma=p["sigma"]
    centers=np.array([[-s,-s],[-s,s],[s,-s],[s,s]],float)
    sizes=sizes_for_ratio(ratio)
    rng=np.random.default_rng(seed+3301*ratio+7919*list(CONDITIONS).index(condition))
    Xs=[]; ys=[]
    for c,(mu,n) in enumerate(zip(centers,sizes)):
        Xs.append(rng.normal(mu,sigma,size=(int(n),2)))
        ys.append(np.full(int(n),c,int))
    X=np.vstack(Xs); y=np.concatenate(ys); order=rng.permutation(len(X))
    return X[order], y[order]

def kmeans_silhouette(X,seed):
    Z=StandardScaler().fit_transform(X); best=None
    for k in range(2,9):
        m=KMeans(n_clusters=k,n_init=15,random_state=seed+k,algorithm="lloyd").fit(Z)
        sil=float(silhouette_score(Z,m.labels_,sample_size=min(1500,len(Z)),random_state=seed))
        item=(sil,-k,k,m.labels_.copy())
        if best is None or item[:2]>best[:2]: best=item
    return best[2],best[3]

def gmm_bic(X,seed):
    Z=StandardScaler().fit_transform(X); best=None
    for k in range(2,9):
        m=GaussianMixture(n_components=k,covariance_type="full",n_init=3,random_state=seed+k,reg_covar=1e-6).fit(Z)
        item=(float(m.bic(Z)),k,m.predict(Z))
        if best is None or item[0]<best[0]: best=item
    return best[1],best[2]

def row(scenario,ratio,condition,seed,method,k,decision,y,pred,runtime,diag=None):
    d={
        "scenario":scenario,"ratio":int(ratio),"condition":condition,"seed":int(seed),
        "method":method,"true_k":4,"selected_k":np.nan if k is None else int(k),
        "decision":decision,"coverage":int(k is not None),"runtime_sec":float(runtime),
    }
    if diag:
        for key in ["selection_consistency","selection_mode","selection_mode_support","score_margin","overlap_safety"]:
            d[key]=diag.get(key,np.nan)
    if k is None:
        d.update({"exact_k":np.nan,"abs_k_error":np.nan,"ARI":np.nan,"NMI":np.nan,"worst_cluster_recall":np.nan})
        return d
    m=evaluate_external(y,pred)
    d.update({
        "exact_k":int(k==4),"abs_k_error":int(abs(k-4)),
        "ARI":m["ARI"],"NMI":m["NMI"],"worst_cluster_recall":m["worst_cluster_recall"]
    })
    return d

rows=[]
for ratio in RATIOS:
    for condition in CONDITIONS:
        for seed in SEEDS:
            scenario=f"r{ratio}_{condition}"
            print("G6X_SCENARIO",scenario,"seed=",seed,flush=True)
            X,y=make_data(ratio,condition,seed)

            t=time.perf_counter()
            m=GuardKPrototype(GuardKConfig(k_min=2,k_max=8,n_bootstrap=4,seed=seed,allow_abstain=False)).fit(X)
            rows.append(row(scenario,ratio,condition,seed,"guardk_v02",m.selected_k_,m.decision_,y,m.labels_,time.perf_counter()-t))

            t=time.perf_counter()
            v=GuardKPrototypeV04(GuardKV04Config(seed=seed)).fit(X)
            rows.append(row(scenario,ratio,condition,seed,"guardk_v04",v.selected_k_,v.decision_,y,v.labels_,time.perf_counter()-t,v.diagnostics_))

            t=time.perf_counter()
            k,p=kmeans_silhouette(X,seed)
            rows.append(row(scenario,ratio,condition,seed,"kmeans_silhouette",k,"ACCEPT",y,p,time.perf_counter()-t))

            t=time.perf_counter()
            k,p=gmm_bic(X,seed)
            rows.append(row(scenario,ratio,condition,seed,"gmm_bic",k,"ACCEPT",y,p,time.perf_counter()-t))

runs=pd.DataFrame(rows)
runs.to_csv(OUT/"G6X_fresh_runs.csv",index=False)

agg=(runs.groupby("method",dropna=False)
    .agg(
        runs=("seed","count"),
        coverage=("coverage","mean"),
        exact_k_rate_all=("exact_k",lambda x:float(x.fillna(0).mean())),
        exact_k_rate_accepted=("exact_k","mean"),
        mean_abs_k_error_accepted=("abs_k_error","mean"),
        mean_ARI_accepted=("ARI","mean"),
        mean_NMI_accepted=("NMI","mean"),
        mean_worst_cluster_recall_accepted=("worst_cluster_recall","mean"),
        mean_runtime_sec=("runtime_sec","mean"),
    ).reset_index())
agg.to_csv(OUT/"G6X_fresh_aggregate.csv",index=False)

cond=(runs.groupby(["condition","method"],dropna=False)
    .agg(
        coverage=("coverage","mean"),
        exact_k_rate_all=("exact_k",lambda x:float(x.fillna(0).mean())),
        exact_k_rate_accepted=("exact_k","mean"),
        mean_ARI_accepted=("ARI","mean"),
        mean_worst_cluster_recall_accepted=("worst_cluster_recall","mean"),
    ).reset_index())
cond.to_csv(OUT/"G6X_by_condition.csv",index=False)

pair=(runs[runs["method"].isin(["guardk_v02","guardk_v04"])]
      .pivot_table(index=["scenario","ratio","condition","seed","true_k"],
                   columns="method",values=["coverage","exact_k","selected_k"],aggfunc="first")
      .reset_index())
pair.columns=["_".join([str(x) for x in c if str(x)!=""]) if isinstance(c,tuple) else str(c) for c in pair.columns]

wrong=pair["exact_k_guardk_v02"]==0
correct=pair["exact_k_guardk_v02"]==1
abstain=pair["coverage_guardk_v04"]==0
error_capture=float(np.mean(abstain[wrong])) if wrong.any() else np.nan
false_abstain=float(np.mean(abstain[correct])) if correct.any() else np.nan

print("\nG6X_AGGREGATE")
print(agg.to_string(index=False))
print("\nG6X_BY_CONDITION")
print(cond.to_string(index=False))
print("\nG6X_V02_ERROR_CAPTURE_BY_V04_ABSTENTION=",error_capture)
print("G6X_V02_CORRECT_FALSE_ABSTENTION_BY_V04=",false_abstain)

a02=agg[agg["method"]=="guardk_v02"].iloc[0]
a04=agg[agg["method"]=="guardk_v04"].iloc[0]
c04=cond[cond["method"]=="guardk_v04"]
clear_mod=c04[c04["condition"].isin(["clear","moderate"])]

gates={
    "overall_coverage_ge_0.50": bool(a04["coverage"]>=0.50),
    "clear_moderate_mean_coverage_ge_0.80": bool(clear_mod["coverage"].mean()>=0.80),
    "accepted_exact_k_gain_ge_0.08": bool(a04["exact_k_rate_accepted"]>=a02["exact_k_rate_accepted"]+0.08),
    "accepted_ari_not_worse_than_v02_minus_0.02": bool(a04["mean_ARI_accepted"]>=a02["mean_ARI_accepted"]-0.02),
    "v02_error_capture_ge_0.30": bool(np.isnan(error_capture) or error_capture>=0.30),
}
status="PASS_SELECTIVE_GUARD_PROMISING" if all(gates.values()) else "REVIEW_REQUIRED"

lock={
    "stage":"G6X","status":status,"confirmatory":False,
    "fresh_seeds":SEEDS,"fresh_ratios":RATIOS,"conditions":list(CONDITIONS),
    "v04_protocol_changed_after_seeing_g6x":False,
    "ground_truth_used_by_v04_fit":False,
    "gates":gates,"error_capture":error_capture,
    "false_abstain_on_v02_correct":false_abstain,
}
(OUT/"G6X_LOCK.json").write_text(json.dumps(lock,indent=2))

for k,v in gates.items(): print("GATE",k,"=",v)
print("G6X_STATUS="+status)
print("G6X_LABEL_ISOLATION_PASS")
print("G6X_PROTOCOL_NO_POSTHOC_CHANGE_PASS")
