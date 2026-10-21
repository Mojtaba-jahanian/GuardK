from pathlib import Path
import os, sys, time, json
import numpy as np, pandas as pd, torch
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT=Path.home()/"GuardK_V6"
REPO=ROOT/"baselines/modern/DeepDPM_locked_95c9376"
OUT=ROOT/"results/g9f_deepdpm"
TMP=ROOT/"tmp/deepdpm_data"
OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(REPO)); os.chdir(REPO)
import DeepDPM

def wr(y,p):
    a=np.unique(y); b=np.unique(p); C=np.zeros((len(a),len(b)),dtype=int)
    for i,x in enumerate(a):
        for j,z in enumerate(b): C[i,j]=np.sum((y==x)&(p==z))
    if C.size==0:return 0.0
    ri,ci=linear_sum_assignment(-C); r=np.zeros(len(a))
    for i,j in zip(ri,ci):
        den=np.sum(y==a[i]); r[i]=C[i,j]/den if den else 0
    return float(r.min())

sets={
 "iris":load_iris(),
 "wine":load_wine(),
 "breast_cancer_wisconsin":load_breast_cancer(),
 "digits":load_digits()
}
seeds=[16001,17011,18013]
rows=[]

# Smoke: 60 epochs reaches the official default split/merge start at epoch 55.
b=sets["iris"]; X=np.asarray(b.data,dtype=np.float32)
sd=TMP/"smoke"; sd.mkdir(parents=True,exist_ok=True)
xt=torch.from_numpy(X)
torch.save(xt,sd/"train_data.pt")
torch.save(xt.clone(),sd/"test_data.pt")
sys.argv=["DeepDPM.py","--dataset","custom","--dir",str(sd),"--offline","--seed","16001","--device","cpu","--max_epochs","60","--init_k","1","--transform_input_data","standard"]
pred=np.asarray(DeepDPM.train_cluster_net())
assert len(pred)==len(X)
print("G9E_DEEPDPM_EXECUTION_SMOKE_PASS selected_k=",len(np.unique(pred)),flush=True)

for name,b in sets.items():
    X=np.asarray(b.data,dtype=np.float32); y=np.asarray(b.target); tk=len(np.unique(y))
    dd=TMP/name; dd.mkdir(parents=True,exist_ok=True)
    xt=torch.from_numpy(X)
    torch.save(xt,dd/"train_data.pt")
    torch.save(xt.clone(),dd/"test_data.pt")
    for seed in seeds:
        print("G9F_RUN",name,"seed=",seed,"n=",len(X),"d=",X.shape[1],flush=True)
        np.random.seed(seed); torch.manual_seed(seed)
        sys.argv=["DeepDPM.py","--dataset","custom","--dir",str(dd),"--offline","--seed",str(seed),"--device","cpu","--max_epochs","500","--init_k","1","--transform_input_data","standard"]
        t=time.perf_counter(); p=np.asarray(DeepDPM.train_cluster_net()); dt=time.perf_counter()-t
        k=len(np.unique(p))
        row=dict(dataset=name,seed=seed,method="DeepDPM",true_k=tk,selected_k=k,
                 exact_k=int(k==tk),abs_k_error=abs(k-tk),
                 ARI=adjusted_rand_score(y,p),NMI=normalized_mutual_info_score(y,p),
                 worst_cluster_recall=wr(y,p),runtime_sec=dt)
        rows.append(row)
        print(" selected_k=",k,"ARI=",round(row["ARI"],6),"NMI=",round(row["NMI"],6),"runtime=",round(dt,2),flush=True)

runs=pd.DataFrame(rows); runs.to_csv(OUT/"G9F_DeepDPM_runs.csv",index=False)
agg=(runs.groupby("dataset").agg(runs=("seed","count"),exact_k_rate=("exact_k","mean"),
 mean_abs_k_error=("abs_k_error","mean"),mean_ARI=("ARI","mean"),mean_NMI=("NMI","mean"),
 mean_worst_cluster_recall=("worst_cluster_recall","mean"),mean_runtime_sec=("runtime_sec","mean")).reset_index())
agg.to_csv(OUT/"G9F_DeepDPM_aggregate.csv",index=False)
print("G9F_DEEPDPM_AGGREGATE"); print(agg.to_string(index=False))
(OUT/"G9F_DeepDPM_LOCK.json").write_text(json.dumps({
 "status":"COMPLETE_OFFICIAL_DEEPDPM_CPU_COMMON_DATA","official_source_modified":False,
 "ground_truth_passed_to_fit":False,"init_k":1,"max_epochs":500,"device":"cpu","post_hoc_tuning":False},indent=2))
print("G9F_DEEPDPM_LABEL_ISOLATION_PASS")
