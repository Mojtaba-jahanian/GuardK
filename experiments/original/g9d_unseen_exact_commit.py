from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.optimize import linear_sum_assignment

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
)
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

from unseen import UNSEEN_DEC

ROOT = Path.home() / "GuardK_V6"
OUT = ROOT / "results/g9d_unseen"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [16001, 17011, 18013]


def datasets():
    funcs = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer_wisconsin": load_breast_cancer,
        "digits": load_digits,
    }

    d = {}

    for name, fn in funcs.items():
        b = fn()
        X = np.asarray(b.data, dtype=np.float32)
        y = np.asarray(b.target)

        d[name] = (
            X,
            y,
            int(len(np.unique(y))),
        )

    return d


def worst_recall(y_true, y_pred):
    tv = np.unique(y_true)
    pv = np.unique(y_pred)

    C = np.zeros(
        (len(tv), len(pv)),
        dtype=np.int64,
    )

    for i, t in enumerate(tv):
        for j, p in enumerate(pv):
            C[i, j] = int(
                np.sum(
                    (y_true == t)
                    & (y_pred == p)
                )
            )

    if C.size == 0:
        return 0.0

    ri, ci = linear_sum_assignment(-C)

    vals = np.zeros(len(tv), dtype=float)

    for i, j in zip(ri, ci):
        den = np.sum(y_true == tv[i])
        vals[i] = C[i, j] / den if den else 0.0

    return float(vals.min())


rows = []

for dataset, (X, y, true_k) in datasets().items():
    Z = StandardScaler().fit_transform(X).astype(np.float32)

    for seed in SEEDS:
        print(
            "G9D_RUN",
            dataset,
            "seed=",
            seed,
            "n=",
            len(Z),
            "d=",
            Z.shape[1],
            flush=True,
        )

        np.random.seed(seed)
        torch.manual_seed(seed)

        m = UNSEEN_DEC(
            n_clusters_init=12,
            dying_threshold=0.5,
            batch_size=256,
            pretrain_epochs=100,
            clustering_epochs=150,
            embedding_size=10,
            device=torch.device("cpu"),
            random_state=seed,
        )

        t0 = time.perf_counter()

        # Labels deliberately NOT supplied.
        m.fit(Z)

        dt = time.perf_counter() - t0

        labels = np.asarray(m.labels_)
        k = int(len(np.unique(labels)))

        row = {
            "dataset": dataset,
            "seed": int(seed),
            "method": "UNSEEN_DEC",
            "true_k": int(true_k),
            "selected_k": k,
            "exact_k": int(k == true_k),
            "abs_k_error": int(abs(k - true_k)),
            "ARI": float(adjusted_rand_score(y, labels)),
            "NMI": float(normalized_mutual_info_score(y, labels)),
            "worst_cluster_recall": worst_recall(y, labels),
            "runtime_sec": float(dt),
        }

        rows.append(row)

        print(
            " selected_k=",
            k,
            "ARI=",
            round(row["ARI"], 6),
            "NMI=",
            round(row["NMI"], 6),
            "runtime=",
            round(dt, 2),
            flush=True,
        )


runs = pd.DataFrame(rows)

runs.to_csv(
    OUT / "G9D_UNSEEN_runs.csv",
    index=False,
)

agg = (
    runs
    .groupby("dataset", dropna=False)
    .agg(
        runs=("seed", "count"),
        exact_k_rate=("exact_k", "mean"),
        mean_abs_k_error=("abs_k_error", "mean"),
        mean_ARI=("ARI", "mean"),
        mean_NMI=("NMI", "mean"),
        mean_worst_cluster_recall=("worst_cluster_recall", "mean"),
        mean_runtime_sec=("runtime_sec", "mean"),
    )
    .reset_index()
)

agg.to_csv(
    OUT / "G9D_UNSEEN_aggregate.csv",
    index=False,
)

print()
print("G9D_UNSEEN_AGGREGATE")
print(agg.to_string(index=False))

lock = {
    "stage": "G9D",
    "status": "COMPLETE_OFFICIAL_UNSEEN_EXACT_DEPENDENCY_COMMIT",
    "ground_truth_passed_to_fit": False,
    "n_clusters_init": 12,
    "post_hoc_tuning": False,
    "device": "cpu",
}

(
    OUT / "G9D_UNSEEN_LOCK.json"
).write_text(
    json.dumps(lock, indent=2)
)

print("G9D_UNSEEN_LABEL_ISOLATION_PASS")
print("G9D_STATUS=COMPLETE_OFFICIAL_UNSEEN_EXACT_DEPENDENCY_COMMIT")
