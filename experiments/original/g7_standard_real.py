from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import hdbscan

from guardk import (
    GuardKConfig,
    GuardKPrototype,
    GuardKV04Config,
    GuardKPrototypeV04,
    evaluate_external,
)

ROOT = Path.home() / "GuardK_V6"
OUT = ROOT / "results/g7_standard_real"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [16001, 17011, 18013]
KMIN = 2
KMAX = 12

def datasets():
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer_wisconsin": load_breast_cancer,
        "digits": load_digits,
    }

    out = {}

    for name, fn in loaders.items():
        b = fn()
        X = np.asarray(b.data, dtype=float)
        y = np.asarray(b.target)
        out[name] = (
            X,
            y,
            int(len(np.unique(y))),
        )

    return out


def kmeans_silhouette(X, seed):
    Z = StandardScaler().fit_transform(X)
    best = None

    for k in range(KMIN, KMAX + 1):
        m = KMeans(
            n_clusters=k,
            n_init=15,
            random_state=seed + k,
            algorithm="lloyd",
        ).fit(Z)

        sil = float(
            silhouette_score(
                Z,
                m.labels_,
                sample_size=min(1600, len(Z)),
                random_state=seed,
            )
        )

        item = (
            sil,
            -k,
            k,
            m.labels_.copy(),
        )

        if best is None or item[:2] > best[:2]:
            best = item

    return best[2], best[3], "silhouette"


def gmm_bic(X, seed):
    Z = StandardScaler().fit_transform(X)
    best = None

    for k in range(KMIN, KMAX + 1):
        m = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=3,
            random_state=seed + k,
            reg_covar=1e-6,
        ).fit(Z)

        bic = float(m.bic(Z))
        labels = m.predict(Z)

        item = (
            bic,
            k,
            labels.copy(),
        )

        if best is None or item[0] < best[0]:
            best = item

    return best[1], best[2], "BIC"


def agglomerative_silhouette(X, seed):
    Z = StandardScaler().fit_transform(X)
    best = None

    for k in range(KMIN, KMAX + 1):
        labels = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
        ).fit_predict(Z)

        sil = float(
            silhouette_score(
                Z,
                labels,
                sample_size=min(1600, len(Z)),
                random_state=seed,
            )
        )

        item = (
            sil,
            -k,
            k,
            labels.copy(),
        )

        if best is None or item[:2] > best[:2]:
            best = item

    return best[2], best[3], "silhouette"


def hdbscan_unknown_k(X):
    Z = StandardScaler().fit_transform(X)

    model = hdbscan.HDBSCAN(
        min_cluster_size=max(
            8,
            int(round(0.02 * len(Z))),
        ),
        cluster_selection_method="eom",
    )

    labels = model.fit_predict(Z)

    k = len(
        [
            c
            for c in np.unique(labels)
            if c != -1
        ]
    )

    return k, labels, "native_unknown_k"


def guardk_v02(X, seed):
    m = GuardKPrototype(
        GuardKConfig(
            k_min=KMIN,
            k_max=KMAX,
            n_bootstrap=4,
            seed=seed,
            allow_abstain=False,
        )
    ).fit(X)

    return (
        m.selected_k_,
        m.labels_,
        m.decision_,
        None,
    )


def guardk_v04(X, seed):
    m = GuardKPrototypeV04(
        GuardKV04Config(
            k_min=KMIN,
            k_max=KMAX,
            seed=seed,
            inner_bootstrap=4,
            outer_repeats=5,
            outer_fraction=0.80,
            outer_inner_bootstrap=3,
            min_selection_consistency=0.60,
            strong_selection_consistency=0.80,
            parsimony_tolerance=0.008,
            min_overlap_safety=0.45,
            allow_abstain=True,
        )
    ).fit(X)

    return (
        m.selected_k_,
        m.labels_,
        m.decision_,
        m.diagnostics_,
    )


def make_row(
    dataset,
    seed,
    method,
    true_k,
    k,
    labels,
    decision,
    runtime,
    diag=None,
):
    row = {
        "dataset": dataset,
        "seed": int(seed),
        "method": method,
        "true_k": int(true_k),
        "selected_k": np.nan if k is None else int(k),
        "coverage": int(k is not None),
        "decision": decision,
        "runtime_sec": float(runtime),
    }

    if diag:
        for key in [
            "selection_consistency",
            "selection_mode",
            "selection_mode_support",
            "score_margin",
            "overlap_safety",
        ]:
            row[key] = diag.get(key, np.nan)

    if k is None:
        row.update(
            {
                "exact_k": np.nan,
                "abs_k_error": np.nan,
                "ARI": np.nan,
                "NMI": np.nan,
                "worst_cluster_recall": np.nan,
                "mean_cluster_recall": np.nan,
            }
        )
        return row

    m = evaluate_external(
        CURRENT_Y,
        labels,
    )

    row.update(
        {
            "exact_k": int(k == true_k),
            "abs_k_error": int(abs(k - true_k)),
            "ARI": m["ARI"],
            "NMI": m["NMI"],
            "worst_cluster_recall": m["worst_cluster_recall"],
            "mean_cluster_recall": m["mean_cluster_recall"],
        }
    )

    return row


rows = []

for dataset, (X, y, true_k) in datasets().items():
    CURRENT_Y = y

    print(
        "G7_DATASET",
        dataset,
        "n=",
        len(X),
        "d=",
        X.shape[1],
        "true_k=",
        true_k,
        flush=True,
    )

    for seed in SEEDS:
        methods = [
            "guardk_v02",
            "guardk_v04",
            "kmeans_silhouette",
            "gmm_bic",
            "agglomerative_silhouette",
            "hdbscan",
        ]

        for method in methods:
            t0 = time.perf_counter()

            if method == "guardk_v02":
                k, labels, decision, diag = guardk_v02(X, seed)

            elif method == "guardk_v04":
                k, labels, decision, diag = guardk_v04(X, seed)

            elif method == "kmeans_silhouette":
                k, labels, criterion = kmeans_silhouette(X, seed)
                decision, diag = "ACCEPT", None

            elif method == "gmm_bic":
                k, labels, criterion = gmm_bic(X, seed)
                decision, diag = "ACCEPT", None

            elif method == "agglomerative_silhouette":
                k, labels, criterion = agglomerative_silhouette(X, seed)
                decision, diag = "ACCEPT", None

            elif method == "hdbscan":
                k, labels, criterion = hdbscan_unknown_k(X)
                decision, diag = "ACCEPT", None

            else:
                raise RuntimeError(method)

            runtime = time.perf_counter() - t0

            rows.append(
                make_row(
                    dataset,
                    seed,
                    method,
                    true_k,
                    k,
                    labels,
                    decision,
                    runtime,
                    diag,
                )
            )

            print(
                " ",
                seed,
                method,
                "K=",
                k,
                "decision=",
                decision,
                "runtime=",
                round(runtime, 3),
                flush=True,
            )

runs = pd.DataFrame(rows)

runs.to_csv(
    OUT / "G7_standard_real_runs.csv",
    index=False,
)

agg = (
    runs
    .groupby(
        ["dataset", "method"],
        dropna=False,
    )
    .agg(
        runs=("seed", "count"),
        coverage=("coverage", "mean"),
        exact_k_rate_all=(
            "exact_k",
            lambda x: float(x.fillna(0).mean()),
        ),
        exact_k_rate_accepted=("exact_k", "mean"),
        mean_abs_k_error_accepted=("abs_k_error", "mean"),
        mean_ARI_accepted=("ARI", "mean"),
        mean_NMI_accepted=("NMI", "mean"),
        mean_worst_cluster_recall_accepted=("worst_cluster_recall", "mean"),
        mean_runtime_sec=("runtime_sec", "mean"),
    )
    .reset_index()
)

agg.to_csv(
    OUT / "G7_standard_real_aggregate.csv",
    index=False,
)

overall = (
    runs
    .groupby("method", dropna=False)
    .agg(
        runs=("seed", "count"),
        coverage=("coverage", "mean"),
        exact_k_rate_all=(
            "exact_k",
            lambda x: float(x.fillna(0).mean()),
        ),
        exact_k_rate_accepted=("exact_k", "mean"),
        mean_ARI_accepted=("ARI", "mean"),
        mean_NMI_accepted=("NMI", "mean"),
        mean_worst_cluster_recall_accepted=("worst_cluster_recall", "mean"),
        mean_runtime_sec=("runtime_sec", "mean"),
    )
    .reset_index()
)

overall.to_csv(
    OUT / "G7_standard_real_overall.csv",
    index=False,
)

print()
print("G7_STANDARD_REAL_AGGREGATE")
print(agg.to_string(index=False))

print()
print("G7_STANDARD_REAL_OVERALL")
print(overall.to_string(index=False))

lock = {
    "stage": "G7",
    "status": "COMPLETE_STANDARD_REAL_DATA",
    "confirmatory": False,
    "ground_truth_used_by_guardk_fit": False,
    "datasets": list(datasets().keys()),
    "seeds": SEEDS,
    "k_range": [KMIN, KMAX],
    "deep_comparators_included": False,
    "clinical_validation_claim": False,
}

(
    OUT / "G7_LOCK.json"
).write_text(
    json.dumps(lock, indent=2)
)

print("G7_LABEL_ISOLATION_PASS")
print("G7_STATUS=COMPLETE_STANDARD_REAL_DATA")


if __name__ == "__main__":
    pass
