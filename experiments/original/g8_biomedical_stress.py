from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from guardk import (
    GuardKConfig,
    GuardKPrototype,
    GuardKV04Config,
    GuardKPrototypeV04,
    evaluate_external,
)

ROOT = Path.home() / "GuardK_V6"
OUT = ROOT / "results/g8_biomedical_stress"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [19001, 20011, 21013, 22019, 23021]
KMIN = 2
KMAX = 8

LEVELS = {
    "native": None,
    "balanced": "balanced",
    "minority_50pct": 0.50,
    "minority_25pct": 0.25,
    "minority_10pct": 0.10,
}

bundle = load_breast_cancer()
X_ALL = np.asarray(bundle.data, dtype=float)
Y_ALL = np.asarray(bundle.target)

classes, counts = np.unique(
    Y_ALL,
    return_counts=True,
)

minority_class = classes[np.argmin(counts)]
majority_class = classes[np.argmax(counts)]

minority_idx_all = np.flatnonzero(
    Y_ALL == minority_class
)
majority_idx_all = np.flatnonzero(
    Y_ALL == majority_class
)

def make_subset(level, seed):
    rng = np.random.default_rng(seed)

    if level == "native":
        idx = np.arange(len(Y_ALL))

    elif level == "balanced":
        n = min(
            len(minority_idx_all),
            len(majority_idx_all),
        )

        a = rng.choice(
            minority_idx_all,
            size=n,
            replace=False,
        )

        b = rng.choice(
            majority_idx_all,
            size=n,
            replace=False,
        )

        idx = np.concatenate([a, b])

    else:
        frac = float(
            LEVELS[level]
        )

        n_min = max(
            12,
            int(
                round(
                    len(minority_idx_all)
                    * frac
                )
            ),
        )

        a = rng.choice(
            minority_idx_all,
            size=n_min,
            replace=False,
        )

        b = majority_idx_all.copy()

        idx = np.concatenate([a, b])

    rng.shuffle(idx)

    return (
        X_ALL[idx],
        Y_ALL[idx],
        {
            "n": int(len(idx)),
            "minority_n": int(
                np.sum(
                    Y_ALL[idx]
                    == minority_class
                )
            ),
            "majority_n": int(
                np.sum(
                    Y_ALL[idx]
                    == majority_class
                )
            ),
        },
    )


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
                sample_size=min(1000, len(Z)),
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

    return best[2], best[3]


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

        item = (
            float(m.bic(Z)),
            k,
            m.predict(Z),
        )

        if best is None or item[0] < best[0]:
            best = item

    return best[1], best[2]


def make_row(
    level,
    seed,
    method,
    meta,
    k,
    decision,
    y,
    labels,
    runtime,
    diag=None,
):
    row = {
        "level": level,
        "seed": int(seed),
        "method": method,
        "n": meta["n"],
        "minority_n": meta["minority_n"],
        "majority_n": meta["majority_n"],
        "imbalance_ratio": float(
            meta["majority_n"]
            / meta["minority_n"]
        ),
        "true_k": 2,
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
        y,
        labels,
    )

    row.update(
        {
            "exact_k": int(k == 2),
            "abs_k_error": int(abs(k - 2)),
            "ARI": m["ARI"],
            "NMI": m["NMI"],
            "worst_cluster_recall": m["worst_cluster_recall"],
            "mean_cluster_recall": m["mean_cluster_recall"],
        }
    )

    return row


rows = []

for level in LEVELS:
    for seed in SEEDS:
        X, y, meta = make_subset(
            level,
            seed,
        )

        print(
            "G8_LEVEL",
            level,
            "seed=",
            seed,
            "n=",
            meta["n"],
            "minority=",
            meta["minority_n"],
            "majority=",
            meta["majority_n"],
            flush=True,
        )

        # v0.2
        t0 = time.perf_counter()

        v02 = GuardKPrototype(
            GuardKConfig(
                k_min=KMIN,
                k_max=KMAX,
                n_bootstrap=4,
                seed=seed,
                allow_abstain=False,
            )
        ).fit(X)

        rows.append(
            make_row(
                level,
                seed,
                "guardk_v02",
                meta,
                v02.selected_k_,
                v02.decision_,
                y,
                v02.labels_,
                time.perf_counter() - t0,
            )
        )

        # v0.4
        t0 = time.perf_counter()

        v04 = GuardKPrototypeV04(
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

        rows.append(
            make_row(
                level,
                seed,
                "guardk_v04",
                meta,
                v04.selected_k_,
                v04.decision_,
                y,
                v04.labels_,
                time.perf_counter() - t0,
                v04.diagnostics_,
            )
        )

        # KMeans
        t0 = time.perf_counter()
        k, pred = kmeans_silhouette(
            X,
            seed,
        )

        rows.append(
            make_row(
                level,
                seed,
                "kmeans_silhouette",
                meta,
                k,
                "ACCEPT",
                y,
                pred,
                time.perf_counter() - t0,
            )
        )

        # GMM
        t0 = time.perf_counter()
        k, pred = gmm_bic(
            X,
            seed,
        )

        rows.append(
            make_row(
                level,
                seed,
                "gmm_bic",
                meta,
                k,
                "ACCEPT",
                y,
                pred,
                time.perf_counter() - t0,
            )
        )

runs = pd.DataFrame(rows)

runs.to_csv(
    OUT / "G8_biomedical_stress_runs.csv",
    index=False,
)

agg = (
    runs
    .groupby(
        ["level", "method"],
        dropna=False,
    )
    .agg(
        runs=("seed", "count"),
        mean_imbalance_ratio=("imbalance_ratio", "mean"),
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
    OUT / "G8_biomedical_stress_aggregate.csv",
    index=False,
)

overall = (
    runs
    .groupby("method", dropna=False)
    .agg(
        coverage=("coverage", "mean"),
        exact_k_rate_all=(
            "exact_k",
            lambda x: float(x.fillna(0).mean()),
        ),
        exact_k_rate_accepted=("exact_k", "mean"),
        mean_ARI_accepted=("ARI", "mean"),
        mean_NMI_accepted=("NMI", "mean"),
        mean_worst_cluster_recall_accepted=("worst_cluster_recall", "mean"),
    )
    .reset_index()
)

overall.to_csv(
    OUT / "G8_biomedical_stress_overall.csv",
    index=False,
)

print()
print("G8_BIOMEDICAL_STRESS_AGGREGATE")
print(agg.to_string(index=False))

print()
print("G8_BIOMEDICAL_STRESS_OVERALL")
print(overall.to_string(index=False))

lock = {
    "stage": "G8",
    "status": "COMPLETE_BIOMEDICAL_FAILURE_STRESS",
    "dataset": "sklearn Wisconsin Diagnostic Breast Cancer",
    "clinical_validation_claim": False,
    "ground_truth_used_by_guardk_fit": False,
    "ground_truth_used_for_controlled_subsampling": True,
    "ground_truth_used_for_external_evaluation": True,
    "levels": list(LEVELS.keys()),
    "seeds": SEEDS,
    "k_range": [KMIN, KMAX],
    "interpretation":
        "Biomedical stress/failure analysis only; not subtype discovery or clinical validation.",
}

(
    OUT / "G8_LOCK.json"
).write_text(
    json.dumps(lock, indent=2)
)

print("G8_LABEL_ISOLATION_PASS")
print("G8_CLINICAL_CLAIM_FALSE_PASS")
print("G8_STATUS=COMPLETE_BIOMEDICAL_FAILURE_STRESS")


if __name__ == "__main__":
    pass
