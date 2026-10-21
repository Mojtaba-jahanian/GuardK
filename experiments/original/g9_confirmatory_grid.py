from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import hdbscan

from guardk import (
    GuardKConfig,
    GuardKPrototype,
    GuardKV04Config,
    GuardKPrototypeV04,
    evaluate_external,
)

ROOT = Path.home() / "GuardK_V6"
OUT = ROOT / "results/g9_confirmatory"
OUT.mkdir(parents=True, exist_ok=True)

K_VALUES = [4, 8]
RATIOS = [1, 5, 20, 50]
OVERLAPS = {
    "low": (7.0, 0.75),
    "medium": (5.0, 1.15),
    "high": (3.2, 1.55),
}
NOISE_FRACTIONS = [0.00, 0.10]
SAMPLE_SIZES = [500, 2000]
SEEDS = [25117, 26119, 27127]
D = 8
KMIN = 2
KMAX = 12


def make_centers(k: int, d: int, radius: float) -> np.ndarray:
    # Deterministic approximately separated centers on a sinusoidal frame.
    j = np.arange(k)[:, None]
    q = np.arange(d)[None, :]
    base = (
        np.sin((j + 1) * (q + 1) * 0.73)
        + np.cos((j + 1) * (q + 1) * 1.17)
    )
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return radius * base / norms


def imbalanced_sizes(n: int, k: int, ratio: int) -> np.ndarray:
    if ratio == 1:
        raw = np.ones(k, dtype=float)
    else:
        raw = np.geomspace(1.0, 1.0 / ratio, num=k)

    raw = raw / raw.sum()
    sizes = np.maximum(5, np.floor(raw * n).astype(int))

    diff = n - int(sizes.sum())

    order = np.argsort(-raw)

    i = 0
    while diff != 0:
        idx = int(order[i % len(order)])

        if diff > 0:
            sizes[idx] += 1
            diff -= 1
        else:
            if sizes[idx] > 5:
                sizes[idx] -= 1
                diff += 1

        i += 1

        if i > 100000:
            raise RuntimeError("size allocation failed")

    return sizes


def make_data(
    *,
    true_k: int,
    ratio: int,
    overlap: str,
    noise_fraction: float,
    n: int,
    seed: int,
):
    radius, sigma = OVERLAPS[overlap]
    centers = make_centers(true_k, D, radius)
    sizes = imbalanced_sizes(n, true_k, ratio)

    rng = np.random.default_rng(
        seed
        + 1009 * true_k
        + 2003 * ratio
        + 4001 * list(OVERLAPS).index(overlap)
        + 8009 * int(round(noise_fraction * 100))
        + 16001 * n
    )

    Xs = []
    ys = []

    for c, (mu, nc) in enumerate(zip(centers, sizes)):
        Xs.append(
            rng.normal(
                loc=mu,
                scale=sigma,
                size=(int(nc), D),
            )
        )

        ys.append(
            np.full(
                int(nc),
                c,
                dtype=int,
            )
        )

    X = np.vstack(Xs)
    y = np.concatenate(ys)

    if noise_fraction > 0:
        m = max(
            1,
            int(
                round(
                    noise_fraction * len(X)
                )
            ),
        )

        idx = rng.choice(
            len(X),
            size=m,
            replace=False,
        )

        # Feature corruption while preserving external class identity.
        X[idx] += rng.normal(
            loc=0.0,
            scale=2.5 * sigma,
            size=(m, D),
        )

    order = rng.permutation(len(X))

    return X[order], y[order], sizes


def km_silhouette(X, seed):
    Z = StandardScaler().fit_transform(X)
    best = None

    for k in range(KMIN, KMAX + 1):
        m = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=seed + k,
            algorithm="lloyd",
        ).fit(Z)

        sil = float(
            silhouette_score(
                Z,
                m.labels_,
                sample_size=min(1200, len(Z)),
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
            covariance_type="diag",
            n_init=2,
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


def agglom_silhouette(X, seed):
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
                sample_size=min(1200, len(Z)),
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

    return best[2], best[3]


def hdbscan_unknown(X):
    Z = StandardScaler().fit_transform(X)

    m = hdbscan.HDBSCAN(
        min_cluster_size=max(
            10,
            int(round(0.02 * len(Z))),
        ),
        cluster_selection_method="eom",
    )

    labels = m.fit_predict(Z)

    k = len(
        [
            z
            for z in np.unique(labels)
            if z != -1
        ]
    )

    return k, labels


def v02(X, seed):
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


def v04(X, seed):
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


def external_row(
    *,
    true_k,
    ratio,
    overlap,
    noise_fraction,
    n,
    seed,
    method,
    selected_k,
    labels,
    decision,
    runtime_sec,
    y,
    diag=None,
):
    row = {
        "true_k": int(true_k),
        "imbalance_ratio": int(ratio),
        "overlap": overlap,
        "noise_fraction": float(noise_fraction),
        "n": int(n),
        "seed": int(seed),
        "method": method,
        "selected_k": np.nan if selected_k is None else int(selected_k),
        "coverage": int(selected_k is not None),
        "decision": decision,
        "runtime_sec": float(runtime_sec),
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

    if selected_k is None:
        row.update(
            {
                "exact_k": np.nan,
                "abs_k_error": np.nan,
                "ARI": np.nan,
                "NMI": np.nan,
                "worst_cluster_recall": np.nan,
            }
        )
        return row

    m = evaluate_external(
        y,
        labels,
    )

    row.update(
        {
            "exact_k": int(selected_k == true_k),
            "abs_k_error": int(abs(selected_k - true_k)),
            "ARI": m["ARI"],
            "NMI": m["NMI"],
            "worst_cluster_recall": m["worst_cluster_recall"],
        }
    )

    return row


rows = []

total = (
    len(K_VALUES)
    * len(RATIOS)
    * len(OVERLAPS)
    * len(NOISE_FRACTIONS)
    * len(SAMPLE_SIZES)
    * len(SEEDS)
)

case_id = 0

for true_k in K_VALUES:
    for ratio in RATIOS:
        for overlap in OVERLAPS:
            for noise_fraction in NOISE_FRACTIONS:
                for n in SAMPLE_SIZES:
                    for seed in SEEDS:
                        case_id += 1

                        print(
                            f"G9_CASE {case_id}/{total}",
                            "K=",
                            true_k,
                            "ratio=",
                            ratio,
                            "overlap=",
                            overlap,
                            "noise=",
                            noise_fraction,
                            "n=",
                            n,
                            "seed=",
                            seed,
                            flush=True,
                        )

                        X, y, sizes = make_data(
                            true_k=true_k,
                            ratio=ratio,
                            overlap=overlap,
                            noise_fraction=noise_fraction,
                            n=n,
                            seed=seed,
                        )

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
                                k, labels, decision, diag = v02(X, seed)

                            elif method == "guardk_v04":
                                k, labels, decision, diag = v04(X, seed)

                            elif method == "kmeans_silhouette":
                                k, labels = km_silhouette(X, seed)
                                decision, diag = "ACCEPT", None

                            elif method == "gmm_bic":
                                k, labels = gmm_bic(X, seed)
                                decision, diag = "ACCEPT", None

                            elif method == "agglomerative_silhouette":
                                k, labels = agglom_silhouette(X, seed)
                                decision, diag = "ACCEPT", None

                            elif method == "hdbscan":
                                k, labels = hdbscan_unknown(X)
                                decision, diag = "ACCEPT", None

                            else:
                                raise RuntimeError(method)

                            rows.append(
                                external_row(
                                    true_k=true_k,
                                    ratio=ratio,
                                    overlap=overlap,
                                    noise_fraction=noise_fraction,
                                    n=n,
                                    seed=seed,
                                    method=method,
                                    selected_k=k,
                                    labels=labels,
                                    decision=decision,
                                    runtime_sec=time.perf_counter() - t0,
                                    y=y,
                                    diag=diag,
                                )
                            )

runs = pd.DataFrame(rows)

runs.to_csv(
    OUT / "G9_confirmatory_runs.csv",
    index=False,
)

agg = (
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
        mean_abs_k_error_accepted=("abs_k_error", "mean"),
        mean_ARI_accepted=("ARI", "mean"),
        mean_NMI_accepted=("NMI", "mean"),
        mean_worst_cluster_recall_accepted=("worst_cluster_recall", "mean"),
        mean_runtime_sec=("runtime_sec", "mean"),
    )
    .reset_index()
)

agg.to_csv(
    OUT / "G9_confirmatory_aggregate.csv",
    index=False,
)

by_overlap = (
    runs
    .groupby(["overlap", "method"], dropna=False)
    .agg(
        coverage=("coverage", "mean"),
        exact_k_rate_all=(
            "exact_k",
            lambda x: float(x.fillna(0).mean()),
        ),
        exact_k_rate_accepted=("exact_k", "mean"),
        mean_ARI_accepted=("ARI", "mean"),
        mean_worst_cluster_recall_accepted=("worst_cluster_recall", "mean"),
    )
    .reset_index()
)

by_overlap.to_csv(
    OUT / "G9_by_overlap.csv",
    index=False,
)

by_ratio = (
    runs
    .groupby(["imbalance_ratio", "method"], dropna=False)
    .agg(
        coverage=("coverage", "mean"),
        exact_k_rate_all=(
            "exact_k",
            lambda x: float(x.fillna(0).mean()),
        ),
        exact_k_rate_accepted=("exact_k", "mean"),
        mean_ARI_accepted=("ARI", "mean"),
        mean_worst_cluster_recall_accepted=("worst_cluster_recall", "mean"),
    )
    .reset_index()
)

by_ratio.to_csv(
    OUT / "G9_by_ratio.csv",
    index=False,
)


def bootstrap_ci(
    values,
    *,
    stat=np.mean,
    seed=919191,
    B=2000,
):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    stats = np.empty(B, dtype=float)

    for b in range(B):
        sample = rng.choice(
            x,
            size=len(x),
            replace=True,
        )
        stats[b] = stat(sample)

    return (
        float(np.quantile(stats, 0.025)),
        float(np.quantile(stats, 0.975)),
    )


ci_rows = []

for method, d in runs.groupby("method"):
    coverage = d["coverage"].to_numpy(dtype=float)

    exact_all = (
        d["exact_k"]
        .fillna(0)
        .to_numpy(dtype=float)
    )

    accepted = d[
        d["coverage"] == 1
    ]

    metrics = {
        "coverage": coverage,
        "exact_k_rate_all": exact_all,
        "exact_k_rate_accepted":
            accepted["exact_k"].to_numpy(dtype=float),
        "ARI_accepted":
            accepted["ARI"].to_numpy(dtype=float),
        "NMI_accepted":
            accepted["NMI"].to_numpy(dtype=float),
        "worst_cluster_recall_accepted":
            accepted["worst_cluster_recall"].to_numpy(dtype=float),
    }

    for metric, values in metrics.items():
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]

        mean = float(np.mean(vals)) if len(vals) else np.nan
        lo, hi = bootstrap_ci(vals)

        ci_rows.append(
            {
                "method": method,
                "metric": metric,
                "mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "n": int(len(vals)),
            }
        )

cis = pd.DataFrame(ci_rows)

cis.to_csv(
    OUT / "G9_confirmatory_bootstrap_CI.csv",
    index=False,
)

v02 = agg[
    agg["method"] == "guardk_v02"
].iloc[0]

v04 = agg[
    agg["method"] == "guardk_v04"
].iloc[0]

# Predeclared interpretation gates for the selective claim.
gates = {
    "coverage_ge_0.45":
        bool(v04["coverage"] >= 0.45),

    "accepted_exact_k_gain_ge_0.08_vs_v02":
        bool(
            v04["exact_k_rate_accepted"]
            >= v02["exact_k_rate_accepted"] + 0.08
        ),

    "accepted_ari_not_worse_vs_v02_minus_0.02":
        bool(
            v04["mean_ARI_accepted"]
            >= v02["mean_ARI_accepted"] - 0.02
        ),

    "accepted_worst_recall_not_worse_vs_v02_minus_0.02":
        bool(
            v04["mean_worst_cluster_recall_accepted"]
            >= v02["mean_worst_cluster_recall_accepted"] - 0.02
        ),
}

status = (
    "PASS_CONFIRMATORY_SELECTIVE_CLAIM"
    if all(gates.values())
    else "CONFIRMATORY_CLAIM_NOT_SUPPORTED"
)

lock = {
    "stage": "G9B",
    "status": status,
    "confirmatory": True,
    "protocol_changed_after_run": False,
    "method_changed_after_run": False,
    "ground_truth_used_by_guardk_fit": False,
    "gates": gates,
    "note":
        "This stage tests the frozen selective claim. It does not establish final SOTA because modern/deep comparators are still deferred.",
}

(
    OUT / "G9_CONFIRMATORY_LOCK.json"
).write_text(
    json.dumps(lock, indent=2)
)

print()
print("G9_CONFIRMATORY_AGGREGATE")
print(agg.to_string(index=False))

print()
print("G9_BY_OVERLAP")
print(by_overlap.to_string(index=False))

print()
print("G9_BY_RATIO")
print(by_ratio.to_string(index=False))

print()
print("G9_BOOTSTRAP_CI")
print(cis.to_string(index=False))

print()
for key, val in gates.items():
    print(
        "GATE",
        key,
        "=",
        val,
    )

print("G9_STATUS=" + status)
print("G9_LABEL_ISOLATION_PASS")
print("G9_PROTOCOL_NO_POSTHOC_CHANGE_PASS")
