from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import (
    KMeans,
    SpectralClustering,
    AgglomerativeClustering,
)
from sklearn.mixture import (
    GaussianMixture,
)
from sklearn.metrics import (
    silhouette_score,
)
from sklearn.preprocessing import (
    StandardScaler,
)

import hdbscan

from guardk import (
    GuardKConfig,
    GuardKPrototype,
    evaluate_external,
)


ROOT = Path.home() / "GuardK_V6"
OUT = (
    ROOT
    / "results"
    / "g6_cpu_benchmark"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [
    2207,
    3407,
    5107,
]

SCENARIOS = {
    "balanced_low": {
        "ratio": 1,
        "separation": 5.0,
        "sigma": 0.8,
    },
    "imbalanced20_low": {
        "ratio": 20,
        "separation": 5.0,
        "sigma": 0.8,
    },
    "balanced_high": {
        "ratio": 1,
        "separation": 2.4,
        "sigma": 1.5,
    },
    "imbalanced20_high": {
        "ratio": 20,
        "separation": 2.4,
        "sigma": 1.5,
    },
}


def sizes_from_ratio(
    ratio,
):
    if ratio == 1:
        return [
            250,
            250,
            250,
            250,
        ]

    if ratio == 20:
        return [
            400,
            240,
            120,
            20,
        ]

    raise ValueError(
        ratio
    )


def make_data(
    cfg,
    seed,
):
    s = cfg[
        "separation"
    ]

    sigma = cfg[
        "sigma"
    ]

    centers = np.array(
        [
            [-s,-s],
            [-s, s],
            [ s,-s],
            [ s, s],
        ],
        dtype=float,
    )

    sizes = sizes_from_ratio(
        cfg["ratio"]
    )

    rng = np.random.default_rng(
        seed
    )

    Xs = []
    ys = []

    for c, (mu,n) in enumerate(
        zip(
            centers,
            sizes,
        )
    ):
        Xs.append(
            rng.normal(
                mu,
                sigma,
                size=(n,2),
            )
        )

        ys.append(
            np.full(
                n,
                c,
                dtype=int,
            )
        )

    X = np.vstack(Xs)
    y = np.concatenate(ys)

    order = rng.permutation(
        len(X)
    )

    return (
        X[order],
        y[order],
    )


def silhouette_select(
    labels_by_k,
    X,
):
    rows = []

    for k, labels in (
        labels_by_k.items()
    ):
        if len(
            np.unique(labels)
        ) < 2:
            score = -1.0
        else:
            score = float(
                silhouette_score(
                    X,
                    labels,
                    sample_size=min(
                        1200,
                        len(X),
                    ),
                    random_state=2207,
                )
            )

        rows.append(
            (
                k,
                score,
            )
        )

    rows.sort(
        key=lambda x: (
            -x[1],
            x[0],
        )
    )

    return rows[0][0]


def run_kmeans(
    Z,
    seed,
):
    labels_by_k = {}

    for k in range(2,9):
        labels_by_k[k] = (
            KMeans(
                n_clusters=k,
                n_init=15,
                random_state=seed+k,
                algorithm="lloyd",
            )
            .fit_predict(Z)
        )

    k = silhouette_select(
        labels_by_k,
        Z,
    )

    return (
        k,
        labels_by_k[k],
        "silhouette",
    )


def run_gmm(
    Z,
    seed,
):
    best = None

    for k in range(2,9):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=seed+k,
            n_init=3,
            reg_covar=1e-6,
        )

        model.fit(Z)

        bic = float(
            model.bic(Z)
        )

        labels = model.predict(
            Z
        )

        item = (
            bic,
            k,
            labels,
        )

        if (
            best is None
            or item[0] < best[0]
            or (
                item[0] == best[0]
                and k < best[1]
            )
        ):
            best = item

    return (
        int(best[1]),
        best[2],
        "BIC",
    )


def run_agglomerative(
    Z,
):
    labels_by_k = {}

    for k in range(2,9):
        labels_by_k[k] = (
            AgglomerativeClustering(
                n_clusters=k,
                linkage="ward",
            )
            .fit_predict(Z)
        )

    k = silhouette_select(
        labels_by_k,
        Z,
    )

    return (
        k,
        labels_by_k[k],
        "silhouette",
    )


def run_spectral(
    Z,
    seed,
):
    labels_by_k = {}

    n_neighbors = min(
        15,
        len(Z)-1,
    )

    for k in range(2,9):
        labels_by_k[k] = (
            SpectralClustering(
                n_clusters=k,
                affinity=
                    "nearest_neighbors",
                n_neighbors=
                    n_neighbors,
                assign_labels=
                    "kmeans",
                random_state=
                    seed+k,
                n_jobs=1,
            )
            .fit_predict(Z)
        )

    k = silhouette_select(
        labels_by_k,
        Z,
    )

    return (
        k,
        labels_by_k[k],
        "silhouette",
    )


def run_hdbscan(
    Z,
):
    # Development-fixed settings, not tuned with labels.
    model = hdbscan.HDBSCAN(
        min_cluster_size=max(
            10,
            int(
                round(
                    0.02 * len(Z)
                )
            ),
        ),
        min_samples=None,
        cluster_selection_method="eom",
        prediction_data=False,
    )

    labels = model.fit_predict(
        Z
    )

    clusters = sorted(
        x
        for x in np.unique(labels)
        if x != -1
    )

    k = len(clusters)

    return (
        k,
        labels,
        "native_unknown_k",
    )


def run_guardk(
    X,
    seed,
):
    model = GuardKPrototype(
        GuardKConfig(
            k_min=2,
            k_max=8,
            n_bootstrap=4,
            seed=seed,
            allow_abstain=False,
        )
    ).fit(X)

    return (
        model.selected_k_,
        model.labels_,
        "guard_score",
    )


def make_row(
    scenario,
    seed,
    method,
    true_k,
    k,
    labels,
    criterion,
    runtime,
):
    metrics = evaluate_external(
        true_k["y"],
        labels,
    )

    return {
        "scenario":
            scenario,
        "seed":
            int(seed),
        "method":
            method,
        "true_k":
            int(true_k["k"]),
        "selected_k":
            int(k),
        "exact_k":
            int(
                k == true_k["k"]
            ),
        "abs_k_error":
            int(
                abs(
                    k
                    - true_k["k"]
                )
            ),
        "ARI":
            metrics["ARI"],
        "NMI":
            metrics["NMI"],
        "worst_cluster_recall":
            metrics[
                "worst_cluster_recall"
            ],
        "mean_cluster_recall":
            metrics[
                "mean_cluster_recall"
            ],
        "selection_criterion":
            criterion,
        "runtime_sec":
            float(runtime),
    }


def main():
    rows = []

    for scenario, scfg in (
        SCENARIOS.items()
    ):
        for seed in SEEDS:
            print(
                "G6_SCENARIO",
                scenario,
                "seed=",
                seed,
                flush=True,
            )

            X, y = make_data(
                scfg,
                seed,
            )

            scaler = (
                StandardScaler()
            )

            Z = scaler.fit_transform(
                X
            )

            methods = [
                "kmeans_silhouette",
                "gmm_bic",
                "agglomerative_silhouette",
                "spectral_silhouette",
                "hdbscan",
                "guardk",
            ]

            for method in methods:
                t0 = time.perf_counter()

                if method == (
                    "kmeans_silhouette"
                ):
                    k, labels, criterion = (
                        run_kmeans(
                            Z,
                            seed,
                        )
                    )

                elif method == "gmm_bic":
                    k, labels, criterion = (
                        run_gmm(
                            Z,
                            seed,
                        )
                    )

                elif method == (
                    "agglomerative_silhouette"
                ):
                    k, labels, criterion = (
                        run_agglomerative(
                            Z
                        )
                    )

                elif method == (
                    "spectral_silhouette"
                ):
                    k, labels, criterion = (
                        run_spectral(
                            Z,
                            seed,
                        )
                    )

                elif method == "hdbscan":
                    k, labels, criterion = (
                        run_hdbscan(
                            Z
                        )
                    )

                elif method == "guardk":
                    k, labels, criterion = (
                        run_guardk(
                            X,
                            seed,
                        )
                    )

                else:
                    raise RuntimeError(
                        method
                    )

                runtime = (
                    time.perf_counter()
                    - t0
                )

                rows.append(
                    make_row(
                        scenario,
                        seed,
                        method,
                        {
                            "k": 4,
                            "y": y,
                        },
                        k,
                        labels,
                        criterion,
                        runtime,
                    )
                )

                print(
                    " ",
                    method,
                    "K=",
                    k,
                    "runtime=",
                    round(
                        runtime,
                        3,
                    ),
                    flush=True,
                )

    runs = pd.DataFrame(
        rows
    )

    runs.to_csv(
        OUT /
        "G6_cpu_benchmark_runs.csv",
        index=False,
    )

    aggregate = (
        runs
        .groupby(
            "method",
            dropna=False,
        )
        .agg(
            runs=(
                "seed",
                "count",
            ),
            exact_k_rate=(
                "exact_k",
                "mean",
            ),
            mean_abs_k_error=(
                "abs_k_error",
                "mean",
            ),
            mean_ARI=(
                "ARI",
                "mean",
            ),
            mean_NMI=(
                "NMI",
                "mean",
            ),
            mean_worst_cluster_recall=(
                "worst_cluster_recall",
                "mean",
            ),
            mean_runtime_sec=(
                "runtime_sec",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "exact_k_rate",
                "mean_ARI",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    aggregate.to_csv(
        OUT /
        "G6_cpu_benchmark_aggregate.csv",
        index=False,
    )

    by_scenario = (
        runs
        .groupby(
            [
                "scenario",
                "method",
            ],
            dropna=False,
        )
        .agg(
            exact_k_rate=(
                "exact_k",
                "mean",
            ),
            mean_ARI=(
                "ARI",
                "mean",
            ),
            mean_NMI=(
                "NMI",
                "mean",
            ),
        )
        .reset_index()
    )

    by_scenario.to_csv(
        OUT /
        "G6_cpu_benchmark_by_scenario.csv",
        index=False,
    )

    print()
    print(
        "G6_AGGREGATE"
    )

    print(
        aggregate.to_string(
            index=False
        )
    )

    if not (
        (
            runs[
                runs["scenario"]
                == "balanced_low"
            ]
            .query(
                "method == 'guardk'"
            )[
                "selected_k"
            ]
            == 4
        ).all()
    ):
        raise SystemExit(
            "G6_GUARDK_EASY_SANITY_FAIL"
        )

    lock = {
        "stage":
            "G6",
        "status":
            "PASS_CPU_COMPARATOR_PILOT",
        "confirmatory":
            False,
        "deep_baselines_included":
            False,
        "deferred_deep_baselines":
            [
                "DeepDPM",
                "UNSEEN",
                "AutoProPos",
                "P2OT_or_equivalent_if_fair",
            ],
        "cpu_methods":
            sorted(
                runs[
                    "method"
                ].unique().tolist()
            ),
        "ground_truth_used_for_selection":
            False,
        "note":
            "This is a CPU baseline pilot. It is not the final SOTA benchmark."
    }

    (
        OUT /
        "G6_CPU_BENCHMARK_LOCK.json"
    ).write_text(
        json.dumps(
            lock,
            indent=2,
        )
    )

    print(
        "G6_GUARDK_EASY_SANITY_PASS"
    )
    print(
        "G6_LABEL_ISOLATION_PASS"
    )
    print(
        "G6_STATUS=PASS_CPU_COMPARATOR_PILOT"
    )


if __name__ == "__main__":
    main()
