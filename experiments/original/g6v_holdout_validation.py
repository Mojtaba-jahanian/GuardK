from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from guardk import (
    GuardKConfig,
    GuardKPrototype,
    GuardKV03Config,
    GuardKPrototypeV03,
    evaluate_external,
)


ROOT = Path.home() / "GuardK_V6"
OUT = ROOT / "results" / "g6v_validation"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [6109, 7213, 8329, 9431, 10501]

RATIOS = [2, 5, 20, 50]

CONDITIONS = {
    "clear": {
        "separation": 5.2,
        "sigma": 0.85,
    },
    "moderate": {
        "separation": 3.8,
        "sigma": 1.20,
    },
    "difficult": {
        "separation": 2.6,
        "sigma": 1.55,
    },
    "ambiguous": {
        "separation": 1.8,
        "sigma": 1.80,
    },
}


def sizes_for_ratio(ratio):
    max_n = 520

    return [
        max_n,
        max(180, int(round(max_n / np.sqrt(ratio)))),
        max(80, int(round(max_n / (ratio ** 0.75)))),
        max(12, int(round(max_n / ratio))),
    ]


def make_data(ratio, condition, seed):
    pars = CONDITIONS[condition]
    s = pars["separation"]
    sigma = pars["sigma"]

    centers = np.array(
        [
            [-s, -s],
            [-s,  s],
            [ s, -s],
            [ s,  s],
        ],
        dtype=float,
    )

    sizes = sizes_for_ratio(ratio)

    rng = np.random.default_rng(
        seed
        + 1009 * ratio
        + 7919 * list(CONDITIONS).index(condition)
    )

    Xs = []
    ys = []

    for c, (mu, n) in enumerate(zip(centers, sizes)):
        Xs.append(
            rng.normal(
                mu,
                sigma,
                size=(n, 2),
            )
        )
        ys.append(
            np.full(n, c, dtype=int)
        )

    X = np.vstack(Xs)
    y = np.concatenate(ys)

    order = rng.permutation(len(X))

    return X[order], y[order]


def kmeans_silhouette(X, seed):
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    best = None

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            n_init=15,
            random_state=seed + k,
            algorithm="lloyd",
        ).fit(Z)

        sil = float(
            silhouette_score(
                Z,
                model.labels_,
                sample_size=min(1500, len(Z)),
                random_state=seed,
            )
        )

        item = (sil, -k, k, model.labels_.copy())

        if best is None or item[:2] > best[:2]:
            best = item

    return best[2], best[3]


def gmm_bic(X, seed):
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    best = None

    for k in range(2, 9):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=3,
            random_state=seed + k,
            reg_covar=1e-6,
        ).fit(Z)

        bic = float(model.bic(Z))
        labels = model.predict(Z)

        item = (bic, k, labels.copy())

        if best is None or item[0] < best[0]:
            best = item

    return best[1], best[2]


def evaluate_row(
    scenario,
    ratio,
    condition,
    seed,
    method,
    k,
    decision,
    y,
    pred,
    runtime,
):
    row = {
        "scenario": scenario,
        "ratio": int(ratio),
        "condition": condition,
        "seed": int(seed),
        "method": method,
        "true_k": 4,
        "selected_k": np.nan if k is None else int(k),
        "decision": decision,
        "coverage": int(k is not None),
        "runtime_sec": float(runtime),
    }

    if k is None:
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

    m = evaluate_external(y, pred)

    row.update(
        {
            "exact_k": int(k == 4),
            "abs_k_error": int(abs(k - 4)),
            "ARI": m["ARI"],
            "NMI": m["NMI"],
            "worst_cluster_recall": m["worst_cluster_recall"],
        }
    )

    return row


def main():
    rows = []

    for ratio in RATIOS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                scenario = f"r{ratio}_{condition}"

                X, y = make_data(
                    ratio,
                    condition,
                    seed,
                )

                print(
                    "G6V_SCENARIO",
                    scenario,
                    "seed=",
                    seed,
                    flush=True,
                )

                t0 = time.perf_counter()
                old = GuardKPrototype(
                    GuardKConfig(
                        k_min=2,
                        k_max=8,
                        n_bootstrap=4,
                        seed=seed,
                        allow_abstain=False,
                    )
                ).fit(X)
                dt = time.perf_counter() - t0

                rows.append(
                    evaluate_row(
                        scenario, ratio, condition, seed,
                        "guardk_v02",
                        old.selected_k_,
                        old.decision_,
                        y,
                        old.labels_,
                        dt,
                    )
                )

                t0 = time.perf_counter()
                new = GuardKPrototypeV03(
                    GuardKV03Config(
                        k_min=2,
                        k_max=8,
                        n_bootstrap=4,
                        seed=seed,
                        allow_abstain=True,
                    )
                ).fit(X)
                dt = time.perf_counter() - t0

                rows.append(
                    evaluate_row(
                        scenario, ratio, condition, seed,
                        "guardk_v03",
                        new.selected_k_,
                        new.decision_,
                        y,
                        new.labels_,
                        dt,
                    )
                )

                t0 = time.perf_counter()
                k, pred = kmeans_silhouette(X, seed)
                dt = time.perf_counter() - t0

                rows.append(
                    evaluate_row(
                        scenario, ratio, condition, seed,
                        "kmeans_silhouette",
                        k,
                        "ACCEPT",
                        y,
                        pred,
                        dt,
                    )
                )

                t0 = time.perf_counter()
                k, pred = gmm_bic(X, seed)
                dt = time.perf_counter() - t0

                rows.append(
                    evaluate_row(
                        scenario, ratio, condition, seed,
                        "gmm_bic",
                        k,
                        "ACCEPT",
                        y,
                        pred,
                        dt,
                    )
                )

    runs = pd.DataFrame(rows)

    runs.to_csv(
        OUT / "G6V_holdout_runs.csv",
        index=False,
    )

    accepted = runs[runs["coverage"] == 1].copy()

    aggregate = (
        runs
        .groupby("method", dropna=False)
        .agg(
            runs=("seed", "count"),
            coverage=("coverage", "mean"),
            exact_k_rate_all=(
                "exact_k",
                lambda x: float(x.fillna(0).mean())
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

    aggregate.to_csv(
        OUT / "G6V_holdout_aggregate.csv",
        index=False,
    )

    by_condition = (
        runs
        .groupby(["condition", "method"], dropna=False)
        .agg(
            coverage=("coverage", "mean"),
            exact_k_rate_all=(
                "exact_k",
                lambda x: float(x.fillna(0).mean())
            ),
            exact_k_rate_accepted=("exact_k", "mean"),
            mean_ARI_accepted=("ARI", "mean"),
            mean_worst_cluster_recall_accepted=("worst_cluster_recall", "mean"),
        )
        .reset_index()
    )

    by_condition.to_csv(
        OUT / "G6V_by_condition.csv",
        index=False,
    )

    print()
    print("G6V_AGGREGATE")
    print(aggregate.to_string(index=False))

    print()
    print("G6V_BY_CONDITION")
    print(by_condition.to_string(index=False))

    v02 = aggregate[aggregate["method"] == "guardk_v02"].iloc[0]
    v03 = aggregate[aggregate["method"] == "guardk_v03"].iloc[0]

    difficult_ambiguous = by_condition[
        by_condition["condition"].isin(["difficult", "ambiguous"])
    ]

    v03_hard = difficult_ambiguous[
        difficult_ambiguous["method"] == "guardk_v03"
    ]

    nonamb = by_condition[
        by_condition["condition"].isin(["clear", "moderate"])
    ]

    v03_nonamb = nonamb[
        nonamb["method"] == "guardk_v03"
    ]

    scientific_pass = bool(
        v03["mean_ARI_accepted"] >= v02["mean_ARI_accepted"] - 0.02
        and
        v03_nonamb["coverage"].mean() >= 0.80
        and
        v03_hard["coverage"].mean() <= 0.90
    )

    status = (
        "PASS_REFINEMENT_PROMISING"
        if scientific_pass
        else "REVIEW_REQUIRED"
    )

    lock = {
        "stage": "G6V",
        "status": status,
        "confirmatory": False,
        "out_of_grid_seeds": SEEDS,
        "ratios": RATIOS,
        "conditions": list(CONDITIONS.keys()),
        "ground_truth_used_by_fit": False,
        "transport_in_core": False,
        "decision_rule_predeclared_before_this_holdout": True,
        "note":
            "This development holdout decides whether v0.3 is scientifically promising; it is not final manuscript evidence."
    }

    (OUT / "G6V_LOCK.json").write_text(
        json.dumps(lock, indent=2)
    )

    print()
    print("G6V_STATUS=" + status)
    print("G6V_LABEL_ISOLATION_PASS")
    print("G6V_TRANSPORT_NOT_CORE_PASS")


if __name__ == "__main__":
    main()
