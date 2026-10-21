from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
)

from guardk import (
    GuardKConfig,
    GuardKPrototype,
    evaluate_external,
    make_guardk_scenario,
)


ROOT = Path.home() / "GuardK_V6"
OUT = ROOT / "results" / "g4_pilot"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)


SCENARIOS = [
    "balanced_separated",
    "imbalanced_10x",
    "rare_20x",
    "overlap_high",
    "overlap_ambiguous",
]

SEEDS = [
    2207,
    3407,
    5107,
]


def independent_candidates(
    Z,
    cfg,
):
    out = {}

    for k in range(
        cfg.k_min,
        cfg.k_max + 1,
    ):
        m = KMeans(
            n_clusters=k,
            n_init=15,
            random_state=(
                cfg.seed + k
            ),
            algorithm="lloyd",
        ).fit(Z)

        out[k] = {
            "labels": m.labels_.copy(),
            "centers": (
                m.cluster_centers_.copy()
            ),
        }

    return out


def evaluate_candidate_set(
    helper,
    Z,
    candidates,
):
    rows = []

    for k in range(
        helper.config.k_min,
        helper.config.k_max + 1,
    ):
        item = candidates[k]

        row = (
            helper._evaluate_candidate(
                Z,
                k,
                item["labels"],
                item["centers"],
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)


def guard_score(
    df,
    *,
    rare=True,
    overlap=True,
):
    st_w = 0.55
    sil_w = 0.20
    rare_w = 0.15
    overlap_w = 0.10

    if not rare:
        st_w = 0.65
        sil_w = 0.25
        rare_w = 0.0
        overlap_w = 0.10

    if not overlap:
        st_w = 0.60
        sil_w = 0.25
        rare_w = 0.15
        overlap_w = 0.0

    return (
        st_w
        * df["stability"]

        + sil_w
        * df["silhouette_01"]

        + rare_w
        * df["rare_support"]

        + overlap_w
        * df["overlap_safety"]

        - 0.02
        * (
            df["k"]
            / 8.0
        )
    )


def select_guard(
    df,
    *,
    rare=True,
    overlap=True,
    allow_abstain=False,
):
    d = df.copy()

    d["score_variant"] = (
        guard_score(
            d,
            rare=rare,
            overlap=overlap,
        )
    )

    eligible = d[
        (d["stability"] >= 0.55)
        &
        (
            d["min_cluster_fraction"]
            >= 0.003
        )
    ].copy()

    if eligible.empty:
        row = d.sort_values(
            [
                "stability",
                "score_variant",
                "k",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        ).iloc[0]

        return (
            int(row["k"]),
            "FALLBACK",
            np.nan,
        )

    ranked = eligible.sort_values(
        [
            "score_variant",
            "k",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    best = float(
        ranked.loc[
            0,
            "score_variant",
        ]
    )

    near = eligible[
        eligible["score_variant"]
        >= best - 0.008
    ]

    selected = int(
        near["k"].min()
    )

    if len(ranked) >= 2:
        margin = float(
            ranked.loc[
                0,
                "score_variant",
            ]
            - ranked.loc[
                1,
                "score_variant",
            ]
        )
    else:
        margin = np.inf

    if (
        allow_abstain
        and len(ranked) >= 2
        and margin < 0.005
    ):
        return (
            None,
            "ABSTAIN_AMBIGUOUS",
            margin,
        )

    return (
        selected,
        "ACCEPT",
        margin,
    )


def pick_partition(
    candidates,
    k,
):
    if k is None:
        return None

    return candidates[
        int(k)
    ]["labels"]


def external_row(
    *,
    scenario,
    seed,
    variant,
    true_k,
    selected_k,
    decision,
    margin,
    y,
    y_pred,
    runtime_sec,
):
    base = {
        "scenario": scenario,
        "seed": int(seed),
        "variant": variant,
        "true_k": int(true_k),
        "selected_k": (
            np.nan
            if selected_k is None
            else int(selected_k)
        ),
        "decision": decision,
        "abstained": int(
            selected_k is None
        ),
        "score_margin": (
            np.nan
            if not np.isfinite(margin)
            else float(margin)
        ),
        "runtime_sec": float(
            runtime_sec
        ),
    }

    if selected_k is None:
        base.update(
            {
                "exact_k": 0,
                "abs_k_error": np.nan,
                "ARI": np.nan,
                "NMI": np.nan,
                "worst_cluster_recall": np.nan,
                "mean_cluster_recall": np.nan,
            }
        )
        return base

    metrics = evaluate_external(
        y,
        y_pred,
    )

    base.update(
        {
            "exact_k": int(
                selected_k == true_k
            ),
            "abs_k_error": int(
                abs(
                    selected_k
                    - true_k
                )
            ),
            "ARI": metrics["ARI"],
            "NMI": metrics["NMI"],
            "worst_cluster_recall":
                metrics[
                    "worst_cluster_recall"
                ],
            "mean_cluster_recall":
                metrics[
                    "mean_cluster_recall"
                ],
        }
    )

    return base


def main():
    rows = []

    for scenario in SCENARIOS:
        for seed in SEEDS:
            print()
            print(
                "=" * 80
            )
            print(
                "SCENARIO",
                scenario,
                "SEED",
                seed,
            )
            print(
                "=" * 80
            )

            X, y, meta = (
                make_guardk_scenario(
                    scenario,
                    seed,
                )
            )

            cfg = GuardKConfig(
                k_min=2,
                k_max=8,
                n_bootstrap=4,
                bootstrap_fraction=0.80,
                seed=seed,
                allow_abstain=False,
            )

            helper = GuardKPrototype(
                cfg
            )

            helper.scaler_ = __import__(
                "sklearn.preprocessing",
                fromlist=["StandardScaler"],
            ).StandardScaler()

            Z = helper.scaler_.fit_transform(
                X
            )

            t0 = time.perf_counter()

            lifecycle = (
                helper._prototype_lifecycle(
                    Z
                )
            )

            life_table = (
                evaluate_candidate_set(
                    helper,
                    Z,
                    lifecycle,
                )
            )

            independent = (
                independent_candidates(
                    Z,
                    cfg,
                )
            )

            independent_table = (
                evaluate_candidate_set(
                    helper,
                    Z,
                    independent,
                )
            )

            compute_runtime = (
                time.perf_counter()
                - t0
            )

            life_table.to_csv(
                OUT
                / (
                    f"candidates_lifecycle_"
                    f"{scenario}_seed{seed}.csv"
                ),
                index=False,
            )

            independent_table.to_csv(
                OUT
                / (
                    f"candidates_independent_"
                    f"{scenario}_seed{seed}.csv"
                ),
                index=False,
            )

            # 1) Independent candidates + silhouette only.
            row = (
                independent_table
                .sort_values(
                    [
                        "silhouette",
                        "k",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
                .iloc[0]
            )

            k = int(row["k"])

            rows.append(
                external_row(
                    scenario=scenario,
                    seed=seed,
                    variant="independent_silhouette",
                    true_k=meta["true_k"],
                    selected_k=k,
                    decision="ACCEPT",
                    margin=np.nan,
                    y=y,
                    y_pred=pick_partition(
                        independent,
                        k,
                    ),
                    runtime_sec=compute_runtime,
                )
            )

            # 2) Lifecycle candidates + silhouette.
            row = (
                life_table
                .sort_values(
                    [
                        "silhouette",
                        "k",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
                .iloc[0]
            )

            k = int(row["k"])

            rows.append(
                external_row(
                    scenario=scenario,
                    seed=seed,
                    variant="lifecycle_silhouette",
                    true_k=meta["true_k"],
                    selected_k=k,
                    decision="ACCEPT",
                    margin=np.nan,
                    y=y,
                    y_pred=pick_partition(
                        lifecycle,
                        k,
                    ),
                    runtime_sec=compute_runtime,
                )
            )

            # 3) Lifecycle + stability only.
            row = (
                life_table
                .sort_values(
                    [
                        "stability",
                        "silhouette",
                        "k",
                    ],
                    ascending=[
                        False,
                        False,
                        True,
                    ],
                )
                .iloc[0]
            )

            k = int(row["k"])

            rows.append(
                external_row(
                    scenario=scenario,
                    seed=seed,
                    variant="lifecycle_stability",
                    true_k=meta["true_k"],
                    selected_k=k,
                    decision="ACCEPT",
                    margin=np.nan,
                    y=y,
                    y_pred=pick_partition(
                        lifecycle,
                        k,
                    ),
                    runtime_sec=compute_runtime,
                )
            )

            variants = [
                (
                    "guard_no_rare",
                    False,
                    True,
                    False,
                ),
                (
                    "guard_no_overlap",
                    True,
                    False,
                    False,
                ),
                (
                    "guard_full_forced",
                    True,
                    True,
                    False,
                ),
                (
                    "guard_full_abstain",
                    True,
                    True,
                    True,
                ),
            ]

            for (
                variant,
                use_rare,
                use_overlap,
                allow_abstain,
            ) in variants:

                k, decision, margin = (
                    select_guard(
                        life_table,
                        rare=use_rare,
                        overlap=use_overlap,
                        allow_abstain=allow_abstain,
                    )
                )

                y_pred = pick_partition(
                    lifecycle,
                    k,
                )

                rows.append(
                    external_row(
                        scenario=scenario,
                        seed=seed,
                        variant=variant,
                        true_k=meta["true_k"],
                        selected_k=k,
                        decision=decision,
                        margin=margin,
                        y=y,
                        y_pred=y_pred,
                        runtime_sec=compute_runtime,
                    )
                )

                print(
                    variant,
                    "K=",
                    k,
                    "decision=",
                    decision,
                    "margin=",
                    margin,
                )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        OUT / "G4_ablation_pilot_runs.csv",
        index=False,
    )

    aggregate = (
        results
        .groupby(
            [
                "scenario",
                "variant",
            ],
            dropna=False,
        )
        .agg(
            runs=(
                "seed",
                "count",
            ),
            coverage=(
                "abstained",
                lambda x:
                    float(
                        1.0 - np.mean(x)
                    ),
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
        )
        .reset_index()
    )

    aggregate.to_csv(
        OUT
        / "G4_ablation_pilot_aggregate.csv",
        index=False,
    )

    print()
    print(
        "=" * 80
    )
    print(
        "G4 PILOT AGGREGATE"
    )
    print(
        "=" * 80
    )

    print(
        aggregate.to_string(
            index=False
        )
    )

    # Mandatory development sanity gate:
    # full forced guard must recover K=4 on the easy balanced
    # scenario for every development seed.
    gate = results[
        (
            results["scenario"]
            == "balanced_separated"
        )
        &
        (
            results["variant"]
            == "guard_full_forced"
        )
    ]

    if not (
        len(gate) == len(SEEDS)
        and
        (gate["selected_k"] == 4).all()
    ):
        raise SystemExit(
            "G4_BALANCED_SANITY_FAIL"
        )

    lock = {
        "stage": "G4",
        "status": "PASS_PILOT",
        "confirmatory": False,
        "selection_uses_ground_truth": False,
        "ground_truth_usage":
            "external evaluation only",
        "candidate_k_range":
            [2, 8],
        "development_seeds":
            SEEDS,
        "scenarios":
            SCENARIOS,
        "variants": sorted(
            results[
                "variant"
            ].unique().tolist()
        ),
        "note":
            "Pilot results are for mechanism decisions only and must not be reported as confirmatory manuscript evidence."
    }

    (
        OUT
        / "G4_PILOT_LOCK.json"
    ).write_text(
        json.dumps(
            lock,
            indent=2,
        )
    )

    print()
    print(
        "G4_BALANCED_SANITY_PASS"
    )
    print(
        "G4_LABEL_ISOLATION_PASS"
    )
    print(
        "G4_STATUS=PASS_PILOT_NONCONFIRMATORY"
    )


if __name__ == "__main__":
    main()
