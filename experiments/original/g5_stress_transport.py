from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from guardk import (
    GuardKConfig,
    GuardKPrototype,
    evaluate_external,
)

from guardk.transport import (
    transport_assign,
)


ROOT = Path.home() / "GuardK_V6"
OUT = ROOT / "results" / "g5_stress"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEEDS = [
    2207,
    3407,
    5107,
]

IMBALANCE_RATIOS = [
    1,
    10,
    20,
    50,
]

OVERLAP_LEVELS = {
    "low": {
        "separation": 5.5,
        "sigma": 0.75,
    },
    "medium": {
        "separation": 4.0,
        "sigma": 1.25,
    },
    "high": {
        "separation": 2.4,
        "sigma": 1.55,
    },
}


def scenario_sizes(
    ratio: int,
) -> list[int]:

    if ratio == 1:
        return [
            360,
            360,
            360,
            360,
        ]

    max_n = 600

    if ratio == 10:
        return [
            600,
            420,
            260,
            60,
        ]

    if ratio == 20:
        return [
            600,
            360,
            180,
            30,
        ]

    if ratio == 50:
        return [
            600,
            320,
            160,
            12,
        ]

    raise ValueError(
        ratio
    )


def make_scenario(
    ratio: int,
    overlap: str,
    seed: int,
):
    cfg = OVERLAP_LEVELS[
        overlap
    ]

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

    sizes = scenario_sizes(
        ratio
    )

    rng = np.random.default_rng(
        seed
        + ratio * 101
        + {
            "low": 1,
            "medium": 2,
            "high": 3,
        }[overlap]
    )

    Xs = []
    ys = []

    for c, (mu, n) in enumerate(
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
        {
            "true_k": 4,
            "ratio": ratio,
            "overlap": overlap,
            "sizes": sizes,
            "sigma": sigma,
            "separation": s,
        },
    )


def row_from_prediction(
    *,
    ratio,
    overlap,
    seed,
    variant,
    true_k,
    selected_k,
    decision,
    y,
    pred,
    runtime_sec,
    transport_diag=None,
):
    out = {
        "imbalance_ratio": int(ratio),
        "overlap": overlap,
        "seed": int(seed),
        "variant": variant,
        "true_k": int(true_k),
        "selected_k": (
            np.nan
            if selected_k is None
            else int(selected_k)
        ),
        "decision": decision,
        "runtime_sec": float(
            runtime_sec
        ),
    }

    if selected_k is None:
        out.update(
            {
                "exact_k": 0,
                "abs_k_error": np.nan,
                "ARI": np.nan,
                "NMI": np.nan,
                "worst_cluster_recall":
                    np.nan,
                "mean_cluster_recall":
                    np.nan,
            }
        )
        return out

    metrics = evaluate_external(
        y,
        pred,
    )

    out.update(
        {
            "exact_k": int(
                selected_k
                == true_k
            ),
            "abs_k_error": int(
                abs(
                    selected_k
                    - true_k
                )
            ),
            "ARI": metrics[
                "ARI"
            ],
            "NMI": metrics[
                "NMI"
            ],
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

    if transport_diag:
        for k,v in (
            transport_diag.items()
        ):
            out[
                "transport_" + k
            ] = v

    return out


def main():
    rows = []
    candidate_rows = []

    for ratio in IMBALANCE_RATIOS:
        for overlap in OVERLAP_LEVELS:
            for seed in SEEDS:
                print(
                    "G5_SCENARIO",
                    "ratio=",
                    ratio,
                    "overlap=",
                    overlap,
                    "seed=",
                    seed,
                    flush=True,
                )

                X, y, meta = (
                    make_scenario(
                        ratio,
                        overlap,
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

                t0 = time.perf_counter()

                model = GuardKPrototype(
                    cfg
                ).fit(X)

                core_runtime = (
                    time.perf_counter()
                    - t0
                )

                for _, r in (
                    model
                    .candidate_table_
                    .iterrows()
                ):
                    candidate_rows.append(
                        {
                            "imbalance_ratio":
                                ratio,
                            "overlap":
                                overlap,
                            "seed":
                                seed,
                            **r.to_dict(),
                        }
                    )

                k = model.selected_k_

                if (
                    k is None
                    or model.labels_ is None
                    or model.selected_centers_
                    is None
                ):
                    rows.append(
                        row_from_prediction(
                            ratio=ratio,
                            overlap=overlap,
                            seed=seed,
                            variant=
                                "guard_core_no_ot",
                            true_k=
                                meta["true_k"],
                            selected_k=None,
                            decision=
                                model.decision_,
                            y=y,
                            pred=None,
                            runtime_sec=
                                core_runtime,
                        )
                    )
                    continue

                rows.append(
                    row_from_prediction(
                        ratio=ratio,
                        overlap=overlap,
                        seed=seed,
                        variant=
                            "guard_core_no_ot",
                        true_k=
                            meta["true_k"],
                        selected_k=k,
                        decision=
                            model.decision_,
                        y=y,
                        pred=
                            model.labels_,
                        runtime_sec=
                            core_runtime,
                    )
                )

                Z = (
                    model
                    .scaler_
                    .transform(X)
                )

                centers = (
                    model
                    .selected_centers_
                )

                for mode, variant in [
                    (
                        "balanced",
                        "guard_plus_balanced_ot",
                    ),
                    (
                        "uot",
                        "guard_plus_uot",
                    ),
                ]:
                    tt = time.perf_counter()

                    pred, diag = (
                        transport_assign(
                            Z,
                            centers,
                            mode,
                            reg=0.05,
                            reg_m=1.0,
                        )
                    )

                    truntime = (
                        time.perf_counter()
                        - tt
                    )

                    rows.append(
                        row_from_prediction(
                            ratio=ratio,
                            overlap=overlap,
                            seed=seed,
                            variant=
                                variant,
                            true_k=
                                meta["true_k"],
                            selected_k=k,
                            decision=
                                model.decision_,
                            y=y,
                            pred=pred,
                            runtime_sec=
                                core_runtime
                                + truntime,
                            transport_diag=
                                diag,
                        )
                    )

    runs = pd.DataFrame(
        rows
    )

    candidates = pd.DataFrame(
        candidate_rows
    )

    runs.to_csv(
        OUT /
        "G5_stress_transport_runs.csv",
        index=False,
    )

    candidates.to_csv(
        OUT /
        "G5_candidate_diagnostics.csv",
        index=False,
    )

    aggregate = (
        runs
        .groupby(
            [
                "imbalance_ratio",
                "overlap",
                "variant",
            ],
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
    )

    aggregate.to_csv(
        OUT /
        "G5_stress_transport_aggregate.csv",
        index=False,
    )

    overall = (
        runs
        .groupby(
            "variant",
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
    )

    overall.to_csv(
        OUT /
        "G5_transport_overall.csv",
        index=False,
    )

    print()
    print(
        "G5_OVERALL"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    easy = runs[
        (
            runs[
                "imbalance_ratio"
            ] == 1
        )
        &
        (
            runs["overlap"]
            == "low"
        )
        &
        (
            runs["variant"]
            == "guard_core_no_ot"
        )
    ]

    if not (
        len(easy) == len(SEEDS)
        and
        (easy["selected_k"] == 4).all()
    ):
        raise SystemExit(
            "G5_EASY_SANITY_FAIL"
        )

    lock = {
        "stage":
            "G5",
        "status":
            "PASS_STRESS_PILOT",
        "confirmatory":
            False,
        "ground_truth_used_for_selection":
            False,
        "imbalance_ratios":
            IMBALANCE_RATIOS,
        "overlap_levels":
            list(
                OVERLAP_LEVELS.keys()
            ),
        "seeds":
            SEEDS,
        "transport_ablation":
            [
                "none",
                "balanced_ot",
                "uot",
            ],
        "transport_role":
            "assignment-only operator with K and prototypes held fixed",
        "note":
            "G5 remains development evidence; transport inclusion in the manuscript depends on matched results."
    }

    (
        OUT /
        "G5_STRESS_LOCK.json"
    ).write_text(
        json.dumps(
            lock,
            indent=2,
        )
    )

    print(
        "G5_EASY_SANITY_PASS"
    )
    print(
        "G5_LABEL_ISOLATION_PASS"
    )
    print(
        "G5_STATUS=PASS_STRESS_PILOT_NONCONFIRMATORY"
    )


if __name__ == "__main__":
    main()
