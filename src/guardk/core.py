"""
GuardK development implementation, G3 v0.2.

Scientific status
-----------------
This code is a DEVELOPMENT implementation used for mechanism validation
and ablation only. It is not yet a confirmatory/final manuscript lock.

Ground-truth labels are never accepted by fit() and never used for
candidate-K selection.

Main components implemented here:
1) sequential prototype lifecycle via split proposals,
2) label-free perturbation stability,
3) imbalance-aware rare-prototype support,
4) overlap safety,
5) explicit guarded K selection,
6) optional ambiguity abstention.

Balanced OT / UOT are intentionally NOT part of the G3 core.
They remain optional operators for later matched ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GuardKConfig:
    k_min: int = 2
    k_max: int = 8

    n_bootstrap: int = 5
    bootstrap_fraction: float = 0.80

    seed: int = 2207

    min_stability: float = 0.55
    min_cluster_fraction: float = 0.003

    parsimony_tolerance: float = 0.008
    ambiguity_margin: float = 0.005

    allow_abstain: bool = False

    # Development-only G3/G4 weights.
    stability_weight: float = 0.55
    silhouette_weight: float = 0.20
    rare_support_weight: float = 0.15
    overlap_weight: float = 0.10
    complexity_penalty: float = 0.02


class GuardKPrototype:
    """
    Label-free GuardK research prototype.

    The public fit() API accepts X only.
    External labels belong exclusively to evaluation code.
    """

    def __init__(self, config: Optional[GuardKConfig] = None):
        self.config = config or GuardKConfig()

        self.scaler_: Optional[StandardScaler] = None
        self.selected_k_: Optional[int] = None
        self.selected_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.candidate_table_: Optional[pd.DataFrame] = None
        self.lifecycle_events_: list[dict] = []
        self.decision_: Optional[str] = None
        self.score_margin_: Optional[float] = None

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if len(X) < max(40, 6 * self.config.k_max):
            raise ValueError("Dataset too small for requested candidate-K range.")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")

        if self.config.k_min < 2:
            raise ValueError("k_min must be >= 2.")

        if self.config.k_max < self.config.k_min:
            raise ValueError("k_max must be >= k_min.")

        return X

    @staticmethod
    def _fit_kmeans(
        X: np.ndarray,
        k: int,
        seed: int,
        *,
        init="k-means++",
        n_init: int = 10,
    ) -> KMeans:

        model = KMeans(
            n_clusters=k,
            init=init,
            n_init=n_init,
            random_state=seed,
            algorithm="lloyd",
        )
        model.fit(X)
        return model

    def _prototype_lifecycle(self, X: np.ndarray) -> dict[int, dict]:
        """
        Sequential split lifecycle.

        Starts at k_min, then:
          - identifies the active prototype with largest
            dispersion x sqrt(cluster size),
          - splits it with local 2-means,
          - globally refines all prototypes with one initialized KMeans step.

        This is the first explicit adaptive lifecycle implementation.
        Merge/prune policies are deferred to later stages.
        """

        cfg = self.config

        first = self._fit_kmeans(
            X,
            cfg.k_min,
            cfg.seed + 17,
            n_init=20,
        )

        labels = first.labels_.copy()
        centers = first.cluster_centers_.copy()

        candidates = {
            cfg.k_min: {
                "labels": labels.copy(),
                "centers": centers.copy(),
                "event": {
                    "type": "initialize",
                    "k": cfg.k_min,
                },
            }
        }

        events = [
            {
                "type": "initialize",
                "k": cfg.k_min,
            }
        ]

        for k in range(cfg.k_min, cfg.k_max):
            dispersion_scores = []

            for c in range(k):
                pts = X[labels == c]

                if len(pts) < 4:
                    dispersion_scores.append(-np.inf)
                    continue

                msd = float(
                    np.mean(
                        np.sum(
                            (pts - centers[c]) ** 2,
                            axis=1,
                        )
                    )
                )

                # Size-aware but sublinear so minority prototypes are not
                # automatically suppressed.
                dispersion_scores.append(
                    msd * np.sqrt(len(pts))
                )

            parent = int(np.argmax(dispersion_scores))
            parent_pts = X[labels == parent]

            local = self._fit_kmeans(
                parent_pts,
                2,
                cfg.seed + 100 * k + parent,
                n_init=10,
            )

            keep = np.arange(k) != parent

            init = np.vstack(
                [
                    centers[keep],
                    local.cluster_centers_,
                ]
            )

            global_model = self._fit_kmeans(
                X,
                k + 1,
                cfg.seed + 1000 + k,
                init=init,
                n_init=1,
            )

            labels = global_model.labels_.copy()
            centers = global_model.cluster_centers_.copy()

            event = {
                "type": "split",
                "from_k": k,
                "to_k": k + 1,
                "parent_cluster": parent,
                "parent_size": int(len(parent_pts)),
                "parent_dispersion_score": float(
                    dispersion_scores[parent]
                ),
            }

            events.append(event)

            candidates[k + 1] = {
                "labels": labels.copy(),
                "centers": centers.copy(),
                "event": event,
            }

        self.lifecycle_events_ = events
        return candidates

    def _partition_stability(
        self,
        X: np.ndarray,
        k: int,
        base_centers: np.ndarray,
    ) -> float:
        """
        Label-free stability.

        Each perturbation fits KMeans on a subsample initialized from the
        same candidate prototypes, then predicts the common full reference
        population. ARI is used only as agreement between model partitions;
        no external labels are involved.
        """

        cfg = self.config

        rng = np.random.default_rng(
            cfg.seed + 10007 * k
        )

        n = len(X)

        n_sub = max(
            6 * k,
            int(round(cfg.bootstrap_fraction * n)),
        )

        partitions = []

        for b in range(cfg.n_bootstrap):
            idx = rng.choice(
                n,
                size=n_sub,
                replace=False,
            )

            model = self._fit_kmeans(
                X[idx],
                k,
                cfg.seed + 997 * b + 31 * k,
                init=base_centers,
                n_init=1,
            )

            partitions.append(
                model.predict(X)
            )

        pair_scores = [
            adjusted_rand_score(
                partitions[i],
                partitions[j],
            )
            for i, j in combinations(
                range(len(partitions)),
                2,
            )
        ]

        return float(np.mean(pair_scores))

    @staticmethod
    def _cluster_geometry(
        X: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        k = len(centers)

        sizes = np.bincount(
            labels,
            minlength=k,
        )

        radii = np.zeros(
            k,
            dtype=np.float64,
        )

        for c in range(k):
            pts = X[labels == c]

            if len(pts) == 0:
                radii[c] = np.inf
                continue

            radii[c] = (
                np.sqrt(
                    np.mean(
                        np.sum(
                            (pts - centers[c]) ** 2,
                            axis=1,
                        )
                    )
                )
                + 1e-12
            )

        return sizes, radii

    @staticmethod
    def _bounded_ratio(x: float) -> float:
        if not np.isfinite(x):
            return 0.0

        x = max(0.0, float(x))
        return x / (1.0 + x)

    def _rare_support(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        sizes: np.ndarray,
        radii: np.ndarray,
    ) -> float:

        n = len(X)

        median_size = float(
            np.median(sizes)
        )

        threshold = max(
            0.10 * n,
            0.50 * median_size,
        )

        rare_ids = np.flatnonzero(
            sizes <= threshold
        )

        if len(rare_ids) == 0:
            rare_ids = np.arange(
                len(centers)
            )

        scores = []

        for c in rare_ids:
            d = np.linalg.norm(
                centers - centers[c],
                axis=1,
            )

            d[c] = np.inf

            nearest = float(
                np.min(d)
            )

            support_ratio = (
                nearest / (radii[c] + 1e-12)
            )

            scores.append(
                self._bounded_ratio(
                    support_ratio
                )
            )

        return float(
            np.min(scores)
        )

    def _overlap_safety(
        self,
        centers: np.ndarray,
        radii: np.ndarray,
    ) -> float:

        vals = []

        for i, j in combinations(
            range(len(centers)),
            2,
        ):

            d = float(
                np.linalg.norm(
                    centers[i] - centers[j]
                )
            )

            pooled_radius = (
                np.sqrt(
                    radii[i] ** 2
                    + radii[j] ** 2
                )
                + 1e-12
            )

            vals.append(
                self._bounded_ratio(
                    d / pooled_radius
                )
            )

        if not vals:
            return 0.0

        return float(
            np.min(vals)
        )

    def _evaluate_candidate(
        self,
        X: np.ndarray,
        k: int,
        labels: np.ndarray,
        centers: np.ndarray,
    ) -> dict:

        cfg = self.config

        stability = self._partition_stability(
            X,
            k,
            centers,
        )

        silhouette = float(
            silhouette_score(
                X,
                labels,
                sample_size=min(
                    2500,
                    len(X),
                ),
                random_state=cfg.seed,
            )
        )

        sizes, radii = (
            self._cluster_geometry(
                X,
                labels,
                centers,
            )
        )

        rare_support = self._rare_support(
            X,
            labels,
            centers,
            sizes,
            radii,
        )

        overlap_safety = (
            self._overlap_safety(
                centers,
                radii,
            )
        )

        sil01 = (
            silhouette + 1.0
        ) / 2.0

        complexity = (
            k / cfg.k_max
        )

        score = (
            cfg.stability_weight
            * stability

            + cfg.silhouette_weight
            * sil01

            + cfg.rare_support_weight
            * rare_support

            + cfg.overlap_weight
            * overlap_safety

            - cfg.complexity_penalty
            * complexity
        )

        return {
            "k": int(k),
            "stability": float(stability),
            "silhouette": silhouette,
            "silhouette_01": float(sil01),
            "rare_support": float(
                rare_support
            ),
            "overlap_safety": float(
                overlap_safety
            ),
            "min_cluster_fraction": float(
                sizes.min() / len(X)
            ),
            "max_cluster_fraction": float(
                sizes.max() / len(X)
            ),
            "guard_score": float(score),
        }

    def _select_from_table(
        self,
        table: pd.DataFrame,
    ) -> tuple[Optional[int], str, float]:

        cfg = self.config

        eligible = table[
            (table["stability"]
             >= cfg.min_stability)
            &
            (table["min_cluster_fraction"]
             >= cfg.min_cluster_fraction)
        ].copy()

        if eligible.empty:
            fallback = table.sort_values(
                [
                    "stability",
                    "guard_score",
                    "k",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            ).iloc[0]

            return (
                int(fallback["k"]),
                "FALLBACK_NO_ELIGIBLE",
                np.nan,
            )

        ranked = eligible.sort_values(
            [
                "guard_score",
                "k",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(drop=True)

        best_score = float(
            ranked.loc[
                0,
                "guard_score",
            ]
        )

        near_best = eligible[
            eligible["guard_score"]
            >= (
                best_score
                - cfg.parsimony_tolerance
            )
        ]

        selected_k = int(
            near_best["k"].min()
        )

        if len(ranked) >= 2:
            margin = float(
                ranked.loc[0, "guard_score"]
                - ranked.loc[1, "guard_score"]
            )
        else:
            margin = np.inf

        if (
            cfg.allow_abstain
            and len(ranked) >= 2
            and margin
            < cfg.ambiguity_margin
        ):
            return (
                None,
                "ABSTAIN_AMBIGUOUS",
                margin,
            )

        return (
            selected_k,
            "ACCEPT",
            margin,
        )

    def fit(
        self,
        X: np.ndarray,
    ) -> "GuardKPrototype":

        X = self._validate_X(X)

        self.scaler_ = StandardScaler()

        Z = self.scaler_.fit_transform(
            X
        )

        candidates = (
            self._prototype_lifecycle(
                Z
            )
        )

        rows = []

        for k in range(
            self.config.k_min,
            self.config.k_max + 1,
        ):
            item = candidates[k]

            row = self._evaluate_candidate(
                Z,
                k,
                item["labels"],
                item["centers"],
            )

            rows.append(row)

        table = pd.DataFrame(rows)

        selected_k, decision, margin = (
            self._select_from_table(
                table
            )
        )

        table["eligible"] = (
            (
                table["stability"]
                >= self.config.min_stability
            )
            &
            (
                table[
                    "min_cluster_fraction"
                ]
                >= self.config.min_cluster_fraction
            )
        )

        table["selected"] = (
            False
            if selected_k is None
            else table["k"].eq(
                selected_k
            )
        )

        self.candidate_table_ = table
        self.selected_k_ = selected_k
        self.decision_ = decision
        self.score_margin_ = margin

        if selected_k is None:
            self.labels_ = None
            self.selected_centers_ = None

        else:
            self.labels_ = candidates[
                selected_k
            ]["labels"].copy()

            self.selected_centers_ = candidates[
                selected_k
            ]["centers"].copy()

        return self

    def fit_predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        self.fit(X)

        if self.labels_ is None:
            raise RuntimeError(
                "GuardK abstained; no partition was accepted."
            )

        return self.labels_.copy()

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        if (
            self.scaler_ is None
            or self.selected_centers_ is None
        ):
            raise RuntimeError(
                "Model is not fitted with an accepted partition."
            )

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        Z = self.scaler_.transform(
            X
        )

        d2 = (
            (
                Z[:, None, :]
                - self.selected_centers_[
                    None, :, :
                ]
            )
            ** 2
        ).sum(axis=2)

        return np.argmin(
            d2,
            axis=1,
        )
