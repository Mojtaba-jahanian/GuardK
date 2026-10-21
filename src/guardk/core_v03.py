from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class GuardKV03Config:
    k_min: int = 2
    k_max: int = 8
    n_bootstrap: int = 5
    bootstrap_fraction: float = 0.80
    seed: int = 2207

    min_stability: float = 0.55
    min_rare_persistence: float = 0.50
    max_silhouette_drop: float = 0.12
    max_quality_drop: float = 0.08
    min_local_split_gain: float = 0.15
    min_cluster_fraction: float = 0.002

    abstain_overlap_safety: float = 0.30
    abstain_rare_persistence: float = 0.60
    abstain_quality: float = 0.65

    allow_abstain: bool = True


class GuardKPrototypeV03:
    """
    GuardK v0.3 development core.

    Differences from v0.2:
      1) explicit local split-gain logging,
      2) bootstrap prototype-membership persistence,
      3) guarded sequential split acceptance,
      4) ambiguity abstention separated from K selection,
      5) no OT/UOT in the core.

    fit() accepts X only. Ground truth is never accepted.
    """

    def __init__(
        self,
        config: Optional[GuardKV03Config] = None,
    ):
        self.config = config or GuardKV03Config()
        self.scaler_: Optional[StandardScaler] = None
        self.selected_k_: Optional[int] = None
        self.labels_: Optional[np.ndarray] = None
        self.selected_centers_: Optional[np.ndarray] = None
        self.candidate_table_: Optional[pd.DataFrame] = None
        self.transition_table_: Optional[pd.DataFrame] = None
        self.decision_: Optional[str] = None
        self.lifecycle_events_: list[dict] = []

    @staticmethod
    def _fit_kmeans(
        X,
        k,
        seed,
        *,
        init="k-means++",
        n_init=10,
    ):
        model = KMeans(
            n_clusters=k,
            init=init,
            n_init=n_init,
            random_state=seed,
            algorithm="lloyd",
        )
        model.fit(X)
        return model

    def _validate(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")

        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")

        if len(X) < max(50, 7 * self.config.k_max):
            raise ValueError("X too small.")

        return X

    def _lifecycle(self, X):
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
                "local_split_gain": np.nan,
                "parent_size": np.nan,
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

                if len(pts) < 6:
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

                dispersion_scores.append(
                    msd * np.sqrt(len(pts))
                )

            parent = int(np.argmax(dispersion_scores))
            parent_pts = X[labels == parent]

            parent_center = parent_pts.mean(axis=0)

            inertia1 = float(
                np.sum(
                    (parent_pts - parent_center) ** 2
                )
            )

            local = self._fit_kmeans(
                parent_pts,
                2,
                cfg.seed + 100 * k + parent,
                n_init=15,
            )

            inertia2 = float(local.inertia_)

            local_gain = (
                (inertia1 - inertia2) / inertia1
                if inertia1 > 0
                else 0.0
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
                "local_split_gain": float(local_gain),
            }

            events.append(event)

            candidates[k + 1] = {
                "labels": labels.copy(),
                "centers": centers.copy(),
                "local_split_gain": float(local_gain),
                "parent_size": int(len(parent_pts)),
            }

        self.lifecycle_events_ = events
        return candidates

    @staticmethod
    def _cluster_geometry(
        X,
        labels,
        centers,
    ):
        k = len(centers)
        sizes = np.bincount(labels, minlength=k)
        radii = np.zeros(k, dtype=float)

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
    def _bounded_ratio(x):
        if not np.isfinite(x):
            return 0.0
        x = max(0.0, float(x))
        return x / (1.0 + x)

    def _overlap_safety(
        self,
        centers,
        radii,
    ):
        vals = []

        for i, j in combinations(range(len(centers)), 2):
            d = float(np.linalg.norm(centers[i] - centers[j]))

            pooled = (
                np.sqrt(radii[i] ** 2 + radii[j] ** 2)
                + 1e-12
            )

            vals.append(
                self._bounded_ratio(d / pooled)
            )

        return float(min(vals)) if vals else 0.0

    @staticmethod
    def _match_to_base(
        base_labels,
        pred_labels,
        k,
    ):
        contingency = np.zeros((k, k), dtype=np.int64)

        for i in range(k):
            for j in range(k):
                contingency[i, j] = int(
                    np.sum(
                        (base_labels == i)
                        &
                        (pred_labels == j)
                    )
                )

        row_ind, col_ind = linear_sum_assignment(-contingency)

        pred_to_base = {
            int(p): int(b)
            for b, p in zip(row_ind, col_ind)
        }

        mapped = np.array(
            [
                pred_to_base.get(int(p), -1)
                for p in pred_labels
            ],
            dtype=int,
        )

        return mapped

    def _persistence(
        self,
        X,
        k,
        base_labels,
        base_centers,
    ):
        cfg = self.config
        rng = np.random.default_rng(cfg.seed + 17011 * k)

        n = len(X)
        n_sub = max(
            7 * k,
            int(round(cfg.bootstrap_fraction * n)),
        )

        ari_scores = []
        cluster_jaccards = [[] for _ in range(k)]

        for b in range(cfg.n_bootstrap):
            idx = rng.choice(
                n,
                size=n_sub,
                replace=False,
            )

            model = self._fit_kmeans(
                X[idx],
                k,
                cfg.seed + 5003 * b + 101 * k,
                init=base_centers,
                n_init=1,
            )

            pred = model.predict(X)

            mapped = self._match_to_base(
                base_labels,
                pred,
                k,
            )

            ari_scores.append(
                adjusted_rand_score(
                    base_labels,
                    mapped,
                )
            )

            for c in range(k):
                A = base_labels == c
                B = mapped == c
                union = int(np.sum(A | B))
                inter = int(np.sum(A & B))

                cluster_jaccards[c].append(
                    float(inter / union)
                    if union
                    else 0.0
                )

        persistence = np.array(
            [
                np.mean(v)
                for v in cluster_jaccards
            ],
            dtype=float,
        )

        sizes = np.bincount(base_labels, minlength=k)
        median_size = float(np.median(sizes))
        rare_ids = np.flatnonzero(sizes <= median_size)

        if len(rare_ids) == 0:
            rare_ids = np.arange(k)

        return {
            "stability": float(np.mean(ari_scores)),
            "min_persistence": float(np.min(persistence)),
            "mean_persistence": float(np.mean(persistence)),
            "rare_persistence": float(np.min(persistence[rare_ids])),
        }

    def _candidate_row(
        self,
        X,
        k,
        item,
    ):
        labels = item["labels"]
        centers = item["centers"]

        persistence = self._persistence(
            X,
            k,
            labels,
            centers,
        )

        sil = float(
            silhouette_score(
                X,
                labels,
                sample_size=min(2500, len(X)),
                random_state=self.config.seed,
            )
        )

        sizes, radii = self._cluster_geometry(
            X,
            labels,
            centers,
        )

        inertia = float(
            np.sum(
                (X - centers[labels]) ** 2
            )
        )

        overlap = self._overlap_safety(
            centers,
            radii,
        )

        sil01 = (sil + 1.0) / 2.0

        quality = (
            0.40 * persistence["stability"]
            + 0.25 * persistence["rare_persistence"]
            + 0.20 * sil01
            + 0.15 * overlap
        )

        return {
            "k": int(k),
            "stability": persistence["stability"],
            "min_persistence": persistence["min_persistence"],
            "mean_persistence": persistence["mean_persistence"],
            "rare_persistence": persistence["rare_persistence"],
            "silhouette": sil,
            "silhouette_01": float(sil01),
            "overlap_safety": float(overlap),
            "min_cluster_fraction": float(sizes.min() / len(X)),
            "max_cluster_fraction": float(sizes.max() / len(X)),
            "inertia": inertia,
            "local_split_gain": float(item["local_split_gain"])
                if np.isfinite(item["local_split_gain"])
                else np.nan,
            "quality": float(quality),
        }

    def _select(
        self,
        table,
    ):
        cfg = self.config

        selected = int(cfg.k_min)
        transitions = []

        indexed = table.set_index("k")

        for next_k in range(cfg.k_min + 1, cfg.k_max + 1):
            cur = indexed.loc[selected]
            nxt = indexed.loc[next_k]

            silhouette_drop = float(
                cur["silhouette"] - nxt["silhouette"]
            )

            quality_drop = float(
                cur["quality"] - nxt["quality"]
            )

            checks = {
                "stability_ok":
                    bool(nxt["stability"] >= cfg.min_stability),
                "rare_persistence_ok":
                    bool(nxt["rare_persistence"] >= cfg.min_rare_persistence),
                "cluster_fraction_ok":
                    bool(nxt["min_cluster_fraction"] >= cfg.min_cluster_fraction),
                "silhouette_drop_ok":
                    bool(silhouette_drop <= cfg.max_silhouette_drop),
                "quality_drop_ok":
                    bool(quality_drop <= cfg.max_quality_drop),
                "local_split_gain_ok":
                    bool(nxt["local_split_gain"] >= cfg.min_local_split_gain),
            }

            accept = all(checks.values())

            transitions.append(
                {
                    "from_k": int(selected),
                    "to_k": int(next_k),
                    "silhouette_drop": silhouette_drop,
                    "quality_drop": quality_drop,
                    "local_split_gain": float(nxt["local_split_gain"]),
                    "stability": float(nxt["stability"]),
                    "rare_persistence": float(nxt["rare_persistence"]),
                    **checks,
                    "accepted": bool(accept),
                }
            )

            if accept:
                selected = int(next_k)
            else:
                break

        row = indexed.loc[selected]

        ambiguous = (
            row["overlap_safety"] < cfg.abstain_overlap_safety
            and
            (
                row["rare_persistence"] < cfg.abstain_rare_persistence
                or
                row["quality"] < cfg.abstain_quality
            )
        )

        if cfg.allow_abstain and ambiguous:
            return None, "ABSTAIN_AMBIGUOUS", transitions

        return selected, "ACCEPT", transitions

    def fit(self, X):
        X = self._validate(X)

        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(X)

        candidates = self._lifecycle(Z)

        rows = [
            self._candidate_row(Z, k, candidates[k])
            for k in range(self.config.k_min, self.config.k_max + 1)
        ]

        table = pd.DataFrame(rows)

        selected_k, decision, transitions = self._select(table)

        table["selected"] = (
            False
            if selected_k is None
            else table["k"].eq(selected_k)
        )

        self.candidate_table_ = table
        self.transition_table_ = pd.DataFrame(transitions)
        self.selected_k_ = selected_k
        self.decision_ = decision

        if selected_k is None:
            self.labels_ = None
            self.selected_centers_ = None
        else:
            self.labels_ = candidates[selected_k]["labels"].copy()
            self.selected_centers_ = candidates[selected_k]["centers"].copy()

        return self

    def fit_predict(self, X):
        self.fit(X)

        if self.labels_ is None:
            raise RuntimeError("GuardK v0.3 abstained.")

        return self.labels_.copy()
