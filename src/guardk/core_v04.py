from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from typing import Optional
import numpy as np
from .core import GuardKConfig, GuardKPrototype

@dataclass(frozen=True)
class GuardKV04Config:
    k_min: int = 2
    k_max: int = 8
    seed: int = 2207
    inner_bootstrap: int = 4
    outer_repeats: int = 5
    outer_fraction: float = 0.80
    outer_inner_bootstrap: int = 3
    min_selection_consistency: float = 0.60
    strong_selection_consistency: float = 0.80
    parsimony_tolerance: float = 0.008
    min_overlap_safety: float = 0.45
    allow_abstain: bool = True

class GuardKPrototypeV04:
    """
    v0.4 preserves the stronger v0.2 K selector and adds a label-free
    reproducibility guard on the selected K itself. fit() accepts X only.
    """
    def __init__(self, config: Optional[GuardKV04Config] = None):
        self.config = config or GuardKV04Config()
        self.base_model_ = None
        self.selected_k_ = None
        self.labels_ = None
        self.selected_centers_ = None
        self.decision_ = None
        self.diagnostics_ = {}

    def _base_cfg(self, seed, n_bootstrap):
        c = self.config
        return GuardKConfig(
            k_min=c.k_min, k_max=c.k_max,
            n_bootstrap=n_bootstrap, bootstrap_fraction=0.80,
            seed=seed, min_stability=0.55, min_cluster_fraction=0.003,
            parsimony_tolerance=c.parsimony_tolerance,
            ambiguity_margin=0.005, allow_abstain=False,
            stability_weight=0.55, silhouette_weight=0.20,
            rare_support_weight=0.15, overlap_weight=0.10,
            complexity_penalty=0.02,
        )

    @staticmethod
    def _margin(table):
        x = table["guard_score"].sort_values(ascending=False).to_numpy()
        return float(x[0]-x[1]) if len(x) >= 2 else float("inf")

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        c = self.config
        base = GuardKPrototype(self._base_cfg(c.seed, c.inner_bootstrap)).fit(X)
        self.base_model_ = base

        if base.selected_k_ is None or base.labels_ is None:
            self.decision_ = "ABSTAIN_BASE_FAILURE"
            return self

        full_k = int(base.selected_k_)
        row = base.candidate_table_.loc[base.candidate_table_["k"] == full_k].iloc[0]
        margin = self._margin(base.candidate_table_)
        overlap = float(row["overlap_safety"])

        rng = np.random.default_rng(c.seed + 44041)
        n = len(X)
        n_sub = max(10*c.k_max, int(round(c.outer_fraction*n)))
        outer = []

        for r in range(c.outer_repeats):
            idx = rng.choice(n, size=n_sub, replace=False)
            seed_r = c.seed + 1009*(r+1)
            m = GuardKPrototype(self._base_cfg(seed_r, c.outer_inner_bootstrap)).fit(X[idx])
            if m.selected_k_ is not None:
                outer.append(int(m.selected_k_))

        if outer:
            counts = Counter(outer)
            mode_k, mode_count = counts.most_common(1)[0]
            mode_support = mode_count/len(outer)
            consistency = sum(k == full_k for k in outer)/len(outer)
        else:
            mode_k, mode_support, consistency = None, 0.0, 0.0

        reproducible = consistency >= c.min_selection_consistency and mode_k == full_k
        strong = consistency >= c.strong_selection_consistency and mode_k == full_k
        low_margin = margin <= c.parsimony_tolerance
        low_overlap = overlap < c.min_overlap_safety

        if not reproducible:
            decision = "ABSTAIN_K_SELECTION_INSTABILITY"
            accept = False
        elif low_margin and not strong:
            decision = "ABSTAIN_SCORE_AMBIGUITY"
            accept = False
        elif low_overlap and not strong:
            decision = "ABSTAIN_OVERLAP_AMBIGUITY"
            accept = False
        else:
            decision = "ACCEPT"
            accept = True

        self.diagnostics_ = {
            "base_selected_k": full_k,
            "outer_selected_ks": list(outer),
            "selection_consistency": float(consistency),
            "selection_mode": None if mode_k is None else int(mode_k),
            "selection_mode_support": float(mode_support),
            "score_margin": float(margin),
            "overlap_safety": float(overlap),
            "low_margin": bool(low_margin),
            "low_overlap": bool(low_overlap),
            "reproducible": bool(reproducible),
            "strongly_reproducible": bool(strong),
            "decision": decision,
        }
        self.decision_ = decision

        if accept:
            self.selected_k_ = full_k
            self.labels_ = base.labels_.copy()
            self.selected_centers_ = base.selected_centers_.copy()

        return self
