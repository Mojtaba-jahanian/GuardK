from __future__ import annotations

import numpy as np

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def matched_cluster_recalls(
    y_true,
    y_pred,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_vals = np.unique(y_true)
    pred_vals = np.unique(y_pred)

    contingency = np.zeros(
        (
            len(true_vals),
            len(pred_vals),
        ),
        dtype=np.int64,
    )

    for i, t in enumerate(true_vals):
        for j, p in enumerate(pred_vals):
            contingency[i, j] = int(
                np.sum(
                    (y_true == t)
                    &
                    (y_pred == p)
                )
            )

    cost = -contingency

    row_ind, col_ind = (
        linear_sum_assignment(
            cost
        )
    )

    recalls = {
        str(t): 0.0
        for t in true_vals
    }

    for i, j in zip(
        row_ind,
        col_ind,
    ):
        denom = int(
            np.sum(
                y_true
                == true_vals[i]
            )
        )

        recalls[
            str(true_vals[i])
        ] = (
            float(
                contingency[i, j]
                / denom
            )
            if denom
            else 0.0
        )

    return recalls


def evaluate_external(
    y_true,
    y_pred,
):
    """
    External evaluation only.

    This function is intentionally separate from GuardK.fit().
    """

    recalls = matched_cluster_recalls(
        y_true,
        y_pred,
    )

    values = list(
        recalls.values()
    )

    return {
        "ARI": float(
            adjusted_rand_score(
                y_true,
                y_pred,
            )
        ),
        "NMI": float(
            normalized_mutual_info_score(
                y_true,
                y_pred,
            )
        ),
        "worst_cluster_recall": float(
            np.min(values)
        ),
        "mean_cluster_recall": float(
            np.mean(values)
        ),
        "cluster_recalls": recalls,
    }
