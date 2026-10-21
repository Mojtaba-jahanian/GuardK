from __future__ import annotations

import numpy as np
import ot


def _scaled_sqeuclidean(
    X: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    M = ot.dist(
        X,
        centers,
        metric="sqeuclidean",
    )

    positive = M[M > 0]

    scale = (
        float(np.median(positive))
        if positive.size
        else 1.0
    )

    if (
        not np.isfinite(scale)
        or scale <= 0
    ):
        scale = 1.0

    return M / scale


def transport_assign(
    X: np.ndarray,
    centers: np.ndarray,
    mode: str,
    *,
    reg: float = 0.05,
    reg_m: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Assignment-only transport ablation.

    Important:
    - K and prototypes are held fixed.
    - No ground-truth labels are used.
    - This isolates the effect of balanced vs unbalanced transport.
    """

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    centers = np.asarray(
        centers,
        dtype=np.float64,
    )

    n = len(X)
    k = len(centers)

    if n == 0 or k < 2:
        raise ValueError(
            "Invalid X/centers."
        )

    M = _scaled_sqeuclidean(
        X,
        centers,
    )

    a = np.full(
        n,
        1.0 / n,
        dtype=np.float64,
    )

    b = np.full(
        k,
        1.0 / k,
        dtype=np.float64,
    )

    if mode == "balanced":
        plan = ot.sinkhorn(
            a,
            b,
            M,
            reg=reg,
            numItermax=5000,
            stopThr=1e-9,
        )

    elif mode == "uot":
        plan = (
            ot.unbalanced
            .sinkhorn_unbalanced(
                a,
                b,
                M,
                reg=reg,
                reg_m=reg_m,
                numItermax=5000,
                stopThr=1e-9,
            )
        )

    else:
        raise ValueError(
            f"Unknown transport mode: {mode}"
        )

    plan = np.asarray(
        plan,
        dtype=np.float64,
    )

    if plan.shape != (n, k):
        raise RuntimeError(
            f"Transport plan shape {plan.shape} != {(n,k)}"
        )

    if not np.isfinite(plan).all():
        raise RuntimeError(
            "Non-finite transport plan."
        )

    if (plan < -1e-12).any():
        raise RuntimeError(
            "Negative transport mass."
        )

    row_mass = plan.sum(axis=1)

    labels = np.argmax(
        plan,
        axis=1,
    ).astype(int)

    counts = np.bincount(
        labels,
        minlength=k,
    )

    diagnostics = {
        "mode": mode,
        "reg": float(reg),
        "reg_m": (
            None
            if mode == "balanced"
            else float(reg_m)
        ),
        "plan_mass": float(
            plan.sum()
        ),
        "min_row_mass": float(
            row_mass.min()
        ),
        "max_row_mass": float(
            row_mass.max()
        ),
        "min_cluster_fraction": float(
            counts.min() / n
        ),
        "max_cluster_fraction": float(
            counts.max() / n
        ),
    }

    return labels, diagnostics
