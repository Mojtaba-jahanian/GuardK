from __future__ import annotations

import numpy as np


def _square_centers(
    separation,
):
    s = float(separation)

    return np.array(
        [
            [-s, -s],
            [-s,  s],
            [ s, -s],
            [ s,  s],
        ],
        dtype=float,
    )


def make_guardk_scenario(
    name: str,
    seed: int,
):
    """
    Synthetic labels are generated ONLY for external evaluation.
    They are not available to the GuardK selection procedure.
    """

    rng = np.random.default_rng(
        seed
    )

    if name == "balanced_separated":
        centers = _square_centers(6.0)
        sizes = [400, 400, 400, 400]
        sigma = 0.75

    elif name == "imbalanced_10x":
        centers = _square_centers(6.0)
        sizes = [800, 400, 200, 80]
        sigma = 0.80

    elif name == "rare_20x":
        centers = _square_centers(6.0)
        sizes = [1000, 500, 200, 50]
        sigma = 0.80

    elif name == "overlap_high":
        centers = _square_centers(2.0)
        sizes = [400, 400, 400, 400]
        sigma = 1.40

    elif name == "overlap_ambiguous":
        centers = _square_centers(1.5)
        sizes = [350, 350, 350, 350]
        sigma = 1.80

    else:
        raise ValueError(
            f"Unknown scenario: {name}"
        )

    X_parts = []
    y_parts = []

    for c, (center, n) in enumerate(
        zip(
            centers,
            sizes,
        )
    ):
        X_parts.append(
            rng.normal(
                loc=center,
                scale=sigma,
                size=(n, 2),
            )
        )

        y_parts.append(
            np.full(
                n,
                c,
                dtype=int,
            )
        )

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    order = rng.permutation(
        len(X)
    )

    return (
        X[order],
        y[order],
        {
            "scenario": name,
            "true_k": 4,
            "sizes": sizes,
            "sigma": sigma,
            "seed": int(seed),
        },
    )
