"""Feature-matrix diagnostics for identification readiness.

Before fitting a structural or IRL estimator, check that the reward features can
actually be recovered from observed choices. The key check is the action-contrast
rank, not the raw design rank: a feature that is constant across actions within a
state cancels out of every discrete-choice comparison and leaves its parameter
unidentified.
"""

from __future__ import annotations

import numpy as np


def feature_diagnostics(feature_matrix: np.ndarray) -> dict[str, float | int]:
    """Rank and conditioning of a reward-feature design, raw and action-differenced.

    A reward parameter is identified from behavior only if its feature varies
    across actions within a state. The raw stacked design ``(S*A, K)`` can be
    full rank while a parameter is still unrecoverable from choices, because any
    feature that is constant across actions differences out of every choice
    comparison. The action-contrast design ``phi(s, a) - phi(s, A)`` is the one
    that drives identification, so ``contrast_rank`` is the check that bites. For
    example, a gridworld with a state-only step penalty and distance features has
    raw rank 3 but contrast rank 1.

    Parameters
    ----------
    feature_matrix : numpy.ndarray
        Reward features with shape ``(S, A, K)``: states, actions, features.

    Returns
    -------
    dict
        Diagnostics with keys:

        - ``num_features`` : number of features ``K``.
        - ``feature_rank`` : rank of the raw design ``(S*A, K)``.
        - ``condition_number`` : condition number of the raw design.
        - ``contrast_rank`` : rank of the action-contrast design
          ``phi(s, a) - phi(s, A)``. Compare this to ``num_features``; if it is
          smaller, some reward parameters are not identified from choices.
        - ``contrast_condition_number`` : condition number of the contrast design.
    """
    phi = np.asarray(feature_matrix, dtype=np.float64)
    S, A, K = phi.shape
    design = phi.reshape(S * A, K)
    rank = int(np.linalg.matrix_rank(design))
    svals = np.linalg.svd(design, compute_uv=False)
    smin = float(svals.min())
    cond = float(svals.max() / smin) if smin > 0 else float("inf")
    contrast = (phi - phi[:, -1:, :]).reshape(S * A, K)
    c_rank = int(np.linalg.matrix_rank(contrast))
    c_svals = np.linalg.svd(contrast, compute_uv=False)
    c_pos = c_svals[c_svals > 1e-12]
    c_cond = float(c_pos.max() / c_pos.min()) if c_pos.size else float("inf")
    return {
        "num_features": K,
        "feature_rank": rank,
        "condition_number": cond,
        "contrast_rank": c_rank,
        "contrast_condition_number": c_cond,
    }
