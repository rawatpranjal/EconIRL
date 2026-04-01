"""Comparison metrics for identification experiments.

All metrics operate on non-absorbing states and non-exit actions
to focus on the economically meaningful part of the reward and
policy. The absorbing state has zero reward by construction and
the exit action is anchored to zero, so including them would
deflate error metrics.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def reward_mse(
    r_estimated: jnp.ndarray,
    r_true: jnp.ndarray,
    absorbing_state: int,
    exit_action: int,
) -> float:
    """Mean squared error of estimated vs true reward.

    Computed over non-absorbing states and non-exit actions.
    """
    mask = _reward_mask(r_true.shape, absorbing_state, exit_action)
    diff = np.asarray(r_estimated - r_true)
    return float(np.mean(diff[mask] ** 2))


def reward_correlation(
    r_estimated: jnp.ndarray,
    r_true: jnp.ndarray,
    absorbing_state: int,
    exit_action: int,
) -> float:
    """Pearson correlation of estimated vs true reward.

    Computed over non-absorbing states and non-exit actions.
    Returns 0.0 if either vector has zero variance.
    """
    mask = _reward_mask(r_true.shape, absorbing_state, exit_action)
    est = np.asarray(r_estimated)[mask]
    true = np.asarray(r_true)[mask]

    if est.std() < 1e-12 or true.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(est, true)[0, 1])


def ccp_error(
    ccp_estimated: jnp.ndarray,
    ccp_oracle: jnp.ndarray,
    absorbing_state: int,
) -> float:
    """Mean absolute CCP difference over non-absorbing states."""
    est = np.asarray(ccp_estimated)
    oracle = np.asarray(ccp_oracle)
    # Exclude absorbing state row
    mask = np.ones(est.shape[0], dtype=bool)
    mask[absorbing_state] = False
    return float(np.mean(np.abs(est[mask] - oracle[mask])))


def max_ccp_error(
    ccp_estimated: jnp.ndarray,
    ccp_oracle: jnp.ndarray,
    absorbing_state: int,
) -> float:
    """Maximum absolute CCP difference over non-absorbing states."""
    est = np.asarray(ccp_estimated)
    oracle = np.asarray(ccp_oracle)
    mask = np.ones(est.shape[0], dtype=bool)
    mask[absorbing_state] = False
    return float(np.max(np.abs(est[mask] - oracle[mask])))


def _reward_mask(
    shape: tuple[int, int],
    absorbing_state: int,
    exit_action: int,
) -> np.ndarray:
    """Boolean mask excluding absorbing state and exit action."""
    S, A = shape
    mask = np.ones((S, A), dtype=bool)
    mask[absorbing_state, :] = False
    mask[:, exit_action] = False
    return mask
