"""Tests for standard-error formula wiring."""

import jax.numpy as jnp
import numpy as np
import pytest

from econirl.core.types import Panel, Trajectory
from econirl.inference.standard_errors import compute_standard_errors


def _toy_panel() -> Panel:
    return Panel(
        trajectories=[
            Trajectory(
                states=jnp.array([0, 1], dtype=jnp.int32),
                actions=jnp.array([0, 1], dtype=jnp.int32),
                next_states=jnp.array([1, 0], dtype=jnp.int32),
                individual_id=10,
            ),
            Trajectory(
                states=jnp.array([2], dtype=jnp.int32),
                actions=jnp.array([0], dtype=jnp.int32),
                next_states=jnp.array([2], dtype=jnp.int32),
                individual_id=20,
            ),
        ]
    )


def test_robust_sandwich_differs_from_asymptotic_when_information_equality_fails():
    """Robust SE should not collapse to asymptotic when OPG meat differs from Hessian."""
    params = jnp.array([1.0, 2.0])
    hessian = -jnp.array([[10.0, 0.0], [0.0, 5.0]])
    scores = jnp.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]])

    asymptotic = compute_standard_errors(params, hessian=hessian, method="asymptotic")
    robust = compute_standard_errors(
        params,
        hessian=hessian,
        gradient_contributions=scores,
        method="robust",
    )

    assert np.all(np.isfinite(np.asarray(asymptotic.standard_errors)))
    assert np.all(np.isfinite(np.asarray(robust.standard_errors)))
    assert not np.allclose(
        np.asarray(asymptotic.standard_errors),
        np.asarray(robust.standard_errors),
    )


def test_clustered_se_matches_manual_cluster_sandwich():
    """Clustered SE should use individual-summed score meat."""
    panel = _toy_panel()
    params = jnp.array([1.0, 2.0])
    hessian = -jnp.eye(2)
    scores = jnp.array([[1.0, 0.0], [2.0, 1.0], [0.0, 4.0]])

    result = compute_standard_errors(
        params,
        hessian=hessian,
        gradient_contributions=scores,
        panel=panel,
        method="clustered",
    )

    cluster_sums = np.array([[3.0, 1.0], [0.0, 4.0]])
    meat = cluster_sums.T @ cluster_sums
    correction = (2 / 1) * ((3 - 1) / (3 - 2))
    expected_var_cov = correction * meat

    np.testing.assert_allclose(
        np.asarray(result.variance_covariance),
        expected_var_cov,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.standard_errors),
        np.sqrt(np.diag(expected_var_cov)),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.details["n_clusters"] == 2


def test_bootstrap_resamples_whole_trajectories_not_rows():
    """Pairs-cluster bootstrap should preserve complete trajectory objects."""
    panel = _toy_panel()
    params = jnp.array([1.0, 2.0])
    seen_bootstrap_lengths: list[tuple[int, ...]] = []

    def estimate_fn(bootstrap_panel: Panel) -> jnp.ndarray:
        seen_bootstrap_lengths.append(tuple(len(traj) for traj in bootstrap_panel.trajectories))
        return jnp.array([
            float(sum(int(traj.individual_id) for traj in bootstrap_panel.trajectories)),
            float(sum(len(traj) for traj in bootstrap_panel.trajectories)),
        ])

    result = compute_standard_errors(
        params,
        panel=panel,
        method="bootstrap",
        n_bootstrap=8,
        seed=123,
        estimate_fn=estimate_fn,
    )

    assert result.details["successful_bootstraps"] == 8
    assert result.details["failed_bootstraps"] == 0
    assert all(
        tuple(sorted(lengths)) in {(1, 1), (1, 2), (2, 2)}
        for lengths in seen_bootstrap_lengths
    )
    assert np.all(np.isfinite(np.asarray(result.standard_errors)))


def test_bootstrap_requires_reestimation_callback():
    with pytest.raises(ValueError, match="estimate_fn required"):
        compute_standard_errors(
            jnp.array([1.0]),
            panel=_toy_panel(),
            method="bootstrap",
            n_bootstrap=2,
        )


def test_full_likelihood_bhhh_returns_structural_block_of_joint_opg_inverse():
    params = jnp.array([1.0, 2.0])
    scores = jnp.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 2.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ]
    )

    result = compute_standard_errors(
        params,
        gradient_contributions=scores,
        method="full_likelihood_bhhh",
    )

    joint_cov = np.linalg.inv(np.asarray(scores.T @ scores))
    expected_structural = joint_cov[:2, :2]

    np.testing.assert_allclose(
        np.asarray(result.variance_covariance),
        expected_structural,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result.standard_errors),
        np.sqrt(np.diag(expected_structural)),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.details["n_joint_parameters"] == 3
