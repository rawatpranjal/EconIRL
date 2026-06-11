"""Unit tests for the ArrayMDP injector and the random_mdp generator.

These verify the injection contract (shapes, validation, reward), seed
reproducibility of the generator, and that an injected environment plugs into
the existing ``simulate_panel`` harness with no harness changes. They are cheap
and do not exercise estimators (that is the benchmark engine's job).
"""

from __future__ import annotations

import numpy as np
import pytest

from econirl.environments import ArrayMDP, random_mdp
from econirl.simulation.synthetic import simulate_panel


def _identity_transitions(S: int, A: int) -> np.ndarray:
    T = np.zeros((A, S, S), dtype=np.float64)
    T[:, np.arange(S), np.arange(S)] = 1.0
    return T


def _toy_env(S: int = 5, A: int = 2, K: int = 2, seed: int = 0) -> ArrayMDP:
    rng = np.random.default_rng(seed)
    T = np.zeros((A, S, S), dtype=np.float64)
    for a in range(A):
        for s in range(S):
            w = rng.dirichlet(np.ones(S))
            T[a, s] = w
    phi = rng.normal(size=(S, A, K))
    return ArrayMDP(T, phi, theta=np.array([-1.0, 0.5]), discount_factor=0.9, seed=seed)


# ---------------------------------------------------------------------------
# Shape and reward contracts
# ---------------------------------------------------------------------------


def test_array_mdp_shapes() -> None:
    env = _toy_env(S=5, A=2, K=2)
    assert env.num_states == 5
    assert env.num_actions == 2
    assert env.num_features == 2
    assert env.transition_matrices.shape == (2, 5, 5)
    assert env.feature_matrix.shape == (5, 2, 2)
    assert env.true_reward_matrix.shape == (5, 2)
    row_sums = np.asarray(env.transition_matrices).sum(axis=2)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)


def test_array_mdp_reward_is_linear_in_theta() -> None:
    S, A, K = 4, 2, 3
    phi = np.random.default_rng(1).normal(size=(S, A, K))
    theta = np.array([0.3, -0.7, 1.1])
    env = ArrayMDP(_identity_transitions(S, A), phi, theta)
    expected = np.einsum("sak,k->sa", phi, theta)
    np.testing.assert_allclose(np.asarray(env.true_reward_matrix), expected, atol=1e-4)
    # _compute_flow_utility agrees with the matrix.
    assert env._compute_flow_utility(2, 1) == pytest.approx(float(expected[2, 1]), abs=1e-4)


def test_theta_as_mapping_sets_named_parameters() -> None:
    S, A, K = 3, 2, 2
    phi = np.ones((S, A, K))
    env = ArrayMDP(_identity_transitions(S, A), phi, theta={"cost": -1.0, "value": 0.5})
    assert env.parameter_names == ["cost", "value"]
    assert env.true_parameters == {"cost": -1.0, "value": 0.5}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_unnormalized_transitions() -> None:
    S, A, K = 3, 2, 1
    T = np.ones((A, S, S))  # rows sum to S, not 1
    phi = np.ones((S, A, K))
    with pytest.raises(ValueError, match="sum to 1"):
        ArrayMDP(T, phi, theta=[1.0])


def test_rejects_feature_transition_mismatch() -> None:
    S, A, K = 4, 2, 2
    T = _identity_transitions(S, A)
    phi = np.ones((S + 1, A, K))  # wrong S
    with pytest.raises(ValueError, match="inconsistent"):
        ArrayMDP(T, phi, theta=[1.0, 1.0])


def test_rejects_theta_length_mismatch() -> None:
    S, A, K = 3, 2, 2
    T = _identity_transitions(S, A)
    phi = np.ones((S, A, K))
    with pytest.raises(ValueError, match="match"):
        ArrayMDP(T, phi, theta=[1.0, 1.0, 1.0])  # K=2 but 3 params


# ---------------------------------------------------------------------------
# Plugs into the existing harness
# ---------------------------------------------------------------------------


def test_array_mdp_simulates_panel() -> None:
    env = _toy_env(S=6, A=2, K=2)
    panel = simulate_panel(env, n_individuals=8, n_periods=15, seed=3)
    assert len(panel.trajectories) == 8
    traj = panel.trajectories[0]
    assert len(traj.states) == 15
    # States and actions stay in range.
    assert int(np.asarray(traj.states).max()) < env.num_states
    assert int(np.asarray(traj.actions).max()) < env.num_actions


# ---------------------------------------------------------------------------
# random_mdp generator
# ---------------------------------------------------------------------------


def test_random_mdp_is_valid_and_sparse() -> None:
    env = random_mdp(num_states=20, num_actions=3, num_features=3, branching=4, seed=7)
    assert env.transition_matrices.shape == (3, 20, 20)
    row_sums = np.asarray(env.transition_matrices).sum(axis=2)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)
    # Branching: each row has at most branching+1 nonzeros (support + self-loop).
    nnz = (np.asarray(env.transition_matrices) > 0).sum(axis=2)
    assert nnz.max() <= 4 + 1


def test_random_mdp_seed_reproducible() -> None:
    a = random_mdp(num_states=15, num_actions=2, seed=42)
    b = random_mdp(num_states=15, num_actions=2, seed=42)
    np.testing.assert_allclose(
        np.asarray(a.transition_matrices), np.asarray(b.transition_matrices)
    )
    np.testing.assert_allclose(
        np.asarray(a.true_reward_matrix), np.asarray(b.true_reward_matrix)
    )


def test_random_mdp_different_seeds_differ() -> None:
    a = random_mdp(num_states=15, num_actions=2, seed=1)
    b = random_mdp(num_states=15, num_actions=2, seed=2)
    assert not np.allclose(
        np.asarray(a.transition_matrices), np.asarray(b.transition_matrices)
    )


def test_random_mdp_simulates_panel() -> None:
    env = random_mdp(num_states=12, num_actions=2, branching=3, seed=5)
    panel = simulate_panel(env, n_individuals=5, n_periods=10, seed=9)
    assert len(panel.trajectories) == 5
