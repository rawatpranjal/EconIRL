"""Tests for F2: road_network generator and graph form factory.

AC1 -- shapes, stochastic rows, theta+names, true_parameters, connectivity
       (every node reaches goal via finite shortest-path).
AC2 -- feature_diagnostics(env.feature_matrix)["contrast_rank"] == 3.
AC5 -- transition_noise=0.3 still gives valid stochastic rows and a connected
       graph (all nodes reachable from goal via finite shortest path).
"""

from __future__ import annotations

import numpy as np
import pytest

from econirl.environments.road_network import road_network
from econirl.preprocessing.diagnostics import feature_diagnostics


# ---------------------------------------------------------------------------
# AC1: shapes, stochastic rows, theta, names, true_parameters, connectivity
# ---------------------------------------------------------------------------


def test_road_network_shapes_default():
    env = road_network()
    assert env.num_states == 20
    assert env.num_actions == 4
    assert env.num_features == 3
    phi = np.asarray(env.feature_matrix)
    assert phi.shape == (20, 4, 3), f"feature_matrix shape {phi.shape}"
    T = np.asarray(env.transition_matrices)
    assert T.shape == (4, 20, 20), f"transitions shape {T.shape}"


def test_road_network_stochastic_rows():
    """All transition rows must sum to 1 (±1e-4 tolerance matching ArrayMDP validator)."""
    env = road_network(num_nodes=15, num_actions=3, seed=1)
    T = np.asarray(env.transition_matrices)
    row_sums = T.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"max row-sum deviation: {np.abs(row_sums - 1.0).max():.2e}"
    )
    assert (T >= 0).all(), "negative transition probability found"


def test_road_network_theta_and_names():
    """Known-truth theta must be [1.0, 0.5, 1.0] with the canonical names."""
    env = road_network(seed=0)
    expected_theta = [1.0, 0.5, 1.0]
    expected_names = ["edge_cost", "amenity", "goal"]
    assert env.parameter_names == expected_names, (
        f"names: {env.parameter_names!r}"
    )
    tp = env.true_parameters
    assert set(tp.keys()) == set(expected_names)
    for name, val in zip(expected_names, expected_theta):
        assert abs(tp[name] - val) < 1e-8, (
            f"true_parameters[{name!r}] = {tp[name]}, expected {val}"
        )


def test_road_network_connectivity():
    """Every node must reach the goal node via a finite shortest path.

    This holds because road_network adds the MST, so the graph is always
    connected regardless of the radius-based edges.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    env = road_network(num_nodes=20, num_actions=4, connectivity=0.15, seed=42)

    # Reconstruct adjacency from the transition tensor (any positive off-diagonal)
    T = np.asarray(env.transition_matrices)
    N = env.num_states
    # Adjacency: state j is reachable from state s via some action a
    adj_reach = (T > 0).any(axis=0)  # (S, S): adj_reach[s, s'] = some action can go to s'
    sp = shortest_path(csr_matrix(adj_reach.astype(float)), directed=False)
    # All pairwise distances must be finite (no disconnected nodes)
    assert np.isfinite(sp).all(), (
        f"Disconnected graph: {(~np.isfinite(sp)).sum()} infinite distances"
    )


# ---------------------------------------------------------------------------
# AC2: action-contrast rank == K == 3
# ---------------------------------------------------------------------------


def test_road_network_contrast_rank():
    """feature_diagnostics must report contrast_rank == 3 (all params identified)."""
    env = road_network(num_nodes=20, num_actions=4, seed=0)
    diag = feature_diagnostics(np.asarray(env.feature_matrix))
    assert diag["contrast_rank"] == 3, (
        f"contrast_rank={diag['contrast_rank']}, expected 3.  "
        f"Full diagnostics: {diag}"
    )


def test_road_network_contrast_rank_small():
    """Smaller graph still achieves contrast_rank == 3."""
    env = road_network(num_nodes=10, num_actions=3, seed=7)
    diag = feature_diagnostics(np.asarray(env.feature_matrix))
    assert diag["contrast_rank"] == 3, (
        f"contrast_rank={diag['contrast_rank']}.  diagnostics: {diag}"
    )


# ---------------------------------------------------------------------------
# AC5: transition_noise=0.3 still valid
# ---------------------------------------------------------------------------


def test_road_network_noisy_stochastic_rows():
    """With transition_noise=0.3 rows still sum to 1 and are non-negative."""
    env = road_network(num_nodes=15, num_actions=3, transition_noise=0.3, seed=2)
    T = np.asarray(env.transition_matrices)
    row_sums = T.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"max deviation: {np.abs(row_sums - 1.0).max():.2e}"
    )
    assert (T >= 0).all(), "negative probability with noise"


def test_road_network_noisy_connected():
    """With transition_noise=0.3 the graph is still fully connected."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    env = road_network(num_nodes=15, num_actions=3, transition_noise=0.3, seed=3)
    T = np.asarray(env.transition_matrices)
    adj_reach = (T > 0).any(axis=0)
    sp = shortest_path(csr_matrix(adj_reach.astype(float)), directed=False)
    assert np.isfinite(sp).all(), "Disconnected graph under transition noise"


# ---------------------------------------------------------------------------
# ValueError / NotImplementedError guards
# ---------------------------------------------------------------------------


def test_road_network_bad_reward_form():
    with pytest.raises(ValueError, match="reward_form"):
        road_network(reward_form="bogus")


def test_road_network_nonlinear_not_implemented():
    with pytest.raises(NotImplementedError):
        road_network(reward_form="nonlinear")


# ---------------------------------------------------------------------------
# Graph form factory via make_form
# ---------------------------------------------------------------------------


def test_make_form_graph_linear():
    from econirl.forms import make_form, Form

    form = make_form("graph", reward_form="linear", num_nodes=10, num_actions=3, seed=0)
    assert isinstance(form, Form)
    assert form.spec.topology == "graph"
    assert form.spec.reward_form == "linear"
    assert form.spec.num_states == 10
    assert form.spec.num_actions == 3
    assert form.spec.has_transitions is True
    tp = form.env.true_parameters
    assert isinstance(tp, dict) and len(tp) == 3
    phi = np.asarray(form.env.feature_matrix)
    assert phi.ndim == 3 and phi.shape == (10, 3, 3)


def test_make_form_graph_nonlinear_raises():
    from econirl.forms import make_form

    with pytest.raises(NotImplementedError):
        make_form("graph", reward_form="nonlinear")


# ---------------------------------------------------------------------------
# Known-truth recovery (SLOW)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_graph_recovery_nfxp_ccp():
    """NFXP and CCP must recover road-network theta within RMSE < 0.2.

    Small graph (10 nodes, 3 actions, K=3) with 400 individuals × 50 periods.
    True theta = [1.0, 0.5, 1.0].  Actual RMSE is typically < 0.05 when
    identification is clean; 0.2 catches any regression to garbage.
    """
    from econirl.forms import make_form, run_form

    form = make_form(
        "graph", reward_form="linear",
        num_nodes=10, num_actions=3, seed=0,
    )
    rr = run_form(
        form,
        estimators=["NFXP", "CCP"],
        n_individuals=400,
        n_periods=50,
        seed=0,
    )

    true_theta = np.array(list(form.env.true_parameters.values()))

    assert "NFXP" in rr.results, f"NFXP missing; skipped={rr.skipped}"
    assert "CCP" in rr.results, f"CCP missing; skipped={rr.skipped}"

    for name in ("NFXP", "CCP"):
        res = rr.results[name]
        converged = getattr(res, "converged", None)
        assert converged, f"{name}: converged={converged!r}"

        params = getattr(res, "parameters", None)
        assert params is not None, f"{name}: parameters is None"
        recovered = np.asarray(params).reshape(-1)

        assert recovered.shape == true_theta.shape, (
            f"{name}: shape mismatch {recovered.shape} vs {true_theta.shape}"
        )
        rmse = float(np.sqrt(np.mean((recovered - true_theta) ** 2)))
        print(
            f"  {name}: true={true_theta.tolist()}, "
            f"recovered={recovered.tolist()}, RMSE={rmse:.4f}"
        )
        assert rmse < 0.2, (
            f"{name}: RMSE={rmse:.4f} exceeds 0.2 threshold.  "
            f"true={true_theta}, recovered={recovered}"
        )
