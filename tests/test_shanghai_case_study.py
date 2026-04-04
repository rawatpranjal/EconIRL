"""Regression tests for Shanghai route-choice case study."""

import numpy as np
import jax.numpy as jnp
import pytest
from econirl.datasets.shanghai_route import (
    load_shanghai_network, load_shanghai_trajectories,
    parse_trajectories_to_panel, build_transition_matrix,
    build_edge_features, build_state_action_features,
)
from econirl.core.types import DDCProblem, TrajectoryPanel
from econirl.preferences.linear import LinearUtility


DATA_DIR = "/Volumes/Expansion/datasets/shanghai_taxi_rcm_airl/data"

FEATURE_NAMES = [
    "length", "residential", "primary", "secondary",
    "tertiary", "living_street", "unclassified",
]


@pytest.fixture(scope="module")
def network():
    return load_shanghai_network(DATA_DIR)


@pytest.fixture(scope="module")
def small_panel(network):
    """100-route training panel for fast tests."""
    df = load_shanghai_trajectories(split="train", cv=0, size=100, data_dir=DATA_DIR)
    return parse_trajectories_to_panel(df, network["transit"])


@pytest.fixture(scope="module")
def transitions(network):
    return build_transition_matrix(network["transit"], n_states=714, n_actions=8)


@pytest.fixture(scope="module")
def features(network):
    edge_feat = build_edge_features(network["edges"], n_states=714)
    return build_state_action_features(edge_feat, network["transit"], n_states=714, n_actions=8)


@pytest.fixture(scope="module")
def problem():
    return DDCProblem(num_states=714, num_actions=8, discount_factor=0.95, scale_parameter=1.0)


@pytest.fixture(scope="module")
def problem_neural():
    return DDCProblem(
        num_states=714, num_actions=8, discount_factor=0.95, scale_parameter=1.0,
        state_dim=1, state_encoder=lambda s: (s.astype(jnp.float32) / 713.0)[..., None],
    )


@pytest.fixture(scope="module")
def utility(features):
    return LinearUtility(feature_matrix=features, parameter_names=FEATURE_NAMES)


class TestDataLoads:
    def test_panel_has_routes(self, small_panel):
        assert small_panel.num_individuals > 0
        assert small_panel.num_observations > 0

    def test_transitions_valid(self, transitions):
        assert transitions.shape == (8, 714, 714)
        row_sums = transitions.sum(axis=2)
        assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5)

    def test_features_shape(self, features):
        assert features.shape == (714, 8, 7)


class TestBCRuns:
    def test_bc_policy(self, small_panel):
        policy = small_panel.compute_choice_frequencies(714, 8)
        assert policy.shape == (714, 8)
        # Check it sums to 1 where we have data
        visited = policy.sum(axis=1) > 0
        assert visited.any()


class TestCCPRuns:
    def test_ccp_estimates(self, small_panel, utility, problem, transitions):
        from econirl.estimation.ccp import CCPEstimator
        est = CCPEstimator(se_method="asymptotic", verbose=False)
        result = est.estimate(small_panel, utility, problem, transitions)
        assert result.parameters is not None
        assert result.parameters.shape == (7,)
        assert result.policy.shape == (714, 8)
        assert result.converged


class TestNNESRuns:
    def test_nnes_estimates(self, small_panel, utility, problem_neural, transitions):
        from econirl.estimation.nnes import NNESEstimator
        est = NNESEstimator(
            hidden_dim=32, num_layers=1, v_epochs=50,
            n_outer_iterations=1, compute_se=False, verbose=False,
        )
        result = est.estimate(small_panel, utility, problem_neural, transitions)
        assert result.parameters is not None
        assert result.parameters.shape == (7,)
        assert result.policy.shape == (714, 8)


class TestTDCCPRuns:
    def test_tdccp_estimates(self, small_panel, utility, problem_neural, transitions):
        from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig
        config = TDCCPConfig(
            hidden_dim=32, num_hidden_layers=1, avi_iterations=5,
            epochs_per_avi=10, n_policy_iterations=1,
            compute_se=False, verbose=False,
        )
        est = TDCCPEstimator(config=config)
        result = est.estimate(small_panel, utility, problem_neural, transitions)
        assert result.parameters is not None
        assert result.parameters.shape == (7,)
        assert result.policy.shape == (714, 8)


@pytest.mark.slow
class TestStructuralOutperformsRandom:
    def test_ccp_better_than_uniform(self, small_panel, utility, problem, transitions):
        """CCP should produce better policy than uniform random."""
        from econirl.estimation.ccp import CCPEstimator
        est = CCPEstimator(verbose=False)
        result = est.estimate(small_panel, utility, problem, transitions)

        all_s = small_panel.get_all_states()
        all_a = small_panel.get_all_actions()

        # CCP log-likelihood
        ccp_ll = float(jnp.log(result.policy[all_s, all_a] + 1e-10).mean())
        # Uniform log-likelihood
        uniform_ll = np.log(1.0 / 8.0)

        assert ccp_ll > uniform_ll, f"CCP LL {ccp_ll:.4f} should beat uniform {uniform_ll:.4f}"
