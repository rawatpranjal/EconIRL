"""Tests for the Shanghai taxi route-choice dataset loader."""

import pytest
import jax.numpy as jnp
import numpy as np

from econirl.datasets.shanghai_route import (
    load_shanghai_network,
    load_shanghai_trajectories,
    build_transition_matrix,
    build_edge_features,
    build_state_action_features,
    add_destination_feature,
    parse_trajectories_to_panel,
    N_STATES,
    N_ACTIONS,
)


DATA_DIR = "/Volumes/Expansion/datasets/shanghai_taxi_rcm_airl/data"


@pytest.fixture(scope="module")
def network():
    return load_shanghai_network(DATA_DIR)


@pytest.fixture(scope="module")
def transit(network):
    return network["transit"]


@pytest.fixture(scope="module")
def edges_df(network):
    return network["edges"]


@pytest.fixture(scope="module")
def nodes_df(network):
    return network["nodes"]


@pytest.fixture(scope="module")
def transition_matrix(transit):
    return build_transition_matrix(transit)


@pytest.fixture(scope="module")
def edge_features(edges_df):
    return build_edge_features(edges_df)


class TestLoadNetwork:
    def test_nodes_count(self, network):
        assert len(network["nodes"]) == 320

    def test_edges_count(self, network):
        assert len(network["edges"]) == 714

    def test_transit_count(self, network):
        assert network["transit"].shape == (1737, 3)

    def test_n_states(self, network):
        assert network["n_states"] == 714

    def test_n_actions(self, network):
        assert network["n_actions"] == 8

    def test_node_columns(self, network):
        assert "osmid" in network["nodes"].columns
        assert "lat" in network["nodes"].columns
        assert "lon" in network["nodes"].columns


class TestTransitionMatrix:
    def test_shape(self, transition_matrix):
        assert transition_matrix.shape == (N_ACTIONS, N_STATES, N_STATES)

    def test_rows_sum_to_one(self, transition_matrix):
        row_sums = transition_matrix.sum(axis=2)
        assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-6)

    def test_deterministic(self, transition_matrix, transit):
        """Valid (s, a) pairs from transit should have exactly one non-zero successor."""
        for row in transit:
            from_s, action, to_s = int(row[0]), int(row[1]), int(row[2])
            row_vals = transition_matrix[action, from_s]
            assert (row_vals > 0).sum() == 1
            assert float(row_vals[to_s]) == 1.0

    def test_non_negative(self, transition_matrix):
        assert (transition_matrix >= 0).all()


class TestEdgeFeatures:
    def test_shape(self, edge_features):
        assert edge_features.shape == (N_STATES, 7)

    def test_length_normalized(self, edge_features):
        """Length feature (column 0) should be in [0, 1]."""
        lengths = edge_features[:, 0]
        assert float(lengths.min()) >= 0.0
        assert float(lengths.max()) <= 1.0 + 1e-6
        # At least one edge should have max length = 1.0
        assert float(lengths.max()) > 0.99

    def test_highway_onehot(self, edge_features):
        """Each edge should have exactly one highway type active."""
        highway_cols = edge_features[:, 1:]
        row_sums = highway_cols.sum(axis=1)
        assert (row_sums == 1.0).all()


class TestStateActionFeatures:
    def test_shape(self, edge_features, transit):
        sa_features = build_state_action_features(edge_features, transit)
        assert sa_features.shape == (N_STATES, N_ACTIONS, 7)

    def test_valid_transitions_nonzero(self, edge_features, transit):
        """Features for valid (s, a) from transit should be non-zero."""
        sa_features = build_state_action_features(edge_features, transit)
        for row in transit[:20]:  # Check first 20
            from_s, action, to_s = int(row[0]), int(row[1]), int(row[2])
            assert float(jnp.abs(sa_features[from_s, action]).sum()) > 0

    def test_matches_edge_features(self, edge_features, transit):
        """SA features at (from_s, action) should equal edge_features[to_s]."""
        sa_features = build_state_action_features(edge_features, transit)
        for row in transit[:20]:
            from_s, action, to_s = int(row[0]), int(row[1]), int(row[2])
            assert jnp.allclose(sa_features[from_s, action], edge_features[to_s])


class TestTrajectoryParsing:
    def test_parse_trajectories(self, transit):
        traj_df = load_shanghai_trajectories(split="train", cv=0, size=1000, data_dir=DATA_DIR)
        panel = parse_trajectories_to_panel(traj_df.head(10), transit)

        assert len(panel) == 10

        for traj in panel:
            # States should be valid IDs
            assert (traj.states >= 0).all()
            assert (traj.states < N_STATES).all()
            assert (traj.next_states >= 0).all()
            assert (traj.next_states < N_STATES).all()
            assert (traj.actions >= 0).all()
            assert (traj.actions < N_ACTIONS).all()

    def test_trajectory_panel_shape(self, transit):
        traj_df = load_shanghai_trajectories(split="train", cv=0, size=1000, data_dir=DATA_DIR)
        panel = parse_trajectories_to_panel(traj_df.head(50), transit)

        assert panel.num_individuals == 50
        assert panel.num_observations > 50  # Each route has multiple steps

    def test_state_continuity(self, transit):
        """next_states[t] should match states[t+1] within a trajectory."""
        traj_df = load_shanghai_trajectories(split="train", cv=0, size=1000, data_dir=DATA_DIR)
        panel = parse_trajectories_to_panel(traj_df.head(10), transit)

        for traj in panel:
            if len(traj) > 1:
                assert jnp.array_equal(traj.next_states[:-1], traj.states[1:])


class TestDestinationFeature:
    def test_edge_features_2d(self, edge_features, nodes_df, edges_df):
        result = add_destination_feature(edge_features, nodes_df, edges_df, destination_nid=0)
        assert result.shape == (N_STATES, 8)
        # Distance feature should be non-negative
        assert (result[:, -1] >= 0).all()
        # Destination itself should have distance ~0
        assert float(result[0, -1]) < 0.01

    def test_sa_features_3d(self, edge_features, nodes_df, edges_df, transit):
        sa_features = build_state_action_features(edge_features, transit)
        result = add_destination_feature(sa_features, nodes_df, edges_df, destination_nid=100)
        assert result.shape == (N_STATES, N_ACTIONS, 8)
        assert (result[:, :, -1] >= 0).all()
