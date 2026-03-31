"""Tests for the BinaryworldEnvironment."""

import numpy as np
import pytest
import torch

from econirl.environments.binaryworld import BinaryworldEnvironment
from econirl.core.types import DDCProblem, Panel


class TestConstruction:
    """Basic construction and property tests."""

    def test_default_construction(self):
        """Default BinaryworldEnvironment should construct without error."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        assert env.num_states == 64
        assert env.num_actions == 5

    def test_num_states_matches_grid(self):
        """num_states should equal grid_size squared."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        assert env.num_states == 64

    def test_num_actions_is_five(self):
        """There should be exactly 5 actions."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        assert env.num_actions == 5


class TestTransitions:
    """Tests for the deterministic transition matrices."""

    def test_shape(self):
        """Transition matrices should have shape (5, 64, 64) for grid_size=8."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        T = env.transition_matrices
        assert T.shape == (5, 64, 64)

    def test_rows_sum_to_one(self):
        """All rows should sum to 1."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        T = env.transition_matrices
        for a in range(5):
            row_sums = T[a].sum(dim=1)
            assert torch.allclose(row_sums, torch.ones(64))

    def test_deterministic(self):
        """Each row should have a single 1.0 entry (deterministic transitions)."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        T = env.transition_matrices
        for a in range(5):
            assert T[a].max(dim=1).values.min().item() == pytest.approx(1.0)


class TestFeatures:
    """Tests for the binary neighborhood features."""

    def test_shape(self):
        """Features should have shape (64, 5, 9) for grid_size=8."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        F = env.feature_matrix
        assert F.shape == (64, 5, 9)

    def test_binary_values_only(self):
        """Feature values should be only 0 or 1."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        F = env.feature_matrix
        unique_vals = torch.unique(F)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_features_identical_across_actions(self):
        """State-only features should be the same for all 5 actions."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        F = env.feature_matrix
        for a in range(1, 5):
            assert torch.equal(F[:, 0, :], F[:, a, :])


class TestReward:
    """Tests for the reward function."""

    def test_shape(self):
        """true_reward should have shape (num_states,)."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        R = env.true_reward
        assert R.shape == (64,)

    def test_values_in_set(self):
        """Reward values should be in {-1, 0, 1}."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        R = env.true_reward
        for val in R:
            assert val.item() in {-1.0, 0.0, 1.0}

    def test_reward_matches_blue_count(self):
        """Critical test: every state reward must match blue count rule.

        For every state, count the blue cells in the 3x3 neighborhood
        (with zero-padding for out-of-bounds), then verify:
        +1 if exactly 4 are blue, -1 if exactly 5 are blue, 0 otherwise.
        """
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        R = env.true_reward
        F = env.feature_matrix
        for s in range(env.num_states):
            # Features for action 0 are the 9 binary neighborhood values
            neighborhood = F[s, 0, :].numpy()
            blue_count = int(neighborhood.sum())
            if blue_count == 4:
                expected = 1.0
            elif blue_count == 5:
                expected = -1.0
            else:
                expected = 0.0
            assert R[s].item() == expected, (
                f"State {s}: blue_count={blue_count}, "
                f"expected reward={expected}, got={R[s].item()}"
            )


class TestNeighborhood:
    """Tests for neighborhood feature computation."""

    def test_center_cell_neighborhood(self):
        """Center cell (3,3) on an 8x8 grid has 9 valid neighbors.

        All 9 entries in the feature vector should be 0 or 1, and
        since the cell is interior, all neighbors are in-bounds.
        """
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        F = env.feature_matrix
        # State for cell (3,3) in an 8x8 grid: row * 8 + col = 3*8+3 = 27
        state = 3 * 8 + 3
        neighborhood = F[state, 0, :]
        assert neighborhood.shape == (9,)
        # All 9 should be valid binary values
        for val in neighborhood:
            assert val.item() in {0.0, 1.0}

    def test_corner_cell_zero_padding(self):
        """Corner cell (0,0) has at most 4 in-bounds neighbors.

        Out-of-bounds neighbors are zero-padded, so at most 4 of the 9
        neighborhood entries can be blue (the 4 in-bounds cells including
        the cell itself: (0,0), (0,1), (1,0), (1,1)).
        """
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        F = env.feature_matrix
        # State for cell (0,0): state index 0
        neighborhood = F[0, 0, :]
        blue_count = int(neighborhood.sum().item())
        # The 3x3 neighborhood around (0,0) has 5 out-of-bounds cells,
        # so at most 4 can be blue
        assert blue_count <= 4, (
            f"Corner (0,0) has {blue_count} blue neighbors but max is 4"
        )


class TestSeedReproducibility:
    """Tests for random seed behavior."""

    def test_same_seed_same_result(self):
        """Same seed should produce identical color map, reward, and features."""
        env1 = BinaryworldEnvironment(grid_size=8, seed=42)
        env2 = BinaryworldEnvironment(grid_size=8, seed=42)
        assert torch.equal(env1.true_reward, env2.true_reward)
        assert torch.equal(env1.feature_matrix, env2.feature_matrix)

    def test_different_seed_different_result(self):
        """Different seeds should produce different color maps."""
        env1 = BinaryworldEnvironment(grid_size=8, seed=0)
        env2 = BinaryworldEnvironment(grid_size=8, seed=999)
        # With different random color assignments, rewards will differ
        assert not torch.equal(env1.true_reward, env2.true_reward)


class TestProblemSpec:
    """Tests for the DDCProblem specification."""

    def test_problem_spec(self):
        """problem_spec should have correct fields."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        spec = env.problem_spec
        assert isinstance(spec, DDCProblem)
        assert spec.num_states == 64
        assert spec.num_actions == 5
        assert spec.discount_factor == 0.9


class TestDemonstrations:
    """Tests for the simulate_demonstrations method."""

    def test_produces_valid_panel(self):
        """simulate_demonstrations should return a Panel with valid data."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        panel = env.simulate_demonstrations(n_demos=4, max_steps=50, seed=0)
        assert isinstance(panel, Panel)
        assert panel.num_individuals == 4

    def test_states_and_actions_in_range(self):
        """Demonstrated states and actions should be within valid ranges."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        panel = env.simulate_demonstrations(n_demos=4, max_steps=50, seed=0)
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        assert (all_states >= 0).all()
        assert (all_states < 64).all()
        assert (all_actions >= 0).all()
        assert (all_actions < 5).all()

    def test_demonstration_reproducibility(self):
        """Same seed should produce identical demonstrations."""
        env = BinaryworldEnvironment(grid_size=8, seed=0)
        panel1 = env.simulate_demonstrations(n_demos=4, max_steps=50, seed=0)
        panel2 = env.simulate_demonstrations(n_demos=4, max_steps=50, seed=0)
        states1 = panel1.get_all_states()
        states2 = panel2.get_all_states()
        assert torch.equal(states1, states2)
