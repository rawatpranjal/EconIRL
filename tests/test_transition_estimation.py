"""Tests for transition probability estimation."""

import pytest
import numpy as np
from econirl.estimation.transitions import estimate_transition_probs, estimate_transition_probs_by_group
from econirl.datasets import load_rust_bus


class TestTransitionEstimation:
    """Tests for first-stage transition probability estimation."""

    def test_probs_sum_to_one(self):
        """Transition probabilities must sum to 1."""
        df = load_rust_bus(group=1)
        probs = estimate_transition_probs(df)

        assert np.isclose(probs.sum(), 1.0, atol=1e-6)

    def test_probs_non_negative(self):
        """All probabilities must be non-negative."""
        df = load_rust_bus(group=1)
        probs = estimate_transition_probs(df)

        assert (probs >= 0).all()

    def test_returns_three_probs(self):
        """Should return exactly 3 probabilities (0, 1, 2 bin increments)."""
        df = load_rust_bus(group=1)
        probs = estimate_transition_probs(df)

        assert len(probs) == 3

    def test_matches_rust_estimates(self):
        """Estimates should be close to Rust's Table IV values."""
        # Rust (1987) Table IV estimates for Group 4:
        # theta_0 = 0.3919, theta_1 = 0.5953, theta_2 = 0.0128
        df = load_rust_bus(group=4)
        probs = estimate_transition_probs(df)

        # Allow 10% relative tolerance due to data differences
        assert np.isclose(probs[0], 0.3919, rtol=0.1)
        assert np.isclose(probs[1], 0.5953, rtol=0.1)
        assert np.isclose(probs[2], 0.0128, rtol=0.2)  # Small prob, more tolerance

    def test_estimate_by_group(self):
        """Test that by-group estimation returns valid probabilities for each group."""
        df = load_rust_bus()  # Load all groups
        probs_by_group = estimate_transition_probs_by_group(df)

        # Should have multiple groups
        assert len(probs_by_group) > 0

        # Each group's probabilities should be valid
        for group, probs in probs_by_group.items():
            assert len(probs) == 3, f"Group {group} should have 3 probabilities"
            assert np.isclose(probs.sum(), 1.0, atol=1e-6), f"Group {group} probs must sum to 1"
            assert (probs >= 0).all(), f"Group {group} probs must be non-negative"
