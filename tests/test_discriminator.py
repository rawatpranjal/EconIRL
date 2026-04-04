"""Tests for tabular discriminators."""

import pytest
import jax.numpy as jnp
import numpy as np

from econirl.estimation.adversarial.discriminator import (
    TabularDiscriminator,
    LinearDiscriminator,
)


class TestTabularDiscriminator:
    """Tests for TabularDiscriminator."""

    def test_init_zeros(self):
        """Discriminator should initialize with zeros."""
        disc = TabularDiscriminator(n_states=5, n_actions=2, init="zeros")
        assert disc.logits.shape == (5, 2)
        np.testing.assert_allclose(np.asarray(disc.logits), np.asarray(jnp.zeros((5, 2))))

    def test_forward_returns_logits(self):
        """Forward pass should return logits for state-action pairs."""
        disc = TabularDiscriminator(n_states=5, n_actions=2, init="zeros")
        states = jnp.array([0, 1, 2], dtype=jnp.int32)
        actions = jnp.array([0, 1, 0], dtype=jnp.int32)
        logits = disc.forward(states, actions)
        assert logits.shape == (3,)

    def test_probability_is_sigmoid(self):
        """Probability should be sigmoid of logits."""
        disc = TabularDiscriminator(n_states=5, n_actions=2, init="zeros")
        disc.logits = disc.logits.at[0, 0].set(2.0)  # Set known value
        states = jnp.array([0], dtype=jnp.int32)
        actions = jnp.array([0], dtype=jnp.int32)
        prob = disc.probability(states, actions)
        expected = 1.0 / (1.0 + jnp.exp(-jnp.array(2.0)))
        np.testing.assert_allclose(np.asarray(prob), np.asarray(jnp.expand_dims(expected, 0)))

    def test_update_changes_logits(self):
        """Update should modify discriminator logits."""
        disc = TabularDiscriminator(n_states=5, n_actions=2, init="zeros")
        expert_states = jnp.array([0, 1], dtype=jnp.int32)
        expert_actions = jnp.array([0, 0], dtype=jnp.int32)
        policy_states = jnp.array([2, 3], dtype=jnp.int32)
        policy_actions = jnp.array([1, 1], dtype=jnp.int32)

        old_logits = jnp.array(disc.logits)
        disc.update(
            expert_states, expert_actions,
            policy_states, policy_actions,
            learning_rate=0.1,
        )
        assert not jnp.allclose(disc.logits, old_logits)


class TestLinearDiscriminator:
    """Tests for LinearDiscriminator."""

    def test_init_with_features(self):
        """Linear discriminator should use feature matrix."""
        np.random.seed(0)
        features = jnp.array(np.random.randn(5, 2, 3))  # (n_states, n_actions, n_features)
        disc = LinearDiscriminator(features)
        assert disc.weights.shape == (3,)

    def test_forward_is_linear(self):
        """Forward should compute linear combination of features."""
        features = jnp.ones((5, 2, 3))  # All ones
        disc = LinearDiscriminator(features)
        disc.weights = jnp.array([1.0, 2.0, 3.0])
        states = jnp.array([0], dtype=jnp.int32)
        actions = jnp.array([0], dtype=jnp.int32)
        logits = disc.forward(states, actions)
        # Features are all 1, weights are [1,2,3], so dot product = 6
        np.testing.assert_allclose(np.asarray(logits), np.asarray(jnp.array([6.0])))
