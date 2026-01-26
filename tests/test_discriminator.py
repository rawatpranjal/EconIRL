"""Tests for tabular discriminators."""

import pytest
import torch

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
        assert torch.allclose(disc.logits, torch.zeros(5, 2))

    def test_forward_returns_logits(self):
        """Forward pass should return logits for state-action pairs."""
        disc = TabularDiscriminator(n_states=5, n_actions=2, init="zeros")
        states = torch.tensor([0, 1, 2])
        actions = torch.tensor([0, 1, 0])
        logits = disc.forward(states, actions)
        assert logits.shape == (3,)

    def test_probability_is_sigmoid(self):
        """Probability should be sigmoid of logits."""
        disc = TabularDiscriminator(n_states=5, n_actions=2, init="zeros")
        disc.logits[0, 0] = 2.0  # Set known value
        states = torch.tensor([0])
        actions = torch.tensor([0])
        prob = disc.probability(states, actions)
        expected = torch.sigmoid(torch.tensor(2.0))
        assert torch.allclose(prob, expected.unsqueeze(0))

    def test_update_changes_logits(self):
        """Update should modify discriminator logits."""
        disc = TabularDiscriminator(n_states=5, n_actions=2, init="zeros")
        expert_states = torch.tensor([0, 1])
        expert_actions = torch.tensor([0, 0])
        policy_states = torch.tensor([2, 3])
        policy_actions = torch.tensor([1, 1])

        old_logits = disc.logits.clone()
        disc.update(
            expert_states, expert_actions,
            policy_states, policy_actions,
            learning_rate=0.1,
        )
        assert not torch.allclose(disc.logits, old_logits)


class TestLinearDiscriminator:
    """Tests for LinearDiscriminator."""

    def test_init_with_features(self):
        """Linear discriminator should use feature matrix."""
        features = torch.randn(5, 2, 3)  # (n_states, n_actions, n_features)
        disc = LinearDiscriminator(features)
        assert disc.weights.shape == (3,)

    def test_forward_is_linear(self):
        """Forward should compute linear combination of features."""
        features = torch.ones(5, 2, 3)  # All ones
        disc = LinearDiscriminator(features)
        disc.weights = torch.tensor([1.0, 2.0, 3.0])
        states = torch.tensor([0])
        actions = torch.tensor([0])
        logits = disc.forward(states, actions)
        # Features are all 1, weights are [1,2,3], so dot product = 6
        assert torch.allclose(logits, torch.tensor([6.0]))
