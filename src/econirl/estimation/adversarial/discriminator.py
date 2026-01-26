"""Tabular discriminators for adversarial IRL.

This module implements discriminators that classify state-action pairs as
expert or policy-generated. For tabular MDPs, we use direct matrix storage
rather than neural networks.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


class TabularDiscriminator:
    """Tabular discriminator storing D(s,a) as a matrix.

    For discrete state-action spaces, we directly store discriminator
    values in a matrix rather than using a neural network.

    Attributes:
        logits: Discriminator logits D(s,a), shape (n_states, n_actions)
        n_states: Number of states
        n_actions: Number of actions
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        init: Literal["zeros", "uniform", "small_positive"] = "zeros",
    ):
        """Initialize tabular discriminator.

        Args:
            n_states: Number of states in the MDP
            n_actions: Number of actions
            init: Initialization method for logits
        """
        self.n_states = n_states
        self.n_actions = n_actions

        if init == "zeros":
            self.logits = torch.zeros(n_states, n_actions)
        elif init == "uniform":
            self.logits = torch.rand(n_states, n_actions) * 2 - 1  # [-1, 1]
        elif init == "small_positive":
            self.logits = torch.ones(n_states, n_actions) * 0.1
        else:
            raise ValueError(f"Unknown init method: {init}")

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute discriminator logits for state-action pairs.

        Args:
            states: State indices, shape (batch_size,)
            actions: Action indices, shape (batch_size,)

        Returns:
            Discriminator logits, shape (batch_size,)
        """
        return self.logits[states, actions]

    def probability(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute probability that (s,a) is from expert.

        Args:
            states: State indices, shape (batch_size,)
            actions: Action indices, shape (batch_size,)

        Returns:
            P(expert | s, a) = sigmoid(D(s,a)), shape (batch_size,)
        """
        logits = self.forward(states, actions)
        return torch.sigmoid(logits)

    def update(
        self,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor,
        learning_rate: float = 0.01,
    ) -> float:
        """Update discriminator with binary cross-entropy gradient.

        Maximizes: E_expert[log D] + E_policy[log(1-D)]

        Args:
            expert_states: States from expert demonstrations
            expert_actions: Actions from expert demonstrations
            policy_states: States from current policy
            policy_actions: Actions from current policy
            learning_rate: Step size for gradient update

        Returns:
            Binary cross-entropy loss (for monitoring)
        """
        # Expert samples: want D(s,a) to be high (label = 1)
        expert_logits = self.forward(expert_states, expert_actions)
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits)
        )

        # Policy samples: want D(s,a) to be low (label = 0)
        policy_logits = self.forward(policy_states, policy_actions)
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_logits, torch.zeros_like(policy_logits)
        )

        total_loss = expert_loss + policy_loss

        # Compute gradients manually (tabular, so gradient is simple)
        # For BCE: dL/d(logit) = sigmoid(logit) - label
        expert_grad = torch.sigmoid(expert_logits) - 1.0  # Want to increase
        policy_grad = torch.sigmoid(policy_logits) - 0.0  # Want to decrease

        # Apply gradients
        for i, (s, a) in enumerate(zip(expert_states, expert_actions)):
            self.logits[s, a] -= learning_rate * expert_grad[i]

        for i, (s, a) in enumerate(zip(policy_states, policy_actions)):
            self.logits[s, a] -= learning_rate * policy_grad[i]

        return total_loss.item()

    def get_reward_matrix(
        self, reward_type: Literal["gail", "airl"] = "gail"
    ) -> torch.Tensor:
        """Convert discriminator to reward matrix.

        Args:
            reward_type: How to derive reward from discriminator
                - "gail": R = -log(1 - D) = log(1 + exp(logit))
                - "airl": R = logit (the discriminator output directly)

        Returns:
            Reward matrix R(s,a), shape (n_states, n_actions)
        """
        if reward_type == "gail":
            # R = -log(1 - sigmoid(D)) = log(1 + exp(D)) = softplus(D)
            return F.softplus(self.logits)
        elif reward_type == "airl":
            # For AIRL, the reward is directly the logit
            return self.logits.clone()
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")


class LinearDiscriminator:
    """Linear discriminator: D(s,a) = w^T * phi(s,a).

    Uses a linear function of features for interpretability.
    The discriminator output is a weighted sum of state-action features.

    Attributes:
        feature_matrix: Features phi(s,a,k), shape (n_states, n_actions, n_features)
        weights: Linear weights w, shape (n_features,)
    """

    def __init__(
        self,
        feature_matrix: torch.Tensor,
        init: Literal["zeros", "uniform", "ones"] = "zeros",
    ):
        """Initialize linear discriminator.

        Args:
            feature_matrix: State-action features, shape (n_states, n_actions, n_features)
            init: Initialization for weights
        """
        if feature_matrix.ndim != 3:
            raise ValueError(
                f"feature_matrix must be 3D (n_states, n_actions, n_features), "
                f"got shape {feature_matrix.shape}"
            )

        self.feature_matrix = feature_matrix
        self.n_states, self.n_actions, self.n_features = feature_matrix.shape

        if init == "zeros":
            self.weights = torch.zeros(self.n_features)
        elif init == "uniform":
            self.weights = torch.rand(self.n_features) * 2 - 1
        elif init == "ones":
            self.weights = torch.ones(self.n_features)
        else:
            raise ValueError(f"Unknown init method: {init}")

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute discriminator logits as linear combination of features.

        D(s,a) = w^T * phi(s,a)

        Args:
            states: State indices, shape (batch_size,)
            actions: Action indices, shape (batch_size,)

        Returns:
            Discriminator logits, shape (batch_size,)
        """
        # Get features for requested state-action pairs
        features = self.feature_matrix[states, actions, :]  # (batch, n_features)
        return features @ self.weights  # (batch,)

    def probability(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute probability that (s,a) is from expert."""
        logits = self.forward(states, actions)
        return torch.sigmoid(logits)

    def update(
        self,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor,
        learning_rate: float = 0.01,
    ) -> float:
        """Update discriminator weights with BCE gradient.

        Args:
            expert_states: States from expert demonstrations
            expert_actions: Actions from expert demonstrations
            policy_states: States from current policy
            policy_actions: Actions from current policy
            learning_rate: Step size

        Returns:
            Loss value
        """
        # Expert: want high probability
        expert_logits = self.forward(expert_states, expert_actions)
        expert_probs = torch.sigmoid(expert_logits)
        expert_features = self.feature_matrix[expert_states, expert_actions, :]

        # Policy: want low probability
        policy_logits = self.forward(policy_states, policy_actions)
        policy_probs = torch.sigmoid(policy_logits)
        policy_features = self.feature_matrix[policy_states, policy_actions, :]

        # Gradient: dL/dw = sum(sigmoid(logit) - label) * features
        expert_grad = ((expert_probs - 1.0).unsqueeze(1) * expert_features).mean(dim=0)
        policy_grad = ((policy_probs - 0.0).unsqueeze(1) * policy_features).mean(dim=0)

        self.weights -= learning_rate * (expert_grad + policy_grad)

        # Compute loss for monitoring
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits)
        )
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_logits, torch.zeros_like(policy_logits)
        )

        return (expert_loss + policy_loss).item()

    def get_reward_matrix(
        self, reward_type: Literal["gail", "airl"] = "gail"
    ) -> torch.Tensor:
        """Convert discriminator to full reward matrix.

        Args:
            reward_type: How to derive reward

        Returns:
            Reward matrix R(s,a), shape (n_states, n_actions)
        """
        # Compute D(s,a) for all state-action pairs
        # D(s,a) = phi(s,a) @ w
        logits = torch.einsum("sak,k->sa", self.feature_matrix, self.weights)

        if reward_type == "gail":
            return F.softplus(logits)
        elif reward_type == "airl":
            return logits
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")
