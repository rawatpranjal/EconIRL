"""Tabular discriminators for adversarial IRL.

This module implements discriminators that classify state-action pairs as
expert or policy-generated. For tabular MDPs, we use direct matrix storage
rather than neural networks.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


def _bce_with_logits(logits, labels):
    """Binary cross-entropy with logits (numerically stable)."""
    return jnp.mean(jnp.logaddexp(0.0, logits) - labels * logits)


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
        key: jax.Array | None = None,
    ):
        """Initialize tabular discriminator.

        Args:
            n_states: Number of states in the MDP
            n_actions: Number of actions
            init: Initialization method for logits
            key: PRNG key (required for "uniform" init)
        """
        self.n_states = n_states
        self.n_actions = n_actions

        if init == "zeros":
            self.logits = jnp.zeros((n_states, n_actions))
        elif init == "uniform":
            if key is None:
                key = jax.random.key(0)
            self.logits = jax.random.uniform(key, (n_states, n_actions)) * 2 - 1
        elif init == "small_positive":
            self.logits = jnp.ones((n_states, n_actions)) * 0.1
        else:
            raise ValueError(f"Unknown init method: {init}")

    def forward(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Compute discriminator logits for state-action pairs.

        Args:
            states: State indices, shape (batch_size,)
            actions: Action indices, shape (batch_size,)

        Returns:
            Discriminator logits, shape (batch_size,)
        """
        return self.logits[states, actions]

    def probability(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Compute probability that (s,a) is from expert.

        Args:
            states: State indices, shape (batch_size,)
            actions: Action indices, shape (batch_size,)

        Returns:
            P(expert | s, a) = sigmoid(D(s,a)), shape (batch_size,)
        """
        logits = self.forward(states, actions)
        return jax.nn.sigmoid(logits)

    def update(
        self,
        expert_states: jnp.ndarray,
        expert_actions: jnp.ndarray,
        policy_states: jnp.ndarray,
        policy_actions: jnp.ndarray,
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
        expert_loss = _bce_with_logits(expert_logits, jnp.ones_like(expert_logits))

        # Policy samples: want D(s,a) to be low (label = 0)
        policy_logits = self.forward(policy_states, policy_actions)
        policy_loss = _bce_with_logits(policy_logits, jnp.zeros_like(policy_logits))

        total_loss = expert_loss + policy_loss

        # Compute gradients manually (tabular, so gradient is simple)
        # For BCE: dL/d(logit) = sigmoid(logit) - label
        expert_grad = jax.nn.sigmoid(expert_logits) - 1.0
        policy_grad = jax.nn.sigmoid(policy_logits) - 0.0

        # Apply gradients functionally (JAX arrays are immutable)
        new_logits = jnp.array(self.logits)
        # Build gradient update via numpy for the scatter loop
        grad_update = np.zeros((self.n_states, self.n_actions))
        es_np = np.asarray(expert_states)
        ea_np = np.asarray(expert_actions)
        eg_np = np.asarray(expert_grad)
        for i in range(len(es_np)):
            grad_update[es_np[i], ea_np[i]] += eg_np[i]
        ps_np = np.asarray(policy_states)
        pa_np = np.asarray(policy_actions)
        pg_np = np.asarray(policy_grad)
        for i in range(len(ps_np)):
            grad_update[ps_np[i], pa_np[i]] += pg_np[i]

        self.logits = new_logits - learning_rate * jnp.array(grad_update)

        return float(total_loss)

    def get_reward_matrix(
        self, reward_type: Literal["gail", "airl"] = "gail"
    ) -> jnp.ndarray:
        """Convert discriminator to reward matrix.

        Args:
            reward_type: How to derive reward from discriminator
                - "gail": R = -log(1 - D) = log(1 + exp(logit))
                - "airl": R = logit (the discriminator output directly)

        Returns:
            Reward matrix R(s,a), shape (n_states, n_actions)
        """
        if reward_type == "gail":
            return jax.nn.softplus(self.logits)
        elif reward_type == "airl":
            return jnp.array(self.logits)
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
        feature_matrix: jnp.ndarray,
        init: Literal["zeros", "uniform", "ones"] = "zeros",
        key: jax.Array | None = None,
    ):
        """Initialize linear discriminator.

        Args:
            feature_matrix: State-action features, shape (n_states, n_actions, n_features)
            init: Initialization for weights
            key: PRNG key (required for "uniform" init)
        """
        if feature_matrix.ndim != 3:
            raise ValueError(
                f"feature_matrix must be 3D (n_states, n_actions, n_features), "
                f"got shape {feature_matrix.shape}"
            )

        self.feature_matrix = feature_matrix
        self.n_states, self.n_actions, self.n_features = feature_matrix.shape

        if init == "zeros":
            self.weights = jnp.zeros(self.n_features)
        elif init == "uniform":
            if key is None:
                key = jax.random.key(0)
            self.weights = jax.random.uniform(key, (self.n_features,)) * 2 - 1
        elif init == "ones":
            self.weights = jnp.ones(self.n_features)
        else:
            raise ValueError(f"Unknown init method: {init}")

    def forward(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Compute discriminator logits as linear combination of features.

        D(s,a) = w^T * phi(s,a)

        Args:
            states: State indices, shape (batch_size,)
            actions: Action indices, shape (batch_size,)

        Returns:
            Discriminator logits, shape (batch_size,)
        """
        features = self.feature_matrix[states, actions, :]
        return features @ self.weights

    def probability(self, states: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Compute probability that (s,a) is from expert."""
        logits = self.forward(states, actions)
        return jax.nn.sigmoid(logits)

    def update(
        self,
        expert_states: jnp.ndarray,
        expert_actions: jnp.ndarray,
        policy_states: jnp.ndarray,
        policy_actions: jnp.ndarray,
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
        expert_probs = jax.nn.sigmoid(expert_logits)
        expert_features = self.feature_matrix[expert_states, expert_actions, :]

        # Policy: want low probability
        policy_logits = self.forward(policy_states, policy_actions)
        policy_probs = jax.nn.sigmoid(policy_logits)
        policy_features = self.feature_matrix[policy_states, policy_actions, :]

        # Gradient: dL/dw = mean((sigmoid(logit) - label) * features)
        expert_grad = ((expert_probs - 1.0)[:, None] * expert_features).mean(axis=0)
        policy_grad = ((policy_probs - 0.0)[:, None] * policy_features).mean(axis=0)

        self.weights = self.weights - learning_rate * (expert_grad + policy_grad)

        # Compute loss for monitoring
        expert_loss = _bce_with_logits(expert_logits, jnp.ones_like(expert_logits))
        policy_loss = _bce_with_logits(policy_logits, jnp.zeros_like(policy_logits))

        return float(expert_loss + policy_loss)

    def get_reward_matrix(
        self, reward_type: Literal["gail", "airl"] = "gail"
    ) -> jnp.ndarray:
        """Convert discriminator to full reward matrix.

        Args:
            reward_type: How to derive reward

        Returns:
            Reward matrix R(s,a), shape (n_states, n_actions)
        """
        logits = jnp.einsum("sak,k->sa", self.feature_matrix, self.weights)

        if reward_type == "gail":
            return jax.nn.softplus(logits)
        elif reward_type == "airl":
            return logits
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")
