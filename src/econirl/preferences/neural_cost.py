"""Neural network cost function for Guided Cost Learning.

This module implements a neural network cost function c(s,a) that maps
discrete state-action pairs to costs via learned embeddings and an MLP.

Used in Guided Cost Learning (Finn et al. 2016) where the cost function
is parameterized as a neural network instead of linear features.

Reference:
    Finn, C., Levine, S., & Abbeel, P. (2016). Guided Cost Learning:
    Deep Inverse Optimal Control via Policy Optimization. ICML.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from econirl.preferences.base import BaseUtilityFunction


class NeuralCostFunction(BaseUtilityFunction):
    """Neural network cost function for discrete state-action spaces.

    Implements c(s, a) = MLP(Embed(s) || Embed(a)) where:
    - Embed(s) is a learned state embedding
    - Embed(a) is a learned action embedding
    - || denotes concatenation
    - MLP is a multi-layer perceptron

    This is used in Guided Cost Learning (GCL) where the cost function
    is learned from demonstrations via importance sampling.

    Parameters
    ----------
    n_states : int
        Number of discrete states.
    n_actions : int
        Number of discrete actions.
    embed_dim : int, default=32
        Dimension of state and action embeddings.
    hidden_dims : list[int], default=[64, 64]
        Hidden layer dimensions for the MLP.
    activation : str, default="relu"
        Activation function: "relu", "tanh", or "leaky_relu".
    key : jax.Array
        PRNG key for weight initialization.

    Examples
    --------
    >>> key = jax.random.key(0)
    >>> cost_fn = NeuralCostFunction(n_states=100, n_actions=4, key=key)
    >>> states = jnp.array([0, 5, 10])
    >>> actions = jnp.array([1, 2, 0])
    >>> costs = cost_fn.forward(states, actions)  # shape: (3,)
    >>> cost_matrix = cost_fn.compute()   # shape: (100, 4)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        embed_dim: int = 32,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        *,
        key: jax.Array | None = None,
    ):
        if hidden_dims is None:
            hidden_dims = [64, 64]

        param_names = ["neural_cost_params"]
        super().__init__(
            num_states=n_states,
            num_actions=n_actions,
            parameter_names=param_names,
        )

        self._embed_dim = embed_dim
        self._hidden_dims = hidden_dims

        if key is None:
            key = jax.random.key(0)

        k1, k2, k3 = jax.random.split(key, 3)

        # Embedding matrices (simple lookup tables)
        self.state_embedding = jax.random.normal(k1, (n_states, embed_dim)) * 0.1
        self.action_embedding = jax.random.normal(k2, (n_actions, embed_dim)) * 0.1

        # Build MLP using Equinox
        input_dim = 2 * embed_dim

        if activation == "relu":
            act_fn = jax.nn.relu
        elif activation == "tanh":
            act_fn = jnp.tanh
        elif activation == "leaky_relu":
            act_fn = jax.nn.leaky_relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.mlp = eqx.nn.MLP(
            in_size=input_dim,
            out_size=1,
            width_size=hidden_dims[0] if hidden_dims else 64,
            depth=len(hidden_dims),
            activation=act_fn,
            key=k3,
        )

    @property
    def embed_dim(self) -> int:
        """Embedding dimension."""
        return self._embed_dim

    @property
    def hidden_dims(self) -> list[int]:
        """Hidden layer dimensions."""
        return self._hidden_dims.copy()

    def forward(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute costs for batched state-action pairs.

        Parameters
        ----------
        states : jnp.ndarray
            State indices, shape (batch_size,).
        actions : jnp.ndarray
            Action indices, shape (batch_size,).

        Returns
        -------
        costs : jnp.ndarray
            Cost values, shape (batch_size,).
        """
        # Get embeddings via lookup
        state_embed = self.state_embedding[states]  # (batch, embed_dim)
        action_embed = self.action_embedding[actions]  # (batch, embed_dim)

        # Concatenate and pass through MLP
        combined = jnp.concatenate([state_embed, action_embed], axis=-1)
        costs = jax.vmap(self.mlp)(combined).squeeze(-1)  # (batch,)

        return costs

    def compute(self, parameters: jnp.ndarray | None = None) -> jnp.ndarray:
        """Compute the full cost matrix c(s, a) for all state-action pairs.

        For GCL, the cost function is parameterized by neural network weights,
        not an explicit parameter vector. The `parameters` argument is ignored.

        Parameters
        ----------
        parameters : jnp.ndarray, optional
            Ignored. Kept for interface compatibility.

        Returns
        -------
        cost_matrix : jnp.ndarray
            Cost matrix of shape (n_states, n_actions).
        """
        n_states = self.num_states
        n_actions = self.num_actions

        # Create all state-action pairs
        states = jnp.arange(n_states)
        actions = jnp.arange(n_actions)

        # Create meshgrid of all (state, action) combinations
        states_grid = jnp.repeat(states, n_actions)
        actions_grid = jnp.tile(actions, n_states)

        # Compute costs
        costs_flat = self.forward(states_grid, actions_grid)

        # Reshape to matrix
        cost_matrix = costs_flat.reshape(n_states, n_actions)

        return cost_matrix

    def compute_gradient(self, parameters: jnp.ndarray | None = None) -> jnp.ndarray:
        """Compute gradient of cost w.r.t. parameters.

        For neural network cost functions, gradients are computed via
        JAX autodiff, not analytically. This method returns zeros.

        Parameters
        ----------
        parameters : jnp.ndarray, optional
            Ignored.

        Returns
        -------
        gradient : jnp.ndarray
            Zero array of shape (n_states, n_actions, 1).
        """
        return jnp.zeros(
            (self.num_states, self.num_actions, 1),
            dtype=jnp.float32,
        )

    def get_reward_matrix(self) -> jnp.ndarray:
        """Get the reward matrix R(s, a) = -c(s, a).

        In IRL, we typically work with rewards (higher is better),
        while GCL learns costs (lower is better). This method returns
        the negated cost matrix.

        Returns
        -------
        reward_matrix : jnp.ndarray
            Reward matrix of shape (n_states, n_actions).
        """
        return -self.compute()

    def __repr__(self) -> str:
        return (
            f"NeuralCostFunction("
            f"n_states={self.num_states}, "
            f"n_actions={self.num_actions}, "
            f"embed_dim={self._embed_dim}, "
            f"hidden_dims={self._hidden_dims})"
        )
