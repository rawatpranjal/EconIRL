"""Action-dependent reward function for IRL.

This extends the standard IRL reward to handle action-dependent features,
as required for structural models like Rust (1987) bus engine replacement.

Unlike LinearReward (state-only features), ActionDependentReward supports
different features per action, enabling Rust-style utility:
    U(s, keep) = -theta_c * s (operating cost grows with mileage)
    U(s, replace) = -RC (fixed replacement cost)

This specification bridges the gap between:
- Traditional IRL: R(s) state-only rewards
- Structural econometrics: U(s,a) action-dependent utilities
"""
from __future__ import annotations

import jax.numpy as jnp

from econirl.preferences.base import BaseUtilityFunction


class ActionDependentReward(BaseUtilityFunction):
    """Action-dependent reward for IRL: R(s,a) = theta * phi(s,a).

    Unlike LinearReward (state-only), this supports different features
    per action, enabling Rust-style utility:
        U(s, keep) = -theta_c * s
        U(s, replace) = -RC

    Parameters
    ----------
    feature_matrix : jnp.ndarray
        Shape (num_states, num_actions, num_features).
    parameter_names : list[str]
        Names for each parameter.

    Example
    -------
    >>> # Create Rust-style features
    >>> n_states = 90
    >>> features = jnp.zeros((n_states, 2, 2))
    >>> for s in range(n_states):
    ...     features[s, 0, 0] = -s  # Keep: operating cost
    ...     features[s, 1, 1] = -1  # Replace: fixed cost
    >>>
    >>> reward = ActionDependentReward(features, ["theta_c", "RC"])
    >>> params = jnp.array([0.001, 3.0])
    >>> R = reward.compute(params)  # shape (90, 2)
    """

    def __init__(
        self,
        feature_matrix: jnp.ndarray,
        parameter_names: list[str],
    ):
        if feature_matrix.ndim != 3:
            raise ValueError(
                f"feature_matrix must be 3D (states, actions, features), "
                f"got shape {feature_matrix.shape}"
            )

        num_states, num_actions, num_features = feature_matrix.shape

        if len(parameter_names) != num_features:
            raise ValueError(
                f"parameter_names length {len(parameter_names)} != "
                f"num_features {num_features}"
            )

        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            parameter_names=parameter_names,
            anchor_action=None,
        )

        self._feature_matrix = jnp.array(feature_matrix)

    @property
    def feature_matrix(self) -> jnp.ndarray:
        """Return the feature matrix."""
        return self._feature_matrix

    def compute(self, params: jnp.ndarray) -> jnp.ndarray:
        """Compute reward matrix R(s,a).

        Args:
            params: Parameter vector of shape (num_features,).

        Returns:
            Reward matrix of shape (num_states, num_actions).
        """
        self.validate_parameters(params)
        # R[s, a] = sum_k params[k] * features[s, a, k]
        return jnp.einsum("sak,k->sa", self._feature_matrix, params)

    def compute_gradient(self, params: jnp.ndarray) -> jnp.ndarray:
        """Compute gradient dR/dtheta.

        For linear reward R = theta * phi, the gradient is the feature matrix.

        Args:
            params: Parameter vector (unused, kept for interface consistency).

        Returns:
            Gradient tensor of shape (num_states, num_actions, num_features).
        """
        # Gradient is just the feature matrix
        return jnp.array(self._feature_matrix)

    def compute_hessian(self, params: jnp.ndarray) -> jnp.ndarray:
        """Compute Hessian d^2R/dtheta^2.

        For linear reward, the Hessian is zero.

        Args:
            params: Parameter vector (unused).

        Returns:
            Zero tensor of shape (num_states, num_actions, num_params, num_params).
        """
        return jnp.zeros(
            (self.num_states, self.num_actions, self.num_parameters, self.num_parameters),
            dtype=self._feature_matrix.dtype,
        )

    def subset_states(self, state_indices: jnp.ndarray) -> "ActionDependentReward":
        """Create a new reward function for a subset of states.

        Args:
            state_indices: Indices of states to keep.

        Returns:
            New ActionDependentReward with only selected states.
        """
        new_features = self._feature_matrix[state_indices, :, :]
        return ActionDependentReward(
            feature_matrix=new_features,
            parameter_names=self._parameter_names.copy(),
        )

    @classmethod
    def from_rust_environment(cls, env) -> "ActionDependentReward":
        """Create from RustBusEnvironment.

        Args:
            env: RustBusEnvironment instance with feature_matrix and parameter_names.

        Returns:
            ActionDependentReward with correct feature structure.
        """
        return cls(
            feature_matrix=env.feature_matrix,
            parameter_names=env.parameter_names,
        )

    def __repr__(self) -> str:
        return (
            f"ActionDependentReward("
            f"num_states={self.num_states}, "
            f"num_actions={self.num_actions}, "
            f"num_features={self.num_parameters}, "
            f"parameters={self.parameter_names})"
        )
