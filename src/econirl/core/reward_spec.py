"""Unified feature specification for all estimators.

RewardSpec replaces the three separate classes (LinearUtility, LinearReward,
ActionDependentReward) with a single, clean interface. Internally it always
stores features as a (S, A, K) array and exposes the same compute/gradient/
hessian interface that estimators rely on.

Backward-compatibility adapters (.to_linear_utility(), etc.) allow gradual
migration without breaking existing code.

Usage:
    >>> features_sak = jnp.zeros((10, 2, 3))
    >>> spec = RewardSpec(features_sak, names=["cost", "benefit", "distance"])
    >>> R = spec.compute(jnp.array([1.0, -0.5, 0.3]))  # shape (10, 2)

    >>> features_sk = jnp.zeros((10, 3))
    >>> spec = RewardSpec(features_sk, names=["a", "b", "c"], n_actions=4)

    >>> spec = RewardSpec.state_dependent(features_sk, names=["a", "b", "c"], n_actions=4)
    >>> spec = RewardSpec.state_action_dependent(features_sak, names=["cost", "benefit", "distance"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    pass


class RewardSpec:
    """Unified feature specification for structural estimation and IRL.

    Stores features as a (S, A, K) array and provides compute, gradient,
    and hessian methods compatible with the BaseUtilityFunction protocol.

    Parameters
    ----------
    features : jnp.ndarray
        Either (S, A, K) for action-dependent features, or (S, K) for
        state-only features (broadcast to all actions).
    names : list[str]
        Human-readable name for each feature/parameter dimension.
    n_actions : int, optional
        Required when ``features`` is (S, K) to specify the number of
        actions for broadcasting. Ignored when ``features`` is (S, A, K).
    """

    def __init__(
        self,
        features: jnp.ndarray,
        names: list[str],
        n_actions: int | None = None,
    ):
        if features.ndim == 3:
            # (S, A, K) — action-dependent features
            S, A, K = features.shape
            if n_actions is not None and n_actions != A:
                raise ValueError(
                    f"features already have {A} actions on axis 1 "
                    f"but n_actions={n_actions} was also provided"
                )
            self._feature_matrix = jnp.array(features)
            self._is_state_only = False

        elif features.ndim == 2:
            # (S, K) — state-only features, broadcast to (S, A, K)
            S, K = features.shape
            if n_actions is None:
                raise ValueError(
                    "n_actions is required when features is 2D (S, K)"
                )
            if n_actions < 1:
                raise ValueError(f"n_actions must be >= 1, got {n_actions}")
            self._feature_matrix = jnp.broadcast_to(
                features[:, None, :], (S, n_actions, K)
            ).copy()
            self._is_state_only = True

        else:
            raise ValueError(
                f"features must be 2D (S, K) or 3D (S, A, K), "
                f"got {features.ndim}D with shape {features.shape}"
            )

        K = self._feature_matrix.shape[2]
        if len(names) != K:
            raise ValueError(
                f"names must have {K} elements to match feature dimension, "
                f"got {len(names)}"
            )

        self._parameter_names = list(names)

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def state_dependent(
        cls,
        state_features: jnp.ndarray,
        names: list[str],
        n_actions: int,
    ) -> RewardSpec:
        """Create from state-only features (S, K), broadcast to all actions.

        Parameters
        ----------
        state_features : jnp.ndarray
            Shape (S, K).
        names : list[str]
            One name per feature.
        n_actions : int
            Number of actions to broadcast to.
        """
        if state_features.ndim != 2:
            raise ValueError(
                f"state_features must be 2D (S, K), got shape {state_features.shape}"
            )
        return cls(features=state_features, names=names, n_actions=n_actions)

    @classmethod
    def state_action_dependent(
        cls,
        features: jnp.ndarray,
        names: list[str],
    ) -> RewardSpec:
        """Create from action-dependent features (S, A, K).

        Parameters
        ----------
        features : jnp.ndarray
            Shape (S, A, K).
        names : list[str]
            One name per feature.
        """
        if features.ndim != 3:
            raise ValueError(
                f"features must be 3D (S, A, K), got shape {features.shape}"
            )
        return cls(features=features, names=names)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_matrix(self) -> jnp.ndarray:
        """Feature array of shape (S, A, K)."""
        return self._feature_matrix

    @property
    def parameter_names(self) -> list[str]:
        """Human-readable names for each parameter."""
        return self._parameter_names.copy()

    @property
    def num_parameters(self) -> int:
        """Number of parameters (K)."""
        return self._feature_matrix.shape[2]

    @property
    def num_states(self) -> int:
        """Number of states (S)."""
        return self._feature_matrix.shape[0]

    @property
    def num_actions(self) -> int:
        """Number of actions (A)."""
        return self._feature_matrix.shape[1]

    @property
    def is_state_only(self) -> bool:
        """Whether the spec was constructed from state-only features."""
        return self._is_state_only

    # ------------------------------------------------------------------
    # Compute interface (matches BaseUtilityFunction protocol)
    # ------------------------------------------------------------------

    def compute(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """Compute reward matrix R(s, a) = sum_k params[k] * features[s, a, k].

        Parameters
        ----------
        parameters : jnp.ndarray
            Shape (K,).

        Returns
        -------
        jnp.ndarray
            Shape (S, A).
        """
        self.validate_parameters(parameters)
        return jnp.einsum("sak,k->sa", self._feature_matrix, parameters)

    def compute_gradient(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """Gradient of reward w.r.t. parameters.

        For linear specification the gradient is the feature matrix itself,
        independent of the parameter values.

        Parameters
        ----------
        parameters : jnp.ndarray
            Shape (K,). Unused but kept for protocol compatibility.

        Returns
        -------
        jnp.ndarray
            Shape (S, A, K).
        """
        return jnp.array(self._feature_matrix)

    def compute_hessian(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """Hessian of reward w.r.t. parameters.

        For linear specification the Hessian is identically zero.

        Parameters
        ----------
        parameters : jnp.ndarray
            Shape (K,). Unused.

        Returns
        -------
        jnp.ndarray
            Shape (S, A, K, K) of zeros.
        """
        K = self.num_parameters
        return jnp.zeros(
            (self.num_states, self.num_actions, K, K),
            dtype=self._feature_matrix.dtype,
        )

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def get_initial_parameters(self) -> jnp.ndarray:
        """Return zeros of shape (K,) as a starting point."""
        return jnp.zeros(self.num_parameters, dtype=jnp.float32)

    def get_parameter_bounds(
        self,
    ) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
        """Return (None, None) indicating unbounded parameters."""
        return (None, None)

    def validate_parameters(self, parameters: jnp.ndarray) -> None:
        """Check that parameters have shape (K,).

        Raises
        ------
        ValueError
            If shape does not match.
        """
        if parameters.shape != (self.num_parameters,):
            raise ValueError(
                f"Expected parameters of shape ({self.num_parameters},), "
                f"got {parameters.shape}"
            )

    # ------------------------------------------------------------------
    # Subset utilities
    # ------------------------------------------------------------------

    def subset_states(self, indices: jnp.ndarray) -> RewardSpec:
        """Return a new RewardSpec containing only the specified states.

        Parameters
        ----------
        indices : jnp.ndarray
            1-D integer array of state indices to keep.
        """
        new = RewardSpec.__new__(RewardSpec)
        new._feature_matrix = self._feature_matrix[indices, :, :]
        new._parameter_names = self._parameter_names.copy()
        new._is_state_only = self._is_state_only
        return new

    # ------------------------------------------------------------------
    # Backward-compatibility adapters
    # ------------------------------------------------------------------

    def to_linear_utility(self) -> "LinearUtility":
        """Convert to a LinearUtility with the same (S, A, K) feature matrix.

        Returns
        -------
        LinearUtility
            Equivalent LinearUtility instance.
        """
        from econirl.preferences.linear import LinearUtility

        return LinearUtility(
            feature_matrix=jnp.array(self._feature_matrix),
            parameter_names=self._parameter_names.copy(),
        )

    def to_action_dependent_reward(self) -> "ActionDependentReward":
        """Convert to an ActionDependentReward with the same (S, A, K) features.

        Returns
        -------
        ActionDependentReward
            Equivalent ActionDependentReward instance.
        """
        from econirl.preferences.action_reward import ActionDependentReward

        return ActionDependentReward(
            feature_matrix=jnp.array(self._feature_matrix),
            parameter_names=self._parameter_names.copy(),
        )

    def to_linear_reward(self) -> "LinearReward":
        """Convert to a LinearReward with state-only (S, K) features.

        This only works when features are truly state-only (identical
        across all actions). If features differ across actions, a
        ValueError is raised.

        Returns
        -------
        LinearReward
            Equivalent LinearReward instance.

        Raises
        ------
        ValueError
            If features vary across actions.
        """
        from econirl.preferences.reward import LinearReward

        # Check that all actions have identical features
        ref = self._feature_matrix[:, 0:1, :]  # (S, 1, K)
        if not jnp.allclose(
            self._feature_matrix, jnp.broadcast_to(ref, self._feature_matrix.shape)
        ):
            raise ValueError(
                "Cannot convert to LinearReward: features differ across "
                "actions. LinearReward requires state-only features."
            )

        state_features = self._feature_matrix[:, 0, :]  # (S, K)
        return LinearReward(
            state_features=jnp.array(state_features),
            parameter_names=self._parameter_names.copy(),
            n_actions=self.num_actions,
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        kind = "state_only" if self._is_state_only else "state_action"
        return (
            f"RewardSpec("
            f"num_states={self.num_states}, "
            f"num_actions={self.num_actions}, "
            f"num_parameters={self.num_parameters}, "
            f"kind={kind}, "
            f"parameters={self._parameter_names})"
        )
