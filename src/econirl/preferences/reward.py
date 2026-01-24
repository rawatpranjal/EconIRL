"""Linear reward specification for Inverse Reinforcement Learning.

This module implements a linear reward function for IRL:

    R(s; θ) = θ · φ(s)

Unlike utility functions U(s, a; θ) that vary by action, reward functions
in IRL are state-only. The reward is broadcast to all actions since the
agent receives the same reward for being in a state regardless of action.

This specification is used in:
- Maximum Entropy IRL (Ziebart et al., 2008)
- Bayesian IRL (Ramachandran & Amir, 2007)
- Feature-matching IRL approaches
"""

from __future__ import annotations

import torch

from econirl.preferences.base import BaseUtilityFunction


class LinearReward(BaseUtilityFunction):
    """Linear reward function for IRL: R(s) = theta * phi(s).

    Unlike utility U(s,a), reward is state-only but expanded
    to (n_states, n_actions) by broadcasting.

    In IRL, we observe expert demonstrations and want to recover a reward
    function that explains the behavior. The linear specification assumes
    reward is a linear combination of state features:

        R(s; θ) = Σ_k θ_k φ_k(s)

    This reward is the same for all actions in a given state, reflecting
    the intuition that reward depends on where you are, not what you do.

    Attributes:
        state_features: Tensor of shape (num_states, num_features)

    Example:
        >>> # Create state features for a grid world
        >>> features = torch.randn(100, 5)  # 100 states, 5 features
        >>> reward = LinearReward(
        ...     state_features=features,
        ...     parameter_names=["goal", "obstacle", "distance", "safety", "comfort"],
        ...     n_actions=4,  # up, down, left, right
        ... )
        >>>
        >>> # Compute reward
        >>> theta = torch.tensor([1.0, -2.0, -0.1, 0.5, 0.3])
        >>> R = reward.compute(theta)  # shape (100, 4)
        >>> # R[s, a] is the same for all actions a in state s
    """

    def __init__(
        self,
        state_features: torch.Tensor,
        parameter_names: list[str],
        n_actions: int = 2,
    ):
        """Initialize the linear reward function.

        Args:
            state_features: Feature tensor of shape (num_states, num_features)
                           where state_features[s, k] = φ_k(s)
            parameter_names: Names for each parameter (must match num_features)
            n_actions: Number of actions. Reward is broadcast to all actions.
        """
        if state_features.ndim != 2:
            raise ValueError(
                f"state_features must be 2D (states, features), "
                f"got shape {state_features.shape}"
            )

        num_states, num_features = state_features.shape

        if len(parameter_names) != num_features:
            raise ValueError(
                f"parameter_names must have {num_features} elements, "
                f"got {len(parameter_names)}"
            )

        if n_actions < 1:
            raise ValueError(f"n_actions must be at least 1, got {n_actions}")

        super().__init__(
            num_states=num_states,
            num_actions=n_actions,
            parameter_names=parameter_names,
            anchor_action=None,  # No anchor for reward functions
        )

        self._state_features = state_features.clone()

    @property
    def state_features(self) -> torch.Tensor:
        """Return the state feature matrix."""
        return self._state_features

    def compute(self, params: torch.Tensor) -> torch.Tensor:
        """Compute reward matrix R(s; θ) = θ · φ(s), broadcast to all actions.

        Args:
            params: Parameter vector θ of shape (num_parameters,)

        Returns:
            Reward matrix of shape (num_states, num_actions)
            where result[s, a] = R(s; θ) for all actions a
        """
        self.validate_parameters(params)

        # R[s] = Σ_k θ[k] * φ[s, k]
        reward_per_state = torch.einsum("sk,k->s", self._state_features, params)

        # Broadcast to all actions: (num_states,) -> (num_states, num_actions)
        return reward_per_state.unsqueeze(1).expand(-1, self.num_actions)

    def compute_gradient(self, params: torch.Tensor) -> torch.Tensor:
        """Compute gradient ∂R/∂θ = φ(s), broadcast to all actions.

        For linear reward, the gradient is simply the state feature matrix,
        independent of the parameter values.

        Args:
            params: Parameter vector (unused, but kept for interface consistency)

        Returns:
            Gradient tensor of shape (num_states, num_actions, num_parameters)
            where result[s, a, k] = ∂R(s;θ)/∂θ_k = φ_k(s) for all actions a
        """
        # For linear reward, gradient w.r.t. params is the feature matrix
        # Shape: (num_states, num_features)
        # Need to broadcast to (num_states, num_actions, num_features)
        gradient = self._state_features.unsqueeze(1).expand(
            -1, self.num_actions, -1
        )
        return gradient.clone()

    def compute_hessian(self, params: torch.Tensor) -> torch.Tensor:
        """Compute Hessian ∂²R/∂θ².

        For linear reward, the Hessian is zero (reward is linear in θ).

        Args:
            params: Parameter vector (unused)

        Returns:
            Zero tensor of shape (num_states, num_actions, num_parameters, num_parameters)
        """
        return torch.zeros(
            (self.num_states, self.num_actions, self.num_parameters, self.num_parameters),
            dtype=self._state_features.dtype,
            device=self._state_features.device,
        )

    def to(self, device: torch.device | str) -> "LinearReward":
        """Move state features to specified device."""
        return LinearReward(
            state_features=self._state_features.to(device),
            parameter_names=self._parameter_names.copy(),
            n_actions=self.num_actions,
        )

    def subset_states(self, state_indices: torch.Tensor) -> "LinearReward":
        """Create a new reward function for a subset of states.

        Useful for state aggregation or focusing on specific regions.

        Args:
            state_indices: Indices of states to keep

        Returns:
            New LinearReward with only selected states
        """
        new_features = self._state_features[state_indices, :]
        return LinearReward(
            state_features=new_features,
            parameter_names=self._parameter_names.copy(),
            n_actions=self.num_actions,
        )

    def __repr__(self) -> str:
        return (
            f"LinearReward("
            f"num_states={self.num_states}, "
            f"num_actions={self.num_actions}, "
            f"num_features={self.num_parameters}, "
            f"parameters={self.parameter_names})"
        )
