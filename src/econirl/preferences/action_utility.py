"""Action-dependent utility specification for structural DDC models.

This module implements the ActionDependentUtility class for dynamic discrete
choice models where utility depends on both state and action in an action-specific
way. This is the standard specification in structural econometrics.

The canonical example is Rust (1987) bus engine replacement:
    U(s, keep) = -theta_c * s / scale + epsilon_keep
    U(s, replace) = -RC + epsilon_replace

where:
- s is the state (mileage bin)
- theta_c is the operating cost parameter
- RC is the replacement cost
- scale normalizes the state variable
- epsilon are i.i.d. Type I Extreme Value shocks

This specification is used in:
- Rust (1987) bus engine replacement
- Keane-Wolpin (1994) occupational choice
- Most structural DDC models in economics

Reference:
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical
    Model of Harold Zurcher." Econometrica, 55(5), 999-1033.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from econirl.preferences.base import BaseUtilityFunction


class ActionDependentUtility(BaseUtilityFunction):
    """Action-dependent utility function for structural DDC models.

    Implements utility specifications where each action has its own utility
    structure. This is the standard specification in structural econometrics.

    For the Rust (1987) model:
        U(s, 0) = -theta_c * s / scale  (keep)
        U(s, 1) = -RC                    (replace)

    where:
    - theta_c is the operating cost parameter
    - RC is the replacement cost
    - scale normalizes the state variable (typically num_states)

    The class supports:
    - Computing utility matrices for given parameters
    - Computing gradients for optimization
    - Providing initial parameter values and bounds

    Attributes:
        state_scale: Scale factor for normalizing state variable

    Example:
        >>> # Create for Rust (1987) model
        >>> utility = ActionDependentUtility.for_rust_model(
        ...     num_states=90,
        ...     state_scale=90.0,
        ... )
        >>> params = jnp.array([0.001, 3.0])  # [theta_c, RC]
        >>> U = utility.compute(params)  # shape (90, 2)
        >>> print(f"U[45, keep]: {U[45, 0]:.4f}")  # Should be negative
        >>> print(f"U[45, replace]: {U[45, 1]:.4f}")  # Should be -3.0
    """

    # Action constants for Rust model
    KEEP = 0
    REPLACE = 1

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        parameter_names: list[str],
        state_scale: float = 1.0,
        utility_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
        gradient_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
    ):
        """Initialize the action-dependent utility function.

        Args:
            num_states: Number of states in the model
            num_actions: Number of actions in the model
            parameter_names: Names for each parameter (e.g., ["theta_c", "RC"])
            state_scale: Scale factor for normalizing state variable (default: 1.0)
            utility_fn: Optional custom utility function. If None, uses Rust model.
                       Signature: (params: ndarray, states: ndarray) -> ndarray[num_states, num_actions]
            gradient_fn: Optional custom gradient function. If None, computes analytically.
                        Signature: (params: ndarray, states: ndarray) -> ndarray[num_states, num_actions, num_params]
        """
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            parameter_names=parameter_names,
            anchor_action=None,
        )

        self._state_scale = state_scale
        self._utility_fn = utility_fn
        self._gradient_fn = gradient_fn

        # Pre-compute state indices tensor
        self._state_indices = jnp.arange(num_states, dtype=jnp.float32)

    @property
    def state_scale(self) -> float:
        """Scale factor for state normalization."""
        return self._state_scale

    def compute(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """Compute utility matrix U(s, a; params) for all state-action pairs.

        For the default Rust model:
            U[:, 0] = -theta_c * state_indices / scale  (keep)
            U[:, 1] = -RC                                (replace)

        Args:
            parameters: Parameter vector of shape (num_parameters,)
                       For Rust model: [theta_c, RC]

        Returns:
            Utility matrix of shape (num_states, num_actions)
        """
        self.validate_parameters(parameters)

        if self._utility_fn is not None:
            return self._utility_fn(parameters, self._state_indices)

        # Default: Rust (1987) model
        return self._compute_rust_utility(parameters)

    def _compute_rust_utility(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """Compute Rust (1987) utility structure.

        U(s, keep) = -theta_c * s / scale
        U(s, replace) = -RC

        Args:
            parameters: [theta_c, RC]

        Returns:
            Utility matrix of shape (num_states, num_actions)
        """
        theta_c = parameters[0]
        rc = parameters[1]

        # Initialize utility matrix
        U = jnp.zeros((self.num_states, self.num_actions), dtype=parameters.dtype)

        # Keep action: U = -theta_c * s / scale
        U = U.at[:, self.KEEP].set(-theta_c * self._state_indices / self._state_scale)

        # Replace action: U = -RC
        U = U.at[:, self.REPLACE].set(-rc)

        return U

    def compute_gradient(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """Compute gradient of utility with respect to parameters.

        For the Rust model:
            dU/d(theta_c)[s, keep] = -s / scale
            dU/d(theta_c)[s, replace] = 0
            dU/d(RC)[s, keep] = 0
            dU/d(RC)[s, replace] = -1

        Args:
            parameters: Parameter vector of shape (num_parameters,)

        Returns:
            Gradient tensor of shape (num_states, num_actions, num_parameters)
            where result[s, a, k] = dU(s, a; params) / d(params[k])
        """
        self.validate_parameters(parameters)

        if self._gradient_fn is not None:
            return self._gradient_fn(parameters, self._state_indices)

        # Default: Rust (1987) model gradient
        return self._compute_rust_gradient(parameters)

    def _compute_rust_gradient(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """Compute gradient for Rust (1987) utility.

        dU/d(theta_c) = [-s/scale, 0]  for [keep, replace]
        dU/d(RC) = [0, -1]             for [keep, replace]

        Args:
            parameters: Parameter vector (not used but kept for interface)

        Returns:
            Gradient tensor of shape (num_states, num_actions, num_parameters)
        """
        grad = jnp.zeros(
            (self.num_states, self.num_actions, self.num_parameters),
            dtype=parameters.dtype,
        )

        # dU/d(theta_c): only affects keep action
        grad = grad.at[:, self.KEEP, 0].set(-self._state_indices / self._state_scale)

        # dU/d(RC): only affects replace action
        grad = grad.at[:, self.REPLACE, 1].set(-1.0)

        return grad

    def compute_hessian(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """Compute Hessian of utility with respect to parameters.

        For linear utility structures like Rust (1987), the Hessian is zero
        since utility is linear in parameters.

        Args:
            parameters: Parameter vector (unused)

        Returns:
            Zero tensor of shape (num_states, num_actions, num_parameters, num_parameters)
        """
        return jnp.zeros(
            (self.num_states, self.num_actions, self.num_parameters, self.num_parameters),
            dtype=parameters.dtype,
        )

    def get_initial_parameters(self) -> jnp.ndarray:
        """Return reasonable initial parameter values for optimization.

        For the Rust model, returns conservative starting values:
        - theta_c: small positive value (0.01)
        - RC: moderate positive value (1.0)

        Returns:
            Initial parameter tensor of shape (num_parameters,)
        """
        if self.num_parameters == 2:
            # Rust model defaults
            return jnp.array([0.01, 1.0], dtype=jnp.float32)
        else:
            # Generic: start near zero
            return jnp.zeros(self.num_parameters, dtype=jnp.float32)

    def get_parameter_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return lower and upper bounds for parameters.

        For the Rust model:
        - theta_c: [0, inf) - operating cost must be non-negative
        - RC: [0, inf) - replacement cost must be non-negative

        Returns:
            Tuple of (lower_bounds, upper_bounds), each of shape (num_parameters,)
        """
        if self.num_parameters == 2:
            # Rust model: both parameters must be non-negative
            lower = jnp.array([0.0, 0.0], dtype=jnp.float32)
            upper = jnp.array([float("inf"), float("inf")], dtype=jnp.float32)
            return lower, upper
        else:
            # Generic: no bounds
            return super().get_parameter_bounds()

    @classmethod
    def for_rust_model(
        cls,
        num_states: int,
        state_scale: float | None = None,
    ) -> "ActionDependentUtility":
        """Create ActionDependentUtility configured for Rust (1987) model.

        Convenience constructor that sets up the standard Rust bus engine
        replacement model.

        Args:
            num_states: Number of mileage bins
            state_scale: Scale factor for state normalization.
                        If None, defaults to num_states.

        Returns:
            ActionDependentUtility configured for Rust model
        """
        if state_scale is None:
            state_scale = float(num_states)

        return cls(
            num_states=num_states,
            num_actions=2,
            parameter_names=["theta_c", "RC"],
            state_scale=state_scale,
        )

    @classmethod
    def from_environment(cls, env, state_scale: float = 1.0) -> "ActionDependentUtility":
        """Create ActionDependentUtility from a DDCEnvironment.

        Convenience constructor that extracts configuration from an environment.

        Args:
            env: A DDCEnvironment instance (e.g., RustBusEnvironment)
            state_scale: Scale factor for state normalization.
                        Default is 1.0 to match environment's convention.

        Returns:
            ActionDependentUtility configured for this environment
        """
        return cls(
            num_states=env.num_states,
            num_actions=env.num_actions,
            parameter_names=env.parameter_names,
            state_scale=state_scale,
        )

    def __repr__(self) -> str:
        return (
            f"ActionDependentUtility("
            f"num_states={self.num_states}, "
            f"num_actions={self.num_actions}, "
            f"parameters={self.parameter_names}, "
            f"state_scale={self._state_scale})"
        )
