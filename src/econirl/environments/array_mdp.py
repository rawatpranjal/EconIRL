"""Custom dynamic discrete choice environment from explicit arrays.

``ArrayMDP`` is the injection primitive of the synthetic generator suite. It
turns a user-supplied transition tensor, feature tensor, and linear reward
parameter vector into a full :class:`DDCEnvironment`, so any hand-built or
programmatically generated MDP can be simulated and estimated through the
existing ``simulate_panel`` harness without writing a new subclass.

Shapeshifter generates parameterized-random MDPs and cannot take an injected
DGP. ``ArrayMDP`` fills that gap: you bring the arrays, it builds the
environment.

Conventions follow the rest of the package:

- ``transitions`` has shape ``(num_actions, num_states, num_states)`` with
  ``transitions[a, s, s'] = P(s' | s, a)`` and rows summing to one.
- ``features`` has shape ``(num_states, num_actions, num_features)``.
- the flow reward is linear, ``R(s, a) = theta . phi(s, a)``.

Reward is linear by design. Nonlinear-reward regimes are served by
``ShapeshifterEnvironment(reward_type="neural")``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


class ArrayMDP(DDCEnvironment):
    """A DDC environment defined by explicit transition, feature, and reward arrays.

    Args:
        transitions: ``(A, S, S)`` transition probabilities ``P(s'|s,a)``.
        features: ``(S, A, K)`` feature tensor ``phi(s, a)``.
        theta: Linear reward parameters. Either a length-``K`` array, or a
            mapping ``{name: value}`` of length ``K`` (its keys become the
            parameter names, preserving insertion order).
        discount_factor: Time discount ``beta`` in ``[0, 1)``.
        scale_parameter: Logit scale ``sigma > 0``.
        parameter_names: Optional names for the ``K`` parameters. Ignored when
            ``theta`` is a mapping (the mapping keys are used instead). Defaults
            to ``["theta_0", ..., "theta_{K-1}"]``.
        initial_distribution: Optional ``(S,)`` initial-state distribution.
            Defaults to uniform.
        seed: Random seed for the environment RNG used by ``reset``/``step``.

    Example:
        >>> import numpy as np
        >>> S, A, K = 5, 2, 2
        >>> T = np.zeros((A, S, S)); T[:, np.arange(S), np.arange(S)] = 1.0
        >>> phi = np.random.default_rng(0).normal(size=(S, A, K))
        >>> env = ArrayMDP(T, phi, theta={"cost": -1.0, "value": 0.5})
        >>> from econirl.simulation.synthetic import simulate_panel
        >>> panel = simulate_panel(env, n_individuals=10, n_periods=20, seed=1)
    """

    def __init__(
        self,
        transitions: np.ndarray | jnp.ndarray,
        features: np.ndarray | jnp.ndarray,
        theta: Sequence[float] | np.ndarray | jnp.ndarray | Mapping[str, float],
        discount_factor: float = 0.95,
        scale_parameter: float = 1.0,
        parameter_names: Sequence[str] | None = None,
        initial_distribution: np.ndarray | jnp.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        transitions = np.asarray(transitions, dtype=np.float64)
        features = np.asarray(features, dtype=np.float64)

        # Resolve theta and its names.
        if isinstance(theta, Mapping):
            names = list(theta.keys())
            theta_vec = np.asarray([float(theta[n]) for n in names], dtype=np.float64)
        else:
            theta_vec = np.asarray(theta, dtype=np.float64).reshape(-1)
            if parameter_names is not None:
                names = list(parameter_names)
            else:
                names = [f"theta_{i}" for i in range(theta_vec.shape[0])]

        self._validate(transitions, features, theta_vec, names, initial_distribution)

        self._A, self._S = transitions.shape[0], transitions.shape[1]
        self._K = features.shape[2]

        self._transition_matrices_arr = jnp.asarray(transitions, dtype=jnp.float32)
        self._feature_matrix_arr = jnp.asarray(features, dtype=jnp.float32)
        self._theta = jnp.asarray(theta_vec, dtype=jnp.float32)
        self._parameter_names = names
        self._true_parameters = {n: float(v) for n, v in zip(names, theta_vec)}

        if initial_distribution is None:
            self._initial_dist = np.ones(self._S, dtype=np.float64) / self._S
        else:
            init = np.asarray(initial_distribution, dtype=np.float64)
            self._initial_dist = init / init.sum()

        self.observation_space = spaces.Discrete(self._S)
        self.action_space = spaces.Discrete(self._A)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(
        transitions: np.ndarray,
        features: np.ndarray,
        theta_vec: np.ndarray,
        names: Sequence[str],
        initial_distribution: np.ndarray | jnp.ndarray | None,
    ) -> None:
        if transitions.ndim != 3:
            raise ValueError(
                f"transitions must be 3D (A, S, S); got shape {transitions.shape}."
            )
        A, S, S2 = transitions.shape
        if S != S2:
            raise ValueError(
                f"transitions must be square in the state axes; got {transitions.shape}."
            )
        row_sums = transitions.sum(axis=2)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            worst = float(np.abs(row_sums - 1.0).max())
            raise ValueError(
                "transition rows must sum to 1 (max abs deviation "
                f"{worst:.2e}). Check that P(s'|s,a) is a valid distribution."
            )
        if (transitions < -1e-8).any():
            raise ValueError("transitions must be non-negative.")

        if features.ndim != 3:
            raise ValueError(
                f"features must be 3D (S, A, K); got shape {features.shape}."
            )
        if features.shape[0] != S or features.shape[1] != A:
            raise ValueError(
                f"features shape {features.shape} is inconsistent with "
                f"transitions (A={A}, S={S}); expected (S, A, K) = ({S}, {A}, K)."
            )
        K = features.shape[2]
        if theta_vec.shape[0] != K:
            raise ValueError(
                f"theta has length {theta_vec.shape[0]} but features have "
                f"K={K} parameters; they must match."
            )
        if len(names) != K:
            raise ValueError(
                f"parameter_names has length {len(names)} but K={K}."
            )
        if initial_distribution is not None:
            init = np.asarray(initial_distribution, dtype=np.float64)
            if init.shape != (S,):
                raise ValueError(
                    f"initial_distribution must have shape ({S},); got {init.shape}."
                )
            if (init < -1e-8).any() or init.sum() <= 0:
                raise ValueError("initial_distribution must be non-negative and sum > 0.")

    # ------------------------------------------------------------------
    # Required DDCEnvironment properties
    # ------------------------------------------------------------------

    @property
    def num_states(self) -> int:
        return self._S

    @property
    def num_actions(self) -> int:
        return self._A

    @property
    def num_features(self) -> int:
        return self._K

    @property
    def transition_matrices(self) -> jnp.ndarray:
        return self._transition_matrices_arr

    @property
    def feature_matrix(self) -> jnp.ndarray:
        return self._feature_matrix_arr

    @property
    def true_parameters(self) -> dict[str, float]:
        return dict(self._true_parameters)

    @property
    def parameter_names(self) -> list[str]:
        return list(self._parameter_names)

    @property
    def true_reward_matrix(self) -> jnp.ndarray:
        """Ground-truth flow reward ``R(s, a) = theta . phi(s, a)``, shape (S, A)."""
        return jnp.einsum("sak,k->sa", self._feature_matrix_arr, self._theta)

    # ------------------------------------------------------------------
    # Required DDCEnvironment methods
    # ------------------------------------------------------------------

    def _get_initial_state_distribution(self) -> np.ndarray:
        return np.asarray(self._initial_dist)

    def _compute_flow_utility(self, state: int, action: int) -> float:
        phi = self._feature_matrix_arr[int(state), int(action)]
        return float(jnp.dot(phi, self._theta))

    def _sample_next_state(self, state: int, action: int) -> int:
        probs = np.asarray(self._transition_matrices_arr[int(action), int(state)])
        total = probs.sum()
        if total <= 0:
            return int(self._np_random.integers(0, self._S))
        probs = probs / total
        return int(self._np_random.choice(self._S, p=probs))

    @classmethod
    def info(cls) -> dict[str, Any]:
        return {
            "name": cls.__name__,
            "description": "User-injected MDP from explicit (A,S,S), (S,A,K), theta arrays.",
        }
