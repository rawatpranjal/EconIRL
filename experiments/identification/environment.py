"""Serialized content consumption environment for identification experiments.

This environment models a reader consuming episodes of serialized fiction.
At each episode the reader can buy (advance to next episode), wait (stay
at current episode), or exit (leave permanently). The exit action has
zero reward and sends the reader to an absorbing state, providing the
anchor normalization from Lee, Sudhir & Wang (2026) Theorems 1-3.

Transitions are deterministic:
    buy:  s -> s+1 (capped at last regular episode)
    wait: s -> s   (stay at current episode)
    exit: s -> absorbing state
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from econirl.environments.base import DDCEnvironment

from .config import DGPConfig


# Action indices
BUY = 0
WAIT = 1
EXIT = 2


class SerializedContentEnvironment(DDCEnvironment):
    """Serialized content consumption with anchor identification.

    The state space has num_episodes regular states (episodes 0..E-1)
    plus one absorbing terminal state. The absorbing state has zero
    reward for all actions and transitions only to itself.

    The reward is action-dependent with 5 free parameters. The exit
    action is anchored to zero reward everywhere, which (combined
    with the absorbing state having V=0) uniquely identifies the
    reward function from observed choice probabilities.

    Parameters
    ----------
    config : DGPConfig
        Environment specification with true reward parameters.
    """

    def __init__(self, config: DGPConfig | None = None):
        if config is None:
            config = DGPConfig()
        self.config = config

        super().__init__(
            discount_factor=config.discount_factor,
            scale_parameter=config.scale_parameter,
        )

        self.observation_space = None
        self.action_space = None

        # Precompute matrices
        self._transitions = self._build_transitions()
        self._features = self._build_features()
        self._reward = self._build_reward()

    # --- State features ---

    def quality(self, s: int) -> float:
        """Episode quality: declines linearly with episode index."""
        if s >= self.config.num_episodes:
            return 0.0
        return 1.0 - 0.01 * s

    def wait_cost(self, s: int) -> float:
        """Wait-cost feature: non-monotone sinusoidal variation."""
        if s >= self.config.num_episodes:
            return 0.0
        return 0.5 + 0.5 * np.sin(2.0 * np.pi * s / self.config.num_episodes)

    # --- DDCEnvironment abstract properties ---

    @property
    def num_states(self) -> int:
        return self.config.num_states

    @property
    def num_actions(self) -> int:
        return self.config.num_actions

    @property
    def transition_matrices(self) -> jnp.ndarray:
        return self._transitions

    @property
    def feature_matrix(self) -> jnp.ndarray:
        return self._features

    @property
    def true_parameters(self) -> dict[str, float]:
        return {
            "alpha_buy": self.config.alpha_buy,
            "theta_e": self.config.theta_e,
            "theta_w": self.config.theta_w,
            "theta_wait_base": self.config.theta_wait_base,
            "theta_w_wait": self.config.theta_w_wait,
        }

    @property
    def parameter_names(self) -> list[str]:
        return ["alpha_buy", "theta_e", "theta_w", "theta_wait_base", "theta_w_wait"]

    # --- Internal construction ---

    def _build_transitions(self) -> jnp.ndarray:
        """Build deterministic transition matrices.

        Shape: (num_actions, num_states, num_states)
        Convention: T[a, s, s'] = P(s' | s, a)
        """
        S = self.config.num_states
        E = self.config.num_episodes
        absorbing = self.config.absorbing_state
        T = np.zeros((3, S, S))

        for s in range(E):
            # Buy: advance to next episode (capped at E-1)
            T[BUY, s, min(s + 1, E - 1)] = 1.0
            # Wait: stay at current episode
            T[WAIT, s, s] = 1.0
            # Exit: go to absorbing state
            T[EXIT, s, absorbing] = 1.0

        # Absorbing state: all actions loop back
        T[BUY, absorbing, absorbing] = 1.0
        T[WAIT, absorbing, absorbing] = 1.0
        T[EXIT, absorbing, absorbing] = 1.0

        return jnp.array(T, dtype=jnp.float32)

    def _build_features(self) -> jnp.ndarray:
        """Build feature matrix for linear utility.

        Shape: (num_states, num_actions, num_features)
        5 features: [alpha_buy_indicator, quality, w_buy, wait_base_indicator, w_wait]

        Features are designed so that:
            r(s, buy)  = alpha_buy * 1 + theta_e * quality(s) + theta_w * w(s) + 0 + 0
            r(s, wait) = 0 + 0 + 0 + theta_wait_base * 1 + theta_w_wait * w(s)
            r(s, exit) = 0 + 0 + 0 + 0 + 0  (anchor)
        """
        S = self.config.num_states
        E = self.config.num_episodes
        features = np.zeros((S, 3, 5))

        for s in range(E):
            q = self.quality(s)
            w = self.wait_cost(s)

            # Buy features
            features[s, BUY, 0] = 1.0       # alpha_buy indicator
            features[s, BUY, 1] = q          # quality
            features[s, BUY, 2] = w          # wait cost (for buy action)

            # Wait features
            features[s, WAIT, 3] = 1.0       # theta_wait_base indicator
            features[s, WAIT, 4] = w          # wait cost (for wait action)

            # Exit features: all zero (anchor)

        # Absorbing state: all features zero for all actions
        # (already initialized to zero)

        return jnp.array(features, dtype=jnp.float32)

    def _build_reward(self) -> jnp.ndarray:
        """Build the true reward matrix from parameters and features."""
        params = self.get_true_parameter_vector()
        return jnp.einsum("sak,k->sa", self._features, params)

    @property
    def reward_matrix(self) -> jnp.ndarray:
        """True reward matrix, shape (num_states, num_actions)."""
        return self._reward

    # --- DDCEnvironment abstract methods ---

    def _get_initial_state_distribution(self) -> np.ndarray:
        """All individuals start at episode 0."""
        dist = np.zeros(self.config.num_states)
        dist[0] = 1.0
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        return float(self._reward[state, action])

    def _sample_next_state(self, state: int, action: int) -> int:
        absorbing = self.config.absorbing_state
        if state == absorbing:
            return absorbing
        if action == EXIT:
            return absorbing
        if action == WAIT:
            return state
        if action == BUY:
            return min(state + 1, self.config.num_episodes - 1)
        raise ValueError(f"Unknown action {action}")

    # --- Counterfactual transition builders ---

    def build_skip_transitions(self, skip: int) -> jnp.ndarray:
        """Build transitions where buy advances by `skip` episodes.

        This is the Type II counterfactual: changing P(s'|s, buy)
        from s+1 to s+skip. Wait and exit transitions are unchanged.
        """
        S = self.config.num_states
        E = self.config.num_episodes
        absorbing = self.config.absorbing_state
        T = np.zeros((3, S, S))

        for s in range(E):
            T[BUY, s, min(s + skip, E - 1)] = 1.0
            T[WAIT, s, s] = 1.0
            T[EXIT, s, absorbing] = 1.0

        T[BUY, absorbing, absorbing] = 1.0
        T[WAIT, absorbing, absorbing] = 1.0
        T[EXIT, absorbing, absorbing] = 1.0

        return jnp.array(T, dtype=jnp.float32)

    def build_shifted_reward(self, wait_shift: float) -> jnp.ndarray:
        """Build reward matrix with shifted wait-cost feature.

        This is the Type I counterfactual: modifying w(s) by adding
        wait_shift to every state, keeping all parameters fixed.
        """
        S = self.config.num_states
        E = self.config.num_episodes
        shifted_features = np.array(self._features)

        for s in range(E):
            w_new = self.wait_cost(s) + wait_shift
            shifted_features[s, BUY, 2] = w_new
            shifted_features[s, WAIT, 4] = w_new

        params = self.get_true_parameter_vector()
        return jnp.einsum("sak,k->sa", jnp.array(shifted_features), params)
