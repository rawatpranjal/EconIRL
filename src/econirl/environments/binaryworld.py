"""Binaryworld environment for deep IRL benchmarking.

This module implements the Binaryworld environment from Wulfmeier et al. (2016)
for evaluating deep inverse reinforcement learning. The environment is an N x N
grid where each cell is randomly assigned a color (blue or red). The reward at
each state depends on the count of blue cells in the 3x3 neighborhood centered
on that cell.

The key property of Binaryworld is that the reward depends on a nonlinear
function of the binary features (a count threshold), which a linear model
cannot capture. This makes it a natural benchmark for comparing linear
MaxEnt IRL against deep IRL methods.

State space:
    N^2 states indexed as row * N + col. No terminal or absorbing state.

Action space:
    5 actions: Left (0), Right (1), Up (2), Down (3), Stay (4).

Features:
    A binary vector of length 9 encoding the 3x3 neighborhood colors centered
    on the cell. Boundary cells use zero-padding for out-of-bounds neighbors.
    Features are state-only and broadcast identically to all 5 actions.

Reward:
    +1 if exactly 4 of the 9 neighborhood cells are blue.
    -1 if exactly 5 of the 9 neighborhood cells are blue.
     0 otherwise.

References:
    Wulfmeier, M., Ondruska, P., & Posner, I. (2016). Maximum entropy deep
        inverse reinforcement learning.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment
from econirl.environments.objectworld import _build_grid_transitions, LEFT, RIGHT, UP, DOWN, STAY
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration
from econirl.core.types import Panel, Trajectory, DDCProblem


class BinaryworldEnvironment(DDCEnvironment):
    """N x N grid with binary cell colors and count-based rewards.

    Each cell is randomly assigned blue (1) or red (0). The reward at each
    state depends on how many of the 9 cells in the 3x3 neighborhood
    (centered on the current cell, with zero-padding at boundaries) are blue.
    Exactly 4 blue cells gives reward +1, exactly 5 gives -1, and all other
    counts give 0.

    This environment tests whether an IRL algorithm can recover reward
    structure that depends on higher-order feature interactions. A linear
    reward function over the 9 binary neighborhood features cannot represent
    the count-based thresholds, so deep IRL methods should outperform linear
    MaxEnt IRL on this environment.

    Example:
        >>> env = BinaryworldEnvironment(grid_size=8, seed=0)
        >>> print(f"States: {env.num_states}, Features: {env.feature_matrix.shape}")
    """

    def __init__(
        self,
        grid_size: int = 32,
        discount_factor: float = 0.9,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize the Binaryworld environment.

        Args:
            grid_size: Side length N of the N x N grid.
            discount_factor: Time discount factor beta in [0, 1).
            scale_parameter: Logit scale parameter sigma > 0.
            seed: Random seed for reproducible color assignment.
        """
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        self._grid_size = grid_size
        self._n_states = grid_size * grid_size

        # Set up Gymnasium spaces
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(5)

        # Randomly assign each cell blue (1) or red (0)
        self._color_map = self._np_random.integers(0, 2, size=self._n_states)

        # Build structural components
        self._transition_matrices = _build_grid_transitions(grid_size)
        self._feature_matrix = self._build_feature_matrix()
        self._true_reward = self._compute_reward()

    def _get_neighborhood(self, state: int) -> np.ndarray:
        """Get the 3x3 neighborhood colors for a state with zero-padding.

        The neighborhood is a flat array of 9 values ordered row-major:
        top-left, top-center, top-right, mid-left, center, mid-right,
        bottom-left, bottom-center, bottom-right.

        Out-of-bounds cells are padded with 0 (red).

        Args:
            state: Flat state index.

        Returns:
            Array of length 9 with binary color values.
        """
        row = state // self._grid_size
        col = state % self._grid_size
        neighborhood = np.zeros(9, dtype=np.float32)

        idx = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r = row + dr
                c = col + dc
                if 0 <= r < self._grid_size and 0 <= c < self._grid_size:
                    neighbor_state = r * self._grid_size + c
                    neighborhood[idx] = float(self._color_map[neighbor_state])
                # else: remains 0 (zero-padding)
                idx += 1

        return neighborhood

    def _build_feature_matrix(self) -> jnp.ndarray:
        """Build the binary neighborhood feature matrix.

        Each state gets a 9-dimensional binary feature vector encoding the
        colors of its 3x3 neighborhood. Features are state-only and broadcast
        identically to all 5 actions.

        Returns:
            Tensor of shape (n_states, 5, 9).
        """
        features_per_state = np.zeros((self._n_states, 9), dtype=np.float32)
        for s in range(self._n_states):
            features_per_state[s] = self._get_neighborhood(s)

        state_features = jnp.array(features_per_state, dtype=jnp.float32)
        # Broadcast state-only features to all 5 actions: (S, 9) -> (S, 5, 9)
        feature_matrix = jnp.broadcast_to(
            jnp.expand_dims(state_features, axis=1),
            (self._n_states, 5, 9),
        ).copy()
        return feature_matrix

    def _compute_reward(self) -> jnp.ndarray:
        """Compute the reward for each state based on blue neighbor count.

        The reward rule is:
            +1 if exactly 4 of the 9 neighborhood cells are blue.
            -1 if exactly 5 of the 9 neighborhood cells are blue.
             0 otherwise.

        Returns:
            Tensor of shape (n_states,) with reward values.
        """
        # Build with numpy then convert (JAX arrays are immutable)
        reward = np.zeros(self._n_states, dtype=np.float32)
        for s in range(self._n_states):
            neighborhood = self._get_neighborhood(s)
            blue_count = int(neighborhood.sum())
            if blue_count == 4:
                reward[s] = 1.0
            elif blue_count == 5:
                reward[s] = -1.0
        return jnp.array(reward)

    # ------------------------------------------------------------------
    # DDCEnvironment abstract property implementations
    # ------------------------------------------------------------------

    @property
    def num_states(self) -> int:
        return self._n_states

    @property
    def num_actions(self) -> int:
        return 5

    @property
    def transition_matrices(self) -> jnp.ndarray:
        return self._transition_matrices

    @property
    def feature_matrix(self) -> jnp.ndarray:
        return self._feature_matrix

    @property
    def true_reward(self) -> jnp.ndarray:
        """Return the ground-truth reward vector of shape (num_states,)."""
        return self._true_reward

    @property
    def true_parameters(self) -> dict[str, float]:
        """Return true parameters.

        The Binaryworld reward is not a linear function of the features
        (it depends on the count of blue neighbors, a nonlinear threshold).
        This returns placeholder values used only for the DDCEnvironment
        interface.
        """
        return {f"neighbor_{i}_weight": 1.0 for i in range(9)}

    @property
    def parameter_names(self) -> list[str]:
        return [f"neighbor_{i}_weight" for i in range(9)]

    @property
    def grid_size(self) -> int:
        """Return the side length of the grid."""
        return self._grid_size

    @property
    def state_dim(self) -> int:
        """Two-dimensional grid position."""
        return 2

    def encode_states(self, states: jnp.ndarray) -> jnp.ndarray:
        """Encode flat state indices to (row, col) normalized to [0, 1].

        Args:
            states: Tensor of flat state indices.

        Returns:
            Tensor of shape (batch, 2) with normalized row and column.
        """
        states_f = states.astype(jnp.float32)
        rows = (states_f // self._grid_size) / max(self._grid_size - 1, 1)
        cols = (states_f % self._grid_size) / max(self._grid_size - 1, 1)
        return jnp.stack([rows, cols], axis=-1)

    # ------------------------------------------------------------------
    # DDCEnvironment abstract method implementations
    # ------------------------------------------------------------------

    def _get_initial_state_distribution(self) -> np.ndarray:
        """Return uniform initial state distribution."""
        return np.ones(self._n_states) / self._n_states

    def _compute_flow_utility(self, state: int, action: int) -> float:
        """Return the reward for the given state (action-independent)."""
        return float(self._true_reward[state])

    def _sample_next_state(self, state: int, action: int) -> int:
        """Return deterministic next state."""
        row = state // self._grid_size
        col = state % self._grid_size

        if action == LEFT:
            col = max(col - 1, 0)
        elif action == RIGHT:
            col = min(col + 1, self._grid_size - 1)
        elif action == UP:
            row = max(row - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self._grid_size - 1)
        # STAY: no change

        return row * self._grid_size + col

    # ------------------------------------------------------------------
    # Demonstration generation
    # ------------------------------------------------------------------

    def simulate_demonstrations(
        self,
        n_demos: int,
        max_steps: int = 50,
        noise_fraction: float = 0.3,
        seed: int = 0,
    ) -> Panel:
        """Generate demonstration trajectories from the optimal policy.

        Solves for the optimal policy under the true reward using policy
        iteration, then samples trajectories. Each action is replaced with
        a uniformly random action with probability noise_fraction.

        Args:
            n_demos: Number of trajectories to generate.
            max_steps: Length of each trajectory.
            noise_fraction: Probability of replacing the optimal action
                with a uniform random action at each step.
            seed: Random seed for trajectory sampling.

        Returns:
            Panel containing the generated trajectories.
        """
        rng = np.random.default_rng(seed)

        # Build the reward matrix (S, A) from the state-only reward
        reward_matrix = jnp.broadcast_to(
            jnp.expand_dims(self._true_reward, axis=1),
            (self._n_states, 5),
        ).copy()

        # Solve for optimal policy
        problem = self.problem_spec
        operator = SoftBellmanOperator(problem, self._transition_matrices)
        result = policy_iteration(operator, reward_matrix)
        policy = result.policy  # (S, A)

        trajectories = []
        for i in range(n_demos):
            # Sample initial state uniformly
            state = int(rng.integers(0, self._n_states))
            states_list = []
            actions_list = []
            next_states_list = []

            for _ in range(max_steps):
                # Choose action: optimal with probability (1 - noise_fraction),
                # uniform random otherwise
                if rng.random() < noise_fraction:
                    action = int(rng.integers(0, 5))
                else:
                    probs = np.array(policy[state])
                    action = int(rng.choice(5, p=probs))

                next_state = self._sample_next_state(state, action)

                states_list.append(state)
                actions_list.append(action)
                next_states_list.append(next_state)

                state = next_state

            trajectories.append(
                Trajectory(
                    states=jnp.array(states_list, dtype=jnp.int32),
                    actions=jnp.array(actions_list, dtype=jnp.int32),
                    next_states=jnp.array(next_states_list, dtype=jnp.int32),
                    individual_id=i,
                )
            )

        return Panel(trajectories=trajectories)

    def describe(self) -> str:
        """Return a human-readable description of the environment."""
        N = self._grid_size
        blue_count = int(self._color_map.sum())
        return (
            f"Binaryworld Environment ({N}x{N})\n"
            f"{'=' * 40}\n"
            f"States: {self.num_states} ({N}x{N} grid)\n"
            f"Actions: Left (0), Right (1), Up (2), Down (3), Stay (4)\n"
            f"Blue cells: {blue_count} / {self.num_states}\n"
            f"Discount factor: {self._discount_factor}\n"
        )
