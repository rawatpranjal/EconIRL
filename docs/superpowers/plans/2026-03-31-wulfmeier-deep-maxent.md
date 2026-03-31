# Wulfmeier (2016) Deep MaxEnt IRL Replication — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate Objectworld and Binaryworld benchmarks from Wulfmeier et al. (2016), validating MCEIRLNeural against the paper's EVD curves.

**Architecture:** Two new environment classes (Objectworld, Binaryworld) inherit from DDCEnvironment with shared grid transition logic. One benchmark script runs both environments across multiple demonstration counts and compares MCEIRLNeural vs linear MCE-IRL using Expected Value Difference.

**Tech Stack:** PyTorch, econirl.environments.base.DDCEnvironment, econirl.estimators.mceirl_neural.MCEIRLNeural, econirl.estimation.mce_irl.MCEIRLEstimator

---

### Task 1: Objectworld Environment

**Files:**
- Create: `src/econirl/environments/objectworld.py`
- Test: `tests/test_objectworld.py`

- [ ] **Step 1: Write failing tests for Objectworld**

```python
# tests/test_objectworld.py
"""Tests for Objectworld environment (Levine et al. 2011 / Wulfmeier 2016)."""

import pytest
import torch

from econirl.environments.objectworld import ObjectworldEnvironment


class TestObjectworldConstruction:
    def test_default_creation(self):
        env = ObjectworldEnvironment(grid_size=8, seed=42)
        assert env.num_states == 64
        assert env.num_actions == 5

    def test_transition_shape(self):
        env = ObjectworldEnvironment(grid_size=8, seed=42)
        T = env.transition_matrices
        assert T.shape == (5, 64, 64)

    def test_transition_rows_sum_to_one(self):
        env = ObjectworldEnvironment(grid_size=8, seed=42)
        T = env.transition_matrices
        row_sums = T.sum(dim=2)
        assert torch.allclose(row_sums, torch.ones_like(row_sums))

    def test_transition_deterministic(self):
        env = ObjectworldEnvironment(grid_size=8, seed=42)
        T = env.transition_matrices
        assert (T.max(dim=2).values == 1.0).all()

    def test_feature_matrix_continuous(self):
        env = ObjectworldEnvironment(grid_size=8, n_colors=2, feature_type="continuous", seed=42)
        F = env.feature_matrix
        assert F.shape == (64, 5, 2)
        assert (F >= 0).all()
        assert (F <= 1).all()  # normalized by grid_size

    def test_feature_matrix_discrete(self):
        env = ObjectworldEnvironment(grid_size=8, n_colors=2, feature_type="discrete", seed=42)
        F = env.feature_matrix
        # C * M = 2 * 8 = 16 binary features
        assert F.shape == (64, 5, 16)
        assert set(F.unique().tolist()).issubset({0.0, 1.0})

    def test_reward_values(self):
        env = ObjectworldEnvironment(grid_size=8, seed=42)
        r = env.true_reward
        assert r.shape == (64,)
        assert set(r.unique().tolist()).issubset({-1.0, 0.0, 1.0})

    def test_seed_reproducibility(self):
        env1 = ObjectworldEnvironment(grid_size=8, seed=42)
        env2 = ObjectworldEnvironment(grid_size=8, seed=42)
        assert torch.equal(env1.true_reward, env2.true_reward)
        assert torch.equal(env1.feature_matrix, env2.feature_matrix)

    def test_different_seeds(self):
        env1 = ObjectworldEnvironment(grid_size=8, seed=42)
        env2 = ObjectworldEnvironment(grid_size=8, seed=99)
        assert not torch.equal(env1.true_reward, env2.true_reward)

    def test_problem_spec(self):
        env = ObjectworldEnvironment(grid_size=8, seed=42)
        p = env.problem_spec
        assert p.num_states == 64
        assert p.num_actions == 5
        assert p.discount_factor == 0.9


class TestObjectworldDemonstrations:
    def test_simulate_demos(self):
        env = ObjectworldEnvironment(grid_size=8, seed=42)
        panel = env.simulate_demonstrations(n_demos=4, max_steps=20, seed=0)
        assert panel.num_individuals == 4
        states = panel.get_all_states()
        actions = panel.get_all_actions()
        assert (states >= 0).all() and (states < 64).all()
        assert (actions >= 0).all() and (actions < 5).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_objectworld.py -v`
Expected: ImportError — `objectworld` module does not exist

- [ ] **Step 3: Implement ObjectworldEnvironment**

```python
# src/econirl/environments/objectworld.py
"""Objectworld environment for IRL benchmarking (Levine et al. 2011).

An N x N grid with randomly placed colored objects. The reward depends
on distances to objects of specific colors. Used by Wulfmeier et al.
(2016) to benchmark Deep MaxEnt IRL.

State space:
    N^2 states indexed as row * N + col. No terminal state.

Action space:
    5 actions: Left (0), Right (1), Up (2), Down (3), Stay (4).

Reward:
    +1 if within distance 3 of color 0 AND distance 2 of color 1.
    -1 if within distance 3 of color 0 but NOT within distance 2 of color 1.
     0 otherwise.

Features (continuous):
    C dimensions: normalized min-distance to nearest object of each color.

Features (discrete):
    C * M binary dimensions: indicator for color c within distance d,
    for d in 1..M.

Reference:
    Levine, S., Popovic, Z., & Koltun, V. (2011). Nonlinear inverse
        reinforcement learning with Gaussian processes. NeurIPS.
    Wulfmeier, M., Ondruska, P., & Posner, I. (2016). Maximum entropy
        deep inverse reinforcement learning. arXiv:1507.04888.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


def _build_grid_transitions(grid_size: int) -> torch.Tensor:
    """Build deterministic 5-action grid transition matrices.

    Actions: 0=Left, 1=Right, 2=Up, 3=Down, 4=Stay.
    Walls cause the agent to stay in place. No terminal state.

    Returns:
        Tensor of shape (5, N^2, N^2).
    """
    n_states = grid_size * grid_size
    T = torch.zeros((5, n_states, n_states), dtype=torch.float32)

    for s in range(n_states):
        row, col = divmod(s, grid_size)

        # Left
        nc = max(col - 1, 0)
        T[0, s, row * grid_size + nc] = 1.0
        # Right
        nc = min(col + 1, grid_size - 1)
        T[1, s, row * grid_size + nc] = 1.0
        # Up
        nr = max(row - 1, 0)
        T[2, s, nr * grid_size + col] = 1.0
        # Down
        nr = min(row + 1, grid_size - 1)
        T[3, s, nr * grid_size + col] = 1.0
        # Stay
        T[4, s, s] = 1.0

    return T


class ObjectworldEnvironment(DDCEnvironment):
    """Objectworld: grid with colored objects and distance-based rewards.

    Args:
        grid_size: Side length N of the N x N grid.
        n_colors: Number of object colors C.
        n_objects_per_color: Objects placed per color.
        feature_type: "continuous" (C features) or "discrete" (C*M features).
        discount_factor: Discount gamma.
        seed: Random seed for object placement.
    """

    LEFT, RIGHT, UP, DOWN, STAY = 0, 1, 2, 3, 4

    def __init__(
        self,
        grid_size: int = 32,
        n_colors: int = 2,
        n_objects_per_color: int = 3,
        feature_type: str = "continuous",
        discount_factor: float = 0.9,
        scale_parameter: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )
        if feature_type not in ("continuous", "discrete"):
            raise ValueError(f"feature_type must be 'continuous' or 'discrete', got '{feature_type}'")

        self._grid_size = grid_size
        self._n_colors = n_colors
        self._n_objects_per_color = n_objects_per_color
        self._feature_type = feature_type
        self._n_states = grid_size * grid_size

        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(5)

        # Place objects randomly
        rng = np.random.RandomState(seed)
        # objects[c] = list of (row, col) positions for color c
        self._objects: list[list[tuple[int, int]]] = []
        for _ in range(n_colors):
            positions = []
            for _ in range(n_objects_per_color):
                r = rng.randint(0, grid_size)
                c = rng.randint(0, grid_size)
                positions.append((r, c))
            self._objects.append(positions)

        # Precompute distance maps, features, reward, transitions
        self._min_distances = self._compute_min_distances()
        self._true_reward = self._compute_reward()
        self._transition_matrices = _build_grid_transitions(grid_size)
        self._feature_matrix = self._compute_features()

    def _compute_min_distances(self) -> torch.Tensor:
        """Compute min Euclidean distance from each state to each color.

        Returns:
            Tensor of shape (n_states, n_colors).
        """
        N = self._grid_size
        dists = torch.full((self._n_states, self._n_colors), float("inf"))
        for c, positions in enumerate(self._objects):
            for s in range(self._n_states):
                sr, sc = divmod(s, N)
                for or_, oc in positions:
                    d = math.sqrt((sr - or_) ** 2 + (sc - oc) ** 2)
                    if d < dists[s, c]:
                        dists[s, c] = d
        return dists

    def _compute_reward(self) -> torch.Tensor:
        """Compute reward: +1 near both colors, -1 near only color 0, else 0."""
        d = self._min_distances
        near_c0 = d[:, 0] <= 3.0
        near_c1 = d[:, 1] <= 2.0
        reward = torch.zeros(self._n_states)
        reward[near_c0 & near_c1] = 1.0
        reward[near_c0 & ~near_c1] = -1.0
        return reward

    def _compute_features(self) -> torch.Tensor:
        """Compute state features, broadcast to all actions."""
        N = self._grid_size
        C = self._n_colors
        d = self._min_distances  # (S, C)

        if self._feature_type == "continuous":
            # C features: normalized min distance per color
            f = d / N  # normalize to [0, 1]
            f = f.clamp(max=1.0)
        else:
            # Discrete: C * M binary features
            M = N
            f = torch.zeros(self._n_states, C * M)
            for c in range(C):
                for m in range(M):
                    threshold = float(m + 1)
                    f[:, c * M + m] = (d[:, c] <= threshold).float()

        # Broadcast to (S, A, K) — same features for all actions
        K = f.shape[1]
        features = f.unsqueeze(1).expand(self._n_states, 5, K).clone()
        return features

    @property
    def num_states(self) -> int:
        return self._n_states

    @property
    def num_actions(self) -> int:
        return 5

    @property
    def transition_matrices(self) -> torch.Tensor:
        return self._transition_matrices

    @property
    def feature_matrix(self) -> torch.Tensor:
        return self._feature_matrix

    @property
    def true_reward(self) -> torch.Tensor:
        """Ground-truth reward vector of shape (S,)."""
        return self._true_reward

    @property
    def true_parameters(self) -> dict[str, float]:
        return {"reward_positive": 1.0, "reward_negative": -1.0}

    @property
    def parameter_names(self) -> list[str]:
        return ["reward_positive", "reward_negative"]

    def simulate_demonstrations(
        self,
        n_demos: int,
        max_steps: int = 50,
        noise_fraction: float = 0.3,
        seed: int = 0,
    ) -> "Panel":
        """Sample expert demonstrations with random action noise.

        Computes the optimal policy under the true reward, then samples
        trajectories where each action is replaced with a uniform random
        action with probability noise_fraction.

        Args:
            n_demos: Number of trajectories.
            max_steps: Steps per trajectory.
            noise_fraction: Probability of taking a random action.
            seed: Random seed for sampling.

        Returns:
            Panel of demonstration trajectories.
        """
        from econirl.core.bellman import SoftBellmanOperator
        from econirl.core.solvers import policy_iteration
        from econirl.core.types import Panel, Trajectory

        # Build reward matrix (S, A) — same reward for all actions
        reward_sa = self._true_reward.unsqueeze(1).expand(self._n_states, 5)

        operator = SoftBellmanOperator(self.problem_spec, self._transition_matrices)
        result = policy_iteration(operator, reward_sa.double(), tol=1e-10, max_iter=200)
        policy = result.policy.float()  # (S, A)

        rng = np.random.RandomState(seed)
        trajectories = []

        for i in range(n_demos):
            states, actions, next_states = [], [], []
            s = rng.randint(0, self._n_states)

            for _ in range(max_steps):
                # With noise_fraction probability, take random action
                if rng.random() < noise_fraction:
                    a = rng.randint(0, 5)
                else:
                    probs = policy[s].numpy()
                    a = rng.choice(5, p=probs / probs.sum())

                # Deterministic transition
                s_next = self._transition_matrices[a, s].argmax().item()

                states.append(s)
                actions.append(a)
                next_states.append(s_next)
                s = s_next

            trajectories.append(Trajectory(
                states=torch.tensor(states, dtype=torch.long),
                actions=torch.tensor(actions, dtype=torch.long),
                next_states=torch.tensor(next_states, dtype=torch.long),
                individual_id=i,
            ))

        return Panel(trajectories=trajectories)

    def _get_initial_state_distribution(self) -> np.ndarray:
        return np.ones(self._n_states) / self._n_states

    def _compute_flow_utility(self, state: int, action: int) -> float:
        return self._true_reward[state].item()

    def _sample_next_state(self, state: int, action: int) -> int:
        return self._transition_matrices[action, state].argmax().item()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_objectworld.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/econirl/environments/objectworld.py tests/test_objectworld.py
git commit -m "feat: add Objectworld environment for Wulfmeier (2016) replication"
```

---

### Task 2: Binaryworld Environment

**Files:**
- Create: `src/econirl/environments/binaryworld.py`
- Test: `tests/test_binaryworld.py`

- [ ] **Step 1: Write failing tests for Binaryworld**

```python
# tests/test_binaryworld.py
"""Tests for Binaryworld environment (Wulfmeier 2016)."""

import pytest
import torch

from econirl.environments.binaryworld import BinaryworldEnvironment


class TestBinaryworldConstruction:
    def test_default_creation(self):
        env = BinaryworldEnvironment(grid_size=8, seed=42)
        assert env.num_states == 64
        assert env.num_actions == 5

    def test_transition_shape(self):
        env = BinaryworldEnvironment(grid_size=8, seed=42)
        T = env.transition_matrices
        assert T.shape == (5, 64, 64)

    def test_transition_rows_sum_to_one(self):
        env = BinaryworldEnvironment(grid_size=8, seed=42)
        T = env.transition_matrices
        row_sums = T.sum(dim=2)
        assert torch.allclose(row_sums, torch.ones_like(row_sums))

    def test_feature_shape_and_values(self):
        env = BinaryworldEnvironment(grid_size=8, seed=42)
        F = env.feature_matrix
        # 9 binary features (3x3 neighborhood)
        assert F.shape == (64, 5, 9)
        assert set(F.unique().tolist()).issubset({0.0, 1.0})

    def test_reward_values(self):
        env = BinaryworldEnvironment(grid_size=8, seed=42)
        r = env.true_reward
        assert r.shape == (64,)
        assert set(r.unique().tolist()).issubset({-1.0, 0.0, 1.0})

    def test_neighborhood_features_center_cell(self):
        """Verify that center cells have correct 3x3 neighborhood features."""
        env = BinaryworldEnvironment(grid_size=8, seed=42)
        # Check a non-boundary cell (row=3, col=3, state=27)
        s = 3 * 8 + 3
        feat = env.feature_matrix[s, 0, :]  # same for all actions
        # Feature should have exactly 9 entries, all 0 or 1
        assert feat.shape == (9,)
        assert feat.sum() >= 0  # at least 0 blue neighbors
        assert feat.sum() <= 9  # at most 9 blue neighbors

    def test_neighborhood_features_corner(self):
        """Corner cell (0,0) should have zero-padding for out-of-bounds neighbors."""
        env = BinaryworldEnvironment(grid_size=8, seed=42)
        feat = env.feature_matrix[0, 0, :]  # state 0 = (0,0)
        # 4 out of 9 neighbors are out-of-bounds (zero-padded)
        # So at most 5 can be blue (the cell itself + 2 right + 1 below + 1 diagonal)
        assert feat.sum() <= 5

    def test_reward_matches_blue_count(self):
        """Reward should be +1 when 4 blue, -1 when 5 blue, 0 otherwise."""
        env = BinaryworldEnvironment(grid_size=8, seed=42)
        for s in range(env.num_states):
            blue_count = env.feature_matrix[s, 0, :].sum().item()
            r = env.true_reward[s].item()
            if blue_count == 4:
                assert r == 1.0, f"State {s}: blue_count=4 but reward={r}"
            elif blue_count == 5:
                assert r == -1.0, f"State {s}: blue_count=5 but reward={r}"
            else:
                assert r == 0.0, f"State {s}: blue_count={blue_count} but reward={r}"

    def test_seed_reproducibility(self):
        env1 = BinaryworldEnvironment(grid_size=8, seed=42)
        env2 = BinaryworldEnvironment(grid_size=8, seed=42)
        assert torch.equal(env1.true_reward, env2.true_reward)

    def test_simulate_demos(self):
        env = BinaryworldEnvironment(grid_size=8, seed=42)
        panel = env.simulate_demonstrations(n_demos=4, max_steps=20, seed=0)
        assert panel.num_individuals == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_binaryworld.py -v`
Expected: ImportError — `binaryworld` module does not exist

- [ ] **Step 3: Implement BinaryworldEnvironment**

```python
# src/econirl/environments/binaryworld.py
"""Binaryworld environment for IRL benchmarking (Wulfmeier 2016).

An N x N grid where each cell is randomly colored blue (1) or red (0).
The reward depends on the count of blue cells in the 3x3 neighborhood.
This creates higher-order feature interactions that linear models cannot
capture, demonstrating the advantage of neural reward approximation.

State space:
    N^2 states indexed as row * N + col. No terminal state.

Action space:
    5 actions: Left (0), Right (1), Up (2), Down (3), Stay (4).

Features:
    9 binary dimensions encoding the 3x3 neighborhood colors (including
    the cell itself). Boundary cells use zero-padding.

Reward:
    +1 if exactly 4 of 9 neighborhood cells are blue.
    -1 if exactly 5 of 9 neighborhood cells are blue.
     0 otherwise.

Reference:
    Wulfmeier, M., Ondruska, P., & Posner, I. (2016). Maximum entropy
        deep inverse reinforcement learning. arXiv:1507.04888.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment
from econirl.environments.objectworld import _build_grid_transitions


class BinaryworldEnvironment(DDCEnvironment):
    """Binaryworld: grid with binary colors and neighborhood-count rewards.

    Args:
        grid_size: Side length N of the N x N grid.
        discount_factor: Discount gamma.
        seed: Random seed for color assignment.
    """

    LEFT, RIGHT, UP, DOWN, STAY = 0, 1, 2, 3, 4

    def __init__(
        self,
        grid_size: int = 32,
        discount_factor: float = 0.9,
        scale_parameter: float = 1.0,
        seed: int = 0,
    ):
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )
        self._grid_size = grid_size
        self._n_states = grid_size * grid_size

        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(5)

        # Randomly assign colors: 1 = blue, 0 = red
        rng = np.random.RandomState(seed)
        self._colors = torch.tensor(
            rng.randint(0, 2, size=(grid_size, grid_size)),
            dtype=torch.float32,
        )

        self._transition_matrices = _build_grid_transitions(grid_size)
        self._feature_matrix = self._compute_features()
        self._true_reward = self._compute_reward()

    def _compute_features(self) -> torch.Tensor:
        """Compute 3x3 neighborhood binary features for each state.

        Returns:
            Tensor of shape (S, 5, 9) — same features for all actions.
        """
        N = self._grid_size
        features = torch.zeros(self._n_states, 9)

        for s in range(self._n_states):
            row, col = divmod(s, N)
            idx = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < N and 0 <= nc < N:
                        features[s, idx] = self._colors[nr, nc]
                    # else: zero-padding (already zero)
                    idx += 1

        # Broadcast to (S, A, K)
        return features.unsqueeze(1).expand(self._n_states, 5, 9).clone()

    def _compute_reward(self) -> torch.Tensor:
        """Reward: +1 if 4 blue neighbors, -1 if 5 blue, 0 otherwise."""
        blue_counts = self._feature_matrix[:, 0, :].sum(dim=1)  # (S,)
        reward = torch.zeros(self._n_states)
        reward[blue_counts == 4] = 1.0
        reward[blue_counts == 5] = -1.0
        return reward

    @property
    def num_states(self) -> int:
        return self._n_states

    @property
    def num_actions(self) -> int:
        return 5

    @property
    def transition_matrices(self) -> torch.Tensor:
        return self._transition_matrices

    @property
    def feature_matrix(self) -> torch.Tensor:
        return self._feature_matrix

    @property
    def true_reward(self) -> torch.Tensor:
        return self._true_reward

    @property
    def true_parameters(self) -> dict[str, float]:
        return {"reward_4blue": 1.0, "reward_5blue": -1.0}

    @property
    def parameter_names(self) -> list[str]:
        return ["reward_4blue", "reward_5blue"]

    def simulate_demonstrations(
        self,
        n_demos: int,
        max_steps: int = 50,
        noise_fraction: float = 0.3,
        seed: int = 0,
    ) -> "Panel":
        """Sample expert demonstrations with random action noise."""
        from econirl.core.bellman import SoftBellmanOperator
        from econirl.core.solvers import policy_iteration
        from econirl.core.types import Panel, Trajectory

        reward_sa = self._true_reward.unsqueeze(1).expand(self._n_states, 5)
        operator = SoftBellmanOperator(self.problem_spec, self._transition_matrices)
        result = policy_iteration(operator, reward_sa.double(), tol=1e-10, max_iter=200)
        policy = result.policy.float()

        rng = np.random.RandomState(seed)
        trajectories = []

        for i in range(n_demos):
            states, actions, next_states = [], [], []
            s = rng.randint(0, self._n_states)

            for _ in range(max_steps):
                if rng.random() < noise_fraction:
                    a = rng.randint(0, 5)
                else:
                    probs = policy[s].numpy()
                    a = rng.choice(5, p=probs / probs.sum())

                s_next = self._transition_matrices[a, s].argmax().item()
                states.append(s)
                actions.append(a)
                next_states.append(s_next)
                s = s_next

            trajectories.append(Trajectory(
                states=torch.tensor(states, dtype=torch.long),
                actions=torch.tensor(actions, dtype=torch.long),
                next_states=torch.tensor(next_states, dtype=torch.long),
                individual_id=i,
            ))

        return Panel(trajectories=trajectories)

    def _get_initial_state_distribution(self) -> np.ndarray:
        return np.ones(self._n_states) / self._n_states

    def _compute_flow_utility(self, state: int, action: int) -> float:
        return self._true_reward[state].item()

    def _sample_next_state(self, state: int, action: int) -> int:
        return self._transition_matrices[action, state].argmax().item()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_binaryworld.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/econirl/environments/binaryworld.py tests/test_binaryworld.py
git commit -m "feat: add Binaryworld environment for Wulfmeier (2016) replication"
```

---

### Task 3: Register Environments

**Files:**
- Modify: `src/econirl/environments/__init__.py`

- [ ] **Step 1: Add exports**

```python
# Add to src/econirl/environments/__init__.py
from econirl.environments.binaryworld import BinaryworldEnvironment
from econirl.environments.objectworld import ObjectworldEnvironment
```

Add `"BinaryworldEnvironment"` and `"ObjectworldEnvironment"` to the `__all__` list.

- [ ] **Step 2: Verify imports work**

Run: `python3 -c "from econirl.environments import ObjectworldEnvironment, BinaryworldEnvironment; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/econirl/environments/__init__.py
git commit -m "feat: register Objectworld and Binaryworld in environments module"
```

---

### Task 4: Benchmark Script

**Files:**
- Create: `examples/wulfmeier-deep-maxent/replicate.py`

- [ ] **Step 1: Create directory and script**

```bash
mkdir -p examples/wulfmeier-deep-maxent
```

```python
# examples/wulfmeier-deep-maxent/replicate.py
#!/usr/bin/env python3
"""Wulfmeier (2016) Deep MaxEnt IRL — Objectworld & Binaryworld Benchmarks.

Replicates Figures 5 and 6 from:
    Wulfmeier, M., Ondruska, P., & Posner, I. (2016). Maximum entropy
    deep inverse reinforcement learning. arXiv:1507.04888.

Compares MCEIRLNeural (DeepIRL) against linear MCE-IRL on two benchmarks:
    1. Objectworld: distance-based features, mostly linear reward
    2. Binaryworld: binary neighborhood features, nonlinear reward

Metric: Expected Value Difference (EVD) — lower is better.

Usage:
    python examples/wulfmeier-deep-maxent/replicate.py
"""

import json
import os
import time

import numpy as np
import torch

from econirl.environments.objectworld import ObjectworldEnvironment
from econirl.environments.binaryworld import BinaryworldEnvironment
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration


# ── Configuration ──
GRID_SIZE = 32
DEMO_COUNTS = [8, 16, 32, 64, 128]
N_SEEDS = 5
MAX_STEPS = 50
NOISE_FRACTION = 0.3
DISCOUNT = 0.9
N_TRANSFER = 5  # number of transfer environments per seed

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


def compute_evd(true_reward, learned_policy, transitions, problem):
    """Compute Expected Value Difference.

    EVD = mean_s [V*(s; r_true) - V^pi_learned(s; r_true)]

    where V* is the value under optimal policy for true reward,
    and V^pi_learned is the value of the learned policy under true reward.
    """
    n_states = problem.num_states
    n_actions = problem.num_actions
    operator = SoftBellmanOperator(problem, transitions)

    # Reward matrix: broadcast state reward to all actions
    reward_sa = true_reward.unsqueeze(1).expand(n_states, n_actions).double()

    # Optimal value under true reward
    opt_result = policy_iteration(operator, reward_sa, tol=1e-10, max_iter=200)
    v_star = opt_result.value_function  # (S,)

    # Value of learned policy under true reward
    # V^pi(s) = sum_a pi(a|s) [r(s,a) + gamma * sum_s' P(s'|s,a) V^pi(s')]
    # Solve: V = R_pi + gamma * P_pi * V
    pi = learned_policy.double()  # (S, A)
    r_pi = (pi * reward_sa).sum(dim=1)  # (S,)
    P_pi = torch.einsum("sa,ast->st", pi, transitions.double())  # (S, S)
    I = torch.eye(n_states, dtype=torch.float64)
    v_learned = torch.linalg.solve(I - DISCOUNT * P_pi, r_pi)

    evd = (v_star - v_learned).mean().item()
    return evd


def run_estimator_mce_neural(panel, env, features):
    """Run MCEIRLNeural (DeepIRL) and return learned policy."""
    from econirl.estimators.mceirl_neural import MCEIRLNeural

    model = MCEIRLNeural(
        n_states=env.num_states,
        n_actions=env.num_actions,
        discount=DISCOUNT,
        reward_type="state",
        reward_hidden_dim=64,
        reward_num_layers=2,
        max_epochs=200,
        lr=0.01,
        inner_solver="hybrid",
        inner_tol=1e-8,
        inner_max_iter=5000,
        verbose=False,
    )

    # Build DataFrame from panel
    import pandas as pd
    rows = []
    for traj in panel.trajectories:
        for t in range(len(traj.states)):
            rows.append({
                "agent_id": traj.individual_id,
                "state": traj.states[t].item(),
                "action": traj.actions[t].item(),
            })
    df = pd.DataFrame(rows)

    model.fit(
        data=df,
        state="state",
        action="action",
        id="agent_id",
        transitions=env.transition_matrices.numpy(),
        features=features,
    )
    return torch.tensor(model.policy_, dtype=torch.float32)


def run_estimator_mce_linear(panel, env):
    """Run linear MCE-IRL and return learned policy."""
    from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
    from econirl.preferences.linear import LinearUtility

    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=[f"f{i}" for i in range(env.feature_matrix.shape[2])],
    )

    estimator = MCEIRLEstimator(config=MCEIRLConfig(
        optimizer="L-BFGS-B",
        inner_solver="hybrid",
        inner_max_iter=5000,
        inner_tol=1e-8,
        outer_max_iter=500,
        outer_tol=1e-6,
        compute_se=False,
        verbose=False,
    ))

    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    return result.policy.float()


def run_benchmark(env_class, env_name):
    """Run the full benchmark for one environment class."""
    print(f"\n{'='*60}")
    print(f"  {env_name} Benchmark (Wulfmeier 2016)")
    print(f"{'='*60}")

    results = {}

    for n_demos in DEMO_COUNTS:
        deep_evds = []
        linear_evds = []

        for seed in range(N_SEEDS):
            env = env_class(grid_size=GRID_SIZE, discount_factor=DISCOUNT, seed=seed)
            panel = env.simulate_demonstrations(
                n_demos=n_demos, max_steps=MAX_STEPS,
                noise_fraction=NOISE_FRACTION, seed=seed * 1000,
            )

            # Feature array for neural estimator (S, K)
            features = env.feature_matrix[:, 0, :].numpy()  # state features (same across actions)

            print(f"  N={n_demos}, seed={seed}: ", end="", flush=True)

            # DeepIRL
            t0 = time.time()
            try:
                deep_policy = run_estimator_mce_neural(panel, env, features)
                deep_evd = compute_evd(env.true_reward, deep_policy, env.transition_matrices, env.problem_spec)
                deep_evds.append(deep_evd)
                dt = time.time() - t0
                print(f"Deep={deep_evd:.2f} ({dt:.1f}s)", end="  ")
            except Exception as e:
                print(f"Deep=FAIL ({e})", end="  ")

            # Linear MCE-IRL
            t0 = time.time()
            try:
                linear_policy = run_estimator_mce_linear(panel, env)
                linear_evd = compute_evd(env.true_reward, linear_policy, env.transition_matrices, env.problem_spec)
                linear_evds.append(linear_evd)
                dt = time.time() - t0
                print(f"Linear={linear_evd:.2f} ({dt:.1f}s)")
            except Exception as e:
                print(f"Linear=FAIL ({e})")

        results[n_demos] = {
            "deep_mean": float(np.mean(deep_evds)) if deep_evds else None,
            "deep_std": float(np.std(deep_evds)) if deep_evds else None,
            "linear_mean": float(np.mean(linear_evds)) if linear_evds else None,
            "linear_std": float(np.std(linear_evds)) if linear_evds else None,
        }

    # Print summary
    print(f"\n  {'N':>5}  {'DeepIRL':>16}  {'Linear MCE':>16}")
    print(f"  {'-'*5}  {'-'*16}  {'-'*16}")
    for n_demos in DEMO_COUNTS:
        r = results[n_demos]
        deep_str = f"{r['deep_mean']:.2f} +/- {r['deep_std']:.2f}" if r['deep_mean'] is not None else "FAIL"
        lin_str = f"{r['linear_mean']:.2f} +/- {r['linear_std']:.2f}" if r['linear_mean'] is not None else "FAIL"
        print(f"  {n_demos:>5}  {deep_str:>16}  {lin_str:>16}")

    return results


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Wulfmeier (2016) Deep MaxEnt IRL Replication")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, Discount: {DISCOUNT}")
    print(f"Demos: {DEMO_COUNTS}, Seeds: {N_SEEDS}")

    all_results = {}

    all_results["objectworld"] = run_benchmark(ObjectworldEnvironment, "Objectworld")
    all_results["binaryworld"] = run_benchmark(BinaryworldEnvironment, "Binaryworld")

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on a small scale to verify it works**

Run: `python3 -c "
from econirl.environments.objectworld import ObjectworldEnvironment
env = ObjectworldEnvironment(grid_size=8, seed=42)
panel = env.simulate_demonstrations(n_demos=4, max_steps=20, seed=0)
print(f'States: {env.num_states}, Demos: {panel.num_individuals}, Obs: {panel.num_observations}')
print(f'Reward: {env.true_reward.unique()}')
"`
Expected: prints environment info without errors

- [ ] **Step 3: Commit**

```bash
git add examples/wulfmeier-deep-maxent/replicate.py
git commit -m "feat: add Wulfmeier (2016) Deep MaxEnt replication benchmark script"
```

---

### Task 5: Integration Test

**Files:**
- Create: `tests/test_wulfmeier_benchmark.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_wulfmeier_benchmark.py
"""Integration test for Wulfmeier (2016) Deep MaxEnt IRL replication.

Runs a small-scale version (8x8 grid) to verify that MCEIRLNeural
achieves lower EVD than linear MCE-IRL on Binaryworld, where the
nonlinear reward structure gives neural methods a clear advantage.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from econirl.environments.binaryworld import BinaryworldEnvironment
from econirl.environments.objectworld import ObjectworldEnvironment
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration


def _compute_evd(true_reward, learned_policy, transitions, problem):
    """Compute Expected Value Difference."""
    n_states = problem.num_states
    n_actions = problem.num_actions
    gamma = problem.discount_factor
    operator = SoftBellmanOperator(problem, transitions)

    reward_sa = true_reward.unsqueeze(1).expand(n_states, n_actions).double()
    opt_result = policy_iteration(operator, reward_sa, tol=1e-10, max_iter=200)
    v_star = opt_result.value_function

    pi = learned_policy.double()
    r_pi = (pi * reward_sa).sum(dim=1)
    P_pi = torch.einsum("sa,ast->st", pi, transitions.double())
    I = torch.eye(n_states, dtype=torch.float64)
    v_learned = torch.linalg.solve(I - gamma * P_pi, r_pi)

    return (v_star - v_learned).mean().item()


def _panel_to_df(panel):
    rows = []
    for traj in panel.trajectories:
        for t in range(len(traj.states)):
            rows.append({
                "agent_id": traj.individual_id,
                "state": traj.states[t].item(),
                "action": traj.actions[t].item(),
            })
    return pd.DataFrame(rows)


@pytest.mark.slow
class TestWulfmeierBinaryworld:
    """On Binaryworld, DeepIRL should beat linear MCE-IRL."""

    def test_deep_beats_linear_on_binaryworld(self):
        torch.manual_seed(42)
        env = BinaryworldEnvironment(grid_size=8, discount_factor=0.9, seed=42)
        panel = env.simulate_demonstrations(n_demos=32, max_steps=30, seed=0)
        features = env.feature_matrix[:, 0, :].numpy()
        df = _panel_to_df(panel)

        # DeepIRL
        from econirl.estimators.mceirl_neural import MCEIRLNeural
        deep_model = MCEIRLNeural(
            n_states=64, n_actions=5, discount=0.9,
            reward_type="state", reward_hidden_dim=32,
            reward_num_layers=2, max_epochs=100, lr=0.01,
            verbose=False,
        )
        deep_model.fit(df, state="state", action="action", id="agent_id",
                       transitions=env.transition_matrices.numpy(), features=features)
        deep_policy = torch.tensor(deep_model.policy_, dtype=torch.float32)
        deep_evd = _compute_evd(env.true_reward, deep_policy, env.transition_matrices, env.problem_spec)

        # Linear MCE-IRL
        from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
        from econirl.preferences.linear import LinearUtility
        utility = LinearUtility(
            feature_matrix=env.feature_matrix,
            parameter_names=[f"f{i}" for i in range(9)],
        )
        linear_est = MCEIRLEstimator(config=MCEIRLConfig(
            optimizer="L-BFGS-B", inner_solver="hybrid",
            inner_max_iter=5000, inner_tol=1e-8,
            outer_max_iter=500, outer_tol=1e-6,
            compute_se=False, verbose=False,
        ))
        linear_result = linear_est.estimate(panel, utility, env.problem_spec, env.transition_matrices)
        linear_evd = _compute_evd(env.true_reward, linear_result.policy.float(),
                                  env.transition_matrices, env.problem_spec)

        print(f"Binaryworld 8x8: Deep EVD={deep_evd:.2f}, Linear EVD={linear_evd:.2f}")

        # DeepIRL should have lower EVD (better reward recovery)
        assert deep_evd < linear_evd, (
            f"DeepIRL EVD ({deep_evd:.2f}) should be less than "
            f"Linear EVD ({linear_evd:.2f}) on Binaryworld"
        )


@pytest.mark.slow
class TestWulfmeierObjectworld:
    """On Objectworld, both methods should achieve reasonable EVD."""

    def test_deep_runs_on_objectworld(self):
        torch.manual_seed(42)
        env = ObjectworldEnvironment(grid_size=8, feature_type="continuous", seed=42)
        panel = env.simulate_demonstrations(n_demos=32, max_steps=30, seed=0)
        features = env.feature_matrix[:, 0, :].numpy()
        df = _panel_to_df(panel)

        from econirl.estimators.mceirl_neural import MCEIRLNeural
        model = MCEIRLNeural(
            n_states=64, n_actions=5, discount=0.9,
            reward_type="state", reward_hidden_dim=32,
            reward_num_layers=2, max_epochs=100, lr=0.01,
            verbose=False,
        )
        model.fit(df, state="state", action="action", id="agent_id",
                  transitions=env.transition_matrices.numpy(), features=features)

        policy = torch.tensor(model.policy_, dtype=torch.float32)
        evd = _compute_evd(env.true_reward, policy, env.transition_matrices, env.problem_spec)
        print(f"Objectworld 8x8: Deep EVD={evd:.2f}")

        # EVD should be finite and non-negative
        assert 0 <= evd < 100, f"EVD out of range: {evd}"
```

- [ ] **Step 2: Run integration test**

Run: `python3 -m pytest tests/test_wulfmeier_benchmark.py -v -s`
Expected: Both tests PASS, DeepIRL EVD < Linear EVD on Binaryworld

- [ ] **Step 3: Run full test suite for regressions**

Run: `python3 -m pytest tests/test_objectworld.py tests/test_binaryworld.py tests/test_wulfmeier_benchmark.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_wulfmeier_benchmark.py
git commit -m "test: add Wulfmeier benchmark integration tests"
```

---

### Task 6: Final Verification and Push

- [ ] **Step 1: Run the benchmark on small scale**

Run: `python3 -c "
from econirl.environments.objectworld import ObjectworldEnvironment
from econirl.environments.binaryworld import BinaryworldEnvironment
for cls, name in [(ObjectworldEnvironment, 'Objectworld'), (BinaryworldEnvironment, 'Binaryworld')]:
    env = cls(grid_size=8, seed=42)
    panel = env.simulate_demonstrations(n_demos=8, max_steps=30, seed=0)
    print(f'{name}: {env.num_states} states, reward unique={env.true_reward.unique().tolist()}, demos={panel.num_observations} obs')
"`

- [ ] **Step 2: Run existing environment tests for regressions**

Run: `python3 -m pytest tests/test_gridworld.py tests/test_objectworld.py tests/test_binaryworld.py -v`
Expected: All PASS

- [ ] **Step 3: Push**

```bash
git push
```
