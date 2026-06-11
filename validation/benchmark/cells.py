"""Difficulty ladder of synthetic DGP cells for the benchmark suite.

Each :class:`BenchmarkCell` is a named data-generating process with a
``difficulty`` rank (for ladder ordering), a one-line ``stresses`` tag (used by
the failure-mode map), and a ``build()`` that returns a fresh
:class:`~econirl.environments.base.DDCEnvironment`. Cells go from a canonical
recoverable problem to regimes designed to separate estimators.

The ladder deliberately reuses existing environments where one already fits:
RustBus for the canonical recoverable cell, Shapeshifter for the neural-reward
cell, and the new ``random_mdp`` / ``ArrayMDP`` generators for the abstract and
high-dimensional cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from econirl.environments import (
    ArrayMDP,
    RustBusEnvironment,
    ShapeshifterConfig,
    ShapeshifterEnvironment,
    random_mdp,
)
from econirl.environments.base import DDCEnvironment


@dataclass(frozen=True)
class BenchmarkCell:
    """A named synthetic DGP on the difficulty ladder.

    Attributes:
        cell_id: Stable machine identifier.
        label: Human-readable name for tables and figures.
        difficulty: Ladder rank (0 = simplest). Used only for ordering.
        stresses: Short tag naming the DGP condition this cell exercises
            (feeds the failure-mode map).
        description: One-sentence description of the regime.
        builder: Zero-argument callable returning a fresh environment.
        n_individuals: Panel size for simulation.
        n_periods: Periods per individual.
        seed: Base seed for the cell (replications offset from it).
    """

    cell_id: str
    label: str
    difficulty: int
    stresses: str
    description: str
    builder: Callable[[], DDCEnvironment]
    n_individuals: int = 500
    n_periods: int = 80
    seed: int = 42

    def build(self) -> DDCEnvironment:
        return self.builder()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _simple_binary() -> DDCEnvironment:
    """Canonical Rust bus: small, recoverable, every estimator should pass."""
    return RustBusEnvironment(
        num_mileage_bins=20,
        operating_cost=0.01,
        replacement_cost=2.0,
        discount_factor=0.95,
    )


def _stochastic_mid() -> DDCEnvironment:
    """Abstract Garnet MDP, moderate size, genuinely stochastic transitions."""
    return random_mdp(
        num_states=30,
        num_actions=2,
        num_features=3,
        branching=4,
        discount_factor=0.95,
        seed=202,
    )


def _high_beta() -> DDCEnvironment:
    """Near-unit discount: stresses inner-loop accuracy and CI coverage."""
    return random_mdp(
        num_states=20,
        num_actions=2,
        num_features=3,
        branching=4,
        discount_factor=0.99,
        seed=303,
    )


def _neural_reward() -> DDCEnvironment:
    """Nonlinear (frozen MLP) reward: separates linear from neural estimators."""
    return ShapeshifterEnvironment(
        ShapeshifterConfig(
            num_states=24,
            num_actions=3,
            num_features=4,
            reward_type="neural",
            feature_type="linear",
            action_dependent=True,
            stochastic_transitions=True,
            discount_factor=0.95,
            seed=404,
        )
    )


def _large_sparse() -> DDCEnvironment:
    """Large abstract MDP: scaling test (dense densification is the ceiling)."""
    return random_mdp(
        num_states=400,
        num_actions=2,
        num_features=3,
        branching=5,
        discount_factor=0.95,
        seed=505,
    )


def _rank_deficient() -> DDCEnvironment:
    """Collinear features: pre-estimation rank check should flag this cell."""
    S, A = 24, 2
    rng = np.random.default_rng(606)
    T = np.zeros((A, S, S), dtype=np.float64)
    for a in range(A):
        for s in range(S):
            support = rng.choice(S, size=min(4, S), replace=False)
            T[a, s, support] = rng.dirichlet(np.ones(support.shape[0]))
    T /= T.sum(axis=2, keepdims=True)

    grid = np.arange(S, dtype=np.float64) / (S - 1)
    f0 = np.ones(S)
    f1 = grid
    f2 = 2.0 * grid  # deliberately collinear with f1 -> rank deficient
    base = np.stack([f0, f1, f2], axis=1)  # (S, 3)
    phi = np.zeros((S, A, 3), dtype=np.float64)
    phi[:, 1, :] = base  # action 0 is the zeroed outside option
    return ArrayMDP(
        T, phi, theta=np.array([-0.5, 1.0, 0.3]), discount_factor=0.95, seed=606
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


CELLS: tuple[BenchmarkCell, ...] = (
    BenchmarkCell(
        cell_id="simple_binary",
        label="Simple binary (Rust bus)",
        difficulty=0,
        stresses="none (canonical recoverable)",
        description="Small Rust bus engine-replacement DGP; every estimator should recover it.",
        builder=_simple_binary,
    ),
    BenchmarkCell(
        cell_id="stochastic_mid",
        label="Stochastic mid-size",
        difficulty=1,
        stresses="stochastic transitions, moderate state count",
        description="Abstract Garnet MDP with 30 states and genuinely stochastic dynamics.",
        builder=_stochastic_mid,
    ),
    BenchmarkCell(
        cell_id="high_beta",
        label="Near-unit discount",
        difficulty=2,
        stresses="discount near 1 (continuation-value sensitivity, CI coverage)",
        description="Abstract MDP with discount 0.99; stresses inner-loop accuracy and coverage.",
        builder=_high_beta,
    ),
    BenchmarkCell(
        cell_id="neural_reward",
        label="Nonlinear reward",
        difficulty=3,
        stresses="nonlinear reward (linear-utility misspecification)",
        description="Frozen-MLP reward; linear-utility estimators are misspecified here.",
        builder=_neural_reward,
    ),
    BenchmarkCell(
        cell_id="large_sparse",
        label="Larger state space",
        difficulty=4,
        stresses="larger state space (checks the methods stay cheap at scale)",
        description=(
            "400-state abstract MDP. The dense (num_actions, num_states, "
            "num_states) transition tensor that tabular estimators consume is "
            "the real ceiling; it bites in the low thousands of states, not at "
            "400. This cell checks that the structural methods stay cheap here, "
            "not that they break."
        ),
        builder=_large_sparse,
        n_individuals=400,
        n_periods=60,
    ),
    BenchmarkCell(
        cell_id="rank_deficient",
        label="Rank-deficient features",
        difficulty=5,
        stresses="collinear features (rank-deficient design matrix)",
        description="Deliberately collinear features; the pre-estimation rank check should flag it.",
        builder=_rank_deficient,
    ),
)


CELLS_BY_ID: dict[str, BenchmarkCell] = {c.cell_id: c for c in CELLS}


def get_cell(cell_id: str) -> BenchmarkCell:
    """Look up a cell by id, with a helpful error listing valid ids."""
    if cell_id not in CELLS_BY_ID:
        valid = ", ".join(CELLS_BY_ID)
        raise KeyError(f"Unknown cell '{cell_id}'. Valid cells: {valid}.")
    return CELLS_BY_ID[cell_id]
