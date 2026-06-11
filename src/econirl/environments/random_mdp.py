"""Garnet-style random MDP generator for abstract benchmarking.

``random_mdp`` builds an abstract :class:`ArrayMDP` with sparse stochastic
transitions and linear reward, parameterized the way the Garnet testbench is:
by the number of states, the number of actions, and a branching factor that
controls how many next-states each state-action pair can reach. This is the
standard way to generate abstract MDPs that are representative of the structure
estimators face in practice while being cheap to construct at scale.

Reference: Bhatnagar et al, "Adaptive Bases for Reinforcement Learning"
(https://arxiv.org/pdf/1005.0125), which introduced the GARNET family.

Honest note on scale: the returned environment stores a dense
``(A, S, S)`` transition tensor because the tabular estimators consume dense
transitions. The generator builds large MDPs cheaply, but how far each
estimator runs is bounded by that dense representation and is reported as part
of the benchmark failure map, not promised here.
"""

from __future__ import annotations

import numpy as np

from econirl.environments.array_mdp import ArrayMDP


def _build_sparse_transitions(
    num_states: int,
    num_actions: int,
    branching: int,
    self_loop: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build an ``(A, S, S)`` sparse stochastic tensor (Garnet rows).

    Each row places a Dirichlet draw on a random subset of ``branching``
    next-states. An optional ``self_loop`` mass on the current state guarantees
    aperiodicity, mirroring the standard Garnet construction.
    """
    b = int(min(branching, num_states))
    T = np.zeros((num_actions, num_states, num_states), dtype=np.float64)
    for a in range(num_actions):
        for s in range(num_states):
            support = rng.choice(num_states, size=b, replace=False)
            weights = rng.dirichlet(np.ones(b))
            T[a, s, support] = weights
            if self_loop > 0.0:
                T[a, s] *= 1.0 - self_loop
                T[a, s, s] += self_loop
    # Renormalize defensively for float drift.
    T /= T.sum(axis=2, keepdims=True)
    return T


def _build_polynomial_features(
    num_states: int,
    num_actions: int,
    num_features: int,
    action_dependent: bool,
) -> np.ndarray:
    """Polynomial features in the normalized state index, shape ``(S, A, K)``.

    Feature 0 is a bias; the rest are increasing powers of ``s/(S-1)``. When
    ``action_dependent`` is set, action 0 is the normalized outside option
    (zero features) and each other action adds an action-specific shift to the
    last feature, giving finite-theta estimators an identified gauge (the same
    convention Shapeshifter uses).
    """
    denom = max(num_states - 1, 1)
    grid = np.arange(num_states, dtype=np.float64) / denom  # (S,)

    base = np.zeros((num_states, num_features), dtype=np.float64)
    base[:, 0] = 1.0
    for k in range(1, num_features):
        base[:, k] = grid ** k

    feats = np.zeros((num_states, num_actions, num_features), dtype=np.float64)
    if action_dependent:
        for a in range(1, num_actions):
            feats[:, a, :] = base
            feats[:, a, -1] += float(a)
    else:
        feats[:] = base[:, None, :]
    return feats


def random_mdp(
    num_states: int = 30,
    num_actions: int = 3,
    num_features: int = 3,
    branching: int = 4,
    discount_factor: float = 0.95,
    reward_scale: float = 1.0,
    self_loop: float = 0.05,
    action_dependent: bool = True,
    scale_parameter: float = 1.0,
    seed: int = 0,
) -> ArrayMDP:
    """Construct a Garnet-style random :class:`ArrayMDP`.

    Args:
        num_states: Number of discrete states ``S``.
        num_actions: Number of discrete actions ``A``.
        num_features: Number of linear reward features ``K``.
        branching: Number of reachable next-states per state-action pair.
        discount_factor: Time discount ``beta`` in ``[0, 1)``.
        reward_scale: Scale of the random linear reward parameters.
        self_loop: Probability mass placed on staying in the current state
            (guarantees aperiodicity). Set to 0 to disable.
        action_dependent: Whether features vary across actions.
        scale_parameter: Logit scale ``sigma``.
        seed: Random seed. Same seed reproduces the same MDP exactly.

    Returns:
        An :class:`ArrayMDP` with sparse transitions and linear reward.
    """
    rng = np.random.default_rng(seed)

    transitions = _build_sparse_transitions(
        num_states, num_actions, branching, self_loop, rng
    )
    features = _build_polynomial_features(
        num_states, num_actions, num_features, action_dependent
    )
    theta = rng.normal(size=num_features) * 0.5 * reward_scale
    names = [f"theta_{i}" for i in range(num_features)]

    return ArrayMDP(
        transitions=transitions,
        features=features,
        theta=theta,
        discount_factor=discount_factor,
        scale_parameter=scale_parameter,
        parameter_names=names,
        seed=seed,
    )
