"""Pure-numpy problem generator for JAX vs PyTorch benchmarks.

Generates Rust-bus-style DDC problems at various state-space sizes.
No torch or jax dependency — outputs are numpy arrays consumed by both frameworks.
"""

import numpy as np
from scipy.special import logsumexp


# Problem sizes to benchmark: (name, n_states)
PROBLEM_SIZES = [
    ("tiny", 20),
    ("rust_bus", 90),
    ("medium", 250),
    ("large", 500),
    ("xlarge", 1000),
]

# Rust (1987) mileage transition probabilities
DEFAULT_PROBS = (0.3919, 0.5953, 0.0128)

# True parameters for data generation
TRUE_PARAMS = np.array([0.001, 3.0], dtype=np.float64)

# Defaults
DEFAULT_BETA = 0.9999
DEFAULT_SIGMA = 1.0


def make_transitions(n_states, probs=DEFAULT_PROBS):
    """Build Rust bus transition matrices P(s'|s,a).

    Matches RustBusEnvironment._build_transition_matrices().

    Returns:
        np.ndarray of shape (2, n_states, n_states) with dtype float64.
        transitions[0] = keep action, transitions[1] = replace action.
    """
    T = np.zeros((2, n_states, n_states), dtype=np.float64)

    # Keep action: mileage increases by 0, 1, or 2
    for s in range(n_states):
        for delta, prob in enumerate(probs):
            next_s = min(s + delta, n_states - 1)
            T[0, s, next_s] += prob

    # Replace action: reset to 0, then increase by 0, 1, or 2
    for delta, prob in enumerate(probs):
        next_s = min(delta, n_states - 1)
        T[1, :, next_s] = prob

    return T


def make_features(n_states):
    """Build feature matrix phi(s,a,k).

    Matches RustBusEnvironment._build_feature_matrix().

    Returns:
        np.ndarray of shape (n_states, 2, 2) with dtype float64.
        features[:, 0, 0] = -mileage (keep action, operating cost feature)
        features[:, 1, 1] = -1 (replace action, replacement cost feature)
    """
    features = np.zeros((n_states, 2, 2), dtype=np.float64)
    mileage = np.arange(n_states, dtype=np.float64)
    features[:, 0, 0] = -mileage
    features[:, 1, 1] = -1.0
    return features


def compute_utility(params, features):
    """Compute flow utility U(s,a) = sum_k theta_k * phi(s,a,k).

    Returns:
        np.ndarray of shape (n_states, n_actions).
    """
    return np.einsum("sak,k->sa", features, params)


def solve_true_policy(transitions, features, params, beta=DEFAULT_BETA,
                      sigma=DEFAULT_SIGMA, tol=1e-10, max_iter=100000):
    """Solve for the optimal policy via value iteration in pure numpy.

    Returns:
        policy: np.ndarray of shape (n_states, n_actions), choice probabilities.
        V: np.ndarray of shape (n_states,), converged value function.
    """
    n_states = transitions.shape[1]
    utility = compute_utility(params, features)
    V = np.zeros(n_states, dtype=np.float64)

    for _ in range(max_iter):
        EV = np.einsum("ast,t->as", transitions, V)  # (2, S)
        Q = utility + beta * EV.T  # (S, 2)
        V_new = sigma * logsumexp(Q / sigma, axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    # Final policy
    EV = np.einsum("ast,t->as", transitions, V)
    Q = utility + beta * EV.T
    Q_scaled = Q / sigma
    Q_scaled -= Q_scaled.max(axis=1, keepdims=True)  # numerical stability
    exp_Q = np.exp(Q_scaled)
    policy = exp_Q / exp_Q.sum(axis=1, keepdims=True)

    return policy, V


def simulate_panel(policy, transitions, n_individuals=200, n_periods=100,
                   seed=42):
    """Simulate panel data from a given policy.

    Returns:
        dict with keys:
            'states': int64 array of shape (n_obs,)
            'actions': int64 array of shape (n_obs,)
            'next_states': int64 array of shape (n_obs,)
            'n_individuals': int
            'n_periods': int
    """
    rng = np.random.RandomState(seed)
    n_states = policy.shape[0]
    n_actions = policy.shape[1]

    all_states = []
    all_actions = []
    all_next_states = []

    for _ in range(n_individuals):
        state = 0  # start at mileage 0
        for _ in range(n_periods):
            action = rng.choice(n_actions, p=policy[state])
            next_state = rng.choice(n_states, p=transitions[action, state])
            all_states.append(state)
            all_actions.append(action)
            all_next_states.append(next_state)
            state = next_state

    return {
        "states": np.array(all_states, dtype=np.int64),
        "actions": np.array(all_actions, dtype=np.int64),
        "next_states": np.array(all_next_states, dtype=np.int64),
        "n_individuals": n_individuals,
        "n_periods": n_periods,
    }


def generate_problem(n_states, params=TRUE_PARAMS, beta=DEFAULT_BETA,
                     sigma=DEFAULT_SIGMA, n_individuals=200, n_periods=100,
                     seed=42):
    """Generate a complete benchmark problem.

    Returns:
        dict with keys: transitions, features, utility, policy, V, panel,
        params, beta, sigma, n_states.
    """
    transitions = make_transitions(n_states)
    features = make_features(n_states)
    utility = compute_utility(params, features)
    policy, V = solve_true_policy(transitions, features, params, beta, sigma)
    panel = simulate_panel(policy, transitions, n_individuals, n_periods, seed)

    return {
        "transitions": transitions,
        "features": features,
        "utility": utility,
        "policy": policy,
        "V": V,
        "panel": panel,
        "params": params,
        "beta": beta,
        "sigma": sigma,
        "n_states": n_states,
    }
