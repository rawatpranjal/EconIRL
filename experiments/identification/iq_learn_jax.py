"""JAX-native IQ-Learn for tabular MDPs.

Implements the chi-squared IQ-Learn objective from Garg et al. (2021)
using JAX and optax. This avoids the PyTorch dependency in the main
contrib/iq_learn.py and stays consistent with the JAX-based codebase.

The chi-squared offline objective (Eq. 12 in Garg et al.):
    min_Q  -E_expert[Q(s,a) - V*(s)] + (1/4*alpha) * E_expert[td^2]
where:
    V*(s) = sigma * logsumexp(Q(s,:) / sigma)
    td(s,a) = Q(s,a) - gamma * sum_{s'} P(s'|s,a) V*(s')

At the optimum, Q* satisfies the soft Bellman equation and the
reward can be recovered via the inverse Bellman operator:
    r(s,a) = Q(s,a) - gamma * sum_{s'} P(s'|s,a) V*(s')
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass
class IQLearnJAXConfig:
    """Configuration for JAX IQ-Learn."""

    alpha: float = 1.0
    learning_rate: float = 0.005
    max_iter: int = 5000
    convergence_tol: float = 1e-6
    verbose: bool = False


def estimate_iq_learn(
    expert_states: jnp.ndarray,
    expert_actions: jnp.ndarray,
    expert_next_states: jnp.ndarray,
    transitions: jnp.ndarray,
    n_states: int,
    n_actions: int,
    gamma: float,
    sigma: float = 1.0,
    config: IQLearnJAXConfig | None = None,
) -> dict:
    """Estimate Q-function via IQ-Learn chi-squared objective.

    Parameters
    ----------
    expert_states, expert_actions, expert_next_states
        Expert demonstration data as integer index arrays.
    transitions
        Transition matrices, shape (n_actions, n_states, n_states).
    n_states, n_actions
        State and action space sizes.
    gamma
        Discount factor.
    sigma
        Logit scale parameter.
    config
        Optimization hyperparameters.

    Returns
    -------
    dict with keys: q_table, reward_matrix, policy, value_function,
        converged, num_iterations, log_likelihood
    """
    if config is None:
        config = IQLearnJAXConfig()

    alpha = config.alpha

    # Initialize Q(s,a) as flat array for optax
    q_params = jnp.zeros(n_states * n_actions)
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(q_params)

    # Precompute EV matrix: EV[s, a] = sum_{s'} P(s'|s,a) V*(s')
    # This depends on V* which depends on Q, so we compute it inside the loss.
    # But we can precompute the transition matmul structure.

    def loss_fn(q_flat):
        Q = q_flat.reshape(n_states, n_actions)

        # V*(s) = sigma * logsumexp(Q(s,:) / sigma)
        V_star = sigma * jax.nn.logsumexp(Q / sigma, axis=1)

        # E[V*(s')] for each (s, a): sum_{s'} P(s'|s,a) V*(s')
        # transitions: (A, S, S'), V_star: (S',) -> EV: (A, S)
        EV = jnp.einsum("ast,t->as", transitions, V_star)  # (A, S)
        EV = EV.T  # (S, A)

        # Temporal difference
        td = Q - gamma * EV

        # Expert terms
        Q_expert = Q[expert_states, expert_actions]
        V_expert = V_star[expert_states]
        td_expert = td[expert_states, expert_actions]

        # Chi-squared objective:
        # min -E[Q(s,a) - V*(s)] + (1/4*alpha) E[td^2]
        loss = -(Q_expert - V_expert).mean() + (1.0 / (4.0 * alpha)) * (td_expert ** 2).mean()
        return loss

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Optimization loop
    converged = False
    best_loss = float("inf")
    for iteration in range(config.max_iter):
        loss_val, grads = loss_and_grad(q_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        q_params = optax.apply_updates(q_params, updates)

        loss_float = float(loss_val)
        if loss_float < best_loss:
            best_loss = loss_float

        grad_norm = float(jnp.linalg.norm(grads))
        if grad_norm < config.convergence_tol:
            converged = True
            break

        if config.verbose and iteration % 500 == 0:
            print(f"  IQ-Learn iter {iteration}: loss={loss_float:.6f}, grad_norm={grad_norm:.6f}")

    # Extract results
    Q_table = q_params.reshape(n_states, n_actions)
    V_star = sigma * jax.nn.logsumexp(Q_table / sigma, axis=1)
    policy = jax.nn.softmax(Q_table / sigma, axis=1)

    # Reward via inverse Bellman: r(s,a) = Q(s,a) - gamma * E[V*(s')]
    EV = jnp.einsum("ast,t->as", transitions, V_star).T
    reward_matrix = Q_table - gamma * EV

    # Log-likelihood
    log_probs = jax.nn.log_softmax(Q_table / sigma, axis=1)
    ll = float(log_probs[expert_states, expert_actions].sum())

    return {
        "q_table": Q_table,
        "reward_matrix": reward_matrix,
        "policy": policy,
        "value_function": V_star,
        "log_likelihood": ll,
        "converged": converged,
        "num_iterations": iteration + 1,
        "final_loss": best_loss,
    }
