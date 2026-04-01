"""JAX implementations of NFXP-NK and AIRL kernels for benchmarking.

Ports the core computational kernels from econirl (PyTorch) to JAX,
using lax.while_loop for value iteration and lax.scan for trajectory sampling.
"""

import jax
import jax.numpy as jnp
import jax.scipy
import optax

# Enable float64 for parity with PyTorch structural estimation
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Bellman operator and solvers
# ---------------------------------------------------------------------------

@jax.jit
def bellman_step(transitions, utility, V, beta, sigma):
    """Single soft Bellman operator application.

    Matches SoftBellmanOperator.apply() in bellman.py:80-108.

    Args:
        transitions: (A, S, S) transition matrices
        utility: (S, A) flow utility
        V: (S,) current value function
        beta: discount factor
        sigma: scale parameter

    Returns:
        (V_new, policy) tuple
    """
    EV = jnp.einsum("ast,t->as", transitions, V)  # (A, S)
    Q = utility + beta * EV.T  # (S, A)
    V_new = sigma * jax.scipy.special.logsumexp(Q / sigma, axis=1)
    policy = jax.nn.softmax(Q / sigma, axis=1)
    return V_new, policy


def _vi_body(state, transitions, utility, beta, sigma):
    """Body function for value iteration while_loop."""
    V, _, k = state
    V_new, policy = bellman_step(transitions, utility, V, beta, sigma)
    error = jnp.max(jnp.abs(V_new - V))
    return V_new, error, k + 1


def _vi_cond(state, tol, max_iter):
    """Condition function for value iteration while_loop."""
    _, error, k = state
    return jnp.logical_and(error > tol, k < max_iter)


def value_iteration(transitions, utility, V_init, beta, sigma, tol=1e-10,
                    max_iter=100000):
    """Value iteration using jax.lax.while_loop.

    The entire convergence loop is compiled to a single XLA program,
    eliminating Python dispatch overhead per iteration.

    Returns:
        (V, policy, n_iter) tuple
    """
    init_state = (V_init, jnp.float64(tol + 1.0), jnp.int32(0))

    def body(state):
        return _vi_body(state, transitions, utility, beta, sigma)

    def cond(state):
        return _vi_cond(state, tol, max_iter)

    V, _, n_iter = jax.lax.while_loop(cond, body, init_state)

    # Final policy
    _, policy = bellman_step(transitions, utility, V, beta, sigma)
    return V, policy, n_iter


# JIT-compiled version with static tol/max_iter
_value_iteration_jit = jax.jit(value_iteration, static_argnames=["tol", "max_iter"])


@jax.jit
def nk_step(transitions, utility, V, beta, sigma):
    """Single Newton-Kantorovich step.

    Matches _newton_kantorovich_step() in solvers.py:297-350.

    Returns:
        (V_new, post_residual_norm) tuple
    """
    n_states = V.shape[0]

    # Apply Bellman to get T(V)
    V_bellman, policy = bellman_step(transitions, utility, V, beta, sigma)
    residual = V_bellman - V

    # Policy-weighted transition: P_pi[s,s'] = sum_a pi(a|s) P(s'|s,a)
    P_pi = jnp.einsum("sa,ast->st", policy, transitions)

    # Solve (I - beta*P_pi) @ delta = residual
    A = jnp.eye(n_states) - beta * P_pi
    delta = jnp.linalg.solve(A, residual)

    V_new = V + delta

    # Post-update residual
    V_check, _ = bellman_step(transitions, utility, V_new, beta, sigma)
    post_residual = jnp.max(jnp.abs(V_check - V_new))

    return V_new, post_residual


def hybrid_iteration(transitions, utility, V_init, beta, sigma,
                     tol=1e-10, max_iter=100000, switch_tol=1e-3,
                     max_nk_iter=20):
    """Hybrid SA + NK solver.

    Matches hybrid_iteration() in solvers.py:353-456.
    Phase 1 uses lax.while_loop for contraction, Phase 2 uses NK steps.

    Returns:
        (V, policy, n_iter) tuple
    """
    # Phase 1: Contraction iterations via while_loop
    def sa_body(state):
        V, _, k = state
        V_new, _ = bellman_step(transitions, utility, V, beta, sigma)
        error = jnp.max(jnp.abs(V_new - V))
        return V_new, error, k + 1

    def sa_cond(state):
        _, error, k = state
        return jnp.logical_and(error > switch_tol, k < max_iter)

    init_state = (V_init, jnp.float64(switch_tol + 1.0), jnp.int32(0))
    V, sa_error, sa_iters = jax.lax.while_loop(sa_cond, sa_body, init_state)

    # Phase 2: NK iterations (unrolled, small fixed count)
    def nk_body(state):
        V, error, k = state
        V_new, post_error = nk_step(transitions, utility, V, beta, sigma)
        return V_new, post_error, k + 1

    def nk_cond(state):
        _, error, k = state
        return jnp.logical_and(error > tol, k < max_nk_iter)

    V, nk_error, nk_iters = jax.lax.while_loop(
        nk_cond, nk_body, (V, sa_error, jnp.int32(0))
    )

    _, policy = bellman_step(transitions, utility, V, beta, sigma)
    return V, policy, sa_iters + nk_iters


_hybrid_iteration_jit = jax.jit(
    hybrid_iteration, static_argnames=["tol", "max_iter", "switch_tol", "max_nk_iter"]
)


# ---------------------------------------------------------------------------
# NFXP implicit differentiation and BHHH
# ---------------------------------------------------------------------------

@jax.jit
def compute_scores(transitions, features, V, policy, beta, sigma,
                   obs_states, obs_actions):
    """Compute per-observation scores via implicit differentiation.

    Matches NFXPEstimator._compute_analytical_score() in nfxp.py:228-312.

    Args:
        transitions: (A, S, S)
        features: (S, A, K) feature matrix
        V: (S,) converged value function
        policy: (S, A) choice probabilities
        beta: discount factor
        sigma: scale parameter
        obs_states: (N,) int array of observed states
        obs_actions: (N,) int array of observed actions

    Returns:
        scores: (N, K) per-observation score matrix
    """
    n_states = V.shape[0]

    # F = I - beta * P_pi
    P_pi = jnp.einsum("sa,ast->st", policy, transitions)
    F = jnp.eye(n_states) - beta * P_pi

    # dT/dtheta[s,k] = sum_a pi(a|s) * phi(s,a,k)
    dT_dtheta = jnp.einsum("sa,sak->sk", policy, features)

    # Solve F @ dV/dtheta = dT/dtheta
    dV_dtheta = jnp.linalg.solve(F, dT_dtheta)

    # dQ/dtheta[s,a,k] = phi(s,a,k) + beta * sum_s' P(s'|s,a) * dV(s')/dtheta_k
    EV_deriv = jnp.einsum("ast,tk->ask", transitions, dV_dtheta)  # (A, S, K)
    dQ_dtheta = features + beta * jnp.transpose(EV_deriv, (1, 0, 2))  # (S, A, K)

    # E_pi[dQ] = sum_a pi(a|s) * dQ(s,a,k)
    E_dQ = jnp.einsum("sa,sak->sk", policy, dQ_dtheta)

    # Per-observation score
    dQ_obs = dQ_dtheta[obs_states, obs_actions]  # (N, K)
    E_dQ_obs = E_dQ[obs_states]  # (N, K)
    scores = (1.0 / sigma) * (dQ_obs - E_dQ_obs)

    return scores


@jax.jit
def compute_log_choice_probs(transitions, utility, V, beta, sigma):
    """Compute log P(a|s) for all state-action pairs.

    Returns:
        (S, A) array of log choice probabilities
    """
    EV = jnp.einsum("ast,t->as", transitions, V)
    Q = utility + beta * EV.T
    return jax.nn.log_softmax(Q / sigma, axis=1)


def nfxp_estimate(initial_params, transitions, features, beta, sigma,
                  obs_states, obs_actions, outer_tol=1e-6, outer_max_iter=200,
                  inner_tol=1e-10, inner_max_iter=100000, switch_tol=1e-3):
    """Full NFXP-NK estimation with BHHH optimizer.

    Python loop for outer BHHH (variable-length line search),
    calling JIT-compiled inner functions.

    Returns:
        (params, ll, n_outer_iter, converged) tuple
    """
    n_states = transitions.shape[1]
    n_params = features.shape[2]
    params = initial_params.copy()
    prev_ll = -jnp.inf

    for iteration in range(outer_max_iter):
        # Compute utility
        utility = jnp.einsum("sak,k->sa", features, params)

        # Solve inner problem
        V_init = jnp.zeros(n_states)
        V, policy, _ = _hybrid_iteration_jit(
            transitions, utility, V_init, beta, sigma,
            tol=inner_tol, max_iter=inner_max_iter, switch_tol=switch_tol,
        )

        # Scores and log-likelihood
        scores = compute_scores(
            transitions, features, V, policy, beta, sigma,
            obs_states, obs_actions,
        )
        log_probs = compute_log_choice_probs(transitions, utility, V, beta, sigma)
        ll = float(log_probs[obs_states, obs_actions].sum())

        # Gradient
        grad = scores.sum(axis=0)
        grad_norm = float(jnp.abs(grad).max())
        ll_change = abs(ll - prev_ll) if prev_ll > -jnp.inf else float("inf")

        if grad_norm < outer_tol or (iteration > 10 and ll_change < 1e-10):
            return params, ll, iteration + 1, True

        prev_ll = ll

        # BHHH Hessian and direction
        H_bhhh = scores.T @ scores + 1e-8 * jnp.eye(n_params)
        direction = jnp.linalg.solve(H_bhhh, grad)

        # Step-halving line search
        step_size = 1.0
        for _ in range(15):
            new_params = params + step_size * direction
            new_utility = jnp.einsum("sak,k->sa", features, new_params)
            V_new, _, _ = _hybrid_iteration_jit(
                transitions, new_utility, V_init, beta, sigma,
                tol=inner_tol, max_iter=inner_max_iter, switch_tol=switch_tol,
            )
            new_log_probs = compute_log_choice_probs(
                transitions, new_utility, V_new, beta, sigma
            )
            new_ll = float(new_log_probs[obs_states, obs_actions].sum())
            if new_ll > ll:
                break
            step_size *= 0.5

        params = new_params

    return params, ll, outer_max_iter, False


# ---------------------------------------------------------------------------
# AIRL kernels
# ---------------------------------------------------------------------------

def sample_trajectory_scan(key, policy, transitions, n_steps, initial_state=0):
    """Sample trajectory using jax.lax.scan (compiled to single XLA kernel).

    Replaces the Python for-loop in airl.py:256-265.

    Returns:
        (states, actions, next_states) each of shape (n_steps,)
    """
    n_states = policy.shape[0]

    def step_fn(carry, _):
        state, key = carry
        key, k1, k2 = jax.random.split(key, 3)

        # Sample action from policy
        action = jax.random.categorical(k1, jnp.log(policy[state] + 1e-10))

        # Sample next state from transition
        next_state = jax.random.categorical(
            k2, jnp.log(transitions[action, state] + 1e-10)
        )

        # Cast to int32 to match carry dtype
        return (next_state.astype(jnp.int32), key), (state, action.astype(jnp.int32), next_state.astype(jnp.int32))

    init_carry = (jnp.int32(initial_state), key)
    _, (states, actions, next_states) = jax.lax.scan(
        step_fn, init_carry, None, length=n_steps
    )

    return states, actions, next_states


_sample_trajectory_jit = jax.jit(
    sample_trajectory_scan, static_argnames=["n_steps", "initial_state"]
)


def airl_discriminator_loss(reward_params, features, V, policy,
                            expert_states, expert_actions, expert_next_states,
                            policy_states, policy_actions, policy_next_states,
                            beta, use_shaping=True):
    """AIRL discriminator BCE loss, differentiable w.r.t. reward_params.

    Matches the inner loop of airl.py:454-489.
    """
    reward_matrix = jnp.einsum("sak,k->sa", features, reward_params)

    def logits(states, actions, next_states):
        r_sa = reward_matrix[states, actions]
        if use_shaping:
            f = r_sa + beta * V[next_states] - V[states]
        else:
            f = r_sa
        log_pi = jnp.log(policy[states, actions] + 1e-10)
        return f - log_pi

    expert_logits = logits(expert_states, expert_actions, expert_next_states)
    policy_logits = logits(policy_states, policy_actions, policy_next_states)

    # BCE loss
    expert_loss = jnp.mean(
        jnp.logaddexp(0.0, -expert_logits)  # -log sigmoid(x) = log(1+exp(-x))
    )
    policy_loss = jnp.mean(
        jnp.logaddexp(0.0, policy_logits)  # -log(1-sigmoid(x)) = log(1+exp(x))
    )
    return expert_loss + policy_loss


_airl_disc_loss_and_grad = jax.jit(
    jax.value_and_grad(airl_discriminator_loss),
    static_argnames=["use_shaping"],
)


def airl_estimate(key, transitions, features, expert_states, expert_actions,
                  expert_next_states, beta, sigma, max_rounds=20,
                  discriminator_steps=5, reward_lr=0.01, convergence_tol=1e-4,
                  inner_tol=1e-8, inner_max_iter=5000, use_shaping=True):
    """Full AIRL training loop.

    Returns:
        (reward_params, policy, V, n_rounds, per_round_times) tuple
    """
    import time

    n_states = transitions.shape[1]
    n_features = features.shape[2]
    n_expert = len(expert_states)

    reward_params = jnp.zeros(n_features)
    optimizer = optax.adam(reward_lr)
    opt_state = optimizer.init(reward_params)

    policy = jnp.ones((n_states, 2)) / 2.0
    V = jnp.zeros(n_states)

    sampling_times = []
    vi_times = []
    disc_times = []

    for round_idx in range(max_rounds):
        old_policy = policy

        # Sample from current policy
        key, subkey = jax.random.split(key)
        t0 = time.perf_counter()
        pol_states, pol_actions, pol_next_states = _sample_trajectory_jit(
            subkey, policy, transitions, n_steps=n_expert,
        )
        # Force completion
        pol_states.block_until_ready()
        sampling_times.append(time.perf_counter() - t0)

        # Discriminator updates
        t0 = time.perf_counter()
        for _ in range(discriminator_steps):
            loss, grads = _airl_disc_loss_and_grad(
                reward_params, features, V, policy,
                expert_states, expert_actions, expert_next_states,
                pol_states, pol_actions, pol_next_states,
                beta, use_shaping=use_shaping,
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            reward_params = optax.apply_updates(reward_params, updates)
        # Force completion
        reward_params.block_until_ready()
        disc_times.append(time.perf_counter() - t0)

        # Policy update via value iteration
        reward_matrix = jnp.einsum("sak,k->sa", features, reward_params)
        t0 = time.perf_counter()
        V, policy, _ = _value_iteration_jit(
            transitions, reward_matrix, jnp.zeros(n_states), beta, sigma,
            tol=inner_tol, max_iter=inner_max_iter,
        )
        policy.block_until_ready()
        vi_times.append(time.perf_counter() - t0)

        # Check convergence
        policy_change = float(jnp.max(jnp.abs(policy - old_policy)))
        if policy_change < convergence_tol:
            break

    per_round_times = {
        "sampling": sampling_times,
        "discriminator": disc_times,
        "value_iteration": vi_times,
    }
    return reward_params, policy, V, round_idx + 1, per_round_times
