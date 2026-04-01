"""PyTorch vs JAX speed benchmark for NFXP-NK and AIRL.

Runs kernel micro-benchmarks and end-to-end estimation at multiple problem sizes.
Both frameworks run on CPU.

Usage:
    python benchmarks/jax_comparison/run_benchmark.py
    python benchmarks/jax_comparison/run_benchmark.py --sizes 90 250
    python benchmarks/jax_comparison/run_benchmark.py --skip-airl
"""

import argparse
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import torch

jax.config.update("jax_enable_x64", True)

from benchmarks.jax_comparison.jax_kernels import (
    _hybrid_iteration_jit,
    _sample_trajectory_jit,
    _value_iteration_jit,
    airl_estimate,
    bellman_step,
    compute_log_choice_probs,
    compute_scores,
    nfxp_estimate,
    nk_step,
)
from benchmarks.jax_comparison.problem_setup import generate_problem


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_jax(fn, *args, n_warmup=5, n_runs=20, **kwargs):
    """Time a JAX function, handling JIT warmup and block_until_ready."""
    # JIT warmup (first call compiles)
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    elif isinstance(result, tuple):
        for r in result:
            if hasattr(r, "block_until_ready"):
                r.block_until_ready()
    jit_time = time.perf_counter() - t0

    # Additional warmup
    for _ in range(n_warmup - 1):
        result = fn(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, tuple):
            for r in result:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, tuple):
            for r in result:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()
        times.append(time.perf_counter() - t0)

    return {"jit_time": jit_time, "median": median(times), "times": times}


def time_torch(fn, *args, n_warmup=5, n_runs=20, **kwargs):
    """Time a PyTorch function."""
    # Warmup
    for _ in range(n_warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    return {"median": median(times), "times": times}


def fmt_ms(seconds):
    """Format seconds as milliseconds string."""
    return f"{seconds * 1000:.2f}ms"


def fmt_speedup(torch_time, jax_time):
    """Format speedup ratio."""
    if jax_time == 0:
        return "inf"
    ratio = torch_time / jax_time
    return f"{ratio:.1f}x"


# ---------------------------------------------------------------------------
# PyTorch kernels (using econirl internals)
# ---------------------------------------------------------------------------

def torch_bellman_step(transitions_t, utility_t, V_t, beta, sigma):
    """Single Bellman step via econirl."""
    EV = torch.einsum("ast,t->as", transitions_t, V_t)
    Q = utility_t + beta * EV.T
    V_new = sigma * torch.logsumexp(Q / sigma, dim=1)
    policy = torch.nn.functional.softmax(Q / sigma, dim=1)
    return V_new, policy


def torch_value_iteration(transitions_t, utility_t, V_init_t, beta, sigma,
                          tol=1e-10, max_iter=100000):
    """Value iteration in PyTorch (Python for-loop)."""
    V = V_init_t.clone()
    for k in range(max_iter):
        V_new, policy = torch_bellman_step(transitions_t, utility_t, V, beta, sigma)
        error = torch.abs(V_new - V).max().item()
        V = V_new
        if error < tol:
            break
    _, policy = torch_bellman_step(transitions_t, utility_t, V, beta, sigma)
    return V, policy, k + 1


def torch_nk_step(transitions_t, utility_t, V_t, beta, sigma):
    """Single NK step in PyTorch."""
    n = V_t.shape[0]
    V_bell, policy = torch_bellman_step(transitions_t, utility_t, V_t, beta, sigma)
    residual = V_bell - V_t
    P_pi = torch.einsum("sa,ast->st", policy, transitions_t)
    A = torch.eye(n, dtype=V_t.dtype) - beta * P_pi
    delta = torch.linalg.solve(A, residual)
    V_new = V_t + delta
    V_check, _ = torch_bellman_step(transitions_t, utility_t, V_new, beta, sigma)
    post_err = torch.abs(V_check - V_new).max().item()
    return V_new, post_err


def torch_hybrid_iteration(transitions_t, utility_t, V_init_t, beta, sigma,
                           tol=1e-10, max_iter=100000, switch_tol=1e-3,
                           max_nk_iter=20):
    """Hybrid SA+NK in PyTorch (Python for-loop)."""
    V = V_init_t.clone()
    for k in range(max_iter):
        V_new, _ = torch_bellman_step(transitions_t, utility_t, V, beta, sigma)
        error = torch.abs(V_new - V).max().item()
        V = V_new
        if error < tol:
            break
        if error < switch_tol:
            break

    if error >= tol and error < switch_tol:
        for nk in range(max_nk_iter):
            V, post_err = torch_nk_step(transitions_t, utility_t, V, beta, sigma)
            if post_err < tol:
                break

    _, policy = torch_bellman_step(transitions_t, utility_t, V, beta, sigma)
    return V, policy


def torch_compute_scores(transitions_t, features_t, V_t, policy_t, beta, sigma,
                         obs_states_t, obs_actions_t):
    """Implicit differentiation scores in PyTorch."""
    n = V_t.shape[0]
    P_pi = torch.einsum("sa,ast->st", policy_t, transitions_t)
    F = torch.eye(n, dtype=V_t.dtype) - beta * P_pi
    dT = torch.einsum("sa,sak->sk", policy_t, features_t)
    dV = torch.linalg.solve(F, dT)
    EV_d = torch.einsum("ast,tk->ask", transitions_t, dV)
    dQ = features_t + beta * EV_d.permute(1, 0, 2)
    E_dQ = torch.einsum("sa,sak->sk", policy_t, dQ)
    dQ_obs = dQ[obs_states_t, obs_actions_t]
    E_dQ_obs = E_dQ[obs_states_t]
    return (1.0 / sigma) * (dQ_obs - E_dQ_obs)


def torch_sample_trajectory(policy_t, transitions_t, n_steps, initial_state=0):
    """Sequential trajectory sampling in PyTorch (Python for-loop)."""
    states, actions, next_states_list = [], [], []
    state = initial_state
    for _ in range(n_steps):
        action = torch.multinomial(policy_t[state], 1).item()
        next_state = torch.multinomial(transitions_t[action, state], 1).item()
        states.append(state)
        actions.append(action)
        next_states_list.append(next_state)
        state = next_state
    return (
        torch.tensor(states, dtype=torch.long),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(next_states_list, dtype=torch.long),
    )


def torch_nfxp_estimate(transitions_t, features_t, initial_params_t, beta, sigma,
                        obs_states_t, obs_actions_t, outer_tol=1e-6,
                        outer_max_iter=200, inner_tol=1e-10, inner_max_iter=100000,
                        switch_tol=1e-3):
    """Full NFXP-NK estimation in PyTorch."""
    n_states = transitions_t.shape[1]
    n_params = features_t.shape[2]
    params = initial_params_t.clone().double()
    prev_ll = -float("inf")

    for iteration in range(outer_max_iter):
        utility = torch.einsum("sak,k->sa", features_t, params)
        V_init = torch.zeros(n_states, dtype=transitions_t.dtype)
        V, policy = torch_hybrid_iteration(
            transitions_t, utility, V_init, beta, sigma,
            tol=inner_tol, max_iter=inner_max_iter, switch_tol=switch_tol,
        )

        scores = torch_compute_scores(
            transitions_t, features_t, V, policy, beta, sigma,
            obs_states_t, obs_actions_t,
        )
        EV = torch.einsum("ast,t->as", transitions_t, V)
        Q = utility + beta * EV.T
        log_probs = torch.nn.functional.log_softmax(Q / sigma, dim=1)
        ll = log_probs[obs_states_t, obs_actions_t].sum().item()

        grad = scores.sum(dim=0)
        grad_norm = grad.abs().max().item()
        ll_change = abs(ll - prev_ll) if prev_ll > -float("inf") else float("inf")

        if grad_norm < outer_tol or (iteration > 10 and ll_change < 1e-10):
            return params.numpy(), ll, iteration + 1, True

        prev_ll = ll

        H = scores.T @ scores + 1e-8 * torch.eye(n_params, dtype=scores.dtype)
        direction = torch.linalg.solve(H, grad)

        step_size = 1.0
        for _ in range(15):
            new_params = params + step_size * direction
            new_u = torch.einsum("sak,k->sa", features_t, new_params)
            V_new, _ = torch_hybrid_iteration(
                transitions_t, new_u, V_init, beta, sigma,
                tol=inner_tol, max_iter=inner_max_iter, switch_tol=switch_tol,
            )
            EV_new = torch.einsum("ast,t->as", transitions_t, V_new)
            Q_new = new_u + beta * EV_new.T
            lp_new = torch.nn.functional.log_softmax(Q_new / sigma, dim=1)
            new_ll = lp_new[obs_states_t, obs_actions_t].sum().item()
            if new_ll > ll:
                break
            step_size *= 0.5

        params = new_params

    return params.numpy(), ll, outer_max_iter, False


def torch_airl_estimate(transitions_t, features_t, expert_states_t, expert_actions_t,
                        expert_next_states_t, beta, sigma, max_rounds=20,
                        discriminator_steps=5, reward_lr=0.01, inner_tol=1e-8,
                        inner_max_iter=5000, use_shaping=True):
    """Full AIRL loop in PyTorch (matching econirl's approach)."""
    n_states = transitions_t.shape[1]
    n_features = features_t.shape[2]
    n_expert = len(expert_states_t)

    reward_weights = torch.nn.Parameter(torch.zeros(n_features, dtype=torch.float32))
    features_f = features_t.float()
    optimizer = torch.optim.Adam([reward_weights], lr=reward_lr)
    policy = torch.ones(n_states, 2) / 2.0
    V = torch.zeros(n_states)

    sampling_times, vi_times, disc_times = [], [], []

    for round_idx in range(max_rounds):
        old_policy = policy.clone()

        # Sample trajectory
        t0 = time.perf_counter()
        pol_s, pol_a, pol_ns = torch_sample_trajectory(
            policy, transitions_t.float(), n_expert,
        )
        sampling_times.append(time.perf_counter() - t0)

        # Discriminator updates
        t0 = time.perf_counter()
        for _ in range(discriminator_steps):
            optimizer.zero_grad()
            reward_matrix = torch.einsum("sak,k->sa", features_f, reward_weights)

            def _logits(s, a, ns):
                r = reward_matrix[s, a]
                if use_shaping:
                    f = r + beta * V[ns] - V[s]
                else:
                    f = r
                return f - torch.log(policy[s, a] + 1e-10)

            e_log = _logits(expert_states_t, expert_actions_t, expert_next_states_t)
            p_log = _logits(pol_s, pol_a, pol_ns)
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    e_log, torch.ones_like(e_log)
                )
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    p_log, torch.zeros_like(p_log)
                )
            )
            loss.backward()
            optimizer.step()
        disc_times.append(time.perf_counter() - t0)

        # Policy update via VI
        t0 = time.perf_counter()
        with torch.no_grad():
            reward_matrix = torch.einsum(
                "sak,k->sa", features_t, reward_weights.data.double()
            )
            V_init = torch.zeros(n_states, dtype=torch.float64)
            V_d, policy_d, _ = torch_value_iteration(
                transitions_t, reward_matrix, V_init, beta, sigma,
                tol=inner_tol, max_iter=inner_max_iter,
            )
            V = V_d.float()
            policy = policy_d.float()
        vi_times.append(time.perf_counter() - t0)

        policy_change = torch.abs(policy - old_policy).max().item()
        if policy_change < 1e-4:
            break

    per_round_times = {
        "sampling": sampling_times,
        "discriminator": disc_times,
        "value_iteration": vi_times,
    }
    return reward_weights.detach().numpy(), policy.numpy(), round_idx + 1, per_round_times


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_kernel_benchmarks(prob, n_warmup=5, n_runs=20):
    """Run micro-benchmarks on individual kernels."""
    n_states = prob["n_states"]
    beta = prob["beta"]
    sigma = prob["sigma"]

    # Convert to framework-specific arrays
    T_j = jnp.array(prob["transitions"])
    U_j = jnp.array(prob["utility"])
    V_j = jnp.zeros(n_states)
    F_j = jnp.array(prob["features"])
    obs_s_j = jnp.array(prob["panel"]["states"])
    obs_a_j = jnp.array(prob["panel"]["actions"])

    T_t = torch.tensor(prob["transitions"])
    U_t = torch.tensor(prob["utility"])
    V_t = torch.zeros(n_states, dtype=torch.float64)
    F_t = torch.tensor(prob["features"])
    obs_s_t = torch.tensor(prob["panel"]["states"])
    obs_a_t = torch.tensor(prob["panel"]["actions"])

    results = {}

    # 1. Single Bellman step
    jax_r = time_jax(bellman_step, T_j, U_j, V_j, beta, sigma,
                     n_warmup=n_warmup, n_runs=n_runs)
    torch_r = time_torch(torch_bellman_step, T_t, U_t, V_t, beta, sigma,
                         n_warmup=n_warmup, n_runs=n_runs)
    results["bellman_step"] = {"jax": jax_r, "torch": torch_r}

    # 2. Value iteration to convergence
    jax_r = time_jax(_value_iteration_jit, T_j, U_j, V_j, beta, sigma,
                     tol=1e-10, max_iter=100000,
                     n_warmup=min(n_warmup, 2), n_runs=min(n_runs, 5))
    torch_r = time_torch(torch_value_iteration, T_t, U_t, V_t, beta, sigma,
                         n_warmup=min(n_warmup, 2), n_runs=min(n_runs, 5))
    results["value_iteration"] = {"jax": jax_r, "torch": torch_r}

    # 3. Hybrid iteration to convergence
    jax_r = time_jax(_hybrid_iteration_jit, T_j, U_j, V_j, beta, sigma,
                     tol=1e-10, max_iter=100000, switch_tol=1e-3,
                     n_warmup=min(n_warmup, 2), n_runs=min(n_runs, 5))
    torch_r = time_torch(torch_hybrid_iteration, T_t, U_t, V_t, beta, sigma,
                         n_warmup=min(n_warmup, 2), n_runs=min(n_runs, 5))
    results["hybrid_iteration"] = {"jax": jax_r, "torch": torch_r}

    # 4. Implicit differentiation scores
    # Get converged V and policy first
    V_conv_j, pol_conv_j, _ = _hybrid_iteration_jit(
        T_j, U_j, V_j, beta, sigma, tol=1e-10, max_iter=100000, switch_tol=1e-3
    )
    V_conv_t, pol_conv_t = torch_hybrid_iteration(
        T_t, U_t, V_t, beta, sigma
    )

    jax_r = time_jax(compute_scores, T_j, F_j, V_conv_j, pol_conv_j,
                     beta, sigma, obs_s_j, obs_a_j,
                     n_warmup=n_warmup, n_runs=n_runs)
    torch_r = time_torch(torch_compute_scores, T_t, F_t, V_conv_t, pol_conv_t,
                         beta, sigma, obs_s_t, obs_a_t,
                         n_warmup=n_warmup, n_runs=n_runs)
    results["implicit_diff_scores"] = {"jax": jax_r, "torch": torch_r}

    # 5. Trajectory sampling (10K steps)
    n_sample = 10000
    key = jax.random.key(0)
    pol_f_j = jnp.array(prob["policy"])
    T_f_j = T_j
    pol_f_t = torch.tensor(prob["policy"], dtype=torch.float32)
    T_f_t = T_t.float()

    jax_r = time_jax(_sample_trajectory_jit, key, pol_f_j, T_f_j,
                     n_steps=n_sample,
                     n_warmup=n_warmup, n_runs=n_runs)
    torch_r = time_torch(torch_sample_trajectory, pol_f_t, T_f_t, n_sample,
                         n_warmup=min(n_warmup, 2), n_runs=min(n_runs, 5))
    results["trajectory_sampling_10k"] = {"jax": jax_r, "torch": torch_r}

    return results


def run_nfxp_benchmark(prob, n_runs=3):
    """End-to-end NFXP-NK estimation benchmark."""
    n_states = prob["n_states"]
    beta = prob["beta"]
    sigma = prob["sigma"]

    T_j = jnp.array(prob["transitions"])
    F_j = jnp.array(prob["features"])
    obs_s_j = jnp.array(prob["panel"]["states"])
    obs_a_j = jnp.array(prob["panel"]["actions"])
    init_j = jnp.array([0.0005, 2.0])

    T_t = torch.tensor(prob["transitions"])
    F_t = torch.tensor(prob["features"])
    obs_s_t = torch.tensor(prob["panel"]["states"])
    obs_a_t = torch.tensor(prob["panel"]["actions"])
    init_t = torch.tensor([0.0005, 2.0], dtype=torch.float64)

    # JAX
    jax_times = []
    jax_result = None
    for i in range(n_runs + 1):
        t0 = time.perf_counter()
        params, ll, n_iter, conv = nfxp_estimate(
            init_j, T_j, F_j, beta, sigma, obs_s_j, obs_a_j,
        )
        # block until ready
        params.block_until_ready()
        elapsed = time.perf_counter() - t0
        if i > 0:  # skip first (JIT warmup)
            jax_times.append(elapsed)
        if jax_result is None:
            jax_result = (np.array(params), ll, n_iter, conv)

    # PyTorch
    torch_times = []
    torch_result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        params, ll, n_iter, conv = torch_nfxp_estimate(
            T_t, F_t, init_t, beta, sigma, obs_s_t, obs_a_t,
        )
        elapsed = time.perf_counter() - t0
        torch_times.append(elapsed)
        if torch_result is None:
            torch_result = (params, ll, n_iter, conv)

    # Verify agreement
    cosine_sim = np.dot(jax_result[0], torch_result[0]) / (
        np.linalg.norm(jax_result[0]) * np.linalg.norm(torch_result[0]) + 1e-15
    )

    return {
        "jax": {"median": median(jax_times), "times": jax_times,
                "params": jax_result[0], "ll": jax_result[1],
                "n_iter": jax_result[2], "converged": jax_result[3]},
        "torch": {"median": median(torch_times), "times": torch_times,
                  "params": torch_result[0], "ll": torch_result[1],
                  "n_iter": torch_result[2], "converged": torch_result[3]},
        "cosine_sim": cosine_sim,
    }


def run_airl_benchmark(prob, max_rounds=20, n_runs=2):
    """End-to-end AIRL benchmark."""
    n_states = prob["n_states"]
    beta = prob["beta"]
    sigma = prob["sigma"]

    T_j = jnp.array(prob["transitions"])
    F_j = jnp.array(prob["features"])
    panel = prob["panel"]
    obs_s_j = jnp.array(panel["states"])
    obs_a_j = jnp.array(panel["actions"])
    obs_ns_j = jnp.array(panel["next_states"])

    T_t = torch.tensor(prob["transitions"])
    F_t = torch.tensor(prob["features"])
    obs_s_t = torch.tensor(panel["states"])
    obs_a_t = torch.tensor(panel["actions"])
    obs_ns_t = torch.tensor(panel["next_states"])

    # JAX
    jax_times = []
    jax_per_round = None
    for i in range(n_runs + 1):
        key = jax.random.key(42 + i)
        t0 = time.perf_counter()
        rp, pol, V, nr, prt = airl_estimate(
            key, T_j, F_j, obs_s_j, obs_a_j, obs_ns_j,
            beta, sigma, max_rounds=max_rounds,
        )
        pol.block_until_ready()
        elapsed = time.perf_counter() - t0
        if i > 0:
            jax_times.append(elapsed)
            if jax_per_round is None:
                jax_per_round = prt

    # PyTorch
    torch_times = []
    torch_per_round = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        rp, pol, nr, prt = torch_airl_estimate(
            T_t, F_t, obs_s_t, obs_a_t, obs_ns_t,
            beta, sigma, max_rounds=max_rounds,
        )
        elapsed = time.perf_counter() - t0
        torch_times.append(elapsed)
        if torch_per_round is None:
            torch_per_round = prt

    return {
        "jax": {"median": median(jax_times), "times": jax_times,
                "per_round": jax_per_round},
        "torch": {"median": median(torch_times), "times": torch_times,
                  "per_round": torch_per_round},
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_kernel_table(all_results):
    """Print kernel micro-benchmark results."""
    sizes = sorted(all_results.keys())
    kernels = [
        "bellman_step", "value_iteration", "hybrid_iteration",
        "implicit_diff_scores", "trajectory_sampling_10k",
    ]

    print("\n" + "=" * 80)
    print("KERNEL MICRO-BENCHMARKS (median time)")
    print("=" * 80)

    # Header
    header = f"{'Kernel':<28}"
    for s in sizes:
        header += f"  {'S=' + str(s):^22}"
    print(header)

    sub_header = f"{'':<28}"
    for _ in sizes:
        sub_header += f"  {'PyTorch':>8} {'JAX':>7} {'Ratio':>5}"
    print(sub_header)
    print("-" * 80)

    for kernel in kernels:
        row = f"{kernel:<28}"
        for s in sizes:
            if kernel in all_results[s]:
                r = all_results[s][kernel]
                t_ms = r["torch"]["median"]
                j_ms = r["jax"]["median"]
                ratio = t_ms / j_ms if j_ms > 0 else float("inf")
                row += f"  {fmt_ms(t_ms):>8} {fmt_ms(j_ms):>7} {ratio:>4.1f}x"
            else:
                row += f"  {'—':>8} {'—':>7} {'—':>5}"
        print(row)

    # JIT overhead
    print("\nJAX JIT Compilation Overhead:")
    for s in sizes:
        print(f"  S={s}:")
        for kernel in kernels:
            if kernel in all_results[s]:
                jit_t = all_results[s][kernel]["jax"]["jit_time"]
                print(f"    {kernel}: {fmt_ms(jit_t)}")


def print_nfxp_table(all_results):
    """Print NFXP end-to-end results."""
    sizes = sorted(all_results.keys())

    print("\n" + "=" * 80)
    print("NFXP-NK END-TO-END ESTIMATION")
    print("=" * 80)

    header = f"{'Metric':<22}"
    for s in sizes:
        header += f"  {'S=' + str(s):^22}"
    print(header)
    print("-" * 80)

    for label, key in [("PyTorch (s)", "torch"), ("JAX (s)", "jax")]:
        row = f"{label:<22}"
        for s in sizes:
            if s in all_results:
                t = all_results[s][key]["median"]
                row += f"  {t:>22.3f}"
            else:
                row += f"  {'—':>22}"
        print(row)

    row = f"{'Speedup':<22}"
    for s in sizes:
        if s in all_results:
            ratio = all_results[s]["torch"]["median"] / max(all_results[s]["jax"]["median"], 1e-9)
            row += f"  {ratio:>21.1f}x"
        else:
            row += f"  {'—':>22}"
    print(row)

    row = f"{'Cosine sim':<22}"
    for s in sizes:
        if s in all_results:
            row += f"  {all_results[s]['cosine_sim']:>22.6f}"
        else:
            row += f"  {'—':>22}"
    print(row)

    # Print converged params
    print("\nParameter recovery:")
    for s in sizes:
        if s in all_results:
            r = all_results[s]
            print(f"  S={s}: PyTorch={r['torch']['params']}  JAX={r['jax']['params']}")
            print(f"        PyTorch LL={r['torch']['ll']:.2f}  JAX LL={r['jax']['ll']:.2f}")


def print_airl_table(all_results):
    """Print AIRL end-to-end results."""
    sizes = sorted(all_results.keys())

    print("\n" + "=" * 80)
    print("AIRL END-TO-END (20 rounds)")
    print("=" * 80)

    header = f"{'Metric':<22}"
    for s in sizes:
        header += f"  {'S=' + str(s):^22}"
    print(header)
    print("-" * 80)

    for label, key in [("PyTorch total (s)", "torch"), ("JAX total (s)", "jax")]:
        row = f"{label:<22}"
        for s in sizes:
            if s in all_results:
                row += f"  {all_results[s][key]['median']:>22.3f}"
            else:
                row += f"  {'—':>22}"
        print(row)

    row = f"{'Speedup':<22}"
    for s in sizes:
        if s in all_results:
            ratio = all_results[s]["torch"]["median"] / max(all_results[s]["jax"]["median"], 1e-9)
            row += f"  {ratio:>21.1f}x"
        else:
            row += f"  {'—':>22}"
    print(row)

    # Per-round breakdown
    print("\nPer-round median breakdown (last available size):")
    if sizes:
        s = sizes[-1]
        r = all_results[s]
        for component in ["sampling", "discriminator", "value_iteration"]:
            jax_med = median(r["jax"]["per_round"][component]) if r["jax"]["per_round"] else 0
            torch_med = median(r["torch"]["per_round"][component]) if r["torch"]["per_round"] else 0
            ratio = torch_med / max(jax_med, 1e-9)
            print(f"  {component:<20}: PyTorch {fmt_ms(torch_med):>10}  JAX {fmt_ms(jax_med):>10}  Ratio: {ratio:.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PyTorch vs JAX DDC benchmark")
    parser.add_argument("--sizes", nargs="+", type=int, default=None,
                        help="State sizes to benchmark (default: 90 250 500)")
    parser.add_argument("--skip-airl", action="store_true",
                        help="Skip AIRL benchmarks")
    parser.add_argument("--skip-nfxp-e2e", action="store_true",
                        help="Skip NFXP end-to-end (just run kernels)")
    args = parser.parse_args()

    sizes = args.sizes or [90, 250, 500]

    print("=" * 80)
    print("PyTorch vs JAX Speed Benchmark for DDC Estimation")
    print("=" * 80)
    print(f"Platform: {sys.platform}")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Problem sizes: {sizes}")
    print()

    # Generate problems
    print("Generating benchmark problems...")
    problems = {}
    for s in sizes:
        print(f"  S={s}...", end=" ", flush=True)
        t0 = time.perf_counter()
        problems[s] = generate_problem(s)
        print(f"done ({time.perf_counter() - t0:.1f}s)")

    # Kernel benchmarks
    print("\nRunning kernel micro-benchmarks...")
    kernel_results = {}
    for s in sizes:
        print(f"\n  S={s}:")
        kernel_results[s] = run_kernel_benchmarks(problems[s])
        for k, v in kernel_results[s].items():
            t_ms = v["torch"]["median"] * 1000
            j_ms = v["jax"]["median"] * 1000
            ratio = t_ms / max(j_ms, 0.001)
            print(f"    {k:<28}: PT={t_ms:>8.2f}ms  JAX={j_ms:>7.2f}ms  ({ratio:.1f}x)")

    print_kernel_table(kernel_results)

    # NFXP end-to-end
    if not args.skip_nfxp_e2e:
        print("\nRunning NFXP-NK end-to-end benchmarks...")
        nfxp_results = {}
        for s in sizes:
            print(f"  S={s}...", end=" ", flush=True)
            t0 = time.perf_counter()
            nfxp_results[s] = run_nfxp_benchmark(problems[s])
            print(f"done ({time.perf_counter() - t0:.1f}s)")

        print_nfxp_table(nfxp_results)

    # AIRL
    if not args.skip_airl:
        print("\nRunning AIRL benchmarks...")
        airl_results = {}
        for s in sizes:
            print(f"  S={s}...", end=" ", flush=True)
            t0 = time.perf_counter()
            airl_results[s] = run_airl_benchmark(problems[s])
            print(f"done ({time.perf_counter() - t0:.1f}s)")

        print_airl_table(airl_results)

    print("\n" + "=" * 80)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
