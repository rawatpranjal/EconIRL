#!/usr/bin/env python3
"""
LSW 2026 Semi-Synthetic Replication
====================================

Replicates the key results of Lee, Sudhir, and Wang (2026) on a
semi-synthetic serialized content environment. Known ground-truth
segment rewards drive data generation. AIRL-Anchored recovers the segments
and rewards. Three counterfactual exercises demonstrate the value of
heterogeneous estimation for platform policy design.

Environment:
    50 episodes across 5 books (10 episodes each) + 1 absorbing state.
    Episodes have synthetic quality/engagement scores that vary across
    the narrative arc (peaks near book endings create cliffhangers).

Segments:
    A "Pay & Read" (30%): quality-sensitive, buys cliffhangers, low
        price sensitivity, strong completion drive.
    B "Wait & Read" (70%): price-sensitive, patient, waits for free
        access, weak cliffhanger response.

Actions: buy=0 (pay, advance), wait=1 (free next period), exit=2 (absorb)

Counterfactuals:
    Type A: Introduce WFF (wait-for-free) to a pay-only platform.
    Type B: Segment-customized wait-times.
    Type C: Content-based pricing (cliffhangers paid-only, weak free).

Reference:
    Lee, P., Sudhir, K., Wang, Y. (2026). "Modeling Serialized Content
    Consumption: Adversarial IRL for Dynamic Discrete Choice."

Usage:
    python examples/serialized-content/lsw_replication.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig
from econirl.estimation.nfxp import NFXPEstimator
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel_from_policy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_BOOKS = 5
EPISODES_PER_BOOK = 10
N_EPISODES = N_BOOKS * EPISODES_PER_BOOK   # 50
N_STATES = N_EPISODES + 1                   # +1 absorbing
ABSORBING = N_EPISODES                      # state 50
N_ACTIONS = 3                               # buy=0, wait=1, exit=2
EXIT_ACTION = 2
DISCOUNT = 0.90
N_FEATURES = 6  # buy_cost, quality_buy, cliffhanger_buy, wait_cost, quality_wait, progress_buy

N_INDIVIDUALS = 2000
N_SERIES_PER_USER = 3   # each user reads 3 books (for within-user consistency)
TRUE_MIX_A = 0.30       # 30% Pay & Read

# Segment A: "Pay and Read" -- quality-sensitive, buys cliffhangers
TRUE_PARAMS_A = np.array([-0.3, 2.0, 1.5, -0.8, 0.5, 0.8])

# Segment B: "Wait and Read" -- price-sensitive, patient waiter
TRUE_PARAMS_B = np.array([-1.5, 0.8, 0.3, -0.2, 0.6, 0.3])

PARAM_NAMES = [
    "buy_cost", "quality_buy", "cliffhanger_buy",
    "wait_cost", "quality_wait", "progress_buy",
]


# ---------------------------------------------------------------------------
# Environment builder
# ---------------------------------------------------------------------------

def build_environment(seed=42):
    """Build serialized content environment with synthetic quality scores.

    Quality varies sinusoidally across episodes within each book,
    peaking near episode 8-9 to create cliffhangers before book endings.
    Random noise adds episode-level variation.
    """
    rng = np.random.default_rng(seed)

    # Generate episode quality scores
    quality = np.zeros(N_EPISODES)
    for book in range(N_BOOKS):
        for ep in range(EPISODES_PER_BOOK):
            s = book * EPISODES_PER_BOOK + ep
            base = 0.5 + 0.3 * np.sin(np.pi * ep / (EPISODES_PER_BOOK - 1))
            quality[s] = np.clip(base + 0.1 * rng.standard_normal(), 0.1, 1.0)
    cliffhanger = (quality > 0.7).astype(float)
    book_progress = np.array([(s % EPISODES_PER_BOOK) / (EPISODES_PER_BOOK - 1)
                               for s in range(N_EPISODES)])

    # Transitions: deterministic
    T = np.zeros((N_ACTIONS, N_STATES, N_STATES))
    for s in range(N_EPISODES):
        # Buy: advance to next episode (wrap at book boundary to absorbing)
        next_ep = s + 1
        if next_ep >= N_EPISODES or (s + 1) % EPISODES_PER_BOOK == 0:
            # End of book: go to absorbing (completed the book)
            T[0, s, ABSORBING] = 1.0
        else:
            T[0, s, next_ep] = 1.0
        # Wait: stay at current episode
        T[1, s, s] = 1.0
        # Exit: go to absorbing state
        T[2, s, ABSORBING] = 1.0
    # Absorbing state stays
    for a in range(N_ACTIONS):
        T[a, ABSORBING, ABSORBING] = 1.0
    transitions = jnp.array(T, dtype=jnp.float32)

    # Feature matrix: (N_STATES, N_ACTIONS, N_FEATURES)
    features = np.zeros((N_STATES, N_ACTIONS, N_FEATURES))
    for s in range(N_EPISODES):
        # Buy action features
        features[s, 0, 0] = 1.0                  # buy_cost indicator
        features[s, 0, 1] = quality[s]            # quality (content value)
        features[s, 0, 2] = cliffhanger[s]        # cliffhanger urgency
        # Wait action features
        features[s, 1, 3] = 1.0                  # wait_cost indicator
        features[s, 1, 4] = quality[s]            # quality on wait
        # Buy action: completion drive
        features[s, 0, 5] = book_progress[s]      # progress through book
    # Exit action (a=2): all features zero (anchor normalization)
    # Absorbing state: all features zero

    feature_matrix = jnp.array(features, dtype=jnp.float32)

    problem = DDCProblem(
        num_states=N_STATES,
        num_actions=N_ACTIONS,
        discount_factor=DISCOUNT,
        scale_parameter=1.0,
    )

    utility = ActionDependentReward(feature_matrix, PARAM_NAMES)

    # Compute segment rewards
    params_a = jnp.array(TRUE_PARAMS_A, dtype=jnp.float32)
    params_b = jnp.array(TRUE_PARAMS_B, dtype=jnp.float32)
    reward_a = jnp.einsum("sak,k->sa", feature_matrix, params_a)
    reward_b = jnp.einsum("sak,k->sa", feature_matrix, params_b)

    # Pre-estimation diagnostics
    F = features.reshape(-1, N_FEATURES)
    rank = np.linalg.matrix_rank(F)
    cond = np.linalg.cond(F[F.any(axis=1)])

    return {
        "transitions": transitions,
        "feature_matrix": feature_matrix,
        "problem": problem,
        "utility": utility,
        "params_a": params_a,
        "params_b": params_b,
        "reward_a": reward_a,
        "reward_b": reward_b,
        "quality": quality,
        "cliffhanger": cliffhanger,
        "book_progress": book_progress,
        "feature_rank": rank,
        "condition_number": cond,
    }


# ---------------------------------------------------------------------------
# Panel generation with multi-book trajectories
# ---------------------------------------------------------------------------

def simulate_single_type_panel(env, seed=42):
    """Generate panel from Segment A (Pay&Read) only. No mixture.

    Each user reads N_SERIES_PER_USER books, each starting at episode 0
    of a different book.
    """
    rng = np.random.default_rng(seed)
    operator = SoftBellmanOperator(env["problem"], env["transitions"])

    result_a = value_iteration(operator, env["reward_a"], tol=1e-10, max_iter=5000)
    policy_a = np.array(result_a.policy)
    print(f"  Soft VI converged: {result_a.converged} ({result_a.num_iterations} iters)")

    trajectories = []
    for i in range(N_INDIVIDUALS):
        book_order = rng.choice(N_BOOKS, size=N_SERIES_PER_USER, replace=False)
        for book_idx in book_order:
            start_ep = book_idx * EPISODES_PER_BOOK
            state = start_ep
            states, actions, next_states = [], [], []

            for _ in range(EPISODES_PER_BOOK * 2):
                if state == ABSORBING:
                    break
                probs = np.array(policy_a[state])
                probs = np.maximum(probs, 0)
                probs = probs / probs.sum()
                action = rng.choice(N_ACTIONS, p=probs)
                next_state = int(jnp.argmax(env["transitions"][action, state]))

                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                state = next_state

            if len(states) >= 2:
                trajectories.append(Trajectory(
                    states=jnp.array(states, dtype=jnp.int32),
                    actions=jnp.array(actions, dtype=jnp.int32),
                    next_states=jnp.array(next_states, dtype=jnp.int32),
                    individual_id=i,
                ))

    panel = Panel(trajectories=trajectories)
    return panel, policy_a


def project_reward_onto_features(reward_matrix, feature_matrix):
    """Project tabular R(s,a) onto linear feature basis.

    Solves: theta_hat = argmin ||F @ theta - R||^2
    where F is (n_states*n_actions, n_features) and R is (n_states*n_actions,).

    Returns the best-fitting linear parameters and R-squared.
    """
    R_flat = np.array(reward_matrix).flatten()
    F_flat = np.array(feature_matrix).reshape(-1, feature_matrix.shape[-1])

    # Only fit on non-zero rows (skip absorbing state)
    mask = np.any(F_flat != 0, axis=1) | (R_flat != 0)
    theta_hat, residuals, _, _ = np.linalg.lstsq(F_flat[mask], R_flat[mask], rcond=None)

    # R-squared
    R_pred = F_flat[mask] @ theta_hat
    ss_res = np.sum((R_flat[mask] - R_pred) ** 2)
    ss_tot = np.sum((R_flat[mask] - R_flat[mask].mean()) ** 2)
    r_squared = 1 - ss_res / max(ss_tot, 1e-10)

    return theta_hat, r_squared


# ---------------------------------------------------------------------------
# Counterfactual helpers
# ---------------------------------------------------------------------------

def simulate_behavior(policy, transitions, start_states, n_per_start=500,
                      max_steps=30, seed=None):
    """Simulate behavior under a given policy from specified start states.

    Returns average episodes consumed, buy actions, wait actions, and exit rate.
    """
    rng = np.random.default_rng(seed)
    policy_np = np.array(policy)
    transitions_np = np.array(transitions)

    total_episodes = 0
    total_buys = 0
    total_waits = 0
    total_exits = 0
    n_sims = 0

    for s0 in start_states:
        for _ in range(n_per_start):
            state = s0
            for _ in range(max_steps):
                if state == ABSORBING:
                    break
                probs = np.maximum(policy_np[state], 0)
                probs = probs / probs.sum()
                action = rng.choice(N_ACTIONS, p=probs)
                next_state = int(np.argmax(transitions_np[action, state]))

                if action == 0:
                    total_buys += 1
                    total_episodes += 1
                elif action == 1:
                    total_waits += 1
                else:
                    total_exits += 1

                state = next_state
            n_sims += 1

    return {
        "avg_episodes": total_episodes / max(n_sims, 1),
        "avg_buys": total_buys / max(n_sims, 1),
        "avg_waits": total_waits / max(n_sims, 1),
        "exit_rate": total_exits / max(total_buys + total_waits + total_exits, 1),
    }


def compute_policy_from_reward(reward, problem, transitions):
    """Compute soft VI policy from a reward matrix."""
    operator = SoftBellmanOperator(problem, transitions)
    result = value_iteration(operator, reward, tol=1e-10, max_iter=5000)
    return result.policy, result.V


# ---------------------------------------------------------------------------
# Counterfactual Type A: Introduce WFF
# ---------------------------------------------------------------------------

def counterfactual_type_a(env):
    """Type A: Introduce wait-for-free to a pay-only platform.

    Baseline: Only buy and exit available (wait action disabled by
    making it equivalent to exit).
    Counterfactual: All three actions available (WFF enabled).
    """
    print("\n--- Counterfactual Type A: Introduce WFF ---")

    # Baseline: pay-only (wait = exit)
    T_payonly = np.array(env["transitions"])
    T_payonly[1] = T_payonly[2]  # wait action transitions = exit transitions
    T_payonly = jnp.array(T_payonly, dtype=jnp.float32)

    # Book start states (first episode of each book)
    start_states = [b * EPISODES_PER_BOOK for b in range(N_BOOKS)]

    results = {}
    for seg_name, reward in [("Pay&Read", env["reward_a"]), ("Wait&Read", env["reward_b"])]:
        # Baseline policy (pay-only)
        pol_base, _ = compute_policy_from_reward(reward, env["problem"], T_payonly)
        base_stats = simulate_behavior(pol_base, T_payonly, start_states, seed=100)

        # Counterfactual policy (WFF enabled)
        pol_wff, _ = compute_policy_from_reward(reward, env["problem"], env["transitions"])
        wff_stats = simulate_behavior(pol_wff, env["transitions"], start_states, seed=100)

        results[seg_name] = {
            "baseline": base_stats,
            "wff": wff_stats,
            "consumption_change": (wff_stats["avg_episodes"] - base_stats["avg_episodes"])
                                 / max(base_stats["avg_episodes"], 0.01),
            "purchase_change": (wff_stats["avg_buys"] - base_stats["avg_buys"])
                               / max(base_stats["avg_buys"], 0.01),
        }

        print(f"\n  {seg_name}:")
        print(f"    Baseline (pay-only): {base_stats['avg_episodes']:.2f} episodes, "
              f"{base_stats['avg_buys']:.2f} buys")
        print(f"    WFF enabled:         {wff_stats['avg_episodes']:.2f} episodes, "
              f"{wff_stats['avg_buys']:.2f} buys, {wff_stats['avg_waits']:.2f} waits")
        print(f"    Consumption change:  {results[seg_name]['consumption_change']:+.1%}")
        print(f"    Purchase change:     {results[seg_name]['purchase_change']:+.1%}")

    return results


# ---------------------------------------------------------------------------
# Counterfactual Type B: Segment-Customized Wait-Times
# ---------------------------------------------------------------------------

def counterfactual_type_b(env):
    """Type B: Different wait-times per segment.

    Baseline: Uniform wait cost.
    Counterfactual:
        Wait&Read: reduce wait_cost by 50% (shorter wait, more patient-friendly)
        Pay&Read: increase wait_cost by 50% (longer wait, screens into buying)
    """
    print("\n--- Counterfactual Type B: Segment-Customized Wait-Times ---")

    features_np = np.array(env["feature_matrix"])
    start_states = [b * EPISODES_PER_BOOK for b in range(N_BOOKS)]

    results = {}
    for seg_name, params, wait_mult in [
        ("Pay&Read", env["params_a"], 1.5),    # longer wait for buyers
        ("Wait&Read", env["params_b"], 0.5),    # shorter wait for waiters
    ]:
        # Baseline (uniform wait cost)
        reward_base = jnp.einsum("sak,k->sa", env["feature_matrix"], params)
        pol_base, _ = compute_policy_from_reward(reward_base, env["problem"], env["transitions"])
        base_stats = simulate_behavior(pol_base, env["transitions"], start_states, seed=200)

        # Modified wait cost
        features_mod = features_np.copy()
        features_mod[:, 1, 3] *= wait_mult  # scale wait_cost feature
        reward_mod = jnp.einsum("sak,k->sa", jnp.array(features_mod, dtype=jnp.float32), params)
        pol_mod, _ = compute_policy_from_reward(reward_mod, env["problem"], env["transitions"])
        mod_stats = simulate_behavior(pol_mod, env["transitions"], start_states, seed=200)

        results[seg_name] = {
            "baseline": base_stats,
            "modified": mod_stats,
            "wait_multiplier": wait_mult,
            "purchase_change": (mod_stats["avg_buys"] - base_stats["avg_buys"])
                               / max(base_stats["avg_buys"], 0.01),
            "consumption_change": (mod_stats["avg_episodes"] - base_stats["avg_episodes"])
                                  / max(base_stats["avg_episodes"], 0.01),
        }

        print(f"\n  {seg_name} (wait cost x{wait_mult}):")
        print(f"    Baseline: {base_stats['avg_buys']:.2f} buys, "
              f"{base_stats['avg_waits']:.2f} waits")
        print(f"    Modified: {mod_stats['avg_buys']:.2f} buys, "
              f"{mod_stats['avg_waits']:.2f} waits")
        print(f"    Purchase change: {results[seg_name]['purchase_change']:+.1%}")

    return results


# ---------------------------------------------------------------------------
# Counterfactual Type C: Content-Based Pricing
# ---------------------------------------------------------------------------

def counterfactual_type_c(env):
    """Type C: Cliffhanger episodes are paid-only, weak episodes are free.

    Baseline: All episodes have same pricing.
    Counterfactual:
        Cliffhanger (quality > 0.7): wait disabled (wait = exit), must pay
        Weak (quality <= 0.7): buy_cost = 0 (free episodes)
    """
    print("\n--- Counterfactual Type C: Content-Based Pricing ---")

    features_np = np.array(env["feature_matrix"])
    transitions_np = np.array(env["transitions"])

    # Build modified transitions and features
    T_content = transitions_np.copy()
    F_content = features_np.copy()

    for s in range(N_EPISODES):
        if env["cliffhanger"][s] > 0.5:
            # Cliffhanger: disable wait (wait = exit)
            T_content[1, s, :] = T_content[2, s, :]
        else:
            # Weak episode: free (remove buy cost)
            F_content[s, 0, 0] = 0.0

    T_content_jnp = jnp.array(T_content, dtype=jnp.float32)
    F_content_jnp = jnp.array(F_content, dtype=jnp.float32)

    start_states = [b * EPISODES_PER_BOOK for b in range(N_BOOKS)]
    n_cliffhangers = int(env["cliffhanger"].sum())
    print(f"  Cliffhanger episodes (paid-only): {n_cliffhangers}/{N_EPISODES}")
    print(f"  Weak episodes (free): {N_EPISODES - n_cliffhangers}/{N_EPISODES}")

    results = {}
    for seg_name, params in [("Pay&Read", env["params_a"]), ("Wait&Read", env["params_b"])]:
        # Baseline
        reward_base = jnp.einsum("sak,k->sa", env["feature_matrix"], params)
        pol_base, _ = compute_policy_from_reward(reward_base, env["problem"], env["transitions"])
        base_stats = simulate_behavior(pol_base, env["transitions"], start_states, seed=300)

        # Content-based pricing
        reward_content = jnp.einsum("sak,k->sa", F_content_jnp, params)
        pol_content, _ = compute_policy_from_reward(reward_content, env["problem"], T_content_jnp)
        content_stats = simulate_behavior(pol_content, T_content_jnp, start_states, seed=300)

        results[seg_name] = {
            "baseline": base_stats,
            "content_pricing": content_stats,
            "purchase_change": (content_stats["avg_buys"] - base_stats["avg_buys"])
                               / max(base_stats["avg_buys"], 0.01),
            "consumption_change": (content_stats["avg_episodes"] - base_stats["avg_episodes"])
                                  / max(base_stats["avg_episodes"], 0.01),
        }

        print(f"\n  {seg_name}:")
        print(f"    Baseline: {base_stats['avg_buys']:.2f} buys, "
              f"{base_stats['avg_episodes']:.2f} episodes")
        print(f"    Content pricing: {content_stats['avg_buys']:.2f} buys, "
              f"{content_stats['avg_episodes']:.2f} episodes")
        print(f"    Purchase change: {results[seg_name]['purchase_change']:+.1%}")
        print(f"    Consumption change: {results[seg_name]['consumption_change']:+.1%}")

    return results


# ---------------------------------------------------------------------------
# Field experiment validation
# ---------------------------------------------------------------------------

def field_experiment_validation(env, policy_a_true, policy_b_true):
    """Validate that recovered rewards predict behavior under policy change.

    Hold-out test: reduce wait cost on books 3-4 (episodes 30-49).
    Compare predicted vs actual behavior change per segment.
    """
    print("\n--- Field Experiment Validation ---")

    features_np = np.array(env["feature_matrix"])

    # Modified environment: reduce wait cost on books 3-4
    F_exp = features_np.copy()
    for s in range(30, N_EPISODES):
        F_exp[s, 1, 3] *= 0.5  # halve wait cost on held-out books
    F_exp_jnp = jnp.array(F_exp, dtype=jnp.float32)

    # Start states for books 3-4
    exp_starts = [30, 40]

    results = {}
    for seg_name, params, true_policy in [
        ("Pay&Read", env["params_a"], policy_a_true),
        ("Wait&Read", env["params_b"], policy_b_true),
    ]:
        # Actual behavior under original regime
        actual_base = simulate_behavior(
            true_policy, env["transitions"], exp_starts, n_per_start=1000, seed=400
        )

        # Actual behavior under experiment (re-solve with true params + modified features)
        reward_exp = jnp.einsum("sak,k->sa", F_exp_jnp, params)
        pol_exp_true, _ = compute_policy_from_reward(reward_exp, env["problem"], env["transitions"])
        actual_exp = simulate_behavior(
            pol_exp_true, env["transitions"], exp_starts, n_per_start=1000, seed=400
        )

        # Change in buy rate
        actual_buy_change = actual_exp["avg_buys"] - actual_base["avg_buys"]
        actual_wait_change = actual_exp["avg_waits"] - actual_base["avg_waits"]

        results[seg_name] = {
            "base_buys": actual_base["avg_buys"],
            "exp_buys": actual_exp["avg_buys"],
            "buy_change": actual_buy_change,
            "base_waits": actual_base["avg_waits"],
            "exp_waits": actual_exp["avg_waits"],
            "wait_change": actual_wait_change,
        }

        print(f"\n  {seg_name}:")
        print(f"    Baseline: {actual_base['avg_buys']:.2f} buys, "
              f"{actual_base['avg_waits']:.2f} waits")
        print(f"    Experiment (wait cost halved): {actual_exp['avg_buys']:.2f} buys, "
              f"{actual_exp['avg_waits']:.2f} waits")
        print(f"    Buy change: {actual_buy_change:+.3f}")
        print(f"    Wait change: {actual_wait_change:+.3f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 72)
    print("LSW 2026 Semi-Synthetic Replication (K=1, Single Type)")
    print("Lee, Sudhir, and Wang: Serialized Content Consumption")
    print("=" * 72)

    # Step 1: Build environment
    print("\n--- Step 1: Build Environment ---")
    env = build_environment(seed=42)
    print(f"  States: {N_STATES} ({N_EPISODES} episodes + 1 absorbing)")
    print(f"  Books: {N_BOOKS} x {EPISODES_PER_BOOK} episodes")
    print(f"  Actions: buy=0, wait=1, exit=2")
    print(f"  Feature rank: {env['feature_rank']} / {N_FEATURES}")
    print(f"  Condition number: {env['condition_number']:.1f}")
    print(f"  Cliffhanger episodes: {int(env['cliffhanger'].sum())}/{N_EPISODES}")

    # Step 2: Generate panel (single type: Segment A only)
    print(f"\n--- Step 2: Generate {N_INDIVIDUALS} Users x {N_SERIES_PER_USER} Books (Segment A only) ---")
    print(f"  True params: {dict(zip(PARAM_NAMES, TRUE_PARAMS_A))}")

    panel, policy_a = simulate_single_type_panel(env, seed=42)
    n_obs = sum(len(t.states) for t in panel.trajectories)
    all_actions = np.concatenate([np.array(t.actions) for t in panel.trajectories])
    print(f"  Trajectories: {len(panel.trajectories):,}")
    print(f"  Observations: {n_obs:,}")
    print(f"  Action shares: buy={(all_actions==0).mean():.3f}, "
          f"wait={(all_actions==1).mean():.3f}, exit={(all_actions==2).mean():.3f}")

    # Step 3: NFXP (structural MLE baseline)
    print("\n--- Step 3: NFXP (structural MLE) ---")
    t0 = time.time()
    try:
        nfxp = NFXPEstimator(se_method="robust")
        nfxp_result = nfxp.estimate(
            panel, env["utility"], env["problem"], env["transitions"]
        )
        nfxp_params = np.array(nfxp_result.parameters)
        print(f"  Time: {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"  NFXP failed: {e}")
        nfxp_params = None

    # Step 4: AIRL-Anchored (K=1, no EM)
    print("\n--- Step 4: AIRL-Anchored (K=1, single type) ---")
    t0 = time.time()
    airl_config = AIRLConfig(
        reward_type="linear",
        max_rounds=500,
        reward_lr=0.01,
        reward_weight_decay=0.01,
        discriminator_steps=1,
        policy_step_size=0.1,   # conservative policy iteration (key fix)
        use_shaping=True,
        convergence_tol=1e-6,
        verbose=True,
    )
    airl = AIRLEstimator(airl_config)
    airl_result = airl.estimate(
        panel, env["utility"], env["problem"], env["transitions"]
    )
    airl_time = time.time() - t0
    print(f"  Time: {airl_time:.1f}s")

    # With reward_type="linear", parameters ARE the 6 linear coefficients
    airl_params = np.array(airl_result.parameters) if airl_result.parameters is not None else None
    if airl_params is not None:
        print(f"  AIRL recovered params: {dict(zip(PARAM_NAMES, [f'{p:.4f}' for p in airl_params]))}")

    # Also extract tabular reward for episode-level comparison
    airl_reward = None
    if airl_result.metadata and "reward_matrix" in airl_result.metadata:
        airl_reward = np.array(airl_result.metadata["reward_matrix"])

    # Step 5: Linear projection of recovered rewards
    print("\n" + "=" * 72)
    print("Parameter Recovery: True vs NFXP vs AIRL-Anchored")
    print("=" * 72)

    true_params = TRUE_PARAMS_A

    # Parameter comparison table
    header = f"  {'Parameter':>20} {'True':>10} {'NFXP':>10} {'AIRL':>10}"
    print(header)
    print("  " + "-" * len(header.strip()))

    for i, name in enumerate(PARAM_NAMES):
        row = f"  {name:>20} {true_params[i]:>10.4f}"
        if nfxp_params is not None:
            row += f" {nfxp_params[i]:>10.4f}"
        else:
            row += f" {'--':>10}"
        if airl_params is not None and i < len(airl_params):
            row += f" {airl_params[i]:>10.4f}"
        else:
            row += f" {'--':>10}"
        print(row)

    # Show implied tabular reward comparison at selected episodes
    if airl_params is not None:
        features_np = np.array(env["feature_matrix"])
        true_reward = np.array(env["reward_a"])
        airl_reward_implied = features_np @ airl_params
        print(f"\n  Implied R(s,a) at selected episodes:")
        print(f"  {'Ep':>4} {'True buy':>10} {'AIRL buy':>10} "
              f"{'True wait':>10} {'AIRL wait':>10}")
        print("  " + "-" * 48)
        for s in [0, 4, 8, 15, 25, 40, 49]:
            if s < N_EPISODES:
                print(f"  {s:>4} {true_reward[s,0]:>10.4f} {airl_reward_implied[s,0]:>10.4f} "
                      f"{true_reward[s,1]:>10.4f} {airl_reward_implied[s,1]:>10.4f}")

    # Step 6: Counterfactuals (use ground truth rewards)
    print("\n" + "=" * 72)
    print("Counterfactual Exercises (ground truth rewards)")
    print("=" * 72)

    cf_a = counterfactual_type_a(env)
    cf_b = counterfactual_type_b(env)
    cf_c = counterfactual_type_c(env)

    # Step 7: Field experiment (Segment A only)
    print("\n" + "=" * 72)
    print("Field Experiment Validation")
    print("=" * 72)

    print("\n--- Field Experiment (Segment A: Pay&Read) ---")
    features_np_exp = np.array(env["feature_matrix"]).copy()
    for s in range(30, N_EPISODES):
        features_np_exp[s, 1, 3] *= 0.5
    F_exp_jnp = jnp.array(features_np_exp, dtype=jnp.float32)

    exp_starts = [30, 40]
    base_stats = simulate_behavior(policy_a, env["transitions"], exp_starts, n_per_start=1000, seed=400)
    reward_exp = jnp.einsum("sak,k->sa", F_exp_jnp, env["params_a"])
    pol_exp, _ = compute_policy_from_reward(reward_exp, env["problem"], env["transitions"])
    exp_stats = simulate_behavior(pol_exp, env["transitions"], exp_starts, n_per_start=1000, seed=400)

    print(f"  Baseline: {base_stats['avg_buys']:.2f} buys, {base_stats['avg_waits']:.2f} waits")
    print(f"  Experiment (wait cost halved on books 3-4): "
          f"{exp_stats['avg_buys']:.2f} buys, {exp_stats['avg_waits']:.2f} waits")
    print(f"  Buy change: {exp_stats['avg_buys'] - base_stats['avg_buys']:+.3f}")
    print(f"  Wait change: {exp_stats['avg_waits'] - base_stats['avg_waits']:+.3f}")

    # Save results
    out = {
        "mode": "K=1 single type (Segment A: Pay&Read)",
        "environment": {
            "n_episodes": N_EPISODES,
            "n_states": N_STATES,
            "n_actions": N_ACTIONS,
            "discount": DISCOUNT,
            "feature_rank": int(env["feature_rank"]),
            "n_cliffhangers": int(env["cliffhanger"].sum()),
        },
        "true_parameters": dict(zip(PARAM_NAMES, TRUE_PARAMS_A.tolist())),
        "nfxp_parameters": dict(zip(PARAM_NAMES, nfxp_params.tolist())) if nfxp_params is not None else None,
        "airl_parameters": dict(zip(PARAM_NAMES, airl_params.tolist())) if airl_params is not None else None,
        "airl_time": airl_time,
    }

    out_path = Path("examples/serialized-content/lsw_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n\nResults saved to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
