"""Integration test for Wulfmeier (2016) Deep MaxEnt IRL.

Runs small-scale (8x8) benchmarks verifying that MCEIRLNeural
outperforms linear MCE-IRL on Binaryworld where the reward has
nonlinear feature interactions.
"""

import pytest
import numpy as np
import pandas as pd
import jax.numpy as jnp

from econirl.environments.binaryworld import BinaryworldEnvironment
from econirl.environments.objectworld import ObjectworldEnvironment
from econirl.core.types import DDCProblem, Panel
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.estimators.mceirl_neural import MCEIRLNeural
from econirl.preferences.linear import LinearUtility


def _panel_to_df(panel: Panel) -> pd.DataFrame:
    """Convert a Panel object to a DataFrame for the sklearn-style API."""
    rows = []
    for traj in panel.trajectories:
        for t in range(len(traj.states)):
            rows.append({
                "agent_id": traj.individual_id,
                "state": int(traj.states[t]),
                "action": int(traj.actions[t]),
            })
    return pd.DataFrame(rows)


def _compute_evd(
    true_reward: jnp.ndarray,
    learned_policy: jnp.ndarray,
    transitions: jnp.ndarray,
    problem: DDCProblem,
) -> float:
    """Compute Expected Value Difference between optimal and learned policy.

    EVD measures how much worse the learned policy performs compared to
    the optimal policy, both evaluated under the true reward function.
    A lower EVD means the learned policy is closer to optimal.

    The computation solves for the optimal value function under the true
    reward, then evaluates the learned policy under the same reward via
    a linear system solve, and returns the mean difference across states.
    """
    n_states = problem.num_states
    n_actions = problem.num_actions
    gamma = problem.discount_factor

    # Build the reward matrix (S, A) from the state-only reward
    reward_sa = jnp.tile(true_reward[:, None], (1, n_actions)).astype(jnp.float64)

    # Optimal value under true reward
    operator = SoftBellmanOperator(problem, transitions)
    opt_result = policy_iteration(operator, reward_sa.astype(jnp.float32), tol=1e-10, max_iter=200)
    v_star = opt_result.V.astype(jnp.float64)

    # Value of learned policy under true reward
    pi = learned_policy.astype(jnp.float64)
    r_pi = (pi * reward_sa).sum(axis=1)
    P_pi = jnp.einsum("sa,ast->st", pi, transitions.astype(jnp.float64))
    I = jnp.eye(n_states, dtype=jnp.float64)
    v_learned = jnp.linalg.solve(I - gamma * P_pi, r_pi)

    return float((v_star - v_learned).mean())


@pytest.mark.slow
class TestWulfmeierBinaryworld:
    """Test that deep MCE-IRL outperforms linear MCE-IRL on Binaryworld.

    Binaryworld has a nonlinear reward structure that depends on the count
    of blue neighbors in a 3x3 neighborhood. A linear model over the 9
    binary neighborhood features cannot represent the count-based thresholds,
    so the neural reward function should achieve lower Expected Value
    Difference than the linear baseline.
    """

    def test_deep_beats_linear_on_binaryworld(self):
        np.random.seed(42)

        # Create environment and generate demonstrations
        env = BinaryworldEnvironment(grid_size=8, seed=42)
        panel = env.simulate_demonstrations(n_demos=32, max_steps=30, seed=0)
        df = _panel_to_df(panel)

        n_states = env.num_states
        n_actions = env.num_actions
        n_features = env.feature_matrix.shape[2]
        transitions = env.transition_matrices
        features = env.feature_matrix

        # Build a state encoder that maps state indices to the 9-dimensional
        # binary neighborhood feature vectors. The neural network needs these
        # features as input to learn the nonlinear count-based reward.
        state_features = features[:, 0, :]  # (S, 9) -- same across actions
        def binaryworld_encoder(states: jnp.ndarray) -> jnp.ndarray:
            return state_features[states.astype(jnp.int32)]

        # --- Deep MCE-IRL (neural reward) ---
        np.random.seed(42)
        deep_model = MCEIRLNeural(
            n_states=n_states,
            n_actions=n_actions,
            discount=0.9,
            reward_type="state",
            reward_hidden_dim=32,
            reward_num_layers=2,
            max_epochs=200,
            lr=0.01,
            state_encoder=binaryworld_encoder,
            state_dim=n_features,
            verbose=False,
        )
        deep_model.fit(
            df,
            state="state",
            action="action",
            id="agent_id",
            transitions=np.asarray(transitions),
            features=features,
        )
        deep_policy = jnp.array(deep_model.policy_).astype(jnp.float32)

        deep_evd = _compute_evd(
            env.true_reward, deep_policy, transitions, env.problem_spec
        )

        # --- Linear MCE-IRL ---
        utility = LinearUtility(
            feature_matrix=features,
            parameter_names=[f"f{i}" for i in range(n_features)],
        )
        estimator = MCEIRLEstimator(
            config=MCEIRLConfig(
                optimizer="L-BFGS-B",
                inner_solver="hybrid",
                inner_max_iter=5000,
                inner_tol=1e-8,
                outer_max_iter=500,
                outer_tol=1e-6,
                compute_se=False,
                verbose=False,
            )
        )
        result = estimator.estimate(
            panel, utility, env.problem_spec, transitions
        )
        linear_policy = result.policy

        linear_evd = _compute_evd(
            env.true_reward, linear_policy, transitions, env.problem_spec
        )

        print(f"\nBinaryworld 8x8 EVD results:")
        print(f"  Deep MCE-IRL EVD:   {deep_evd:.4f}")
        print(f"  Linear MCE-IRL EVD: {linear_evd:.4f}")
        print(f"  Deep wins by:       {linear_evd - deep_evd:.4f}")

        # The neural model should achieve lower EVD because the reward
        # depends on count thresholds that are not linearly separable
        # in the binary neighborhood features.
        assert deep_evd < linear_evd, (
            f"Expected deep EVD ({deep_evd:.4f}) < linear EVD ({linear_evd:.4f}). "
            f"The neural reward should capture nonlinear structure that linear cannot."
        )


@pytest.mark.slow
class TestWulfmeierObjectworld:
    """Sanity check that deep MCE-IRL runs on Objectworld without errors.

    Objectworld has distance-based features where the reward depends on
    proximity to objects of different colors. This test verifies that the
    neural estimator produces a reasonable policy (finite EVD) on this
    environment.
    """

    def test_deep_runs_on_objectworld(self):
        np.random.seed(42)

        # Create environment and generate demonstrations
        env = ObjectworldEnvironment(
            grid_size=8, feature_type="continuous", seed=42
        )
        panel = env.simulate_demonstrations(n_demos=32, max_steps=30, seed=0)
        df = _panel_to_df(panel)

        n_states = env.num_states
        n_actions = env.num_actions
        n_features = env.feature_matrix.shape[2]
        transitions = env.transition_matrices
        features = env.feature_matrix

        # Build a state encoder from the environment features
        state_features = features[:, 0, :]  # (S, K) -- same across actions
        def objectworld_encoder(states: jnp.ndarray) -> jnp.ndarray:
            return state_features[states.astype(jnp.int32)]

        # Run deep MCE-IRL
        np.random.seed(42)
        deep_model = MCEIRLNeural(
            n_states=n_states,
            n_actions=n_actions,
            discount=0.9,
            reward_type="state",
            reward_hidden_dim=32,
            reward_num_layers=2,
            max_epochs=200,
            lr=0.01,
            state_encoder=objectworld_encoder,
            state_dim=n_features,
            verbose=False,
        )
        deep_model.fit(
            df,
            state="state",
            action="action",
            id="agent_id",
            transitions=np.asarray(transitions),
            features=features,
        )
        deep_policy = jnp.array(deep_model.policy_).astype(jnp.float32)

        evd = _compute_evd(
            env.true_reward, deep_policy, transitions, env.problem_spec
        )

        print(f"\nObjectworld 8x8 EVD results:")
        print(f"  Deep MCE-IRL EVD: {evd:.4f}")

        # Sanity check: EVD should be finite and non-negative
        assert 0 <= evd < 100, (
            f"Expected EVD in [0, 100), got {evd:.4f}. "
            f"The neural reward should produce a reasonable policy."
        )
