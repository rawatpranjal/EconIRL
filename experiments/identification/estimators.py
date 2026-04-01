"""Unified estimator wrappers for identification experiments.

Each method returns a dict with:
    - name: method identifier
    - reward_matrix: (S, A) recovered reward (or None for RF-Q)
    - q_table: (S, A) Q-values (or None if not applicable)
    - policy: (S, A) estimated CCPs
    - value_function: (S,) estimated V
    - log_likelihood: scalar

The oracle computes the true policy from known reward and transitions.
All estimation methods take (panel, env) and return the same dict format.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem

from .config import EstimationConfig
from .environment import SerializedContentEnvironment, BUY, WAIT, EXIT
from .iq_learn_jax import estimate_iq_learn, IQLearnJAXConfig


def estimate_oracle(env: SerializedContentEnvironment) -> dict:
    """Oracle: compute true policy from known reward and transitions."""
    problem = env.problem_spec
    operator = SoftBellmanOperator(problem, env.transition_matrices)
    result = value_iteration(operator, env.reward_matrix, tol=1e-10, max_iter=10000)

    log_probs = jax.nn.log_softmax(
        result.Q / problem.scale_parameter, axis=1
    )

    return {
        "name": "Oracle",
        "reward_matrix": env.reward_matrix,
        "q_table": result.Q,
        "policy": result.policy,
        "value_function": result.V,
        "log_likelihood": None,
    }


def estimate_reduced_form_q(
    panel,
    env: SerializedContentEnvironment,
    config: EstimationConfig | None = None,
) -> dict:
    """Reduced-form Q: fit Q(s,a) via MLE logit (behavioral cloning).

    This is John's proposed simplification. Parametrize Q(s,a) as a
    free tabular matrix and maximize the log-likelihood of observed
    choices. The resulting Q recovers the advantage function A*(s,a)
    but cannot separate reward from continuation values.
    """
    if config is None:
        config = EstimationConfig()

    n_states = env.num_states
    n_actions = env.num_actions
    sigma = env._scale_parameter

    states = np.asarray(panel.get_all_states(), dtype=np.int32)
    actions = np.asarray(panel.get_all_actions(), dtype=np.int32)

    # Initialize Q as zeros
    q_params = jnp.zeros(n_states * n_actions)
    optimizer = optax.adam(config.rf_learning_rate)
    opt_state = optimizer.init(q_params)

    states_jnp = jnp.array(states)
    actions_jnp = jnp.array(actions)

    def nll_fn(q_flat):
        Q = q_flat.reshape(n_states, n_actions)
        log_probs = jax.nn.log_softmax(Q / sigma, axis=1)
        return -log_probs[states_jnp, actions_jnp].mean()

    nll_and_grad = jax.jit(jax.value_and_grad(nll_fn))

    for iteration in range(config.rf_max_iter):
        loss, grads = nll_and_grad(q_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        q_params = optax.apply_updates(q_params, updates)

        if float(jnp.linalg.norm(grads)) < 1e-6:
            break

    Q_table = q_params.reshape(n_states, n_actions)
    policy = jax.nn.softmax(Q_table / sigma, axis=1)
    V = sigma * jax.nn.logsumexp(Q_table / sigma, axis=1)

    log_probs = jax.nn.log_softmax(Q_table / sigma, axis=1)
    ll = float(log_probs[states_jnp, actions_jnp].sum())

    return {
        "name": "Reduced-Form Q",
        "reward_matrix": None,  # RF-Q cannot recover reward
        "q_table": Q_table,
        "policy": policy,
        "value_function": V,
        "log_likelihood": ll,
    }


def estimate_airl_with_anchors(
    panel,
    env: SerializedContentEnvironment,
    config: EstimationConfig | None = None,
) -> dict:
    """AIRL with anchor enforcement (LSW Theorems 1-3).

    Runs vanilla AIRL with tabular reward, then post-processes:
    1. Zero out reward at exit action for all states
    2. Zero out reward at absorbing state for all actions
    3. Re-solve Bellman with these constraints

    The anchor constraints make g* = r* exactly.
    """
    if config is None:
        config = EstimationConfig()

    from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig

    airl_config = AIRLConfig(
        reward_type="tabular",
        reward_lr=config.airl_reward_lr,
        discriminator_steps=config.airl_disc_steps,
        max_rounds=config.airl_max_rounds,
        use_shaping=True,
        convergence_tol=config.airl_convergence_tol,
        compute_se=False,
        verbose=False,
    )

    from econirl.preferences.action_reward import ActionDependentReward
    utility = ActionDependentReward(env.feature_matrix, env.parameter_names)
    problem = env.problem_spec
    transitions = env.transition_matrices

    estimator = AIRLEstimator(config=airl_config)
    summary = estimator.estimate(panel, utility, problem, transitions)

    # Extract raw reward matrix from metadata
    reward_raw = jnp.array(summary.metadata["reward_matrix"])

    # Anchor enforcement: zero exit action and absorbing state
    absorbing = env.config.absorbing_state
    reward_anchored = jnp.array(reward_raw)
    reward_anchored = reward_anchored.at[:, EXIT].set(0.0)
    reward_anchored = reward_anchored.at[absorbing, :].set(0.0)

    # Re-solve with anchored reward
    operator = SoftBellmanOperator(problem, transitions)
    vi_result = value_iteration(operator, reward_anchored, tol=1e-10, max_iter=10000)

    return {
        "name": "AIRL + Anchors",
        "reward_matrix": reward_anchored,
        "q_table": vi_result.Q,
        "policy": vi_result.policy,
        "value_function": vi_result.V,
        "log_likelihood": summary.log_likelihood,
    }


def estimate_airl_no_anchors(
    panel,
    env: SerializedContentEnvironment,
    config: EstimationConfig | None = None,
) -> dict:
    """AIRL without anchor enforcement.

    Standard AIRL that recovers a shaped reward g* = r* + shaping residual.
    Works in-sample but fails on Type II counterfactuals because the
    shaping term contaminates the re-solved Bellman equation.
    """
    if config is None:
        config = EstimationConfig()

    from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig

    airl_config = AIRLConfig(
        reward_type="tabular",
        reward_lr=config.airl_reward_lr,
        discriminator_steps=config.airl_disc_steps,
        max_rounds=config.airl_max_rounds,
        use_shaping=True,
        convergence_tol=config.airl_convergence_tol,
        compute_se=False,
        verbose=False,
    )

    from econirl.preferences.action_reward import ActionDependentReward
    utility = ActionDependentReward(env.feature_matrix, env.parameter_names)
    problem = env.problem_spec
    transitions = env.transition_matrices

    estimator = AIRLEstimator(config=airl_config)
    summary = estimator.estimate(panel, utility, problem, transitions)

    reward_raw = jnp.array(summary.metadata["reward_matrix"])

    # No anchor enforcement: use reward as-is
    operator = SoftBellmanOperator(problem, transitions)
    vi_result = value_iteration(operator, reward_raw, tol=1e-10, max_iter=10000)

    return {
        "name": "AIRL No Anchors",
        "reward_matrix": reward_raw,
        "q_table": vi_result.Q,
        "policy": vi_result.policy,
        "value_function": vi_result.V,
        "log_likelihood": summary.log_likelihood,
    }


def estimate_iq_learn(
    panel,
    env: SerializedContentEnvironment,
    config: EstimationConfig | None = None,
) -> dict:
    """IQ-Learn: inverse soft-Q learning (Garg et al. 2021).

    Learns Q(s,a) by enforcing Bellman consistency, then recovers
    reward via the inverse Bellman operator. No adversarial training.
    """
    if config is None:
        config = EstimationConfig()

    from .iq_learn_jax import estimate_iq_learn as _iq_learn, IQLearnJAXConfig

    iq_config = IQLearnJAXConfig(
        alpha=config.iq_alpha,
        learning_rate=config.iq_learning_rate,
        max_iter=config.iq_max_iter,
    )

    states = panel.get_all_states()
    actions = panel.get_all_actions()
    next_states = panel.get_all_next_states()

    result = _iq_learn(
        expert_states=states,
        expert_actions=actions,
        expert_next_states=next_states,
        transitions=env.transition_matrices,
        n_states=env.num_states,
        n_actions=env.num_actions,
        gamma=env._discount_factor,
        sigma=env._scale_parameter,
        config=iq_config,
    )

    return {
        "name": "IQ-Learn",
        "reward_matrix": result["reward_matrix"],
        "q_table": result["q_table"],
        "policy": result["policy"],
        "value_function": result["value_function"],
        "log_likelihood": result["log_likelihood"],
    }


# --- Analytical / population-level methods ---

def oracle_counterfactual_ccps(
    env: SerializedContentEnvironment,
    shifted_reward: jnp.ndarray | None = None,
    new_transitions: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute oracle CCPs under counterfactual using true reward.

    For Type I: provide shifted_reward with original transitions.
    For Type II: provide new_transitions with true reward.
    """
    problem = env.problem_spec
    reward = shifted_reward if shifted_reward is not None else env.reward_matrix
    transitions = new_transitions if new_transitions is not None else env.transition_matrices

    operator = SoftBellmanOperator(problem, transitions)
    result = value_iteration(operator, reward, tol=1e-10, max_iter=10000)
    return result.policy
