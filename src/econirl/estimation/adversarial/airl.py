"""Adversarial Inverse Reinforcement Learning (AIRL) for tabular MDPs.

This module implements AIRL (Fu et al. 2018) adapted for discrete choice models.
AIRL recovers a reward function that is robust to changes in dynamics by using
a specific discriminator structure that disentangles reward from shaping.

Algorithm:
    1. Initialize reward function r(s,a), shaping potential h(s), and policy pi
    2. Repeat:
       a) Compute discriminator: D(s,a,s') = exp(f) / (exp(f) + pi(a|s))
          where f(s,a,s') = r(s,a) + gamma*h(s') - h(s)
       b) Update discriminator to classify expert vs policy
       c) Update policy using the configured generator reward

Reference:
    Fu, J., Luo, K., & Levine, S. (2018). "Learning robust rewards with
    adversarial inverse reinforcement learning." ICLR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
from tqdm import tqdm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import backward_induction, hybrid_iteration, value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.adversarial.base import AdversarialEstimatorBase
from econirl.estimation.base import EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.base import BaseUtilityFunction
from econirl.preferences.reward import LinearReward


@dataclass
class AIRLConfig:
    """Configuration for tabular AIRL.

    Attributes:
        reward_type: Parameterization of reward ("tabular" or "linear")
        reward_arg: Whether reward is state-only or state-action.
        anchor_action: Optional action whose reward is pinned to zero.
            Only used for state-action rewards.
        absorbing_state: Optional absorbing state whose reward row is pinned
            to zero when an anchor action is used.
        reward_lr: Learning rate for reward updates
        discriminator_steps: Discriminator updates per round
        generator_solver: Inner solver for policy
        generator_tol: Tolerance for value iteration
        generator_max_iter: Max iterations for value iteration
        max_rounds: Maximum training rounds
        use_shaping: Whether to use potential shaping f = r + gamma*h(s') - h(s)
        shaping_coef: Coefficient for shaping term (typically gamma)
        shaping_l2_penalty: Small L2 penalty on shaping/reward parameters to
            remove gauge drift during adversarial training.
        shaping_bellman_penalty: Optional penalty tying the shaping potential to
            the soft Bellman value of the recovered reward. Default 0 preserves
            the original unconstrained AIRL discriminator.
        generator_reward: Reward used by the policy update. "recovered" solves
            the current recovered reward g. "log_odds" uses AIRL's discriminator
            log-odds reward f - log pi. "f" uses the shaped discriminator score.
        discriminator_data_mode: Source for discriminator negatives. "sampled"
            uses sampled policy transitions. "occupancy" uses exact finite-horizon
            tabular policy transition occupancy. "conditional_occupancy" uses
            exact tabular action/next-state conditionals given each observed
            state, removing state-marginal occupancy differences.
        policy_sample_mode: How to draw policy negatives for the discriminator.
            "chain" preserves the historical single-chain sampler; "rollout"
            draws one rollout per expert trajectory from the empirical initial
            distribution, matching tabular demonstrations with fixed starts.
        negative_history_size: Number of previous policy-sample batches to mix
            into the discriminator's negative class. Fu et al. use the previous
            20 iterations in their experiment details; default 0 preserves the
            historical econirl training path.
        min_rounds: Minimum adversarial rounds before policy-change convergence
            can stop training.
        convergence_tol: Tolerance for policy convergence
        compute_se: Whether to compute standard errors
        se_method: Method for standard errors
        n_bootstrap: Number of bootstrap samples
        verbose: Whether to print progress
    """

    reward_type: Literal["tabular", "linear"] = "tabular"
    reward_arg: Literal["state", "state_action"] = "state"
    """Reward parametrization. Per Fu et al. (2018) Theorems 5.1-5.2 the
    disentanglement / dynamics-transfer guarantees only hold when the
    reward is a function of state alone, g_theta(s). State-action rewards
    g_theta(s, a) recover a shaped advantage and lose the transfer
    property. Default 'state' matches the original paper; 'state_action'
    is the legacy econirl behavior."""
    anchor_action: int | None = None
    absorbing_state: int | None = None
    reward_lr: float = 0.01
    reward_weight_decay: float = 0.0  # L2 regularization on reward params
    discriminator_steps: int = 5
    generator_solver: Literal["value", "hybrid"] = "hybrid"
    generator_tol: float = 1e-8
    generator_max_iter: int = 5000
    policy_step_size: float = 1.0  # Conservative policy iteration mixing.
    # 1.0 = full VI update (original). 0.1 = mix 10% new, 90% old.
    # Lower values prevent reward divergence in tabular settings by
    # dampening the policy update, mimicking PPO's small steps.
    max_rounds: int = 200
    use_shaping: bool = True
    shaping_coef: float | None = None  # If None, uses discount_factor
    shaping_l2_penalty: float = 1e-8
    shaping_bellman_penalty: float = 0.0
    generator_reward: Literal["recovered", "log_odds", "f"] = "recovered"
    discriminator_data_mode: Literal["sampled", "occupancy", "conditional_occupancy"] = "sampled"
    policy_sample_mode: Literal["chain", "rollout"] = "chain"
    negative_history_size: int = 0
    min_rounds: int = 20
    convergence_tol: float = 1e-4
    compute_se: bool = True
    se_method: Literal["bootstrap", "asymptotic"] = "bootstrap"
    n_bootstrap: int = 100
    verbose: bool = False

    # Unified public AIRL facade. These fields are ignored by the legacy
    # AIRLEstimator concrete class and consumed by the AIRL facade below.
    version: Literal["state_only", "anchored", "heterogeneous"] = "state_only"
    num_segments: int = 1
    exit_action: int | None = None
    max_em_iterations: int = 50
    em_convergence_tol: float = 1e-3
    consistency_weight: float = 0.1
    prior_smoothing: float = 0.01
    prior_min: float = 0.0
    prior_damping: float = 0.0
    normalize_reward: bool = False
    unit_normalize_reward: bool = False
    gradient_clip_norm: float = 0.0
    antisymmetric_init: bool = False
    initialization: Literal["random", "behavioral_anchor"] = "random"
    initialization_smoothing: float = 1.0
    initialization_l2_penalty: float = 0.0
    seed: int = 42


class AIRLEstimator(AdversarialEstimatorBase):
    """Adversarial Inverse Reinforcement Learning for tabular MDPs.

    AIRL learns a disentangled reward function that is robust to changes
    in dynamics. The key insight is using a discriminator of the form:

        D(s,a,s') = exp(f) / (exp(f) + pi(a|s))

    where f(s,a,s') = r(s,a) + gamma*h(s') - h(s).

    This structure allows recovery of the reward r(s,a) independent of
    the shaping term h.

    Parameters
    ----------
    config : AIRLConfig, optional
        Configuration object with algorithm parameters.
    **kwargs
        Override individual config parameters.

    Examples
    --------
    >>> from econirl.estimation.adversarial import AIRLEstimator, AIRLConfig
    >>> config = AIRLConfig(max_rounds=100, verbose=True)
    >>> estimator = AIRLEstimator(config=config)
    >>> result = estimator.estimate(panel, reward_fn, problem, transitions)
    """

    def __init__(
        self,
        config: AIRLConfig | None = None,
        **kwargs,
    ):
        if config is None:
            config = AIRLConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        super().__init__(
            se_method=config.se_method if config.compute_se else "asymptotic",
            compute_hessian=False,
            verbose=config.verbose,
        )
        self.config = config

    @property
    def name(self) -> str:
        return "AIRL (Fu et al. 2018)"

    def estimate(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate reward function using AIRL.

        Overrides base class to handle reward parameters properly.

        Args:
            panel: Expert demonstrations
            utility: Utility/reward function specification
            problem: Problem specification
            transitions: Transition matrices
            initial_params: Initial parameters (optional)

        Returns:
            EstimationSummary with learned parameters and policy
        """
        import time as time_module

        start_time = time_module.time()

        # Coerce transitions to a JAX array so a plain numpy (A, S, S) array does
        # not crash deep in the traced scan with a cryptic TracerArrayConversionError.
        transitions = jnp.asarray(transitions, dtype=jnp.float32)

        # Run optimization
        result = self._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            initial_params=initial_params,
            **kwargs,
        )

        # Generate parameter names
        if self.config.reward_type == "linear":
            param_names = utility.parameter_names
        else:
            # Tabular reward: one parameter per (state, action) pair
            param_names = [
                f"R({s},{a})" for s in range(problem.num_states) for a in range(problem.num_actions)
            ]

        # Create standard errors (NaN for adversarial methods)
        standard_errors = jnp.full_like(result.parameters, float("nan"))

        # Goodness of fit
        n_obs = panel.num_observations
        n_params = len(result.parameters)
        ll = result.log_likelihood

        goodness_of_fit = GoodnessOfFit(
            log_likelihood=ll,
            num_parameters=n_params,
            num_observations=n_obs,
            aic=-2 * ll + 2 * n_params,
            bic=-2 * ll + n_params * jnp.log(jnp.array(n_obs)).item(),
            prediction_accuracy=self._compute_prediction_accuracy(panel, result.policy),
        )

        total_time = time_module.time() - start_time

        # Expanded diagnostics: data / pre-estimation / first-stage transition.
        # Auxiliary reporting only -- never let it break a real fit.
        from econirl.inference.results import compute_fit_diagnostics

        dataset = pre_estimation = transition_first_stage = None
        try:
            feature_matrix = getattr(utility, "feature_matrix", None)
            dataset, pre_estimation, transition_first_stage = compute_fit_diagnostics(
                panel,
                problem.num_states,
                problem.num_actions,
                feature_matrix=feature_matrix,
            )
        except Exception:  # noqa: BLE001 - diagnostics are non-critical
            pass

        return EstimationSummary(
            parameters=result.parameters,
            parameter_names=param_names,
            standard_errors=standard_errors,
            hessian=None,
            variance_covariance=None,
            method=self.name,
            num_observations=n_obs,
            num_individuals=panel.num_individuals,
            num_periods=max(panel.num_periods_per_individual),
            discount_factor=problem.discount_factor,
            scale_parameter=problem.scale_parameter,
            log_likelihood=ll,
            goodness_of_fit=goodness_of_fit,
            identification=None,
            converged=result.converged,
            num_iterations=result.num_iterations,
            convergence_message=result.message,
            value_function=result.value_function,
            policy=result.policy,
            estimation_time=total_time,
            num_states=problem.num_states,
            num_actions=problem.num_actions,
            dataset=dataset,
            pre_estimation=pre_estimation,
            transition_first_stage=transition_first_stage,
            metadata=result.metadata,
        )

    def _sample_transitions_from_panel(
        self,
        panel: Panel,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample (s, a, s') transitions from expert demonstrations.

        Returns:
            Tuple of (states, actions, next_states) tensors
        """
        return (
            panel.get_all_states(),
            panel.get_all_actions(),
            panel.get_all_next_states(),
        )

    def _sample_policy_rollouts_like_panel(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        trajectory_lengths: jnp.ndarray,
        initial_dist: jnp.ndarray,
        key: jax.Array,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample policy transitions with one rollout per expert trajectory."""
        n_trajectories = int(trajectory_lengths.shape[0])
        max_periods = int(jnp.max(trajectory_lengths))

        rollout_keys = jax.random.split(key, n_trajectories)

        def one_rollout(rollout_key):
            init_key, scan_key = jax.random.split(rollout_key)
            init_state = jax.random.categorical(init_key, jnp.log(initial_dist + 1e-10)).astype(
                jnp.int32
            )
            step_keys = jax.random.split(scan_key, max_periods)

            def step_fn(carry, step_key):
                state = carry
                k1, k2 = jax.random.split(step_key)
                action = jax.random.categorical(k1, jnp.log(policy[state] + 1e-10))
                next_state = jax.random.categorical(k2, jnp.log(transitions[action, state] + 1e-10))
                return next_state.astype(jnp.int32), (
                    state,
                    action.astype(jnp.int32),
                    next_state.astype(jnp.int32),
                )

            _, rollout = jax.lax.scan(step_fn, init_state, step_keys)
            return rollout

        states, actions, next_states = jax.vmap(one_rollout)(rollout_keys)
        mask = jnp.arange(max_periods)[None, :] < trajectory_lengths[:, None]
        return states[mask], actions[mask], next_states[mask]

    def _panel_transition_weights(
        self,
        panel: Panel,
        n_states: int,
        n_actions: int,
    ) -> jnp.ndarray:
        """Empirical transition distribution as weights over (s, a, s')."""
        states, actions, next_states = self._sample_transitions_from_panel(panel)
        weights = jnp.zeros((n_states, n_actions, n_states), dtype=jnp.float32)
        weights = weights.at[states, actions, next_states].add(1.0)
        return weights / jnp.maximum(weights.sum(), 1.0)

    def _policy_transition_occupancy(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        initial_dist: jnp.ndarray,
        trajectory_lengths: jnp.ndarray,
    ) -> jnp.ndarray:
        """Exact finite-horizon policy transition occupancy over (s, a, s')."""
        max_periods = int(jnp.max(trajectory_lengths))
        total_steps = jnp.maximum(jnp.sum(trajectory_lengths), 1)
        trans_sas = jnp.transpose(transitions, (1, 0, 2))

        weights = jnp.zeros_like(trans_sas)
        state_dist = initial_dist
        for period in range(max_periods):
            active = jnp.sum(trajectory_lengths > period)
            step_mass = active / total_steps
            state_action = state_dist[:, None] * policy
            weights = weights + step_mass * state_action[:, :, None] * trans_sas
            state_dist = jnp.einsum("sa,ast->t", policy * state_dist[:, None], transitions)
        return weights / jnp.maximum(weights.sum(), 1e-12)

    # Uses _sample_transitions_from_policy from AdversarialEstimatorBase (lax.scan)

    def _compute_airl_logits(
        self,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        next_states: jnp.ndarray,
        reward_matrix: jnp.ndarray,
        V: jnp.ndarray,
        policy: jnp.ndarray,
        gamma: float,
    ) -> jnp.ndarray:
        """Compute AIRL discriminator logits.

        D(s,a,s') = sigmoid(f - log pi(a|s))
        where f = r(s,a) + gamma*V(s') - V(s)

        Returns logits = f - log pi(a|s)
        """
        # f(s,a,s') = r(s,a) + gamma*V(s') - V(s)
        # When reward_arg == "state" we project the reward onto the
        # state-only subspace per Fu et al. (2018) Theorems 5.1-5.2.
        if self.config.reward_arg == "state":
            r_state = reward_matrix.mean(axis=1)
            r_sa = r_state[states]
        else:
            r_sa = reward_matrix[states, actions]
        if self.config.use_shaping:
            shaping_coef = self.config.shaping_coef if self.config.shaping_coef else gamma
            f = r_sa + shaping_coef * V[next_states] - V[states]
        else:
            f = r_sa

        # log pi(a|s)
        log_pi = jnp.log(policy[states, actions] + 1e-10)

        # AIRL logit
        return f - log_pi

    def _compute_initial_distribution(
        self,
        panel: Panel,
        n_states: int,
    ) -> jnp.ndarray:
        """Compute initial state distribution from data."""
        init_states = jnp.array(
            [traj.states[0].item() for traj in panel.trajectories if len(traj) > 0],
            dtype=jnp.int32,
        )
        counts = jnp.zeros(n_states, dtype=jnp.float32)
        counts = counts.at[init_states].add(1.0)

        if counts.sum() > 0:
            return counts / counts.sum()
        return jnp.ones(n_states) / n_states

    def _compute_policy(
        self,
        reward_matrix: jnp.ndarray,
        operator: SoftBellmanOperator,
        num_periods: int | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute optimal policy given reward matrix.

        For finite horizon (num_periods set), uses backward induction and
        returns period-0 policy/value for compatibility.
        """
        if num_periods is not None:
            fh_result = backward_induction(operator, reward_matrix, num_periods)
            return fh_result.policy[0], fh_result.V[0]

        if self.config.generator_solver == "hybrid":
            result = hybrid_iteration(
                operator,
                reward_matrix,
                tol=self.config.generator_tol,
                max_iter=self.config.generator_max_iter,
            )
        else:
            result = value_iteration(
                operator,
                reward_matrix,
                tol=self.config.generator_tol,
                max_iter=self.config.generator_max_iter,
            )
        return result.policy, result.V

    def _compute_generator_reward(
        self,
        reward_matrix: jnp.ndarray,
        shaping_potential: jnp.ndarray,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        gamma: float,
    ) -> jnp.ndarray:
        """Compute the state-action reward used by the generator update."""
        if not self.config.use_shaping:
            shaped_score = reward_matrix
        else:
            shaping_coef = (
                self.config.shaping_coef if self.config.shaping_coef is not None else gamma
            )
            expected_next_potential = jnp.einsum("ast,t->sa", transitions, shaping_potential)
            shaped_score = (
                reward_matrix + shaping_coef * expected_next_potential - shaping_potential[:, None]
            )

        if self.config.generator_reward == "recovered":
            return reward_matrix
        if self.config.generator_reward == "f":
            return shaped_score
        if self.config.generator_reward == "log_odds":
            return shaped_score - jnp.log(policy + 1e-10)
        raise ValueError(f"unknown generator_reward={self.config.generator_reward!r}")

    def _enforce_anchor_reward(self, reward_matrix: jnp.ndarray) -> jnp.ndarray:
        """Apply AIRL anchor normalization for state-action rewards.

        This mirrors the anchored AIRL gauge: one action is pinned to zero reward,
        and the absorbing state's reward row is pinned to zero when available.
        The anchor is deliberately not applied to state-only AIRL because doing
        so would turn a state reward into an action-dependent object.
        """
        if self.config.anchor_action is None:
            return reward_matrix
        if self.config.reward_arg != "state_action":
            raise ValueError("anchor_action is only valid for state_action AIRL")

        anchor_action = int(self.config.anchor_action)
        if not 0 <= anchor_action < reward_matrix.shape[1]:
            raise ValueError(f"anchor_action={anchor_action} is outside the action space")
        anchored = reward_matrix.at[:, anchor_action].set(0.0)
        if self.config.absorbing_state is not None:
            absorbing = int(self.config.absorbing_state)
            if not 0 <= absorbing < reward_matrix.shape[0]:
                raise ValueError(f"absorbing_state={absorbing} is outside the state space")
            anchored = anchored.at[absorbing, :].set(0.0)
        return anchored

    def _optimize(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run AIRL optimization.

        Args:
            panel: Expert demonstrations
            utility: Reward function specification
            problem: Problem specification
            transitions: Transition matrices
            initial_params: Initial reward parameters (optional)

        Returns:
            EstimationResult with learned reward and policy
        """
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        gamma = problem.discount_factor
        operator = SoftBellmanOperator(problem, transitions)

        import optax

        # Initialize reward parameters with optax Adam
        if self.config.reward_type == "linear":
            if isinstance(utility, ActionDependentReward):
                feature_matrix = utility.feature_matrix
                n_features = feature_matrix.shape[2]
            elif isinstance(utility, LinearReward):
                sf = utility.state_features
                feature_matrix = jnp.broadcast_to(
                    sf[:, None, :], (sf.shape[0], n_actions, sf.shape[1])
                ).copy()
                n_features = sf.shape[1]
            else:
                raise TypeError(f"Unsupported utility type: {type(utility)}")

            if initial_params is not None:
                reward_params = jnp.array(initial_params, dtype=jnp.float32)
            else:
                reward_params = jnp.zeros(n_features)

            if self.config.reward_weight_decay > 0:
                optimizer = optax.adamw(
                    self.config.reward_lr,
                    weight_decay=self.config.reward_weight_decay,
                )
            else:
                optimizer = optax.adam(self.config.reward_lr)

            def get_reward_matrix(params):
                return jnp.einsum("sak,k->sa", feature_matrix, params)

        else:
            # Tabular reward
            if initial_params is None:
                reward_params = jnp.zeros((n_states, n_actions))
            else:
                initial = jnp.asarray(initial_params, dtype=jnp.float32)
                if initial.shape == (n_states, n_actions):
                    reward_params = initial
                elif initial.shape == (n_states,):
                    reward_params = jnp.broadcast_to(initial[:, None], (n_states, n_actions)).copy()
                elif initial.size == n_states * n_actions:
                    reward_params = initial.reshape((n_states, n_actions))
                else:
                    raise ValueError(
                        "tabular AIRL initial_params must have shape "
                        f"({n_states}, {n_actions}), ({n_states},), or "
                        f"flat length {n_states * n_actions}; got {initial.shape}"
                    )
            feature_matrix = None

            if self.config.reward_weight_decay > 0:
                optimizer = optax.adamw(
                    self.config.reward_lr,
                    weight_decay=self.config.reward_weight_decay,
                )
            else:
                optimizer = optax.adam(self.config.reward_lr)

            def get_reward_matrix(params):
                return params

        disc_params = {
            "reward": reward_params,
            "shaping": jnp.zeros(n_states, dtype=jnp.float32),
        }
        opt_state = optimizer.init(disc_params)

        # Initial state distribution
        initial_dist = self._compute_initial_distribution(panel, n_states)

        # Sample expert transitions once
        expert_states, expert_actions, expert_next_states = self._sample_transitions_from_panel(
            panel
        )
        n_expert = len(expert_states)
        trajectory_lengths = jnp.asarray(
            [len(traj) for traj in panel.trajectories if len(traj) > 0],
            dtype=jnp.int32,
        )

        # Initialize policy
        policy = jnp.ones((n_states, n_actions)) / n_actions

        # AIRL discriminator loss (differentiable w.r.t. reward and h params)
        use_shaping = self.config.use_shaping
        shaping_coef = self.config.shaping_coef
        shaping_l2_penalty = self.config.shaping_l2_penalty
        shaping_bellman_penalty = self.config.shaping_bellman_penalty

        reward_arg_state = self.config.reward_arg == "state"
        negative_history_size = max(0, int(self.config.negative_history_size))
        negative_history: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []
        occupancy_history: list[jnp.ndarray] = []
        expert_transition_weights = self._panel_transition_weights(panel, n_states, n_actions)
        expert_state_weights = expert_transition_weights.sum(axis=(1, 2))
        expert_conditional_transition_weights = (
            expert_transition_weights / jnp.maximum(expert_state_weights[:, None, None], 1e-12)
        ) * expert_state_weights[:, None, None]

        def disc_loss_fn(
            params,
            policy_fixed,
            exp_s,
            exp_a,
            exp_ns,
            pol_s,
            pol_a,
            pol_ns,
            pol_log_pi,
        ):
            reward_matrix = get_reward_matrix(params["reward"])
            if reward_arg_state:
                reward_matrix = jnp.broadcast_to(
                    reward_matrix.mean(axis=1, keepdims=True),
                    reward_matrix.shape,
                )
            else:
                reward_matrix = self._enforce_anchor_reward(reward_matrix)
            shaping_potential = params["shaping"]

            def expert_logits(states, actions, next_states):
                r_sa = reward_matrix[states, actions]
                if use_shaping:
                    sc = shaping_coef if shaping_coef is not None else gamma
                    f = r_sa + sc * shaping_potential[next_states] - shaping_potential[states]
                else:
                    f = r_sa
                log_pi = jnp.log(policy_fixed[states, actions] + 1e-10)
                return f - log_pi

            def policy_logits(states, actions, next_states, log_pi):
                r_sa = reward_matrix[states, actions]
                if use_shaping:
                    sc = shaping_coef if shaping_coef is not None else gamma
                    f = r_sa + sc * shaping_potential[next_states] - shaping_potential[states]
                else:
                    f = r_sa
                return f - log_pi

            e_logits = expert_logits(exp_s, exp_a, exp_ns)
            p_logits = policy_logits(pol_s, pol_a, pol_ns, pol_log_pi)

            e_loss = jnp.mean(jnp.logaddexp(0.0, -e_logits))
            p_loss = jnp.mean(jnp.logaddexp(0.0, p_logits))
            l2_penalty = shaping_l2_penalty * (
                jnp.mean(reward_matrix**2) + jnp.mean(shaping_potential**2)
            )
            bellman_penalty = 0.0
            if use_shaping and shaping_bellman_penalty > 0:
                sc = shaping_coef if shaping_coef is not None else gamma
                expected_next = jnp.einsum("ast,t->sa", transitions, shaping_potential)
                bellman_target = jax.nn.logsumexp(reward_matrix + sc * expected_next, axis=1)
                bellman_penalty = shaping_bellman_penalty * jnp.mean(
                    (shaping_potential - bellman_target) ** 2
                )
            return e_loss + p_loss + l2_penalty + bellman_penalty

        disc_loss_and_grad = jax.value_and_grad(disc_loss_fn)

        def occupancy_disc_loss_fn(
            params,
            policy_fixed,
            expert_weights,
            policy_weight_batches,
            policy_log_prob_batches,
        ):
            reward_matrix = get_reward_matrix(params["reward"])
            if reward_arg_state:
                reward_matrix = jnp.broadcast_to(
                    reward_matrix.mean(axis=1, keepdims=True),
                    reward_matrix.shape,
                )
            else:
                reward_matrix = self._enforce_anchor_reward(reward_matrix)
            shaping_potential = params["shaping"]

            if use_shaping:
                sc = shaping_coef if shaping_coef is not None else gamma
                f = (
                    reward_matrix[:, :, None]
                    + sc * shaping_potential[None, None, :]
                    - shaping_potential[:, None, None]
                )
            else:
                f = reward_matrix[:, :, None]

            expert_log_pi = jnp.log(policy_fixed + 1e-10)[:, :, None]
            e_logits = f - expert_log_pi
            p_logits = f[None, :, :, :] - policy_log_prob_batches[:, :, :, None]

            e_loss = jnp.sum(expert_weights * jnp.logaddexp(0.0, -e_logits))
            p_loss = (
                jnp.sum(policy_weight_batches * jnp.logaddexp(0.0, p_logits))
                / policy_weight_batches.shape[0]
            )
            l2_penalty = shaping_l2_penalty * (
                jnp.mean(reward_matrix**2) + jnp.mean(shaping_potential**2)
            )
            bellman_penalty = 0.0
            if use_shaping and shaping_bellman_penalty > 0:
                sc = shaping_coef if shaping_coef is not None else gamma
                expected_next = jnp.einsum("ast,t->sa", transitions, shaping_potential)
                bellman_target = jax.nn.logsumexp(reward_matrix + sc * expected_next, axis=1)
                bellman_penalty = shaping_bellman_penalty * jnp.mean(
                    (shaping_potential - bellman_target) ** 2
                )
            return e_loss + p_loss + l2_penalty + bellman_penalty

        occupancy_disc_loss_and_grad = jax.value_and_grad(occupancy_disc_loss_fn)

        # Training metrics
        disc_losses = []
        policy_changes = []
        converged = False
        round_idx = 0
        key = jax.random.key(42)

        pbar = tqdm(
            range(self.config.max_rounds),
            desc="AIRL",
            disable=not self.config.verbose,
        )

        for round_idx in pbar:
            old_policy = jnp.array(policy)

            # Update reward and shaping potential via optax Adam.
            disc_loss = 0.0
            if self.config.discriminator_data_mode in (
                "occupancy",
                "conditional_occupancy",
            ):
                if self.config.discriminator_data_mode == "conditional_occupancy":
                    trans_sas = jnp.transpose(transitions, (1, 0, 2))
                    expert_weights_for_loss = expert_conditional_transition_weights
                    policy_weights = (
                        expert_state_weights[:, None, None] * policy[:, :, None] * trans_sas
                    )
                else:
                    expert_weights_for_loss = expert_transition_weights
                    policy_weights = self._policy_transition_occupancy(
                        policy, transitions, initial_dist, trajectory_lengths
                    )
                policy_log_prob_matrix = jnp.log(policy + 1e-10)
                if occupancy_history:
                    history = occupancy_history[-negative_history_size:]
                    policy_weight_batches = jnp.stack([policy_weights, *history])
                else:
                    policy_weight_batches = policy_weights[None, :, :, :]
                policy_log_prob_batches = jnp.broadcast_to(
                    policy_log_prob_matrix,
                    (
                        policy_weight_batches.shape[0],
                        policy_log_prob_matrix.shape[0],
                        policy_log_prob_matrix.shape[1],
                    ),
                )

                for _ in range(self.config.discriminator_steps):
                    loss, grads = occupancy_disc_loss_and_grad(
                        disc_params,
                        policy,
                        expert_weights_for_loss,
                        policy_weight_batches,
                        policy_log_prob_batches,
                    )
                    updates, opt_state = optimizer.update(grads, opt_state, params=disc_params)
                    disc_params = optax.apply_updates(disc_params, updates)
                    disc_loss = float(loss)
                if negative_history_size:
                    occupancy_history.append(policy_weights)
                    if len(occupancy_history) > negative_history_size:
                        occupancy_history = occupancy_history[-negative_history_size:]
            else:
                # Sample from current policy using lax.scan
                key, subkey = jax.random.split(key)
                if self.config.policy_sample_mode == "rollout":
                    policy_states, policy_actions, policy_next_states = (
                        self._sample_policy_rollouts_like_panel(
                            policy,
                            transitions,
                            trajectory_lengths,
                            initial_dist,
                            subkey,
                        )
                    )
                else:
                    policy_states, policy_actions, policy_next_states = (
                        self._sample_transitions_from_policy(
                            policy, transitions, n_expert, initial_dist, subkey
                        )
                    )
                if negative_history:
                    history = negative_history[-negative_history_size:]
                    disc_policy_states = jnp.concatenate(
                        [policy_states, *(batch[0] for batch in history)]
                    )
                    disc_policy_actions = jnp.concatenate(
                        [policy_actions, *(batch[1] for batch in history)]
                    )
                    disc_policy_next_states = jnp.concatenate(
                        [policy_next_states, *(batch[2] for batch in history)]
                    )
                else:
                    disc_policy_states = policy_states
                    disc_policy_actions = policy_actions
                    disc_policy_next_states = policy_next_states
                disc_policy_log_probs = jnp.log(
                    policy[disc_policy_states, disc_policy_actions] + 1e-10
                )

                for _ in range(self.config.discriminator_steps):
                    loss, grads = disc_loss_and_grad(
                        disc_params,
                        policy,
                        expert_states,
                        expert_actions,
                        expert_next_states,
                        disc_policy_states,
                        disc_policy_actions,
                        disc_policy_next_states,
                        disc_policy_log_probs,
                    )
                    updates, opt_state = optimizer.update(grads, opt_state, params=disc_params)
                    disc_params = optax.apply_updates(disc_params, updates)
                    disc_loss = float(loss)
                if negative_history_size:
                    negative_history.append((policy_states, policy_actions, policy_next_states))
                    if len(negative_history) > negative_history_size:
                        negative_history = negative_history[-negative_history_size:]

            disc_losses.append(disc_loss)

            # Update policy via soft value iteration under the configured
            # generator reward. The recovered reward g remains the object
            # reported after training.
            current_reward = get_reward_matrix(disc_params["reward"])
            if reward_arg_state:
                current_reward = jnp.broadcast_to(
                    current_reward.mean(axis=1, keepdims=True),
                    current_reward.shape,
                )
            else:
                current_reward = self._enforce_anchor_reward(current_reward)
            shaping_potential = disc_params["shaping"]
            generator_reward = self._compute_generator_reward(
                current_reward,
                shaping_potential,
                policy,
                transitions,
                gamma,
            )
            new_policy, _ = self._compute_policy(
                generator_reward,
                operator,
                problem.num_periods,
            )

            # Conservative policy iteration: mix old and new policy
            alpha = self.config.policy_step_size
            if alpha < 1.0:
                policy = (1 - alpha) * old_policy + alpha * new_policy
                # Renormalize to valid distribution
                policy = policy / policy.sum(axis=1, keepdims=True)
            else:
                policy = new_policy

            # Check convergence
            policy_change = float(jnp.abs(policy - old_policy).max())
            policy_changes.append(policy_change)

            r_range = float(jnp.max(current_reward) - jnp.min(current_reward))
            pbar.set_postfix(
                {
                    "d_loss": f"{disc_loss:.4f}",
                    "d_pol": f"{policy_change:.4f}",
                    "R_rng": f"{r_range:.2f}",
                    "P(R|hi)": f"{float(policy[-10:, 1].mean()):.3f}",
                }
            )

            if (
                round_idx + 1 >= self.config.min_rounds
                and policy_change < self.config.convergence_tol
            ):
                converged = True
                break

        pbar.close()

        # Final values
        final_reward = get_reward_matrix(disc_params["reward"])
        if reward_arg_state:
            final_reward = jnp.broadcast_to(
                final_reward.mean(axis=1, keepdims=True),
                final_reward.shape,
            )
        else:
            final_reward = self._enforce_anchor_reward(final_reward)

        final_policy, final_value = self._compute_policy(
            final_reward,
            operator,
            problem.num_periods,
        )

        # Compute log-likelihood
        log_probs = operator.compute_log_choice_probabilities(final_reward, final_value)
        ll = float(log_probs[panel.get_all_states(), panel.get_all_actions()].sum())

        # Extract parameters
        if self.config.reward_type == "linear":
            parameters = jnp.array(disc_params["reward"])
        else:
            parameters = disc_params["reward"].flatten()

        optimization_time = time.time() - start_time

        return EstimationResult(
            parameters=parameters,
            log_likelihood=ll,
            value_function=final_value,
            policy=final_policy,
            hessian=None,
            converged=converged,
            num_iterations=round_idx + 1,
            num_function_evals=round_idx + 1,
            message="Converged" if converged else "Max rounds reached",
            optimization_time=optimization_time,
            metadata={
                "reward_type": self.config.reward_type,
                "reward_arg": self.config.reward_arg,
                "anchor_action": self.config.anchor_action,
                "absorbing_state": self.config.absorbing_state,
                "use_shaping": self.config.use_shaping,
                "learned_shaping": True,
                "shaping_bellman_penalty": self.config.shaping_bellman_penalty,
                "generator_reward": self.config.generator_reward,
                "discriminator_data_mode": self.config.discriminator_data_mode,
                "policy_sample_mode": self.config.policy_sample_mode,
                "negative_history_size": negative_history_size,
                "min_rounds": self.config.min_rounds,
                "final_disc_loss": disc_losses[-1] if disc_losses else None,
                "disc_losses": disc_losses,
                "policy_changes": policy_changes,
                "shaping_potential": jnp.array(disc_params["shaping"]).tolist(),
                "reward_matrix": jnp.array(final_reward).tolist(),
            },
        )


class AIRL:
    """Unified public AIRL entry point for identified AIRL variants.

    The facade exposes only the variants with a paper-backed identification
    claim:

    - ``version="state_only"``: Fu, Luo, and Levine (2018) state-only AIRL.
    - ``version="anchored"``: Lee, Sudhir, and Wang (2026) anchored
      action-dependent AIRL with one segment.
    - ``version="heterogeneous"``: Lee, Sudhir, and Wang (2026) anchored
      action-dependent AIRL with latent segments.

    The unanchored state-action AIRL diagnostic remains available through the
    legacy ``AIRLEstimator`` concrete class, but it is deliberately rejected by
    this public facade because it recovers a shaped advantage, not an identified
    structural reward.
    """

    def __init__(
        self,
        config: AIRLConfig | Any | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = AIRLConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config
        self.version = self._infer_version(config)
        self.delegate = self._build_delegate(config)

    @property
    def name(self) -> str:
        if self.version == "state_only":
            return "AIRL (Fu et al. 2018, state-only)"
        if self.version == "anchored":
            return "AIRL (Lee, Sudhir & Wang 2026, anchored)"
        return "AIRL (Lee, Sudhir & Wang 2026, heterogeneous)"

    def estimate(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs: Any,
    ) -> EstimationSummary:
        result = self.delegate.estimate(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            initial_params=initial_params,
            **kwargs,
        )
        result.method = self.name
        result.metadata = {
            **(result.metadata or {}),
            "airl_version": self.version,
            "airl_delegate": type(self.delegate).__name__,
        }
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    @staticmethod
    def _infer_version(config: Any) -> str:
        if isinstance(config, AIRLConfig):
            return config.version
        if getattr(config, "num_segments", 1) == 1:
            return "anchored"
        return "heterogeneous"

    def _build_delegate(self, config: Any) -> AIRLEstimator | Any:
        from econirl.estimation.adversarial.airl_het import (
            AIRLHetConfig,
            AIRLHetEstimator,
        )

        if isinstance(config, AIRLHetConfig):
            self._validate_anchored_config(
                version=self.version,
                exit_action=config.exit_action,
                absorbing_state=config.absorbing_state,
                num_segments=config.num_segments,
            )
            return AIRLHetEstimator(config)

        if not isinstance(config, AIRLConfig):
            raise TypeError(
                f"AIRL expects an AIRLConfig or AIRLHetConfig; got {type(config).__name__}"
            )

        if config.version == "state_only":
            if config.reward_arg != "state":
                raise ValueError(
                    "Unanchored state-action AIRL is not identified as a "
                    "structural reward. Use version='anchored' or "
                    "version='heterogeneous' with exit_action and "
                    "absorbing_state, or use AIRLEstimator directly for the "
                    "diagnostic shaped-advantage experiment."
                )
            if (
                config.exit_action is not None
                or config.anchor_action is not None
                or config.absorbing_state is not None
            ):
                raise ValueError(
                    "State-only AIRL does not use exit_action, anchor_action, "
                    "or absorbing_state. Use version='anchored' for the "
                    "identified action-dependent model."
                )
            return AIRLEstimator(replace(config, reward_arg="state"))

        if config.version in {"anchored", "heterogeneous"}:
            exit_action = (
                config.exit_action if config.exit_action is not None else config.anchor_action
            )
            num_segments = 1 if config.version == "anchored" else config.num_segments
            self._validate_anchored_config(
                version=config.version,
                exit_action=exit_action,
                absorbing_state=config.absorbing_state,
                num_segments=num_segments,
            )
            het_config = AIRLHetConfig(
                num_segments=num_segments,
                exit_action=int(exit_action),
                absorbing_state=int(config.absorbing_state),
                reward_type=config.reward_type,
                reward_lr=config.reward_lr,
                discriminator_steps=config.discriminator_steps,
                generator_solver=config.generator_solver,
                generator_tol=config.generator_tol,
                generator_max_iter=config.generator_max_iter,
                max_airl_rounds=config.max_rounds,
                airl_convergence_tol=config.convergence_tol,
                max_em_iterations=config.max_em_iterations,
                em_convergence_tol=config.em_convergence_tol,
                consistency_weight=config.consistency_weight,
                prior_smoothing=config.prior_smoothing,
                prior_min=config.prior_min,
                prior_damping=config.prior_damping,
                reward_weight_decay=config.reward_weight_decay,
                normalize_reward=config.normalize_reward,
                unit_normalize_reward=config.unit_normalize_reward,
                gradient_clip_norm=config.gradient_clip_norm,
                antisymmetric_init=config.antisymmetric_init,
                initialization=config.initialization,
                initialization_smoothing=config.initialization_smoothing,
                initialization_l2_penalty=config.initialization_l2_penalty,
                use_shaping=config.use_shaping,
                shaping_coef=config.shaping_coef,
                shaping_l2_penalty=config.shaping_l2_penalty,
                generator_reward=config.generator_reward,
                policy_step_size=config.policy_step_size,
                min_airl_rounds=config.min_rounds,
                verbose=config.verbose,
                seed=config.seed,
            )
            return AIRLHetEstimator(het_config)

        raise ValueError(f"unknown AIRL version={config.version!r}")

    @staticmethod
    def _validate_anchored_config(
        *,
        version: str,
        exit_action: int | object | None,
        absorbing_state: int | object | None,
        num_segments: int,
    ) -> None:
        if exit_action is None:
            raise ValueError(
                f"version={version!r} requires exit_action for anchored AIRL anchor identification"
            )
        if absorbing_state is None:
            raise ValueError(
                f"version={version!r} requires absorbing_state for anchored AIRL "
                "anchor identification"
            )
        if version == "anchored" and num_segments != 1:
            raise ValueError("version='anchored' is the one-segment anchored AIRL case")
        if version == "heterogeneous" and num_segments < 2:
            raise ValueError("version='heterogeneous' requires num_segments >= 2")
