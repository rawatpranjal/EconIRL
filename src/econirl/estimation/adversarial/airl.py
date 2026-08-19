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
from dataclasses import dataclass
from typing import Literal

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
        generator_reward: Reward used by the policy update. "recovered" solves
            the current recovered reward g. "log_odds" uses AIRL's discriminator
            log-odds reward f - log pi. "f" uses the shaped discriminator score.
        min_rounds: Minimum adversarial rounds before policy-change convergence
            can stop training.
        convergence_tol: Tolerance for policy convergence
        compute_se: Whether to compute standard errors
        se_method: Method for standard errors
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for policy-transition sampling
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
    generator_reward: Literal["recovered", "log_odds", "f"] = "recovered"
    min_rounds: int = 20
    convergence_tol: float = 1e-4
    compute_se: bool = True
    se_method: Literal["bootstrap", "asymptotic"] = "bootstrap"
    n_bootstrap: int = 100
    seed: int = 42
    verbose: bool = False


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

        This mirrors the AIRL2 normalization: one action is pinned to zero reward,
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
            reward_params = jnp.zeros((n_states, n_actions))
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

        # Initialize policy
        policy = jnp.ones((n_states, n_actions)) / n_actions

        # AIRL discriminator loss (differentiable w.r.t. reward and h params)
        use_shaping = self.config.use_shaping
        shaping_coef = self.config.shaping_coef
        shaping_l2_penalty = self.config.shaping_l2_penalty

        reward_arg_state = self.config.reward_arg == "state"

        def disc_loss_fn(
            params,
            policy_fixed,
            exp_s,
            exp_a,
            exp_ns,
            pol_s,
            pol_a,
            pol_ns,
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

            def logits(states, actions, next_states):
                r_sa = reward_matrix[states, actions]
                if use_shaping:
                    sc = shaping_coef if shaping_coef is not None else gamma
                    f = r_sa + sc * shaping_potential[next_states] - shaping_potential[states]
                else:
                    f = r_sa
                log_pi = jnp.log(policy_fixed[states, actions] + 1e-10)
                return f - log_pi

            e_logits = logits(exp_s, exp_a, exp_ns)
            p_logits = logits(pol_s, pol_a, pol_ns)

            e_loss = jnp.mean(jnp.logaddexp(0.0, -e_logits))
            p_loss = jnp.mean(jnp.logaddexp(0.0, p_logits))
            l2_penalty = shaping_l2_penalty * (
                jnp.mean(reward_matrix**2) + jnp.mean(shaping_potential**2)
            )
            return e_loss + p_loss + l2_penalty

        disc_loss_and_grad = jax.value_and_grad(disc_loss_fn)

        # Training metrics
        disc_losses = []
        policy_changes = []
        converged = False
        round_idx = 0
        key = jax.random.key(self.config.seed)

        pbar = tqdm(
            range(self.config.max_rounds),
            desc="AIRL",
            disable=not self.config.verbose,
        )

        for round_idx in pbar:
            old_policy = jnp.array(policy)

            # Sample from current policy using lax.scan
            key, subkey = jax.random.split(key)
            policy_states, policy_actions, policy_next_states = (
                self._sample_transitions_from_policy(
                    policy, transitions, n_expert, initial_dist, subkey
                )
            )

            # Update reward and shaping potential via optax Adam.
            disc_loss = 0.0
            for _ in range(self.config.discriminator_steps):
                loss, grads = disc_loss_and_grad(
                    disc_params,
                    policy,
                    expert_states,
                    expert_actions,
                    expert_next_states,
                    policy_states,
                    policy_actions,
                    policy_next_states,
                )
                updates, opt_state = optimizer.update(grads, opt_state, params=disc_params)
                disc_params = optax.apply_updates(disc_params, updates)
                disc_loss = float(loss)

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
                "generator_reward": self.config.generator_reward,
                "min_rounds": self.config.min_rounds,
                "final_disc_loss": disc_losses[-1] if disc_losses else None,
                "disc_losses": disc_losses,
                "policy_changes": policy_changes,
                "shaping_potential": jnp.array(disc_params["shaping"]).tolist(),
                "reward_matrix": jnp.array(final_reward).tolist(),
            },
        )
