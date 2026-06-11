"""f-IRL: Inverse Reinforcement Learning via State Marginal Matching.

Recovers reward functions by matching the state-marginal distribution of
the policy to the expert's empirical state-marginal, using f-divergence
minimization instead of feature expectation matching.

Algorithm:
    1. Compute expert state or state-action marginal from demonstrations
    2. Initialize tabular reward R(s) or R(s, a)
    3. For each iteration:
       a. Solve MDP under current R to get policy pi
       b. Compute policy marginal via forward propagation
       c. Compute f-divergence gradient between expert and policy marginals
       d. Update R in the divergence-gradient direction
    4. Return R and induced policy

Supported f-divergences:
    - fkl (or "kl"): forward KL D_KL(p_expert || p_policy),
        gradient = log(p_expert / p_policy)
    - rkl: reverse KL D_KL(p_policy || p_expert) (mode-seeking),
        gradient = log(p_policy / p_expert)
    - js: Jensen-Shannon divergence, symmetric mixture-based form,
        gradient = log(p_expert / mean) - log(p_policy / mean) where
        mean = (p_expert + p_policy) / 2
    - chi2: chi-squared (econirl extension), gradient = (p_expert / p_policy) - 1
    - tv: total variation (econirl extension), gradient = sign(p_expert - p_policy)

Reference:
    Ni, T., Sikchi, H., Wang, Y., Gupta, T., Lee, L., & Eysenbach, B. (2022).
    "f-IRL: Inverse Reinforcement Learning via State Marginal Matching."
    CoRL.
"""

from __future__ import annotations

import time
from typing import Literal

import jax.numpy as jnp

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.preferences.base import UtilityFunction


class FIRLEstimator(BaseEstimator):
    """f-IRL estimator via state-marginal matching.

    Recovers a tabular reward function R(s,a) by minimizing the
    f-divergence between the expert's state-action marginal and the
    policy's state-action marginal. This avoids the feature-matching
    assumption of MaxEnt IRL.

    Attributes:
        f_divergence: Which f-divergence to use ("kl", "chi2", "tv").
        lr: Learning rate for reward updates.
        max_iter: Maximum number of gradient iterations.
        inner_tol: Convergence tolerance for MDP solver.
        inner_max_iter: Maximum iterations for MDP solver.
        horizon: Horizon for state visitation computation.

    Example:
        >>> estimator = FIRLEstimator(f_divergence="kl", lr=0.5)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
    """

    def __init__(
        self,
        f_divergence: Literal["kl", "fkl", "rkl", "js", "chi2", "tv"] = "fkl",
        lr: float = 0.5,
        max_iter: int = 500,
        inner_tol: float = 1e-8,
        inner_max_iter: int = 5000,
        horizon: int = 100,
        reward_clip: float = 10.0,
        marginal_space: Literal["state_action", "state"] = "state_action",
        reward_scope: Literal["state_action", "state"] = "state_action",
        selection_metric: Literal["log_likelihood", "occupancy_l1"] = "log_likelihood",
        compute_se: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            se_method="asymptotic",
            compute_hessian=False,
            verbose=verbose,
        )
        # "kl" is a back-compat alias for "fkl" (forward KL) per Ni et al. 2022.
        self._f_divergence = "fkl" if f_divergence == "kl" else f_divergence
        self._lr = lr
        self._max_iter = max_iter
        self._inner_tol = inner_tol
        self._inner_max_iter = inner_max_iter
        self._horizon = horizon
        self._reward_clip = reward_clip
        if marginal_space not in {"state_action", "state"}:
            raise ValueError("marginal_space must be 'state_action' or 'state'")
        if reward_scope not in {"state_action", "state"}:
            raise ValueError("reward_scope must be 'state_action' or 'state'")
        if selection_metric not in {"log_likelihood", "occupancy_l1"}:
            raise ValueError("selection_metric must be 'log_likelihood' or 'occupancy_l1'")
        if reward_scope == "state" and marginal_space != "state":
            raise ValueError("state reward_scope requires state marginal_space")
        self._marginal_space = marginal_space
        self._reward_scope = reward_scope
        self._selection_metric = selection_metric

    @property
    def name(self) -> str:
        return f"f-IRL ({self._f_divergence}, Ni et al. 2022)"

    def _compute_expert_marginal(
        self,
        panel: Panel,
        n_states: int,
        n_actions: int,
    ) -> jnp.ndarray:
        """Compute empirical marginal from demonstrations.

        Returns:
            State marginal, shape (n_states,), or state-action marginal,
            shape (n_states, n_actions). The returned marginal sums to 1.
        """
        all_states = panel.get_all_states()
        if self._marginal_space == "state":
            counts = jnp.zeros(n_states)
            counts = counts.at[all_states.astype(jnp.int32)].add(1.0)
        else:
            all_actions = panel.get_all_actions()
            idx = (all_states * n_actions + all_actions).astype(jnp.int32)
            counts = jnp.zeros(n_states * n_actions)
            counts = counts.at[idx].add(1.0)
            counts = counts.reshape(n_states, n_actions)
        total = counts.sum()
        return counts / jnp.maximum(total, 1.0)

    def _compute_state_visitation(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        problem: DDCProblem,
        panel: Panel,
    ) -> jnp.ndarray:
        """Compute discounted state visitation under a policy."""
        n_states = problem.num_states
        beta = problem.discount_factor

        # Initial state distribution from data
        init_counts = jnp.zeros(n_states)
        init_states = jnp.array(
            [traj.states[0].item() for traj in panel.trajectories if len(traj) > 0],
            dtype=jnp.int32,
        )
        init_counts = init_counts.at[init_states].add(1.0)
        mu = init_counts / jnp.maximum(init_counts.sum(), 1.0)

        state_vis = mu
        P_pi = jnp.einsum("sa,ast->st", policy, transitions)

        for t in range(1, self._horizon):
            mu = mu @ P_pi
            state_vis += (beta ** t) * mu

        return state_vis / state_vis.sum()

    def _compute_policy_marginal(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        problem: DDCProblem,
        panel: Panel,
    ) -> jnp.ndarray:
        """Compute model marginal under policy via forward propagation."""
        n_states = problem.num_states
        state_vis = self._compute_state_visitation(policy, transitions, problem, panel)
        if self._marginal_space == "state":
            return state_vis
        return state_vis[:, None] * policy

    def _f_divergence_gradient(
        self,
        p_expert: jnp.ndarray,
        p_policy: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute gradient of f-divergence w.r.t. reward.

        The reward update direction is proportional to the f-divergence
        gradient evaluated at the density ratio.

        Args:
            p_expert: Expert marginal, shape (n_states, n_actions).
            p_policy: Policy marginal, shape (n_states, n_actions).

        Returns:
            Gradient direction, shape (n_states, n_actions).
        """
        eps = 1e-10
        p_policy_safe = jnp.clip(p_policy, min=eps)
        p_expert_safe = jnp.clip(p_expert, min=eps)

        if self._f_divergence == "fkl":
            # Forward KL D_KL(p_E || p_pi). Mass-covering: upweights states
            # where expert has support but policy does not.
            return jnp.log(p_expert_safe / p_policy_safe)
        elif self._f_divergence == "rkl":
            # Reverse KL D_KL(p_pi || p_E). Mode-seeking: penalizes states
            # where the policy puts mass that the expert does not.
            return jnp.log(p_policy_safe / p_expert_safe)
        elif self._f_divergence == "js":
            # Jensen-Shannon symmetric divergence with mixture
            # M = (p_E + p_pi) / 2.
            mean = 0.5 * (p_expert_safe + p_policy_safe)
            return jnp.log(p_expert_safe / mean) - jnp.log(p_policy_safe / mean)
        elif self._f_divergence == "chi2":
            return (p_expert_safe / p_policy_safe) - 1.0
        elif self._f_divergence == "tv":
            return jnp.sign(p_expert - p_policy)
        else:
            raise ValueError(f"Unknown f-divergence: {self._f_divergence}")

    def _reward_matrix_from_params(
        self,
        reward_params: jnp.ndarray,
        n_actions: int,
    ) -> jnp.ndarray:
        if self._reward_scope == "state":
            return jnp.repeat(reward_params[:, None], n_actions, axis=1)
        return reward_params

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run f-IRL optimization."""
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        operator = SoftBellmanOperator(problem, transitions)

        # Expert marginal
        expert_marginal = self._compute_expert_marginal(panel, n_states, n_actions)

        # Initialize tabular reward
        if self._reward_scope == "state":
            reward_params = jnp.zeros(n_states)
        else:
            reward_params = jnp.zeros((n_states, n_actions))

        best_score = float("-inf")
        best_ll = float("-inf")
        best_policy = None
        best_V = None
        best_reward_params = None
        best_reward_matrix = None
        best_policy_marginal = None
        best_occupancy_l1 = float("inf")

        self._log(f"f-IRL ({self._f_divergence}): {self._max_iter} iterations")

        from tqdm import tqdm
        pbar = tqdm(
            range(self._max_iter),
            desc=f"f-IRL ({self._f_divergence})",
            disable=not self._verbose,
            leave=True,
        )
        for it in pbar:
            # Solve MDP under current reward
            reward_matrix = self._reward_matrix_from_params(reward_params, n_actions)
            solver_result = value_iteration(
                operator, reward_matrix,
                tol=self._inner_tol,
                max_iter=self._inner_max_iter,
            )
            policy = solver_result.policy

            # Compute policy marginal
            policy_marginal = self._compute_policy_marginal(
                policy, transitions, problem, panel,
            )

            # Compute divergence gradient.
            grad = self._f_divergence_gradient(expert_marginal, policy_marginal)

            # Track the best policy/reward pair before taking the next update.
            log_probs = operator.compute_log_choice_probabilities(
                reward_matrix, solver_result.V,
            )
            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            ll = log_probs[all_states, all_actions].sum().item()
            occupancy_l1 = float(jnp.abs(expert_marginal - policy_marginal).sum())
            score = ll if self._selection_metric == "log_likelihood" else -occupancy_l1

            if score > best_score:
                best_score = score
                best_ll = ll
                best_policy = jnp.array(policy)
                best_V = jnp.array(solver_result.V)
                best_reward_params = jnp.array(reward_params)
                best_reward_matrix = jnp.array(
                    self._reward_matrix_from_params(reward_params, n_actions)
                )
                best_policy_marginal = jnp.array(policy_marginal)
                best_occupancy_l1 = occupancy_l1

            reward_params = reward_params + self._lr * grad
            reward_params = jnp.clip(reward_params, -self._reward_clip, self._reward_clip)

            div = occupancy_l1
            current_reward = self._reward_matrix_from_params(reward_params, n_actions)
            r_range = float(jnp.max(current_reward) - jnp.min(current_reward))
            pbar.set_postfix({
                "LL": f"{ll:.2f}",
                "best": f"{best_ll:.2f}",
                "div": f"{div:.4f}",
                "R_rng": f"{r_range:.2f}",
            })

        elapsed = time.time() - start_time

        return EstimationResult(
            parameters=best_reward_params.flatten(),
            log_likelihood=best_ll,
            value_function=best_V,
            policy=best_policy,
            hessian=None,
            converged=True,
            num_iterations=self._max_iter,
            message=f"f-IRL ({self._f_divergence}): {self._max_iter} iterations",
            optimization_time=elapsed,
            metadata={
                "reward_matrix": best_reward_matrix,
                "state_reward_vector": (
                    best_reward_params if self._reward_scope == "state" else None
                ),
                "expert_marginal": expert_marginal,
                "policy_marginal": best_policy_marginal,
                "occupancy_l1": best_occupancy_l1,
                "reward_range": float(
                    jnp.max(best_reward_matrix) - jnp.min(best_reward_matrix)
                ),
                "f_divergence": self._f_divergence,
                "marginal_space": self._marginal_space,
                "reward_scope": self._reward_scope,
                "selection_metric": self._selection_metric,
                "counterfactual_reward_normalization": (
                    "affine" if self._reward_scope == "state" else None
                ),
            },
        )

    def estimate(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate via f-IRL.

        Overrides base to handle tabular reward output.
        """
        result = self._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            initial_params=initial_params,
            **kwargs,
        )

        n_obs = panel.num_observations
        if self._reward_scope == "state":
            n_params = problem.num_states
        else:
            n_params = problem.num_states * problem.num_actions

        goodness_of_fit = GoodnessOfFit(
            log_likelihood=result.log_likelihood,
            num_parameters=n_params,
            num_observations=n_obs,
            aic=-2 * result.log_likelihood + 2 * n_params,
            bic=-2 * result.log_likelihood
            + n_params * jnp.log(jnp.array(n_obs)).item(),
            prediction_accuracy=self._compute_prediction_accuracy(
                panel, result.policy
            ),
        )

        if self._reward_scope == "state":
            param_names = [f"R(s={s})" for s in range(problem.num_states)]
        else:
            param_names = [
                f"R(s={s},a={a})"
                for s in range(problem.num_states)
                for a in range(problem.num_actions)
            ]

        return EstimationSummary(
            parameters=result.parameters,
            parameter_names=param_names,
            standard_errors=jnp.full_like(result.parameters, float("nan")),
            hessian=None,
            variance_covariance=None,
            method=self.name,
            num_observations=n_obs,
            num_individuals=panel.num_individuals,
            num_periods=max(panel.num_periods_per_individual),
            discount_factor=problem.discount_factor,
            scale_parameter=problem.scale_parameter,
            log_likelihood=result.log_likelihood,
            goodness_of_fit=goodness_of_fit,
            identification=None,
            converged=True,
            num_iterations=result.num_iterations,
            convergence_message=result.message,
            value_function=result.value_function,
            policy=result.policy,
            estimation_time=result.optimization_time,
            metadata=result.metadata,
        )
