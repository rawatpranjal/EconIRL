"""Receding Horizon Inverse Planning (RHIP).

Source: Barnes, Abueg, Wulfmeier et al., "Massively Scalable Inverse
Reinforcement Learning in Google Maps", ICLR 2024 (arXiv 2305.11290).

RHIP is not a new objective. It is a single knob -- the planning horizon ``H`` --
over the MaxEnt-family IRL machinery the package already has. Within ``H`` steps
of a state the agent plans with the expensive stochastic soft-Bellman policy;
beyond ``H`` it falls back to a cheap deterministic planner under the current
reward. The horizon recovers three classic methods as special cases:

    H = inf  ->  Max Causal Entropy IRL (fully stochastic, robust, expensive)
    H = 1    ->  Bayesian-IRL-like middle ground
    H = 0    ->  Max-Margin-Planning-like (fully deterministic, cheap, brittle)

The learning rule is the same for every ``H``: a MaxEnt feature-matching
gradient ascent

    grad = E_demo[phi] - E_pi[phi]

where ``E_pi[phi]`` is the expected (discounted) state-action feature occupancy
under the *receding-horizon policy* ``pi_H`` induced by the current reward. The
only thing the horizon changes is how ``pi_H`` is computed:

* ``H = inf``: the full soft value-iteration fixed point. This delegates to
  :class:`~econirl.estimation.mce_irl.MCEIRLEstimator`, so the endpoint is
  identical to MCE-IRL by construction.
* ``H`` finite: a deterministic (hard-max) value-iteration tail under the
  current reward gives the continuation value ``V_det``; ``H`` soft-Bellman
  backups are then run on top of ``V_det`` to produce time-indexed soft
  Q-values, and ``pi_H(a|s) = softmax(Q_H(s,a)/sigma)`` is the policy used to
  drive the occupancy and the choice-probability log-likelihood. ``H = 0`` runs
  no soft backups, so the policy is the softmax over the deterministic
  continuation value -- the Max-Margin-Planning end.

Linear reward only (this chunk). A neural-reward variant is deferred (v2),
following the ``NeuralUFXP`` exception pattern.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.base import BaseUtilityFunction
from econirl.preferences.reward import LinearReward


def _is_infinite_horizon(horizon) -> bool:
    """Return True when ``horizon`` denotes the H=inf (MCE-IRL) endpoint."""
    if horizon is None:
        return True
    if isinstance(horizon, float) and math.isinf(horizon):
        return True
    return False


@dataclass
class RHIPConfig:
    """Configuration for RHIP estimation.

    Attributes:
        horizon: Planning horizon ``H``. ``float('inf')`` or ``None`` selects
            the MCE-IRL endpoint. ``0`` selects the deterministic
            (Max-Margin-Planning-like) endpoint. Any positive integer is a
            receding-horizon interpolation.
        learning_rate: Adam step size for the feature-matching gradient ascent.
        outer_max_iter: Maximum outer (gradient) iterations.
        outer_tol: Gradient-norm convergence tolerance.
        gradient_clip: Max gradient norm before clipping.
        det_tail_iter: Iterations of the deterministic value-iteration tail
            used as the continuation value beyond the horizon (finite H only).
        det_tail_tol: Convergence tolerance for the deterministic tail.
        svf_tol / svf_max_iter: Forward state-visitation fixed-point controls.
        compute_se: Whether to compute standard errors (bootstrap).
        n_bootstrap: Bootstrap replications for SEs.
        verbose: Print progress.
    """

    horizon: float | int | None = float("inf")

    # Outer optimisation (feature matching, shared with MCE-IRL).
    learning_rate: float = 0.05
    outer_max_iter: int = 100
    outer_tol: float = 1e-6
    gradient_clip: float = 1.0
    use_adam: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # Deterministic continuation-value tail (finite horizon only).
    det_tail_iter: int = 2000
    det_tail_tol: float = 1e-8

    # Forward occupancy.
    svf_tol: float = 1e-8
    svf_max_iter: int = 1000
    occupancy_tol: float = 1e-3

    # Inference.
    compute_se: bool = False
    n_bootstrap: int = 100

    verbose: bool = False

    # ---- MCE-IRL endpoint passthrough ----
    # When H=inf, RHIP delegates to MCE-IRL; these map the shared knobs onto an
    # MCEIRLConfig so the endpoint matches the MCE-IRL config used elsewhere.
    inner_max_iter: int = 2000
    # An explicit MCEIRLConfig to delegate to at H=inf. When supplied it is used
    # verbatim, so RHIP(H=inf) reproduces *that exact* MCE-IRL configuration
    # bit-for-bit (the Oracle-A equivalence backbone). When None, the config
    # below is derived from the shared RHIP knobs.
    mce_config: MCEIRLConfig | None = None

    def to_mce_config(self) -> MCEIRLConfig:
        """Build the MCE-IRL config the H=inf endpoint delegates to."""
        if self.mce_config is not None:
            return self.mce_config
        return MCEIRLConfig(
            optimizer="gradient",
            learning_rate=self.learning_rate,
            outer_tol=self.outer_tol,
            outer_max_iter=self.outer_max_iter,
            gradient_clip=self.gradient_clip,
            use_adam=self.use_adam,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_eps=self.adam_eps,
            inner_max_iter=self.inner_max_iter,
            svf_tol=self.svf_tol,
            svf_max_iter=self.svf_max_iter,
            occupancy_tol=self.occupancy_tol,
            compute_se=self.compute_se,
            n_bootstrap=self.n_bootstrap,
            verbose=self.verbose,
        )


class RHIPEstimator(BaseEstimator):
    """Receding Horizon Inverse Planning estimator (Barnes et al. 2024).

    A horizon-parameterised MaxEnt-family IRL. See the module docstring for the
    method. The estimator satisfies the same low-level contract as the other
    package estimators (``estimate(panel, utility, problem, transitions)`` ->
    :class:`~econirl.inference.results.EstimationSummary` exposing
    ``.parameters``, ``.standard_errors``, ``.policy``, ``.value_function`` and
    ``.converged``).

    Parameters
    ----------
    config : RHIPConfig, optional
        Configuration object. If ``None`` a default is built from ``**kwargs``.
    horizon : float | int, optional
        Convenience shortcut for ``config.horizon``.
    **kwargs
        Override individual config fields.
    """

    def __init__(
        self,
        config: RHIPConfig | None = None,
        horizon: float | int | None = None,
        **kwargs,
    ):
        if config is None:
            config = RHIPConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        if horizon is not None:
            config.horizon = horizon

        super().__init__(
            se_method="bootstrap" if config.compute_se else "asymptotic",
            compute_hessian=config.compute_se,
            verbose=config.verbose,
        )
        self.config = config

        # Reuse MCE-IRL's occupancy/feature helpers verbatim so the finite-H
        # feature-matching gradient is the exact same computation as H=inf,
        # differing only in the policy that drives the occupancy.
        self._mce = MCEIRLEstimator(config=config.to_mce_config())

    @property
    def name(self) -> str:
        h = self.config.horizon
        label = "inf" if _is_infinite_horizon(h) else str(int(h))
        return f"RHIP (Barnes et al. 2024, H={label})"

    # ------------------------------------------------------------------
    # Receding-horizon policy
    # ------------------------------------------------------------------

    def _deterministic_tail_value(
        self,
        operator: SoftBellmanOperator,
        reward_matrix: jnp.ndarray,
        problem: DDCProblem,
    ) -> jnp.ndarray:
        """Deterministic (hard-max) continuation value under the current reward.

        Runs plain value iteration with the *hard* Bellman backup

            V(s) = max_a [ R(s,a) + gamma * E_{s'} V(s') ]

        i.e. the sigma -> 0 limit of the soft operator. This is the cheap
        planner used beyond the receding horizon.
        """
        beta = problem.discount_factor
        transitions = operator.transitions
        n_states = problem.num_states
        V = jnp.zeros(n_states, dtype=reward_matrix.dtype)
        for _ in range(self.config.det_tail_iter):
            EV = jnp.einsum("ast,t->as", transitions, V).T  # (S, A)
            Q = reward_matrix + beta * EV
            V_new = jnp.max(Q, axis=1)
            delta = float(jnp.abs(V_new - V).max())
            V = V_new
            if delta < self.config.det_tail_tol:
                break
        return V

    def _receding_horizon_policy(
        self,
        operator: SoftBellmanOperator,
        reward_matrix: jnp.ndarray,
        problem: DDCProblem,
        horizon: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Soft policy after ``horizon`` soft backups over a deterministic tail.

        The continuation value beyond the horizon is the deterministic
        (hard-max) optimum ``V_det``. We then run ``horizon`` soft-Bellman
        backups on top of it:

            V_0 = V_det
            Q_k(s,a) = R(s,a) + gamma * E_{s'} V_{k-1}(s')
            V_k(s)   = sigma * logsumexp_a Q_k(s,a) / sigma

        and return ``(V_H, policy_H)`` where ``policy_H = softmax(Q_H/sigma)``.
        ``horizon = 0`` runs no soft backups, so the policy is the softmax over
        the deterministic continuation value (the Max-Margin-Planning end);
        as ``horizon -> inf`` the soft backups drive V to the soft fixed point
        (the MCE-IRL end).
        """
        beta = problem.discount_factor
        sigma = problem.scale_parameter
        transitions = operator.transitions

        V = self._deterministic_tail_value(operator, reward_matrix, problem)

        # H=0: one soft choice over the deterministic continuation value.
        # H>=1: H soft backups. Either way the final policy is a softmax over
        # the soft Q built on the (H-step-refined) continuation value.
        EV = jnp.einsum("ast,t->as", transitions, V).T  # (S, A)
        Q = reward_matrix + beta * EV
        policy = jax.nn.softmax(Q / sigma, axis=1)
        V_soft = sigma * jax.scipy.special.logsumexp(Q / sigma, axis=1)

        for _ in range(int(horizon)):
            EV = jnp.einsum("ast,t->as", transitions, V_soft).T
            Q = reward_matrix + beta * EV
            policy = jax.nn.softmax(Q / sigma, axis=1)
            V_soft = sigma * jax.scipy.special.logsumexp(Q / sigma, axis=1)

        return V_soft, policy

    def _log_choice_probabilities(
        self,
        reward_matrix: jnp.ndarray,
        V_soft: jnp.ndarray,
        problem: DDCProblem,
        transitions: jnp.ndarray,
    ) -> jnp.ndarray:
        """log pi(a|s) consistent with the receding-horizon Q at convergence."""
        beta = problem.discount_factor
        sigma = problem.scale_parameter
        EV = jnp.einsum("ast,t->as", transitions, V_soft).T
        Q = reward_matrix + beta * EV
        return jax.nn.log_softmax(Q / sigma, axis=1)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def _optimize(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        true_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        # ---- H = inf: delegate to MCE-IRL so the endpoint is identical ----
        if _is_infinite_horizon(self.config.horizon):
            result = self._mce._optimize(
                panel=panel,
                utility=utility,
                problem=problem,
                transitions=transitions,
                initial_params=initial_params,
                true_params=true_params,
                **kwargs,
            )
            md = dict(result.metadata or {})
            md["horizon"] = "inf"
            md["estimator"] = "RHIP (H=inf, delegated to MCE-IRL)"
            return EstimationResult(
                parameters=result.parameters,
                log_likelihood=result.log_likelihood,
                value_function=result.value_function,
                policy=result.policy,
                hessian=result.hessian,
                gradient_contributions=result.gradient_contributions,
                converged=result.converged,
                num_iterations=result.num_iterations,
                num_function_evals=result.num_function_evals,
                message=result.message,
                optimization_time=result.optimization_time,
                metadata=md,
            )

        # ---- Finite horizon: receding-horizon feature matching ----
        horizon = int(self.config.horizon)
        start_time = time.time()

        reward_fn = utility
        transitions_f64 = transitions.astype(jnp.float64)
        operator = SoftBellmanOperator(problem, transitions_f64)

        if initial_params is None:
            params = reward_fn.get_initial_parameters()
        else:
            params = jnp.array(initial_params)
        params = jnp.asarray(params, dtype=jnp.float64)

        # Empirical (discounted) feature occupancy -- identical to MCE-IRL.
        empirical_features = self._mce._compute_empirical_features(
            panel, reward_fn, problem.num_states, problem.num_actions,
            discount=problem.discount_factor,
        ).astype(jnp.float64)
        initial_dist = self._mce._compute_initial_distribution(
            panel, problem.num_states
        ).astype(jnp.float64)
        D_demo, D_sa_demo = self._mce._compute_empirical_state_occupancy(
            panel, problem.num_states, problem.num_actions,
            discount=problem.discount_factor,
        )
        D_demo = D_demo.astype(jnp.float64)
        D_sa_demo = D_sa_demo.astype(jnp.float64)

        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()

        # Adam state.
        m = jnp.zeros_like(params)
        v = jnp.zeros_like(params)

        best_obj = float("inf")
        best_params = jnp.array(params)
        patience_counter = 0
        max_patience = 20
        converged = False
        grad_norm = float("inf")

        pbar = tqdm(
            range(self.config.outer_max_iter),
            desc=f"RHIP (H={horizon})",
            disable=not self.config.verbose,
            leave=True,
        )

        for i in pbar:
            reward_matrix = reward_fn.compute(
                params.astype(jnp.float32)
            ).astype(jnp.float64)

            V_soft, policy = self._receding_horizon_policy(
                operator, reward_matrix, problem, horizon
            )

            # Expected (discounted) feature occupancy under the RH policy --
            # the same MCE-IRL occupancy computation, just a different policy.
            expected_features = self._mce._compute_expected_features(
                panel, policy, reward_fn,
                transitions=transitions_f64, initial_dist=initial_dist,
                discount=problem.discount_factor,
            )

            gradient = empirical_features - expected_features  # ascent direction
            grad_norm = float(jnp.linalg.norm(gradient))
            if self.config.gradient_clip > 0 and grad_norm > self.config.gradient_clip:
                gradient = gradient * (self.config.gradient_clip / grad_norm)
                grad_norm = self.config.gradient_clip

            obj = float(0.5 * jnp.sum((empirical_features - expected_features) ** 2))

            D_policy = self._mce._compute_state_visitation(
                policy, transitions_f64, problem, initial_dist
            )
            occ_dist = float(jnp.max(jnp.abs(D_demo - D_policy)))

            if obj < best_obj:
                best_obj = obj
                best_params = jnp.array(params)
                patience_counter = 0
            else:
                patience_counter += 1

            postfix = {"obj": f"{obj:.6f}", "||g||": f"{grad_norm:.4f}",
                       "occ": f"{occ_dist:.4f}"}
            if true_params is not None:
                rmse = float(jnp.sqrt(jnp.mean((params - true_params) ** 2)))
                postfix["RMSE"] = f"{rmse:.6f}"
            pbar.set_postfix(postfix)

            if grad_norm < self.config.outer_tol:
                converged = True
                break
            if occ_dist < self.config.occupancy_tol:
                converged = True
                break
            if patience_counter > max_patience:
                break

            # Adam step.
            if self.config.use_adam:
                t = i + 1
                m = self.config.adam_beta1 * m + (1 - self.config.adam_beta1) * gradient
                v = self.config.adam_beta2 * v + (1 - self.config.adam_beta2) * (gradient ** 2)
                m_hat = m / (1 - self.config.adam_beta1 ** t)
                v_hat = v / (1 - self.config.adam_beta2 ** t)
                params = params + self.config.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.config.adam_eps)
            else:
                params = params + self.config.learning_rate * gradient

        pbar.close()
        final_params = best_params

        # Final solution at the best parameters.
        reward_matrix = reward_fn.compute(
            final_params.astype(jnp.float32)
        ).astype(jnp.float64)
        V_soft, policy = self._receding_horizon_policy(
            operator, reward_matrix, problem, horizon
        )
        log_probs = self._log_choice_probabilities(
            reward_matrix, V_soft, problem, transitions_f64
        )
        ll = float(log_probs[all_states, all_actions].sum())

        D = self._mce._compute_state_visitation(
            policy, transitions_f64, problem, initial_dist
        )
        D_sa = D[:, None] * policy
        occupancy_moment_residual = float(jnp.max(jnp.abs(D_sa_demo - D_sa)))
        final_expected = self._mce._compute_expected_features(
            panel, policy, reward_fn,
            transitions=transitions_f64, initial_dist=initial_dist,
            discount=problem.discount_factor,
        )
        feature_diff = float(jnp.linalg.norm(empirical_features - final_expected))

        standard_errors = None
        if self.config.compute_se:
            standard_errors = self._bootstrap_inference(
                panel, reward_fn, problem, transitions_f64, final_params, initial_dist,
                horizon,
            )

        optimization_time = time.time() - start_time

        return EstimationResult(
            parameters=final_params.astype(jnp.float32),
            log_likelihood=ll,
            value_function=V_soft,
            policy=policy,
            hessian=None,
            converged=converged,
            num_iterations=i + 1,
            num_function_evals=i + 1,
            message="Converged" if converged else "Max iterations / early stop",
            optimization_time=optimization_time,
            metadata={
                "horizon": horizon,
                "estimator": f"RHIP (H={horizon})",
                "empirical_features": np.asarray(empirical_features).tolist(),
                "final_expected_features": np.asarray(final_expected).tolist(),
                "feature_difference": feature_diff,
                "feature_diff": feature_diff,
                "occupancy_moment_residual": occupancy_moment_residual,
                "state_visitation": np.asarray(D).tolist(),
                "state_action_visitation": np.asarray(D_sa).tolist(),
                "standard_errors": np.asarray(standard_errors).tolist()
                if standard_errors is not None else None,
            },
        )

    def _bootstrap_inference(
        self,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        point_estimate: jnp.ndarray,
        initial_dist: jnp.ndarray,
        horizon: int,
    ) -> jnp.ndarray:
        """Bootstrap SEs for the finite-horizon path."""
        n_params = len(point_estimate)
        operator = SoftBellmanOperator(problem, transitions)
        bootstrap_estimates = np.zeros((self.config.n_bootstrap, n_params))
        trajectories = panel.trajectories
        n_traj = len(trajectories)

        for b in range(self.config.n_bootstrap):
            indices = np.random.choice(n_traj, size=n_traj, replace=True)
            boot_panel = Panel(trajectories=[trajectories[j] for j in indices])
            empirical_features = self._mce._compute_empirical_features(
                boot_panel, reward_fn, problem.num_states, problem.num_actions,
                discount=problem.discount_factor,
            ).astype(jnp.float64)
            boot_initial = self._mce._compute_initial_distribution(
                boot_panel, problem.num_states
            ).astype(jnp.float64)
            params = jnp.asarray(point_estimate, dtype=jnp.float64)
            for _ in range(50):
                reward_matrix = reward_fn.compute(
                    params.astype(jnp.float32)
                ).astype(jnp.float64)
                _, policy = self._receding_horizon_policy(
                    operator, reward_matrix, problem, horizon
                )
                expected = self._mce._compute_expected_features(
                    boot_panel, policy, reward_fn,
                    transitions=transitions, initial_dist=boot_initial,
                    discount=problem.discount_factor,
                )
                gradient = empirical_features - expected
                params = params + 0.1 * gradient
                if float(jnp.linalg.norm(gradient)) < 0.01:
                    break
            bootstrap_estimates[b] = np.asarray(params)

        return jnp.array(bootstrap_estimates.std(axis=0))


# ---------------------------------------------------------------------------
# Sklearn-style protocol wrapper
# ---------------------------------------------------------------------------


class RHIP:
    """Sklearn-style Receding Horizon Inverse Planning estimator.

    Satisfies :class:`~econirl.estimators.protocol.EstimatorProtocol`. The
    horizon ``H`` is the single knob over the MaxEnt-family IRL: ``inf`` is
    MCE-IRL, ``0`` is Max-Margin-Planning-like, intermediate values
    interpolate.

    This wrapper accepts a pre-built :class:`Panel` plus ``features`` and
    ``transitions`` (the form the studies use) and exposes the fitted policy,
    value function, reward matrix and recovered parameters.

    Parameters
    ----------
    horizon : float | int, default=inf
        Planning horizon ``H``.
    n_actions : int, optional
        Number of actions (inferred from features when omitted).
    discount : float, default=0.95
        Discount factor ``beta``.
    scale : float, default=1.0
        Logit scale ``sigma``.
    feature_names : list[str], optional
        Names for the reward features.
    learning_rate, outer_max_iter, verbose, ...
        Forwarded to :class:`RHIPConfig`.
    """

    def __init__(
        self,
        horizon: float | int | None = float("inf"),
        n_actions: int | None = None,
        discount: float = 0.95,
        scale: float = 1.0,
        feature_names: list[str] | None = None,
        learning_rate: float = 0.05,
        outer_max_iter: int = 100,
        compute_se: bool = False,
        verbose: bool = False,
        **config_kwargs,
    ):
        self.horizon = horizon
        self.n_actions = n_actions
        self.discount = discount
        self.scale = scale
        self.feature_names = feature_names
        self._config_kwargs = dict(
            horizon=horizon,
            learning_rate=learning_rate,
            outer_max_iter=outer_max_iter,
            compute_se=compute_se,
            verbose=verbose,
            **config_kwargs,
        )

        # Protocol attributes (set after fit).
        self.params_: dict[str, float] | None = None
        self.se_: dict[str, float] | None = None
        self.pvalues_: dict[str, float] | None = None
        self.policy_: np.ndarray | None = None
        self.value_: np.ndarray | None = None
        self.reward_matrix_: np.ndarray | None = None
        self.converged_: bool | None = None

        self._result = None
        self._reward_fn = None

    def fit(
        self,
        data: Panel,
        features: np.ndarray | None = None,
        transitions: np.ndarray | None = None,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        **kwargs,
    ) -> "RHIP":
        """Fit RHIP to a Panel with action-dependent ``features``."""
        if not isinstance(data, Panel):
            raise TypeError(
                "RHIP.fit expects a Panel; pass features=(S,A,K) and "
                "transitions=(A,S,S)."
            )
        if features is None or transitions is None:
            raise ValueError("RHIP.fit requires features=(S,A,K) and transitions=(A,S,S).")

        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 3:
            raise ValueError(
                f"features must be 3D (states, actions, K); got shape {features.shape}"
            )
        n_states, n_actions, n_features = features.shape
        transitions = np.asarray(transitions, dtype=np.float64)

        names = self.feature_names or [f"theta_{k}" for k in range(n_features)]
        reward_fn = ActionDependentReward(jnp.array(features), names)
        self._reward_fn = reward_fn

        problem = DDCProblem(
            num_states=n_states,
            num_actions=n_actions,
            discount_factor=self.discount,
            scale_parameter=self.scale,
        )

        est = RHIPEstimator(**self._config_kwargs)
        summary = est.estimate(data, reward_fn, problem, jnp.array(transitions))
        self._result = summary

        params = np.asarray(summary.parameters)
        param_names = list(summary.parameter_names)
        self.params_ = {n: float(v) for n, v in zip(param_names, params)}
        se = np.asarray(summary.standard_errors)
        self.se_ = {n: float(v) for n, v in zip(param_names, se)}
        self.policy_ = np.asarray(summary.policy) if summary.policy is not None else None
        self.value_ = (
            np.asarray(summary.value_function)
            if summary.value_function is not None else None
        )
        reward_matrix = reward_fn.compute(jnp.array(params, dtype=jnp.float32))
        self.reward_matrix_ = np.asarray(reward_matrix)
        self.converged_ = bool(summary.converged)
        return self

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        if self.policy_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.policy_[np.asarray(states, dtype=np.int64)]

    def conf_int(self, alpha: float = 0.05) -> dict:
        if self.params_ is None or self.se_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        from scipy.stats import norm as _norm
        z = float(_norm.ppf(1 - alpha / 2))
        return {
            name: (self.params_[name] - z * self.se_[name],
                   self.params_[name] + z * self.se_[name])
            for name in self.params_
        }

    def summary(self) -> str:
        if self._result is None:
            return "RHIP: not fitted. Call fit() first."
        h = self.horizon
        label = "inf" if _is_infinite_horizon(h) else str(int(h))
        lines = [f"RHIP (Barnes et al. 2024), horizon H={label}",
                 f"converged: {self.converged_}"]
        for name, val in (self.params_ or {}).items():
            se = (self.se_ or {}).get(name, float("nan"))
            lines.append(f"  {name:<16} {val:>10.4f}  (se {se:.4f})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        label = "inf" if _is_infinite_horizon(self.horizon) else str(int(self.horizon))
        return f"RHIP(horizon={label}, fitted={self.params_ is not None})"
