"""CCP-based estimators: Hotz-Miller and NPL.

This module implements Conditional Choice Probability (CCP) estimators
for dynamic discrete choice models:

1. Hotz-Miller (1993): Two-step estimator using CCPs from data
2. NPL (Aguirregabiria-Mira 2002): Iterated Hotz-Miller policy updates

Key insight: The value function can be recovered from CCPs without solving
the full Bellman equation, via the Hotz-Miller inversion theorem.

For logit errors:
    e(a,x) = γ - log(P(a|x))  where γ ≈ 0.5772 is Euler's constant

References:
    Hotz, V.J. and Miller, R.A. (1993). "Conditional Choice Probabilities
        and the Estimation of Dynamic Models." RES 60(3), 497-529.
    Aguirregabiria, V. and Mira, P. (2002). "Swapping the Nested Fixed Point
        Algorithm." Econometrica 70(4), 1519-1543.
    Aguirregabiria, V. and Mira, P. (2010). "Dynamic discrete choice structural
        models: A survey." Journal of Econometrics 156(1), 38-67.
"""

from __future__ import annotations

import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.optimizer import minimize_lbfgsb
from econirl.core.solvers import policy_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.full_likelihood import (
    compute_rust_full_likelihood_bhhh_score,
)
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.base import UtilityFunction

# Euler-Mascheroni constant
EULER_GAMMA = 0.5772156649015329


# ── Module-level JIT for the CCP logit step ────────────────────────────────
#
# By making z_tilde / e_tilde / all_states / all_actions EXPLICIT ARGUMENTS
# rather than closed-over variables, JAX can reuse this compiled XLA kernel
# across all NPL iterations (the kernel is keyed on shape+dtype, not on
# the concrete values of the captured arrays).  If z_tilde were in the
# closure instead, every NPL step would create a new Python function with a
# new constant baked into the XLA program, forcing a full recompilation.


def _ccp_logit_neg_ll(params, z_tilde, e_tilde, all_states, all_actions, inv_sigma):
    """CCP logit pseudo-LL with augmented features (value only).

    v(s,a; theta) = z_tilde[s,a,:]@theta + e_tilde[s,a]
    LL = sum_{i} log softmax(v / sigma)[s_i, a_i]
    """
    theta = params.astype(z_tilde.dtype)
    v = jnp.einsum("sak,k->sa", z_tilde, theta) + e_tilde
    log_probs = jax.nn.log_softmax(v * inv_sigma, axis=1)
    return -(log_probs[all_states, all_actions].sum())


# JIT the value+grad together.  jax.jit is lazy — compilation fires on the
# first call, keyed on (function_id, abstract_args=(shape, dtype)).  Since
# _ccp_logit_neg_ll is defined at module level its function_id is stable, so
# the compiled XLA kernel is reused for every NPL step (same shapes, different
# concrete values of z_tilde/e_tilde).
_ccp_logit_neg_ll_and_grad = jax.jit(jax.value_and_grad(_ccp_logit_neg_ll))


class CCPEstimator(BaseEstimator):
    """CCP-based estimator for dynamic discrete choice models.

    Implements both Hotz-Miller (K=1) and NPL (K>1) estimation via the
    `num_policy_iterations` parameter.

    The algorithm:
    1. Estimate CCPs from data using frequency estimator
    2. For k = 1, ..., K:
       a) Compute emax correction: e(a,x) = γ - log(P(a|x))
       b) Compute valuation matrix via matrix inversion (eq 42, A&M 2010)
       c) Maximize pseudo-likelihood to get θ̂_k
       d) Update CCPs from θ̂_k (for NPL)
    3. Return final estimates

    Attributes:
        num_policy_iterations: Number of NPL iterations (K=1 is Hotz-Miller)
        ccp_min_count: Minimum observations per state for CCP estimation
        convergence_tol: Joint tolerance for parameter and policy residuals

    Example:
        >>> # Hotz-Miller (fast, one-step)
        >>> hm = CCPEstimator(num_policy_iterations=1)
        >>> result = hm.estimate(panel, utility, problem, transitions)
        >>>
        >>> # Fixed-stage NPL
        >>> npl = CCPEstimator(num_policy_iterations=10)
        >>> result = npl.estimate(panel, utility, problem, transitions)
    """

    def __init__(
        self,
        mode: Literal["one_step", "npl"] | None = None,
        num_policy_iterations: int | None = None,
        ccp_min_count: int = 1,
        ccp_smoothing: float = 1e-6,
        convergence_tol: float = 1e-10,
        outer_tol: float = 1e-6,
        outer_max_iter: int = 1000,
        se_method: SEMethod = "asymptotic",
        compute_hessian: bool = True,
        verbose: bool = False,
    ):
        """Initialize the CCP estimator.

        Args:
            mode: Estimator mode following the JSS paper Versions paragraph.
                "one_step" stops at K=1 with empirical CCPs (Hotz-Miller 1993).
                "npl" iterates to convergence (Aguirregabiria-Mira 2002).
                If both `mode` and `num_policy_iterations` are None the default
                is "one_step".
            num_policy_iterations: Legacy iteration-count selector. K=1 is the
                one-step estimator, K>1 runs NPL for at most K iterations, and
                K=-1 runs NPL until joint parameter and policy convergence.
                When `mode` is set the K value is derived from it
                (one_step -> 1, npl -> -1).
            ccp_min_count: Minimum observations per state for reliable CCP estimation.
                          States with fewer observations get uniform CCPs.
            ccp_smoothing: Small value added to CCPs to avoid log(0).
            convergence_tol: Tolerance applied to both the parameter L2
                residual and policy maximum residual.
            outer_tol: Tolerance for pseudo-likelihood maximization.
            outer_max_iter: Max iterations for pseudo-likelihood maximization.
            se_method: Method for computing standard errors.
            compute_hessian: Whether to compute Hessian for inference.
            verbose: Whether to print progress messages.
        """
        super().__init__(
            se_method=se_method,
            compute_hessian=compute_hessian,
            verbose=verbose,
        )
        if mode not in {None, "one_step", "npl"}:
            raise ValueError("mode must be one of None, 'one_step', or 'npl'.")
        if mode is not None and num_policy_iterations is not None:
            raise ValueError("Pass either `mode` or `num_policy_iterations`, not both.")
        if mode is not None:
            num_policy_iterations = 1 if mode == "one_step" else -1
        elif num_policy_iterations is None:
            num_policy_iterations = 1
        if num_policy_iterations != -1 and num_policy_iterations < 1:
            raise ValueError("num_policy_iterations must be a positive integer or -1.")
        if ccp_min_count < 1:
            raise ValueError("ccp_min_count must be at least 1.")
        if ccp_smoothing < 0:
            raise ValueError("ccp_smoothing must be non-negative.")
        if convergence_tol <= 0:
            raise ValueError("convergence_tol must be positive.")
        if outer_tol <= 0:
            raise ValueError("outer_tol must be positive.")
        if outer_max_iter < 1:
            raise ValueError("outer_max_iter must be at least 1.")
        self._mode = mode
        self._num_policy_iterations = num_policy_iterations
        self._ccp_min_count = ccp_min_count
        self._ccp_smoothing = ccp_smoothing
        self._convergence_tol = convergence_tol
        self._outer_tol = outer_tol
        self._outer_max_iter = outer_max_iter

    @property
    def name(self) -> str:
        if self._num_policy_iterations == 1:
            return "Hotz-Miller (CCP)"
        elif self._num_policy_iterations == -1:
            return "NPL (until convergence)"
        else:
            return f"NPL (K={self._num_policy_iterations})"

    def _estimate_ccps_from_data(
        self,
        panel: Panel,
        num_states: int,
        num_actions: int,
    ) -> jnp.ndarray:
        """Estimate CCPs from data using frequency estimator.

        P̂(a|s) = N(s,a) / N(s)

        Args:
            panel: Panel data with observed choices
            num_states: Number of states
            num_actions: Number of actions

        Returns:
            CCP matrix of shape (num_states, num_actions)
        """
        dtype = jnp.float64

        # Count state-action frequencies.
        all_states = jnp.asarray(panel.get_all_states(), dtype=jnp.int32)
        all_actions = jnp.asarray(panel.get_all_actions(), dtype=jnp.int32)
        idx = all_states * num_actions + all_actions
        counts = (
            jnp.zeros(num_states * num_actions, dtype=dtype)
            .at[idx]
            .add(jnp.ones(idx.shape[0], dtype=dtype))
            .reshape(num_states, num_actions)
        )

        state_counts = counts.sum(axis=1, keepdims=True)
        smoothing = jnp.asarray(self._ccp_smoothing, dtype=dtype)

        denom = state_counts + num_actions * smoothing
        safe_denom = jnp.where(denom > 0, denom, jnp.ones_like(denom))
        empirical = (counts + smoothing) / safe_denom

        uniform = jnp.full(
            (num_states, num_actions),
            1.0 / num_actions,
            dtype=dtype,
        )
        supported = state_counts >= self._ccp_min_count
        return jnp.where(supported, empirical, uniform)

    def _compute_emax_correction(self, ccps: jnp.ndarray) -> jnp.ndarray:
        """Compute emax correction for logit errors.

        e(a,x) = γ - log(P(a|x))

        where γ ≈ 0.5772 is Euler's constant.

        Args:
            ccps: CCP matrix of shape (num_states, num_actions)

        Returns:
            Emax correction matrix of shape (num_states, num_actions)
        """
        # Clamp CCPs to avoid log(0), even when callers request zero smoothing
        # for exact frequency checks.
        tiny = np.finfo(np.float64).tiny
        floor = jnp.asarray(max(float(self._ccp_smoothing), tiny), dtype=ccps.dtype)
        ccps_safe = jnp.clip(ccps, floor, 1.0)
        return EULER_GAMMA - jnp.log(ccps_safe)

    def _compute_policy_weighted_transitions(
        self,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute policy-weighted transition matrix F_π.

        F_π[s, s'] = Σ_a P(a|s) * P(s'|s,a)

        Args:
            ccps: CCP matrix of shape (num_states, num_actions)
            transitions: Transition matrices of shape (num_actions, num_states, num_states)

        Returns:
            Policy-weighted transition matrix of shape (num_states, num_states)
        """
        num_states = ccps.shape[0]
        num_actions = ccps.shape[1]

        dtype = transitions.dtype
        F_pi = jnp.zeros((num_states, num_states), dtype=dtype)
        for a in range(num_actions):
            # transitions[a] has shape (num_states, num_states)
            # ccps[:, a] has shape (num_states,)
            # We want F_pi[s, s'] += P(a|s) * P(s'|s,a)
            F_pi = F_pi + ccps[:, a : a + 1].astype(dtype) * transitions[a]

        return F_pi

    def _compute_valuation_matrix(
        self,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
        utility: UtilityFunction,
        parameters: jnp.ndarray,
        problem: DDCProblem,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute valuation matrix W^P via matrix inversion.

        W^P = (I - β·F_π)⁻¹ · Σ_a P(a) ⊙ [u(a), e(a)]

        Following equation 42 in Aguirregabiria & Mira (2010).

        Args:
            ccps: CCP matrix of shape (num_states, num_actions)
            transitions: Transition matrices
            utility: Utility function
            parameters: Current parameter estimate
            problem: Problem specification

        Returns:
            Tuple of (W_z, W_e) where:
            - W_z: shape (num_states, num_features) for utility contribution
            - W_e: shape (num_states,) for emax contribution
        """
        beta = problem.discount_factor
        num_states = problem.num_states

        # Compute policy-weighted transition matrix
        F_pi = self._compute_policy_weighted_transitions(ccps, transitions)

        # Compute (I - β·F_π)⁻¹
        dtype = F_pi.dtype
        identity = jnp.eye(num_states, dtype=dtype)
        inv_matrix = jnp.linalg.inv(identity - beta * F_pi)

        # Compute emax corrections
        e = self._compute_emax_correction(ccps)

        # Euler's constant is action invariant. Omitting it fixes the arbitrary
        # value-function level without changing choice probabilities.
        normalized_e = e - EULER_GAMMA
        expected_e = (problem.scale_parameter * (ccps * normalized_e).sum(axis=1)).astype(
            inv_matrix.dtype
        )

        # W_e = (I - β·F_π)⁻¹ · expected_e
        W_e = inv_matrix @ expected_e

        # For utility contribution, we need the feature matrix
        # u(s,a) = θ · φ(s,a), so we compute expected features
        if hasattr(utility, "feature_matrix"):
            features = utility.feature_matrix  # shape (num_states, num_actions, num_features)

            # Compute Σ_a P(a|s) · φ(s,a) for each state
            # expected_features[s, k] = Σ_a P(a|s) · φ(s,a,k)
            dtype = inv_matrix.dtype
            expected_features = jnp.einsum("sa,sak->sk", ccps.astype(dtype), features.astype(dtype))

            # W_z = (I - β·F_π)⁻¹ · expected_features
            W_z = inv_matrix @ expected_features
        else:
            # Fallback: compute utility directly
            flow_utility = utility.compute(parameters)  # shape (num_states, num_actions)
            expected_utility = (ccps * flow_utility).sum(axis=1)  # shape (num_states,)
            W_z = inv_matrix @ expected_utility
            W_z = W_z[:, None]  # shape (num_states, 1)

        return W_z, W_e

    def _compute_choice_specific_values(
        self,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
        utility: UtilityFunction,
        parameters: jnp.ndarray,
        problem: DDCProblem,
    ) -> jnp.ndarray:
        """Compute choice-specific value functions using CCP representation.

        For linear utility u(a,x) = z(a,x)'θ, the Hotz-Miller representation is:
            v(a,x) = z̃(a,x)'θ + ẽ(a,x)

        where:
            z̃(a,x) = z(a,x) + β·E[W_z(x') | x, a]
            ẽ(a,x) = β·E[W_e(x') | x, a]

        This correctly separates linear (in θ) and constant terms.

        Args:
            ccps: CCP matrix
            transitions: Transition matrices
            utility: Utility function
            parameters: Current parameters
            problem: Problem specification

        Returns:
            Choice-specific values of shape (num_states, num_actions)
        """
        beta = problem.discount_factor
        num_states = problem.num_states
        num_actions = problem.num_actions

        # Compute valuation matrix components
        W_z, W_e = self._compute_valuation_matrix(ccps, transitions, utility, parameters, problem)

        if hasattr(utility, "feature_matrix"):
            features = utility.feature_matrix  # shape (num_states, num_actions, num_features)
            num_features = features.shape[2]

            # Compute E[W_z(x') | x, a] for each (x, a)
            # transitions[a, s, s'] = P(s'|s,a), W_z has shape (num_states, num_features)
            E_W_z = jnp.zeros((num_states, num_actions, num_features), dtype=transitions.dtype)
            for a in range(num_actions):
                E_W_z = E_W_z.at[:, a, :].set(transitions[a] @ W_z)  # (num_states, num_features)

            # Compute E[W_e(x') | x, a]
            E_W_e = jnp.zeros((num_states, num_actions), dtype=transitions.dtype)
            for a in range(num_actions):
                E_W_e = E_W_e.at[:, a].set(transitions[a] @ W_e)

            # z̃(a,x) = z(a,x) + β·E[W_z(x') | x, a]
            z_tilde = features.astype(E_W_z.dtype) + beta * E_W_z

            # ẽ(a,x) = β·E[W_e(x') | x, a]
            e_tilde = beta * E_W_e  # (num_states, num_actions)

            # v(a,x) = z̃(a,x)'θ + ẽ(a,x)
            v = jnp.einsum("sak,k->sa", z_tilde, parameters.astype(z_tilde.dtype)) + e_tilde
        else:
            # Fallback for non-linear utility
            flow_utility = utility.compute(parameters)
            W = W_z.squeeze(1) + W_e

            EW = jnp.zeros((num_states, num_actions), dtype=transitions.dtype)
            for a in range(num_actions):
                EW = EW.at[:, a].set(transitions[a] @ W)

            v = flow_utility + beta * EW

        return v

    def _compute_log_likelihood(
        self,
        parameters: jnp.ndarray,
        panel: Panel,
        utility: UtilityFunction,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
        problem: DDCProblem,
    ) -> float:
        """Compute pseudo-log-likelihood given CCPs.

        Args:
            parameters: Current parameter estimate
            panel: Panel data
            utility: Utility function
            ccps: Current CCP estimates
            transitions: Transition matrices
            problem: Problem specification

        Returns:
            Log-likelihood value
        """
        sigma = problem.scale_parameter

        # Compute choice-specific values
        v = self._compute_choice_specific_values(ccps, transitions, utility, parameters, problem)

        # Compute log choice probabilities via softmax
        log_probs = jax.nn.log_softmax(v / sigma, axis=1)

        # Sum log-likelihood over observations
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        ll = float(log_probs[all_states, all_actions].sum())

        return ll

    def _update_ccps_from_values(
        self,
        v: jnp.ndarray,
        sigma: float,
    ) -> jnp.ndarray:
        """Update CCPs from choice-specific values.

        P(a|x) = exp(v(a,x)/σ) / Σ_{a'} exp(v(a',x)/σ)

        Args:
            v: Choice-specific values of shape (num_states, num_actions)
            sigma: Scale parameter

        Returns:
            Updated CCPs of shape (num_states, num_actions)
        """
        return jax.nn.softmax(v / sigma, axis=1)

    def _estimate_initial_params(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
    ) -> jnp.ndarray:
        """Estimate rough starting values from data."""
        n_params = utility.num_parameters
        total_obs = 0
        n_replace = 0
        mileage_at_replace = 0.0

        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        total_obs = all_states.shape[0]
        replace_mask = all_actions == 1
        n_replace = int(replace_mask.sum())
        mileage_at_replace = float(all_states[replace_mask].astype(jnp.float32).sum())

        if n_replace > 0 and total_obs > 0 and n_params == 2:
            replace_rate = n_replace / total_obs
            avg_mileage = mileage_at_replace / n_replace
            n_states = problem.num_states
            op_cost_init = 1.0 / n_states
            rc_init = max(0.5, op_cost_init * avg_mileage / max(replace_rate, 0.01))
            return jnp.array([op_cost_init, rc_init], dtype=jnp.float32)

        return jnp.full((n_params,), 0.01, dtype=jnp.float32)

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run CCP/NPL optimization.

        Args:
            panel: Panel data with observed choices
            utility: Utility function specification
            problem: Problem specification
            transitions: Transition probability matrices
            initial_params: Starting values (defaults to zeros)

        Returns:
            EstimationResult with optimized parameters
        """
        start_time = time.time()

        # Use float64 for high discount factors (condition number ≈ 1/(1-β))
        beta = problem.discount_factor
        if beta > 0.99:
            transitions = jnp.array(transitions, dtype=jnp.float64)

        # Initialize parameters — use data-driven starting values if zeros
        if initial_params is None:
            initial_params = utility.get_initial_parameters()
            if (initial_params == 0).all():
                initial_params = self._estimate_initial_params(panel, utility, problem)

        current_params = jnp.array(initial_params)

        # Step 1: Estimate initial CCPs from data
        self._log("Estimating CCPs from data")
        ccps = self._estimate_ccps_from_data(panel, problem.num_states, problem.num_actions)

        # Track iterations
        num_policy_iterations = 0
        converged = False
        optimizer_failed = False
        inner_optimizer_history: list[dict[str, object]] = []
        npl_residual_history: list[dict[str, float | int]] = []
        final_parameter_residual: float | None = None
        final_policy_residual: float | None = None
        last_candidate_policy: jnp.ndarray | None = None
        total_function_evals = 0
        total_optimizer_iterations = 0
        # NPL-until-convergence cap. The NPL CCP fixed point contracts at rate
        # ~beta, so high-discount problems need many policy iterations to reach it;
        # the old cap of 100 with a loose 1e-6 parameter tolerance stopped well
        # short of the fixed point and missed the MLE (Aguirregabiria-Mira Lemma 2).
        max_iterations = self._num_policy_iterations if self._num_policy_iterations > 0 else 1000

        # ── One-time setup before NPL loop ─────────────────────────────────
        # These are fixed across all NPL iterations and are used to build the
        # logit objective function once (stable Python identity → jaxopt
        # JIT-compiles the solver update step once and reuses it every NPL step).
        beta = problem.discount_factor
        num_actions = problem.num_actions
        sigma = problem.scale_parameter
        lower, upper = utility.get_parameter_bounds()
        if lower is None and upper is None:
            opt_bounds = None
        else:
            if lower is None:
                lower = jnp.full((utility.num_parameters,), -jnp.inf)
            if upper is None:
                upper = jnp.full((utility.num_parameters,), jnp.inf)
            opt_bounds = (
                jnp.asarray(lower, dtype=jnp.float64),
                jnp.asarray(upper, dtype=jnp.float64),
            )
        states_arr = jnp.asarray(panel.get_all_states())
        actions_arr = jnp.asarray(panel.get_all_actions())
        inv_sigma_f64 = jnp.float64(1.0 / sigma)
        _is_linear = hasattr(utility, "feature_matrix")
        initial_ccps = ccps

        if _is_linear:
            features_f64 = jnp.array(utility.feature_matrix, dtype=jnp.float64)

            # neg_ll_base is defined ONCE — stable Python function identity.
            # z64 and e64 are passed as fun_args (dynamic args), so jaxopt
            # compiles the solver update step based on (function, shapes/dtypes)
            # and reuses the compiled kernel across all NPL steps even when
            # z64/e64 change values.
            def neg_ll_base(params_x, z64, e64):
                return _ccp_logit_neg_ll_and_grad(
                    params_x, z64, e64, states_arr, actions_arr, inv_sigma_f64
                )
        else:
            # Fallback for non-linear utility: finite differences
            def neg_ll_base(params_x, z64, e64):
                val = float(
                    -self._compute_log_likelihood(
                        jnp.array(params_x, dtype=jnp.float32),
                        panel,
                        utility,
                        ccps,
                        transitions,
                        problem,
                    )
                )
                n = len(params_x)
                grad = np.zeros(n)
                for i in range(n):
                    eps = max(1e-5, abs(float(params_x[i])) * 1e-4)
                    pf = params_x.at[i].add(eps)
                    pb = params_x.at[i].add(-eps)
                    ll_plus = self._compute_log_likelihood(
                        jnp.array(pf, dtype=jnp.float32),
                        panel,
                        utility,
                        ccps,
                        transitions,
                        problem,
                    )
                    ll_minus = self._compute_log_likelihood(
                        jnp.array(pb, dtype=jnp.float32),
                        panel,
                        utility,
                        ccps,
                        transitions,
                        problem,
                    )
                    grad[i] = (-ll_plus - (-ll_minus)) / (2 * eps)
                return val, jnp.array(grad)

        # Step 2: Policy iteration loop
        from tqdm import tqdm

        pbar = tqdm(
            range(max_iterations),
            desc="CCP NPL",
            disable=not self._verbose,
            leave=True,
        )
        for k in pbar:
            num_policy_iterations = k + 1

            prev_params = jnp.array(current_params)

            # A&M (2002) algorithm: compute the valuation matrix ONCE per NPL step
            # from the fixed CCPs. W_z and W_e depend only on CCPs (not theta), so
            # they do not change during the logit optimization step. Caching them
            # reduces per-call cost from O(S^3) to O(S*A*K) for the pseudo-LL.
            W_z, W_e = self._compute_valuation_matrix(
                ccps, transitions, utility, current_params, problem
            )

            # Compute augmented features z_tilde and e_tilde from cached W_z, W_e.
            # v(s,a; theta) = z_tilde(s,a)' * theta + e_tilde(s,a)
            if _is_linear:
                E_W_z = jnp.stack([transitions[a] @ W_z for a in range(num_actions)], axis=1)
                E_W_e = jnp.stack([transitions[a] @ W_e for a in range(num_actions)], axis=1)
                z64 = features_f64 + beta * E_W_z.astype(jnp.float64)  # (S, A, K)
                e64 = (beta * E_W_e).astype(jnp.float64)  # (S, A)
                fun_args = (z64, e64)
            else:
                fun_args = (None, None)

            result = minimize_lbfgsb(
                neg_ll_base,
                jnp.array(current_params, dtype=jnp.float64),
                bounds=opt_bounds,
                maxiter=self._outer_max_iter,
                tol=self._outer_tol,
                verbose=False,
                desc="CCP",
                value_and_grad=True,
                param_names=list(utility.parameter_names),
                fun_args=fun_args,
            )

            current_params = jnp.array(result.x, dtype=jnp.float32)
            current_ll = -result.fun
            total_function_evals += int(result.nfev)
            total_optimizer_iterations += int(result.nit)
            inner_succeeded = bool(
                result.success or result.convergence_reason == "objective_plateau"
            )
            inner_optimizer_history.append(
                {
                    "policy_iteration": num_policy_iterations,
                    "success": bool(result.success),
                    "accepted": inner_succeeded,
                    "message": result.message,
                    "iterations": int(result.nit),
                    "function_evals": int(result.nfev),
                    "gradient_norm": result.grad_norm,
                    "projected_gradient_norm": result.projected_grad_norm,
                    "convergence_reason": result.convergence_reason,
                }
            )
            if not inner_succeeded:
                optimizer_failed = True
                self._log(
                    f"CCP inner optimizer failed at policy iteration "
                    f"{num_policy_iterations}: {result.message}"
                )
                break

            # A&M's NPL fixed point is a CCP vector P satisfying
            # P = Psi(P, theta). Parameter stability alone does not establish
            # that fixed point, so every accepted stage records both residuals.
            candidate_values = self._compute_choice_specific_values(
                ccps,
                transitions,
                utility,
                current_params,
                problem,
            )
            candidate_policy = self._update_ccps_from_values(
                candidate_values,
                problem.scale_parameter,
            )
            parameter_residual = float(jnp.linalg.norm(current_params - prev_params))
            policy_residual = float(jnp.max(jnp.abs(candidate_policy - ccps)))
            final_parameter_residual = parameter_residual
            final_policy_residual = policy_residual
            last_candidate_policy = candidate_policy
            residual_record: dict[str, float | int] = {
                "policy_iteration": num_policy_iterations,
                "parameter_residual": parameter_residual,
                "policy_residual": policy_residual,
            }
            npl_residual_history.append(residual_record)
            inner_optimizer_history[-1].update(residual_record)

            postfix = {
                "LL": f"{current_ll:.2f}",
                "d_param": f"{parameter_residual:.1e}",
                "d_policy": f"{policy_residual:.1e}",
            }
            for j, nm in enumerate(utility.parameter_names[:3]):
                postfix[nm] = f"{float(current_params[j]):.5f}"
            pbar.set_postfix(postfix)

            joint_fixed_point = (
                parameter_residual <= self._convergence_tol
                and policy_residual <= self._convergence_tol
            )
            if self._num_policy_iterations != 1 and joint_fixed_point:
                converged = True
                pbar.set_postfix({**postfix, "status": "converged"})
                pbar.close()
                self._log("NPL parameter and policy residuals converged")
                break

            # Update CCPs for next iteration (if doing NPL)
            if k < max_iterations - 1:
                ccps = candidate_policy

            # Stop if only doing Hotz-Miller (K=1)
            if self._num_policy_iterations == 1:
                break

        # Compute final value function and policy
        if last_candidate_policy is None:
            v = self._compute_choice_specific_values(
                ccps,
                transitions,
                utility,
                current_params,
                problem,
            )
            final_policy = self._update_ccps_from_values(v, problem.scale_parameter)
        else:
            final_policy = last_candidate_policy

        # Report value in the package's soft-Bellman convention. The CCP
        # inversion uses the Euler-constant emax correction internally; directly
        # applying logsumexp to those CCP choice-specific values returns the
        # same policy but a value level shifted by that representation. For
        # diagnostics and known-truth comparisons, evaluate the recovered policy
        # under the recovered flow reward without the Euler-constant offset.
        sigma = problem.scale_parameter
        recovered_reward = utility.compute(current_params).astype(jnp.float64)
        policy_eval = final_policy.astype(jnp.float64)
        transitions_eval = transitions.astype(jnp.float64)
        clipped_policy = jnp.clip(policy_eval, 1e-12, 1.0)
        reward_pi = jnp.sum(policy_eval * recovered_reward, axis=1)
        entropy_pi = -sigma * jnp.sum(policy_eval * jnp.log(clipped_policy), axis=1)
        transition_pi = jnp.einsum("sa,ast->st", policy_eval, transitions_eval)
        lhs = jnp.eye(problem.num_states, dtype=jnp.float64) - beta * transition_pi
        V = jnp.linalg.solve(lhs, reward_pi + entropy_pi).astype(jnp.float32)

        # Compute final log-likelihood
        final_ll = self._compute_log_likelihood(
            current_params, panel, utility, ccps, transitions, problem
        )

        # Compute Hessian for standard errors
        hessian = None
        gradient_contributions = None
        full_likelihood_metadata = None

        if self._compute_hessian:
            self._log("Computing Hessian for standard errors")

            if self._se_method == "full_likelihood_bhhh":
                if self._num_policy_iterations != -1 or not converged:
                    raise ValueError(
                        "se_method='full_likelihood_bhhh' requires "
                        "num_policy_iterations=-1 and NPL fixed-point convergence."
                    )
                if not _is_linear:
                    raise ValueError("se_method='full_likelihood_bhhh' requires a linear utility.")
                transition_probabilities = kwargs.get("transition_probabilities")
                transition_increments = kwargs.get("transition_increments")
                if transition_probabilities is None or transition_increments is None:
                    raise ValueError(
                        "se_method='full_likelihood_bhhh' requires "
                        "transition_probabilities and transition_increments."
                    )

                operator = SoftBellmanOperator(problem, transitions_eval)
                bellman_result = policy_iteration(
                    operator,
                    recovered_reward,
                    tol=1e-12,
                    max_iter=1000,
                )
                if not bellman_result.converged:
                    raise RuntimeError(
                        "Bellman policy iteration did not converge for "
                        "full-likelihood BHHH inference."
                    )
                bellman_policy_residual = float(
                    jnp.max(jnp.abs(bellman_result.policy - final_policy))
                )
                if bellman_policy_residual > 1e-8:
                    raise RuntimeError(
                        "The converged NPL policy does not match the Bellman policy "
                        "at the fitted parameters "
                        f"(residual={bellman_policy_residual:.3e})."
                    )

                V = bellman_result.V
                final_policy = bellman_result.policy
                gradient_contributions, full_likelihood_metadata = (
                    compute_rust_full_likelihood_bhhh_score(
                        panel=panel,
                        utility=utility,
                        problem=problem,
                        transitions=transitions_eval,
                        value_function=V,
                        policy=final_policy,
                        transition_probabilities=jnp.asarray(
                            transition_probabilities,
                            dtype=jnp.float64,
                        ),
                        transition_increments=jnp.asarray(transition_increments),
                    )
                )
                full_likelihood_metadata.update(
                    {
                        "bellman_policy_residual": bellman_policy_residual,
                        "bellman_iterations": bellman_result.num_iterations,
                    }
                )
            elif _is_linear:
                # Use the same fixed-CCP pseudo-likelihood for estimation and
                # inference. Mixing its score with the full structural Hessian
                # at a finite-stage estimate makes the sandwich inconsistent
                # and can make the Hessian indefinite.
                v_final = (
                    jnp.einsum(
                        "sak,k->sa",
                        z64,
                        current_params.astype(jnp.float64),
                    )
                    + e64
                )
                probs = jax.nn.softmax(v_final * inv_sigma_f64, axis=1)
                mean_features = jnp.einsum("sa,sak->sk", probs, z64)
                centered = z64 - mean_features[:, None, :]
                covariance_by_state = jnp.einsum(
                    "sa,sak,sal->skl",
                    probs,
                    centered,
                    centered,
                )
                state_counts = jnp.bincount(
                    states_arr,
                    length=problem.num_states,
                ).astype(jnp.float64)
                hessian = -jnp.einsum(
                    "s,skl->kl",
                    state_counts,
                    covariance_by_state,
                ) * (inv_sigma_f64**2)
                gradient_contributions = (
                    z64[states_arr, actions_arr] - mean_features[states_arr]
                ) * inv_sigma_f64
            else:
                operator = SoftBellmanOperator(problem, transitions)

                def ll_fn(params):
                    flow_u = utility.compute(params).astype(transitions.dtype)
                    from econirl.core.solvers import value_iteration

                    sol = value_iteration(
                        operator,
                        flow_u,
                        tol=1e-12,
                        max_iter=100_000,
                    )
                    log_probs = operator.compute_log_choice_probabilities(
                        flow_u,
                        sol.V,
                    )
                    return log_probs[states_arr, actions_arr].sum()

                hessian = compute_numerical_hessian(current_params, ll_fn)
                gradient_contributions = self._compute_gradient_contributions(
                    current_params,
                    panel,
                    utility,
                    ccps,
                    transitions,
                    problem,
                )

        optimization_time = time.time() - start_time

        if optimizer_failed:
            termination_reason = "inner_optimizer_failed"
            run_succeeded = False
            message = f"CCP inner optimizer failed after {num_policy_iterations} policy iterations"
        elif self._num_policy_iterations == 1:
            termination_reason = "one_step_complete"
            run_succeeded = True
            message = "Hotz-Miller one-step estimation completed"
        elif self._num_policy_iterations > 1:
            termination_reason = "fixed_point_converged" if converged else "fixed_k_complete"
            run_succeeded = converged or num_policy_iterations == self._num_policy_iterations
            message = (
                f"NPL fixed point converged in {num_policy_iterations} policy iterations"
                if converged
                else f"NPL completed the requested {num_policy_iterations} policy iterations"
            )
        else:
            termination_reason = "fixed_point_converged" if converged else "iteration_cap_reached"
            run_succeeded = converged
            message = (
                f"NPL fixed point converged in {num_policy_iterations} policy iterations"
                if converged
                else f"NPL reached the {max_iterations}-iteration cap without convergence"
            )

        return EstimationResult(
            parameters=current_params,
            log_likelihood=final_ll,
            value_function=V,
            policy=final_policy,
            hessian=hessian,
            gradient_contributions=gradient_contributions,
            converged=run_succeeded,
            num_iterations=num_policy_iterations,
            num_function_evals=total_function_evals,
            num_inner_iterations=total_optimizer_iterations,
            message=message,
            optimization_time=optimization_time,
            metadata={
                "mode": ("one_step" if self._num_policy_iterations == 1 else "npl"),
                "num_policy_iterations": num_policy_iterations,
                "npl_converged": converged,
                "termination_reason": termination_reason,
                "npl_parameter_residual": final_parameter_residual,
                "npl_policy_residual": final_policy_residual,
                "npl_convergence_tolerance": self._convergence_tol,
                "npl_residual_history": npl_residual_history,
                "inner_optimizer_succeeded": not optimizer_failed,
                "inner_optimizer_history": inner_optimizer_history,
                "requested_policy_iterations": self._num_policy_iterations,
                "ccp_min_count": self._ccp_min_count,
                "ccp_smoothing": self._ccp_smoothing,
                "initial_ccps": initial_ccps,
                "final_ccps": final_policy,
                "min_initial_ccp": float(jnp.min(initial_ccps)),
                "min_final_ccp": float(jnp.min(final_policy)),
                "outer_tol": self._outer_tol,
                "outer_max_iter": self._outer_max_iter,
                "optimizer": "L-BFGS-B",
                "se_method_detail": (
                    "joint_full_likelihood_bhhh"
                    if self._se_method == "full_likelihood_bhhh"
                    else (
                        "fixed_ccp_pseudo_likelihood"
                        if _is_linear
                        else "full_structural_likelihood_fallback"
                    )
                ),
                **(
                    {"full_likelihood_bhhh": full_likelihood_metadata}
                    if full_likelihood_metadata is not None
                    else {}
                ),
            },
        )

    def _compute_gradient_contributions(
        self,
        params: jnp.ndarray,
        panel: Panel,
        utility: UtilityFunction,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
        problem: DDCProblem,
        eps: float = 1e-5,
    ) -> jnp.ndarray:
        """Compute per-observation gradient contributions for robust SEs."""
        n_obs = panel.num_observations
        n_params = len(params)

        gradients = jnp.zeros((n_obs, n_params))
        sigma = problem.scale_parameter

        # Compute gradient for each parameter
        for k in range(n_params):
            params_plus = params.at[k].add(eps)
            params_minus = params.at[k].add(-eps)

            v_plus = self._compute_choice_specific_values(
                ccps, transitions, utility, params_plus, problem
            )
            log_probs_plus = jax.nn.log_softmax(v_plus / sigma, axis=1)

            v_minus = self._compute_choice_specific_values(
                ccps, transitions, utility, params_minus, problem
            )
            log_probs_minus = jax.nn.log_softmax(v_minus / sigma, axis=1)

            # Compute gradients for all observations
            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            gradients = gradients.at[:, k].set(
                (log_probs_plus[all_states, all_actions] - log_probs_minus[all_states, all_actions])
                / (2 * eps)
            )

        return gradients
