"""Unnested Fixed Point (UFXP) estimator.

Implements the UFXP estimator of Bray ("Unnesting the Fixed Point in the
Estimation of Dynamic Programs") in the form presented by Oguz and Bray
("Training Neural Networks Embedded in Dynamic Discrete Choice Models", 2026):
estimation from the first-order conditions of Bellman's equation, with the
value-function dependence removed by a dual representation that is computed
once, before the parameter search.

The first-order condition at state ``x`` and action ``a != A`` (the reference
action) under logit shocks with scale ``sigma`` is

    sigma * log(P_a(x) / P_A(x)) = U_a(x) - U_A(x)
                                   + beta * (F_a - F_A) V_P(x),

where ``V_P`` is the value of following the empirical choice probabilities
``P`` (the package's soft-Bellman convention has no Euler-gamma constant:
``V = sigma * logsumexp(Q / sigma)``, so ``V = sum_a P_a (Q_a - sigma log
P_a)`` exactly):

    V_P = (I - beta * F_P)^{-1} [ sum_a P_a o (U_a - sigma * log P_a) ],
    F_P = sum_a diag(P_a) F_a.

(A Euler-gamma term would add a constant per state, which the differenced
transitions ``F_a - F_A`` annihilate, so either convention yields identical
estimates; the package's is used for consistency.)

UFXP scores ``m`` random projections ``Z_i`` of the stacked FOC residuals.
The only ``V_P``-dependent term in each projection is a fixed linear
functional ``w_i' V_P`` with

    w_i' = - beta * sum_{a != A} Z_{ia}' (F_a - F_A),

and Proposition 2 of the paper replaces it by ``lambda_i' u_P`` where
``lambda_i`` solves ``lambda_i = w_i + beta F_P' lambda_i`` -- a fixed point in
the empirical CCPs only, independent of the parameters. All ``m`` duals are
obtained from one factorization of ``(I - beta F_P')``, after which no linear
system is ever solved again: for a linear-in-parameters utility
``U_a(x) = phi(x, a)' theta`` every projected residual is affine in ``theta``
and the UFXP objective ``sum_i residual_i^2`` is an ordinary least-squares
problem with a closed-form solution.

Two weighting modes are implemented. ``weights="optimal"`` is the paper's
OUFXP: per-state optimal weights built from the CCP covariance and the
inverse-CCP Jacobian. For linear utility the weights are independent of theta,
so OUFXP stays a single closed-form weighted moment solve and, by Theorem 2,
is asymptotically as efficient as maximum likelihood; its standard errors come
from the efficient moment variance ``(sum_x z(x)' G(x))^{-1} / N``.
``weights="random"`` is plain UFXP with ``m`` random projections — consistent
but less efficient, kept for the paper-faithful baseline and no standard
errors.

Scope of this implementation:

- Linear utility (``LinearUtility`` / the uniform benchmark path). The paper's
  neural-utility training loop is out of scope here.

First-order conditions are used only at states observed at least
``ccp_min_count`` times; unvisited states still enter ``V_P`` through the
transition structure (with uniform CCPs substituted there, as in the package's
CCP estimator). Under optimal weights, thin states are additionally
downweighted by their sample share and unvisited states drop out entirely.
"""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np

from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.preferences.base import UtilityFunction


class UFXPEstimator(BaseEstimator):
    """Unnested Fixed Point estimator (Bray; Oguz and Bray 2026).

    Two weighting modes:

    - ``weights="optimal"`` (default, the paper's OUFXP): per-state optimal
      weights ``z(x) = [Gamma(x) Sigma(x) Gamma(x)' / eta(x)]^{-1} G(x)`` with
      ``eta(x) = N(x)/N`` the state's sample share, ``Sigma(x)`` the
      multinomial CCP covariance, ``Gamma(x)`` the Jacobian of the inverse CCP
      map, and ``G(x)`` the moment Jacobian. For linear utility every piece is
      independent of theta, so the estimator stays closed form, and Theorem 2
      of the paper gives asymptotic efficiency equal to maximum likelihood.
      Standard errors come from the efficient moment variance
      ``Var(theta) = (sum_x z(x)' G(x))^{-1} / N``, delivered through the
      package's asymptotic-SE pipeline. Thin states are downweighted by
      ``eta(x)``; unvisited states drop out.
    - ``weights="random"`` (plain UFXP): ``m`` random projections of the
      stacked FOCs, consistent but less efficient, no standard errors.

    Attributes:
        weights: ``"optimal"`` (OUFXP) or ``"random"`` (plain UFXP).
        num_projections: Number of random projections ``m`` for
            ``weights="random"`` (must exceed the parameter count).
        ccp_min_count: Minimum visits for a state's FOCs to be scored.
        ccp_smoothing: Additive smoothing for the frequency CCPs.
        seed: Seed for the random projection matrices.
    """

    def __init__(
        self,
        weights: str = "optimal",
        num_projections: int = 32,
        ccp_min_count: int = 1,
        ccp_smoothing: float = 1e-6,
        seed: int = 0,
        compute_hessian: bool = True,
        verbose: bool = False,
    ):
        super().__init__(se_method="asymptotic", compute_hessian=compute_hessian,
                         verbose=verbose)
        if weights not in ("optimal", "random"):
            raise ValueError(f"weights must be 'optimal' or 'random', got {weights!r}")
        if num_projections < 1:
            raise ValueError("num_projections must be positive")
        self._weights = weights
        self._num_projections = num_projections
        self._ccp_min_count = ccp_min_count
        self._ccp_smoothing = ccp_smoothing
        self._seed = seed

    @property
    def name(self) -> str:
        return "UFXP"

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        t0 = time.time()
        F = np.asarray(transitions, dtype=np.float64)  # (A, S, S)
        A, S, _ = F.shape
        phi = np.asarray(utility.feature_matrix, dtype=np.float64)  # (S, A, K)
        K = phi.shape[2]
        beta = float(problem.discount_factor)
        sigma = float(problem.scale_parameter)
        m = self._num_projections
        if self._weights == "random" and m <= K:
            raise ValueError(f"num_projections ({m}) must exceed the parameter "
                             f"count ({K})")

        # --- Pre-computation (independent of theta) -----------------------
        # Frequency CCPs with smoothing; uniform at unsupported states.
        states = np.asarray(panel.get_all_states(), dtype=np.int64)
        actions = np.asarray(panel.get_all_actions(), dtype=np.int64)
        counts = np.zeros((S, A), dtype=np.float64)
        np.add.at(counts, (states, actions), 1.0)
        state_counts = counts.sum(axis=1, keepdims=True)
        P = (counts + self._ccp_smoothing) / np.where(
            state_counts > 0, state_counts + A * self._ccp_smoothing, 1.0)
        supported = (state_counts[:, 0] >= self._ccp_min_count)
        P[~supported] = 1.0 / A

        ref = A - 1  # reference action for the CCP inversion
        log_p = np.log(np.clip(P, 1e-300, 1.0))
        # Inverse CCP map under logit: c_a - c_ref = sigma * log(P_a / P_ref).
        logratio = sigma * (log_p[:, :ref] - log_p[:, ref:ref + 1])  # (S, A-1)
        # Policy-value flow correction: -sigma * sum_a P_a log P_a (the
        # package's gamma-free soft-Bellman convention).
        ent = -sigma * (P * log_p).sum(axis=1)  # (S,)
        # CCP-weighted transition and features.
        F_P = np.einsum("sa,asx->sx", P, F)  # (S, S)
        phi_P = np.einsum("sa,sak->sk", P, phi)  # (S, K)

        dF = F[:ref] - F[ref:ref + 1]  # (A-1, S, S)
        dphi = phi[:, :ref, :] - phi[:, ref:ref + 1, :]  # (S, A-1, K)
        hessian = None

        if self._weights == "optimal":
            # --- OUFXP: optimal per-state weights, closed form -------------
            # One factorization of (I - beta F_P) gives both theta-gradient
            # and entropy components of the policy value:
            #   dV = (I - beta F_P)^{-1} phi_P     (S, K)
            #   v_ent = (I - beta F_P)^{-1} ent    (S,)
            sol_v = np.linalg.solve(np.eye(S) - beta * F_P,
                                    np.hstack([phi_P, ent[:, None]]))
            dV, v_ent = sol_v[:, :K], sol_v[:, K]
            # Moment Jacobian and intercept: residual(x) = y(x) - G(x) theta.
            G = dphi + beta * np.einsum("asx,xk->sak", dF, dV)  # (S, A-1, K)
            y = logratio - beta * np.einsum("asx,x->sa", dF, v_ent)  # (S, A-1)

            # Gamma(x): Jacobian of the inverse CCP map, (A-1, A) per state;
            # Gamma Sigma Gamma' is invariant to the simplex gauge.
            eta = state_counts[:, 0] / max(states.shape[0], 1)  # (S,)
            D = np.zeros((K, K))
            rhs = np.zeros(K)
            n_scored = 0
            for x in range(S):
                if not supported[x] or eta[x] <= 0.0:
                    continue
                Gam = np.zeros((A - 1, A))
                for a in range(A - 1):
                    Gam[a, a] = sigma / P[x, a]
                    Gam[a, ref] = -sigma / P[x, ref]
                Sig = np.diag(P[x]) - np.outer(P[x], P[x])
                GSG = Gam @ Sig @ Gam.T  # (A-1, A-1)
                Wx = eta[x] * np.linalg.pinv(GSG)
                zx = Wx @ G[x]  # (A-1, K)
                D += G[x].T @ Wx @ G[x]
                rhs += zx.T @ y[x]
                n_scored += 1
            rank = int(np.linalg.matrix_rank(D))
            theta = np.linalg.lstsq(D, rhs, rcond=None)[0]
            # Residual of the weighted moment system; zero under full rank,
            # nonzero (and reported) when the design is rank deficient.
            obj = float(np.sum((D @ theta - rhs) ** 2))
            # Efficient variance (sum_x z'G)^{-1}/N, threaded through the
            # asymptotic-SE pipeline as Var = [-hessian]^{-1}.
            if self._compute_hessian:
                hessian = jnp.asarray(-states.shape[0] * D)
            mode_msg = (f"closed-form OUFXP over {n_scored} supported states "
                        f"(optimal weights)")
        else:
            # --- Plain UFXP: m random projections, dual trick --------------
            # Projections are zeroed at unsupported states so only observed
            # FOCs are scored.
            rng = np.random.default_rng(self._seed)
            Z = rng.standard_normal((m, S, A - 1))
            Z[:, ~supported, :] = 0.0

            # Duals: one factorization of (I - beta F_P'), m right-hand sides.
            # w_i = -beta * sum_{a != ref} (F_a - F_A)' Z_i[:, a]
            W = -beta * np.einsum("asx,msa->xm", dF, Z)  # (S, m)
            lam = np.linalg.solve(np.eye(S) - beta * F_P.T, W)  # (S, m)

            # residual_i(theta) = b_i + c_i' theta with
            #   b_i = <Z_i, logratio> + lambda_i' ent
            #   c_i = -sum_{x,a} Z_i[x,a] (phi[x,a] - phi[x,ref]) + phi_P' lambda_i
            b = np.einsum("msa,sa->m", Z, logratio) + lam.T @ ent  # (m,)
            C = -np.einsum("msa,sak->mk", Z, dphi) + lam.T @ phi_P  # (m, K)
            theta, _, rank, _ = np.linalg.lstsq(C, -b, rcond=None)
            obj = float(np.sum((C @ theta + b) ** 2))
            mode_msg = f"closed-form least squares over {m} random projections"

        # --- One model solve at theta_hat for policy and value ------------
        from econirl.core.bellman import SoftBellmanOperator
        from econirl.core.solvers import value_iteration

        theta_j = jnp.asarray(theta)
        reward = jnp.einsum("sak,k->sa", jnp.asarray(phi), theta_j)
        op = SoftBellmanOperator(problem, jnp.asarray(F))
        sol = value_iteration(op, reward)
        policy = np.asarray(sol.policy)

        ll = float(np.log(np.clip(policy[states, actions], 1e-300, None)).sum())

        return EstimationResult(
            parameters=theta_j,
            log_likelihood=ll,
            value_function=sol.V,
            policy=sol.policy,
            hessian=hessian,
            converged=bool(rank == K),
            num_iterations=1,
            message=f"{mode_msg}, objective {obj:.3e}",
            optimization_time=time.time() - t0,
            metadata={
                "weights": self._weights,
                "num_projections": m if self._weights == "random" else None,
                "ufxp_objective": obj,
                "supported_state_share": float(supported.mean()),
                "design_rank": int(rank),
            },
        )
