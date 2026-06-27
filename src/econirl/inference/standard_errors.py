"""Standard error computation methods.

This module provides multiple methods for computing standard errors
of estimated parameters in dynamic discrete choice models:

1. Asymptotic (Hessian-based): Inverse of negative Hessian
2. Robust (Sandwich): Accounts for potential misspecification
3. Bootstrap: Nonparametric resampling
4. Clustered: Cluster-robust SEs for panel data

With JAX, analytical Hessians via jax.hessian and per-observation scores
via jax.vmap(jax.grad(...)) can replace the numerical fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from econirl.core.types import Panel

SEMethod = Literal[
    "asymptotic",
    "robust",
    "bootstrap",
    "clustered",
    "full_likelihood_bhhh",
]


@dataclass
class StandardErrorResult:
    """Result of standard error computation.

    Attributes:
        standard_errors: SE for each parameter
        variance_covariance: Full variance-covariance matrix
        method: Method used to compute SEs
        details: Additional method-specific information
    """

    standard_errors: jnp.ndarray
    variance_covariance: jnp.ndarray
    method: str
    details: dict


def compute_standard_errors(
    parameters: jnp.ndarray,
    hessian: jnp.ndarray | None = None,
    gradient_contributions: jnp.ndarray | None = None,
    panel: Panel | None = None,
    log_likelihood_fn: Callable | None = None,
    method: SEMethod = "asymptotic",
    n_bootstrap: int = 400,
    seed: int | None = None,
    estimate_fn: Callable[[Panel], jnp.ndarray] | None = None,
) -> StandardErrorResult:
    """Compute standard errors using specified method.

    Args:
        parameters: Estimated parameter vector
        hessian: Hessian matrix at optimum (required for asymptotic/robust)
        gradient_contributions: Per-observation gradient contributions
                               (required for robust/clustered)
        panel: Panel data (required for bootstrap/clustered)
        log_likelihood_fn: Function to compute log-likelihood (kept for compatibility)
        method: SE computation method
        n_bootstrap: Number of bootstrap replications
        seed: Random seed for bootstrap
        estimate_fn: Function that takes a Panel and returns parameter estimates
                    (required for bootstrap)

    Returns:
        StandardErrorResult with SEs and variance-covariance matrix
    """
    if method == "asymptotic":
        return _asymptotic_se(parameters, hessian)
    elif method == "robust":
        return _robust_se(parameters, hessian, gradient_contributions)
    elif method == "bootstrap":
        return _bootstrap_se(parameters, panel, log_likelihood_fn, n_bootstrap, seed, estimate_fn)
    elif method == "clustered":
        return _clustered_se(parameters, hessian, gradient_contributions, panel)
    elif method == "full_likelihood_bhhh":
        return _full_likelihood_bhhh_se(parameters, gradient_contributions)
    else:
        raise ValueError(f"Unknown SE method: {method}")


def _asymptotic_se(
    parameters: jnp.ndarray,
    hessian: jnp.ndarray | None,
) -> StandardErrorResult:
    """Compute asymptotic standard errors from Hessian.

    Uses the inverse of the negative Hessian as the variance-covariance
    matrix, which is the standard MLE result under correct specification.

    Var(theta_hat) = [-H(theta_hat)]^{-1}
    """
    if hessian is None:
        raise ValueError("Hessian required for asymptotic SEs")

    n_params = hessian.shape[0]

    # Rust (1994) / Newey-McFadden (1994): MLE variance is inverse information.
    var_cov = None
    for ridge_factor in [0, 1e-8, 1e-6, 1e-4, 1e-2]:
        ridge = ridge_factor * jnp.eye(n_params, dtype=hessian.dtype)
        candidate = jnp.linalg.inv(-hessian + ridge)
        diag = jnp.diag(candidate)
        if bool(jnp.all(diag > 0)) and bool(jnp.all(jnp.isfinite(candidate))):
            var_cov = candidate
            break

    if var_cov is None:
        se = jnp.full((n_params,), float("nan"), dtype=hessian.dtype)
        var_cov = jnp.full((n_params, n_params), float("nan"), dtype=hessian.dtype)
        return StandardErrorResult(
            standard_errors=se,
            variance_covariance=var_cov,
            method="asymptotic",
            details={"hessian_used": True, "singular": True},
        )

    diag = jnp.diag(var_cov)
    se = jnp.sqrt(jnp.maximum(diag, 0.0))

    return StandardErrorResult(
        standard_errors=se,
        variance_covariance=var_cov,
        method="asymptotic",
        details={"hessian_used": True},
    )


def _robust_se(
    parameters: jnp.ndarray,
    hessian: jnp.ndarray | None,
    gradient_contributions: jnp.ndarray | None,
) -> StandardErrorResult:
    """Compute robust (sandwich) standard errors.

    The sandwich estimator is:
        Var(theta_hat) = H^{-1} @ B @ H^{-1}

    where:
        H = Hessian (negative expected information)
        B = sum_i g_i g_i' (outer product of gradients)

    This is robust to misspecification of the likelihood.
    """
    if hessian is None:
        raise ValueError("Hessian required for robust SEs")
    if gradient_contributions is None:
        raise ValueError("Gradient contributions required for robust SEs")

    n_params = hessian.shape[0]
    H_inv = None
    for ridge_factor in [0, 1e-8, 1e-6, 1e-4, 1e-2]:
        ridge = ridge_factor * jnp.eye(n_params, dtype=hessian.dtype)
        candidate = jnp.linalg.inv(-hessian + ridge)
        if bool(jnp.all(jnp.isfinite(candidate))):
            H_inv = candidate
            break

    if H_inv is None:
        se = jnp.full((n_params,), float("nan"), dtype=hessian.dtype)
        var_cov = jnp.full((n_params, n_params), float("nan"), dtype=hessian.dtype)
        return StandardErrorResult(
            standard_errors=se,
            variance_covariance=var_cov,
            method="robust",
            details={"n_observations": gradient_contributions.shape[0], "singular": True},
        )

    # White (1982): misspecification-robust sandwich uses OPG meat.
    B = gradient_contributions.T @ gradient_contributions

    # Sandwich: H^{-1} B H^{-1}
    var_cov = H_inv @ B @ H_inv

    diag = jnp.diag(var_cov)
    se = jnp.sqrt(jnp.maximum(diag, 0.0))

    return StandardErrorResult(
        standard_errors=se,
        variance_covariance=var_cov,
        method="robust",
        details={"n_observations": gradient_contributions.shape[0]},
    )


def _bootstrap_se(
    parameters: jnp.ndarray,
    panel: Panel | None,
    log_likelihood_fn: Callable | None,
    n_bootstrap: int,
    seed: int | None,
    estimate_fn: Callable[[Panel], jnp.ndarray] | None = None,
) -> StandardErrorResult:
    """Compute bootstrap standard errors.

    Resamples individuals (with replacement) and re-estimates the model
    on each bootstrap sample. The SE is the standard deviation of the
    bootstrap distribution.
    """
    if panel is None:
        raise ValueError("Panel required for bootstrap SEs")
    if estimate_fn is None:
        raise ValueError(
            "estimate_fn required for bootstrap SEs. "
            "Bootstrap requires re-estimating the model on each resampled panel."
        )

    rng = np.random.default_rng(seed)
    n_individuals = panel.num_individuals
    n_params = len(parameters)
    bootstrap_estimates: list[np.ndarray] = []
    failed_bootstraps = 0

    # Pairs cluster bootstrap: resample whole individuals (clusters), not
    # observations (Cameron and Miller 2015, p.12). B>=400 advised for SEs.
    for b in range(n_bootstrap):
        indices = rng.choice(n_individuals, size=n_individuals, replace=True)
        resampled_trajectories = [panel.trajectories[i] for i in indices]
        bootstrap_panel = Panel(trajectories=resampled_trajectories)

        try:
            bootstrap_params = estimate_fn(bootstrap_panel)
            bootstrap_estimates.append(np.asarray(bootstrap_params, dtype=float))
        except Exception:
            failed_bootstraps += 1

    if len(bootstrap_estimates) > 1:
        boot_matrix = np.stack(bootstrap_estimates)
        se = jnp.array(np.std(boot_matrix, axis=0, ddof=1))
        if n_params == 1:
            var_cov = jnp.array([[float(np.var(boot_matrix[:, 0], ddof=1))]])
        else:
            var_cov = jnp.array(np.cov(boot_matrix, rowvar=False))
    else:
        var_cov = jnp.full((n_params, n_params), float("nan"), dtype=jnp.float64)
        se = jnp.full((n_params,), float("nan"), dtype=jnp.float64)

    return StandardErrorResult(
        standard_errors=se,
        variance_covariance=var_cov,
        method="bootstrap",
        details={
            "n_bootstrap": n_bootstrap,
            "successful_bootstraps": len(bootstrap_estimates),
            "failed_bootstraps": failed_bootstraps,
            "seed": seed,
        },
    )


def _clustered_se(
    parameters: jnp.ndarray,
    hessian: jnp.ndarray | None,
    gradient_contributions: jnp.ndarray | None,
    panel: Panel | None,
) -> StandardErrorResult:
    """Compute cluster-robust standard errors.

    Accounts for correlation of observations within individuals
    (clusters). The variance estimator is:

        Var(theta_hat) = H^{-1} @ B_cluster @ H^{-1}

    where:
        B_cluster = sum_c (sum_{i in c} g_i)(sum_{i in c} g_i)'

    This sums gradients within clusters before taking outer products.
    """
    if hessian is None:
        raise ValueError("Hessian required for clustered SEs")
    if gradient_contributions is None:
        raise ValueError("Gradient contributions required for clustered SEs")
    if panel is None:
        raise ValueError("Panel required for clustered SEs")

    n_params = hessian.shape[0]
    H_inv = None
    for ridge_factor in [0, 1e-8, 1e-6, 1e-4, 1e-2]:
        ridge = ridge_factor * jnp.eye(n_params, dtype=hessian.dtype)
        candidate = jnp.linalg.inv(-hessian + ridge)
        if bool(jnp.all(jnp.isfinite(candidate))):
            H_inv = candidate
            break

    if H_inv is None:
        se = jnp.full((n_params,), float("nan"), dtype=hessian.dtype)
        var_cov = jnp.full((n_params, n_params), float("nan"), dtype=hessian.dtype)
        return StandardErrorResult(
            standard_errors=se,
            variance_covariance=var_cov,
            method="clustered",
            details={"singular": True},
        )

    n_clusters = panel.num_individuals

    # Sum gradients within each cluster
    cluster_gradients = np.zeros((n_clusters, n_params))
    obs_idx = 0
    for c, traj in enumerate(panel.trajectories):
        n_obs_c = len(traj)
        cluster_gradients[c] = np.asarray(
            gradient_contributions[obs_idx : obs_idx + n_obs_c].sum(axis=0)
        )
        obs_idx += n_obs_c

    cluster_gradients = jnp.array(cluster_gradients)

    # Cameron-Miller (2015): cluster meat sums scores within each cluster first.
    B_cluster = cluster_gradients.T @ cluster_gradients

    # Small sample correction
    G = n_clusters
    N = panel.num_observations
    K = n_params
    # Stata's standard finite-cluster correction (Cameron and Miller 2015, p.11).
    correction = (G / (G - 1)) * (N - 1) / (N - K) if G > 1 and N > K else 1.0

    var_cov = correction * (H_inv @ B_cluster @ H_inv)

    diag = jnp.diag(var_cov)
    se = jnp.sqrt(jnp.maximum(diag, 0.0))

    return StandardErrorResult(
        standard_errors=se,
        variance_covariance=var_cov,
        method="clustered",
        details={
            "n_clusters": n_clusters,
            "n_observations": N,
            "small_sample_correction": correction,
        },
    )


def _full_likelihood_bhhh_se(
    parameters: jnp.ndarray,
    gradient_contributions: jnp.ndarray | None,
) -> StandardErrorResult:
    """Compute structural SEs from a joint full-likelihood BHHH matrix.

    ``gradient_contributions`` must contain per-observation scores for the
    joint parameter vector, with the structural parameters first and nuisance
    transition probabilities after them. The returned covariance is the
    structural block of ``inv(G'G)``.
    """
    if gradient_contributions is None:
        raise ValueError(
            "Gradient contributions required for full_likelihood_bhhh SEs"
        )

    n_structural = len(parameters)
    if gradient_contributions.ndim != 2:
        raise ValueError("gradient_contributions must be a two-dimensional matrix")
    if gradient_contributions.shape[1] < n_structural:
        raise ValueError(
            "gradient_contributions must include all structural score columns"
        )

    scores = jnp.asarray(gradient_contributions, dtype=jnp.float64)
    n_joint = int(scores.shape[1])
    opg = scores.T @ scores

    joint_cov = None
    for ridge_factor in [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        ridge = ridge_factor * jnp.eye(n_joint, dtype=opg.dtype)
        candidate = jnp.linalg.inv(opg + ridge)
        diag = jnp.diag(candidate)
        if bool(jnp.all(diag > 0)) and bool(jnp.all(jnp.isfinite(candidate))):
            joint_cov = candidate
            break

    if joint_cov is None:
        structural_cov = jnp.full(
            (n_structural, n_structural), float("nan"), dtype=scores.dtype
        )
        se = jnp.full((n_structural,), float("nan"), dtype=scores.dtype)
        return StandardErrorResult(
            standard_errors=se,
            variance_covariance=structural_cov,
            method="full_likelihood_bhhh",
            details={
                "n_observations": scores.shape[0],
                "n_joint_parameters": n_joint,
                "n_structural_parameters": n_structural,
                "singular": True,
            },
        )

    structural_cov = joint_cov[:n_structural, :n_structural]
    joint_se = jnp.sqrt(jnp.maximum(jnp.diag(joint_cov), 0.0))
    se = joint_se[:n_structural]

    return StandardErrorResult(
        standard_errors=se,
        variance_covariance=structural_cov,
        method="full_likelihood_bhhh",
        details={
            "n_observations": scores.shape[0],
            "n_joint_parameters": n_joint,
            "n_structural_parameters": n_structural,
            "joint_standard_errors": joint_se,
            "joint_variance_covariance": joint_cov,
        },
    )


def compute_numerical_hessian(
    parameters: jnp.ndarray,
    log_likelihood_fn: Callable,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Compute Hessian numerically via finite differences.

    Uses central differences for better accuracy:
        H[i,j] = (f(x+e_i+e_j) - f(x+e_i-e_j) - f(x-e_i+e_j) + f(x-e_i-e_j)) / (4*eps^2)

    For most cases, prefer compute_analytical_hessian which uses jax.hessian.

    Args:
        parameters: Point at which to compute Hessian
        log_likelihood_fn: Function mapping parameters to log-likelihood
        eps: Step size for finite differences

    Returns:
        Hessian matrix of shape (n_params, n_params)
    """
    n = len(parameters)
    params_np = np.asarray(parameters)
    hessian = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            e_i = np.zeros(n)
            e_j = np.zeros(n)
            e_i[i] = eps
            e_j[j] = eps

            f_pp = float(log_likelihood_fn(jnp.array(params_np + e_i + e_j)))
            f_pm = float(log_likelihood_fn(jnp.array(params_np + e_i - e_j)))
            f_mp = float(log_likelihood_fn(jnp.array(params_np - e_i + e_j)))
            f_mm = float(log_likelihood_fn(jnp.array(params_np - e_i - e_j)))

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
            hessian[j, i] = hessian[i, j]

    return jnp.array(hessian)


def compute_analytical_hessian(
    parameters: jnp.ndarray,
    log_likelihood_fn: Callable,
) -> jnp.ndarray:
    """Compute Hessian analytically via jax.hessian.

    This uses forward-over-reverse mode autodiff (jacfwd of grad),
    which is efficient for moderate parameter counts.

    Args:
        parameters: Point at which to compute Hessian
        log_likelihood_fn: Differentiable function mapping parameters to scalar

    Returns:
        Hessian matrix of shape (n_params, n_params)
    """
    return jax.hessian(log_likelihood_fn)(parameters)


def compute_gradient_contributions(
    parameters: jnp.ndarray,
    panel: Panel,
    log_prob_fn: Callable,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Compute per-observation gradient contributions numerically.

    For analytical computation, use compute_analytical_gradient_contributions
    which leverages jax.vmap(jax.grad(...)).

    Args:
        parameters: Current parameter values
        panel: Panel data
        log_prob_fn: Function(params, state, action) -> log probability
        eps: Step size for finite differences

    Returns:
        Gradient contributions of shape (n_observations, n_params)
    """
    n_obs = panel.num_observations
    n_params = len(parameters)
    params_np = np.asarray(parameters)

    gradients = np.zeros((n_obs, n_params))

    obs_idx = 0
    for traj in panel.trajectories:
        for t in range(len(traj)):
            state = int(traj.states[t])
            action = int(traj.actions[t])

            for k in range(n_params):
                e_k = np.zeros(n_params)
                e_k[k] = eps

                log_p_plus = float(log_prob_fn(jnp.array(params_np + e_k), state, action))
                log_p_minus = float(log_prob_fn(jnp.array(params_np - e_k), state, action))

                gradients[obs_idx, k] = (log_p_plus - log_p_minus) / (2 * eps)

            obs_idx += 1

    return jnp.array(gradients)


def compute_analytical_gradient_contributions(
    parameters: jnp.ndarray,
    obs_states: jnp.ndarray,
    obs_actions: jnp.ndarray,
    log_prob_fn: Callable,
) -> jnp.ndarray:
    """Compute per-observation gradient contributions analytically via jax.vmap.

    This is the JAX-native way to compute per-observation scores for
    sandwich standard errors and BHHH optimization.

    Args:
        parameters: Current parameter values, shape (K,)
        obs_states: Observed states, shape (N,)
        obs_actions: Observed actions, shape (N,)
        log_prob_fn: Function(params, state, action) -> scalar log probability.
                    Must be differentiable w.r.t. params.

    Returns:
        Gradient contributions of shape (N, K)
    """
    per_obs_grad = jax.vmap(
        jax.grad(log_prob_fn, argnums=0),
        in_axes=(None, 0, 0),
    )
    return per_obs_grad(parameters, obs_states, obs_actions)
