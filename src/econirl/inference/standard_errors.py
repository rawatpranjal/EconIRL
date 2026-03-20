"""Standard error computation methods.

This module provides multiple methods for computing standard errors
of estimated parameters in dynamic discrete choice models:

1. Asymptotic (Hessian-based): Inverse of negative Hessian
2. Robust (Sandwich): Accounts for potential misspecification
3. Bootstrap: Nonparametric resampling
4. Clustered: Cluster-robust SEs for panel data

The choice of SE method can significantly affect inference,
especially with panel data where observations within individuals
may be correlated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch

from econirl.core.types import Panel


SEMethod = Literal["asymptotic", "robust", "bootstrap", "clustered"]


@dataclass
class StandardErrorResult:
    """Result of standard error computation.

    Attributes:
        standard_errors: SE for each parameter
        variance_covariance: Full variance-covariance matrix
        method: Method used to compute SEs
        details: Additional method-specific information
    """

    standard_errors: torch.Tensor
    variance_covariance: torch.Tensor
    method: str
    details: dict


def compute_standard_errors(
    parameters: torch.Tensor,
    hessian: torch.Tensor | None = None,
    gradient_contributions: torch.Tensor | None = None,
    panel: Panel | None = None,
    log_likelihood_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    method: SEMethod = "asymptotic",
    n_bootstrap: int = 200,
    seed: int | None = None,
    estimate_fn: Callable[[Panel], torch.Tensor] | None = None,
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
    else:
        raise ValueError(f"Unknown SE method: {method}")


def _asymptotic_se(
    parameters: torch.Tensor,
    hessian: torch.Tensor | None,
) -> StandardErrorResult:
    """Compute asymptotic standard errors from Hessian.

    Uses the inverse of the negative Hessian as the variance-covariance
    matrix, which is the standard MLE result under correct specification.

    Var(θ̂) = [-H(θ̂)]^{-1}
    """
    if hessian is None:
        raise ValueError("Hessian required for asymptotic SEs")

    n_params = hessian.shape[0]

    # Variance-covariance is inverse of negative Hessian
    var_cov = None
    for ridge_factor in [0, 1e-8, 1e-6, 1e-4, 1e-2]:
        try:
            ridge = ridge_factor * torch.eye(n_params, device=hessian.device, dtype=hessian.dtype)
            var_cov = torch.linalg.inv(-hessian + ridge)
            # Check if result is reasonable
            diag = torch.diag(var_cov)
            if (diag > 0).all() and torch.isfinite(var_cov).all():
                break
            var_cov = None
        except RuntimeError:
            continue

    if var_cov is None:
        # Return NaN if we can't invert
        se = torch.full((n_params,), float("nan"), dtype=hessian.dtype)
        var_cov = torch.full((n_params, n_params), float("nan"), dtype=hessian.dtype)
        return StandardErrorResult(
            standard_errors=se,
            variance_covariance=var_cov,
            method="asymptotic",
            details={"hessian_used": True, "singular": True},
        )

    # Ensure positive diagonal (numerical issues can cause small negatives)
    diag = torch.diag(var_cov)
    se = torch.sqrt(torch.clamp(diag, min=0))

    return StandardErrorResult(
        standard_errors=se,
        variance_covariance=var_cov,
        method="asymptotic",
        details={"hessian_used": True},
    )


def _robust_se(
    parameters: torch.Tensor,
    hessian: torch.Tensor | None,
    gradient_contributions: torch.Tensor | None,
) -> StandardErrorResult:
    """Compute robust (sandwich) standard errors.

    The sandwich estimator is:
        Var(θ̂) = H^{-1} @ B @ H^{-1}

    where:
        H = Hessian (negative expected information)
        B = Σ_i g_i g_i' (outer product of gradients)

    This is robust to misspecification of the likelihood.
    """
    if hessian is None:
        raise ValueError("Hessian required for robust SEs")
    if gradient_contributions is None:
        raise ValueError("Gradient contributions required for robust SEs")

    # H^{-1} with progressive ridge regularization
    n_params = hessian.shape[0]
    H_inv = None
    for ridge_factor in [0, 1e-8, 1e-6, 1e-4, 1e-2]:
        try:
            ridge = ridge_factor * torch.eye(n_params, device=hessian.device, dtype=hessian.dtype)
            H_inv = torch.linalg.inv(-hessian + ridge)
            if torch.isfinite(H_inv).all():
                break
            H_inv = None
        except RuntimeError:
            continue

    if H_inv is None:
        se = torch.full((n_params,), float("nan"), dtype=hessian.dtype)
        var_cov = torch.full((n_params, n_params), float("nan"), dtype=hessian.dtype)
        return StandardErrorResult(
            standard_errors=se,
            variance_covariance=var_cov,
            method="robust",
            details={"n_observations": gradient_contributions.shape[0], "singular": True},
        )

    # B = Σ_i g_i g_i' (outer product of gradients)
    # gradient_contributions shape: (n_obs, n_params)
    B = gradient_contributions.T @ gradient_contributions

    # Sandwich: H^{-1} B H^{-1}
    var_cov = H_inv @ B @ H_inv

    diag = torch.diag(var_cov)
    se = torch.sqrt(torch.clamp(diag, min=0))

    return StandardErrorResult(
        standard_errors=se,
        variance_covariance=var_cov,
        method="robust",
        details={"n_observations": gradient_contributions.shape[0]},
    )


def _bootstrap_se(
    parameters: torch.Tensor,
    panel: Panel | None,
    log_likelihood_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    n_bootstrap: int,
    seed: int | None,
    estimate_fn: Callable[[Panel], torch.Tensor] | None = None,
) -> StandardErrorResult:
    """Compute bootstrap standard errors.

    Resamples individuals (with replacement) and re-estimates the model
    on each bootstrap sample. The SE is the standard deviation of the
    bootstrap distribution.

    This is nonparametric and does not require distributional assumptions.

    Args:
        parameters: Point estimates (used as fallback and for shape)
        panel: Panel data to resample from
        log_likelihood_fn: Not used (kept for API compatibility)
        n_bootstrap: Number of bootstrap replications
        seed: Random seed for reproducibility
        estimate_fn: Function that takes a Panel and returns parameter estimates.
                    If None, bootstrap SEs cannot be computed properly.
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

    bootstrap_estimates = torch.zeros((n_bootstrap, n_params))
    successful_bootstraps = 0

    for b in range(n_bootstrap):
        # Resample individuals with replacement
        indices = rng.choice(n_individuals, size=n_individuals, replace=True)
        resampled_trajectories = [panel.trajectories[i] for i in indices]
        bootstrap_panel = Panel(trajectories=resampled_trajectories)

        try:
            # Re-estimate on bootstrap sample
            bootstrap_params = estimate_fn(bootstrap_panel)
            bootstrap_estimates[b] = bootstrap_params
            successful_bootstraps += 1
        except Exception:
            # If estimation fails, use original parameters (conservative)
            bootstrap_estimates[b] = parameters

    # Compute variance from bootstrap distribution
    if successful_bootstraps > 1:
        var_cov = torch.cov(bootstrap_estimates.T)
        se = torch.std(bootstrap_estimates, dim=0)
    else:
        # Not enough successful bootstraps
        var_cov = torch.full((n_params, n_params), float("nan"), dtype=parameters.dtype)
        se = torch.full((n_params,), float("nan"), dtype=parameters.dtype)

    return StandardErrorResult(
        standard_errors=se,
        variance_covariance=var_cov,
        method="bootstrap",
        details={
            "n_bootstrap": n_bootstrap,
            "successful_bootstraps": successful_bootstraps,
            "seed": seed,
        },
    )


def _clustered_se(
    parameters: torch.Tensor,
    hessian: torch.Tensor | None,
    gradient_contributions: torch.Tensor | None,
    panel: Panel | None,
) -> StandardErrorResult:
    """Compute cluster-robust standard errors.

    Accounts for correlation of observations within individuals
    (clusters). The variance estimator is:

        Var(θ̂) = H^{-1} @ B_cluster @ H^{-1}

    where:
        B_cluster = Σ_c (Σ_{i∈c} g_i)(Σ_{i∈c} g_i)'

    This sums gradients within clusters before taking outer products.
    """
    if hessian is None:
        raise ValueError("Hessian required for clustered SEs")
    if gradient_contributions is None:
        raise ValueError("Gradient contributions required for clustered SEs")
    if panel is None:
        raise ValueError("Panel required for clustered SEs")

    # H^{-1} with progressive ridge regularization
    n_params = hessian.shape[0]
    H_inv = None
    for ridge_factor in [0, 1e-8, 1e-6, 1e-4, 1e-2]:
        try:
            ridge = ridge_factor * torch.eye(n_params, device=hessian.device, dtype=hessian.dtype)
            H_inv = torch.linalg.inv(-hessian + ridge)
            if torch.isfinite(H_inv).all():
                break
            H_inv = None
        except RuntimeError:
            continue

    if H_inv is None:
        se = torch.full((n_params,), float("nan"), dtype=hessian.dtype)
        var_cov = torch.full((n_params, n_params), float("nan"), dtype=hessian.dtype)
        return StandardErrorResult(
            standard_errors=se,
            variance_covariance=var_cov,
            method="clustered",
            details={"singular": True},
        )

    n_params = len(parameters)
    n_clusters = panel.num_individuals

    # Sum gradients within each cluster
    cluster_gradients = torch.zeros((n_clusters, n_params))
    obs_idx = 0
    for c, traj in enumerate(panel.trajectories):
        n_obs_c = len(traj)
        cluster_gradients[c] = gradient_contributions[obs_idx : obs_idx + n_obs_c].sum(dim=0)
        obs_idx += n_obs_c

    # B_cluster = Σ_c g_c g_c'
    B_cluster = cluster_gradients.T @ cluster_gradients

    # Small sample correction: multiply by G/(G-1) * N/(N-K)
    # where G = number of clusters, N = total obs, K = number of params
    G = n_clusters
    N = panel.num_observations
    K = n_params
    correction = (G / (G - 1)) * (N - 1) / (N - K) if G > 1 and N > K else 1.0

    # Sandwich with correction
    var_cov = correction * (H_inv @ B_cluster @ H_inv)

    diag = torch.diag(var_cov)
    se = torch.sqrt(torch.clamp(diag, min=0))

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


def compute_numerical_hessian(
    parameters: torch.Tensor,
    log_likelihood_fn: Callable[[torch.Tensor], torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute Hessian numerically via finite differences.

    Uses central differences for better accuracy:
        H[i,j] ≈ (f(x+e_i+e_j) - f(x+e_i-e_j) - f(x-e_i+e_j) + f(x-e_i-e_j)) / (4ε²)

    Args:
        parameters: Point at which to compute Hessian
        log_likelihood_fn: Function mapping parameters to log-likelihood
        eps: Step size for finite differences

    Returns:
        Hessian matrix of shape (n_params, n_params)
    """
    n = len(parameters)
    hessian = torch.zeros((n, n), dtype=parameters.dtype)

    for i in range(n):
        for j in range(i, n):
            # Create perturbation vectors
            e_i = torch.zeros_like(parameters)
            e_j = torch.zeros_like(parameters)
            e_i[i] = eps
            e_j[j] = eps

            # Four-point formula for mixed partial
            f_pp = log_likelihood_fn(parameters + e_i + e_j)
            f_pm = log_likelihood_fn(parameters + e_i - e_j)
            f_mp = log_likelihood_fn(parameters - e_i + e_j)
            f_mm = log_likelihood_fn(parameters - e_i - e_j)

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
            hessian[j, i] = hessian[i, j]  # Symmetric

    return hessian


def compute_gradient_contributions(
    parameters: torch.Tensor,
    panel: Panel,
    log_prob_fn: Callable[[torch.Tensor, int, int], torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute per-observation gradient contributions numerically.

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

    gradients = torch.zeros((n_obs, n_params))

    obs_idx = 0
    for traj in panel.trajectories:
        for t in range(len(traj)):
            state = traj.states[t].item()
            action = traj.actions[t].item()

            # Numerical gradient for this observation
            for k in range(n_params):
                e_k = torch.zeros_like(parameters)
                e_k[k] = eps

                log_p_plus = log_prob_fn(parameters + e_k, state, action)
                log_p_minus = log_prob_fn(parameters - e_k, state, action)

                gradients[obs_idx, k] = (log_p_plus - log_p_minus) / (2 * eps)

            obs_idx += 1

    return gradients
