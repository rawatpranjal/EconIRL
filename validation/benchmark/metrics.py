"""Segmented statistical metrics for the benchmark suite.

Two metric families, kept strictly separate (Skalse et al 2023: reward is only
partially identifiable, so parameter bias is not comparable across estimator
families):

- Behavioral metrics (policy total-variation distance, value RMSE) are computed
  for EVERY estimator, because the recovered policy/value are comparable across
  families.
- Parameter metrics (bias, empirical SE, RMSE, 95% coverage) are computed ONLY
  for the structural family that recovers a finite theta in the same gauge as
  the DGP and returns real standard errors.

Every aggregate carries a Monte Carlo standard error so simulation uncertainty
is visible (Morris, White, Crowther 2019).
"""

from __future__ import annotations

import numpy as np

# Normal-approx z for 95% CIs, matching run_monte_carlo's convention.
_Z95 = 1.959963984540054


def policy_tv(pi_est: np.ndarray, pi_oracle: np.ndarray) -> float:
    """Mean total-variation distance between two policies over states.

    TV(s) = 0.5 * sum_a |pi_est[s,a] - pi_oracle[s,a]|, averaged over states.
    Bounded in [0, 1]; 0 means the policies agree everywhere.
    """
    pi_est = np.asarray(pi_est, dtype=np.float64)
    pi_oracle = np.asarray(pi_oracle, dtype=np.float64)
    tv = 0.5 * np.abs(pi_est - pi_oracle).sum(axis=1)
    return float(tv.mean())


def value_rmse(v_est: np.ndarray | None, v_oracle: np.ndarray) -> float | None:
    """RMSE between estimated and oracle value functions, or None if unavailable."""
    if v_est is None:
        return None
    v_est = np.asarray(v_est, dtype=np.float64).reshape(-1)
    v_oracle = np.asarray(v_oracle, dtype=np.float64).reshape(-1)
    if v_est.shape != v_oracle.shape:
        return None
    return float(np.sqrt(np.mean((v_est - v_oracle) ** 2)))


def feature_diagnostics(feature_matrix: np.ndarray) -> dict[str, float | int]:
    """Rank and condition number of the stacked design matrix (S*A, K).

    Feeds the failure-mode map: a rank-deficient or ill-conditioned design is
    the structural reason some cells break parameter recovery.
    """
    phi = np.asarray(feature_matrix, dtype=np.float64)
    S, A, K = phi.shape
    design = phi.reshape(S * A, K)
    rank = int(np.linalg.matrix_rank(design))
    svals = np.linalg.svd(design, compute_uv=False)
    smin = float(svals.min())
    cond = float(svals.max() / smin) if smin > 0 else float("inf")
    return {"num_features": K, "feature_rank": rank, "condition_number": cond}


def _mc_se_mean(values: np.ndarray) -> float:
    """Monte Carlo SE of a mean: sample sd / sqrt(n)."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    if n <= 1:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(n))


def _mc_se_proportion(p: float, n: int) -> float:
    """Monte Carlo SE of a coverage proportion: sqrt(p(1-p)/n)."""
    if n <= 0:
        return float("nan")
    return float(np.sqrt(p * (1.0 - p) / n))


def behavioral_summary(
    policy_tvs: list[float],
    value_rmses: list[float | None],
) -> dict:
    """Aggregate behavioral metrics across replications with Monte Carlo SE."""
    tv = np.asarray(policy_tvs, dtype=np.float64)
    out: dict = {
        "n": int(tv.shape[0]),
        "policy_tv_mean": float(tv.mean()) if tv.size else None,
        "policy_tv_mc_se": _mc_se_mean(tv) if tv.size else None,
    }
    vr = [v for v in value_rmses if v is not None]
    if vr:
        vr_arr = np.asarray(vr, dtype=np.float64)
        out["value_rmse_mean"] = float(vr_arr.mean())
        out["value_rmse_mc_se"] = _mc_se_mean(vr_arr)
    else:
        out["value_rmse_mean"] = None
        out["value_rmse_mc_se"] = None
    return out


def parameter_summary(
    estimates: np.ndarray,
    standard_errors: np.ndarray,
    true_theta: np.ndarray,
    param_names: list[str],
) -> dict:
    """Aggregate parameter recovery (structural family only).

    Args:
        estimates: ``(n_reps, K)`` parameter estimates.
        standard_errors: ``(n_reps, K)`` standard errors (may be NaN).
        true_theta: ``(K,)`` ground-truth parameters.
        param_names: K names.

    Returns a dict with bias, empirical SE, RMSE, and (only when finite SEs are
    present) 95% coverage, each with Monte Carlo SE. ``se_available`` records
    whether coverage could be computed honestly.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    standard_errors = np.asarray(standard_errors, dtype=np.float64)
    true_theta = np.asarray(true_theta, dtype=np.float64).reshape(-1)
    n_reps = estimates.shape[0]

    mean_est = estimates.mean(axis=0)
    bias = mean_est - true_theta
    bias_mc_se = estimates.std(axis=0, ddof=1) / np.sqrt(n_reps) if n_reps > 1 else np.full_like(bias, np.nan)
    empirical_se = estimates.std(axis=0, ddof=1) if n_reps > 1 else np.full_like(bias, np.nan)
    rmse = np.sqrt(((estimates - true_theta) ** 2).mean(axis=0))

    out: dict = {
        "names": list(param_names),
        "true": true_theta.tolist(),
        "mean_estimate": mean_est.tolist(),
        "bias": bias.tolist(),
        "bias_mc_se": bias_mc_se.tolist(),
        "empirical_se": empirical_se.tolist(),
        "rmse": rmse.tolist(),
    }

    # Fraction of replications where every parameter got a finite SE. A low
    # rate means the estimator routinely fails to deliver usable inference,
    # which a bland "n/a" coverage would otherwise hide.
    out["se_available_rate"] = float(np.isfinite(standard_errors).all(axis=1).mean())

    # Coverage only where SEs are finite for every rep of that parameter.
    finite_se = np.isfinite(standard_errors).all(axis=0)
    if finite_se.any():
        lower = estimates - _Z95 * standard_errors
        upper = estimates + _Z95 * standard_errors
        covered = (lower <= true_theta) & (true_theta <= upper)
        coverage = covered.mean(axis=0)
        cov_list: list[float | None] = []
        cov_se_list: list[float | None] = []
        for k in range(true_theta.shape[0]):
            if finite_se[k]:
                cov_list.append(float(coverage[k]))
                cov_se_list.append(_mc_se_proportion(float(coverage[k]), n_reps))
            else:
                cov_list.append(None)
                cov_se_list.append(None)
        out["coverage_95"] = cov_list
        out["coverage_95_mc_se"] = cov_se_list
        out["se_available"] = True
    else:
        out["coverage_95"] = [None] * true_theta.shape[0]
        out["coverage_95_mc_se"] = [None] * true_theta.shape[0]
        out["se_available"] = False

    return out
