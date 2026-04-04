"""Inference metrics for parameter recovery evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class InferenceMetrics:
    """Metrics for evaluating parameter recovery (inference).

    Attributes:
        mse: Mean squared error between theta-hat and theta-star
        rmse: Root mean squared error
        mae: Mean absolute error
        bias: Per-parameter bias (theta-hat - theta-star)
        correlation: Pearson correlation coefficient
        cosine_similarity: Cosine similarity (direction match)
        relative_error: Per-parameter |theta-hat - theta-star| / |theta-star|
        coverage_90: Fraction of parameters where 90% CI contains theta-star
        coverage_95: Fraction of parameters where 95% CI contains theta-star
    """

    mse: float
    rmse: float
    mae: float
    bias: jnp.ndarray
    correlation: float
    cosine_similarity: float
    relative_error: jnp.ndarray
    coverage_90: float | None = None
    coverage_95: float | None = None


def inference_metrics(
    theta_true: jnp.ndarray,
    theta_hat: jnp.ndarray,
    standard_errors: jnp.ndarray | None = None,
    mask: list[bool] | None = None,
    normalize: bool = False,
) -> InferenceMetrics:
    """Compute parameter recovery metrics.

    Args:
        theta_true: Ground truth parameters
        theta_hat: Estimated parameters
        standard_errors: Standard errors (for coverage computation)
        mask: Which parameters to include (True = include)
        normalize: Normalize both vectors to unit norm before comparison

    Returns:
        InferenceMetrics with all computed values
    """
    theta_true = jnp.asarray(theta_true, dtype=jnp.float32)
    theta_hat = jnp.asarray(theta_hat, dtype=jnp.float32)

    if mask is not None:
        mask_tensor = jnp.array(mask, dtype=jnp.bool_)
        theta_true = theta_true[mask_tensor]
        theta_hat = theta_hat[mask_tensor]
        if standard_errors is not None:
            standard_errors = standard_errors[mask_tensor]

    if normalize:
        theta_true = theta_true / jnp.linalg.norm(theta_true)
        theta_hat = theta_hat / jnp.linalg.norm(theta_hat)

    bias = theta_hat - theta_true

    mse = float((bias**2).mean())
    rmse = mse**0.5
    mae = float(jnp.abs(bias).mean())

    relative_error = jnp.where(
        jnp.abs(theta_true) > 1e-10,
        jnp.abs(theta_hat - theta_true) / jnp.abs(theta_true),
        jnp.zeros_like(theta_true),
    )

    if len(theta_true) > 1:
        mean_true = theta_true.mean()
        mean_hat = theta_hat.mean()
        cov = ((theta_true - mean_true) * (theta_hat - mean_hat)).mean()
        std_true = theta_true.std()
        std_hat = theta_hat.std()
        if std_true > 1e-10 and std_hat > 1e-10:
            correlation = float(cov / (std_true * std_hat))
        else:
            correlation = 1.0 if jnp.allclose(theta_true, theta_hat) else 0.0
    else:
        correlation = 1.0 if jnp.allclose(theta_true, theta_hat) else 0.0

    norm_true = jnp.linalg.norm(theta_true)
    norm_hat = jnp.linalg.norm(theta_hat)
    if norm_true > 1e-10 and norm_hat > 1e-10:
        cosine_similarity = float(
            jnp.dot(theta_true, theta_hat) / (norm_true * norm_hat)
        )
    else:
        cosine_similarity = 1.0 if jnp.allclose(theta_true, theta_hat) else 0.0

    coverage_90 = None
    coverage_95 = None
    if standard_errors is not None:
        standard_errors = jnp.asarray(standard_errors, dtype=jnp.float32)
        z_90 = 1.645
        z_95 = 1.96

        lower_90 = theta_hat - z_90 * standard_errors
        upper_90 = theta_hat + z_90 * standard_errors
        covered_90 = (theta_true >= lower_90) & (theta_true <= upper_90)
        coverage_90 = float(covered_90.astype(jnp.float32).mean())

        lower_95 = theta_hat - z_95 * standard_errors
        upper_95 = theta_hat + z_95 * standard_errors
        covered_95 = (theta_true >= lower_95) & (theta_true <= upper_95)
        coverage_95 = float(covered_95.astype(jnp.float32).mean())

    return InferenceMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        bias=bias,
        correlation=correlation,
        cosine_similarity=cosine_similarity,
        relative_error=relative_error,
        coverage_90=coverage_90,
        coverage_95=coverage_95,
    )
