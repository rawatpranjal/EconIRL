"""JAX configuration for econirl.

This module must be imported before any other econirl module to ensure
float64 precision is enabled. JAX defaults to float32, but structural
estimation requires float64 for inner loop tolerances of 1e-12 and
accurate Hessian computation for standard errors.
"""

import jax

jax.config.update("jax_enable_x64", True)
