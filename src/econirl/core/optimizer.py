"""JAX-native optimization wrappers replacing scipy.optimize.minimize.

All structural estimators use bounded L-BFGS for the outer theta optimization.
This module provides a unified interface backed by jaxopt.LBFGSB (bounded) and
jaxopt.LBFGS (unbounded), replacing scipy.optimize.minimize throughout the
codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import jaxopt
from tqdm import tqdm


@dataclass
class OptimizeResult:
    """Result of JAX-native optimization, matching scipy interface."""
    x: jnp.ndarray
    fun: float
    success: bool
    nit: int
    nfev: int
    message: str


def _run_stepwise(solver, x0, maxiter, tol, desc, param_names, bounds_kw):
    """Run jaxopt solver step by step with rich tqdm diagnostics."""
    state = solver.init_state(x0, **bounds_kw) if bounds_kw else solver.init_state(x0)
    params = x0
    prev_val = float("inf")

    pbar = tqdm(range(maxiter), desc=desc, leave=True)
    for i in pbar:
        if bounds_kw:
            params, state = solver.update(params, state, **bounds_kw)
        else:
            params, state = solver.update(params, state)

        fval = float(state.value)
        gnorm = float(jnp.linalg.norm(state.grad))
        rel_change = abs(fval - prev_val) / max(abs(prev_val), 1e-12)

        postfix = {
            "obj": f"{fval:.4f}",
            "|g|": f"{gnorm:.1e}",
            "dobj": f"{rel_change:.1e}",
        }

        # Show first few parameter values if names are given
        if param_names is not None:
            p = params.ravel()
            for j, name in enumerate(param_names[:3]):
                if j < len(p):
                    postfix[name] = f"{float(p[j]):.5f}"

        pbar.set_postfix(postfix)
        prev_val = fval

        if gnorm < tol:
            pbar.set_postfix({**postfix, "status": "converged"})
            break
    pbar.close()
    return params, state, i + 1


def minimize_lbfgsb(
    fun: Callable,
    x0: jnp.ndarray,
    bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    maxiter: int = 500,
    tol: float = 1e-8,
    verbose: bool = False,
    desc: str = "L-BFGS-B",
    value_and_grad: bool = False,
    param_names: Sequence[str] | None = None,
) -> OptimizeResult:
    """Bounded L-BFGS optimization using jaxopt.

    Parameters
    ----------
    fun : callable
        Objective function. If value_and_grad is False, must be f(x) -> scalar
        and must be JAX-differentiable. If value_and_grad is True, must be
        f(x) -> (scalar, grad_array).
    x0 : jnp.ndarray
        Initial parameter vector.
    bounds : tuple of (lower, upper) arrays, optional
        Box constraints. If None, uses unbounded LBFGS.
    maxiter : int
        Maximum iterations.
    tol : float
        Gradient tolerance for convergence.
    verbose : bool
        If True, show tqdm progress bar with rich diagnostics.
    desc : str
        Description for the progress bar.
    value_and_grad : bool
        If True, fun returns (value, gradient) instead of just the value.
    param_names : list of str, optional
        Names for the first few parameters. Shown in tqdm postfix.

    Returns
    -------
    OptimizeResult
        Result with .x, .fun, .success, .nit, .nfev, .message fields.
    """
    x0 = jnp.asarray(x0, dtype=jnp.float64)

    if bounds is not None:
        lower = jnp.asarray(bounds[0], dtype=jnp.float64)
        upper = jnp.asarray(bounds[1], dtype=jnp.float64)
        solver = jaxopt.LBFGSB(
            fun=fun, value_and_grad=value_and_grad,
            maxiter=maxiter, tol=tol, jit=False, unroll=True,
        )
        bounds_kw = {"bounds": (lower, upper)}
    else:
        solver = jaxopt.LBFGS(
            fun=fun, value_and_grad=value_and_grad,
            maxiter=maxiter, tol=tol, jit=False, unroll=True,
        )
        bounds_kw = {}

    if verbose:
        params, state, nit = _run_stepwise(
            solver, x0, maxiter, tol, desc, param_names, bounds_kw,
        )
    else:
        if bounds_kw:
            result = solver.run(x0, **bounds_kw)
        else:
            result = solver.run(x0)
        params = result.params
        state = result.state
        nit = maxiter

    if value_and_grad:
        final_val, final_grad = fun(params)
        final_val = float(final_val)
        grad_norm = float(jnp.linalg.norm(jnp.asarray(final_grad)))
    else:
        final_val = float(fun(params))
        grad_norm = float(jnp.linalg.norm(jax.grad(fun)(params)))

    converged = grad_norm < tol * 10

    return OptimizeResult(
        x=params,
        fun=final_val,
        success=converged,
        nit=nit,
        nfev=nit,
        message="Converged" if converged else "Maximum iterations reached",
    )
