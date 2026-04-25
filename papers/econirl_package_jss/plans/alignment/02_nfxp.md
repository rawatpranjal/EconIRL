## Estimator: NFXP (Nested Fixed Point)
## Paper(s): Rust 1987 (canonical formulation); Iskhakov, Rust, Schjerning 2016 (SA→NK polyalgorithm and the MPEC-vs-NFXP comparison). Both at `papers/foundational/1987_rust_optimal_replacement.pdf` and `papers/foundational/iskhakov_rust_schjerning_2016_mpec_comment.md`.
## Code: `src/econirl/estimation/nfxp.py`

### Loss / objective

- Paper formula (Rust 1987, eq. 4.1–4.6): the conditional likelihood of observed actions

  ```
  L(theta) = prod_i prod_t Pr(a_it | s_it; theta)
            = prod_i prod_t exp(Q(s_it, a_it; theta) / sigma)
                            / sum_a' exp(Q(s_it, a'; theta) / sigma)
  ```

  where `Q(s, a; theta) = u(s, a; theta) + beta * sum_s' P(s'|s, a) V(s'; theta)` and `V` is the fixed point of the soft Bellman operator.

- Code implementation: `_optimize` builds a pure-JAX objective in `nfxp.py:464-485`. The flow utility comes from `compute_utility_matrix` (linear in theta), the Bellman fixed point comes from the inner solver (default `hybrid_iteration` per `nfxp.py:185-191`), and the conditional log-likelihood is summed over (s, a) observations.

- Match: **yes**. The implementation is the textbook NFXP under the soft Bellman operator. Two implementation choices that diverge from a strict reading of Rust 1987:
  1. **Inner solver**: Rust 1987 used pure value iteration; the package defaults to `hybrid_iteration` (contraction → Newton-Kantorovich), which is the Iskhakov-Rust-Schjerning recommendation for high beta. This is the SA→NK polyalgorithm and is documented in the module docstring.
  2. **Outer optimizer**: Rust 1987 used BHHH; the package uses L-BFGS-B (`minimize_lbfgsb` in `core.optimizer`). The two have the same first-order optimality conditions; BHHH only differs in the Hessian approximation. The package's default Hessian for inference uses the inverse outer-product-of-gradients per Rust 1987 anyway.

### Gradient

- Paper formula: implicit differentiation through the fixed point. For each parameter `theta_k`,

  ```
  d_theta_k V = (I - beta * P_pi)^{-1} * d_theta_k u_pi
  ```

  where `P_pi` is the policy-induced transition matrix. The likelihood gradient then follows by the chain rule.

- Code implementation: JAX autodiff through the fully-compiled `value_iteration_jaxnative` / `hybrid_iteration` solver. The fixed-point loop is wrapped in `jax.lax.while_loop` so `jax.grad` differentiates through it via implicit-function-theorem unrolling (Blondel et al. 2022). See `core/solvers.py` line 89-100.

- Match: **yes**, equivalent in exact arithmetic. The autodiff-through-fixed-point approach is mathematically identical to Rust's analytical gradient at the fixed point.

### Bellman / inner loop

- Paper algorithm: pure VI in Rust 1987; the SA→NK polyalgorithm in Iskhakov-Rust-Schjerning 2016.

- Code algorithm: `nfxp.py:185-191` selects between `value_iteration`, `policy_iteration`, and `hybrid_iteration` via the `inner_solver` constructor argument; default is `"hybrid"`.

- Match: **yes** for the SA→NK story; **superset** (also exposes `"value"` and `"policy"`) for completeness.

### Identification assumptions

- Paper conditions: full-rank feature matrix in the linear-in-theta utility specification; transitions are known (in Rust 1987 they are estimated separately by counting). Logit shock scale is fixed (typically `sigma = 1`).

- Code enforcement: the wrapper accepts a `LinearUtility` with parameter names; transitions are passed as a `(A, S, S)` tensor; `scale_parameter` defaults to `1.0` in `DDCProblem`. The package warns when the feature matrix has rank below `num_features` (per CLAUDE.md "Pre-Estimation Diagnostics"), but the warning is in the wrapper, not the estimator itself.

- Match: **yes**.

### Hyperparameter defaults vs paper defaults

- `inner_solver`: `"hybrid"` (Iskhakov-Rust-Schjerning recommendation). Rust 1987 used VI; either is acceptable.
- `inner_tol`: `1e-10` (paper does not specify; modern NFXP implementations use 1e-8 to 1e-12).
- `outer_tol`: `1e-6`.
- `max_iter` (outer L-BFGS): `200`.
- `compute_hessian`: `True` (for inference).

Match: **yes**. Defaults are reasonable for small to medium panels.

### Findings / fixes applied

- **No code fixes required.** NFXP-on-rust-small is the validated reference (VALIDATION_LOG.md "NFXP on rust-small": **Pass**, theta_c relative error 0.29%, RC 2.4%, SE coverage 95%).

- One pre-existing inconsistency in the paper draft is noted but not a code issue: paper Section 4 cites two different log-likelihoods for NFXP-on-rust-small (-1900.33 vs -4263.20). This is a paper-level artifact-management problem, not a code-vs-paper alignment issue. Tracked in VALIDATION_LOG.md "Cross-cutting finding".

- VALIDATION_LOG.md status: **Pass**.
