## Estimator: SEES (Sieve Estimation of Economic Structural Models)
## Paper(s): Luo & Sang 2024 "Sieve Estimation of Dynamic Discrete Choice Models" (working paper). Doclinged at `papers/foundational/luo_sang_2024_sees.md`.
## Code: `src/econirl/estimation/sees.py`

### Loss / objective

- Paper formula (Luo-Sang 2024, eq. 3.4): replace the value function with a sieve approximation `V_K(s) = sum_{j=1}^K beta_j * b_j(s)` where `b_j` is a basis function (B-spline, polynomial, or Fourier). Joint penalized maximum-likelihood:

  ```
  L(theta, beta) = sum_i sum_t log Pr(a_it | s_it; theta, V_K(beta)) - omega_n * R(beta)
  ```

  where the penalty `R(beta)` is the squared Bellman residual `||V_K - T V_K||^2` and `omega_n` is a weight that diverges as `n -> infinity` (Luo-Sang Theorem 4.1 gives the rate).

- Code implementation: `sees.py:_optimize` constructs the basis matrix `B` (shape `(num_states, K)`), parameterizes V via `V = B @ beta`, and runs joint optimization over `(theta, beta)` with the penalized loss. The basis family is selected via the `basis_type` constructor argument; the dimension `K` via `basis_dim`.

- Match: **yes**, follows the Luo-Sang penalized-MLE formulation.

### Gradient

- Paper formula: standard penalized-MLE gradient.

- Code implementation: JAX autodiff. The penalty term differentiates cleanly because the Bellman residual is a quadratic form in V.

- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: no inner Bellman fixed-point. The Bellman penalty replaces the fixed-point with a soft constraint, and the joint optimization handles both `theta` and `beta` simultaneously.

- Code algorithm: matches.

- Match: **yes**.

### Identification assumptions

- Paper conditions: the basis must be rich enough to approximate the true V (universal-approximation in the basis family); the penalty weight `omega_n` must grow slowly enough to allow the sieve to fit and fast enough to enforce the Bellman constraint asymptotically (Luo-Sang Theorem 4.1's two-sided rate condition).

- Code enforcement: `omega_n` is exposed as `bellman_penalty_weight` (default `1.0`); the user is expected to schedule it. The wrapper does not auto-schedule. **The default `1.0` is too low for asymptotic Bellman enforcement**; the paper's recommended schedule is `omega_n = n^{1/2}` or similar.

- Match: **structure yes**; **default penalty weight too low**.

### Hyperparameter defaults vs paper defaults

- `basis_type`: "bspline" (paper used B-splines for their numerical experiments).
- `basis_dim`: 10 (paper uses K = O(n^{1/5}) for cubic splines; for n=10000, K=6 to 10).
- `bellman_penalty_weight`: 1.0.
- `learning_rate`: 1e-3.

Match: **basis yes**; **penalty weight too low** for the asymptotic regime.

### Findings / fixes applied

- **Default penalty weight could be raised**, but doing so without the paper's full schedule is a half-fix. The Tier 4 ss-spine cell uses default `omega_n=1.0` for now; if SEES under-performs on the cell, the failure_mode will be `policy_drift` and the follow-up is to implement the paper's adaptive schedule. **Not applied** in this audit pass.

- VALIDATION_LOG.md status: **Pending** (Tier 4 ss-spine).
