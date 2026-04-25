## Estimator: TD-CCP (Temporal-Difference CCP)
## Paper(s): Adusumilli & Eckardt 2025 "Temporal-Difference Estimation of Dynamic Discrete Choice Models" (working paper). Doclinged at `papers/foundational/adusumilli_eckardt_2025_td_ccp.md`.
## Code: `src/econirl/estimation/td_ccp.py`

### Loss / objective

- Paper formula (Adusumilli-Eckardt 2025, Section 3): replace the closed-form Hotz-Miller V solve with a temporal-difference Bellman residual, parameterized as either a linear semi-gradient or a neural approximate value iteration. The TD residual at observation `(s, a, s')` is

  ```
  delta(s, a, s'; theta) = u(s, a; theta) + gamma * V(s'; theta) - Q(s, a; theta)
  ```

  and the joint loss combines:

  - **Conditional log-likelihood** of observed actions (Hotz-Miller pseudo-likelihood under V).
  - **TD residual** `||delta||^2` over the panel.

  Critically, the paper avoids estimating the transition density `P(s' | s, a)` explicitly — the residual is computed at the observed `(s, a, s')` triples. This is the main advantage over plain CCP for high-dimensional state spaces.

- Code implementation: `td_ccp.py:_optimize` constructs the joint loss. The `linear_mode` constructor argument selects between the closed-form linear semi-gradient and the neural approximate VI variant. The cross-fitting variance estimator (Adusumilli-Eckardt Section 5) splits the panel by individual (not by row) for valid clustered SEs.

- Match: **yes** for the loss form. The cross-fitting splits by individual, not by row, per the project memory `tdccp_cross_fitting_and_se.md` (which records the recent fix).

### Gradient

- Paper formula: semi-gradient on the TD residual (do not differentiate through the bootstrap V on the next-state side, per the standard TD convention).

- Code implementation: JAX autodiff with `jax.lax.stop_gradient` on the next-state V term to enforce semi-gradient. The clustered sandwich SE construction is implemented post-fit per Adusumilli-Eckardt eq. 5.3.

- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: no inner fixed-point. TD residual replaces the fixed-point.

- Code algorithm: matches.

- Match: **yes**.

### Identification assumptions

- Paper conditions: linear features must be full-rank (semi-gradient identification); the cross-fitting variance estimator requires panel-level (not row-level) splits to avoid leakage; coverage of the asymptotic CI breaks down at very high beta (the project memory records this on a tabular 1D DGP at beta=0.95).

- Code enforcement: cross-fitting splits are by `individual_id` per the recent fix (commit referenced in VALIDATION_LOG.md). The high-beta caveat is not flagged in code; the Tier 4 ss-high-gamma cell will surface it as `parameter_drift` if it bites.

- Match: **yes** post-fix.

### Hyperparameter defaults vs paper defaults

- `linear_mode`: `True` (closed-form linear semi-gradient).
- `cross_fit_folds`: `5`.
- `learning_rate`: `1e-3` (Adam, neural mode only).
- `num_iterations`: `2000` (neural mode).

Match: **yes**.

### Findings / fixes applied

- **No code fixes required.** The cross-fitting individual-split fix is already in (committed before this audit). The clustered sandwich SE formula was also recently corrected for a sign issue.

- **Caveat per project memory**: SE coverage breaks at beta=0.95 in the tabular 1D DGP. The Tier 4 ss-high-gamma cell (beta=0.999) will likely show this as a coverage failure; the failure_mode `inference_unsupported` is appropriate.

- VALIDATION_LOG.md status: **Pass with caveat (cross-fitting variance fragile at high beta)**.
