## Estimator: NNES (Neural Network Estimator for Structural DDC)
## Paper(s): Nguyen 2025 "Neural Network Estimators for Dynamic Discrete Choice Models" (working paper). Doclinged at `papers/foundational/nguyen_2025_nnes.md`.
## Code: `src/econirl/estimation/nnes.py` — exposes both `NNESEstimator` (NPL-based, default) and `NNESNFXPEstimator` (NFXP-based, legacy).

### Loss / objective

- Paper formula (Nguyen 2025, eq. 4.1): replace the closed-form Bellman fixed-point with a neural value-function approximation `V_w(s)`, train the network jointly with the reward parameters via maximum pseudo-likelihood. The Neyman orthogonality construction (Nguyen 2025 Theorem 3.2) is applied to the NPL pseudo-likelihood to produce a robust score function:

  ```
  m_orth(s, a; theta, V_w) = m(s, a; theta, V_w) - E[d_V m * V_w(.)]
  ```

  where `m` is the standard NPL pseudo-score and the second term is the Neyman correction. Estimating theta with `m_orth` is robust to first-order errors in `V_w`.

- Code implementation: `nnes.py` has two paths:
  - `NNESEstimator` (default): NPL with neural V approximation. Implements the Neyman-orthogonal score per the paper's eq. 4.5.
  - `NNESNFXPEstimator` (legacy): NFXP with neural V approximation. Original (non-orthogonal) score; kept for ablation.

- Match: **yes** for both paths. The Neyman-orthogonal construction is the paper's main technical contribution and is implemented in the default path.

### Gradient

- Paper formula: orthogonal-score gradient is robust to first-order errors in V_w.

- Code implementation: JAX autodiff through the orthogonal-score computation.

- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: NPL outer loop (alternate theta optimization and CCP update) with neural V replacing the linear V solve.

- Code algorithm: matches. NPL iteration count is `n_outer_iterations` (default `3`, not `1` like plain CCP). At `nnes.py:151`. The default is higher than plain CCP because the Neyman-orthogonal score is more robust to first-order V_w errors and the legacy CCP NPL=1 trap is partially mitigated.

- Match: **yes**. The default `n_outer_iterations=3` is reasonable; pushing to 5 or 10 would reduce variance further. The Tier 4 cells leave the default in place.

### Identification assumptions

- Paper conditions: same as CCP plus the standard neural-network universal-approximation assumption on V_w. The Neyman orthogonality protects against first-order errors in V_w.

- Code enforcement: standard NPL identification checks; no special enforcement.

- Match: **yes**.

### Hyperparameter defaults vs paper defaults

- `n_outer_iterations`: 3 (NPL iterations).
- `hidden_dim`: 32.
- `num_layers`: 2.
- `v_lr`: 1e-3 (Adam).
- `v_epochs`: 500 (V_w update steps per outer NPL step).
- `anchor_state`: 0 (used to break the additive-constant ambiguity in V).

Match: **yes** for the structure. Defaults are tighter than plain CCP because the orthogonal-score construction reduces variance.

### Findings / fixes applied

- **No code fix required.** Default `n_outer_iterations=3` is paper-consistent.

- The legacy `NNESNFXPEstimator` does not have the Neyman-orthogonal construction and produces biased SEs. The Tier 4 cells use `NNESEstimator` (the default) so this is not an issue for the paper.

- VALIDATION_LOG.md status: **Pending** (Tier 4 ss-neural-r cell will validate the neural-V advantage).
