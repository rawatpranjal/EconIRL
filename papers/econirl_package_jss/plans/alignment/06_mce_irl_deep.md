## Estimator: MCE-IRL Deep (neural reward variant of MCE-IRL)
## Paper(s): same Ziebart 2010 thesis as the tabular version. The neural-reward variant is the imitation library's pattern (Wulfmeier, Ondruska, and Posner 2015 "Maximum Entropy Deep Inverse Reinforcement Learning" is the canonical neural extension; not in `papers/foundational/` and we should add it).
## Code: `src/econirl/estimation/mce_irl.py` — same module, selected via `reward_type="neural"` constructor argument.

### Loss / objective

- Paper formula: same dual MCE objective as the tabular version, with the linear reward `R = phi @ theta` replaced by a neural reward `R = phi_net(s, a; theta)` where `theta` are network weights. The gradient is the same form `mu_pi - mu_E` but contracted with the network's Jacobian `d_theta R(s, a)` rather than the static feature matrix.

- Code implementation: `mce_irl.py` selects between linear and neural via the `reward_type` constructor argument. The neural path swaps `LinearUtility.compute(theta)` for `phi_net(theta)`; the loss and inner-Bellman-solve flow unchanged. Optax Adam is the default optimizer for the neural variant (vs L-BFGS for tabular).

- Match: **yes**, follows the Wulfmeier-Ondruska-Posner pattern. Network architecture defaults are documented in the `MCEIRLConfig` constructor.

### Gradient

- Paper formula: same `mu_pi - mu_E` form with neural Jacobian.

- Code implementation: JAX autodiff through the network and the inner soft Bellman solver. The `value_iteration` solver is wrapped in `jax.lax.while_loop` so `jax.grad` differentiates correctly through the fixed point.

- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: per outer step, recompute `R(s, a)` from the network, run soft VI to convergence, compute occupancy, update theta.

- Code algorithm: matches. The inner `value_iteration` is jit-compiled; the outer Adam step uses `jax.grad` on the dual loss.

- Match: **yes**.

### Identification assumptions

- Paper conditions: same as tabular MCE-IRL (action-dependent inputs to the network) plus the standard neural-network identifiability caveats (rewards identified up to additive constants and scale per Kim et al. 2021). The neural network can in principle absorb any constant shift, so identification post-fit relies on the same anchor-action normalization the tabular case uses.

- Code enforcement: same wrapper issue as tabular — if `feature_matrix=None`, the network input degenerates. The Tier 4 ss-neural-r cell sets `dataset_config.reward_type="neural"` and passes the env's feature matrix through the `LinearUtility.from_environment` path, so the cell does not hit the wrapper bug.

- Match: **same caveat as tabular**.

### Hyperparameter defaults vs paper defaults

- `MCEIRLConfig.network_width`: 64.
- `MCEIRLConfig.network_depth`: 2.
- `MCEIRLConfig.learning_rate`: 1e-3 for Adam.
- `MCEIRLConfig.outer_max_iter`: 500 (more than tabular because Adam needs more steps).

Match: **yes**, standard imitation-library defaults.

### Findings / fixes applied

- No code fixes applied. Same wrapper-default issue as tabular MCE-IRL; same workaround (cells pass `feature_matrix` explicitly).

- A **paper-side gap**: the Wulfmeier-Ondruska-Posner 2015 paper is the canonical reference for neural-reward MaxEnt IRL but is not in `papers/foundational/`. **Action**: docling and add it. Tracked in CLOUD_VERIFICATION_QUEUE.md as a follow-up.

- VALIDATION_LOG.md status: **Pending** (Tier 4 ss-neural-r cell will validate).
