# Changelog

## 0.1.0 (unreleased)

This release completes the applied workflow across nine estimators. Each one
loads panel data through the documented public surface, runs pre-estimation
checks, fits, reports convergence, produces a summary, predicts out of sample,
quantifies uncertainty, runs a counterfactual, and serializes without changing
supported results. Every estimator ships a runnable notebook under `examples/`.

### Estimators qualified in this release

NFXP, CCP, TD-CCP, MCE-IRL, Neural MCE-IRL, AIRL, NeuralAIRL, AIRL2, GLADIUS.

### Added

- `NeuralAIRL`, a standalone nonlinear state-only AIRL estimator.
- `AIRL2` at the package root, the canonical name for anchored heterogeneous
  adversarial IRL.
- `DeterministicTransitions` and `MCEIRLTask` at the package root.
- Bootstrap inference for estimators without usable analytic standard errors,
  with repeated-simulation coverage evidence in `validation/results/`.
- One applied notebook per estimator under `examples/`. Notebooks are excluded
  from the documentation build.
- `docs/comparing_estimators.md`, a single page for choosing between methods.

### Known limitation

GLADIUS is a partial match to Table 2 of its source paper, not a completed
replication. Five of the six sample sizes land inside the paper's mean plus two
standard errors. The largest, 5000 trajectories, does not: 0.26 percent reward
MAPE against a 0.24 bound. The NFXP oracle control in the same harness beats the
paper's own Rust column at every sample size, and the best seeds at 5000 already
reach the paper value, so the miss is variance in the tail. See the GLADIUS
validation page for the per-cell table.

### Changed

- The estimator documentation is split into a core roster and an other roster.
- Repeated data-contract, diagnostics, and uncertainty prose moved to shared
  pages. Estimator pages carry the model, identification, API, and evidence.

### Migration from 0.0.10

Nothing was removed from the public import surface. The renames below keep
working through 0.1.x and emit a `DeprecationWarning` on use.

| 0.0.10 | 0.1.0 | Action |
| --- | --- | --- |
| `AIRLHet` | `AIRL2` | Rename the class. |
| `AIRLHetEstimator` | `AIRL2Estimator` | Rename the class. |
| `AIRLHetConfig` | `AIRL2Config` | Rename the class. |
| `econirl.estimation.adversarial.airl_het` | `econirl.estimation.adversarial.airl2` | Import from the new module. |

Pickles written by 0.0.10 under the `airl_het` module load and re-save through
the canonical `airl2` module.

## 0.0.10 (2026-07-26)

NFXP and CCP qualified against the 0.1 workflow contract. The Rust (1987)
Table IX replication matches the published estimates and standard errors.
