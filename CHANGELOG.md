# Changelog

## 0.1.1

Correctness and disclosure fixes found by an adversarial review of the 0.1.0
diff. No estimator numerics changed.

### Fixed

- `AIRL2` was registered in the estimator capability matrix with no matching
  build recipe in `econirl.forms.loader`, so `run_form` swallowed the generic
  "no build recipe" error into its skip log and silently dropped AIRL2 from
  every roster run. It now fails with a message naming the exit-action and
  absorbing-state anchors it needs, and a test walks the whole capability
  registry so no future entry can be silently skipped. AIRL2 used directly
  was never affected.
- The 0.1.0 migration notes understated the `AIRL` and `NeuralAIRL` change.
  See below.

### Documentation

- The 0.1.0 note "Nothing was removed from the public import surface" was
  true but misleading. `AIRL` and `NeuralAIRL` were rebound to different
  classes, which raises `TypeError` on an old call rather than warning. The
  migration section now says so.

### Known behavior, now pinned by a test

Neural MCE-IRL sets `converged_ = False` on a truncated fit without emitting
a warning. The cross-estimator contract test records this so it stays visible.

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

**`AIRL` and `NeuralAIRL` changed meaning. Neither emits a warning; both raise
`TypeError` on an old call.** In 0.0.10 the package root bound `AIRL` to the
neural, context-aware class (`AIRL = NeuralAIRL` in `econirl/estimators/__init__.py`),
whose constructor took `state_dim` and `context_dim` and whose `fit` needed no
transition tensor. In 0.1.0 `AIRL` is the tabular state-only estimator: it takes
`n_states` and raises `ValueError` if `transitions` is omitted. `NeuralAIRL` is a
new class with its own signature, also requiring `n_states` and transitions.

Because this is a rebinding of an existing name rather than a rename behind a
compatibility shim, upgrading code that calls the 0.0.10 `AIRL` or `NeuralAIRL`
fails immediately with `TypeError` on the unexpected keyword. That is loud, not
silent, but it is not a deprecation path.

There is no replacement for the observed-context conditioning the 0.0.10 class
offered. `NeuralAIRL` rejects `context` and directs callers to `AIRL2`, and
`AIRL2` conditions on latent segments rather than observed context, so a 0.0.10
workflow that passed a context encoder has no 0.1.0 equivalent.

The `AIRLHet` names below are a genuine deprecation: they keep working through
0.1.x and emit a `DeprecationWarning` on use.

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
