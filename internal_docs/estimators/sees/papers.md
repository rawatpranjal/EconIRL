# SEES Paper Context

Primary source: Luo and Sang (2024) for sieve estimation of structural models.

Public citations live in `../../../docs/references.md`. Broader paper routing
lives in `../../papers/index.md`.

Internal context to preserve here: sieve basis construction, Bellman penalty,
simulation objective, continuous-covariate stress tests, default penalty
rationale, and the reason SEES public RTD exposure is currently controlled by
Sphinx exclusion.

## Paper-To-Package Translation

Luo and Sang (2024) motivates SEES as a structural estimator with a sieve value
approximation and a Bellman-equilibrium penalty. The package implementation
keeps that contract: reward is finite-dimensional, value is approximated by a
basis, and Bellman residual is part of the evidence.

The current package result differs from a paper-final empirical design:

- the primary result is the synthetic encoded-state known-truth DGP;
- the basis is an encoded-state full-rank `rbf_svd` construction with 81
  columns;
- the final optimizer convergence flag is false, even though all structural
  gates pass;
- SEES remains excluded from public RTD until the caveat is intentionally
  accepted.

The paper warning is practical: basis adequacy and penalty strength are not
tuning cosmetics. If the sieve cannot represent the value function, or if the
penalty lets the Bellman residual drift, SEES is not a structural estimate.
