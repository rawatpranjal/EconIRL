# NFXP Paper Context

Primary sources: Rust (1987) for the bus replacement model and nested
fixed-point likelihood, and Iskhakov et al. (2016) for the NFXP-MPEC
computational comparison.

Public citations live in `../../../docs/references.md`. Broader paper routing
lives in `../../papers/index.md`.

Internal context to preserve here: Bellman fixed-point derivation, logit CCP
formula, likelihood construction, optimizer behavior, Rust replication notes,
counterfactual definitions, and how exact NFXP acts as the reference estimator
for CCP, MPEC, NNES, and SEES.

## Paper-To-Package Translation

Rust (1987) is an empirical bus replacement model. The current package
validation is not a re-estimation of the original empirical data; it is a
known-truth DGP designed to check the NFXP implementation when the structural
assumptions hold exactly.

The package keeps the Rust logic that matters for estimator mechanics:

- tabular state;
- discrete keep/replace-style actions, generalized to three actions in the
  canonical cell;
- Markov transition law separated from payoff estimation;
- logit shocks with fixed scale;
- nested solution of the dynamic program inside likelihood evaluation.

The package validation differs from the paper in ways that should be stated
explicitly in internal work:

- the canonical cell has a synthetic transition law and synthetic features;
- the true value, Q, reward, and policy objects are known;
- Type A/B/C counterfactual oracles are generated directly from the DGP;
- the exit action and absorbing state pin down payoff location.

Iskhakov et al. (2016) enters through numerical design. The internal default is
the hybrid inner solver, which combines safe successive approximation with
Newton-Kantorovich behavior near the fixed point.
