# NNES Paper Context

Primary sources: Nguyen (2025) for neural value approximation in structural
estimation, with Hotz and Miller (1993) and Aguirregabiria and Mira (2002) for
the CCP and NPL foundations.

Public citations live in `../../../docs/references.md`. Broader paper routing
lives in `../../papers/index.md`.

Internal context to preserve here: value-network target, NPL orthogonality,
paper simulation design, approximation-error limits, standard-error conditions,
and the distinction between neural value approximation and neural reward
recovery.

## Paper-To-Package Translation

Nguyen (2025) is a neural value-approximation paper for structural DDC
estimation. The package implementation should be read through that lens:
rewards remain finite-dimensional, while the continuation object is learned.

The paper's orthogonality result is asymptotic. The package validation turns
that theoretical claim into a finite-sample recovery audit by checking
parameters, reward tables, policies, values, Q functions, and Type A/B/C
counterfactuals against known truth.

Important differences from a paper-final empirical design:

- current evidence is synthetic known-truth evidence;
- the result file validates the high-dimensional cell, not a real Rust
  replication;
- standard errors are not currently validated because the result file stores
  `null` standard errors;
- support coverage is good but not complete, so the release status is
  `Certified with caveat`.
