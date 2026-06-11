# TD-CCP Paper Context

Primary sources: Adusumilli and Eckardt (2025) for temporal-difference
estimation of DDC models and Hotz and Miller (1993) for the CCP foundation.

Public citations live in `../../../docs/references.md`. Broader paper routing
lives in `../../papers/index.md`.

Internal context to preserve here: recursive term estimation, Algorithm 2
locally robust standard errors, cross-fitting, zeta and lambda diagnostics,
finite-dimensional reward assumptions, and why transition-free estimation is
not transition-free counterfactual evaluation.

## Paper-To-Package Translation

Adusumilli and Eckardt (2025) is a finite-dimensional reward-parameter paper.
The package implementation should keep that contract explicit: TD-CCP estimates
theta from known reward features and observed transition tuples. It is not a
generic neural reward learner.

The package validation maps the paper to a synthetic encoded-state hard case:

- state features are encoded and two-dimensional;
- first-stage CCPs are logit with degree-2 state features;
- the recursive terms use the semigradient estimator;
- Algorithm 2 cross-fitting and locally robust standard errors are required;
- action 0 is the reward-normalized baseline.

Transition-density-free means transition density is not modeled in the
parameter-estimation step. It does not mean values, policies, and
counterfactuals can be evaluated without a transition environment.
