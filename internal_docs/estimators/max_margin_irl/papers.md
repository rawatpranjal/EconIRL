# Max Margin IRL Paper Context

Primary source: Abbeel and Ng (2004) for apprenticeship learning and
feature-expectation margin methods. Broader paper routing lives in
`../../papers/index.md`.

## Paper-To-Package Translation

| Paper concept | Package concept | Notes |
| --- | --- | --- |
| Expert feature expectations | Empirical feature sums from panel | State-only or action-dependent. |
| Candidate policies | Violating policies from MDP solves | Added iteratively. |
| Margin constraint | Expert better than each candidate | Implemented through QP/SLSQP-style solve. |
| Reward weights | `theta` | Direction and normalization matter. |
| Norm constraint | Unit norm or anchor parameter | Removes scale ambiguity. |
| Apprenticeship baseline | Contrib/wrapper estimator | Not public release evidence yet. |

## Derivation Checklist

1. Define expert feature expectations.
2. Define policy feature expectations.
3. State the separating margin objective.
4. Explain constraint generation.
5. Explain reward scale normalization.
6. State validation requirements for structural use.

## Release Gap

The package has implementation and tests but no tracked known-truth validation
artifact. Do not use this estimator in public comparison tables as certified
until that artifact exists.
