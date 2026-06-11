# Max Margin IRL Internal Notes

Max Margin IRL is the Abbeel-Ng apprenticeship-learning estimator. It learns
reward weights that separate expert feature expectations from the feature
expectations of alternative policies by a margin. The package keeps it through
contrib and a high-level wrapper, but it does not currently have public RTD
coverage or a tracked validation JSON.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Public RTD page: none currently.
- Package wrapper: `../../../src/econirl/estimators/max_margin_irl.py`.
- Contrib implementation: `../../../src/econirl/contrib/max_margin_irl.py`.
- Tests: `../../../tests/contrib/test_max_margin_irl_core.py` and
  `../../../tests/contrib/test_max_margin_irl_sklearn.py`.
- Validation result: none currently.

## Algorithm

The estimator uses iterative constraint generation:

1. compute expert feature expectations `mu_E`;
2. initialize reward weights `theta`;
3. solve the MDP under current reward to find a violating policy;
4. compute that policy's feature expectations `mu_i`;
5. solve a margin problem so that

```text
theta' mu_E - theta' mu_i >= margin
```

for all collected violating policies;
6. repeat until margin improvement is below tolerance.

The package supports state-only `LinearReward` and action-dependent
`ActionDependentReward`. Identification is handled through either a unit-norm
constraint or an optional anchor parameter.

## Interpretation

Max Margin IRL is useful as a feature-expectation apprenticeship baseline. It
does not naturally return likelihood-based standard errors, and the margin
normalization is not the same as an economic reward scale. Treat the learned
theta as a separating direction unless a validation artifact proves stronger
structural recovery.

## Current Release State

Current evidence:

- contrib implementation exists;
- sklearn-style wrapper exists;
- core and wrapper tests exist;
- no tracked validation JSON exists;
- no public RTD estimator page exists.

Before promotion, add a known-truth validation runner with parameter direction,
feature expectation gap, policy quality, reward gauge, and counterfactual
metrics.

## Debugging Order

1. Verify feature expectations for expert demonstrations.
2. Check reward type: state-only versus action-dependent.
3. Confirm value iteration for violating policies converges.
4. Inspect active constraints and margin improvement.
5. Check unit-norm or anchor normalization.
6. Compare policy quality only after the margin problem is stable.

## Implementation Paths

- Package wrapper: `../../../src/econirl/estimators/max_margin_irl.py`.
- Contrib estimator: `../../../src/econirl/contrib/max_margin_irl.py`.
- Core tests: `../../../tests/contrib/test_max_margin_irl_core.py`.
- Wrapper tests: `../../../tests/contrib/test_max_margin_irl_sklearn.py`.

## Public Documentation Boundary

Keep Max Margin IRL out of public estimator docs until release artifacts exist.
