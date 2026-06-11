# GLADIUS Internal Notes

GLADIUS is the package's neural empirical-risk route for high-dimensional DDC
and IRL settings. It learns Q and expected-value objects from panel transitions,
then projects implied rewards onto interpretable structural features. The method
is useful for model-free approximation and large state spaces, but the current
validation is not counterfactual-certified under strict structural gates.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Public RTD source: `../../../docs/estimators/gladius.md`.
- Neural wrapper: `../../../src/econirl/estimators/neural_gladius.py`.
- Lower-level estimator: `../../../src/econirl/estimation/gladius.py`.
- Validation runner: `../../../validation/estimators/gladius/run.py`.
- Validation results: `../../../validation/results/gladius.json` and
  `../../../validation/results/gladius_scaled.json`.

## Estimator Contract

The paper objective combines a policy likelihood term with a Bellman penalty.
The package implementation uses neural Q and continuation-value components:

```text
V_Q(s) = sigma log sum_a exp(Q(s, a) / sigma)
zeta(s, a) approx E[V_Q(s') | s, a]
r_hat(s, a) = Q(s, a) - beta zeta(s, a).
```

For structural interpretation, the implied reward is projected onto feature
differences:

```text
Delta r_hat(s, a) = r_hat(s, a) - r_hat(s, anchor_action)
Delta phi(s, a)   = phi(s, a) - phi(s, anchor_action)
theta_hat         = argmin_theta ||Delta r_hat - Delta phi theta||^2.
```

The action-difference projection removes a state-dependent Q constant. It does
not by itself certify full Bellman reward recovery or counterfactual validity.

## Paper-To-Package Difference

The original GLADIUS argument uses observed rewards to anchor the Bellman error.
In IRL/DDC package use, rewards are latent. The package therefore monitors raw
Bellman reward diagnostics, projected reward diagnostics, policy imitation,
value/Q recovery, and counterfactual regret separately. A good projected reward
is not enough to claim structural counterfactual validity if raw Bellman reward
or value gates fail.

## Validation Status

Both current GLADIUS artifacts have status
`strict_structural_counterfactual_fail` and
`counterfactual_valid_certified = false`.

| Cell | Converged | Iter. | Time (s) | Parameter cosine | Projected reward NRMSE | Raw Bellman reward NRMSE | Policy TV | Value NRMSE | Q NRMSE | Failed gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gladius_paper_high_state` | true | 235 | 296.7 | 0.975 | 0.198 | 0.571 | 0.0369 | 0.420 | 0.235 | raw reward, value |
| `gladius_paper_high_state_scaled` | true | 161 | 175.0 | 0.986 | 0.134 | 0.590 | 0.0381 | 0.495 | 0.222 | raw reward, value |

Shared support diagnostics:

- feature rank: 4/4;
- condition number: 4.376;
- observed states: 21/21;
- state-action coverage: 1.000;
- single-action states: 0;
- anchor valid: true;
- maximum transition row error: about `2.4e-8`.

The scaled artifact improves projected parameter and reward recovery, but it
does not resolve the raw Bellman reward or value gates. Public docs should
therefore describe GLADIUS as promising and diagnostically useful, not as
certified for structural counterfactuals.

## Counterfactual Evidence

The current counterfactual regrets are small, especially in the scaled artifact:

| Cell | Type A regret | Type B regret | Type C regret |
| --- | ---: | ---: | ---: |
| `gladius_paper_high_state` | 0.00854 | 0.0529 | 0.00852 |
| `gladius_paper_high_state_scaled` | 0.00274 | 0.00742 | 0.000493 |

These numbers are useful but not sufficient. The release rule is strict:
counterfactual-valid certification requires the structural gates to pass, not
only low perturbation regret.

## Debugging Order

1. Verify the panel transition tuples `(s, a, s')` and feature differences.
2. Check Q-network and EV-network scaling before changing structural gates.
3. Inspect final loss and policy TV.
4. Compare raw Bellman reward, projected reward, and anchor-projected reward as
   separate objects.
5. Compare value and Q recovery.
6. Use counterfactual regret as a final diagnostic, not as a replacement for
   failed structural gates.

## Implementation Paths

- Neural wrapper: `../../../src/econirl/estimators/neural_gladius.py`.
- Lower-level estimator: `../../../src/econirl/estimation/gladius.py`.
- Hyperparameter sweep script: `../../../scripts/gladius_hyperparam_sweep.py`.
- Validation runner: `../../../validation/estimators/gladius/run.py`.
- Validation JSON: `../../../validation/results/gladius.json`.
- Scaled validation JSON: `../../../validation/results/gladius_scaled.json`.
- Public docs: `../../../docs/estimators/gladius.md`.

## Public Documentation Boundary

Public RTD should state that GLADIUS currently has strong projected-reward and
policy evidence but is not strict-counterfactual-certified. The internal page
keeps the deeper distinction between raw Bellman reward, projected reward,
value/Q recovery, and counterfactual gates.
