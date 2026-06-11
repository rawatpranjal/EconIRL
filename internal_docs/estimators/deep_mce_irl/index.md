# Deep MCE-IRL Internal Notes

Deep MCE-IRL extends maximum causal entropy IRL by replacing a small linear
reward table with a neural reward map or neural feature encoder. The estimator
still lives inside the MCE contract: it uses known transitions, solves a
soft Bellman problem, compares expert and model occupancies, and validates the
induced policy, value function, Q function, reward surface, and
counterfactuals.

The neural part changes what is recoverable. Raw network weights are not a
structural object. The validated object is the anchored reward map and the
behavior it induces.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Public RTD source: `../../../docs/estimators/deep_mce_irl.md`.
- Package wrapper: `../../../src/econirl/estimators/mceirl_neural.py`.
- Lower-level MCE solver: `../../../src/econirl/estimation/mce_irl.py`.
- Validation runner: `../../../validation/estimators/deep_mce_irl/run.py`.
- Validation result: `../../../validation/results/deep_mce_irl.json`.

## Estimator Contract

The maintained estimator starts from the MCE model

```text
Q(s, a) = r_eta(s, a) + beta * sum_{s'} P(s' | s, a) V(s')
V(s)    = log sum_a exp(Q(s, a))
pi(a|s) = exp(Q(s, a) - V(s)).
```

The neural component enters through one of two routes:

```text
r_eta(s, a) = f_eta(x(s, a))
```

or

```text
r_theta(s, a) = h_eta(x(s, a))' theta.
```

The validation suite separates these cases because their interpretation differs:

- raw neural reward: validate the full anchored reward matrix and induced
  behavior, not neural weights;
- neural features with linear reward: validate the projected reward and only
  compare finite `theta` when the projection is identified;
- neural reward plus neural features: treat the reward map and behavior as the
  target, not a unique parameter vector.

## Identification And Normalization

Deep MCE inherits all MCE reward-gauge problems and adds neural
non-identification. Multiple neural parameter vectors can represent the same
reward map, and multiple reward maps can induce nearly identical policies under
limited support. The internal docs and public RTD should therefore avoid
claiming structural recovery of neural weights.

Use this hierarchy when interpreting a run:

1. transition tensor, state encoding, action encoding, and discount are aligned;
2. action support and state-action coverage are adequate;
3. reward anchor is valid;
4. occupancy moments and feature moments are small;
5. reward map recovery is evaluated under the same gauge;
6. policy, value, Q, and counterfactual metrics pass;
7. projected finite parameters are discussed only when the projection is
   identified.

## Validation Cells

The current validation artifact contains one primary release cell and three
support cells. The primary cell is `deep_mce_neural_reward`.

| Cell | Role | Reward/feature target | Rank | Cond. | Obs. | Reward NRMSE | Policy TV | Value NRMSE | Q NRMSE | Counterfactual regrets | Gate status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `canonical_low_state_only` | support/sanity | projected linear state reward | 3/3 | 6.232 | 80000 | 0.0648 | 0.0111 | 0.0923 | 0.0629 | A 0.0105, B 0.0775, C 0.0011 | pass |
| `deep_mce_neural_reward` | primary | raw anchored neural reward matrix | 4/4 | 39.859 | 160000 | 0.0436 | 0.00475 | 0.0778 | 0.0442 | A 0.00164, B 0.00148, C 0.00191 | pass |
| `deep_mce_neural_features` | support/finite-theta check | projected linear reward on neural features | 4/4 | 478.680 | 160000 | 0.0539 | 0.00238 | 0.00541 | 0.0551 | A 0.000240, B 0.000259, C 0.000154 | pass with theta warning |
| `deep_mce_neural_reward_features` | support/stress | raw neural reward with neural features | 4/4 | 162.486 | 160000 | 0.0380 | 0.00710 | 0.0141 | 0.0429 | A 0.00244, B 0.00235, C 0.000227 | pass |

## Primary Cell Details

The primary cell uses a frozen nonlinear neural reward, linear features, known
stochastic transitions, and full reward-map validation.

| Check | Current value | Interpretation |
| --- | ---: | --- |
| Converged | true | Optimizer reached a reported solution. |
| Iterations | 166 | Neural reward fit solved within the release run. |
| Observations | 160000 | Large simulated panel for the release cell. |
| State count | 32 | Full finite validation environment. |
| State-action coverage | 1.000 | All state-action cells represented. |
| Minimum action share | 0.189 | Weakest action still has support. |
| Anchor action | 0 | Reward map is compared under a fixed gauge. |
| Feature residual | 0.00386 | Feature moments pass the primary tolerance. |
| Occupancy moment residual | 0.00182 | Occupancy moments pass the primary tolerance. |
| Hidden dimension | 32 | Current neural reward architecture. |
| Number of layers | 2 | Current neural reward architecture. |
| Learning rate | 0.003 | Current optimizer setting. |
| Reward normalized RMSE | 0.0436 | Anchored reward map is close to truth. |
| Policy total variation | 0.00475 | Induced behavior is close to truth. |
| Value normalized RMSE | 0.0778 | Value object is close to truth. |
| Q normalized RMSE | 0.0442 | State-action value object is close to truth. |

The primary result has no interpretable parameter vector in the release table.
That is intentional. A neural reward map can be behaviorally and numerically
validated without a unique structural vector of network weights.

## Caveats

- Neural weights are not a structural estimand.
- Projected finite parameters are only meaningful when the projection is
  identified and well-conditioned.
- The `deep_mce_neural_features` support cell has excellent reward and behavior
  recovery but poor raw theta recovery; this is a warning against over-reading
  parameter vectors in neural feature spaces.
- The current validation uses supplied encodings and known transitions.
- Public docs should present Deep MCE as a neural reward-map estimator, not as a
  black-box proof of reward identification in arbitrary environments.

## Debugging Order

1. Confirm the MCE base solver works on the same transition tensor.
2. Validate state/action encodings and reward anchoring.
3. Check support: state coverage, state-action coverage, and minimum action
   share.
4. Inspect feature rank and condition number for projected finite-theta cells.
5. Compare occupancy residuals before reward metrics.
6. Compare anchored reward maps before neural weights.
7. Compare policy, value, Q, and counterfactuals.
8. Treat optimizer settings as a late-stage diagnosis unless moments fail after
   the data and gauge checks pass.

## Implementation Paths

- Package wrapper: `../../../src/econirl/estimators/mceirl_neural.py`.
- Lower-level MCE implementation: `../../../src/econirl/estimation/mce_irl.py`.
- Validation runner: `../../../validation/estimators/deep_mce_irl/run.py`.
- Validation JSON: `../../../validation/results/deep_mce_irl.json`.
- Public docs: `../../../docs/estimators/deep_mce_irl.md`.

## Public Documentation Boundary

Public RTD should explain the neural reward-map contract, required inputs,
basic usage, validation summary, and caveats about neural weight
non-identification. The detailed cell matrix, projection caveats, and debugging
order belong here.
