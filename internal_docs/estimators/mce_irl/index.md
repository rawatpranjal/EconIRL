# MCE-IRL Internal Notes

Maximum causal entropy IRL estimates a reward function from expert state-action
occupancy by matching discounted feature moments under a known transition model.
In this package it is the entropy-regularized IRL estimator closest to a
structural dynamic discrete choice likelihood: the reward enters a soft Bellman
recursion, the policy is causal, and counterfactuals are meaningful only through
the fitted MDP primitives.

## Source Boundary

- Paper context: `papers.md`.
- Implementation links: `links.md`.
- Public RTD source: `../../../docs/estimators/mce_irl.md`.
- Public quick start: `../../../docs/estimators/mce_irl/quick_start.md`.
- Public validation page: `../../../docs/estimators/mce_irl/validation.md`.
- Validation runner: `../../../validation/estimators/mce_irl/run.py`.
- Validation result: `../../../validation/results/mce_irl.json`.

## Estimand

The maintained estimator targets a reward representation

```text
r_theta(s, a) = phi(s, a)' theta
```

given:

- a finite state space;
- a finite action space;
- known transition probabilities `P(s' | s, a)`;
- expert state-action observations;
- feature vectors `phi(s, a)`;
- a discount factor `beta`;
- a reward normalization or anchor that removes the additive gauge.

For a fixed `theta`, the soft value functions solve

```text
Q_theta(s, a) = r_theta(s, a)
              + beta * sum_{s'} P(s' | s, a) V_theta(s')

V_theta(s) = log sum_a exp(Q_theta(s, a))

pi_theta(a | s) = exp(Q_theta(s, a) - V_theta(s)).
```

This policy is causal: at time `t`, action probabilities depend on current
state, the current reward, and continuation values. They do not condition on
future realized states. That is the critical distinction from trajectory
maximum-entropy IRL and is why this estimator is the default entropy IRL route
for dynamic discrete choice comparisons.

## Moment Condition

Let `d_E(s, a)` denote the empirical discounted expert occupancy and
`d_theta(s, a)` the discounted occupancy induced by the soft-optimal policy and
the known transition model. The optimizer solves the feature-matching condition

```text
sum_{s,a} d_E(s, a) phi(s, a)
  = sum_{s,a} d_theta(s, a) phi(s, a)
```

up to numerical tolerance. Equivalently, the gradient of the causal-entropy
dual objective is

```text
grad_theta L(theta)
  = E_E[phi(s, a)] - E_theta[phi(s, a)].
```

The result object reports both feature residuals and occupancy moment residuals
because a small parameter error is not the primary release criterion. The
behavioral object is the induced policy, value function, reward surface, and
counterfactual response.

## Identification

The reward is identified only after removing reward transformations that do not
change the policy in the maintained environment. The operational checks are:

- the feature design must have full rank after any anchor or normalization;
- each action must have enough observed support for the target comparison;
- known transitions must be row-stochastic and align with the observation
  encoding;
- the reward normalization must be applied consistently to truth and estimates;
- parameter comparisons are secondary to reward-map and policy comparisons.

The current validation cell uses an action-dependent feature design with full
rank `8/8`, transition row error below `2e-8`, full state coverage, and no
single-action states. Those are the preconditions that make the finite-theta
comparison interpretable.

## Validation Design

The current release artifact is the `mce_low_high_reward` validation cell. It
simulates a known low/high reward environment, estimates the linear reward, and
checks reward, value, Q, policy, and counterfactual recovery. The hard release
gates are in `../../../validation/results/mce_irl.json`.

| Check | Current value | Interpretation |
| --- | ---: | --- |
| Converged | true | Optimizer reached a reported solution. |
| Observations | 300000 | Large enough for tight moment checks in the release cell. |
| Iterations | 25 | Root optimizer solved the moment system quickly. |
| Feature rank | 8/8 | Linear reward comparison is identified in the anchored design. |
| Feature condition number | 1.373 | Well-conditioned release design. |
| State coverage | 25/25 | All states observed. |
| State-action coverage | 1.000 | All state-action cells represented. |
| Minimum action share | 0.330 | No action is near unsupported. |
| Feature difference | 1.89e-12 | Feature moments match to numerical precision. |
| Occupancy moment residual | 0.00106 | Discounted occupancy moments pass the tolerance gate. |
| Parameter cosine | 0.996 | Direction of finite-theta reward is recovered. |
| Parameter RMSE | 0.00630 | Small absolute finite-theta error. |
| Parameter relative RMSE | 0.0917 | Passes current finite-theta tolerance. |
| Reward normalized RMSE | 0.0823 | Reward map is within release tolerance. |
| Policy total variation | 0.00698 | Induced behavior is close to truth. |
| Value normalized RMSE | 0.0826 | Value object is close to truth. |
| Q normalized RMSE | 0.0816 | State-action value object is close to truth. |

## Release Gates

| Gate | Threshold | Current value | Status |
| --- | ---: | ---: | --- |
| Convergence | true | true | pass |
| Feature residual | <= 0.02 | 1.89e-12 | pass |
| Occupancy moment residual | <= 0.02 | 0.00106 | pass |
| Reward normalized RMSE | <= 0.10 | 0.0823 | pass |
| Policy total variation | <= 0.03 | 0.00698 | pass |
| Value normalized RMSE | <= 0.10 | 0.0826 | pass |
| Q normalized RMSE | <= 0.10 | 0.0816 | pass |
| Type A regret | <= 0.05 | 0.000433 | pass |
| Type B regret | <= 0.05 | 0.000410 | pass |
| Type C regret | <= 0.05 | 0.000094 | pass |

## Counterfactual Evidence

Counterfactuals are evaluated on three perturbation families. The numbers below
come from the same validation artifact as the main estimator gates.

| Counterfactual | Policy TV | Regret | Value RMSE |
| --- | ---: | ---: | ---: |
| Type A | 0.00646 | 0.000433 | 0.000742 |
| Type B | 0.00628 | 0.000410 | 0.000523 |
| Type C | 0.00421 | 0.000094 | 0.000145 |

Interpret these as local validation in the known simulation environment, not a
general guarantee. Counterfactual validity still requires the same transition
law, support, reward representation, and policy-response assumptions used in
the fitted MDP.

## Debugging Order

When MCE-IRL fails or drifts, debug in this order:

1. Verify transition rows and state-action indexing before looking at the
   optimizer.
2. Check feature rank, normalization, and anchor validity.
3. Confirm empirical occupancy and discounted occupancy use the same discount
   and start-state distribution.
4. Inspect action support and single-action states.
5. Compare feature moments before comparing parameters.
6. Compare reward maps under the same gauge.
7. Compare induced policy, value, and Q objects.
8. Only then inspect optimizer hyperparameters.

## Implementation Paths

- Estimator implementation: `src/econirl/estimation/mce_irl.py`.
- Public API export: `src/econirl/estimation/__init__.py`.
- Validation runner: `../../../validation/estimators/mce_irl/run.py`.
- Validation JSON: `../../../validation/results/mce_irl.json`.
- Public docs: `../../../docs/estimators/mce_irl.md`.

## Public Documentation Boundary

The public RTD page should explain when to use MCE-IRL, required inputs,
minimal examples, and validation receipts. It should not reproduce the full
paper derivation, all debugging notes, or every simulation table. Those belong
here for maintainers and AI agents.
