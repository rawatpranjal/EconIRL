# Simulation Study

Deep MCE-IRL runs on a synthetic cell with a fixed nonlinear neural reward,
known stochastic transitions, linear state features, and an anchor action that
normalizes the reward. The cell has 32 states, 3 actions, and full
state-action coverage, so every recovery claim is checked against the oracle
reward matrix, policy, value function, Q function, and counterfactual objects.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/deep_mce_irl/run.py).
It writes the machine-readable results file
[`deep_mce_irl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/deep_mce_irl.json).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/deep_mce_irl/run.py
```

## Design

| Quantity | Value |
| --- | --- |
| States | 32 |
| Actions | 3 |
| Individuals | 2,000 |
| Periods per individual | 80 |
| Observations | 160,000 |
| Discount factor | 0.95 |
| Reward type | Fixed nonlinear neural (2 layers, hidden dim 32) |
| Anchor action | 0 |

## Fit Summary

| Quantity | Value |
| --- | --- |
| Converged | True |
| Log-likelihood | -162257.61 |
| Estimation time | 10.4 seconds |
| Feature residual (L2) | 0.00386 |
| Occupancy moment residual (max) | 0.00182 |

## Recovery Metrics

| Metric | Value |
| --- | --- |
| Reward normalized RMSE | 0.0436 |
| Policy TV | 0.00475 |
| Value normalized RMSE | 0.0778 |
| Q normalized RMSE | 0.0442 |
| Type A regret | 0.00164 |
| Type B regret | 0.00148 |
| Type C regret | 0.00191 |

## Checks

| Check | Value | Threshold | Status |
| --- | --- | --- | --- |
| converged | true | is true | pass |
| occupancy_moment_residual | 0.00182 | at most 0.03 | pass |
| reward_normalized_rmse | 0.0436 | at most 0.15 | pass |
| policy_tv | 0.00475 | at most 0.05 | pass |
| value_normalized_rmse | 0.0778 | at most 0.15 | pass |
| q_normalized_rmse | 0.0442 | at most 0.15 | pass |
| type_a_regret | 0.00164 | at most 0.08 | pass |
| type_b_regret | 0.00148 | at most 0.08 | pass |
| type_c_regret | 0.00191 | at most 0.08 | pass |

## Notes on the Primary Cell

There is no parameter vector in the primary cell results. That is by design: a
neural reward map does not have a unique structural parameter vector that can
be compared across networks, so reward-map recovery and behavioral metrics are
the right scorecard. The support cells (`deep_mce_neural_features` and
`deep_mce_neural_reward_features`) exercise the projected theta path, but
projection quality is contingent on the projection being identified.

Deep MCE-IRL also appears on the
[bus engine](../../simulation_studies/rust_bus.md) and
[gridworld](../../simulation_studies/index.md) pages of the simulation studies,
where it is compared against the full structural and IRL rosters.
