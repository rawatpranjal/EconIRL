# Simulation Study

IQ-Learn runs on three synthetic cells covering low-dimensional tabular,
high-dimensional neural, and state-only reward settings. Each cell has known
transitions, policy, value, Q function, and counterfactual oracle objects, so
every recovery claim is checked against the truth. The primary cell is
`canonical_low_action`.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/iq_learn/run.py).
It writes the results file
[`iq_learn.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/iq_learn.json).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/iq_learn/run.py
```

## Primary Cell: canonical_low_action

### Design

| Quantity | Value |
| --- | --- |
| States | 21 |
| Actions | 3 |
| Individuals | 2,000 |
| Periods per individual | 80 |
| Observations | 160,000 |
| Q type | tabular |
| Divergence | chi2 |
| Alpha | 1.0 |

### Fit Summary

| Quantity | Value |
| --- | --- |
| Converged | True |
| Log-likelihood | -174923.515625 |
| Iterations | 173 |
| Estimation time | 3.49 seconds |
| Expert state coverage | 1.0 |
| Expert state-action coverage | 1.0 |

### Recovery Metrics

| Metric | Value | Gate | Status |
| --- | --- | --- | --- |
| Policy TV | 0.04068339984836971 | at most 0.05 | pass |
| Raw Bellman reward NRMSE | 0.3809617636095332 | at most 0.1 | fail |
| Projected reward NRMSE | 0.27739328652373035 | at most 0.1 | fail |
| Value NRMSE | 0.4855298533917329 | at most 0.1 | fail |
| Q NRMSE | 0.48137314674423415 | at most 0.1 | fail |
| Type A counterfactual regret | 0.011518059257921435 | at most 0.05 | pass |
| Type B counterfactual regret | 0.02155897051624924 | at most 0.05 | pass |
| Type C counterfactual regret | 0.009886935172236239 | at most 0.05 | pass |

The estimator passes imitation and counterfactual regret checks on the primary
cell. Reward, value, and Q recovery fail. Low regret on this cell reflects
that the Q-induced policy happens to produce near-oracle welfare under the
applied interventions, not that the reward or value objects are structurally
accurate.

## Stress Cell: canonical_high_action

| Quantity | Value |
| --- | --- |
| States | 81 |
| Actions | 3 |
| Individuals | 2,000 |
| Periods per individual | 80 |
| Observations | 160,000 |
| Q type | neural |

| Metric | Value | Gate | Status |
| --- | --- | --- | --- |
| Policy TV | 0.069342286794494 | at most 0.05 | fail |
| Raw Bellman reward NRMSE | 0.9969176297594339 | at most 0.1 | fail |
| Projected reward NRMSE | 0.7863419580279364 | at most 0.1 | fail |
| Value NRMSE | 1.7525372602832876 | at most 0.1 | fail |
| Q NRMSE | 1.4222683291851073 | at most 0.1 | fail |
| Type A regret | 0.09202234513797602 | at most 0.05 | fail |
| Type B regret | 0.38998822309194975 | at most 0.05 | fail |
| Type C regret | 0.026453418082599357 | at most 0.05 | pass |

All structural recovery gates and most regret gates fail on the high-dimensional
neural cell.

## Negative Control: canonical_low_state_only

| Quantity | Value |
| --- | --- |
| States | 21 |
| Actions | 3 |
| Individuals | 500 |
| Periods per individual | 80 |
| Observations | 40,000 |
| Q type | tabular |

| Metric | Value | Gate | Status |
| --- | --- | --- | --- |
| Policy TV | 0.03664439528766237 | at most 0.05 | pass |
| Raw Bellman reward NRMSE | 0.7500170275582363 | at most 0.1 | fail |
| Projected reward NRMSE | 0.28081766497887156 | at most 0.1 | fail |
| Value NRMSE | 0.5703591477554589 | at most 0.1 | fail |
| Q NRMSE | 0.5518449193241758 | at most 0.1 | fail |
| Type A regret | 0.034859046266863335 | at most 0.05 | pass |
| Type B regret | 0.05649933858379137 | at most 0.05 | fail |
| Type C regret | 0.020117752969335607 | at most 0.05 | pass |

The state-only cell passes imitation and most regret checks but fails
structural reward, value, and Q recovery.

## Sparse-Support Guard

The sparse-support guard uses a tiny panel with one observed state and one
observed state-action pair (state coverage 0.333, state-action coverage
0.167). Even when all non-coverage metrics are set to pass, the run is not
counterfactual-valid because support gates fail. The guard prevents future
changes from treating small policy or regret numbers as sufficient when the
expert panel does not cover the relevant state-action space.

```bash
PYTHONPATH=src:. python validation/estimators/iq_learn/sparse_support_guard.py
```

Results: [`iq_learn_sparse_support_guard.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/iq_learn_sparse_support_guard.json).

## Simulation Studies

IQ-Learn appears on both cross-estimator simulation-study pages:
the [bus engine](../../simulation_studies/rust_bus.md) and the
[taxi gridworld](../../simulation_studies/taxi_gridworld.md) pages,
where it is compared against the full structural and IRL rosters.
