# Simulation Study

GLADIUS is evaluated on two related high-dimensional-state synthetic cells.
Each cell has 21 discrete states paired with a high-dimensional encoded state-
feature vector, a low-dimensional action-dependent reward basis, and an anchor
action whose rewards are known. All recovery claims are compared against the
truth from the data-generating process.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/gladius/run.py).
It writes two machine-readable results files:
[`gladius.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius.json)
(primary cell) and
[`gladius_scaled.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/gladius_scaled.json)
(scaled cell).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/gladius/run.py
```

## Design

| Quantity | Primary cell | Scaled cell |
| --- | --- | --- |
| States | 21 | 21 |
| State-feature dimension | 64 | 128 |
| Reward features | 4 | 4 |
| Actions | 3 | 3 |
| Individuals | 1,000 | 1,000 |
| Periods per individual | 100 | 100 |
| Observations | 100,000 | 100,000 |
| Transition noise | 0.05 | 0.05 |
| Discount factor | 0.95 | 0.95 |
| Anchor Bellman mode | anchor_moment | anchor_moment |

## Fit Summary

| Quantity | Primary cell | Scaled cell |
| --- | --- | --- |
| Converged | true | true |
| Iterations (epochs) | 235 | 161 |
| Log-likelihood | -109186.558 | -109344.115 |
| Final loss | 0.5460 | 0.5469 |
| Estimation time | 296.7 seconds | 175.0 seconds |

## Support Diagnostics

Both cells share the same structural support:

| Check | Value | Status |
| --- | --- | --- |
| Feature rank | 4 / 4 | pass |
| Feature condition number | 4.376 | pass |
| Observed states | 21 / 21 | pass |
| State-action coverage | 1.000 | pass |
| Single-action states | 0 | pass |
| Anchor valid | true | pass |
| Max transition row error | 2.4e-08 | pass |

## Recovery Metrics

| Metric | Primary cell | Scaled cell |
| --- | --- | --- |
| Parameter cosine similarity | 0.9752 | 0.9857 |
| Parameter relative RMSE | 0.4855 | 0.2348 |
| Projected reward NRMSE | 0.1980 | 0.1344 |
| Raw Bellman reward NRMSE | 0.5715 | 0.5897 |
| Policy TV | 0.0369 | 0.0381 |
| Value NRMSE | 0.4196 | 0.4945 |
| Q NRMSE | 0.2351 | 0.2218 |

## Checks

| Check | Primary cell | Threshold | Status | Scaled cell | Threshold | Status |
| --- | --- | --- | --- | --- | --- | --- |
| converged | true | is true | pass | true | is true | pass |
| final_loss | 0.5460 | at most 2.0 | pass | 0.5469 | at most 2.0 | pass |
| parameter_cosine | 0.9752 | at least 0.90 | pass | 0.9857 | at least 0.90 | pass |
| parameter_relative_rmse | 0.4855 | at most 0.50 | pass | 0.2348 | at most 0.50 | pass |
| raw_bellman_reward_nrmse | 0.5715 | at most 0.30 | **fail** | 0.5897 | at most 0.30 | **fail** |
| projected_reward_nrmse | 0.1980 | at most 0.30 | pass | 0.1344 | at most 0.30 | pass |
| policy_tv | 0.0369 | at most 0.12 | pass | 0.0381 | at most 0.12 | pass |
| value_nrmse | 0.4196 | at most 0.30 | **fail** | 0.4945 | at most 0.30 | **fail** |
| q_nrmse | 0.2351 | at most 0.30 | pass | 0.2218 | at most 0.30 | pass |
| type_a_regret | 0.00854 | at most 0.12 | pass | 0.00274 | at most 0.12 | pass |
| type_b_regret | 0.0529 | at most 0.12 | pass | 0.00742 | at most 0.12 | pass |
| type_c_regret | 0.00852 | at most 0.12 | pass | 0.000493 | at most 0.12 | pass |

The two checks that fail - raw Bellman reward and value NRMSE - are what
prevent GLADIUS from reaching structural counterfactual validity.
Projected reward, policy, and Q recover reasonably; the absolute reward level
implied by the raw Bellman equation does not. Counterfactual regret is small
on both cells, especially the scaled variant, but low regret alone is not
sufficient for structural validity when the upstream reward checks fail.

GLADIUS also appears in the
[bus engine](../../simulation_studies/rust_bus.md) and
[taxi gridworld](../../simulation_studies/taxi_gridworld.md)
simulation studies, where it is compared against the full structural and
behavioral rosters.
