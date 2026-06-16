# Simulation Study

UFXP runs on the same low-dimensional action-dependent synthetic cell as the
other structural estimators. The cell has 21 states, 3 actions, known linear
reward, known transitions, and exact policy, value, Q, and Type A, Type B, and
Type C counterfactual oracle objects, so every recovery claim is checked
against the truth.

The full result generator is
[`ufxp_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/ufxp/run.py).
It writes the results file
[`ufxp_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/ufxp.json).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/ufxp/run.py
```

## Design

| Quantity | Value |
| --- | --- |
| States | 21 |
| Actions | 3 |
| Individuals | 2,000 |
| Periods per individual | 80 |
| Observations | 160,000 |
| Discount factor | 0.95 |
| Weighting | optimal |

## Fit Summary

| Quantity | Value |
| --- | --- |
| Converged | True |
| Log-likelihood | -174875.7871 |
| Estimation time | 0.64 seconds |

## Parameter Recovery

| Parameter | Truth | Estimate | Std. error | Error |
| --- | --- | --- | --- | --- |
| action_0_intercept | 0.1000 | 0.0851 | 0.0295 | -0.0149 |
| action_0_progress | 0.5000 | 0.5269 | 0.0360 | +0.0269 |
| action_1_intercept | 0.0000 | -0.0112 | 0.0367 | -0.0112 |
| action_1_progress | -0.2000 | -0.2020 | 0.0525 | -0.0020 |

## Recovery Metrics

| Metric | Value |
| --- | --- |
| Parameter RMSE | 0.0164 |
| Parameter cosine similarity | 0.9991 |
| Reward RMSE | 0.0083 |
| Value RMSE | 0.0170 |
| Q RMSE | 0.0193 |
| Policy TV | 0.0050 |
| Policy KL | 0.000070 |

## Numerical Checks

| Check | Value | Threshold | Status |
| --- | --- | --- | --- |
| converged | true | is true | pass |
| standard_errors_finite | true | is true | pass |
| parameter_cosine | 0.9991 | at least 0.98 | pass |
| parameter_relative_rmse | 0.0598 | at most 0.15 | pass |
| policy_tv | 0.0050 | at most 0.03 | pass |
| value_rmse | 0.0170 | at most 0.10 | pass |
| q_rmse | 0.0193 | at most 0.10 | pass |
| type_a_regret | 0.00016 | at most 0.05 | pass |
| type_b_regret | 0.00030 | at most 0.05 | pass |
| type_c_regret | 0.00008 | at most 0.05 | pass |

UFXP also appears on every page of the
[simulation studies](../../simulation_studies/index.md), where it is compared
against the full structural and IRL rosters on the bus engine, gridworld, and
abstract MDP benchmarks.
