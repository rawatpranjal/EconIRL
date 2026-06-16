# Simulation Study

AIRL-Het runs on a synthetic serialized-content heterogeneous cell with two
latent segments, repeated books per user, and three actions (read/wait/exit).
The cell has known segment-level rewards, transitions, policies, values, Q
functions, and Type A, Type B, and Type C counterfactual oracle objects, so
every recovery metric is checked against the truth.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/aairl/run.py).
It writes the results file
[`aairl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/aairl.json).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/aairl/run.py
```

## Design

| Quantity | Value |
| --- | --- |
| Segments | 2 |
| States | 61 (60 regular + 1 absorbing) |
| Actions | 3 (read / wait / exit) |
| State dimension | 18 |
| Reward features | 20 |
| Individuals | 800 |
| Books per user | 4 |
| Periods per individual | 16 |
| Observations | 12,368 |
| Discount factor | 0.92 |
| Scale parameter | 0.85 |
| Reward mode | action-dependent |
| Initialization | behavioral_anchor |

## Fit Summary

| Quantity | Value |
| --- | --- |
| Converged | true |
| Convergence message | EM converged |
| EM iterations | 2 |
| Log-likelihood | -11614.763 |
| Estimation time | 30.7 seconds |

## Segment Recovery

| Metric | Value |
| --- | --- |
| Segment assignment accuracy | 0.895 |
| Segment prior L1 | 0.0435 |
| Max segment reward NRMSE | 0.265 |
| Max segment policy TV | 0.0591 |
| Max segment value NRMSE | 0.142 |
| Max segment Q NRMSE | 0.211 |

Metrics are reported over both segments; the max is taken across segments after
aligning estimated segments to true segments by minimum-cost permutation.

## Numerical Checks

| Check | Value | Threshold | Status |
| --- | --- | --- | --- |
| converged | true | is true | pass |
| segment_prior_l1 | 0.0435 | at most 0.35 | pass |
| segment_assignment_accuracy | 0.895 | at least 0.70 | pass |
| max_segment_reward_normalized_rmse | 0.265 | at most 0.30 | pass |
| max_segment_policy_tv | 0.0591 | at most 0.12 | pass |
| max_segment_value_normalized_rmse | 0.142 | at most 0.30 | pass |
| max_segment_q_normalized_rmse | 0.211 | at most 0.30 | pass |
| type_a_max_regret | 0.0145 | at most 0.12 | pass |
| type_b_max_regret | 0.1189 | at most 0.12 | pass |
| type_c_max_regret | 0.00687 | at most 0.12 | pass |

The Type B regret (0.1189) is close to its threshold (0.12). This is expected
for the heterogeneous transition-change counterfactual: each segment's reward
must adapt to a different new dynamic, and the adversarial reward carries more
approximation error under that intervention than under reward shifts or action
removals. The result still passes.
