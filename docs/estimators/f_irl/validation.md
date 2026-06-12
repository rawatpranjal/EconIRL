# Simulation Study

f-IRL is reported on two synthetic cells. The primary cell is the paper-faithful
state-marginal configuration: a fully specified data-generating process with
known reward, transitions, policy, value, Q function, and Type A, B, and C
counterfactual oracle objects. The action-dependent cell is a diagnostic
negative control that fails the reward-range check and is not treated as
structural recovery evidence.

The full result generator is
[`run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/f_irl/run.py).
It writes the machine-readable results file
[`f_irl.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/f_irl.json).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/f_irl/run.py
```

## Design

| Quantity | Primary cell | Diagnostic cell |
| --- | --- | --- |
| Cell ID | `f_irl_paper_state_marginal` | `canonical_low_action` |
| Role | Paper-faithful (state marginal) | Action-dependent negative control |
| States | 8 | 21 |
| Actions | 3 | 3 |
| Individuals | 1,000 | 2,000 |
| Periods per individual | 100 | 80 |
| Observations | 100,000 | 160,000 |
| Discount factor | 0.95 | 0.95 |
| f-divergence | Forward KL | Forward KL |
| Marginal scope | State | State-action |
| Reward scope | State | State-action |
| Iterations | 250 | 500 |

## Fit Summary

| Quantity | Primary cell | Diagnostic cell |
| --- | --- | --- |
| Converged | true | true |
| Log-likelihood | -103951.53 | -175777.97 |
| Estimation time | 27.6 seconds | 73.5 seconds |
| Occupancy L1 | 0.000260 | 0.1609 |
| Reward range | 1.292 | 0.000 |

## Recovery Metrics (Primary Cell)

| Metric | Value |
| --- | --- |
| State marginal L1 | 0.000260 |
| Reward NRMSE | 0.199 |
| Value NRMSE | 0.130 |
| Q NRMSE | 0.100 |
| Policy TV | 0.0121 |

Normalized RMSE is computed relative to the range of the oracle quantity. This
metric is used in place of absolute RMSE because the reward is identified only
up to scale in the IRL setting.

## Checks (Primary Cell)

| Check | Value | Threshold | Status |
| --- | --- | --- | --- |
| converged | true | is true | pass |
| state_marginal_l1 | 0.000260 | at most 0.08 | pass |
| reward_range | 1.292 | at least 0.001 | pass |
| reward_normalized_rmse | 0.199 | at most 0.3 | pass |
| policy_tv | 0.0121 | at most 0.08 | pass |
| value_normalized_rmse | 0.130 | at most 0.3 | pass |
| q_normalized_rmse | 0.100 | at most 0.3 | pass |
| type_a_regret | 0.00708 | at most 0.05 | pass |
| type_b_regret | 0.01236 | at most 0.05 | pass |
| type_c_regret | 0.00273 | at most 0.05 | pass |

## Diagnostic Cell

The action-dependent cell (`canonical_low_action`) fails the reward-range check
with a reward range of 0.000, indicating a flat-reward solution. Policy TV is
0.141 and occupancy L1 is 0.161. This cell is retained as a diagnostic to show
that f-IRL with action-dependent marginal matching does not recover a
non-trivial reward on the standard DDC benchmark. It is not structural recovery
evidence.

f-IRL also appears on the bus engine and gridworld pages of the
[simulation studies](../../simulation_studies/index.md), where it is compared
against the full structural and IRL rosters.
