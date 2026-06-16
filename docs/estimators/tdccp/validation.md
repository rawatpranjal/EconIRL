# Simulation Study

The TD-CCP simulation study uses the `shapeshifter_encoded_state_locally_robust`
synthetic cell: encoded state features, a finite linear reward, and known oracle
objects for every comparison. The simulation asks whether the cross-fitted,
locally robust semigradient path recovers the reward parameters, the implied
dynamic objects, counterfactual behavior, and valid standard errors.

The result generator is
[`tdccp_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/tdccp/run.py).
It writes the results file
[`tdccp_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp.json).

```bash
PYTHONPATH=src:. python validation/estimators/tdccp/run.py --quiet-progress
```

## Design

| Quantity | Value |
| --- | ---: |
| States | 81 |
| Actions | 3 |
| State dimension | 2 |
| Reward parameters | 6 |
| Discount factor | 0.95 |
| Simulated individuals | 2,000 |
| Periods per individual | 60 |
| Observations | 120,000 |
| Reward form | Linear in encoded-state basis |
| Action normalization | Action 0 fixed to zero |

This is not a raw neural reward-recovery test. It is a structural parameter
test with encoded state features and stochastic transitions.

## Fit Summary

The estimator converged on both preliminary folds and both final robust folds.
Known transition tensors were not used to estimate `theta`; they were supplied
after fitting so the simulation harness could compare recovered policies,
values, Q functions, and counterfactual decisions with oracle solutions.

| Quantity | Value |
| --- | ---: |
| Converged | true |
| Outer iterations | 121 |
| Log likelihood | -99831.3015 |
| Estimation time | 10.71 seconds |
| Standard errors | Locally robust |
| Covariance unit | Individual |
| Max standard error | 0.034212 |
| Max locally robust moment norm | 7.91e-06 |
| Max correction residual norm | 0.002610 |
| Max correction residual RMS | 0.292447 |
| Max absolute correction residual | 1.022835 |
| Preliminary optimizer status | 2/2 folds converged |
| Preliminary projected-gradient max | 2.00e-07 |
| Final robust optimizer status | 2/2 folds converged |
| Method | `semigradient` |
| Basis | `encoded`, degree 2 |
| CCP model | `logit`, degree 2 |
| Cross-fitting | true |
| Robust SE | true |

## Repeated-Seed SE Coverage

The standard-error check re-simulates the same encoded-state design 25 times
with 300 individuals and 35 periods per replication. Each replication uses the
same cross-fitted, locally robust TD-CCP settings and individual-clustered
covariances.

The 25-replication run is the current RTD simulation receipt because it is
small enough to regenerate routinely. A paper-final audit should rerun the same command
with `--mc-replications 100` under a CPU budget before claiming final Monte
Carlo precision.

| Quantity | Value |
| --- | ---: |
| Replications completed | 25/25 |
| Overall 95% CI coverage | 0.900 |
| Mean parameter relative RMSE | 0.125000 |
| Strict threshold-passing replications | 15/25 |
| Final robust optimizer success | 25/25 |
| Preliminary optimizer success | 25/25 |
| Preliminary optimizer stationarity | 25/25 |
| Preliminary projected-gradient max | 6.88e-07 |
| Max locally robust moment norm | 2.20e-05 |
| Max correction residual norm | 0.035225 |
| Max correction residual RMS | 0.312355 |
| Max absolute correction residual | 1.137664 |

| Parameter | Bias | RMSE | Empirical SD | Mean SE | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `action_1_intercept` | 0.007756 | 0.079567 | 0.080821 | 0.068993 | 0.960 |
| `action_1_x0` | -0.055920 | 0.094582 | 0.077853 | 0.089330 | 0.920 |
| `action_1_x1` | 0.044520 | 0.118374 | 0.111945 | 0.075693 | 0.880 |
| `action_2_intercept` | -0.058485 | 0.121556 | 0.108759 | 0.091918 | 0.920 |
| `action_2_x0` | 0.025811 | 0.136120 | 0.136406 | 0.115838 | 0.920 |
| `action_2_x1` | 0.075985 | 0.144508 | 0.125452 | 0.107827 | 0.800 |

## Recovery Metrics

These metrics compare the fitted structural objects with the true simulated values. They
are the main evidence that the estimator recovers the finite reward target and
the implied dynamic decisions in this design.

| Metric | Value |
| --- | ---: |
| Parameter cosine similarity | 0.998658 |
| Parameter relative RMSE | 0.059104 |
| Reward normalized RMSE | 0.025307 |
| Policy total variation | 0.004658 |
| Value normalized RMSE | 0.000600 |
| Q normalized RMSE | 0.000733 |

## Counterfactual Metrics

The counterfactual checks are run after fitting, using supplied transition
environments. They test whether the recovered reward leads to the same
counterfactual decisions as the oracle reward.

| Counterfactual | Policy TV | Value RMSE | Regret |
| --- | ---: | ---: | ---: |
| Type A | 0.004643 | 0.001885 | 0.001883 |
| Type B | 0.004840 | 0.001867 | 0.001864 |
| Type C | 0.006953 | 0.003208 | 0.003200 |

## Numerical Checks

| Check | Threshold | Value | Status |
| --- | --- | ---: | --- |
| Converged | true | true | pass |
| Locally robust path | true | true | pass |
| Finite positive standard errors | true | true | pass |
| Moment norm | at most 1e-4 | 7.91e-06 | pass |
| Covariance minimum eigenvalue | at least -1e-10 | 2.27e-05 | pass |
| Parameter cosine | at least 0.99 | 0.998658 | pass |
| Parameter relative RMSE | at most 0.15 | 0.059104 | pass |
| Reward normalized RMSE | at most 0.08 | 0.025307 | pass |
| Policy total variation | at most 0.03 | 0.004658 | pass |
| Value normalized RMSE | at most 0.10 | 0.000600 | pass |
| Q normalized RMSE | at most 0.10 | 0.000733 | pass |
| Type A regret | at most 0.05 | 0.001883 | pass |
| Type B regret | at most 0.05 | 0.001864 | pass |
| Type C regret | at most 0.05 | 0.003200 | pass |

## Diagnostic Records

The `canonical_low_action` cell remains a simple sanity check and passes 10/10
checks. The `canonical_high_action` cell remains a diagnostic stress test and
currently fails 10/10 checks. The raw neural-reward diagnostic passes 5/8 checks
and fails reward, value, and Q recovery. It has no finite true reward
parameter vector, so it is not part of the primary finite-parameter study.

TD-CCP also appears on the [bus engine](../../simulation_studies/rust_bus.md)
and [abstract MDP sanity](../../simulation_studies/abstract_mdp_1_sanity.md)
pages of the simulation studies, where it is compared against the full structural
and IRL rosters.
