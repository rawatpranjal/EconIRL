# Validation

CCP is certified on the `canonical_low_action` known-truth cell. This is the
same low-dimensional action-dependent structural benchmark used for NFXP.

## Design

| Quantity | Value |
| --- | ---: |
| Regular states | 20 |
| Absorbing states | 1 |
| Total states | 21 |
| Actions | 3 |
| Exit action | 2 |
| Discount factor | 0.95 |
| Shock scale | 1.0 |
| Simulated individuals | 2,000 |
| Periods per individual | 80 |
| Observations | 160,000 |

The validation DGP has action-dependent reward features and an exit action
that anchors the reward level.

## Fit Summary

| Quantity | Value |
| --- | ---: |
| NPL delta criterion met | false |
| NPL iterations completed | 10 |
| Log likelihood | -174875.9531 |
| Estimation time | 20.77 seconds |
| Standard errors finite | true |

The convergence flag is false because the hard validation uses a fixed K=10
NPL run. The hard gate checks that at least five NPL iterations completed and
that recovery metrics pass.

## Parameter Recovery

| Parameter | Truth | Estimate | SE | Error |
| --- | ---: | ---: | ---: | ---: |
| `action_0_intercept` | 0.100000 | 0.083895 | 0.035995 | -0.016105 |
| `action_0_progress` | 0.500000 | 0.528519 | 0.039855 | 0.028519 |
| `action_1_intercept` | 0.000000 | -0.014460 | 0.117503 | -0.014460 |
| `action_1_progress` | -0.200000 | -0.200513 | 0.158931 | -0.000513 |

## Recovery Metrics

| Metric | Value |
| --- | ---: |
| Parameter RMSE | 0.017903 |
| Parameter relative RMSE | 0.065372 |
| Parameter cosine similarity | 0.998867 |
| Reward RMSE | 0.009694 |
| Value RMSE | 0.019438 |
| Q RMSE | 0.022432 |
| Policy KL | 9.21e-5 |
| Policy total variation | 0.005697 |
| Policy max state L1 | 0.018905 |

## NPL Comparison

| Estimator | Iterations | Relative RMSE | Cosine | Policy TV | Value RMSE | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hotz-Miller K=1 | 1 | 0.058259 | 0.999211 | 0.005539 | 0.016469 | 1.57 seconds |
| NPL K=3 | 3 | 0.065298 | 0.998871 | 0.005694 | 0.019416 | 4.53 seconds |
| NPL K=10 | 10 | 0.065372 | 0.998867 | 0.005697 | 0.019438 | 20.77 seconds |

In this finite sample, the one-step estimate already recovers the policy well
because empirical CCP support is strong. The certified release claim uses the
K=10 NPL row because it is the structural CCP counterpart to the NFXP fixed
point.

## Hard Gates

| Gate | Threshold | Value | Status |
| --- | --- | ---: | --- |
| NPL iterations | at least 5 | 10 | pass |
| Standard errors finite | true | true | pass |
| Parameter cosine | at least 0.98 | 0.998867 | pass |
| Parameter relative RMSE | at most 0.15 | 0.065372 | pass |
| Policy total variation | at most 0.03 | 0.005697 | pass |
| Value RMSE | at most 0.10 | 0.019438 | pass |
| Q RMSE | at most 0.10 | 0.022432 | pass |
| Type A regret | at most 0.05 | 0.000213 | pass |
| Type B regret | at most 0.05 | 0.000362 | pass |
| Type C regret | at most 0.05 | 0.000086 | pass |

The estimates are not exactly equal to truth because the panel is finite. The
release claim is recovery within strict tolerances in the frozen known-truth
cell.
