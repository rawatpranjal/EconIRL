# Validation

TD-CCP is certified on `shapeshifter_encoded_state_locally_robust`. This is
the paper-faithful hard case: two-dimensional encoded state coordinates enter
a finite linear structural reward, action 0 is normalized to zero, and
Algorithm 2 locally robust standard errors are computed.

These results are not hand-entered examples. They come from the known-truth
validation harness and the TD-CCP primer generator. The validation target is
recovery of structural parameters, rewards, policies, values, Q functions, and
Type A/B/C counterfactual decisions in the finite-theta setting, with
fold-specific zeta moments, individual-clustered sandwich covariance, and
repeated-seed standard-error coverage recorded in the artifact.

The full result generator is
[`tdccp_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/tdccp/tdccp_run.py).
It writes the rendered table source
[`tdccp_results.tex`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/tdccp/tdccp_results.tex)
and the machine-readable artifact
[`tdccp_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/tdccp/tdccp_results.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/tdccp/tdccp_run.py --quiet-progress
```

The hard-case harness flow is:

```python
from papers.econirl_package.primers.tdccp.tdccp_run import (
    build_paper_hard_case_dgp,
    evaluate_paper_hard_case_summary,
    tdccp_paper_hard_case_gates,
)
```

Read the tables as a sequence. The design table states the known-truth cell.
The fit summary reports how the estimator ran. Recovery metrics compare the
estimated structural objects to oracle objects. Hard gates are the reported
thresholds.

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

The hard case is deliberately not raw neural reward recovery. It keeps a
finite structural parameter vector and uses encoded state features with
stochastic transitions.

## Fit Summary

| Quantity | Value |
| --- | ---: |
| Converged | true |
| Outer iterations | 121 |
| Log likelihood | -99831.3015 |
| Estimation time | 10.71 seconds |
| Standard errors | Algorithm 2 locally robust |
| Covariance unit | Individual |
| Max standard error | 0.034212 |
| Max zeta moment norm | 7.91e-06 |
| Max lambda fixed-point residual norm | 0.002610 |
| Max lambda fixed-point residual RMS | 0.292447 |
| Max absolute lambda fixed-point residual | 1.022835 |
| Preliminary plug-in optimizer status | 2/2 folds converged |
| Preliminary projected-gradient max | 2.00e-07 |
| Final robust zeta optimizer status | 2/2 folds converged |
| Method | `semigradient` |
| Basis | `encoded`, degree 2 |
| CCP model | `logit`, degree 2 |
| Cross-fitting | true |
| Robust SE | true |

Known transitions are not used to estimate `theta`. They enter after
estimation so the validation harness can evaluate policies, values, Q
functions, and counterfactual decisions against oracle solutions.

The preliminary plug-in folds and final robust zeta folds all satisfy the
optimizer stationarity checks. The artifact also records projected-gradient
norms and lambda fixed-point residual summaries so this claim is auditable.

## Repeated-Seed SE Coverage

The standard-error check re-simulates the same encoded-state DGP 25 times with
300 individuals and 35 periods per replication. It uses the same TD-CCP
Algorithm 2 settings, with individual-clustered fold covariances.

The 25-replication run is the current RTD evidence because it is light enough
to regenerate routinely. A paper-final audit should rerun the same command
with `--mc-replications 100` under a CPU budget before claiming final Monte
Carlo precision.

| Quantity | Value |
| --- | ---: |
| Replications completed | 25/25 |
| Overall 95% CI coverage | 0.900 |
| Mean parameter relative RMSE | 0.125000 |
| Strict gate-passing replications | 15/25 |
| Final robust zeta optimizer success | 25/25 |
| Preliminary plug-in optimizer success | 25/25 |
| Preliminary plug-in optimizer stationarity | 25/25 |
| Preliminary projected-gradient max | 6.88e-07 |
| Max zeta moment norm | 2.20e-05 |
| Max lambda fixed-point residual norm | 0.035225 |
| Max lambda fixed-point residual RMS | 0.312355 |
| Max absolute lambda fixed-point residual | 1.137664 |

| Parameter | Bias | RMSE | Empirical SD | Mean SE | 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `action_1_intercept` | 0.007756 | 0.079567 | 0.080821 | 0.068993 | 0.960 |
| `action_1_x0` | -0.055920 | 0.094582 | 0.077853 | 0.089330 | 0.920 |
| `action_1_x1` | 0.044520 | 0.118374 | 0.111945 | 0.075693 | 0.880 |
| `action_2_intercept` | -0.058485 | 0.121556 | 0.108759 | 0.091918 | 0.920 |
| `action_2_x0` | 0.025811 | 0.136120 | 0.136406 | 0.115838 | 0.920 |
| `action_2_x1` | 0.075985 | 0.144508 | 0.125452 | 0.107827 | 0.800 |

## Recovery Metrics

| Metric | Value |
| --- | ---: |
| Parameter cosine similarity | 0.998658 |
| Parameter relative RMSE | 0.059104 |
| Reward normalized RMSE | 0.025307 |
| Policy total variation | 0.004658 |
| Value normalized RMSE | 0.000600 |
| Q normalized RMSE | 0.000733 |

## Counterfactual Metrics

| Counterfactual | Policy TV | Value RMSE | Regret |
| --- | ---: | ---: | ---: |
| Type A | 0.004643 | 0.001885 | 0.001883 |
| Type B | 0.004840 | 0.001867 | 0.001864 |
| Type C | 0.006953 | 0.003208 | 0.003200 |

## Hard Gates

| Gate | Threshold | Value | Status |
| --- | --- | ---: | --- |
| Converged | true | true | pass |
| Algorithm 2 locally robust path | true | true | pass |
| Finite positive standard errors | true | true | pass |
| Zeta moment norm | at most 1e-4 | 7.91e-06 | pass |
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
gates. The `canonical_high_action` cell remains a diagnostic stress test and
currently fails 10/10 gates. The raw neural-reward diagnostic passes 5/8 gates
and fails reward, value, and Q recovery. It has no finite true `theta`, so it
is not part of the certified claim.
