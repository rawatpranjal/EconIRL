# Validation

CCP is certified on the `canonical_low_action` known-truth cell. This is the
same low-dimensional action-dependent structural benchmark used for NFXP.

These results are not hand-entered examples. They come from the known-truth
validation harness, where the reward, transition law, optimal policy, value
function, Q function, and counterfactual oracle objects are all known before
estimation starts. The CCP page checks whether the estimator can recover those
known objects from a finite simulated panel.

The full result generator is
[`ccp_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/ccp/ccp_run.py).
It writes the rendered table source
[`ccp_results.tex`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/ccp/ccp_results.tex)
and the machine-readable artifact
[`ccp_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/ccp/ccp_results.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/ccp/ccp_run.py
```

The core harness flow is:

```python
from experiments.known_truth import (
    build_known_truth_dgp,
    get_cell,
    run_estimator,
    simulate_known_truth_panel,
)

cell = get_cell("canonical_low_action")
dgp = build_known_truth_dgp(cell.dgp_config)
panel = simulate_known_truth_panel(dgp, cell.simulation_config)
main_run = run_estimator("CCP", dgp, panel, smoke=False)
```

Read the tables as a sequence. The design table states the known-truth cell.
The fit summary reports how the estimator ran. Parameter recovery compares
estimated reward parameters to truth. Recovery metrics compare the recovered
reward, value, Q function, and policy to oracle objects. Hard gates are the
pass/fail release criteria.

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
| NPL delta criterion met | true |
| NPL iterations completed | 8 |
| Log likelihood | -174875.9375 |
| Estimation time | 16.31 seconds |
| Standard errors finite | true |

The fitted run met the NPL delta criterion before the maximum of ten policy
updates. The hard gate checks that at least five NPL iterations completed and
that recovery metrics pass.

## Parameter Recovery

| Parameter | Truth | Estimate | SE | Error |
| --- | ---: | ---: | ---: | ---: |
| `action_0_intercept` | 0.100000 | 0.083896 | 0.009450 | -0.016104 |
| `action_0_progress` | 0.500000 | 0.528517 | 0.009222 | 0.028517 |
| `action_1_intercept` | 0.000000 | -0.014458 | 0.026148 | -0.014458 |
| `action_1_progress` | -0.200000 | -0.200517 | 0.020787 | -0.000517 |

## Recovery Metrics

| Metric | Value |
| --- | ---: |
| Parameter RMSE | 0.017902 |
| Parameter relative RMSE | 0.065367 |
| Parameter cosine similarity | 0.998867 |
| Reward RMSE | 0.009694 |
| Value RMSE | 0.019436 |
| Q RMSE | 0.022430 |
| Policy KL | 9.21e-5 |
| Policy total variation | 0.005697 |
| Policy max state L1 | 0.018902 |

In this finite sample, the one-step estimate already recovers the policy well
because empirical CCP support is strong. The certified release claim uses the
NPL run with a ten-iteration cap because it is the structural CCP counterpart
to the NFXP fixed point.

## Hard Gates

| Gate | Threshold | Value | Status |
| --- | --- | ---: | --- |
| NPL iterations | at least 5 | 8 | pass |
| Standard errors finite | true | true | pass |
| Parameter cosine | at least 0.98 | 0.998867 | pass |
| Parameter relative RMSE | at most 0.15 | 0.065367 | pass |
| Policy total variation | at most 0.03 | 0.005697 | pass |
| Value RMSE | at most 0.10 | 0.019436 | pass |
| Q RMSE | at most 0.10 | 0.022430 | pass |
| Type A regret | at most 0.05 | 0.000213 | pass |
| Type B regret | at most 0.05 | 0.000362 | pass |
| Type C regret | at most 0.05 | 0.000086 | pass |

The estimates are not exactly equal to truth because the panel is finite. The
release claim is recovery within strict tolerances in the frozen known-truth
cell.
