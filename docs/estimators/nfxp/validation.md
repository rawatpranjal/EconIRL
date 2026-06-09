# Validation

NFXP is certified on the `canonical_low_action` known-truth cell. This cell is
small enough to solve exactly and large enough to expose rank, support, and
normalization issues.

These results are not hand-entered examples. They come from the known-truth
validation harness, where the reward, transition law, optimal policy, value
function, Q function, and counterfactual oracle objects are all known before
estimation starts. The NFXP page checks whether the estimator can recover those
known objects from a finite simulated panel.

The full result generator is
[`nfxp_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nfxp/nfxp_run.py).
It writes the rendered table source
[`nfxp_results.tex`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nfxp/nfxp_results.tex)
and the machine-readable artifact
[`nfxp_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nfxp/nfxp_results.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/nfxp/nfxp_run.py
```

The core harness flow is:

```python
from pathlib import Path

from experiments.known_truth import run_cell_estimator

run_cell_estimator(
    "canonical_low_action",
    "NFXP",
    Path("/tmp/econirl_nfxp_primer_known_truth"),
    smoke=False,
)
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
| Converged | true |
| Outer iterations | 12 |
| Log likelihood | -174875.7719 |
| Estimation time | 5.41 seconds |
| Function evaluations | 80 |
| Total inner iterations | 11259 |
| Final inner iterations | 139 |
| Inner solver | hybrid |
| Outer optimizer | BHHH |

## Parameter Recovery

| Parameter | Truth | Estimate | SE | Error |
| --- | ---: | ---: | ---: | ---: |
| `action_0_intercept` | 0.100000 | 0.083894 | 0.029335 | -0.016106 |
| `action_0_progress` | 0.500000 | 0.528522 | 0.035889 | 0.028522 |
| `action_1_intercept` | 0.000000 | -0.014461 | 0.036733 | -0.014461 |
| `action_1_progress` | -0.200000 | -0.200511 | 0.052502 | -0.000511 |

## Recovery Metrics

| Metric | Value |
| --- | ---: |
| Parameter RMSE | 0.017904 |
| Parameter relative RMSE | 0.065378 |
| Parameter cosine similarity | 0.998867 |
| Reward RMSE | 0.009694 |
| Value RMSE | 0.019445 |
| Q RMSE | 0.022438 |
| Policy KL | 9.21e-5 |
| Policy total variation | 0.005697 |
| Policy max state L1 | 0.018905 |

## Hard Gates

| Gate | Threshold | Value | Status |
| --- | --- | ---: | --- |
| converged | true | true | pass |
| parameter cosine | at least 0.98 | 0.998867 | pass |
| parameter relative RMSE | at most 0.15 | 0.065378 | pass |
| policy total variation | at most 0.03 | 0.005697 | pass |
| value RMSE | at most 0.10 | 0.019445 | pass |
| Type A regret | at most 0.05 | 0.000213 | pass |
| Type B regret | at most 0.05 | 0.000362 | pass |
| Type C regret | at most 0.05 | 0.000086 | pass |

The estimates are not exactly equal to truth because the panel is finite. The
release claim is recovery within strict tolerances in the frozen known-truth
cell.
