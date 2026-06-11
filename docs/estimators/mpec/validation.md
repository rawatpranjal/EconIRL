# Simulation Study

We run MPEC on the `canonical_low_action` known-truth cell, the same
low-dimensional action-dependent structural benchmark used for NFXP and CCP.
The simulation asks whether the constrained-likelihood formulation can recover
the structural reward and counterfactual behavior while satisfying the Bellman
equality constraint. Real bus data cannot answer that question because the
true reward, policy, value function, Q function, and counterfactual oracles are
not observed.

These results are not hand-entered examples. They come from the known-truth
simulation harness. The harness fixes the transition law, action-dependent
reward features, and reward weights before generating the finite panel. Those
objects define the true reward, policy, value function, Q function, and
counterfactual oracles that are held back for evaluation.

The full result generator is
[`mpec_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/mpec/mpec_run.py).
It writes the rendered table source
[`mpec_results.tex`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/mpec/mpec_results.tex)
and the machine-readable artifact
[`mpec_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/mpec/mpec_results.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/mpec/mpec_run.py
```

The core harness flow is:

```python
from pathlib import Path

from experiments.known_truth import run_cell_estimator

run_cell_estimator(
    "canonical_low_action",
    "MPEC",
    Path("/tmp/econirl_mpec_primer_known_truth"),
    smoke=False,
)
```

Read the tables as a sequence. The design table states the known-truth cell.
The fit summary reports convergence, constrained-optimizer iterations, and the
final Bellman constraint violation. Parameter recovery compares estimated
reward parameters to truth. Recovery metrics compare the recovered reward,
value, Q function, and policy to oracle objects. Threshold checks are the
numerical cutoffs recorded by the harness.

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

The simulation DGP has action-dependent reward features and an exit action
that anchors the reward level.

## Fit Summary

| Quantity | Value |
| --- | ---: |
| Converged | true |
| SQP iterations | 19 |
| Log likelihood | -174875.7719 |
| Estimation time | 1.92 seconds |
| Solver | slsqp |
| SciPy success | true |
| Final Bellman constraint violation | 7.72e-12 |
| Standard errors finite | true |

The constrained optimizer satisfies the Bellman equality threshold by several orders
of magnitude. The simulation result still depends on recovery checks, not on
the constraint diagnostic alone.

## Parameter Recovery

| Parameter | Truth | Estimate | SE | Error |
| --- | ---: | ---: | ---: | ---: |
| `action_0_intercept` | 0.100000 | 0.083894 | 0.029336 | -0.016106 |
| `action_0_progress` | 0.500000 | 0.528522 | 0.035890 | 0.028522 |
| `action_1_intercept` | 0.000000 | -0.014461 | 0.036733 | -0.014461 |
| `action_1_progress` | -0.200000 | -0.200511 | 0.052502 | -0.000511 |

## Recovery Metrics

| Metric | Value |
| --- | ---: |
| Parameter RMSE | 0.017905 |
| Parameter relative RMSE | 0.065378 |
| Parameter cosine similarity | 0.998867 |
| Reward RMSE | 0.009694 |
| Value RMSE | 0.019445 |
| Q RMSE | 0.022437 |
| Policy KL | 9.21e-5 |
| Policy total variation | 0.005697 |
| Policy max state L1 | 0.018905 |

## Threshold Checks

| Check | Threshold | Value | Status |
| --- | --- | ---: | --- |
| converged | true | true | pass |
| Bellman constraint violation | at most 0.000001 | 0.000000 | pass |
| Standard errors finite | true | true | pass |
| Parameter cosine | at least 0.98 | 0.998867 | pass |
| Parameter relative RMSE | at most 0.15 | 0.065378 | pass |
| Policy total variation | at most 0.03 | 0.005697 | pass |
| Value RMSE | at most 0.10 | 0.019445 | pass |
| Q RMSE | at most 0.10 | 0.022437 | pass |
| Type A regret | at most 0.05 | 0.000213 | pass |
| Type B regret | at most 0.05 | 0.000362 | pass |
| Type C regret | at most 0.05 | 0.000086 | pass |

The estimates are not exactly equal to truth because the panel is finite. The
study reports recovery within the listed tolerances in the frozen known-truth
cell.

## Counterfactual Recovery

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.005109 | 7.56e-5 | 0.000238 | 0.000213 |
| Type B | 0.005457 | 8.20e-5 | 0.000363 | 0.000362 |
| Type C | 0.003549 | 3.56e-5 | 0.000114 | 0.000086 |
