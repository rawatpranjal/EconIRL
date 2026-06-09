# Validation

SEES is certified on the `canonical_high_action` known-truth cell. The
low-dimensional `canonical_low_action` cell remains in the artifact as a
sanity check for the historical state-index basis. The primary cell uses
encoded states and a richer reward-feature basis, so it tests the SEES path
where the estimator is meant to matter.

These results are not hand-entered examples. They come from the known-truth
validation harness, where the reward, transition law, optimal policy, value
function, Q function, and counterfactual oracle objects are all known before
estimation starts.

The full result generator is
[`sees_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/sees/sees_run.py).
It writes the rendered table source
[`sees_results.tex`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/sees/sees_results.tex)
and the machine-readable artifact
[`sees_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/sees/sees_results.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/sees/sees_run.py --quiet-progress
```

The core harness flow is:

```python
from experiments.known_truth import (
    build_known_truth_dgp,
    get_cell,
    run_estimator,
    simulate_known_truth_panel,
)

cell = get_cell("canonical_high_action")
dgp = build_known_truth_dgp(cell.dgp_config)
panel = simulate_known_truth_panel(dgp, cell.simulation_config)
run = run_estimator("SEES", dgp, panel, smoke=False, enforce_gates=True)
```

## Read the Tables

Read the tables as a sequence. The design rows state which known-truth cells
were run. The fit summary reports the optimizer flag, Bellman residuals,
likelihood, and run time. Recovery metrics compare the estimated structural
object to oracle reward, policy, value, and Q objects. Hard gates are the
pass/fail release criteria.

The optimizer flag is reported exactly as returned by L-BFGS-B. The SEES
release gate is not the optimizer flag alone; it is the Bellman residual,
finite standard errors, known-truth recovery, and counterfactual regret.

## Design

| Quantity | Low-dimensional | High-dimensional primary |
| --- | ---: | ---: |
| Total states | 21 | 81 |
| State dimension | 2 | 16 |
| Actions | 3 | 3 |
| Reward parameters | 4 | 32 |
| Simulated individuals | 2,000 | 2,000 |
| Periods per individual | 80 | 80 |
| Observations | 160,000 | 160,000 |
| Basis source | `state_index` | `encoded_state` |
| Basis dimension | 21 | 81 |
| Penalty weight | 100 | 10000 |

## Fit Summary

| Quantity | Low-dimensional | High-dimensional primary |
| --- | ---: | ---: |
| Optimizer flag | false | false |
| Iterations | 181 | 6 |
| Log likelihood | -174875.7843 | -160289.8089 |
| Estimation time | 3.77 seconds | 3.92 seconds |
| Bellman violation | 5.83e-5 | 3.08e-6 |
| Bellman RMSE | 2.74e-5 | 6.80e-7 |
| Standard errors finite | true | true |

The primary high-dimensional cell ran in 3.92 seconds.

## Recovery Metrics

| Metric | Gate | Low-dimensional | High-dimensional primary |
| --- | --- | ---: | ---: |
| Parameter cosine | at least 0.99 | 0.999146 | 0.999955 |
| Parameter relative RMSE | at most 0.15 | 0.059671 | 0.009528 |
| Reward RMSE | at most 0.03 | 0.008988 | 0.004432 |
| Policy total variation | at most 0.02 | 0.005179 | 0.002117 |
| Value RMSE | at most 0.10 | 0.017591 | 0.037836 |
| Q RMSE | at most 0.10 | 0.020514 | 0.031480 |

## Hard Gates

| Gate | Threshold | Primary value | Status |
| --- | --- | ---: | --- |
| Bellman violation | at most 0.05 | 0.000003 | pass |
| Standard errors finite | true | true | pass |
| Parameter cosine | at least 0.99 | 0.999955 | pass |
| Parameter relative RMSE | at most 0.15 | 0.009528 | pass |
| Reward RMSE | at most 0.03 | 0.004432 | pass |
| Policy total variation | at most 0.02 | 0.002117 | pass |
| Value RMSE | at most 0.10 | 0.037836 | pass |
| Q RMSE | at most 0.10 | 0.031480 | pass |
| Type A regret | at most 0.01 | 0.000113 | pass |
| Type B regret | at most 0.01 | 0.000183 | pass |
| Type C regret | at most 0.01 | 0.000014 | pass |

## Counterfactual Recovery

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.002096 | 2.78e-5 | 0.000234 | 0.000113 |
| Type B | 0.002421 | 3.67e-5 | 0.000309 | 0.000183 |
| Type C | 0.001231 | 6.91e-6 | 0.000027 | 0.000014 |

The estimates are not exactly equal to truth because the panel is finite. The
release claim is recovery within strict tolerances in the frozen known-truth
cells.
