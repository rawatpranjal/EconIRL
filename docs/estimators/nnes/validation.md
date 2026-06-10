# Validation

NNES is reported on the `canonical_high_action` known-truth cell. The
low-dimensional `canonical_low_action` cell remains in the artifact as a
sanity check. The primary cell uses encoded states and a richer reward-feature
basis, which is the setting where the neural value approximation is meant to
matter.

These results are not hand-entered examples. They come from the known-truth
validation harness, where the reward, transition law, optimal policy, value
function, Q function, and counterfactual oracle objects are all known before
estimation starts.

The full result generator is
[`nnes_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nnes/nnes_run.py).
It writes the rendered table source
[`nnes_results.tex`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nnes/nnes_results.tex)
and the machine-readable artifact
[`nnes_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nnes/nnes_results.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/nnes/nnes_run.py --quiet-progress
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
run = run_estimator("NNES", dgp, panel, smoke=False, enforce_gates=True)
```

## Read the Tables

Read the tables as a sequence. The design rows state which known-truth cells
were run. The fit summary reports convergence, L-BFGS-B iterations, final
value-network loss, likelihood, and run time. Recovery metrics compare the
estimated structural object to oracle reward, policy, value, and Q objects.
Hard gates are the reported thresholds.

The validation evidence does not rely on optimizer convergence alone. It also
uses value-network loss, known-truth recovery, and counterfactual regret.

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
| Feature rank | 4 / 4 | 32 / 32 |
| State-action coverage | 1.000 | 0.959 |

## Fit Summary

| Quantity | Low-dimensional | High-dimensional primary |
| --- | ---: | ---: |
| Converged | true | true |
| L-BFGS-B iterations | 10 | 29 |
| Log likelihood | -174875.8093 | -160272.8803 |
| Estimation time | 37.12 seconds | 36.46 seconds |
| Outer NPL iterations | 3 | 3 |
| Final V loss | 0.000399 | 0.029932 |

The primary high-dimensional cell ran in 36.46 seconds.

## Recovery Metrics

| Metric | Gate | Low-dimensional | High-dimensional primary |
| --- | --- | ---: | ---: |
| Parameter cosine | at least 0.95 | 0.998240 | 0.991204 |
| Parameter relative RMSE | at most 0.30 | 0.065179 | 0.135110 |
| Reward RMSE | at most 0.08 | 0.010210 | 0.064012 |
| Policy total variation | at most 0.03 | 0.005646 | 0.023834 |
| Value RMSE | at most 0.20 | 0.019845 | 0.115620 |
| Q RMSE | at most 0.20 | 0.023370 | 0.137145 |

## Hard Gates

| Gate | Threshold | Primary value | Status |
| --- | --- | ---: | --- |
| Outer NPL iterations | at least 3 | 3 | pass |
| Final V loss | at most 0.05 | 0.029932 | pass |
| Parameter cosine | at least 0.95 | 0.991204 | pass |
| Parameter relative RMSE | at most 0.30 | 0.135110 | pass |
| Reward RMSE | at most 0.08 | 0.064012 | pass |
| Policy total variation | at most 0.03 | 0.023834 | pass |
| Value RMSE | at most 0.20 | 0.115620 | pass |
| Q RMSE | at most 0.20 | 0.137145 | pass |
| Type A regret | at most 0.05 | 0.004865 | pass |
| Type B regret | at most 0.05 | 0.005559 | pass |
| Type C regret | at most 0.05 | 0.001314 | pass |

## Counterfactual Recovery

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.021578 | 0.002165 | 0.007880 | 0.004865 |
| Type B | 0.021651 | 0.002128 | 0.007916 | 0.005559 |
| Type C | 0.013053 | 0.000920 | 0.002325 | 0.001314 |

The estimates are not exactly equal to truth because the panel is finite and
the value function is approximated. The reported scope is recovery within the
listed tolerances in the frozen known-truth cells.
