# Validation

MCE-IRL is reported on two known-truth cells. The primary cell is
`mce_low_high_reward`, a compact state-space problem with eight
action-dependent reward features. The sanity cell is `canonical_low_action`.

These results are not hand-entered examples. They come from the known-truth
validation harness, where rewards, transitions, policies, value functions,
Q functions, and counterfactual oracle objects are known before estimation
starts.

The full result generator is
[`mce_irl_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/mce_irl/mce_irl_run.py).
It writes the rendered table source
[`mce_irl_results.tex`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/mce_irl/mce_irl_results.tex)
and the machine-readable artifact
[`mce_irl_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/mce_irl/mce_irl_results.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/mce_irl/mce_irl_run.py --quiet-progress --enforce-gates
```

The compact harness flow is:

```python
from experiments.known_truth import build_known_truth_dgp, run_estimator

dgp = build_known_truth_dgp(cell.dgp_config)
panel = simulate_panel(dgp, cell.simulation_config)
result = run_estimator("MCE-IRL", dgp, panel, smoke=False)
```

Read the tables as a sequence. The design table states the known-truth cells.
The fit summary reports the primary run. Recovery metrics compare the recovered
reward, value, Q function, policy, and feature moments to oracle objects. Hard
gates are the reported thresholds.

## Design

| Quantity | Sanity cell | Primary cell |
| --- | ---: | ---: |
| Cell id | `canonical_low_action` | `mce_low_high_reward` |
| States | 21 | 25 |
| Actions | 3 | 3 |
| Reward features | 4 | 8 |
| Simulated individuals | 2,000 | 3,000 |
| Periods per individual | 80 | 100 |
| Observations | 160,000 | 300,000 |
| Transition law | Known | Known |
| Reward features | Action-dependent | Action-dependent |

## Fit Summary

The primary run uses the low-level `MCEIRLEstimator` with the root
feature-matching optimizer and standard-error computation disabled.

| Quantity | Value |
| --- | ---: |
| Converged | true |
| Outer iterations | 25 |
| Log likelihood | -329244.2754 |
| Estimation time | 2.83 seconds |
| Optimizer | `root` |
| Feature residual | 1.89e-12 |
| Occupancy moment residual | 0.001060 |
| Parameter cosine similarity | 0.995950 |
| Parameter relative RMSE | 0.091725 |

Raw parameter cosine is reported, but it is not a hard gate for MCE-IRL. The
decision-critical checks are feature matching, occupancy moments, recovered
reward/value/Q objects, policy distance, and counterfactual regret.

## Recovery Metrics

| Metric | Sanity cell | Primary cell |
| --- | ---: | ---: |
| Feature residual | 3.46e-13 | 1.89e-12 |
| Occupancy moment residual | 0.000814 | 0.001060 |
| Reward normalized RMSE | 0.007252 | 0.082287 |
| Policy total variation | 0.004529 | 0.006984 |
| Value normalized RMSE | 0.007398 | 0.082646 |
| Q normalized RMSE | 0.026314 | 0.081560 |

## Counterfactual Metrics

The counterfactual checks are run after fitting by changing the oracle problem
and comparing the recovered-reward policy to the oracle policy.

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.006456 | 0.000157 | 0.000742 | 0.000433 |
| Type B | 0.006284 | 0.000142 | 0.000523 | 0.000410 |
| Type C | 0.004211 | 5.98e-5 | 0.000145 | 0.000094 |

## Hard Gates

| Gate | Threshold | Value | Status |
| --- | --- | ---: | --- |
| Converged | true | true | pass |
| Feature residual | at most 0.02 | 1.89e-12 | pass |
| Occupancy moment residual | at most 0.02 | 0.001060 | pass |
| Reward normalized RMSE | at most 0.10 | 0.082287 | pass |
| Policy total variation | at most 0.03 | 0.006984 | pass |
| Value normalized RMSE | at most 0.10 | 0.082646 | pass |
| Q normalized RMSE | at most 0.10 | 0.081560 | pass |
| Type A regret | at most 0.05 | 0.000433 | pass |
| Type B regret | at most 0.05 | 0.000410 | pass |
| Type C regret | at most 0.05 | 0.000094 | pass |

The sanity and primary cells together pass 20 of 20 hard gates.
