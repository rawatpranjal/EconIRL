# Simulation Study

MCE-IRL runs on two synthetic cells. The primary cell is
`mce_low_high_reward`, a compact state-space problem with eight
action-dependent reward features. The sanity cell is `canonical_low_action`.
The simulation checks whether the feature-matching route recovers
reward and counterfactual behavior when the transition law and reward-feature
basis are supplied. Real data cannot answer that question because the reward,
policy, value function, Q function, occupancy measure, and
counterfactual oracles are not observed.

The numbers come from the simulation harness. In that harness, the
transition law, reward features, and reward weights are fixed before generating
the panel. The estimator sees the generated demonstrations, the transition
law, and the supplied reward features. The reward, policy, value function,
Q function, occupancy measure, and counterfactual oracles are held back for
evaluation.

The full result generator is
[`mce_irl_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/mce_irl/run.py).
It writes the results file
[`mce_irl_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/mce_irl.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python validation/estimators/mce_irl/run.py --quiet-progress --enforce-gates
```

Read the tables in order. The design table states the synthetic cells. The
fit summary reports the primary run. Recovery metrics compare the recovered
reward, value, Q function, policy, and feature moments with oracle objects.
Numerical checks list the numerical cutoffs recorded by the harness.

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
| Estimation time | 2.69 seconds |
| Optimizer | `root` |
| Feature residual | 1.89e-12 |
| Occupancy moment residual | 0.001060 |
| Parameter cosine similarity | 0.995950 |
| Parameter relative RMSE | 0.091725 |

Raw parameter cosine is reported for reference. The recovery table focuses on
feature matching, occupancy moments, recovered reward/value/Q objects, policy
distance, and counterfactual regret.

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

The counterfactual checks run after fitting. They change the oracle problem and
compare the recovered-reward policy with the oracle policy.

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.006456 | 0.000157 | 0.000742 | 0.000433 |
| Type B | 0.006284 | 0.000142 | 0.000523 | 0.000410 |
| Type C | 0.004211 | 5.98e-5 | 0.000145 | 0.000094 |

## Numerical Checks

| Check | Threshold | Value | Status |
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

The same set of checks is also recorded for the sanity cell in the JSON
results file.

MCE-IRL also appears on the
[bus engine](../../simulation_studies/rust_bus.md) and
[gridworld](../../simulation_studies/taxi_gridworld.md) simulation study pages,
where it is compared against the full structural and IRL rosters.
