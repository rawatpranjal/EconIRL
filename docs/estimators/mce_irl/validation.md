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

## Evidence

MCE-IRL is compared against the full structural and IRL rosters in six
simulation studies: [bus engine](../../simulation_studies/rust_bus.md),
[taxi gridworld](../../simulation_studies/taxi_gridworld.md),
[route choice](../../simulation_studies/route_choice.md),
[stockpiling](../../simulation_studies/stockpiling.md),
[fleet maintenance](../../simulation_studies/fleet_maintenance.md), and
[vehicle scrappage](../../simulation_studies/vehicle_scrappage.md). See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
