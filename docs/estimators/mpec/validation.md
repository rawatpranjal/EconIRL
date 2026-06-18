# Simulation Study

MPEC runs on the `canonical_low_action` synthetic cell, the same
low-dimensional action-dependent structural benchmark used for NFXP and CCP.
The cell has 21 states, 3 actions, known linear reward, known transitions, and
exact policy, value, Q, and Type A, Type B, and Type C counterfactual oracle
objects, so every recovery claim is checked against fully specified truth.

The full result generator is
[`mpec_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/mpec/run.py).
It writes the results file
[`mpec_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/mpec.json).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/mpec/run.py
```

The simulation DGP has action-dependent reward features and an exit action
that anchors the reward level. The estimates are not exactly equal to truth
because the panel is finite.

## Evidence

MPEC appears in six simulation studies:
[bus engine](../../simulation_studies/rust_bus.md),
[taxi gridworld](../../simulation_studies/taxi_gridworld.md),
[route choice](../../simulation_studies/route_choice.md),
[stockpiling](../../simulation_studies/stockpiling.md),
[fleet maintenance](../../simulation_studies/fleet_maintenance.md), and
[vehicle scrappage](../../simulation_studies/vehicle_scrappage.md). See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
