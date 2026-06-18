# Simulation Study

UFXP runs on the same low-dimensional action-dependent synthetic cell as the
other structural estimators. The cell has 21 states, 3 actions, known linear
reward, known transitions, and exact policy, value, Q, and Type A, Type B, and
Type C counterfactual oracle objects, so every recovery claim is checked
against the truth.

The full result generator is
[`ufxp_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/ufxp/run.py).
It writes the results file
[`ufxp_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/ufxp.json).

```bash
cd /path/to/econirl
PYTHONPATH=src:. python validation/estimators/ufxp/run.py
```

## Evidence

UFXP is compared against the full structural and IRL rosters on the
[bus engine](../../simulation_studies/rust_bus.md),
[taxi gridworld](../../simulation_studies/taxi_gridworld.md), and
[vehicle scrappage](../../simulation_studies/vehicle_scrappage.md) pages. See
the [simulation studies index](../../simulation_studies/index.md) for what
each study shows.
