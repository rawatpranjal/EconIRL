# Simulation Study

NFXP runs on the `canonical_low_action` synthetic cell. The cell has known
rewards, transitions, policies, values, Q functions, and Type A, Type B, and
Type C counterfactual oracles, so every recovery claim is checked against the
truth. Real bus data cannot answer that question because the true reward, value
function, policy, and counterfactual oracles are not observed.

The full result generator is
[`nfxp_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/nfxp/run.py).
It writes the results file
[`nfxp_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/nfxp.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python validation/estimators/nfxp/run.py
```

The synthetic cell has action-dependent reward features and an exit action
that anchors the reward level. The estimates differ from truth because the
panel is finite.

## Evidence

NFXP is compared against the full structural and IRL rosters in six simulation
studies: [bus engine](../../simulation_studies/rust_bus.md),
[taxi gridworld](../../simulation_studies/taxi_gridworld.md),
[route choice](../../simulation_studies/route_choice.md),
[stockpiling](../../simulation_studies/stockpiling.md),
[fleet maintenance](../../simulation_studies/fleet_maintenance.md), and
[vehicle scrappage](../../simulation_studies/vehicle_scrappage.md). See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
