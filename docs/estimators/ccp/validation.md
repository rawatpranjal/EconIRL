# Simulation Study

Read this page as an oracle-object simulation for the CCP route. The question is
whether the inversion and NPL steps recover the same structural objects as the
reference target when support is strong.

We run CCP on the `canonical_low_action` synthetic cell, the same
low-dimensional action-dependent structural benchmark used for NFXP. The
simulation asks whether the CCP/NPL route can recover the same structural
objects from a finite panel when empirical choice support is strong. Real bus
data cannot answer that question because the true reward, policy, value
function, and counterfactual oracles are not observed.

These results come from the simulation harness. The harness fixes the transition
law, action-dependent reward features, and reward weights before generating the
finite panel. Those objects define the true reward, policy, value function,
Q function, and counterfactual oracles that are held back for evaluation. The
estimator sees the generated panel, the transition law, and the action-dependent
reward features, not the oracle dynamic objects.

The full result generator is
[`ccp_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/ccp/run.py).
It writes the results file
[`ccp_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/ccp.json).
To rerun it from the repository root:

```bash
PYTHONPATH=src:. python validation/estimators/ccp/run.py
```

The simulation DGP has action-dependent reward features and an exit action
that anchors the reward level. The estimates are not exactly equal to truth
because the panel is finite.

## Evidence

CCP is compared against the full structural and IRL rosters in six simulation
studies: [bus engine](../../simulation_studies/rust_bus.md),
[taxi gridworld](../../simulation_studies/taxi_gridworld.md),
[route choice](../../simulation_studies/route_choice.md),
[stockpiling](../../simulation_studies/stockpiling.md),
[fleet maintenance](../../simulation_studies/fleet_maintenance.md), and
[vehicle scrappage](../../simulation_studies/vehicle_scrappage.md). See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
