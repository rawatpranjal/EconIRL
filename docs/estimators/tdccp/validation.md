# Simulation Study

The TD-CCP simulation study uses the `shapeshifter_encoded_state_locally_robust`
synthetic cell: encoded state features, a finite linear reward, and known oracle
objects for every comparison. The simulation asks whether the cross-fitted,
locally robust semigradient path recovers the reward parameters, the implied
dynamic objects, counterfactual behavior, and valid standard errors.

The result generator is
[`tdccp_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/estimators/tdccp/run.py).
It writes the results file
[`tdccp_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/validation/results/tdccp.json).

```bash
PYTHONPATH=src:. python validation/estimators/tdccp/run.py --quiet-progress
```

This is not a raw neural reward-recovery test. It is a structural parameter
test with encoded state features and stochastic transitions.

Known transition tensors were not used to estimate `theta`. They were supplied
after fitting so the simulation harness could compare recovered policies,
values, Q functions, and counterfactual decisions with oracle solutions.

The standard-error check re-simulates the same encoded-state design 25 times
with 300 individuals and 35 periods per replication. Each replication uses
cross-fitted, locally robust TD-CCP settings and individual-clustered
covariances. The 25-replication run is the current simulation receipt because
it is small enough to regenerate routinely. A paper-final audit should rerun
with `--mc-replications 100` under a CPU budget before claiming final Monte
Carlo precision.

The `canonical_low_action` cell is a simple sanity check. The
`canonical_high_action` cell is a diagnostic stress test that currently fails
all checks. The raw neural-reward diagnostic passes 5 of 8 checks and fails
reward, value, and Q recovery. It has no finite true reward parameter vector,
so it is not part of the primary finite-parameter study.

## Evidence

TD-CCP appears on the
[bus engine](../../simulation_studies/rust_bus.md) page. See the
[simulation studies index](../../simulation_studies/index.md) for what each
study shows.
