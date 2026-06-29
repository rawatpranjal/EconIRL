# Rust Bus Engine Example

Read this page as a runnable MPEC smoke test on a Rust-style environment. The
important extra output is the Bellman constraint violation, because it tells you
whether the constrained program actually enforced the model equations.

The Rust bus-engine replacement problem is the canonical dynamic discrete
choice example. A bus operator observes mileage and chooses whether to keep
the current engine or replace it.

EconIRL ships a Rust-style environment that is suitable for a quick public
MPEC smoke test.

```python
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.mpec import MPECEstimator, MPECConfig
from econirl.preferences.linear import LinearUtility
from econirl.simulation import simulate_panel

env = RustBusEnvironment(
    operating_cost=0.01,
    replacement_cost=2.0,
    num_mileage_bins=20,
    discount_factor=0.99,
)
panel = simulate_panel(env, n_individuals=100, n_periods=50)
utility = LinearUtility.from_environment(env)

model = MPECEstimator(config=MPECConfig(solver="sqp"))
summary = model.estimate(
    panel=panel,
    utility=utility,
    problem=env.problem_spec,
    transitions=env.transition_matrices,
)

print(summary.parameters)
print(summary.metadata["final_constraint_violation"])
```

## Interpretation

The Rust-style linear utility specification estimates two parameters. The
first is the operating cost slope over mileage states. The second is the
replacement cost. The fitted policy gives the replacement probability by
mileage state.

```python
states = [0, 5, 10, 19]
print(summary.policy[states, 1])
```

## Counterfactual Replacement Cost

MPEC's estimator API does not currently provide the one-call
dataframe-wrapper `counterfactual` method used by NFXP and CCP. For simulation
evidence, counterfactual replacement-cost and transition interventions are
tested through the simulation harness and reported on the
[counterfactuals page](counterfactuals.md).

## Replication Boundary

This page is a package smoke test on the synthetic bus environment, not a full
historical replication of the original study. The estimator's recovery
properties are established on a synthetic cell whose data-generating process
is fully specified; see the [Simulation Study](validation.md) page. The
[bus engine simulation page](../../simulation_studies/rust_bus.md) compares
MPEC against the full estimator roster on a shared synthetic bus engine panel.
