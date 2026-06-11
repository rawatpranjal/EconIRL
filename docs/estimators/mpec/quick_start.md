# Quick Start

The public MPEC surface is the lower-level estimator API. Create an estimator,
call `estimate`, and read the returned summary object.

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
    seed=42,
)
panel = simulate_panel(env, n_individuals=100, n_periods=50, seed=123)
utility = LinearUtility.from_environment(env)

model = MPECEstimator(config=MPECConfig(solver="sqp"), verbose=False)
summary = model.estimate(
    panel=panel,
    utility=utility,
    problem=env.problem_spec,
    transitions=env.transition_matrices,
)

print([float(x) for x in summary.parameters])
print([float(x) for x in summary.standard_errors])
print(summary.policy.shape)
print(summary.metadata["final_constraint_violation"])
```

Output

```text
[0.017559314146637917, 2.244328022003174]
[0.003141143824905157, 0.07669873535633087]
(20, 2)
9.265894274079756e-09
```

The fitted summary exposes structural parameters, standard errors, a policy,
a value function, a likelihood, and constrained-optimizer diagnostics.

| Attribute | Meaning |
| --- | --- |
| `parameters` | Estimated structural reward parameters. |
| `standard_errors` | Standard errors for the structural parameters. |
| `policy` | Estimated action probabilities by state. |
| `value_function` | Estimated value function by state. |
| `log_likelihood` | Maximized constrained choice log likelihood. |
| `metadata` | Solver status and Bellman constraint diagnostics. |

## Counterfactual Example

MPEC does not currently expose the same one-call dataframe wrapper
`counterfactual` method as NFXP and CCP. The counterfactual evidence is
therefore produced through the known-truth simulation harness, which re-solves
oracle Type A, Type B, and Type C counterfactuals against the recovered
structural object.

See [Counterfactuals](counterfactuals.md) for the reported counterfactual
families and [Simulation Study](validation.md) for the generator and artifact
links.

## Lower-Level Control

Use `econirl.estimation.mpec.MPECEstimator` when you need direct control over
panel objects, utility objects, transition tensors, constrained optimizer
options, Bellman constraint tolerance, or diagnostic metadata.
