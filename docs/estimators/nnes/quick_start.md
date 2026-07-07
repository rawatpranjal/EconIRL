# Quick Start

This page shows the wrapper path for NNES. The network shown here approximates
the continuation value; the reported reward parameters remain finite structural
parameters.

The public wrapper accepts dataframe-style panel data. This quick start is the
package smoke path: create a model, call `fit`, and read the fitted
sklearn-style attributes. The full NNES simulation study lives in the
simulation results file linked from [Simulation Study](validation.md).

```python
from econirl.datasets import load_rust_bus
from econirl import NNES

df = load_rust_bus()

model = NNES(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility="linear_cost",
    bellman="npl",
    hidden_dim=32,
    num_layers=2,
    v_epochs=500,
    n_outer_iterations=3,
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.se_)
print(model.policy_.shape)
print(model.v_network_.shape)
```

Output from the package smoke run:

```text
{'theta_c': 0.001034, 'RC': 3.073617}
{'theta_c': 0.000383, 'RC': 0.077527}
(90, 2)
(90,)
```

The fitted wrapper exposes structural parameter estimates, standard errors,
the estimated policy, the value function, the first-stage transition estimate,
and the trained value-network predictions over states.

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated structural reward parameters. |
| `se_` | Standard errors for reward parameters. |
| `policy_` | Estimated action probabilities by state. |
| `value_` | Estimated value function by state. |
| `v_network_` | Neural value-network predictions by state. |
| `transitions_` | First-stage transition matrix used by the wrapper. |

Use `bellman="npl"` for the reported NNES path. The wrapper also accepts
`bellman="nfxp"` as a neural soft-Bellman diagnostic variant.

## Full Estimator API

Use `econirl.estimation.nnes.NNESEstimator` when the model already has
package-native objects or when the simulation study needs exact control over
the DDC problem, transition tensor, reward specification, and NNES training
configuration.

```python
from econirl.estimation.nnes import NNESConfig, NNESEstimator

estimator = NNESEstimator(
    config=NNESConfig(
        hidden_dim=32,
        num_layers=2,
        v_epochs=500,
        n_outer_iterations=3,
        compute_se=True,
    )
)
summary = estimator.estimate(panel, utility, problem, transitions)

print(summary.parameters)
print(summary.metadata["v_loss_per_outer"][-1])
```

No fixed output is shown for the estimator-API snippet because `panel`,
`utility`, `problem`, and `transitions` are caller-supplied objects.

The synthetic data simulation uses the full estimator API because it supplies the
true DDC problem, transition tensor, and reward specification directly. That is
also the route to use when reproducing paper-style designs with custom state
encodings or supplied transition laws.
