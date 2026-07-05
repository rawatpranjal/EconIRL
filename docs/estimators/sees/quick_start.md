# Quick Start

The public wrapper accepts dataframe-style panel data. Create a model, call
`fit`, and read the fitted sklearn-style attributes.

```python
from econirl.datasets import load_rust_bus, rust_bus_reward_spec
from econirl import SEES

df = load_rust_bus()

model = SEES(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility=rust_bus_reward_spec(90),
    solution="value",
    basis_type="fourier",
    basis_dim=8,
    penalty_weight=10.0,
    num_theta_starts=1,
    max_iter=500,
)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.se_)
print(model.policy_.shape)
print(model.alpha_.shape)
```

The fitted wrapper exposes the structural parameter estimates, standard
errors, estimated policy, value function, transition estimate, and sieve
coefficients. The default `solution="value"` is V-SEES. The wrapper also
accepts `solution="q"`, `solution="ev"`, `solution="policy"`, and
`solution="collocation"` for alternative Bellman-SEES formulations.
Use `num_theta_starts > 1` when the finite-sample objective is sensitive to
the initial structural parameters. Treat this wrapper example as a smoke path;
the simulation-study page reports the synthetic cells with explicit
finite-state penalty weights.

| Attribute | Meaning |
| --- | --- |
| `params_` | Estimated structural reward parameters. |
| `se_` | Standard errors for reward parameters. |
| `policy_` | Estimated action probabilities by state. |
| `value_` | Estimated value function by state. |
| `alpha_` | Estimated sieve coefficients. |
| `transitions_` | First-stage transition matrix used by the wrapper. |

## Lower-Level Control

Use `econirl.estimation.sees.SEESEstimator` when the model already has
package-native objects or when the simulation study needs exact control over
the basis and penalty.

```python
from econirl.estimation.sees import SEESEstimator

estimator = SEESEstimator(
    solution="value",
    basis_type="bspline",
    basis_dim=21,
    penalty_weight=100.0,
    num_theta_starts=1,
    max_iter=1000,
    compute_se=True,
)
summary = estimator.estimate(panel, utility, problem, transitions)

print(summary.parameters)
print(summary.metadata["bellman_violation"])
```

Output from the low-dimensional simulation-study objects:

```text
[ 0.084825  0.526234 -0.011571 -0.203976]
5.826541219988712e-05
```

The synthetic data simulation uses the lower-level API because it supplies the
true DDC problem, transition tensor, and reward specification directly.
